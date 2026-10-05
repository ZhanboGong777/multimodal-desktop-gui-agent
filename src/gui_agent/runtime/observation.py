"""One entry point for "what is on screen right now".

Week 2 exposed capture, OCR and contour detection as three separate calls, and
each demo wired them together differently. A closed loop cannot afford that: the
screenshot, the geometry and the elements all have to belong to the same frame,
or a coordinate resolved from one capture gets executed against another.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import DEFAULT_MAX_ELEMENTS, Config
from ..perception.capture import (
    capture_monitor,
    foreground_window_context,
    window_bounds_in_screenshot,
)
from ..perception.ocr import OcrError, create_ocr_engine
from ..perception.ui_detection import detect_ui_candidates
from ..schemas import UIElement
from .schemas import ElementRef, ObservationSnapshot


class ObservationError(RuntimeError):
    """Raised when a frame could not be produced at all."""


def _rank(element: UIElement) -> tuple[int, float]:
    """Sort key: text first, then confidence, so labels outrank bare contours."""
    return (1 if element.text.strip() else 0, element.confidence)


def foreground_matches(snapshot: ObservationSnapshot) -> bool:
    """Recheck window identity and location immediately before dispatch.

    A match between two saved images cannot authorise a click after another
    window has taken focus or the target window has moved again. This check reads
    OS metadata only; it neither captures nor changes the desktop.
    """
    if not snapshot.foreground_stable or not snapshot.window_id or snapshot.window_bounds is None:
        return False
    current = foreground_window_context()
    return bool(
        current.window_id == snapshot.window_id
        and current.window_title == snapshot.window_title
        and current.window_class == snapshot.window_class
        and window_bounds_in_screenshot(current, snapshot.screen_info) == snapshot.window_bounds
    )


class ObservationService:
    """Captures a frame and turns it into an addressable snapshot."""

    def __init__(
        self,
        config: Config,
        *,
        max_elements: int = DEFAULT_MAX_ELEMENTS,
        output_directory: Path | None = None,
        save_images: bool = True,
    ) -> None:
        self.config = config
        self.max_elements = max_elements
        self.output_directory = output_directory
        self.save_images = save_images
        self._selection: Any = None
        self._counter = 0
        self._notices: list[str] = []

    # ── engine lifetime ────────────────────────────────────────────────
    def _engine(self) -> Any:
        """Build the OCR engine once.

        Reloading the model on every step is the difference between a loop that
        runs and one that spends its whole budget in initialisation.
        """
        if self._selection is None:
            self._selection = create_ocr_engine(self.config.perception.ocr)
            self._notices = list(getattr(self._selection, "notices", []))
        return self._selection

    @property
    def notices(self) -> list[str]:
        return list(self._notices)

    def next_id(self) -> str:
        self._counter += 1
        return f"obs-{self._counter:04d}"

    def foreground_matches(self, snapshot: ObservationSnapshot) -> bool:
        """The injectable observer seam used by the runner's dispatch check."""
        return foreground_matches(snapshot)

    # ── the one call the runtime makes ─────────────────────────────────
    def observe(self, *, observation_id: str | None = None) -> ObservationSnapshot:
        """Capture, read and package one frame.

        Errors from OCR or contour detection are recorded on the snapshot rather
        than raised: a run that keeps going on a stale frame is worse than one
        that reports a degraded observation.
        """
        started = time.perf_counter()
        oid = observation_id or self.next_id()
        errors: list[str] = []

        try:
            capture = capture_monitor(
                self.config.perception.monitor_index,
                output_directory=self.output_directory,
                save=self.save_images,
            )
        except Exception as exc:
            raise ObservationError(f"capture failed: {type(exc).__name__}: {exc}") from exc

        elements: list[UIElement] = []
        engine_name = "none"
        try:
            selection = self._engine()
            output = selection.engine.recognize(capture.image)
            elements = list(output.elements)
            engine_name = output.engine
            errors.extend(output.errors)
        except OcrError as exc:
            errors.append(f"ocr unavailable: {exc}")

        if self.config.perception.ui_detection.enabled:
            try:
                boxes = [element.bounding_box for element in elements]
                elements.extend(
                    detect_ui_candidates(
                        capture.image,
                        min_area=self.config.perception.ui_detection.min_area,
                        max_area_ratio=self.config.perception.ui_detection.max_area_ratio,
                        min_rectangularity=self.config.perception.ui_detection.min_rectangularity,
                        max_candidates=self.config.perception.ui_detection.max_candidates,
                        exclude=boxes,
                        exclusion_threshold=self.config.perception.ui_detection.exclusion_threshold,
                        priority_region=getattr(capture, "window_bounds", None),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - contours are optional context
                errors.append(f"ui detection failed: {type(exc).__name__}: {exc}")

        refs = [
            ElementRef(
                element_id=f"{oid}-e{index:03d}",
                text=element.text,
                bounding_box=element.bounding_box,
                center=element.center or element.bounding_box.center,
                confidence=element.confidence,
                source=element.source,
            )
            for index, element in enumerate(self._select(elements))
        ]

        return ObservationSnapshot(
            observation_id=oid,
            captured_at=capture.captured_at or datetime.now(UTC),
            image_path=str(capture.image_path) if capture.image_path else None,
            screen_info=capture.screen_info,
            elements=refs,
            ocr_engine=engine_name,
            processing_time_ms=(time.perf_counter() - started) * 1000.0,
            errors=errors,
            notices=self._engine_notices(),
            # `getattr` because a capture result is a duck-typed seam here: tests and other
            # callers build their own, and a frame without these two is still a valid frame.
            # The window is a decoration on the observation, never a requirement for one.
            window_title=getattr(capture, "window_title", ""),
            window_class=getattr(capture, "window_class", ""),
            window_id=getattr(capture, "window_id", ""),
            window_bounds=getattr(capture, "window_bounds", None),
            foreground_stable=getattr(capture, "foreground_stable", False),
        )

    def _engine_notices(self) -> list[str]:
        """What the OCR engine reported about itself when it was built.

        These are produced once, at construction, and were read by nothing: a run
        that fell back from PaddleOCR to Tesseract recorded `tesseract` and lost
        the reason. 7.1.4 asks for the fallback to be kept.
        """
        if self._selection is None:
            return []
        return list(getattr(self._selection, "notices", []))

    def _select(self, elements: list[UIElement]) -> list[UIElement]:
        """Keep the most useful elements, text before contours.

        Ordering matters beyond the cap: the model reads the list top to bottom,
        and a screen's worth of unlabelled rectangles ahead of the actual labels
        makes the prompt worse, not just longer.
        """
        ordered = sorted(elements, key=_rank, reverse=True)
        return ordered[: self.max_elements]


def describe_elements(snapshot: ObservationSnapshot) -> str:
    """Render the element list for the model prompt.

    T04_20261005_192417 could not name an input or send control: contours survived
    selection but were dropped here. Offer their frame-local ids and geometry
    without inventing a semantic label; the model can compare them with the image.
    The placeholder is not text for re-location: stale contour ids need verified
    current image evidence and a real current candidate, never their old position.
    """
    lines = []
    for item in snapshot.elements:
        label = repr(item.text) if item.text.strip() else "<unlabelled box>"
        box = item.bounding_box
        lines.append(
            f"{item.element_id}  {label}  conf={item.confidence:.2f}  "
            f"center=({item.center.x},{item.center.y})  box=({box.left},{box.top},{box.right},{box.bottom})"
        )
    return "\n".join(lines)
