"""Invariants of the observation service that do not need a screen.

`observe()` itself needs a desktop, so it is exercised through the runner and the
integration tests. What is left here is the part a run depends on before it ever
captures anything: the id a frame is addressed by, and which elements are offered
to the model as targets.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from PIL import Image

from gui_agent.config import Config
from gui_agent.perception.ocr import EngineSelection, OcrError, OcrOutput
from gui_agent.runtime import observation
from gui_agent.runtime.observation import (
    ObservationError,
    ObservationService,
    describe_elements,
)
from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot
from gui_agent.schemas import BoundingBox, Point, ScreenInfo, UIElement


def _service() -> ObservationService:
    # Constructing the service does not capture anything: the OCR engine is built
    # on the first observe(), which is why this needs no desktop.
    return ObservationService(Config())


def test_each_observation_gets_its_own_id() -> None:
    """7.4.2: two observations must not share an id.

    The id is how a step record points back at the frame its coordinate came from.
    A repeat would not just be untidy - it would make the evidence ambiguous, and
    the whole point of a frame-local id is that it can be checked.
    """
    service = _service()

    ids = [service.next_id() for _ in range(3)]

    assert ids == ["obs-0001", "obs-0002", "obs-0003"]
    assert len(set(ids)) == 3


def test_contours_are_not_offered_as_targets() -> None:
    """8.1.4: the element list is what the model aims with, so it holds real text.

    A bare contour box has no label. Listing it would invite the model to describe
    it as "the search box", which is an invention the adapter would then refuse -
    or worse, resolve against the wrong element.
    """
    snapshot = ObservationSnapshot(
        observation_id="obs-0001",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=100, screenshot_height=100, control_width=100, control_height=100
        ),
        elements=[
            ElementRef(
                element_id="obs-0001-e000",
                text="",
                bounding_box=BoundingBox(left=0, top=0, right=20, bottom=20),
                center=Point(x=10, y=10),
                confidence=0.9,
                source="contour",
            ),
            ElementRef(
                element_id="obs-0001-e001",
                text="Search",
                bounding_box=BoundingBox(left=30, top=0, right=80, bottom=20),
                center=Point(x=55, y=10),
                confidence=0.9,
            ),
        ],
    )

    rendered = describe_elements(snapshot)

    assert "obs-0001-e001" in rendered
    assert "obs-0001-e000" not in rendered, "an unlabelled box is not a target"
    assert rendered.count("\n") == 0, "one line per offered element"


def test_the_element_line_carries_what_a_step_needs_to_aim() -> None:
    """The prompt line has to carry the id, the text and the box, in that order."""
    snapshot = ObservationSnapshot(
        observation_id="obs-0002",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=100, screenshot_height=100, control_width=100, control_height=100
        ),
        elements=[
            ElementRef(
                element_id="obs-0002-e007",
                text="Fetch origin",
                bounding_box=BoundingBox(left=10, top=20, right=110, bottom=40),
                center=Point(x=60, y=30),
                confidence=0.94,
            )
        ],
    )

    line = describe_elements(snapshot)

    assert line.startswith("obs-0002-e007  'Fetch origin'")
    assert "conf=0.94" in line
    assert "center=(60,30)" in line
    assert "box=(10,20,110,40)" in line


# ───────── observe(), against a prepared frame instead of a screen ─────────
SCREEN = ScreenInfo(
    screenshot_width=200, screenshot_height=100, control_width=200, control_height=100
)


class _PreparedCapture:
    """One prepared frame, standing in for the screen."""

    def __init__(self) -> None:
        self.image = Image.new("RGB", (200, 100), "white")
        self.screen_info = SCREEN
        self.captured_at = datetime.now(UTC)
        self.image_path = None


class _StubEngine:
    """An OCR backend with a fixed answer."""

    name = "stub"

    def __init__(self, elements: list[UIElement] | None = None, error: str | None = None) -> None:
        self._elements = elements or []
        self._error = error

    def recognize(self, image: object, **kwargs: object) -> OcrOutput:
        if self._error:
            raise OcrError(self._error)
        return OcrOutput(elements=list(self._elements), engine=self.name, elapsed_ms=1.0)


def _element(text: str, index: int = 0) -> UIElement:
    return UIElement(
        text=text,
        bounding_box=BoundingBox(left=index * 20, top=0, right=index * 20 + 18, bottom=10),
        confidence=0.9,
    )


def _patch(monkeypatch: pytest.MonkeyPatch, engine: _StubEngine) -> None:
    monkeypatch.setattr(observation, "capture_monitor", lambda *a, **k: _PreparedCapture())
    monkeypatch.setattr(
        observation, "create_ocr_engine", lambda config: EngineSelection(engine=engine)
    )


def test_observe_builds_one_frame_from_a_prepared_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """16.1 asks for prepared screenshots rather than the operator's desktop.

    The whole of `observe()` had no test at all - it needs a screen, and the suite
    is not allowed to take one. A prepared frame plus a stubbed backend is what the
    section actually asks for, and it runs the id assignment, the geometry and the
    snapshot assembly for real.
    """
    _patch(monkeypatch, _StubEngine([_element("Fetch origin")]))

    snapshot = ObservationService(Config()).observe()

    assert snapshot.observation_id == "obs-0001"
    assert snapshot.ocr_engine == "stub"
    assert snapshot.image_path is None
    assert snapshot.screen_info.screenshot_width == 200
    assert [item.text for item in snapshot.elements if item.text] == ["Fetch origin"]
    assert snapshot.elements[0].element_id == "obs-0001-e000"
    assert snapshot.processing_time_ms >= 0.0


def test_observe_numbers_its_frames_in_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """7.4.2: two observations must not share an id, and the service is what mints
    them. `next_id` was tested; the path that actually uses it was not."""
    _patch(monkeypatch, _StubEngine([_element("Browser")]))
    service = ObservationService(Config())

    ids = [service.observe().observation_id for _ in range(3)]

    assert ids == ["obs-0001", "obs-0002", "obs-0003"]


def test_an_ocr_failure_is_recorded_on_the_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """7.3.8: every OCR error goes into the record.

    A frame with a broken text pass is still a frame - and the alternative, quietly
    reusing the previous frame's elements, would attach today's coordinates to
    yesterday's screen.
    """
    _patch(monkeypatch, _StubEngine(error="tesseract binary missing"))

    snapshot = ObservationService(Config()).observe()

    assert any("tesseract binary missing" in error for error in snapshot.errors), snapshot.errors
    assert snapshot.observation_id == "obs-0001", "the frame is still the current one"


def test_a_capture_failure_stops_the_frame_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    """No screenshot means no observation, and the caller has to hear about it."""

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("monitor_index 1 is out of range")

    monkeypatch.setattr(observation, "capture_monitor", boom)

    with pytest.raises(ObservationError, match="monitor_index 1 is out of range"):
        ObservationService(Config()).observe()


def test_the_element_cap_keeps_the_labelled_elements(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_select` ranks text ahead of unlabelled contours.

    That ordering is what stops a screenful of boxes from pushing the labels out of
    the prompt - the cap is 60 by default, and a busy screen produces more contours
    than that on its own.
    """
    labels = [_element(f"label {index}", index) for index in range(5)]
    blanks = [_element("", index) for index in range(80)]
    _patch(monkeypatch, _StubEngine(labels + blanks))

    snapshot = ObservationService(Config(), max_elements=10).observe()

    assert len(snapshot.elements) == 10
    assert [item.text for item in snapshot.elements if item.text] == [
        f"label {index}" for index in range(5)
    ]


def test_a_contour_failure_is_recorded_and_the_frame_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contours are optional context; the labelled elements are the part that matters.

    Losing them narrows what the model can aim at, which is worth recording - but it
    is not a reason to throw away a frame that OCR read perfectly well.
    """
    _patch(monkeypatch, _StubEngine([_element("Browser")]))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("contour detection is unavailable")

    monkeypatch.setattr(observation, "detect_ui_candidates", boom)

    snapshot = ObservationService(Config()).observe()

    assert any("ui detection failed" in error for error in snapshot.errors), snapshot.errors
    assert [item.text for item in snapshot.elements if item.text] == ["Browser"]
