"""OCR backends that return structured :class:`UIElement` objects.

Tesseract is the fallback backend and PaddleOCR is the primary target. A backend
that cannot be imported must never break the other one: :func:`create_ocr_engine`
degrades to Tesseract and reports what happened.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from ..config import OcrConfig
from ..schemas import BoundingBox, UIElement

DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)
DEFAULT_MIN_CONFIDENCE = 0.5
WHITESPACE = re.compile(r"\s+")

# Tesseract and PaddleOCR use different codes for the same language. The
# configuration stays backend agnostic and each backend translates, otherwise a
# PaddleOCR style "en" would be handed to Tesseract, which only knows "eng".
TESSERACT_LANGUAGE_ALIASES = {
    "en": "eng",
    "eng": "eng",
    "zh": "chi_sim",
    "ch": "chi_sim",
    "ch_sim": "chi_sim",
    "zh_cn": "chi_sim",
    "ch_tra": "chi_tra",
    "zh_tw": "chi_tra",
}


def to_tesseract_languages(languages: Sequence[str]) -> list[str]:
    """Translate configuration language codes into Tesseract language codes."""
    return [TESSERACT_LANGUAGE_ALIASES.get(language, language) for language in languages]


class OcrError(RuntimeError):
    """Raised when an OCR backend is unavailable or fails."""


@dataclass
class OcrOutput:
    """Normalised result produced by every backend."""

    elements: list[UIElement]
    engine: str
    elapsed_ms: float
    errors: list[str] = field(default_factory=list)


@dataclass
class EngineSelection:
    """The engine that was actually built, plus human readable notices."""

    engine: OCREngine
    notices: list[str] = field(default_factory=list)


def clean_text(value: str | None) -> str:
    """Collapse whitespace and strip the result."""
    return WHITESPACE.sub(" ", value or "").strip()


def make_element(
    text: str,
    left: int,
    top: int,
    width: int,
    height: int,
    confidence: float,
    *,
    source: str = "ocr",
) -> UIElement | None:
    """Build a UIElement, returning ``None`` for degenerate boxes."""
    if width <= 0 or height <= 0:
        return None
    return UIElement(
        text=text,
        bounding_box=BoundingBox(
            left=int(left),
            top=int(top),
            right=int(left) + int(width),
            bottom=int(top) + int(height),
        ),
        confidence=max(0.0, min(1.0, float(confidence))),
        source=source,  # type: ignore[arg-type]
    )


def box_to_bounds(box: Sequence[Sequence[float]]) -> tuple[int, int, int, int]:
    """Convert a 4-point polygon (PaddleOCR style) into integer bounds."""
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    left, right = int(min(xs)), int(max(xs))
    top, bottom = int(min(ys)), int(max(ys))
    return left, top, max(1, right - left), max(1, bottom - top)


class OCREngine(ABC):
    """Common interface for every OCR backend."""

    name = "base"

    def __init__(
        self,
        default_languages: Iterable[str] = DEFAULT_LANGUAGES,
        default_min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self.default_languages = tuple(default_languages)
        self.default_min_confidence = float(default_min_confidence)

    @abstractmethod
    def recognize(
        self,
        image: Image.Image | np.ndarray,
        *,
        languages: Sequence[str] | None = None,
        min_confidence: float | None = None,
    ) -> OcrOutput:
        """Return every recognised text region as a UIElement."""

    def _threshold(self, min_confidence: float | None) -> float:
        return self.default_min_confidence if min_confidence is None else float(min_confidence)

    def _languages(self, languages: Sequence[str] | None) -> tuple[str, ...]:
        return tuple(languages) if languages else self.default_languages


def to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return np.stack([array] * 3, axis=-1)
    return array[:, :, :3]


class TesseractOCREngine(OCREngine):
    """Word-level Tesseract backend that keeps confidence and bounding boxes."""

    name = "tesseract"

    def recognize(
        self,
        image: Image.Image | np.ndarray,
        *,
        languages: Sequence[str] | None = None,
        min_confidence: float | None = None,
    ) -> OcrOutput:
        import pytesseract

        threshold = self._threshold(min_confidence)
        language = "+".join(to_tesseract_languages(self._languages(languages)))
        started = time.perf_counter()
        try:
            data = pytesseract.image_to_data(
                image, lang=language, output_type=pytesseract.Output.DICT
            )
        except Exception as exc:
            raise OcrError(f"Tesseract failed: {exc}") from exc

        elements: list[UIElement] = []
        texts = data.get("text", [])
        for index in range(len(texts)):
            text = clean_text(texts[index])
            if not text:
                continue
            try:
                raw_confidence = float(data["conf"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if raw_confidence < 0:
                continue
            confidence = min(1.0, raw_confidence / 100.0)
            if confidence < threshold:
                continue
            element = make_element(
                text,
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
                confidence,
            )
            if element is not None:
                elements.append(element)

        return OcrOutput(
            elements=elements,
            engine=self.name,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


class PaddleOCREngine(OCREngine):
    """PaddleOCR backend; supports both the 2.x and the 3.x result shapes."""

    name = "paddleocr"

    def __init__(
        self,
        default_languages: Iterable[str] = DEFAULT_LANGUAGES,
        default_min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        super().__init__(default_languages, default_min_confidence)
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - depends on the environment
            raise OcrError(
                "PaddleOCR is not installed. Install it with: pip install -r requirements-ocr.txt"
            ) from exc
        self._factory = PaddleOCR
        self._engines: dict[str, Any] = {}

    def _engine_for(self, language: str) -> Any:
        if language not in self._engines:
            try:
                self._engines[language] = self._factory(lang=language)
            except TypeError:
                self._engines[language] = self._factory(lang=language, use_angle_cls=True)
        return self._engines[language]

    def _run(self, engine: Any, array: np.ndarray) -> Any:
        if hasattr(engine, "predict"):
            try:
                return engine.predict(array)
            except TypeError:
                pass
        return engine.ocr(array, cls=True)

    def _parse(self, result: Any, threshold: float) -> list[UIElement]:
        elements: list[UIElement] = []
        for page in result or []:
            if page is None:
                continue
            if hasattr(page, "get"):
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                boxes = page.get("rec_polys") or page.get("dt_polys") or []
                triples = zip(texts, scores, boxes)
            else:
                triples = (
                    (line[1][0], line[1][1], line[0]) for line in page if line and len(line) >= 2
                )
            for text, score, box in triples:
                cleaned = clean_text(text)
                if not cleaned:
                    continue
                confidence = max(0.0, min(1.0, float(score)))
                if confidence < threshold:
                    continue
                left, top, width, height = box_to_bounds(box)
                element = make_element(cleaned, left, top, width, height, confidence)
                if element is not None:
                    elements.append(element)
        return elements

    def recognize(
        self,
        image: Image.Image | np.ndarray,
        *,
        languages: Sequence[str] | None = None,
        min_confidence: float | None = None,
    ) -> OcrOutput:
        threshold = self._threshold(min_confidence)
        language = self._languages(languages)[0]
        started = time.perf_counter()
        try:
            result = self._run(self._engine_for(language), to_rgb_array(image))
        except Exception as exc:
            raise OcrError(f"PaddleOCR failed: {exc}") from exc
        elements = self._parse(result, threshold)
        return OcrOutput(
            elements=elements,
            engine=self.name,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


class FallbackOCREngine(OCREngine):
    """Run a primary backend and switch permanently to a fallback on failure.

    Importing PaddleOCR can succeed while its PaddlePaddle engine is missing, so
    the problem only appears on the first ``recognize`` call. This wrapper keeps
    that from breaking the Tesseract flow, which is what the Week 2 acceptance
    criteria require.
    """

    def __init__(
        self,
        primary: OCREngine,
        fallback: OCREngine,
        notices: list[str] | None = None,
    ) -> None:
        super().__init__(fallback.default_languages, fallback.default_min_confidence)
        self._primary: OCREngine | None = primary
        self._fallback = fallback
        self.notices: list[str] = notices if notices is not None else []

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._primary.name if self._primary is not None else self._fallback.name

    @property
    def using_fallback(self) -> bool:
        return self._primary is None

    def recognize(
        self,
        image: Image.Image | np.ndarray,
        *,
        languages: Sequence[str] | None = None,
        min_confidence: float | None = None,
    ) -> OcrOutput:
        if self._primary is not None:
            try:
                return self._primary.recognize(
                    image, languages=languages, min_confidence=min_confidence
                )
            except OcrError as exc:
                self.notices.append(
                    f"{self._primary.name} failed at runtime ({exc}); "
                    f"switching to {self._fallback.name}."
                )
                self._primary = None
        return self._fallback.recognize(image, languages=languages, min_confidence=min_confidence)


def create_ocr_engine(config: OcrConfig | None = None) -> EngineSelection:
    """Build the configured engine, falling back to Tesseract when needed."""
    config = config or OcrConfig()
    settings = {
        "default_languages": config.languages,
        "default_min_confidence": config.min_confidence,
    }
    notices: list[str] = []

    if config.engine == "tesseract":
        return EngineSelection(TesseractOCREngine(**settings), notices)

    try:
        primary = PaddleOCREngine(**settings)
    except OcrError as exc:
        notices.append(f"PaddleOCR unavailable ({exc}); falling back to Tesseract.")
        if config.fallback_engine == "none":
            raise
        return EngineSelection(TesseractOCREngine(**settings), notices)

    if config.fallback_engine == "tesseract":
        # The engine may only fail on the first call, so keep Tesseract ready.
        return EngineSelection(
            FallbackOCREngine(primary, TesseractOCREngine(**settings), notices), notices
        )
    return EngineSelection(primary, notices)
