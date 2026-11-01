"""OCR backends that return structured :class:`UIElement` objects.

Tesseract is the fallback backend and PaddleOCR is the primary target. A backend
that cannot be imported must never break the other one: :func:`create_ocr_engine`
degrades to Tesseract and reports what happened.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..config import OcrConfig
from ..schemas import BoundingBox, UIElement

DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)
DEFAULT_MIN_CONFIDENCE = 0.5
WHITESPACE = re.compile(r"\s+")
CJK_WORD_GAP = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")

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


def first_present(page: Any, *keys: str) -> Any:
    """Return the first value that is not ``None``, otherwise an empty list.

    ``page.get(key) or fallback`` would be shorter but ``or`` evaluates
    ``bool(value)``, and PaddleOCR hands back NumPy arrays for some result
    shapes: ``bool(array)`` raises ``ValueError`` as soon as the array holds
    more than one element.
    """
    for key in keys:
        value = page.get(key)
        if value is not None:
            return value
    return []


def box_to_bounds(box: Sequence[Sequence[float]]) -> tuple[int, int, int, int]:
    """Convert a 4-point polygon (PaddleOCR style) into integer bounds."""
    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    left, right = int(min(xs)), int(max(xs))
    top, bottom = int(min(ys)), int(max(ys))
    return left, top, max(1, right - left), max(1, bottom - top)


#: Tesseract reports one row per *word*. A vision model reading the same
#: screenshot describes whole labels instead ("Fetch origin", "Summary
#: (required)"), so the words are merged back into the lines a reader would see.
#: The ratio is the second half of that rule: a gap wider than this fraction of
#: the line height means Tesseract put two separate controls on one text row, and
#: merging those would aim the click at the empty space between them.
LINE_GAP_RATIO = 1.2


class WordRow(NamedTuple):
    """One Tesseract word, tagged with the line Tesseract assigned it to."""

    line_key: object
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float


class TextLine(NamedTuple):
    """A whole line of text, as a reader - or a vision model - would see it."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float


def tesseract_line_key(data: dict[str, Any], index: int) -> object:
    """Tesseract's own line assignment for one word row.

    When the columns are missing the key falls back to one unique value per row,
    which degrades to the old one-element-per-word behaviour rather than merging
    the whole screen into a single element.
    """
    try:
        return (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return ("word", index)


def merge_words_into_lines(
    rows: Sequence[WordRow], *, gap_ratio: float = LINE_GAP_RATIO
) -> list[TextLine]:
    """Merge consecutive word rows back into lines.

    Two rows are merged only when Tesseract put them on the same line *and* they
    sit closer together than ``gap_ratio`` times the line height. Without the
    second rule a menu bar reads as one element ("File  Edit  View") whose centre
    is a patch of empty pixels between two menus.
    """
    lines: list[TextLine] = []
    current: list[WordRow] = []

    def flush() -> None:
        if not current:
            return
        left = min(row.left for row in current)
        top = min(row.top for row in current)
        right = max(row.left + row.width for row in current)
        bottom = max(row.top + row.height for row in current)
        lines.append(
            TextLine(
                text=" ".join(row.text for row in current),
                left=left,
                top=top,
                # No clamping: a degenerate box has to reach make_element intact,
                # which is what drops it.
                width=right - left,
                height=bottom - top,
                # The weakest word sets the line's confidence: a line is not more
                # trustworthy than the worst piece of it.
                confidence=min(row.confidence for row in current),
            )
        )
        current.clear()

    for row in rows:
        if current:
            previous = current[-1]
            gap = row.left - (previous.left + previous.width)
            if row.line_key != previous.line_key or gap > gap_ratio * max(previous.height, 1):
                flush()
        current.append(row)
    flush()
    return lines


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


def prepare_screen_text(image: Image.Image | np.ndarray) -> Image.Image:
    """Normalize UI contrast and remove long control borders on an OCR copy.

    Line lengths scale with the current frame, not a label vocabulary. The mask
    keeps short glyph strokes; neither original pixels nor contour input change.
    """
    gray = ImageOps.grayscale(Image.fromarray(to_rgb_array(image)))
    if np.median(np.asarray(gray)) < 127:
        gray = ImageOps.invert(gray)
    gray = ImageOps.autocontrast(gray)
    array = np.array(gray)
    _, ink = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    height, width = array.shape
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((1, max(40, round(width * .08))), dtype=np.uint8))
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((max(40, round(height * .08)), 1), dtype=np.uint8))
    array[(horizontal > 0) | (vertical > 0)] = 255
    return Image.fromarray(array)


class TesseractOCREngine(OCREngine):
    """Word-level Tesseract backend that keeps confidence and bounding boxes."""

    name = "tesseract"

    def __init__(
        self,
        default_languages: Iterable[str] = DEFAULT_LANGUAGES,
        default_min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        *,
        profile: str = "native",
        scale: float = 2.0,
        tessdata_dir: str | Path | None = None,
        transport: str = "subprocess",
        library_path: str | Path | None = None,
    ) -> None:
        super().__init__(default_languages, default_min_confidence)
        if profile not in {"native", "screen"}:
            raise ValueError("Tesseract profile must be native or screen")
        if not 1.0 <= scale <= 3.0:
            raise ValueError("Tesseract scale must be between 1 and 3")
        if tessdata_dir is not None and '"' in str(tessdata_dir):
            raise ValueError("Tessdata directory cannot contain a quote")
        if sys.platform == "win32" and tessdata_dir is not None and any(
            character.isspace() for character in str(tessdata_dir)
        ):
            raise ValueError("On Windows use a tessdata directory without whitespace")
        self.profile = profile
        self.scale = float(scale)
        self.tessdata_dir = tessdata_dir
        if transport not in {"subprocess", "persistent"}:
            raise ValueError("Tesseract transport must be subprocess or persistent")
        self.transport = transport
        self.library_path = library_path
        self._persistent: Any = None
        self._persistent_lock = threading.Lock()

    def close(self) -> None:
        """Release an optional persistent API; default subprocess has no handle."""
        if self._persistent is not None:
            self._persistent.close()

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
        options: dict[str, Any] = {"lang": language, "output_type": pytesseract.Output.DICT}
        source_size: tuple[int, int] | None = None
        actual_scale_x = actual_scale_y = 1.0
        config: list[str] = []
        if self.profile == "screen":
            # A single current-frame pass; no vocabulary, fixture labels or
            # previous screenshot enters recognition. Sparse text segmentation
            # keeps independent UI controls out of a document-style column.
            prepared = Image.fromarray(to_rgb_array(image))
            source_size = prepared.size
            prepared = prepare_screen_text(prepared)
            target = (max(1, round(prepared.width * self.scale)),
                      max(1, round(prepared.height * self.scale)))
            actual_scale_x, actual_scale_y = target[0] / prepared.width, target[1] / prepared.height
            image = prepared.resize(target, Image.Resampling.LANCZOS)
            config.extend(["--psm 11", "--oem 1"])
        if self.tessdata_dir is not None:
            value = str(self.tessdata_dir)
            # pytesseract uses shlex(posix=False) on Windows, which preserves
            # literal quotes in arguments. The constructor rejects whitespace
            # there rather than quietly using a different resource directory.
            config.append(f'--tessdata-dir {value}' if sys.platform == "win32"
                          else f'--tessdata-dir "{value}"')
        if config:
            options["config"] = " ".join(config)
        try:
            if self.transport == "persistent":
                from .tesseract_api import PersistentTesseract

                with self._persistent_lock:
                    if self._persistent is None:
                        self._persistent = PersistentTesseract(self.library_path)
                prepared_image = image if isinstance(image, Image.Image) else Image.fromarray(
                    to_rgb_array(image))
                data = self._persistent.image_to_data(
                    prepared_image, language=language, tessdata_dir=self.tessdata_dir,
                    psm=11 if self.profile == "screen" else 3,
                    oem=1 if self.profile == "screen" else 3,
                )
            else:
                data = pytesseract.image_to_data(image, **options)
        except Exception as exc:
            raise OcrError(f"Tesseract failed: {exc}") from exc

        rows: list[WordRow] = []
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
            try:
                left = int(data["left"][index])
                top = int(data["top"][index])
                width = int(data["width"][index])
                height = int(data["height"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            rows.append(
                WordRow(
                    line_key=tesseract_line_key(data, index),
                    text=text,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    confidence=confidence,
                )
            )

        elements: list[UIElement] = []
        for line in merge_words_into_lines(rows):
            text = CJK_WORD_GAP.sub("", line.text) if self.profile == "screen" else line.text
            left, top = line.left, line.top
            right, bottom = line.left + line.width, line.top + line.height
            if source_size is not None:
                # Outward rounding retains tiny glyphs; clamp to the original
                # screenshot so scaled-pixel coordinates cannot reach execution.
                left = max(0, min(source_size[0], floor(left / actual_scale_x)))
                top = max(0, min(source_size[1], floor(top / actual_scale_y)))
                right = max(left, min(source_size[0], ceil(right / actual_scale_x)))
                bottom = max(top, min(source_size[1], ceil(bottom / actual_scale_y)))
            element = make_element(
                text, left, top, right - left, bottom - top, line.confidence
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
        notices: list[str] | None = None,
        use_doc_preprocessing: bool = False,
    ) -> None:
        super().__init__(default_languages, default_min_confidence)
        self.notices: list[str] = notices if notices is not None else []
        self.use_doc_preprocessing = use_doc_preprocessing
        self._prepare_platform()
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            # Report the underlying error: "not installed" would be misleading when
            # the package is present but fails to import (missing dependency, a CUDA
            # DLL that will not load, or a renamed symbol).
            raise OcrError(
                f"PaddleOCR could not be imported ({type(exc).__name__}: {exc}). "
                "Install or repair it with: pip install -r requirements-ocr.txt"
            ) from exc
        self._factory = PaddleOCR
        self._engines: dict[str, Any] = {}

    def _prepare_platform(self) -> None:
        """Apply the Windows-only workarounds PaddleOCR needs in order to start.

        Both were found by reproducing the failures on a Windows GPU node; the
        experiment report carries the full traces.
        """
        if sys.platform != "win32":
            return

        # 1. paddleocr -> paddlex -> modelscope imports Torch. Torch ships cuDNN
        #    9.5 while PaddlePaddle 3.3 is built against 9.9, so once Torch has
        #    loaded its copy PaddlePaddle's own load dies with
        #    "OSError: [WinError 127] ... cudnn_cnn64_9.dll". The two CUDA stacks
        #    cannot share a process, and whichever loads first wins.
        if sys.modules.get("torch") is not None:
            self.notices.append(
                "torch was already imported, so PaddleOCR may fail to load its cuDNN "
                "libraries. Create the OCR engine before importing torch."
            )
        else:
            sys.modules["torch"] = None
            self.notices.append(
                "torch is blocked in this process to keep its cuDNN build from "
                "clashing with PaddlePaddle."
            )

        # 2. PaddleX's C++ predictor cannot read a model config from a non-ASCII
        #    path, so a Chinese Windows account name makes create_predictor fail
        #    with "json parse error ... attempting to parse an empty input".
        if not os.environ.get("PADDLE_PDX_CACHE_HOME"):
            home = os.path.expanduser("~")
            if home and not home.isascii():
                self.notices.append(
                    f"home directory {home!r} contains non-ASCII characters and "
                    "PADDLE_PDX_CACHE_HOME is unset; PaddleX may fail to read its model "
                    "config. Set PADDLE_PDX_CACHE_HOME to an ASCII path, for example: "
                    r"set PADDLE_PDX_CACHE_HOME=D:\paddlex_cache"
                )

    def _engine_for(self, language: str) -> Any:
        if language not in self._engines:
            options: dict[str, Any] = {"lang": language}
            if not self.use_doc_preprocessing:
                options.update(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            self._engines[language] = self._factory(**options)
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
                texts = first_present(page, "rec_texts")
                scores = first_present(page, "rec_scores")
                boxes = first_present(page, "rec_polys", "dt_polys")
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

    @property
    def primary(self) -> OCREngine | None:
        """The wrapped engine, for diagnostics and tests."""
        return self._primary

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
    tesseract_settings = {
        **settings,
        "profile": config.tesseract_profile,
        "scale": config.tesseract_scale,
        "tessdata_dir": config.tesseract_tessdata_dir,
        "transport": config.tesseract_transport,
        "library_path": config.tesseract_library_path,
    }
    # Only PaddleOCR has a document pipeline; Tesseract must not receive this.
    paddle_settings = {**settings, "use_doc_preprocessing": config.use_doc_preprocessing}
    notices: list[str] = []

    if config.engine == "tesseract":
        return EngineSelection(TesseractOCREngine(**tesseract_settings), notices)

    try:
        primary = PaddleOCREngine(notices=notices, **paddle_settings)
    except OcrError as exc:
        notices.append(f"PaddleOCR unavailable ({exc}); falling back to Tesseract.")
        if config.fallback_engine == "none":
            raise
        return EngineSelection(TesseractOCREngine(**tesseract_settings), notices)

    if config.fallback_engine == "tesseract":
        # The engine may only fail on the first call, so keep Tesseract ready.
        return EngineSelection(
            FallbackOCREngine(primary, TesseractOCREngine(**tesseract_settings), notices), notices
        )
    return EngineSelection(primary, notices)
