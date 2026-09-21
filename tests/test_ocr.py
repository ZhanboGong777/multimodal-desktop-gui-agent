"""Tests for the OCR backends. No test touches the real desktop or Tesseract."""

from __future__ import annotations

import pytest
from PIL import Image

from gui_agent.config import OcrConfig
from gui_agent.perception import ocr as ocr_module
from gui_agent.perception.ocr import (
    OCREngine,
    OcrError,
    PaddleOCREngine,
    TesseractOCREngine,
    box_to_bounds,
    clean_text,
    create_ocr_engine,
    make_element,
)


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  Hello \n world\t ") == "Hello world"
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_make_element_skips_degenerate_boxes() -> None:
    assert make_element("x", 0, 0, 0, 10, 0.9) is None
    assert make_element("x", 0, 0, 10, 0, 0.9) is None
    element = make_element("x", 10, 20, 30, 40, 0.9)
    assert element is not None
    assert element.bounding_box.right == 40
    assert element.bounding_box.bottom == 60


def test_make_element_clamps_confidence_into_range() -> None:
    element = make_element("x", 0, 0, 5, 5, 3.0)
    assert element is not None
    assert element.confidence == 1.0


def test_box_to_bounds_converts_a_quad() -> None:
    quad = [[10, 20], [50, 22], [48, 60], [12, 58]]
    left, top, width, height = box_to_bounds(quad)
    assert (left, top) == (10, 20)
    assert width == 40
    assert height == 40


def test_tesseract_engine_normalises_results(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    payload = {
        "text": ["", "Hello", "   ", "World", "ZeroWidth"],
        "conf": ["-1", "95.5", "80", "10", "99"],
        "left": [0, 10, 0, 50, 70],
        "top": [0, 20, 0, 60, 80],
        "width": [0, 80, 0, 60, 0],
        "height": [0, 20, 0, 18, 10],
    }
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: payload)

    output = TesseractOCREngine(default_min_confidence=0.5).recognize(
        Image.new("RGB", (200, 100), "white")
    )
    assert [element.text for element in output.elements] == ["Hello"]
    assert output.engine == "tesseract"
    assert output.elapsed_ms >= 0.0
    assert output.elements[0].confidence == pytest.approx(0.955)


def test_tesseract_engine_wraps_backend_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tesseract binary missing")

    monkeypatch.setattr(pytesseract, "image_to_data", boom)
    with pytest.raises(OcrError):
        TesseractOCREngine().recognize(Image.new("RGB", (10, 10)))


def test_create_ocr_engine_builds_the_configured_backend() -> None:
    selection = create_ocr_engine(OcrConfig(engine="tesseract"))
    assert selection.engine.name == "tesseract"
    assert selection.notices == []


def test_create_ocr_engine_falls_back_when_paddleocr_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPaddleOCR(PaddleOCREngine):
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OcrError("not installed")

    monkeypatch.setattr(ocr_module, "PaddleOCREngine", BrokenPaddleOCR)
    selection = create_ocr_engine(OcrConfig(engine="paddleocr", fallback_engine="tesseract"))
    assert selection.engine.name == "tesseract"
    assert selection.notices
    assert "falling back" in selection.notices[0]


def test_create_ocr_engine_raises_when_fallback_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenPaddleOCR(PaddleOCREngine):
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OcrError("not installed")

    monkeypatch.setattr(ocr_module, "PaddleOCREngine", BrokenPaddleOCR)
    with pytest.raises(OcrError):
        create_ocr_engine(OcrConfig(engine="paddleocr", fallback_engine="none"))


def test_paddle_engine_parses_both_result_shapes() -> None:
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    version_2 = [[[[[10, 10], [60, 10], [60, 30], [10, 30]], ("Hello", 0.9)]]]
    version_3 = [
        {
            "rec_texts": ["World"],
            "rec_scores": [0.8],
            "rec_polys": [[[5, 5], [70, 5], [70, 25], [5, 25]]],
        }
    ]
    assert [element.text for element in engine._parse(version_2, 0.5)] == ["Hello"]
    assert [element.text for element in engine._parse(version_3, 0.5)] == ["World"]


def test_paddle_engine_applies_the_confidence_threshold() -> None:
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    payload = [[[[[0, 0], [10, 0], [10, 10], [0, 10]], ("weak", 0.2)]]]
    assert engine._parse(payload, 0.5) == []


def test_language_codes_are_translated_for_tesseract() -> None:
    from gui_agent.perception.ocr import TesseractOCREngine, to_tesseract_languages

    assert to_tesseract_languages(["en"]) == ["eng"]
    assert to_tesseract_languages(["en", "zh"]) == ["eng", "chi_sim"]
    assert to_tesseract_languages(["eng"]) == ["eng"]
    # Unknown codes are passed through unchanged rather than silently dropped.
    assert to_tesseract_languages(["deu"]) == ["deu"]
    assert TesseractOCREngine().default_languages == ("en",)


def test_tesseract_engine_sends_translated_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    seen: dict[str, object] = {}

    def capture(image: object, lang: str = "", output_type: object = None) -> dict:
        seen["lang"] = lang
        return {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}

    monkeypatch.setattr(pytesseract, "image_to_data", capture)
    TesseractOCREngine(default_languages=["en"]).recognize(Image.new("RGB", (5, 5)))
    assert seen["lang"] == "eng"


def test_fallback_engine_switches_on_first_runtime_failure() -> None:
    from gui_agent.perception.ocr import FallbackOCREngine, OcrOutput

    class FailingPrimary:
        name = "paddleocr"

        def __init__(self) -> None:
            self.calls = 0

        def recognize(self, image: object, **kwargs: object) -> OcrOutput:
            self.calls += 1
            raise OcrError("Engine 'paddle_static' is unavailable")

    class WorkingFallback:
        name = "tesseract"

        def __init__(self) -> None:
            self.calls = 0
            self.default_languages = ("en",)
            self.default_min_confidence = 0.5

        def recognize(self, image: object, **kwargs: object) -> OcrOutput:
            self.calls += 1
            return OcrOutput(elements=[], engine=self.name, elapsed_ms=0.0)

    primary, fallback = FailingPrimary(), WorkingFallback()
    notices: list[str] = []
    engine = FallbackOCREngine(primary, fallback, notices)  # type: ignore[arg-type]

    assert engine.name == "paddleocr"
    assert not engine.using_fallback

    output = engine.recognize(Image.new("RGB", (5, 5)))
    assert output.engine == "tesseract"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert engine.name == "tesseract"
    assert engine.using_fallback
    assert notices and "switching to tesseract" in notices[0]

    # The primary must not be retried once it has failed.
    engine.recognize(Image.new("RGB", (5, 5)))
    assert primary.calls == 1
    assert fallback.calls == 2


def test_factory_wraps_paddleocr_with_the_tesseract_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gui_agent.perception.ocr import FallbackOCREngine, PaddleOCREngine

    class StubPaddleOCR(PaddleOCREngine):
        def __init__(self, *args: object, **kwargs: object) -> None:
            OCREngine.__init__(self, kwargs.get("default_languages", ("en",)))

        def recognize(self, image: object, **kwargs: object) -> object:
            from gui_agent.perception.ocr import OcrOutput

            return OcrOutput(elements=[], engine=self.name, elapsed_ms=0.0)

    monkeypatch.setattr(ocr_module, "PaddleOCREngine", StubPaddleOCR)
    selection = create_ocr_engine(OcrConfig(engine="paddleocr", fallback_engine="tesseract"))
    assert isinstance(selection.engine, FallbackOCREngine)
    assert selection.engine.name == "paddleocr"


def test_factory_returns_the_bare_engine_when_fallback_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gui_agent.perception.ocr import PaddleOCREngine

    class StubPaddleOCR(PaddleOCREngine):
        def __init__(self, *args: object, **kwargs: object) -> None:
            OCREngine.__init__(self, kwargs.get("default_languages", ("en",)))

    monkeypatch.setattr(ocr_module, "PaddleOCREngine", StubPaddleOCR)
    selection = create_ocr_engine(OcrConfig(engine="paddleocr", fallback_engine="none"))
    assert selection.engine.name == "paddleocr"


def test_paddle_v3_parser_accepts_numpy_arrays() -> None:
    """PaddleOCR can hand back NumPy arrays; `or` on those raises ValueError."""
    np = pytest.importorskip("numpy")
    engine = PaddleOCREngine()

    payload = [
        {
            "rec_texts": np.array(["Hello", "World"]),
            "rec_scores": np.array([0.9, 0.8]),
            "rec_polys": np.array(
                [
                    [[5, 5], [70, 5], [70, 25], [5, 25]],
                    [[5, 30], [70, 30], [70, 50], [5, 50]],
                ]
            ),
        }
    ]

    elements = engine._parse(payload, 0.5)
    assert [element.text for element in elements] == ["Hello", "World"]
    assert elements[0].bounding_box.left == 5
    assert elements[1].bounding_box.top == 30


def test_paddle_v3_parser_handles_missing_polygons() -> None:
    """A page without any polygon key must degrade to no elements, not crash."""
    np = pytest.importorskip("numpy")
    engine = PaddleOCREngine()

    # rec_polys explicitly None and dt_polys absent.
    assert engine._parse([{"rec_texts": np.array(["x"]), "rec_polys": None}], 0.5) == []
    assert engine._parse([{"rec_texts": ["x"]}], 0.5) == []
