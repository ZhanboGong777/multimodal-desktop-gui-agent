"""Screen OCR integration contracts without a real OCR process or desktop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from gui_agent.config import OcrConfig
from gui_agent.perception.ocr import (
    OcrError,
    TesseractOCREngine,
    create_ocr_engine,
    prepare_screen_text,
)


def payload(words: list[tuple[str, int, int, int, int, float, int]]) -> dict:
    return {
        "text": [w[0] for w in words], "left": [w[1] for w in words],
        "top": [w[2] for w in words], "width": [w[3] for w in words],
        "height": [w[4] for w in words], "conf": [w[5] for w in words],
        "block_num": [1 for _ in words], "par_num": [1 for _ in words],
        "line_num": [w[6] for w in words],
    }


def test_native_default_preserves_original_image_and_backend_arguments(monkeypatch) -> None:
    import pytesseract

    original = Image.new("RGB", (103, 71), (32, 35, 40))
    calls = []

    def recognize(image, **kwargs):
        calls.append((image, kwargs))
        return payload([("Exact Label", 7, 11, 54, 12, 95, 1)])

    monkeypatch.setattr(pytesseract, "image_to_data", recognize)
    output = TesseractOCREngine().recognize(original)
    assert calls == [(original, {"lang": "eng", "output_type": pytesseract.Output.DICT})]
    assert output.elements[0].bounding_box.model_dump() == {
        "left": 7, "top": 11, "right": 61, "bottom": 23}


def test_screen_scales_current_pixels_and_maps_rounded_bounds_to_original(monkeypatch) -> None:
    import pytesseract

    seen = {}

    def recognize(image, **kwargs):
        seen.update(size=image.size, mode=image.mode, **kwargs)
        return payload([("Open Project", 13, 19, 102, 23, 91, 1)])

    monkeypatch.setattr(pytesseract, "image_to_data", recognize)
    result = TesseractOCREngine(profile="screen", scale=1.5).recognize(
        Image.new("RGB", (101, 81), "white"))
    assert seen["size"] == (152, 122)
    assert seen["mode"] == "L"
    assert seen["config"] == "--psm 11 --oem 1"
    assert result.elements[0].bounding_box.model_dump() == {
        "left": 8, "top": 12, "right": 77, "bottom": 28}


def test_screen_clamps_backend_overshoot_and_drops_out_of_frame_box(monkeypatch) -> None:
    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: payload([
        ("edge", -2, 6, 205, 190, 90, 1), ("outside", 205, 205, 20, 20, 90, 2)]))
    result = TesseractOCREngine(profile="screen").recognize(Image.new("RGB", (100, 80)))
    assert [item.text for item in result.elements] == ["edge"]
    assert result.elements[0].bounding_box.model_dump() == {
        "left": 0, "top": 3, "right": 100, "bottom": 80}


def test_screen_rejoins_cjk_word_segments_without_dropping_latin_spaces(monkeypatch) -> None:
    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: payload([
        ("保存", 10, 10, 25, 20, 95, 1), ("修改", 38, 10, 25, 20, 93, 1),
        ("Open", 10, 60, 35, 20, 99, 2), ("Project", 51, 60, 65, 20, 98, 2)]))
    image = Image.new("RGB", (300, 120))
    assert [e.text for e in TesseractOCREngine(profile="screen").recognize(image).elements] == [
        "保存修改", "Open Project"]
    assert [e.text for e in TesseractOCREngine().recognize(image).elements] == [
        "保存 修改", "Open Project"]


def test_screen_does_not_merge_separate_controls_or_hallucinate_missing_text(monkeypatch) -> None:
    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: payload([
        ("保存", 10, 10, 25, 20, 95, 1), ("取消", 100, 10, 25, 20, 93, 1)]))
    result = TesseractOCREngine(profile="screen").recognize(Image.new("RGB", (300, 120)))
    assert [e.text for e in result.elements] == ["保存", "取消"]


def test_screen_keeps_confidence_filter_and_weakest_word_confidence(monkeypatch) -> None:
    import pytesseract

    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: payload([
        ("Keep", 10, 10, 30, 20, 95, 1), ("Label", 45, 10, 35, 20, 74, 1),
        ("noise", 10, 80, 30, 20, 49, 2)]))
    result = TesseractOCREngine(profile="screen").recognize(Image.new("RGB", (300, 120)))
    assert [e.text for e in result.elements] == ["Keep Label"]
    assert result.elements[0].confidence == .74


def test_screen_accepts_gray_numpy_and_preserves_input(monkeypatch) -> None:
    import pytesseract

    original = np.arange(99, dtype=np.uint8).reshape((9, 11))
    before = original.copy()
    seen = []
    monkeypatch.setattr(pytesseract, "image_to_data",
                        lambda image, **k: (seen.append(image.size) or payload([])))
    assert TesseractOCREngine(profile="screen").recognize(original).elements == []
    assert seen == [(22, 18)]
    np.testing.assert_array_equal(original, before)


def test_resource_selection_does_not_mutate_global_environment(monkeypatch) -> None:
    import os

    import pytesseract

    monkeypatch.setenv("TESSDATA_PREFIX", "unrelated")
    seen = []
    monkeypatch.setattr(pytesseract, "image_to_data",
                        lambda image, **k: (seen.append(k) or payload([])))
    TesseractOCREngine(profile="screen", tessdata_dir="D:/ocr/tessdata").recognize(
        Image.new("RGB", (50, 40)))
    assert "--tessdata-dir" in seen[0]["config"]
    assert "D:/ocr/tessdata" in seen[0]["config"]
    assert os.environ["TESSDATA_PREFIX"] == "unrelated"


def test_factory_routes_screen_config_to_tesseract() -> None:
    selected = create_ocr_engine(OcrConfig(
        engine="tesseract", languages=["en", "zh"], tesseract_profile="screen",
        tesseract_scale=2.5, tesseract_tessdata_dir=Path("D:/ocr/tessdata")))
    assert isinstance(selected.engine, TesseractOCREngine)
    assert selected.engine.profile == "screen"
    assert selected.engine.scale == 2.5
    assert selected.engine.default_languages == ("en", "zh")
    assert selected.engine.tessdata_dir == Path("D:/ocr/tessdata")


def test_screen_language_resource_failure_remains_visible(monkeypatch) -> None:
    import pytesseract

    def fail(*args, **kwargs):
        raise RuntimeError("Failed loading language chi_sim")

    monkeypatch.setattr(pytesseract, "image_to_data", fail)
    with pytest.raises(OcrError, match="Failed loading language chi_sim"):
        TesseractOCREngine(profile="screen", default_languages=["en", "zh"]).recognize(
            Image.new("RGB", (100, 50)))


def test_configuration_rejects_unbounded_scale_and_unknown_profile() -> None:
    for values in ({"tesseract_scale": .5}, {"tesseract_scale": 3.1},
                   {"tesseract_profile": "untrusted"}):
        with pytest.raises(ValidationError):
            OcrConfig(**values)
    for values in ({"scale": 4}, {"profile": "untrusted"}, {"tessdata_dir": 'a"b'}):
        with pytest.raises(ValueError):
            TesseractOCREngine(**values)


def test_border_suppression_preserves_short_text_strokes_and_original() -> None:
    image = Image.new("RGB", (300, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 180, 100), outline="black")
    draw.line((60, 45, 60, 57), fill="black", width=2)
    before = np.array(image)
    processed = np.array(prepare_screen_text(image))
    assert processed[20, 60] == 255  # long horizontal control border
    assert processed[50, 20] == 255  # long vertical control border
    assert processed[50, 60] == 0  # short glyph stroke
    np.testing.assert_array_equal(np.array(image), before)


def test_dark_theme_normalizes_to_same_ink_geometry() -> None:
    light = Image.new("RGB", (200, 100), (240, 240, 240))
    dark = Image.new("RGB", (200, 100), (15, 15, 15))
    ImageDraw.Draw(light).line((30, 30, 30, 45), fill=(60, 60, 60), width=2)
    ImageDraw.Draw(dark).line((30, 30, 30, 45), fill=(195, 195, 195), width=2)
    np.testing.assert_array_equal(np.array(prepare_screen_text(light)), np.array(prepare_screen_text(dark)))


def test_windows_resource_path_restriction_is_explicit(monkeypatch) -> None:
    from gui_agent.perception import ocr

    monkeypatch.setattr(ocr.sys, "platform", "win32")
    with pytest.raises(ValueError, match="without whitespace"):
        TesseractOCREngine(tessdata_dir="D:/path with spaces")
