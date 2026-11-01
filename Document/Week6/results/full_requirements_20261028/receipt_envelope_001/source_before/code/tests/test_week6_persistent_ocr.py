"""Persistent C API ownership, failure, isolation and factory contracts.

These tests replace the native ABI with recorded calls. The separate paired
benchmark exercises the installed library and checks complete OCR/UI output.
"""

from __future__ import annotations

import ctypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from gui_agent.config import OcrConfig
from gui_agent.perception import tesseract_api
from gui_agent.perception.ocr import OcrError, TesseractOCREngine, create_ocr_engine

PAGE = b"1\t1\t0\t0\t0\t0\t0\t0\t20\t12\t-1\t\n"
WORD = b"5\t1\t1\t1\t1\t1\t2\t3\t10\t6\t95.912345\tSample\n"


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class Native:
    def __init__(self):
        self.calls = []
        self.next_api = 10
        self.init_status = 0
        self.recognize_status = 0
        self.create_null = False
        self.text_null = False
        self.text = PAGE + WORD
        self.buffer = None
        self.active = 0
        self.max_active = 0
        self.counter_lock = threading.Lock()
        for name in (
            "TessBaseAPICreate", "TessBaseAPIInit2", "TessBaseAPISetPageSegMode",
            "TessBaseAPIClearAdaptiveClassifier", "TessBaseAPISetImage",
            "TessBaseAPIRecognize", "TessBaseAPIGetTsvText", "TessDeleteText",
            "TessBaseAPIClear", "TessBaseAPIEnd", "TessBaseAPIDelete",
        ):
            setattr(self, name, Function(lambda *args, name=name: self.invoke(name, args)))

    def invoke(self, name, args):
        self.calls.append((name, args))
        if name == "TessBaseAPICreate":
            self.next_api += 1
            return 0 if self.create_null else self.next_api
        if name == "TessBaseAPIInit2":
            return self.init_status
        if name == "TessBaseAPIRecognize":
            with self.counter_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(.002)
            with self.counter_lock:
                self.active -= 1
            return self.recognize_status
        if name == "TessBaseAPIGetTsvText":
            self.buffer = ctypes.create_string_buffer(self.text)
            return 0 if self.text_null else ctypes.addressof(self.buffer)
        return None

    def count(self, name):
        return sum(call == name for call, _ in self.calls)


@pytest.fixture
def native(monkeypatch):
    library = Native()
    monkeypatch.setattr(tesseract_api, "_library_path", lambda _value: "test-installed-library")
    monkeypatch.setattr(tesseract_api.sys, "platform", "test-native")
    monkeypatch.setattr(tesseract_api.ctypes, "CDLL", lambda _path: library)
    return library


def recognize(engine, language="eng"):
    return engine.image_to_data(
        Image.new("L", (20, 12), "white"), language=language,
        tessdata_dir="/test/tessdata", psm=11, oem=1,
    )


def test_two_frames_reuse_model_but_release_each_text_and_clear_state(native):
    engine = tesseract_api.PersistentTesseract()
    assert recognize(engine) == recognize(engine)
    assert native.count("TessBaseAPIInit2") == 1
    assert native.count("TessBaseAPIClearAdaptiveClassifier") == 2
    assert native.count("TessBaseAPIClear") == 2
    assert native.count("TessDeleteText") == 2
    assert native.count("TessBaseAPIEnd") == 0
    engine.close()
    engine.close()
    assert native.count("TessBaseAPIEnd") == native.count("TessBaseAPIDelete") == 1
    with pytest.raises(RuntimeError, match="closed"):
        recognize(engine)


def test_language_change_releases_previous_model(native):
    engine = tesseract_api.PersistentTesseract()
    recognize(engine)
    recognize(engine, "eng+chi_sim")
    assert native.count("TessBaseAPIInit2") == 2
    assert native.count("TessBaseAPIDelete") == 1
    init_calls = [args for name, args in native.calls if name == "TessBaseAPIInit2"]
    assert init_calls[0][2] == b"eng"
    assert init_calls[1][2] == b"eng+chi_sim"
    engine.close()
    assert native.count("TessBaseAPIDelete") == 2


def test_headerless_tsv_matches_wrapper_integer_confidence_and_empty_page(native):
    engine = tesseract_api.PersistentTesseract()
    result = recognize(engine)
    assert result["conf"] == [-1, 95]
    assert result["text"] == ["", "Sample"]
    assert result["left"] == [0, 2]
    native.text = PAGE
    assert recognize(engine)["text"] == [""]
    engine.close()


@pytest.mark.parametrize("text", [b"", b"wrong\theader\n", PAGE + b"5\ttoo\tfew\n", PAGE + b"\xff\n"])
def test_invalid_tsv_or_utf8_always_frees_pointer_and_clears_frame(native, text):
    native.text = text
    engine = tesseract_api.PersistentTesseract()
    with pytest.raises((RuntimeError, UnicodeDecodeError)):
        recognize(engine)
    assert native.count("TessDeleteText") == native.count("TessBaseAPIClear") == 1
    engine.close()


def test_recognition_error_clears_frame_without_freeing_unallocated_text(native):
    native.recognize_status = -1
    engine = tesseract_api.PersistentTesseract()
    with pytest.raises(RuntimeError, match="Recognize failed"):
        recognize(engine)
    assert native.count("TessBaseAPIClear") == 1
    assert native.count("TessDeleteText") == 0
    engine.close()


def test_null_text_clears_frame_without_freeing_null(native):
    native.text_null = True
    engine = tesseract_api.PersistentTesseract()
    with pytest.raises(RuntimeError, match="GetTsvText returned NULL"):
        recognize(engine)
    assert native.count("TessBaseAPIClear") == 1
    assert native.count("TessDeleteText") == 0
    engine.close()


def test_initialization_error_releases_api_and_retry_can_initialize(native):
    native.init_status = -1
    engine = tesseract_api.PersistentTesseract()
    with pytest.raises(RuntimeError, match="Init2 failed"):
        recognize(engine)
    assert native.count("TessBaseAPIDelete") == 1
    native.init_status = 0
    assert recognize(engine)["text"][-1] == "Sample"
    engine.close()
    assert native.count("TessBaseAPIDelete") == 2


def test_null_create_never_deletes_invalid_pointer(native):
    native.create_null = True
    engine = tesseract_api.PersistentTesseract()
    with pytest.raises(RuntimeError, match="Create returned NULL"):
        recognize(engine)
    engine.close()
    assert native.count("TessBaseAPIDelete") == 0


def test_same_instance_serializes_concurrent_native_access(native):
    engine = tesseract_api.PersistentTesseract()
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda _index: recognize(engine), range(8)))
    assert all(row["text"][-1] == "Sample" for row in rows)
    assert native.max_active == 1
    assert native.count("TessBaseAPIInit2") == 1
    assert native.count("TessBaseAPIClear") == 8
    engine.close()


@pytest.mark.parametrize("mode", ["LA", "P", "1", "I", "F"])
def test_invalid_pixel_channels_are_rejected_before_set_image(native, mode):
    engine = tesseract_api.PersistentTesseract()
    with pytest.raises(ValueError, match="grayscale/RGB/RGBA"):
        engine.image_to_data(Image.new(mode, (20, 12)), language="eng",
                             tessdata_dir=None, psm=11, oem=1)
    assert native.count("TessBaseAPISetImage") == 0
    engine.close()


def test_explicit_missing_library_fails_visibly_without_subprocess_fallback(tmp_path, monkeypatch):
    import pytesseract

    calls = []
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: calls.append(True))
    engine = TesseractOCREngine(transport="persistent", library_path=tmp_path / "missing.dll")
    with pytest.raises(OcrError, match="library does not exist"):
        engine.recognize(Image.new("RGB", (20, 12)))
    assert calls == []


def test_persistent_config_is_explicit_and_does_not_mutate_environment(monkeypatch, native):
    import os

    monkeypatch.setenv("TESSDATA_PREFIX", "independent-existing-value")
    selected = create_ocr_engine(OcrConfig(
        engine="tesseract", tesseract_transport="persistent",
        tesseract_library_path=Path("explicit.dll"),
        tesseract_tessdata_dir=Path("D:/ascii/tessdata"),
        tesseract_profile="screen", languages=["en", "zh"],
    ))
    assert isinstance(selected.engine, TesseractOCREngine)
    selected.engine.recognize(Image.new("RGB", (20, 12)))
    assert selected.engine.transport == "persistent"
    assert os.environ["TESSDATA_PREFIX"] == "independent-existing-value"
    selected.engine.close()


def test_unknown_transport_rejected_and_default_unchanged():
    assert OcrConfig().tesseract_transport == "subprocess"
    assert TesseractOCREngine().transport == "subprocess"
    with pytest.raises(ValidationError):
        OcrConfig(tesseract_transport="hidden-fallback")
    with pytest.raises(ValueError, match="transport"):
        TesseractOCREngine(transport="hidden-fallback")
