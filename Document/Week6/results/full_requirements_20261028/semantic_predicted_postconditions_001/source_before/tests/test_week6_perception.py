"""Offline timing, fresh-frame and exact candidate compatibility regressions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from gui_agent.config import Config
from gui_agent.perception import ui_detection
from gui_agent.perception.capture import CaptureResult
from gui_agent.perception.ocr import EngineSelection, OcrError, OcrOutput
from gui_agent.runtime import observation
from gui_agent.runtime.schemas import ObservationSnapshot
from gui_agent.schemas import BoundingBox, ScreenInfo, UIElement

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Document/Week6/results/perception_benchmark"


@pytest.fixture(scope="module")
def legacy_detector():
    """The measured pre-change implementation, preserved with its actual bytes."""
    source = (EVIDENCE / "baseline_ui_detection.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "65a5dd206cd472d18c4408ffcc516c4a4f399e519f8dec9175700dfa51b8276a"
    )
    spec = importlib.util.spec_from_file_location(
        "gui_agent.perception._week6_compatibility_reference", EVIDENCE / "baseline_ui_detection.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8, 1.0, 1.1])
def test_deduplication_retains_legacy_order_and_nested_controls(legacy_detector, threshold):
    rng = random.Random(42)
    boxes = [BoundingBox(left=0, top=0, right=600, bottom=400)]
    for _ in range(120):
        x, y = rng.randrange(500), rng.randrange(300)
        boxes.append(BoundingBox(left=x, top=y, right=x + rng.randrange(1, 200),
                                 bottom=y + rng.randrange(1, 160)))
    boxes.extend([boxes[8], boxes[1], boxes[8]])
    before = [box.model_dump() for box in boxes]
    actual = ui_detection._deduplicate(boxes, threshold)
    expected = legacy_detector._deduplicate(boxes, threshold)
    assert actual == expected
    assert all(first is second for first, second in zip(actual, expected))
    assert before == [box.model_dump() for box in boxes], "input order/geometry must not mutate"


@pytest.mark.parametrize("settings", [
    {}, {"max_candidates": 1}, {"min_rectangularity": 0.9},
    {"exclusion_threshold": 0.0}, {"min_area": 250, "max_area_ratio": .1},
])
def test_measured_bilingual_fixtures_have_identical_ordered_candidates(legacy_detector, settings):
    manifest = json.loads((EVIDENCE / "fixture_manifest.json").read_text(encoding="utf-8"))
    for fixture in manifest["fixtures"]:
        path = ROOT / fixture["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fixture["sha256"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        exclude = [BoundingBox(left=35, top=100, right=105, bottom=132)]
        priority = BoundingBox(left=0, top=0, right=520, bottom=420)
        assert ui_detection.detect_ui_candidates(
            image, exclude=exclude, priority_region=priority, **settings,
        ) == legacy_detector.detect_ui_candidates(
            image, exclude=exclude, priority_region=priority, **settings,
        )


def test_new_frame_never_reuses_previous_contour_geometry(legacy_detector):
    image = Image.new("RGB", (500, 300), "white")
    ImageDraw.Draw(image).rectangle((20, 40, 160, 120), outline="black", width=2)
    first = ui_detection.detect_ui_candidates(image)
    image.paste("white", (0, 0, 500, 300))
    ImageDraw.Draw(image).rectangle((260, 170, 460, 250), outline="black", width=2)
    second = ui_detection.detect_ui_candidates(image)
    assert first and second
    assert first != second
    assert second == legacy_detector.detect_ui_candidates(image)
    assert all(item.center.x > 250 and item.center.y > 160 for item in second)


def test_observation_stages_split_cold_initialization_and_read_every_frame(monkeypatch):
    clock = {"now": 0.0}
    calls = {"capture": 0, "factory": 0}
    screen = ScreenInfo(screenshot_width=50, screenshot_height=30,
                        control_width=50, control_height=30)

    def capture(*args, **kwargs):
        calls["capture"] += 1
        clock["now"] += 1
        return CaptureResult(Image.new("RGB", (50, 30), (calls["capture"], 0, 0)),
                             screen, datetime.now(UTC))

    class Engine:
        def recognize(self, image):
            clock["now"] += 3
            value = image.getpixel((0, 0))[0]
            return OcrOutput([UIElement(text=f"fresh-{value}",
                                       bounding_box=BoundingBox(left=1, top=1, right=20, bottom=10),
                                       confidence=.9)], "test", 3000)

    def factory(config):
        calls["factory"] += 1
        clock["now"] += 2
        return EngineSelection(Engine())

    def detect(image, **kwargs):
        clock["now"] += 4
        return []

    monkeypatch.setattr(observation, "capture_monitor", capture)
    monkeypatch.setattr(observation, "create_ocr_engine", factory)
    monkeypatch.setattr(observation, "detect_ui_candidates", detect)
    monkeypatch.setattr(observation.time, "perf_counter", lambda: clock["now"])
    service = observation.ObservationService(Config())
    select = service._select

    def selection(elements):
        clock["now"] += .5
        return select(elements)

    monkeypatch.setattr(service, "_select", selection)
    first, second = service.observe(), service.observe()
    assert first.stage_timings_ms == {
        "capture": 1000, "engine_init": 2000, "ocr": 3000,
        "ui_detection": 4000, "selection": 500,
    }
    assert first.processing_time_ms == 10500
    assert second.stage_timings_ms == {**first.stage_timings_ms, "engine_init": 0}
    assert second.processing_time_ms == 8500
    assert calls == {"capture": 2, "factory": 1}
    assert first.elements[0].text == "fresh-1" and second.elements[0].text == "fresh-2"
    assert first.elements[0].element_id != second.elements[0].element_id
    assert ObservationSnapshot.model_validate_json(second.model_dump_json()) == second


@pytest.mark.parametrize("failure", ["factory", "ocr", "ui", "disabled"])
def test_disabled_or_failed_stages_keep_errors_and_explicit_timing(monkeypatch, failure):
    screen = ScreenInfo(screenshot_width=50, screenshot_height=30,
                        control_width=50, control_height=30)
    monkeypatch.setattr(observation, "capture_monitor", lambda *a, **k: CaptureResult(
        Image.new("RGB", (50, 30), "white"), screen, datetime.now(UTC)))

    class Engine:
        def recognize(self, image):
            if failure == "ocr":
                raise OcrError("recognition broke")
            return OcrOutput([], "test", 0)

    def factory(config):
        if failure == "factory":
            raise OcrError("factory broke")
        return EngineSelection(Engine())

    def detect(*args, **kwargs):
        if failure == "ui":
            raise RuntimeError("contours broke")
        return []

    monkeypatch.setattr(observation, "create_ocr_engine", factory)
    monkeypatch.setattr(observation, "detect_ui_candidates", detect)
    config = Config()
    config.perception.ui_detection.enabled = failure != "disabled"
    snapshot = observation.ObservationService(config).observe()
    assert all(value >= 0 for value in snapshot.stage_timings_ms.values())
    assert sum(snapshot.stage_timings_ms.values()) <= snapshot.processing_time_ms
    if failure == "factory":
        assert snapshot.stage_timings_ms["ocr"] == 0
    if failure == "disabled":
        assert snapshot.stage_timings_ms["ui_detection"] == 0
        assert not snapshot.errors
    else:
        assert any("broke" in value for value in snapshot.errors)


def test_legacy_observation_without_stage_timings_loads():
    snapshot = ObservationSnapshot.model_validate({
        "observation_id": "old-frame", "captured_at": "2026-10-01T00:00:00Z",
        "screen_info": {"screenshot_width": 10, "screenshot_height": 10,
                        "control_width": 10, "control_height": 10},
    })
    assert snapshot.stage_timings_ms == {}
