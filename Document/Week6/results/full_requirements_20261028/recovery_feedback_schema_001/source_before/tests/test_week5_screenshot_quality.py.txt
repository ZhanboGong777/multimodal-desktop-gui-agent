"""CPU-only frame integrity tests; none assert message readability or Send safety."""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime

import pytest
from PIL import Image, ImageDraw

from gui_agent.runtime.schemas import ObservationSnapshot
from gui_agent.runtime.screenshot_quality import assess_screenshot_quality, window_identity
from gui_agent.schemas import BoundingBox, ScreenInfo


def _source_frame(tmp_path, **updates):
    path = tmp_path / "actual-source.png"
    image = Image.new("RGB", (200, 150), "#888888")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 189, 139), fill="white", outline="black", width=2)
    draw.text((20, 20), "Actual header", fill="black")
    image.save(path)
    snapshot = ObservationSnapshot(
        observation_id="quality-0001", captured_at=datetime.now(UTC), image_path=str(path),
        screen_info=ScreenInfo(screenshot_width=200, screenshot_height=150,
                               control_width=200, control_height=150),
        window_id="actual:1", window_title="WeChat", window_class="Qt51514QWindowIcon",
        window_bounds=BoundingBox(left=10, top=10, right=190, bottom=140), foreground_stable=True,
    )
    return snapshot.model_copy(update=updates)


def test_quality_pass_only_proves_integrity_and_preserves_actual_source(tmp_path):
    snapshot = _source_frame(tmp_path)
    path = tmp_path / "actual-source.png"
    before = path.read_bytes()
    result = assess_screenshot_quality(snapshot)
    assert result.outcome == "passed" and result.code == "basic_frame_integrity_passed"
    assert result.evidence["source_image_bytes"] == len(before)
    assert result.evidence["source_image_sha256"] == hashlib.sha256(before).hexdigest()
    assert result.evidence["source_image_size"] == [200, 150]
    assert result.evidence["semantic_authorization"] is False
    assert result.evidence["send_authorization"] is False
    assert result.evidence["cursor_occlusion"].startswith("unknown:")
    assert result.evidence["arbitrary_overlay_detection"] == "unsupported"
    assert path.read_bytes() == before


@pytest.mark.parametrize("title", ["ChatGPT", "Codex", "  cOdEx  "])
def test_exact_known_host_title_and_class_block_its_actual_foreground(tmp_path, title):
    snapshot = _source_frame(tmp_path, window_title=title, window_class="Chrome_WidgetWin_1")
    result = assess_screenshot_quality(snapshot)
    assert result.outcome == "blocked" and result.code == "foreground_application_obstruction"
    assert result.evidence["window_title"] == title
    assert result.evidence["foreground_match_rule"].startswith("exact host title")


@pytest.mark.parametrize("title,class_name", [
    ("微信 ChatGPT交流", "Chrome_WidgetWin_1"),
    ("ChatGPT customer chat", "Chrome_WidgetWin_1"),
    ("project - Codex examples", "Chrome_WidgetWin_1"),
    ("ChatGPT", "Qt51514QWindowIcon"),
    ("Codex", "Notepad"),
    ("΢��", "Qt51514QWindowIcon"),
])
def test_title_substrings_or_different_class_do_not_prove_host_obstruction(tmp_path, title, class_name):
    result = assess_screenshot_quality(_source_frame(tmp_path, window_title=title, window_class=class_name))
    assert result.outcome == "passed", "semantic verification must still decide the actual target app"


@pytest.mark.parametrize("field,value", [
    ("window_id", ""), ("window_title", ""), ("window_class", ""), ("window_bounds", None),
])
def test_missing_foreground_evidence_is_blocked_not_retryable(tmp_path, field, value):
    result = assess_screenshot_quality(_source_frame(tmp_path, **{field: value}))
    assert result.outcome == "blocked" and result.code == "foreground_evidence_missing"


def test_valid_frame_captured_during_foreground_change_is_retryable(tmp_path):
    result = assess_screenshot_quality(_source_frame(tmp_path, foreground_stable=False))
    assert result.outcome == "retryable" and result.code == "foreground_capture_unstable"
    assert result.evidence["source_image_sha256"]


def test_known_host_is_blocked_even_when_foreground_capture_is_unstable(tmp_path):
    result = assess_screenshot_quality(_source_frame(
        tmp_path, window_title="ChatGPT", window_class="Chrome_WidgetWin_1", foreground_stable=False,
    ))
    assert result.code == "foreground_application_obstruction" and result.outcome == "blocked"


@pytest.mark.parametrize("bounds", [
    BoundingBox(left=-1, top=10, right=190, bottom=140),
    BoundingBox(left=10, top=10, right=201, bottom=140),
    BoundingBox(left=10, top=10, right=190, bottom=151),
])
def test_clipped_window_is_blocked_instead_of_cropping_missing_evidence(tmp_path, bounds):
    result = assess_screenshot_quality(_source_frame(tmp_path, window_bounds=bounds))
    assert result.code == "foreground_window_clipped" and result.outcome == "blocked"


def test_reported_capture_errors_block_without_retrying(tmp_path):
    result = assess_screenshot_quality(_source_frame(tmp_path, errors=["OCR failed"], foreground_stable=False))
    assert result.code == "observation_degraded" and result.outcome == "blocked"
    assert result.evidence["observation_errors"] == ["OCR failed"]


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 1.1])
def test_inconsistent_control_geometry_cannot_authorize_a_frame(tmp_path, scale):
    snapshot = _source_frame(tmp_path)
    snapshot.screen_info = snapshot.screen_info.model_copy(update={"scale_x": scale})
    result = assess_screenshot_quality(snapshot)
    assert result.code == "screen_scale_inconsistent" and result.outcome == "blocked"


def test_valid_scaled_monitor_geometry_is_supported(tmp_path):
    snapshot = _source_frame(tmp_path)
    snapshot.screen_info = ScreenInfo(screenshot_width=200, screenshot_height=150,
                                     control_width=100, control_height=75, monitor_left=-100)
    assert assess_screenshot_quality(snapshot).outcome == "passed"


def test_missing_source_image_is_blocked_even_for_unstable_capture(tmp_path):
    result = assess_screenshot_quality(_source_frame(tmp_path, image_path=None, foreground_stable=False))
    assert result.code == "source_image_missing" and result.outcome == "blocked"


@pytest.mark.parametrize("invalid_source", ["missing", "corrupt"])
def test_unreadable_source_is_blocked_with_original_read_error(tmp_path, invalid_source):
    snapshot = _source_frame(tmp_path)
    path = tmp_path / "actual-source.png"
    if invalid_source == "missing":
        path.unlink()
    else:
        path.write_bytes(b"actual corrupt PNG data")
    result = assess_screenshot_quality(snapshot)
    assert result.code == "source_image_unreadable" and result.outcome == "blocked"
    assert result.evidence["source_error"]


def test_source_dimensions_must_match_actual_recorded_monitor(tmp_path):
    snapshot = _source_frame(tmp_path)
    Image.new("RGB", (201, 150), "white").save(snapshot.image_path)
    result = assess_screenshot_quality(snapshot)
    assert result.code == "source_image_dimensions_mismatch" and result.outcome == "blocked"
    assert result.evidence["source_image_size"] == [201, 150]


def test_uniform_entire_foreground_window_is_retryable(tmp_path):
    snapshot = _source_frame(tmp_path)
    Image.new("RGB", (200, 150), "white").save(snapshot.image_path)
    result = assess_screenshot_quality(snapshot)
    assert result.code == "foreground_window_uniform" and result.outcome == "retryable"
    assert result.evidence["window_rgb_channel_ranges"] == [0, 0, 0]


def test_blank_editor_and_unproven_blue_overlay_remain_semantic_questions(tmp_path):
    snapshot = _source_frame(tmp_path)
    with Image.open(snapshot.image_path) as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((15, 95, 185, 135), fill="white")  # Valid empty editor.
        draw.ellipse((55, 20, 70, 35), fill="blue")  # Pixels do not prove an OS cursor.
        image.save(snapshot.image_path)
    result = assess_screenshot_quality(snapshot)
    assert result.outcome == "passed"
    assert result.evidence["cursor_occlusion"].startswith("unknown:")
    assert result.evidence["send_authorization"] is False


def test_decoded_frame_and_sha_refer_to_the_same_bytes_if_path_changes(tmp_path, monkeypatch):
    snapshot = _source_frame(tmp_path)
    raw = (tmp_path / "actual-source.png").read_bytes()
    original_open = Image.open
    def open_hashed_bytes(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        Image.new("RGB", (1, 1), "white").save(tmp_path / "actual-source.png")
        return original_open(source, *args, **kwargs)
    monkeypatch.setattr(Image, "open", open_hashed_bytes)
    result = assess_screenshot_quality(snapshot)
    assert result.outcome == "passed"
    assert result.evidence["source_image_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result.evidence["source_image_size"] == [200, 150]


def test_capture_identity_includes_title_class_handle_and_bounds(tmp_path):
    snapshot = _source_frame(tmp_path)
    assert window_identity(snapshot) == (
        "actual:1", "WeChat", "Qt51514QWindowIcon", (10, 10, 190, 140),
    )
    assert window_identity(snapshot.model_copy(update={"window_title": "Other chat"})) != window_identity(snapshot)
