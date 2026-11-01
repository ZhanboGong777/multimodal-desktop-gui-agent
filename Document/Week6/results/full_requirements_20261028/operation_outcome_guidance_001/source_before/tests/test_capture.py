"""Tests for the capture module. MSS is replaced by a fake session.

These tests must never read the real screen, so both the MSS session and the
control-space probe are patched.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import pytest

from gui_agent.perception import capture as capture_module
from gui_agent.perception.capture import (
    CaptureError,
    CaptureRegion,
    average_capture_ms,
    capture_frames,
    capture_monitor,
    capture_only_fps,
    capture_region,
    effective_sequence_fps,
    list_monitors,
)


class FakeShot:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.rgb = bytes(width * height * 3)


class FakeSession:
    monitors: ClassVar[list[dict[str, int]]] = [
        {"left": 0, "top": 0, "width": 1920, "height": 2036},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 215, "top": 1080, "width": 1470, "height": 956},
    ]

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def grab(self, region: dict[str, int]) -> FakeShot:
        if self.fail:
            raise RuntimeError("no display available")
        return FakeShot(region["width"], region["height"])

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def fake_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch MSS and the control-space probe so nothing touches the desktop."""
    monkeypatch.setattr(capture_module.mss, "mss", lambda: FakeSession())
    monkeypatch.setattr(capture_module, "_control_size", lambda: (1920, 1080))


@pytest.fixture
def failing_mss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture_module.mss, "mss", lambda: FakeSession(fail=True))
    monkeypatch.setattr(capture_module, "_control_size", lambda: (1920, 1080))


def test_list_monitors_returns_the_session_monitors(fake_mss: None) -> None:
    monitors = list_monitors()
    assert len(monitors) == 3
    assert monitors[1]["width"] == 1920


def test_capture_monitor_returns_image_and_geometry(fake_mss: None) -> None:
    frame = capture_monitor(1)
    assert frame.image.size == (1920, 1080)
    assert frame.screen_info.screenshot_width == 1920
    assert frame.screen_info.control_width == 1920
    assert frame.screen_info.scale_x == pytest.approx(1.0)
    assert frame.image_path is None
    assert frame.capture_time_ms >= 0.0


def test_capture_monitor_records_scaled_control_size(fake_mss: None) -> None:
    frame = capture_monitor(1, control_size=(960, 540))
    assert frame.screen_info.control_width == 960
    assert frame.screen_info.scale_x == pytest.approx(0.5)


def test_capture_monitor_keeps_the_monitor_offset(fake_mss: None) -> None:
    frame = capture_monitor(2, control_size=(1470, 956))
    assert frame.screen_info.monitor_left == 215
    assert frame.screen_info.monitor_top == 1080


def test_capture_monitor_saves_when_requested(fake_mss: None, tmp_path: Any) -> None:
    frame = capture_monitor(1, output_directory=tmp_path, save=True)
    assert frame.image_path is not None
    assert frame.image_path.exists()


@pytest.mark.parametrize("index", [0, 5, -1])
def test_capture_monitor_rejects_invalid_index(fake_mss: None, index: int) -> None:
    with pytest.raises(CaptureError):
        capture_monitor(index)


def test_capture_monitor_wraps_backend_failures(failing_mss: None) -> None:
    with pytest.raises(CaptureError):
        capture_monitor(1)


def test_capture_region_keeps_the_absolute_offset(fake_mss: None) -> None:
    frame = capture_region(CaptureRegion(left=100, top=50, width=300, height=200))
    assert frame.image.size == (300, 200)
    assert frame.screen_info.monitor_left == 100
    assert frame.screen_info.monitor_top == 50
    assert frame.screen_info.scale_x == pytest.approx(1.0)


def test_capture_region_rejects_a_degenerate_size(fake_mss: None) -> None:
    with pytest.raises(CaptureError):
        capture_region(CaptureRegion(left=0, top=0, width=0, height=10))


def test_capture_frames_returns_the_requested_count(fake_mss: None) -> None:
    frames = capture_frames(3, 0.0, monitor_index=1)
    assert len(frames) == 3
    assert [frame.metadata["frame_index"] for frame in frames] == [0, 1, 2]


def test_capture_frames_statistics(fake_mss: None) -> None:
    frames = capture_frames(3, 0.0, monitor_index=1)
    assert average_capture_ms(frames) > 0.0
    assert capture_only_fps(frames) > 0.0
    assert average_capture_ms([]) == 0.0
    assert capture_only_fps([]) == 0.0


@pytest.mark.parametrize(
    "count, interval",
    [(0, 0.0), (-1, 0.0), (2, -0.5)],
)
def test_capture_frames_rejects_bad_parameters(fake_mss: None, count: int, interval: float) -> None:
    with pytest.raises(CaptureError):
        capture_frames(count, interval)


def test_capture_sequence_reports_the_gaps_between_frames(fake_mss: None) -> None:
    """The reported rate must include the interval, not just the capture cost."""
    frames = capture_frames(4, 0.02, monitor_index=1)

    # Capture-only throughput ignores the 20 ms wait and is therefore much higher.
    assert frames.effective_fps < frames.capture_only_fps
    assert frames.effective_fps == pytest.approx(4 / frames.elapsed_seconds)
    assert frames.average_capture_ms == pytest.approx(average_capture_ms(frames.frames))


def test_capture_sequence_behaves_like_a_list(fake_mss: None) -> None:
    frames = capture_frames(3, 0.0, monitor_index=1)
    assert len(frames) == 3
    assert frames[0].metadata["frame_index"] == 0
    assert [frame.metadata["frame_index"] for frame in frames] == [0, 1, 2]


@pytest.mark.parametrize(
    ("count", "elapsed", "expected"),
    [(0, 1.0, 0.0), (5, 0.0, 0.0), (5, 1.0, 5.0), (10, 2.5, 4.0)],
)
def test_effective_sequence_fps_formula(count: int, elapsed: float, expected: float) -> None:
    assert effective_sequence_fps(count, elapsed) == pytest.approx(expected)


def test_capture_frames_accepts_a_region(fake_mss: None) -> None:
    """A sequence can be limited to a region, and keeps the monitor offset."""
    frames = capture_frames(2, 0.0, monitor_index=1, region=CaptureRegion(10, 20, 40, 30))

    assert len(frames) == 2
    assert frames[0].image.size == (40, 30)
    # The region offset is preserved, so a coordinate measured inside the region
    # still maps back to the right place on the screen.
    info = frames[0].screen_info
    assert (info.monitor_left, info.monitor_top) == (10, 20)
