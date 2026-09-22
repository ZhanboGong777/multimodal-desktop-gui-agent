"""Screen capture built on MSS, with optional frame sequences.

Screenshot pixels are not control coordinates. Every capture therefore returns a
:class:`ScreenInfo` describing how the captured image relates to the monitor, so
callers can map coordinates through :mod:`gui_agent.coordinates`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import mss
from PIL import Image

from ..schemas import ScreenInfo


class CaptureError(RuntimeError):
    """Raised when the screen cannot be captured."""


@dataclass(frozen=True)
class CaptureRegion:
    """A rectangle expressed in monitor screenshot pixels."""

    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass
class CaptureResult:
    """One captured frame plus the geometry needed to interpret it."""

    image: Image.Image
    screen_info: ScreenInfo
    captured_at: datetime
    image_path: Path | None = None
    capture_time_ms: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


def _control_size() -> tuple[int, int] | None:
    """Best-effort control-space size, without hard-failing in headless runs."""
    try:
        import pyautogui
    except Exception:  # noqa: BLE001 - pyautogui may be absent or headless
        return None
    try:
        size = pyautogui.size()
        return int(size.width), int(size.height)
    except Exception:  # noqa: BLE001 - a headless session has no screen size
        return None


def list_monitors() -> list[dict[str, int]]:
    """Return the MSS monitor list; index 0 is the union of all monitors."""
    try:
        with mss.mss() as session:
            return [dict(monitor) for monitor in session.monitors]
    except Exception as exc:
        raise CaptureError(f"Unable to query monitors: {exc}") from exc


def _stamp() -> str:
    # Local time keeps session directories readable for the operator.
    return datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _default_name(prefix: str) -> str:
    return f"{prefix}_{_stamp()}.png"


def _save(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _build_screen_info(
    *,
    capture_width: int,
    capture_height: int,
    monitor_left: int,
    monitor_top: int,
    monitor_width: int,
    monitor_height: int,
    monitor_index: int,
    control_size: tuple[int, int] | None,
) -> ScreenInfo:
    """Describe one capture using a single global scale factor."""
    if control_size is None:
        control_width, control_height = capture_width, capture_height
    else:
        scale_x = control_size[0] / monitor_width
        scale_y = control_size[1] / monitor_height
        control_width = max(1, round(capture_width * scale_x))
        control_height = max(1, round(capture_height * scale_y))

    return ScreenInfo(
        screenshot_width=capture_width,
        screenshot_height=capture_height,
        control_width=control_width,
        control_height=control_height,
        monitor_index=monitor_index,
        monitor_left=monitor_left,
        monitor_top=monitor_top,
    )


def capture_monitor(
    monitor_index: int = 1,
    *,
    output_directory: Path | None = None,
    save: bool = False,
    control_size: tuple[int, int] | None = None,
) -> CaptureResult:
    """Capture one monitor; index 1 is the primary display by MSS convention."""
    started = time.perf_counter()
    try:
        with mss.mss() as session:
            monitors = session.monitors
            if monitor_index < 1 or monitor_index >= len(monitors):
                raise CaptureError(
                    f"monitor_index {monitor_index} is out of range "
                    f"(available 1..{len(monitors) - 1})"
                )
            monitor = monitors[monitor_index]
            shot = session.grab(monitor)
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"Screen capture failed: {exc}") from exc

    image = Image.frombytes("RGB", shot.size, shot.rgb)
    if control_size is None:
        control_size = _control_size()

    screen_info = _build_screen_info(
        capture_width=image.width,
        capture_height=image.height,
        monitor_left=int(monitor["left"]),
        monitor_top=int(monitor["top"]),
        monitor_width=int(monitor["width"]),
        monitor_height=int(monitor["height"]),
        monitor_index=monitor_index,
        control_size=control_size,
    )

    image_path = None
    if save:
        directory = output_directory or Path("outputs/week2")
        image_path = _save(image, directory / _default_name(f"monitor{monitor_index}"))

    return CaptureResult(
        image=image,
        screen_info=screen_info,
        captured_at=datetime.now(UTC),
        image_path=image_path,
        capture_time_ms=(time.perf_counter() - started) * 1000.0,
        metadata={"monitor": dict(monitor)},
    )


def capture_fullscreen(
    *,
    output_directory: Path | None = None,
    save: bool = False,
    control_size: tuple[int, int] | None = None,
) -> CaptureResult:
    """Capture the primary display.

    The primary monitor is the one the agent drives, so "fullscreen" means that
    monitor rather than the union of every attached display.
    """
    return capture_monitor(
        1, output_directory=output_directory, save=save, control_size=control_size
    )


def capture_region(
    region: CaptureRegion,
    *,
    monitor_index: int = 1,
    output_directory: Path | None = None,
    save: bool = False,
    control_size: tuple[int, int] | None = None,
) -> CaptureResult:
    """Capture part of a monitor and keep the absolute monitor offset."""
    if region.width <= 0 or region.height <= 0:
        raise CaptureError(f"region must have a positive size, got {region.width}x{region.height}")

    started = time.perf_counter()
    try:
        with mss.mss() as session:
            monitors = session.monitors
            if monitor_index < 1 or monitor_index >= len(monitors):
                raise CaptureError(
                    f"monitor_index {monitor_index} is out of range "
                    f"(available 1..{len(monitors) - 1})"
                )
            monitor = monitors[monitor_index]
            shot = session.grab(region.as_mss())
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"Region capture failed: {exc}") from exc

    image = Image.frombytes("RGB", shot.size, shot.rgb)
    if control_size is None:
        control_size = _control_size()

    scale_x = 1.0 if control_size is None else control_size[0] / int(monitor["width"])
    scale_y = 1.0 if control_size is None else control_size[1] / int(monitor["height"])

    screen_info = _build_screen_info(
        capture_width=image.width,
        capture_height=image.height,
        monitor_left=int(monitor["left"]) + round(region.left * scale_x),
        monitor_top=int(monitor["top"]) + round(region.top * scale_y),
        monitor_width=image.width,
        monitor_height=image.height,
        monitor_index=monitor_index,
        control_size=None
        if control_size is None
        else (
            round(image.width * scale_x),
            round(image.height * scale_y),
        ),
    )

    image_path = None
    if save:
        directory = output_directory or Path("outputs/week2")
        image_path = _save(image, directory / _default_name("region"))

    return CaptureResult(
        image=image,
        screen_info=screen_info,
        captured_at=datetime.now(UTC),
        image_path=image_path,
        capture_time_ms=(time.perf_counter() - started) * 1000.0,
        metadata={"region": region.as_mss()},
    )


@dataclass
class CaptureSequence:
    """A burst of frames plus how long the whole burst actually took.

    Iterating, indexing and ``len()`` behave exactly like the plain list this
    replaced, so callers that only want the frames are unaffected.
    """

    frames: list[CaptureResult]
    elapsed_seconds: float

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[CaptureResult]:
        return iter(self.frames)

    def __getitem__(self, index: int) -> CaptureResult:
        return self.frames[index]

    @property
    def average_capture_ms(self) -> float:
        """Mean time spent inside the capture call itself."""
        return average_capture_ms(self.frames)

    @property
    def capture_only_fps(self) -> float:
        """Throughput of the capture call alone, ignoring the gaps between frames."""
        return capture_only_fps(self.frames)

    @property
    def effective_fps(self) -> float:
        """Frame rate the sequence actually achieved, gaps included."""
        return effective_sequence_fps(len(self.frames), self.elapsed_seconds)


def capture_frames(
    frame_count: int = 10,
    interval_seconds: float = 0.2,
    *,
    monitor_index: int = 1,
    region: CaptureRegion | None = None,
    output_directory: Path | None = None,
    save_frames: bool = False,
) -> CaptureSequence:
    """Capture a short sequence to prove continuous, near real-time capture.

    Per-frame latency is recorded on each :class:`CaptureResult`, and the burst
    as a whole is timed so callers can report both the capture-only throughput
    and the rate the sequence really ran at. Those two numbers differ a lot:
    with a 0.2 s interval the burst advances about four times per second even
    though each individual capture only costs tens of milliseconds.
    """
    if frame_count < 1:
        raise CaptureError(f"frame_count must be at least 1, got {frame_count}")
    if interval_seconds < 0:
        raise CaptureError(f"interval_seconds must not be negative, got {interval_seconds}")

    control_size = _control_size()
    frames: list[CaptureResult] = []
    started = time.perf_counter()
    for index in range(frame_count):
        frame = (
            capture_region(
                region,
                monitor_index=monitor_index,
                output_directory=output_directory,
                save=save_frames,
                control_size=control_size,
            )
            if region is not None
            else capture_monitor(
                monitor_index,
                output_directory=output_directory,
                save=save_frames,
                control_size=control_size,
            )
        )
        frame.metadata["frame_index"] = index
        frames.append(frame)
        if index < frame_count - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)
    return CaptureSequence(frames=frames, elapsed_seconds=time.perf_counter() - started)


def average_capture_ms(frames: Sequence[CaptureResult]) -> float:
    if not frames:
        return 0.0
    return sum(frame.capture_time_ms for frame in frames) / len(frames)


def capture_only_fps(frames: Sequence[CaptureResult]) -> float:
    """Throughput of the capture call alone, ignoring the wait between frames.

    This is a ceiling rather than the rate a sequence ran at. Use
    :func:`effective_sequence_fps` for the recorded rate.
    """
    average = average_capture_ms(frames)
    if average <= 0:
        return 0.0
    return 1000.0 / average


def effective_sequence_fps(frame_count: int, elapsed_seconds: float) -> float:
    """Frame rate actually achieved, including the wait between frames."""
    if frame_count <= 0 or elapsed_seconds <= 0:
        return 0.0
    return frame_count / elapsed_seconds
