"""Read-only T04 frame integrity checks, never semantic or Send authorization.

No expected recipient, marker, model, cursor API or desktop controller enters
this module. A pass means only that the source frame and geometry are usable;
arbitrary overlays and obscured characters cannot be established from these facts.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .schemas import ObservationSnapshot


@dataclass(frozen=True)
class ScreenshotQuality:
    outcome: Literal["passed", "retryable", "blocked"]
    code: str
    detail: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def window_identity(snapshot: ObservationSnapshot) -> tuple[Any, ...]:
    """Retry captures must retain the observed window, title, class and bounds."""
    box = snapshot.window_bounds
    return (snapshot.window_id, snapshot.window_title, snapshot.window_class,
            (box.left, box.top, box.right, box.bottom) if box else None)


def assess_screenshot_quality(snapshot: ObservationSnapshot) -> ScreenshotQuality:
    """Classify concrete capture failures; do not guess whether text is readable."""
    screen = snapshot.screen_info
    bounds = snapshot.window_bounds
    evidence: dict[str, Any] = {
        "observation_id": snapshot.observation_id, "image_path": snapshot.image_path,
        "captured_at": snapshot.captured_at.isoformat(),
        "window_id": snapshot.window_id, "window_title": snapshot.window_title,
        "window_class": snapshot.window_class,
        "window_bounds": bounds.model_dump() if bounds else None,
        "foreground_stable": snapshot.foreground_stable,
        "screen_info": screen.model_dump(), "observation_errors": snapshot.errors.copy(),
        "scope": "source image, foreground metadata and window geometry only",
        "semantic_authorization": False, "send_authorization": False,
        "cursor_occlusion": "unknown: this snapshot has no cursor-position/overlay evidence",
        "arbitrary_overlay_detection": "unsupported",
    }

    def result(outcome, code, detail):
        return ScreenshotQuality(outcome, code, detail, evidence)

    if not (snapshot.window_id and snapshot.window_title and snapshot.window_class and bounds):
        return result("blocked", "foreground_evidence_missing",
                      "known foreground handle, title, class and bounds are required")
    # T04_20261007_142444_963962_9bd5f870 had exact title ChatGPT and this
    # Chromium host class over WeChat. Substrings in chat titles are not proof.
    host_title = snapshot.window_title.strip().casefold()
    if host_title in {"chatgpt", "codex"} and snapshot.window_class == "Chrome_WidgetWin_1":
        evidence["foreground_match_rule"] = "exact host title plus recorded Chromium window class"
        return result("blocked", "foreground_application_obstruction",
                      "captured foreground is the ChatGPT/Codex host; put the target application in front")
    if snapshot.errors:
        return result("blocked", "observation_degraded", "capture/perception reported errors")
    if not (
        0 <= bounds.left < bounds.right <= screen.screenshot_width
        and 0 <= bounds.top < bounds.bottom <= screen.screenshot_height
    ):
        return result("blocked", "foreground_window_clipped",
                      "the complete foreground window is not inside the recorded monitor")
    if any(not isinstance(scale, (int, float)) or not math.isfinite(scale)
           or not math.isclose(scale, expected) for scale, expected in (
               (screen.scale_x, screen.control_width / screen.screenshot_width),
               (screen.scale_y, screen.control_height / screen.screenshot_height),
           )):
        return result("blocked", "screen_scale_inconsistent", "capture/control geometry is inconsistent")
    if not snapshot.image_path:
        return result("blocked", "source_image_missing", "no saved source image is available")
    try:
        path = Path(snapshot.image_path)
        raw = path.read_bytes()
        evidence["source_image_bytes"] = len(raw)
        evidence["source_image_sha256"] = hashlib.sha256(raw).hexdigest()
        # Decode exactly the hashed bytes, even if a path is replaced meanwhile.
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            evidence["source_image_size"] = list(image.size)
            if image.size != (screen.screenshot_width, screen.screenshot_height):
                return result("blocked", "source_image_dimensions_mismatch",
                              "source image dimensions disagree with the recorded screenshot geometry")
            window = image.convert("RGB").crop((bounds.left, bounds.top, bounds.right, bounds.bottom))
            ranges = [high - low for low, high in window.getextrema()]
            evidence["window_rgb_channel_ranges"] = ranges
            if not snapshot.foreground_stable:
                return result("retryable", "foreground_capture_unstable",
                              "foreground changed during this capture; another unchanged-window capture is required")
            # Only a practically uniform whole window is classified here. A
            # blank editor is valid, and blur/overlay heuristics could erase text.
            if max(ranges) <= 1:
                return result("retryable", "foreground_window_uniform",
                              "the entire foreground crop is effectively uniform; no text evidence is available")
    except (OSError, ValueError) as exc:
        evidence["source_error"] = f"{type(exc).__name__}: {exc}"
        return result("blocked", "source_image_unreadable", "cannot read the saved source frame")
    return result("passed", "basic_frame_integrity_passed",
                  "source image and foreground geometry passed basic checks; semantic verification is still required")
