"""Image/geometry-only cursor parking; never message or Send authorization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from ..coordinates import is_inside_screen, screenshot_to_control
from ..schemas import DesktopAction, Point
from .schemas import ObservationSnapshot, ResolvedAction
from .screenshot_quality import assess_screenshot_quality, window_identity

_MONITOR_INSET_CONTROL = 32
_WINDOW_CLEARANCE_CONTROL = 64


class CursorParkingError(RuntimeError):
    """A fresh capture cannot establish unchanged parking geometry."""


@dataclass(frozen=True)
class CursorParkingDecision:
    outcome: Literal["planned", "skipped", "blocked"]
    code: str
    resolved: ResolvedAction | None
    evidence: dict[str, Any]


def verify_parking_frame(
    reference: ObservationSnapshot, current: ObservationSnapshot,
) -> dict[str, Any]:
    """Require a fresh real capture with identical foreground and monitor geometry."""
    quality = assess_screenshot_quality(current)
    if quality.outcome != "passed":
        raise CursorParkingError(f"parking capture refused: {quality.code}")
    if (
        not reference.foreground_stable
        or reference.errors
        or window_identity(reference) != window_identity(current)
        or reference.screen_info.model_dump() != current.screen_info.model_dump()
    ):
        raise CursorParkingError("foreground or screen geometry changed around cursor parking")
    if (
        reference.observation_id == current.observation_id
        or current.captured_at < reference.captured_at
    ):
        raise CursorParkingError("cursor parking requires a fresh subsequent observation")
    return quality.evidence


def plan_cursor_parking(
    snapshot: ObservationSnapshot, *, step_id: str,
) -> CursorParkingDecision:
    """Choose only a measured window-exterior point, with no task/label inputs.

    Insets are a conservative fixed geometry policy, not a claim that an
    arbitrary overlay or the operating system cursor footprint was measured.
    """
    quality = assess_screenshot_quality(snapshot)
    evidence = {
        "source": quality.evidence,
        "monitor_inset_control_pixels": _MONITOR_INSET_CONTROL,
        "window_clearance_control_pixels": _WINDOW_CLEARANCE_CONTROL,
        "selection": "maximum geometric clearance; area then fixed side order breaks ties",
        "semantic_authorization": False,
        "send_authorization": False,
        "cursor_occlusion_removed": "unproven; strict fresh draft verification is still required",
    }
    if quality.outcome != "passed":
        return CursorParkingDecision("blocked", quality.code, None, evidence)
    screen, bounds = snapshot.screen_info, snapshot.window_bounds
    assert bounds is not None and screen.scale_x is not None and screen.scale_y is not None
    sx, sy = screen.scale_x, screen.scale_y
    ix, iy = math.ceil(_MONITOR_INSET_CONTROL / sx), math.ceil(_MONITOR_INSET_CONTROL / sy)
    px, py = math.ceil(_WINDOW_CLEARANCE_CONTROL / sx), math.ceil(_WINDOW_CLEARANCE_CONTROL / sy)
    left, top = ix, iy
    right, bottom = screen.screenshot_width - ix, screen.screenshot_height - iy
    areas = [
        ("left", (left, top, min(right, bounds.left - px), bottom)),
        ("right", (max(left, bounds.right + px), top, right, bottom)),
        ("above", (left, top, right, min(bottom, bounds.top - py))),
        ("below", (left, max(top, bounds.bottom + py), right, bottom)),
    ]
    candidates = []
    window_left = round(bounds.left * sx) + screen.monitor_left
    window_top = round(bounds.top * sy) + screen.monitor_top
    window_right = round(bounds.right * sx) + screen.monitor_left
    window_bottom = round(bounds.bottom * sy) + screen.monitor_top
    for order, (side, area) in enumerate(areas):
        a, b, c, d = area
        if not (a < c and b < d):
            continue
        native = Point(x=(a + c) // 2, y=(b + d) // 2)
        control = screenshot_to_control(native, screen)
        dx = max(window_left - control.x, 0, control.x - (window_right - 1))
        dy = max(window_top - control.y, 0, control.y - (window_bottom - 1))
        clearance = min(
            math.hypot(dx, dy), control.x - screen.monitor_left,
            control.y - screen.monitor_top,
            screen.monitor_left + screen.control_width - 1 - control.x,
            screen.monitor_top + screen.control_height - 1 - control.y,
        )
        if not is_inside_screen(control, screen) or clearance < _MONITOR_INSET_CONTROL:
            continue
        if window_left <= control.x < window_right and window_top <= control.y < window_bottom:
            continue
        candidates.append((clearance, (c - a) * (d - b), -order, side, area, native, control))
    if not candidates:
        evidence["free_rectangles"] = []
        return CursorParkingDecision("skipped", "no_safe_window_exterior", None, evidence)
    chosen = max(candidates, key=lambda item: item[:3])
    clearance, _, _, side, area, native, control = chosen
    evidence.update(
        side=side, free_rectangle=list(area), clearance_control_pixels=clearance,
        screenshot_point=native.model_dump(), control_point=control.model_dump(),
    )
    resolved = ResolvedAction(
        step_id=step_id, action=DesktopAction(action_type="move", x=control.x, y=control.y),
        screenshot_point=native, control_point=control,
        note="runtime geometry-only cursor parking outside unchanged foreground window",
    )
    return CursorParkingDecision("planned", "measured_window_exterior", resolved, evidence)
