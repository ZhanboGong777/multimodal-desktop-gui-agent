"""Pure helpers that map screenshot coordinates onto control coordinates.

Capture tools and control tools do not always agree on geometry: a Retina
screenshot may be twice the size of the control space, Windows display scaling
may stretch only one axis, and a monitor may be offset. Screenshot coordinates
returned by OCR or contour detection must therefore never be handed to
PyAutoGUI directly.
"""

from __future__ import annotations

from .schemas import Point, ScreenInfo


def clamp(value: int, low: int, high: int) -> int:
    """Clamp ``value`` into ``[low, high]``."""
    if high < low:
        raise ValueError(f"invalid range: low={low} is greater than high={high}")
    return max(low, min(value, high))


def screenshot_to_control(point: Point, screen: ScreenInfo) -> Point:
    """Scale a screenshot pixel into monitor-relative control coordinates.

    The monitor offset is added last so the result is an absolute position that
    PyAutoGUI can use on multi-monitor desktops.
    """
    scale_x = screen.scale_x if screen.scale_x is not None else 1.0
    scale_y = screen.scale_y if screen.scale_y is not None else 1.0
    return Point(
        x=round(point.x * scale_x) + screen.monitor_left,
        y=round(point.y * scale_y) + screen.monitor_top,
    )


def clamp_to_screen(point: Point, screen: ScreenInfo) -> Point:
    """Clamp an absolute control point into the monitor's valid area."""
    max_x = screen.monitor_left + screen.control_width - 1
    max_y = screen.monitor_top + screen.control_height - 1
    return Point(
        x=clamp(point.x, screen.monitor_left, max_x),
        y=clamp(point.y, screen.monitor_top, max_y),
    )


def screenshot_to_control_clamped(point: Point, screen: ScreenInfo) -> Point:
    """Map and clamp in one step; this is what the demos should call."""
    return clamp_to_screen(screenshot_to_control(point, screen), screen)


def is_inside_screen(point: Point, screen: ScreenInfo) -> bool:
    """True when an absolute control point lies on the captured monitor."""
    return (
        screen.monitor_left <= point.x < screen.monitor_left + screen.control_width
        and screen.monitor_top <= point.y < screen.monitor_top + screen.control_height
    )
