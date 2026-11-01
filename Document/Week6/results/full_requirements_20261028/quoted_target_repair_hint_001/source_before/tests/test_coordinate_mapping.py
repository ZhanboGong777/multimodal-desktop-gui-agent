"""Tests for screenshot -> control coordinate mapping."""

from __future__ import annotations

import pytest

from gui_agent.coordinates import (
    clamp,
    clamp_to_screen,
    is_inside_screen,
    screenshot_to_control,
    screenshot_to_control_clamped,
)
from gui_agent.schemas import Point, ScreenInfo


def make_info(**overrides: int) -> ScreenInfo:
    values: dict[str, int] = {
        "screenshot_width": 1000,
        "screenshot_height": 800,
        "control_width": 1000,
        "control_height": 800,
    }
    values.update(overrides)
    return ScreenInfo(**values)


def test_identity_when_screenshot_and_control_sizes_match() -> None:
    assert screenshot_to_control(Point(x=100, y=200), make_info()) == Point(x=100, y=200)


def test_retina_screenshot_is_half_the_control_space() -> None:
    screen = make_info(screenshot_width=2000, screenshot_height=1600)
    assert screenshot_to_control(Point(x=1000, y=800), screen) == Point(x=500, y=400)


def test_axes_can_scale_independently() -> None:
    screen = make_info(screenshot_width=2000, screenshot_height=1000)
    assert screenshot_to_control(Point(x=200, y=200), screen) == Point(x=100, y=160)


def test_monitor_offset_is_added() -> None:
    screen = make_info(monitor_left=215, monitor_top=1080)
    assert screenshot_to_control(Point(x=10, y=10), screen) == Point(x=225, y=1090)


def test_screen_edges_are_inside_but_one_pixel_past_is_not() -> None:
    screen = make_info()
    assert is_inside_screen(Point(x=0, y=0), screen)
    assert is_inside_screen(Point(x=999, y=799), screen)
    assert not is_inside_screen(Point(x=1000, y=799), screen)
    assert not is_inside_screen(Point(x=0, y=800), screen)


def test_clamp_pulls_coordinates_back_onto_the_monitor() -> None:
    screen = make_info(monitor_left=215, monitor_top=1080)
    assert clamp_to_screen(Point(x=-50, y=-50), screen) == Point(x=215, y=1080)
    assert clamp_to_screen(Point(x=99999, y=99999), screen) == Point(x=1214, y=1879)


def test_clamp_helper_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError):
        clamp(5, 10, 0)


def test_mapping_and_clamping_are_combined() -> None:
    screen = make_info(screenshot_width=2000, screenshot_height=1600)
    assert screenshot_to_control_clamped(Point(x=5000, y=5000), screen) == Point(x=999, y=799)


def test_offset_monitor_clamps_within_its_own_bounds() -> None:
    screen = make_info(monitor_left=215, monitor_top=1080, control_width=1470, control_height=956)
    assert clamp_to_screen(Point(x=0, y=0), screen) == Point(x=215, y=1080)
    assert clamp_to_screen(Point(x=5000, y=5000), screen) == Point(x=1684, y=2035)
