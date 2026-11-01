"""Tests for the shared data structures."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gui_agent.schemas import ActionResult, BoundingBox, DesktopAction, Point, ScreenInfo, UIElement


def test_bounding_box_reports_size_and_center() -> None:
    box = BoundingBox(left=10, top=20, right=110, bottom=70)
    assert box.width == 100
    assert box.height == 50
    assert box.center == Point(x=60, y=45)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left": 100, "top": 0, "right": 100, "bottom": 10},
        {"left": 0, "top": 10, "right": 10, "bottom": 10},
        {"left": 50, "top": 0, "right": 10, "bottom": 10},
    ],
)
def test_bounding_box_rejects_invalid_geometry(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        BoundingBox(**kwargs)


def test_screen_info_derives_scale_factors() -> None:
    info = ScreenInfo(
        screenshot_width=2560, screenshot_height=1600, control_width=1280, control_height=800
    )
    assert info.scale_x == pytest.approx(0.5)
    assert info.scale_y == pytest.approx(0.5)


def test_screen_info_keeps_explicit_scale_factors() -> None:
    info = ScreenInfo(
        screenshot_width=1000,
        screenshot_height=1000,
        control_width=1000,
        control_height=1000,
        scale_x=0.25,
        scale_y=0.75,
    )
    assert info.scale_x == pytest.approx(0.25)
    assert info.scale_y == pytest.approx(0.75)


@pytest.mark.parametrize(
    "field",
    ["screenshot_width", "screenshot_height", "control_width", "control_height"],
)
def test_screen_info_rejects_non_positive_dimensions(field: str) -> None:
    values: dict[str, int] = {
        "screenshot_width": 100,
        "screenshot_height": 100,
        "control_width": 100,
        "control_height": 100,
    }
    values[field] = 0
    with pytest.raises(ValidationError):
        ScreenInfo(**values)


def test_ui_element_fills_center_and_default_source() -> None:
    element = UIElement(text="ok", bounding_box=BoundingBox(left=0, top=0, right=10, bottom=10))
    assert element.center == Point(x=5, y=5)
    assert element.source == "ocr"
    assert element.confidence == 0.0


def test_ui_element_rejects_out_of_range_confidence() -> None:
    box = BoundingBox(left=0, top=0, right=10, bottom=10)
    with pytest.raises(ValidationError):
        UIElement(text="bad", bounding_box=box, confidence=1.5)
    with pytest.raises(ValidationError):
        UIElement(text="bad", bounding_box=box, source="unknown")  # type: ignore[arg-type]


def test_point_actions_require_coordinates() -> None:
    with pytest.raises(ValidationError):
        DesktopAction(action_type="click")
    action = DesktopAction(action_type="click", x=1, y=2)
    assert action.points == [Point(x=1, y=2)]


def test_drag_requires_two_distinct_points() -> None:
    with pytest.raises(ValidationError):
        DesktopAction(action_type="drag", start=Point(x=1, y=1))
    with pytest.raises(ValidationError):
        DesktopAction(action_type="drag", x=1, y=1)
    action = DesktopAction(action_type="drag", start=Point(x=1, y=1), end=Point(x=9, y=9))
    assert action.points == [Point(x=1, y=1), Point(x=9, y=9)]


@pytest.mark.parametrize(
    "payload",
    [
        {"action_type": "type_text"},
        {"action_type": "type_text", "text": "   "},
        {"action_type": "key_press"},
        {"action_type": "hotkey", "keys": []},
    ],
)
def test_text_and_key_actions_require_payloads(payload: dict) -> None:
    with pytest.raises(ValidationError):
        DesktopAction(**payload)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Point(x=1, y=2, z=3)  # type: ignore[call-arg]


def test_action_result_defaults_to_empty_metadata() -> None:
    result = ActionResult(success=True, dry_run=True)
    assert result.metadata == {}
    assert result.error is None
