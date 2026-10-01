"""The adapter must refuse rather than guess.

Every failure here is a case where a plausible-looking implementation would pick
something and click it: the first of several matches, a coordinate from a stale
frame, or a key the plan invented.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gui_agent.planning.schemas import PlanStep
from gui_agent.runtime.action_adapter import ActionAdapter, ActionResolutionError
from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot
from gui_agent.schemas import BoundingBox, ScreenInfo


def _snapshot(
    observation_id: str = "obs-0001", texts: tuple[str, ...] = ("File", "Edit", "Search")
) -> ObservationSnapshot:
    elements = [
        ElementRef(
            element_id=f"{observation_id}-e{index:03d}",
            text=text,
            bounding_box=BoundingBox(
                left=10 + index * 100, top=20, right=80 + index * 100, bottom=40
            ),
            center=__import__("gui_agent.schemas", fromlist=["Point"]).Point(
                x=45 + index * 100, y=30
            ),
            confidence=0.9,
        )
        for index, text in enumerate(texts)
    ]
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=1470, screenshot_height=956, control_width=1470, control_height=956
        ),
        elements=elements,
    )


def _step(**kwargs) -> PlanStep:
    payload = {"step_id": "s1", "description": "do it", "action_type": "click"}
    payload.update(kwargs)
    return PlanStep(**payload)


@pytest.fixture
def adapter() -> ActionAdapter:
    return ActionAdapter(platform="darwin")


def test_a_unique_text_target_resolves(adapter: ActionAdapter) -> None:
    resolved = adapter.resolve(_step(target_text="Edit"), _snapshot())
    assert resolved.action.action_type == "click"
    assert (resolved.action.x, resolved.action.y) == (145, 30)
    assert resolved.element_id == "obs-0001-e001"


def test_an_element_id_from_this_frame_resolves(adapter: ActionAdapter) -> None:
    resolved = adapter.resolve(_step(arguments={"element_id": "obs-0001-e002"}), _snapshot())
    assert resolved.element_id == "obs-0001-e002"


def test_an_element_id_from_another_frame_is_refused(adapter: ActionAdapter) -> None:
    """The same id in a new frame means a different element, so it must not resolve."""
    stale = _step(arguments={"element_id": "obs-0001-e000"})
    with pytest.raises(ActionResolutionError, match="not from observation"):
        adapter.resolve(stale, _snapshot("obs-0002"))


def test_an_ambiguous_query_is_refused_not_first_matched(adapter: ActionAdapter) -> None:
    snapshot = _snapshot(texts=("Open", "Open", "Close"))
    with pytest.raises(ActionResolutionError, match="elements match"):
        adapter.resolve(_step(target_text="Open"), snapshot)


def test_a_missing_target_is_refused(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="no element matches"):
        adapter.resolve(_step(target_text="Nowhere"), _snapshot())


def test_a_click_without_any_target_is_refused(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="needs target_text"):
        adapter.resolve(_step(), _snapshot())


def test_scroll_without_a_coordinate_is_refused(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError):
        adapter.resolve(_step(action_type="scroll", target_text="Nowhere"), _snapshot())


def test_scroll_needs_an_amount(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="scroll_amount"):
        adapter.resolve(_step(action_type="scroll", target_text="Edit"), _snapshot())


def test_type_text_rejects_blank_input(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="non-empty"):
        adapter.resolve(_step(action_type="type_text", arguments={"text": "   "}), _snapshot())


def test_type_text_accepts_real_input(adapter: ActionAdapter) -> None:
    resolved = adapter.resolve(
        _step(action_type="type_text", arguments={"text": "GUI agent research"}), _snapshot()
    )
    assert resolved.action.text == "GUI agent research"


def test_an_unsupported_key_is_refused(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="allowed key set"):
        adapter.resolve(_step(action_type="key_press", arguments={"key": "rm"}), _snapshot())


def test_a_supported_key_is_case_folded(adapter: ActionAdapter) -> None:
    resolved = adapter.resolve(
        _step(action_type="key_press", arguments={"key": "Enter"}), _snapshot()
    )
    assert resolved.action.key == "enter"


def test_a_hotkey_is_translated_for_the_platform() -> None:
    step = _step(action_type="hotkey", arguments={"keys": ["ctrl", "l"]})
    on_mac = ActionAdapter(platform="darwin").resolve(step, _snapshot())
    on_windows = ActionAdapter(platform="win32").resolve(step, _snapshot())
    assert on_mac.action.keys == ["command", "l"]
    assert on_windows.action.keys == ["ctrl", "l"]


def test_an_out_of_range_wait_is_refused(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="0-10"):
        adapter.resolve(_step(action_type="wait", arguments={"duration": 600}), _snapshot())


def test_finish_is_never_turned_into_an_action(adapter: ActionAdapter) -> None:
    """finish closes the plan; the runner verifies the task instead."""
    with pytest.raises(ActionResolutionError, match="never executed"):
        adapter.resolve(_step(action_type="finish"), _snapshot())


def test_coordinates_are_scaled_into_control_space(adapter: ActionAdapter) -> None:
    """A scaled screenshot must not produce a click at the wrong place."""
    snapshot = _snapshot()
    scaled = snapshot.model_copy(
        update={
            "screen_info": ScreenInfo(
                screenshot_width=2940,
                screenshot_height=1912,
                control_width=1470,
                control_height=956,
            )
        }
    )
    resolved = adapter.resolve(_step(arguments={"element_id": "obs-0001-e000"}), scaled)
    # element centre is (45,30) in a 2940-wide screenshot -> (22,15) in control space
    assert (resolved.control_point.x, resolved.control_point.y) == (22, 15)
    assert (resolved.screenshot_point.x, resolved.screenshot_point.y) == (45, 30)


def test_a_stale_id_with_a_text_target_is_rebound_in_the_new_frame(adapter: ActionAdapter) -> None:
    """A plan is written from one frame and executed after another.

    Refusing every stale id would make element targeting unusable, since the
    runner always re-observes before acting. The spec allows a re-bind when the
    step also says, in words, what it is aiming at.
    """
    step = _step(arguments={"element_id": "obs-0001-e001"}, target_text="Edit")
    resolved = adapter.resolve(step, _snapshot("obs-0002"))
    assert resolved.element_id == "obs-0002-e001"
    assert "re-bound from obs-0001-e001" in resolved.note


def test_a_stale_id_without_a_text_target_is_still_refused(adapter: ActionAdapter) -> None:
    """Nothing to re-locate against: guessing would be worse than stopping."""
    step = _step(arguments={"element_id": "obs-0001-e001"})
    with pytest.raises(ActionResolutionError, match="names no text target"):
        adapter.resolve(step, _snapshot("obs-0002"))
