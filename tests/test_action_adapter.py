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
from gui_agent.schemas import BoundingBox, Point, ScreenInfo


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
    with pytest.raises(ActionResolutionError, match="0-5"):
        adapter.resolve(_step(action_type="wait", arguments={"duration": 600}), _snapshot())


def test_the_wait_bound_comes_from_the_run_configuration() -> None:
    """10.3 suggests max_wait_seconds = 5, and the run's own limit is what applies.

    Hard-coding it meant the configured value could not have any effect - the same
    trap as a config key nothing reads.
    """
    strict = ActionAdapter(platform="darwin", max_wait_seconds=2)

    with pytest.raises(ActionResolutionError, match="0-2"):
        strict.resolve(_step(action_type="wait", arguments={"duration": 3}), _snapshot())

    allowed = strict.resolve(_step(action_type="wait", arguments={"duration": 2}), _snapshot())
    assert allowed.action.action_type == "wait"


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


def _word_snapshot(
    observation_id: str = "obs-0002",
    words: tuple[tuple[str, int, int], ...] = (),
) -> ObservationSnapshot:
    """A frame whose elements are word-sized, the way a word-level backend emits them."""
    elements = [
        ElementRef(
            element_id=f"{observation_id}-e{index:03d}",
            text=text,
            bounding_box=BoundingBox(left=left, top=top, right=left + 60, bottom=top + 20),
            center=Point(x=left + 30, y=top + 10),
            confidence=0.9,
        )
        for index, (text, left, top) in enumerate(words)
    ]
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=1470, screenshot_height=956, control_width=1470, control_height=956
        ),
        elements=elements,
    )


def test_a_target_split_across_word_elements_is_joined(adapter: ActionAdapter) -> None:
    """Word-level OCR never produces the multi-word label a model asks for.

    "Summary (required)" arrives as two elements, so an exact-text search finds
    nothing even though the label is plainly on screen. This is what stopped every
    real target from resolving on the Windows review machine.
    """
    snapshot = _word_snapshot(words=(("Summary", 10, 40), ("(required)", 80, 40)))
    resolved = adapter.resolve(_step(target_text="Summary (required)"), snapshot)
    assert resolved.element_id == "obs-0002-e000+obs-0002-e001"
    assert "joined from 2 adjacent elements" in resolved.note


def test_the_joined_target_is_centred_on_the_whole_label(adapter: ActionAdapter) -> None:
    snapshot = _word_snapshot(words=(("Current", 10, 10), ("repository", 80, 10)))
    resolved = adapter.resolve(_step(target_text="Current repository"), snapshot)
    # Union box is (10,10)-(140,30), so the click lands at its centre.
    assert (resolved.screenshot_point.x, resolved.screenshot_point.y) == (75, 20)


def test_a_stale_id_split_across_words_is_rebound_to_the_joined_target(
    adapter: ActionAdapter,
) -> None:
    """Both repairs have to work at once: planning ids go stale by design."""
    snapshot = _word_snapshot(words=(("Fetch", 10, 10), ("origin", 80, 10)))
    step = _step(arguments={"element_id": "obs-0001-e003"}, target_text="Fetch origin")
    resolved = adapter.resolve(step, snapshot)
    assert resolved.element_id == "obs-0002-e000+obs-0002-e001"
    assert "re-bound from obs-0001-e003" in resolved.note


def test_two_matching_runs_are_refused(adapter: ActionAdapter) -> None:
    """The same label twice on screen is an ambiguity, not a reason to take the first."""
    snapshot = _word_snapshot(
        words=(
            ("Current", 10, 10),
            ("repository", 80, 10),
            ("Current", 10, 100),
            ("repository", 80, 100),
        )
    )
    with pytest.raises(ActionResolutionError, match="element runs match"):
        adapter.resolve(_step(target_text="Current repository"), snapshot)


def test_a_run_must_consume_the_whole_query(adapter: ActionAdapter) -> None:
    """A run that only covers part of the query is not a match."""
    snapshot = _word_snapshot(words=(("Current", 10, 10), ("repository", 80, 10)))
    with pytest.raises(ActionResolutionError, match="no element matches"):
        adapter.resolve(_step(target_text="Current repository main"), snapshot)


def _element(element_id: str, text: str, left: int, top: int, right: int, bottom: int,
             confidence: float = 0.96) -> ElementRef:
    """An element whose centre follows from its box, as the observation builds it."""
    box = BoundingBox(left=left, top=top, right=right, bottom=bottom)
    return ElementRef(
        element_id=element_id, text=text, bounding_box=box, center=box.center,
        confidence=confidence,
    )


def test_a_stacked_shortcut_label_is_joined_across_an_interleaved_neighbour(
    adapter: ActionAdapter,
) -> None:
    """The two lines of a desktop shortcut are adjacent on screen, not in sort order.

    Measured on the Windows review machine: the Microsoft Edge shortcut is two OCR
    elements, 'Microsoft' at (2433,379)-(2513,393) and 'Edge' at (2452,403)-(2494,421).
    A top-to-bottom sort puts 'MySQL' at (403,412) between them, because its y sits
    between the two lines. The run search therefore never saw the halves as
    neighbours, and T01 failed with "no element matches 'Microsoft Edge'" while the
    icon was plainly on screen and both words were in the element list.
    """
    snapshot = ObservationSnapshot(
        observation_id="obs-0002",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=2560, screenshot_height=1600,
            control_width=2560, control_height=1600, scale_x=1.0, scale_y=1.0,
        ),
        elements=[
            _element("obs-0002-e009", "Microsoft", 2433, 379, 2513, 393),
            # A different column, whose vertical position falls between the two lines.
            _element("obs-0002-e021", "MySQL", 380, 405, 426, 419, confidence=0.94),
            _element("obs-0002-e010", "Edge", 2452, 403, 2494, 421),
        ],
    )
    resolved = adapter.resolve(_step(target_text="Microsoft Edge"), snapshot)
    assert resolved.element_id == "obs-0002-e009+obs-0002-e010"
    assert "joined from 2 adjacent elements" in resolved.note


def test_a_stacked_run_still_refuses_two_of_the_same_shortcut(
    adapter: ActionAdapter,
) -> None:
    """Two shortcuts with the same label remain an ambiguity, not a first-match win."""
    def _pair(top: int) -> list[ElementRef]:
        base = top
        return [
            _element(f"obs-0002-e{base:03d}", "Microsoft", 2433, top, 2513, top + 14),
            _element(f"obs-0002-e{base + 1:03d}", "Edge", 2452, top + 24, 2494, top + 42),
        ]

    snapshot = ObservationSnapshot(
        observation_id="obs-0002",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=2560, screenshot_height=1600,
            control_width=2560, control_height=1600, scale_x=1.0, scale_y=1.0,
        ),
        elements=_pair(100) + _pair(400),
    )
    with pytest.raises(ActionResolutionError, match="element runs match"):
        adapter.resolve(_step(target_text="Microsoft Edge"), snapshot)


def test_a_target_that_maps_outside_the_monitor_is_refused(adapter: ActionAdapter) -> None:
    """7.3.6: an illegal coordinate is refused, never clamped to the edge.

    The realistic shape of this is a screenshot whose scale was never applied: the
    element sits well inside the image, and the control point lands past the edge of
    the monitor. Clamping would still click something - just not the thing asked for.
    """
    snapshot = ObservationSnapshot(
        observation_id="obs-0001",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=2940,
            screenshot_height=1912,
            control_width=1470,
            control_height=956,
            scale_x=1.0,
            scale_y=1.0,
        ),
        elements=[
            ElementRef(
                element_id="obs-0001-e000",
                text="Far away",
                bounding_box=BoundingBox(left=2000, top=100, right=2060, bottom=120),
                center=Point(x=2030, y=110),
                confidence=0.9,
            )
        ],
    )

    with pytest.raises(ActionResolutionError, match="outside the captured monitor"):
        adapter.resolve(_step(target_text="Far away"), snapshot)


def test_a_scroll_amount_of_the_wrong_type_is_refused(adapter: ActionAdapter) -> None:
    """8.3.4 wants parameters complete *and* correctly typed.

    int() on a non-numeric string raises ValueError, and the runner catches
    ActionResolutionError - so an unguarded conversion would leave the run as a
    traceback instead of a recorded failure.
    """
    step = _step(action_type="scroll", target_text="Edit", arguments={"scroll_amount": "lots"})

    with pytest.raises(ActionResolutionError, match="must be a number"):
        adapter.resolve(step, _snapshot())


def test_a_drag_duration_of_the_wrong_type_is_refused(adapter: ActionAdapter) -> None:
    step = _step(
        action_type="drag",
        arguments={
            "start_element_id": "obs-0001-e000",
            "end_element_id": "obs-0001-e001",
            "duration": "slowly",
        },
    )

    with pytest.raises(ActionResolutionError, match="must be a number"):
        adapter.resolve(step, _snapshot())


def test_a_drag_resolves_to_both_endpoints(adapter: ActionAdapter) -> None:
    """A whole action type had no success path under test.

    `_resolve_drag` was only ever exercised by its error branches, so a drag that
    actually resolves had never been checked end to end - including that both
    points go through the coordinate mapping.
    """
    step = _step(
        action_type="drag",
        arguments={"start_element_id": "obs-0001-e000", "end_element_id": "obs-0001-e002"},
    )

    resolved = adapter.resolve(step, _snapshot())

    assert resolved.action.action_type == "drag"
    assert (resolved.action.start.x, resolved.action.start.y) == (45, 30)
    assert (resolved.action.end.x, resolved.action.end.y) == (245, 30)
    assert resolved.action.duration == 0.5, "the default, when the plan names none"


def test_a_drag_naming_an_element_from_nowhere_is_refused(adapter: ActionAdapter) -> None:
    step = _step(
        action_type="drag",
        arguments={"start_element_id": "obs-0001-e000", "end_element_id": "obs-0009-e000"},
    )

    with pytest.raises(ActionResolutionError, match="not from observation"):
        adapter.resolve(step, _snapshot())


def test_a_hotkey_without_any_keys_is_refused(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="non-empty arguments.keys"):
        adapter.resolve(_step(action_type="hotkey", arguments={}), _snapshot())


def test_a_hotkey_token_outside_the_whitelist_is_refused(adapter: ActionAdapter) -> None:
    """The whitelist is what stops the model from driving a shell through a chord."""
    step = _step(action_type="hotkey", arguments={"keys": ["ctrl", "rm"]})

    with pytest.raises(ActionResolutionError, match="not allowed"):
        adapter.resolve(step, _snapshot())


def test_blank_hotkey_tokens_are_skipped(adapter: ActionAdapter) -> None:
    step = _step(action_type="hotkey", arguments={"keys": ["", "  ", "c"]})

    assert adapter.resolve(step, _snapshot()).action.keys == ["c"]


def test_a_hotkey_that_boils_down_to_nothing_is_refused(adapter: ActionAdapter) -> None:
    step = _step(action_type="hotkey", arguments={"keys": ["", "   "]})

    with pytest.raises(ActionResolutionError, match="empty key list"):
        adapter.resolve(step, _snapshot())


def test_key_press_without_a_key_is_refused(adapter: ActionAdapter) -> None:
    with pytest.raises(ActionResolutionError, match="requires arguments.key"):
        adapter.resolve(_step(action_type="key_press", arguments={}), _snapshot())
