"""Literal description conflicts refuse dispatch without locally repairing the model."""
from __future__ import annotations

import json

import pytest
from test_week6_grounding_consistency import frame, native, runtime, step

from gui_agent.runtime.action_adapter import (
    ActionAdapter,
    ActionResolutionError,
    validate_target_consistency,
)
from gui_agent.runtime.schemas import ExecutionOptions


@pytest.mark.parametrize('description', [
    "Click 'Dismiss'", 'Select the "Dismiss" option.',
    'Double-click on “Dismiss”.', 'Tap ‘Dismiss’.',
])
@pytest.mark.parametrize('index', [0, 2])
def test_quoted_target_cannot_name_another_readable_control(description, index):
    observation = frame()
    value = step(index=index)
    value.description = description
    before = value.model_dump()
    with pytest.raises(ActionResolutionError, match='quoted target conflict:'):
        ActionAdapter().resolve(value, observation)
    assert value.model_dump() == before


def test_missing_literal_is_not_permission_to_choose_an_unrelated_existing_label():
    value = step(index=0)
    value.description = "Select the 'Export' button."
    with pytest.raises(ActionResolutionError, match='quoted target conflict:'):
        validate_target_consistency(value, frame())


@pytest.mark.parametrize('arguments', [{}, {'element_id': 'obs-0001-e000'}])
def test_both_text_only_and_bare_id_claims_are_checked(arguments):
    value = step(index=0)
    value.description = "Click 'Dismiss'"
    value.arguments = arguments
    if arguments:
        value.target_text = None
    with pytest.raises(ActionResolutionError, match='quoted target conflict:'):
        ActionAdapter().resolve(value, frame())


@pytest.mark.parametrize('description', [
    "Click 'Accept'", 'Select the "accept" option.',
    "Click the control after 'Dismiss' is shown.",
    "Review the 'Dismiss' result before choosing another control.",
])
def test_matching_literal_and_descriptions_outside_the_narrow_target_syntax_keep_existing_grounding(description):
    value = step(index=0)
    value.description = description
    resolved = ActionAdapter().resolve(value, frame())
    assert (resolved.action.x, resolved.action.y) == (60, 30)


def test_unknown_visual_candidate_does_not_acquire_an_ocr_witness_from_the_description():
    observation = frame()
    observation.elements = [item for item in observation.elements if item.element_id.endswith('002')]
    value = step(index=2, target='')
    value.description = "Click 'Export'"
    validate_target_consistency(value, observation)
    assert observation.elements[0].text == ''


def test_existing_text_id_conflict_guard_still_applies_when_the_quote_matches_the_id():
    value = step(index=1, target='Accept')
    value.description = "Click 'Dismiss'"
    with pytest.raises(ActionResolutionError, match='target consistency conflict:'):
        ActionAdapter().resolve(value, frame())


def test_runner_refuses_original_plan_and_only_a_new_native_answer_can_dispatch(tmp_path):
    frames = [frame(f'obs-{index:04d}', finished=index >= 5) for index in range(1, 7)]
    bad = json.loads(native('obs-0001', 0))
    bad['steps'][0]['description'] = "Select the 'Dismiss' button."
    original = json.dumps(bad)
    corrected = native('obs-0003', 0)
    runner, task, executor, client, _ = runtime(tmp_path, frames, [original, corrected])
    result = runner.run(task, ExecutionOptions(execute=True, confirm=False,
                                             max_actions=1, max_planning_attempts=2))
    assert result.status == 'succeeded'
    assert result.action_count == len(executor.actions) == 1
    assert result.planning_calls == len(client.calls) == 2
    assert result.steps[0].action_result is None
    assert result.steps[0].error.startswith('quoted target conflict:')
    assert json.loads(original) == bad


def test_repeated_literal_conflicts_exhaust_existing_budget_without_any_input(tmp_path):
    frames = [frame(f'obs-{index:04d}') for index in range(1, 5)]
    replies = []
    for oid in ['obs-0001', 'obs-0003']:
        value = json.loads(native(oid, 0))
        value['steps'][0]['description'] = "Select the 'Dismiss' button."
        replies.append(json.dumps(value))
    runner, task, executor, client, _ = runtime(tmp_path, frames, replies)
    result = runner.run(task, ExecutionOptions(execute=True, confirm=False,
                                             max_actions=1, max_planning_attempts=2))
    assert result.status == 'failed' and executor.actions == []
    assert result.planning_calls == len(client.calls) == 2 and result.action_count == 0
