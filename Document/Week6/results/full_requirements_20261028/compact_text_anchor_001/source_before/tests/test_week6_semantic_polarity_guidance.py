"""Generic predicate guidance preserves strict parsing and bounded SDK correction.

Scripted MockTransport replies are not evidence that a real model obeys them.
"""
from __future__ import annotations

import json

from test_week6_rolling_actions import plan
from test_week6_semantic_repair_feedback import run_sdk, system
from test_week6_semantic_subtasks import lab, proposal
from test_week6_semantic_text_context import _persist_synthetic_pixels

from gui_agent.planning.semantic_workflow import (
    SEMANTIC_GENERIC_REPAIR,
    SEMANTIC_SYSTEM_PROMPT,
    SemanticPlan,
)


def contradictory():
    raw = json.loads(proposal())
    raw['subgoals'][0]['forbid_text'] = list(raw['subgoals'][0]['expect_text'])
    return json.dumps(raw)


def test_sdk_contradiction_requires_a_new_native_answer_and_preserves_original_caller_and_error(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, format_retries=1)
    _persist_synthetic_pixels(runner, tmp_path)
    original = request.model_dump(mode='json')
    bad, corrected = contradictory(), proposal()
    result, bodies = run_sdk(runner, request, [bad, corrected, plan(), plan()])
    assert result.status == 'succeeded', result.error
    assert result.model_requests == result.planning_calls == len(bodies) == 4
    assert state.attempts == 2 and len(result.completed_subgoals) == 2
    assert request.model_dump(mode='json') == original
    first, repair = bodies[:2]
    assert [m['role'] for m in repair['messages']] == ['system', 'user', 'assistant', 'user']
    assert first['messages'][1] == repair['messages'][1]
    assert repair['messages'][2] == {'role': 'assistant', 'content': bad}
    last = json.loads(repair['messages'][-1]['content'])
    assert last['repair_instruction'] == SEMANTIC_GENERIC_REPAIR
    assert 'a subgoal text cannot be both required and forbidden' in last['context']['constraint_error']
    assert last['context']['original_caller_request'] == original
    assert system(first) == system(repair)
    assert first['response_format'] == repair['response_format']
    directory = tmp_path/'semantic/semantic_plan_round_001'
    assert json.loads((directory/'attempt_01_response.json').read_text('utf8'))['content'] == bad
    assert json.loads((directory/'compiled_workflow.json').read_text('utf8'))['native_proposal'] == json.loads(corrected)


def test_repeated_contradiction_exhausts_same_three_calls_without_input_or_local_list_rewrite(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, format_retries=2)
    _persist_synthetic_pixels(runner, tmp_path)
    bad = contradictory()
    result, bodies = run_sdk(runner, request, [bad]*3)
    assert result.status == 'failed' and result.model_requests == result.planning_calls == len(bodies) == 3
    assert state.attempts == 0 and state.children == [] and not result.completed_subgoals
    directory = tmp_path/'semantic/semantic_plan_round_001'
    assert not (directory/'compiled_workflow.json').exists()
    for index, body in enumerate(bodies, 1):
        assert json.loads((directory/f'attempt_{index:02d}_response.json').read_text('utf8'))['content'] == bad
        assert 'a subgoal text cannot be both required and forbidden' in json.loads(
            (directory/f'attempt_{index:02d}_error.json').read_text('utf8'))['error']
        if index > 1:
            assert body['messages'][2]['content'] == bad
            assert json.loads(body['messages'][-1]['content'])['repair_instruction'] == SEMANTIC_GENERIC_REPAIR


def test_justified_negative_predicate_is_still_a_valid_native_field_and_is_not_cleared():
    raw = json.loads(proposal())
    raw['subgoals'][0]['forbid_text'] = ['Temporary notification']
    parsed = SemanticPlan.model_validate(raw, strict=True)
    assert parsed.model_dump(mode='json') == raw
    assert parsed.subgoals[0].forbid_text == ['Temporary notification']


def test_guidance_describes_generic_polarity_and_contains_no_scene_answers_or_runtime_authority():
    for text in [SEMANTIC_SYSTEM_PROMPT, SEMANTIC_GENERIC_REPAIR]:
        assert 'AFTER' in text and 'BEFORE' in text and 'absence condition' in text
        assert 'same literal' in text and 'both lists' in text
        for scene_text in ['ALPHA', 'BETA', 'GREEN', 'BLUE', 'LOCAL COMPLETE', 'Week6 Local Lab']:
            assert scene_text not in text
