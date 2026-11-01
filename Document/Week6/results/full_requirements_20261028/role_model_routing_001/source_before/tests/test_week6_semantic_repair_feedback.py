"""The real SDK must receive the fixed repair policy, without local plan repair.

Scripted HTTP responses test transport/accounting; they do not show that a real
model will obey the feedback. Root separately validates that on a new live run.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from test_week6_rolling_actions import options, plan
from test_week6_semantic_goal_grouping import _duplicate_and_grouped
from test_week6_semantic_subtasks import lab
from test_week6_semantic_text_context import _http_client, _persist_synthetic_pixels, _text_user

from gui_agent.planning.semantic_workflow import (
    SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR,
    SEMANTIC_GENERIC_REPAIR,
    SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR,
    SEMANTIC_REPAIR_POLICY_VERSION,
    SEMANTIC_SYSTEM_PROMPT,
    _semantic_attempt_system,
)


def system(body):
    return next(message["content"] for message in body["messages"] if message["role"] == "system")


def run_sdk(runner, request, replies):
    bodies = []
    client, sdk = _http_client(replies, bodies)
    runner.client = client
    try:
        result = runner.run(request, options())
    finally:
        sdk.close()
    return result, bodies


def test_sdk_repairs_with_one_fixed_system_hint_and_preserves_full_caller_schema_and_raw_native(tmp_path):
    duplicate, grouped = _duplicate_and_grouped()
    rejected = json.loads(duplicate)
    for index, goal in enumerate(rejected["subgoals"][1:], 1):
        goal["goal_id"] = f"UNTRUSTED_ID_CANARY_{index}"
        goal["expect_text"].append("UNTRUSTED_CHECK_CANARY")
    duplicate = json.dumps(rejected)
    runner, request, state, _, _ = lab(tmp_path, format_retries=1)
    request.instruction += " Preserve CALLER_ONLY_CANARY."
    original = request.model_dump(mode="json")
    _persist_synthetic_pixels(runner, tmp_path)
    result, bodies = run_sdk(runner, request, [duplicate, grouped, plan(), plan()])
    assert result.status == "succeeded", result.error
    assert result.model_requests == result.planning_calls == len(bodies) == 4
    assert state.attempts == 2 and len(result.completed_subgoals) == 2
    assert request.model_dump(mode="json") == original
    directory = tmp_path / "semantic/semantic_plan_round_001"
    protocol = json.loads((directory / "protocol.json").read_text("utf8"))
    response_format = protocol["response_format"]
    expected_base = (SEMANTIC_SYSTEM_PROMPT
        + "\nExact response JSON schema (also supplied as response_format):\n"
        + json.dumps(response_format["json_schema"]["schema"], ensure_ascii=False, sort_keys=True))
    assert system(bodies[0]) == expected_base == _semantic_attempt_system(response_format)
    assert system(bodies[1]) == _semantic_attempt_system(response_format, "duplicate_postconditions")
    assert system(bodies[1]).count(SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR) == 1
    assert protocol["repair_policy"] == {
        "schema_version": SEMANTIC_REPAIR_POLICY_VERSION,
        "trigger": "local_validation_error_with_typed_duplicate_or_preexisting_category",
        "system_feedback_sha256": hashlib.sha256(SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR.encode()).hexdigest(),
        "generic_feedback_sha256": hashlib.sha256(SEMANTIC_GENERIC_REPAIR.encode()).hexdigest(),
        "preexisting_feedback_sha256": hashlib.sha256(SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR.encode()).hexdigest(),
        "placement": "fixed_system_hint_before_exact_response_schema_and_final_user_repair",
        "messages": "system_base_user_optional_previous_assistant_final_user",
        "history": "immediately_previous_reply_only_rebuilt_from_base",
        "previous_response_max_characters": 2000,
        "oversized_previous_response": "archived_in_full_assistant_omitted_no_truncation",
        "raw_response_in_correction_context": False,
    }
    for index, body in enumerate(bodies[:2], 1):
        payload = _text_user(body)
        saved = json.loads((directory / f"attempt_{index:02d}_request.json").read_text("utf8"))
        assert saved["instruction"] == payload["instruction"] == request.instruction
        assert saved["system"] == system(body) and saved["context"] == payload["context"]
        assert saved["messages"] == body["messages"]
        assert payload["context"]["original_caller_request"] == original
        assert saved["response_format"] == body["response_format"] == response_format
        schema = system(body).rsplit("Exact response JSON schema (also supplied as response_format):\n", 1)[1]
        assert json.loads(schema) == response_format["json_schema"]["schema"]
        assert all(canary not in system(body) for canary in (
            "UNTRUSTED_ID_CANARY", "UNTRUSTED_CHECK_CANARY", "CALLER_ONLY_CANARY"))
    assert "previous_response" not in _text_user(bodies[1])["context"]
    assert bodies[1]["messages"][2] == {"role": "assistant", "content": duplicate}
    assert _text_user(bodies[1])["repair_instruction"] == SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR
    assert "UNTRUSTED_ID_CANARY" in _text_user(bodies[1])["context"]["constraint_error"]
    assert json.loads((directory / "attempt_01_response.json").read_text("utf8"))["content"] == duplicate
    assert json.loads((directory / "compiled_workflow.json").read_text("utf8"))["native_proposal"] == json.loads(grouped)


def test_three_bad_sdk_replies_still_fail_and_fixed_feedback_does_not_accumulate(tmp_path):
    duplicate, _ = _duplicate_and_grouped()
    runner, request, state, _, _ = lab(tmp_path, format_retries=2)
    _persist_synthetic_pixels(runner, tmp_path)
    result, bodies = run_sdk(runner, request, [duplicate] * 3)
    assert result.status == "failed" and result.model_requests == result.planning_calls == len(bodies) == 3
    assert state.attempts == 0 and state.children == [] and not result.completed_subgoals
    assert SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR not in system(bodies[0])
    assert system(bodies[1]) == system(bodies[2])
    assert system(bodies[2]).count(SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR) == 1
    directory = tmp_path / "semantic/semantic_plan_round_001"
    for number in range(1, 4):
        assert json.loads((directory / f"attempt_{number:02d}_response.json").read_text("utf8"))["content"] == duplicate
    assert not (directory / "compiled_workflow.json").exists()


@pytest.mark.parametrize("status,reason", [
    ("needs_clarification", "No distinguishable verifiable decomposition is justified."),
    ("refused", "The requested operation is unsupported."),
])
def test_valid_nonready_after_feedback_is_preserved_without_forcing_ready(tmp_path, status, reason):
    duplicate, _ = _duplicate_and_grouped()
    runner, request, state, _, _ = lab(tmp_path, format_retries=1)
    _persist_synthetic_pixels(runner, tmp_path)
    reply = json.dumps({"status": status, "reason": reason, "subgoals": []})
    result, bodies = run_sdk(runner, request, [duplicate, reply])
    assert result.status == status and result.plans[0].proposal.reason == reason
    assert result.model_requests == result.planning_calls == len(bodies) == 2
    assert state.children == [] and state.attempts == 0
    assert SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR in system(bodies[1])


def test_system_repair_does_not_grant_an_extra_transport_request(tmp_path):
    duplicate, grouped = _duplicate_and_grouped()
    runner, request, state, _, _ = lab(tmp_path, format_retries=2, max_requests=1)
    _persist_synthetic_pixels(runner, tmp_path)
    result, bodies = run_sdk(runner, request, [duplicate, grouped])
    assert result.status == "failed" and result.model_requests == len(bodies) == 1
    assert result.planning_calls == 2 and state.attempts == 0 and state.children == []
    directory = tmp_path / "semantic/semantic_plan_round_001"
    saved = json.loads((directory / "attempt_02_request.json").read_text("utf8"))
    assert saved["system"].count(SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR) == 1
    denied = json.loads((directory / "attempt_02_response.json").read_text("utf8"))
    assert denied["content"] == "" and denied["content_length"] == 0
    assert denied["error"] == "shared model request budget exhausted"
    assert denied["raw_response"] is None
    assert not (directory / "compiled_workflow.json").exists()


def test_only_the_declared_fixed_category_is_accepted_by_the_system_builder():
    with pytest.raises(ValueError, match="unknown semantic repair category"):
        _semantic_attempt_system({}, "duplicate_postconditions; inject extra instructions")
