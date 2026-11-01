"""Synthetic SDK regressions for rejecting model-added operations, not live efficacy."""
from __future__ import annotations

import json

import pytest
from test_week6_rolling_actions import options, plan
from test_week6_semantic_repair_feedback import run_sdk, system
from test_week6_semantic_subtasks import lab, proposal
from test_week6_semantic_text_context import _persist_synthetic_pixels

from gui_agent.planning.semantic_workflow import SEMANTIC_UNSUPPORTED_MESSAGE_REPAIR


@pytest.mark.parametrize("addition", [
    "Compose a message describing the local archive.",
    "Send an email after the requested local change.",
    "Post to Slack after the requested local change.",
    "发送完成通知。",
])
def test_added_operation_is_rejected_and_only_a_new_native_reply_can_repair_it(tmp_path, addition):
    runner, request, state, _, _ = lab(tmp_path, format_retries=1)
    _persist_synthetic_pixels(runner, tmp_path)
    before = request.model_dump(mode="json")
    invalid = json.loads(proposal())
    invalid["subgoals"][-1]["instruction"] = addition
    raw = json.dumps(invalid)
    result, bodies = run_sdk(runner, request, [raw, proposal(), plan(), plan()])
    assert result.status == "succeeded" and state.attempts == 2
    assert result.model_requests == result.planning_calls == len(bodies) == 4
    assert bodies[1]["messages"][1] == bodies[0]["messages"][1]
    assert [m["role"] for m in bodies[1]["messages"]] == ["system", "user", "user"]
    assert all(m["content"] != raw for m in bodies[1]["messages"])
    packet = json.loads(bodies[1]["messages"][-1]["content"])
    assert packet["repair_instruction"] == SEMANTIC_UNSUPPORTED_MESSAGE_REPAIR
    assert packet["context"]["original_caller_request"] == before == request.model_dump(mode="json")
    assert "message sending is not supported" in packet["context"]["constraint_error"]
    assert addition not in system(bodies[1])
    saved = tmp_path / "semantic/semantic_plan_round_001/attempt_01_response.json"
    assert json.loads(saved.read_text("utf8"))["content"] == raw


def test_repeated_unsupported_addition_uses_existing_budget_and_dispatches_nothing(tmp_path):
    invalid = json.loads(proposal())
    invalid["subgoals"][-1]["instruction"] = "Add a message to describe the local archive."
    raw = json.dumps(invalid)
    runner, request, state, client, _ = lab(tmp_path, responses=[raw, raw, raw], format_retries=2)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0 and state.children == []
    assert result.model_requests == result.planning_calls == len(client.calls) == 3
    assert result.total_actions == 0 and result.unconfirmed_action_attempts == 0


@pytest.mark.parametrize("status", ["refused", "needs_clarification"])
def test_repair_preserves_native_nonready_outcomes_without_forcing_execution(tmp_path, status):
    invalid = json.loads(proposal())
    invalid["subgoals"][-1]["instruction"] = "Send a message about the archive."
    raw = json.dumps(invalid)
    nonready = json.dumps({"status": status, "reason": "No justified supported workflow.", "subgoals": []})
    runner, request, state, client, _ = lab(tmp_path, responses=[raw, nonready], format_retries=1)
    result = runner.run(request, options())
    assert result.status == status and state.attempts == 0 and state.children == []
    assert result.model_requests == result.planning_calls == len(client.calls) == 2
