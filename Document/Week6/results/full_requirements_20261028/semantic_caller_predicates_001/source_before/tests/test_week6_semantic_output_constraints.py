"""Bound semantic output without discarding replies or forcing a ready plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal

from gui_agent.planning.semantic_workflow import SemanticGoal, SemanticPlan
from gui_agent.planning.workflow_decomposer import ProposedSubgoal


def _bad_field(field):
    native = json.loads(proposal())
    if field == "reason":
        native["reason"] = "R" * 301
    elif field == "instruction":
        native["subgoals"][0][field] = "Configure " + "X" * 591
    elif field in {"expect_text", "forbid_text"}:
        native["subgoals"][0][field] = ["X" * 201]
    else:
        native["subgoals"][0][field.removesuffix("_count")] = [f"Unique check {i}" for i in range(9)]
    return json.dumps(native)


def test_semantic_success_preserves_output_limits_and_keeps_caller_future_text(tmp_path):
    native = json.loads(proposal())
    native["reason"] = ""
    native["subgoals"][1]["expect_text"] = ["Final accepted"]
    runner, request, state, client, _ = lab(tmp_path, responses=[json.dumps(native), plan(), plan()])
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 2
    assert result.schema_version == "week6.semantic_workflow.v1"
    assert result.plans[0].schema_version == "week6.semantic_subtasks.v4"
    assert result.plans[0].input_modality == "screen_text_context"
    assert result.plans[0].image_attached is False
    schema = client.calls[0]["kwargs"]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["reason"]["maxLength"] == 300
    assert schema["properties"]["subgoals"]["maxItems"] == 8
    goal = schema["$defs"]["SemanticGoal"]["properties"]
    assert goal["instruction"]["maxLength"] == 600
    for field in ("expect_text", "forbid_text"):
        assert goal[field]["maxItems"] == 8 and goal[field]["items"]["maxLength"] == 200
    protocol = json.loads((tmp_path / "semantic/semantic_plan_round_001/protocol.json").read_text("utf8"))
    assert protocol["schema_version"] == result.plans[0].schema_version
    assert protocol["output_constraints"]["reason_max_characters"] == 300
    assert protocol["caller_request"] == request.model_dump(mode="json")


@pytest.mark.parametrize("field", ["reason", "instruction", "expect_text", "forbid_text",
                                  "expect_text_count", "forbid_text_count"])
def test_over_limit_native_fields_fail_before_child_and_keep_exact_reply(tmp_path, field):
    content = _bad_field(field)
    runner, request, state, client, _ = lab(tmp_path, responses=[content])
    original = request.model_dump()
    result = runner.run(request, options())
    assert result.status == "failed" and state.children == [] and state.attempts == 0
    assert result.planning_calls == result.model_requests == len(client.calls) == 1
    assert request.model_dump() == original
    response_path = tmp_path / "semantic/semantic_plan_round_001/attempt_01_response.json"
    response = json.loads(response_path.read_text("utf8"))
    assert response["content"] == response["raw_response"]["unabridged"] == content
    assert (response_path.parent / "attempt_01_error.json").is_file()
    assert not (response_path.parent / "compiled_workflow.json").exists()


def test_exact_length_boundaries_are_accepted_locally_without_relaxing_legacy():
    goal = SemanticGoal(goal_id="Configure", instruction="Configure " + "X" * 590,
                        expect_text=["E" * 200, *[f"Expected {i}" for i in range(7)]],
                        forbid_text=["F" * 200, *[f"Forbidden {i}" for i in range(7)]])
    assert len(goal.instruction) == 600 and len(goal.expect_text) == len(goal.forbid_text) == 8
    assert SemanticPlan(status="refused", reason="R" * 300, subgoals=[]).reason == "R" * 300
    # These narrower limits belong to the new semantic protocol, not old APIs.
    old = ProposedSubgoal(instruction="Configure " + "X" * 591,
                          expect_text=["E" * 201, *[f"Expected {i}" for i in range(8)]])
    assert len(old.instruction) == 601 and len(old.expect_text) == 9


@pytest.mark.parametrize("status,reason", [
    ("needs_clarification", "The request does not identify which local document to modify."),
    ("refused", "The requested operation would send content to another person."),
])
def test_short_non_ready_responses_remain_valid_and_are_never_forced_ready(tmp_path, status, reason):
    native = {"status": status, "reason": reason, "subgoals": []}
    runner, request, state, client, _ = lab(tmp_path, responses=[json.dumps(native)])
    result = runner.run(request, options())
    assert result.status == status and result.plans[0].proposal.reason == reason
    assert result.plans[0].workflow is None and state.children == [] and state.attempts == 0
    assert len(client.calls) == result.planning_calls == result.model_requests == 1


@pytest.mark.parametrize("field", ["reason", "instruction"])
def test_repeated_complete_clauses_below_size_cap_are_rejected(tmp_path, field):
    native = json.loads(proposal())
    repeated = "The requested option is not available. The requested option is not available."
    assert len(repeated) < 300
    if field == "reason":
        native["reason"] = repeated
    else:
        native["subgoals"][0][field] = repeated
    runner, request, state, _, _ = lab(tmp_path, responses=[json.dumps(native)])
    result = runner.run(request, options())
    assert result.status == "failed" and state.children == [] and state.attempts == 0
    assert "repeated clauses" in result.plans[0].error


def test_large_loop_reply_is_archived_fully_but_not_echoed_into_counted_repair(tmp_path):
    # Reproduces the observed long non-ready reason loop, without fixture labels.
    bad = json.dumps({"status": "needs_clarification", "reason":
                      "The requested option is unavailable and another option seems selected. " * 350,
                      "subgoals": []})
    assert len(bad) > 21000
    runner, request, state, client, _ = lab(tmp_path, responses=[bad, proposal(), plan(), plan()],
                                          format_retries=1)
    original = request.model_dump()
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 2
    assert result.model_requests == result.planning_calls == len(client.calls) == 4
    assert result.plans[0].planning_calls == result.plans[0].model_requests == 2
    directory = tmp_path / "semantic/semantic_plan_round_001"
    first_response = json.loads((directory / "attempt_01_response.json").read_text("utf8"))
    assert first_response["content"] == first_response["raw_response"]["unabridged"] == bad
    correction = json.loads((directory / "attempt_02_request.json").read_text("utf8"))["context"]
    assert "previous_response" not in correction
    assert correction["previous_response_characters"] == len(bad)
    assert Path(correction["previous_response_record"]) == (directory / "attempt_01_response.json").resolve()
    assert correction["original_caller_request"] == original == request.model_dump()
    assert json.loads(client.calls[1]["messages"][-1]["content"])["context"] == correction
    assert "max_tokens" not in client.calls[0]["kwargs"]


def test_loop_repair_cannot_add_calls_after_shared_request_cap(tmp_path):
    bad = json.dumps({"status": "needs_clarification", "reason": "Repeated unavailable option. " * 900,
                      "subgoals": []})
    runner, request, state, client, _ = lab(tmp_path, responses=[bad, proposal()],
                                          format_retries=2, max_requests=1)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0 and state.children == []
    assert result.model_requests == len(client.calls) == 1
    assert result.planning_calls == 2  # rejected wrapper call is still a planning attempt


def test_nonrepeated_sentence_can_contain_repeated_words():
    reason = "The document named Settings needs a location before Settings can be modified."
    assert SemanticPlan(status="needs_clarification", reason=reason, subgoals=[]).reason == reason
