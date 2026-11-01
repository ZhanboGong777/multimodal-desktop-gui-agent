"""Untrusted intermediate predictions retain independent caller-final authority.

SDK wire tests use MockTransport; no external API, capture or OS input occurs.
These tests do not establish real model planning quality.
"""

from __future__ import annotations

import json

import pytest
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal
from test_week6_semantic_text_context import _http_client, _persist_synthetic_pixels, _text_user

from gui_agent.planning.semantic_workflow import SemanticDecomposer
from gui_agent.planning.workflow_decomposer import BudgetedModelClient


@pytest.mark.parametrize("marker", ["[not specified]", "[unspecified]", "[unknown]", "<TBD>", "[未指定]", "[待填写]"])
@pytest.mark.parametrize("location", ["instruction", "predicate"])
def test_reserved_unresolved_forms_stop_before_observation_http_or_input(tmp_path, marker, location):
    runner, request, state, client, _ = lab(tmp_path)
    if location == "instruction":
        request.instruction += f" Required local destination: {marker}."
    else:
        request.expect_text.append(f"Destination {marker}")
    original = request.model_dump(mode="json")
    result = runner.run(request, options())
    assert result.status == "needs_clarification" and marker in result.error
    assert state.captures == state.attempts == result.model_requests == result.planning_calls == 0
    assert not state.children and not client.calls and not result.plans
    directory = tmp_path / "semantic"
    assert json.loads((directory / "semantic_request.json").read_text("utf8"))["caller_request"] == original
    assert json.loads((directory / "semantic_preflight.json").read_text("utf8"))["supported"] is False
    assert not (directory / "initial_observation").exists()


@pytest.mark.parametrize("word", ["unknown", "TBD", "unknown-looking", "[well known]"])
def test_unbracketed_words_are_not_claimed_to_be_missing_parameter_detection(tmp_path, word):
    refusal = {"status": "needs_clarification", "reason": "The caller has not identified the destination.", "subgoals": []}
    runner, request, state, client, _ = lab(tmp_path, responses=[json.dumps(refusal)])
    request.instruction += f" Preserve the document named {word}."
    result = runner.run(request, options())
    assert result.status == "needs_clarification" and result.error == refusal["reason"]
    assert result.model_requests == len(client.calls) == 1 and state.captures == 1
    assert state.attempts == 0


@pytest.mark.parametrize("kind", ["long_positive", "long_negative", "single", "case_alias", "contradiction"])
def test_only_final_contradiction_still_preflights_not_old_pool_representation_limits(tmp_path, kind):
    refusal = {"status": "needs_clarification", "reason": "A verifiable intermediate result is unclear.", "subgoals": []}
    runner, request, state, client, _ = lab(tmp_path, responses=[json.dumps(refusal)])
    if kind == "long_positive":
        request.expect_text.append("P" * 201)
    elif kind == "long_negative":
        request.forbid_text.append("N" * 201)
    elif kind == "single":
        request.expect_text, request.forbid_text = ["One signal"], []
    elif kind == "case_alias":
        request.expect_text, request.forbid_text = ["One signal", "ONE SIGNAL"], []
    else:
        request.expect_text, request.forbid_text = ["Result accepted", "Saved"], ["ACCEPTED"]
    original = request.model_dump(mode="json")
    result = runner.run(request, options())
    assert result.status == "needs_clarification"
    if kind == "contradiction":
        assert state.captures == result.model_requests == result.planning_calls == 0 and not client.calls
    else:
        assert result.error == refusal["reason"] and state.captures == result.model_requests == result.planning_calls == 1
        assert len(client.calls) == 1
    assert not state.children and state.attempts == 0
    assert request.model_dump(mode="json") == original


@pytest.mark.parametrize("field,bad,meaning", [
    ("expect_text", "ERROR", "unobserved_prediction"), ("forbid_text", "Verified part 1", "final_conflict"),
    ("expect_text", " Verified part 1", "raw_rejection"), ("expect_text", "Verified part 1 ", "raw_rejection"),
    ("expect_text", "VERIFIED PART 1", "observed_prediction"), ("forbid_text", " ERROR ", "raw_rejection"),
    ("expect_text", "Invented future completion label", "unobserved_prediction"),
])
def test_predictions_are_not_pool_restricted_but_raw_integrity_final_conflicts_and_actual_evidence_still_apply(tmp_path, field, bad, meaning):
    native = json.loads(proposal())
    native["subgoals"][0][field] = [bad]
    if meaning == "final_conflict":
        # Keep this contract internally consistent so it reaches the distinct
        # immutable-caller conflict guard rather than the earlier shape guard.
        native["subgoals"][0]["expect_text"] = ["Verified part 2"]
    content = json.dumps(native)
    runner, request, state, client, _ = lab(tmp_path, responses=[content, plan(), plan()], goal_calls=1)
    result = runner.run(request, options())
    directory = tmp_path / "semantic/semantic_plan_round_001"
    assert json.loads((directory / "attempt_01_response.json").read_text("utf8"))["content"] == content
    if meaning in {"raw_rejection", "final_conflict"}:
        assert result.status == "failed" and state.attempts == 0 and state.children == [] and len(client.calls) == 1
        assert ("without stripping" if meaning == "raw_rejection" else "conflicts") in result.error
        assert not (directory / "compiled_workflow.json").exists()
    else:
        compiled = json.loads((directory / "compiled_workflow.json").read_text("utf8"))
        assert compiled["native_proposal"] == native and compiled["workflow"]["subgoals"][0][field] == [bad]
        if meaning == "observed_prediction":
            assert result.status == "succeeded" and state.attempts == 2
        else:
            assert result.status != "succeeded" and state.attempts == 1
            assert result.goal_outcomes[0]["status"] == "failed" and not result.completed_subgoals


def test_missing_native_list_is_not_filled_by_pydantic_defaults(tmp_path):
    native = json.loads(proposal())
    native["subgoals"][0].pop("forbid_text")
    runner, request, state, _, _ = lab(tmp_path, responses=[json.dumps(native)])
    result = runner.run(request, options())
    assert result.status == "failed" and "strict strings" in result.error
    assert state.children == [] and state.attempts == 0


@pytest.mark.parametrize("empty_side", ["expect_text", "forbid_text"])
def test_sdk_schema_bounds_predictions_even_when_caller_has_an_empty_side(tmp_path, empty_side):
    native = {"status": "needs_clarification", "reason": "These predicates do not identify verifiable stages.", "subgoals": []}
    runner, request, state, _, _ = lab(tmp_path)
    if empty_side == "expect_text":
        request.expect_text, request.forbid_text = [], ["Advance control", "Pending work"]
    else:
        request.forbid_text = []
    original = request.model_dump(mode="json")
    _persist_synthetic_pixels(runner, tmp_path)
    bodies = []
    client, sdk = _http_client([json.dumps(native)], bodies)
    runner.client = client
    try:
        result = runner.run(request, options())
    finally:
        sdk.close()
    assert result.status == "needs_clarification" and state.attempts == 0
    assert len(bodies) == 1
    schema = bodies[0]["response_format"]["json_schema"]["schema"]
    fields = schema["$defs"]["SemanticGoal"]["properties"]
    for field in ("expect_text", "forbid_text"):
        assert fields[field]["maxItems"] == 8 and "enum" not in fields[field]["items"]
        assert fields[field]["items"]["maxLength"] == 200
    system = bodies[0]["messages"][0]["content"]
    embedded = system.rsplit("Exact response JSON schema (also supplied as response_format):\n", 1)[1]
    assert json.loads(embedded) == schema
    actual = _text_user(bodies[0])
    assert actual["context"]["original_caller_request"] == original
    assert "current_observation" not in actual["context"]
    assert all(isinstance(message["content"], str) for message in bodies[0]["messages"])


def test_revision_context_sends_logical_history_and_failure_category_not_screen_evidence(tmp_path):
    runner, request, _, _, clock = lab(tmp_path)
    completed = json.loads(proposal())["subgoals"][0]
    completed = {**completed, "verification": {"evidence": {"text": "SECRET OCR", "image_path": "SECRET PIXELS"}},
                 "evidence_directory": "SECRET DIRECTORY", "status": "succeeded"}
    failure = {"status": "failed", "error": "SECRET ERROR", "verification": {"window_id": "SECRET WINDOW"}}
    bodies = []
    client, sdk = _http_client([proposal(json.loads(proposal())["subgoals"][1])], bodies)
    before = runner.initial_observer(tmp_path / "source")
    budgeted = BudgetedModelClient(client, max_requests=1, deadline=clock() + 50, clock=clock)
    try:
        result = SemanticDecomposer(max_format_retries=0, clock=clock).decompose(
            request, evidence_directory=tmp_path / "revision", budgeted_client=budgeted, initial_observation=before,
            completed_subgoals=[completed], max_calls=1, deadline=clock() + 50, revision=True, failure=failure)
    finally:
        sdk.close()
    assert result.status == "ready" and result.model_requests == 1
    saved = json.loads((tmp_path / "revision/attempt_01_request.json").read_text("utf8"))
    context = saved["context"]
    assert _text_user(bodies[0])["context"] == context
    assert all(isinstance(message["content"], str) for message in bodies[0]["messages"])
    assert context["completed_subgoals"] == [{key: completed[key] for key in
                                             ("goal_id", "instruction", "expect_text", "forbid_text")}]
    assert context["failure"] == {"category": "semantic_postconditions_unsatisfied"}
    assert "SECRET" not in json.dumps(context) and before.image_path not in json.dumps(context)
    archive = json.loads((tmp_path / "revision/context_source_archive.json").read_text("utf8"))
    assert archive["completed_subgoals"] == [completed] and archive["failure"] == failure
    assert json.loads((tmp_path / "revision/snapshot_archive.json").read_text("utf8")) == before.model_dump(mode="json")


def test_distinct_predicates_appearing_together_do_not_credit_a_second_unexecuted_goal(tmp_path):
    runner, request, state, _, _ = lab(tmp_path)
    original_factory = runner.runner_factory

    def factory(task, index, directory, budgeted):
        child = original_factory(task, index, directory, budgeted)
        execute = child.executor.execute

        def all_at_once(action, **kwargs):
            outcome = execute(action, **kwargs)
            if outcome.success and not outcome.dry_run:
                state.stage = 2
            return outcome

        child.executor.execute = all_at_once
        return child

    runner.runner_factory = factory
    result = runner.run(request, options())
    assert result.status == "blocked" and "already hold" in result.error
    assert result.total_actions == state.attempts == 1 and len(result.completed_subgoals) == 1
    assert len(state.children) == 1 and result.final_verification is None


def test_preexisting_predicted_control_text_cannot_be_used_as_initial_goal_credit(tmp_path):
    native = json.loads(proposal())
    native["subgoals"][0]["expect_text"] = ["Advance control"]
    runner, request, state, _, _ = lab(tmp_path, responses=[json.dumps(native)])
    result = runner.run(request, options())
    assert result.status == "failed" and "already hold" in result.error
    assert result.model_requests == 1 and state.attempts == 0 and state.children == []


def test_predicted_intermediate_conditions_execute_with_single_independent_caller_final(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, responses=[proposal(), plan(), plan()])
    request.expect_text, request.forbid_text = ["Final accepted"], []
    original = request.model_dump(mode="json")
    result = runner.run(request, options())
    assert result.status == "succeeded" and result.total_actions == state.attempts == 2
    assert len(result.completed_subgoals) == 2 and result.final_verification.passed
    assert request.model_dump(mode="json") == original
    assert all(not set(entry["expect_text"]) <= set(original["expect_text"]) for entry in result.completed_subgoals)
