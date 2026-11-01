"""Model-owned goal grouping and counted repair, without a real model or GUI.

The fixed replies test the repair contract and immutable evidence. They do not
claim a real model follows the revised prompt; live acceptance checks that.
"""

from __future__ import annotations

import json

import pytest
from test_week6_automatic_decomposition import _frame
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal

from gui_agent.planning.semantic_workflow import (
    SEMANTIC_SYSTEM_PROMPT,
    SemanticDecomposer,
    SemanticPlan,
)


def _duplicate_and_grouped():
    first = {"goal_id": "ApplyPreference", "instruction": "Apply the requested preference.",
             "expect_text": ["Verified part 1"], "forbid_text": []}
    verify = {"goal_id": "VerifyConfig", "instruction": "Verify the resulting configuration.",
              "expect_text": ["Verified part 2", "Final accepted"], "forbid_text": []}
    persist = {"goal_id": "PersistConfig", "instruction": "Persist the verified configuration locally.",
               "expect_text": ["Final accepted", "Verified part 2"], "forbid_text": []}
    grouped = {"goal_id": "VerifyAndPersist", "instruction": "Verify the resulting configuration and persist it locally.",
               "expect_text": ["Verified part 2", "Final accepted"], "forbid_text": []}
    return proposal(first, verify, persist), proposal(first, grouped)


def test_counted_model_repair_groups_shared_outcome_without_local_plan_rewrite(tmp_path):
    duplicate, grouped = _duplicate_and_grouped()
    runner, request, state, client, _ = lab(tmp_path, responses=[duplicate, grouped, plan(), plan()], format_retries=1)
    request.instruction = "Apply the requested preference, verify the resulting configuration, and persist it locally."
    original = request.model_dump(mode="json")
    result = runner.run(request, options())
    assert result.status == "succeeded", result.error
    assert result.model_requests == result.planning_calls == len(client.calls) == 4
    assert result.plans[0].planning_calls == 2 and state.attempts == 2
    assert [entry["goal_id"] for entry in result.completed_subgoals] == ["ApplyPreference", "VerifyAndPersist"]
    assert request.model_dump(mode="json") == original
    assert all(request.instruction in task.instruction for task in state.tasks)
    directory = tmp_path / "semantic/semantic_plan_round_001"
    assert json.loads((directory / "attempt_01_response.json").read_text("utf8"))["content"] == duplicate
    assert json.loads((directory / "attempt_02_response.json").read_text("utf8"))["content"] == grouped
    error = json.loads((directory / "attempt_01_error.json").read_text("utf8"))["error"]
    assert "VerifyConfig" in error and "PersistConfig" in error
    assert "combine their unfinished required operations" in error and "preserve every requested operation" in error
    repair = json.loads((directory / "attempt_02_request.json").read_text("utf8"))
    assert repair["context"]["constraint_error"] == error
    assert repair["context"]["previous_response"] == duplicate
    assert json.loads((directory / "native_proposal.json").read_text("utf8")) == json.loads(grouped)
    compiled = json.loads((directory / "compiled_workflow.json").read_text("utf8"))
    assert compiled["native_proposal"] == json.loads(grouped)
    assert len(compiled["mapping"]) == 2  # No executable version of the bad three-goal reply.
    assert compiled["mapping"][1]["native_instruction"] == json.loads(grouped)["subgoals"][1]["instruction"]
    assert result.final_verification.passed


def test_three_identical_bad_replies_remain_rejected_without_forced_grouping(tmp_path):
    duplicate, _ = _duplicate_and_grouped()
    runner, request, state, client, _ = lab(tmp_path, responses=[duplicate] * 3, format_retries=2)
    result = runner.run(request, options())
    assert result.status == "failed" and result.model_requests == result.planning_calls == len(client.calls) == 3
    assert state.children == [] and state.attempts == 0 and not result.completed_subgoals
    directory = tmp_path / "semantic/semantic_plan_round_001"
    for number in range(1, 4):
        assert json.loads((directory / f"attempt_{number:02d}_response.json").read_text("utf8"))["content"] == duplicate
    assert not (directory / "compiled_workflow.json").exists()


def test_grouping_repair_cannot_exceed_actual_http_cap(tmp_path):
    duplicate, grouped = _duplicate_and_grouped()
    runner, request, state, client, _ = lab(tmp_path, responses=[duplicate, grouped], format_retries=2, max_requests=1)
    result = runner.run(request, options())
    assert result.status == "failed" and len(client.calls) == result.model_requests == 1
    assert result.planning_calls == 2 and state.attempts == 0 and state.children == []


@pytest.mark.parametrize("identity", ["goal_id", "instruction"])
def test_duplicate_identity_is_not_told_to_combine_different_objectives(tmp_path, identity):
    native = json.loads(proposal())
    native["subgoals"][1][identity] = native["subgoals"][0][identity]
    runner, request, state, _, _ = lab(tmp_path, responses=[json.dumps(native)])
    result = runner.run(request, options())
    assert result.status == "failed" and "duplicate" in result.error
    assert "combine their unfinished required operations" not in result.error
    assert state.children == [] and state.attempts == 0


@pytest.mark.parametrize("identity", ["goal_id", "instruction", "checks"])
def test_completed_ledger_conflicts_never_recommend_merging_or_replaying_completed_work(tmp_path, identity):
    _, request, _, _, _ = lab(tmp_path)
    completed = json.loads(proposal())["subgoals"][0]
    candidate = json.loads(proposal())["subgoals"][1]
    if identity == "checks":
        candidate["expect_text"], candidate["forbid_text"] = completed["expect_text"], completed["forbid_text"]
    else:
        candidate[identity] = completed[identity]
    native = SemanticPlan.model_validate(json.loads(proposal(candidate)), strict=True)
    with pytest.raises(ValueError, match="completed") as error:
        SemanticDecomposer()._validate(native, request, _frame("before", ["Pending work"]), [completed], revision=True)
    assert "cannot be replayed" in str(error.value) and "keep completed objectives unchanged" in str(error.value)
    assert "combine their unfinished required operations" not in str(error.value)


def test_generic_prompt_covers_all_operations_and_places_overall_final_outcome_last():
    text = " ".join(SEMANTIC_SYSTEM_PROMPT.split())
    assert "covering ALL requirements" in text and "at least TWO distinct" in text
    assert "same COMPLETE expect_text/forbid_text set may belong to only ONE goal" in text
    assert "Do not drop any requested operation" in text
    assert "never combine new work into an already completed goal" in text
    assert "Put the overall final-state conjunction on the last goal" in text
    assert all(label not in text for label in ("ALPHA", "BETA", "GREEN", "BLUE", "LOCAL COMPLETE", "Week6 Local Lab"))
