"""Offline recovery and tamper boundaries; no real model or desktop input."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal

from gui_agent.planning.semantic_workflow import SemanticGoal, _goal_task, _logical_failure
from gui_agent.runtime.schemas import ObservationSnapshot


def initial_proposal():
    native = json.loads(proposal())
    native["subgoals"][1]["expect_text"] = ["Advance control", "Verified part 1"]
    return json.dumps(native)


def remaining_proposal():
    return proposal({"goal_id": "PersistRevised", "instruction": "Persist the remaining configuration",
                     "expect_text": ["Verified part 2"], "forbid_text": []})


def run_recovery(tmp_path, **kwargs):
    runner, request, state, client, clock = lab(
        tmp_path, responses=[initial_proposal(), plan(), remaining_proposal(), plan()],
        rounds=2, **kwargs)
    result = runner.run(request, options())
    return result, request, state, client, clock


def test_preexisting_later_contract_replans_only_unfinished_work_without_completion_credit(tmp_path):
    result, request, state, client, _ = run_recovery(tmp_path)
    assert result.status == "succeeded", result.error
    assert result.total_actions == state.attempts == len(state.tasks) == 2
    assert result.planning_calls == result.model_requests == len(client.calls) == 4
    assert [row["goal_id"] for row in result.completed_subgoals] == ["SetPreference", "PersistRevised"]
    assert len(result.goal_outcomes) == len(result.workflow_results) == 2
    assert len(result.rejected_goal_contracts) == 1
    rejected = result.rejected_goal_contracts[0]
    assert rejected["goal"]["goal_id"] == "PersistConfig"
    assert rejected["confirmed_inputs"] == 0 and rejected["eligible_for_semantic_credit"] is False
    root = tmp_path / "semantic"
    assert json.loads((root / "goal_rejection_001_002.json").read_text("utf8")) == rejected
    assert json.loads((root / "semantic_rejected_contracts.json").read_text("utf8")) == [rejected]
    assert not (root / "goal_contract_001_002.json").exists()
    assert not (root / "goal_001_002_PersistConfig").exists()
    revision = json.loads((root / "semantic_plan_round_002/attempt_01_request.json").read_text("utf8"))
    assert revision["context"]["original_caller_request"] == request.model_dump(mode="json")
    assert revision["context"]["completed_subgoals"] == [{
        key: result.completed_subgoals[0][key]
        for key in ("goal_id", "instruction", "expect_text", "forbid_text")}]
    assert revision["context"]["failure"]["rejected_goal"] == rejected["goal"]
    assert revision["context"]["failure"]["category"] == "semantic_condition_preexisting_before_input"
    assert revision["image_path"] is None and not revision["image_attached"]
    assert result.final_verification.passed and result.unconfirmed_action_attempts == 0


def test_preexisting_contract_still_blocks_when_round_budget_is_exhausted(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, responses=[initial_proposal(), plan()], rounds=1)
    result = runner.run(request, options())
    assert result.status == "blocked" and "already hold before input" in result.error
    assert state.attempts == 1 and len(client.calls) == 2
    assert len(result.completed_subgoals) == len(result.rejected_goal_contracts) == 1
    assert result.final_verification is None


def test_preexisting_recovery_does_not_reset_http_budget(tmp_path):
    result, _, state, client, _ = run_recovery(tmp_path, max_requests=2)
    assert result.status != "succeeded" and state.attempts == 1
    assert result.model_requests == len(client.calls) == 2
    assert len(result.completed_subgoals) == 1 and len(result.rejected_goal_contracts) == 1
    assert result.final_verification is None


def test_unchanged_rejected_contract_is_not_executed_under_a_new_goal_id(tmp_path):
    repeated = json.loads(initial_proposal())["subgoals"][1]
    repeated["goal_id"] = "RenamedStillPreexisting"
    runner, request, state, client, _ = lab(
        tmp_path, responses=[initial_proposal(), plan(), proposal(repeated)], rounds=2)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 1 and len(client.calls) == 3
    assert len(result.rejected_goal_contracts) == 1 and len(result.completed_subgoals) == 1
    assert "already hold" in result.error


def test_unexpected_caller_final_does_not_rescue_rejected_contract_or_trigger_more_input(tmp_path):
    runner, request, state, client, _ = lab(
        tmp_path, responses=[initial_proposal(), plan()], rounds=2,
        extra_text_at_stage={1: ["Unexpected final"]})
    request.expect_text = ["Unexpected final"]
    result = runner.run(request, options())
    assert result.status == "blocked" and state.attempts == 1 and len(client.calls) == 2
    assert len(result.rejected_goal_contracts) == 1 and result.final_verification is None


def test_unknown_input_is_never_converted_to_preexisting_contract_recovery(tmp_path):
    result, _, state, client, _ = run_recovery(tmp_path, unknown_at=1)
    assert result.status != "succeeded" and state.attempts == 1
    assert result.unconfirmed_action_attempts == 1 and len(client.calls) == 2
    assert not result.rejected_goal_contracts and not result.completed_subgoals


def test_rejection_feedback_excludes_observation_paths_and_other_diagnostics():
    goal = json.loads(initial_proposal())["subgoals"][1]
    logical = _logical_failure({"status": "rejected_before_input", "goal": goal,
                                "before_reference": "PrivatePathCanary", "pixels": "PixelCanary",
                                "reason": "RawReasonCanary"})
    assert logical["rejected_goal"] == goal
    assert all(value not in json.dumps(logical) for value in
               ["PrivatePathCanary", "PixelCanary", "RawReasonCanary"])


@pytest.fixture
def audit_reader():
    path = Path(__file__).resolve().parents[1] / "scripts/week6_audit_semantic_acceptance.py"
    spec = importlib.util.spec_from_file_location("preexisting_contract_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_rejection(tmp_path, audit_reader, mutation=None, child_artifact=False):
    result, request, _, _, _ = run_recovery(tmp_path / "run")
    root = tmp_path / "run/semantic"
    rejected = result.rejected_goal_contracts[0]
    if mutation:
        mutation(rejected)
        (root / "goal_rejection_001_002.json").write_text(json.dumps(rejected), encoding="utf8")
    if child_artifact:
        (root / "goal_001_002_PersistConfig").mkdir()
    goal = SemanticGoal.model_validate(json.loads(initial_proposal())["subgoals"][1])
    before = ObservationSnapshot.model_validate(json.loads(
        (root / "contract_before_001_002/observation.json").read_text("utf8")))
    return audit_reader.audit_pre_input_rejection(
        audit_reader.Evidence(tmp_path), root, 1, 2, goal, _goal_task(goal, request), before)


def test_saved_rejected_contract_is_independently_recomputed(tmp_path, audit_reader):
    result = audit_rejection(tmp_path, audit_reader)
    assert result["confirmed_inputs"] == 0 and result["eligible_for_semantic_credit"] is False


@pytest.mark.parametrize("key,value", [
    ("confirmed_inputs", 1), ("confirmed_inputs", False), ("eligible_for_semantic_credit", True),
    ("round", True), ("before_reference", "borrowed_observation.json"),
    ("status", "succeeded"),
])
def test_saved_rejection_tampering_cannot_earn_execution_or_completion(tmp_path, audit_reader, key, value):
    with pytest.raises(audit_reader.AuditError, match="rejected_pre_input_contract_changed"):
        audit_rejection(tmp_path, audit_reader, lambda row: row.update({key: value}))


def test_rejected_contract_cannot_have_a_child_execution_directory(tmp_path, audit_reader):
    with pytest.raises(audit_reader.AuditError, match="has_execution_artifacts"):
        audit_rejection(tmp_path, audit_reader, child_artifact=True)
