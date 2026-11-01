"""A repeated failed contract gets bounded generation repair, never another input."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal

from gui_agent.planning import semantic_workflow as semantic
from gui_agent.runtime.schemas import ObservationSnapshot


def initial():
    native = json.loads(proposal())
    native["subgoals"][0]["expect_text"] = ["Verified part 1", "Unseen checkpoint"]
    return native


def repeated(*, rename=False, casefold=False):
    goal = copy.deepcopy(initial()["subgoals"][0])
    if rename:
        goal["goal_id"] = "RenamedFailure"
    if casefold:
        goal["instruction"] = goal["instruction"].upper()
        goal["expect_text"] = list(reversed([value.upper() for value in goal["expect_text"]]))
    return proposal(goal)


def remaining():
    return proposal({"goal_id": "PersistRemaining", "instruction": "Persist the remaining configuration",
                     "expect_text": ["Verified part 2", "Final accepted"], "forbid_text": []})


def repair_run(tmp_path, *, repeat=None, **kwargs):
    runner, request, state, client, clock = lab(tmp_path,
        responses=[json.dumps(initial()), plan(), repeat or repeated(), remaining(), plan()],
        rounds=2, format_retries=1, goal_calls=1, **kwargs)
    before = request.model_dump(mode="json")
    result = runner.run(request, options())
    assert request.model_dump(mode="json") == before
    return result, request, state, client, clock


@pytest.mark.parametrize("rename,casefold", [(False, False), (True, False), (True, True)])
def test_native_repeated_failure_is_repaired_before_any_replay(tmp_path, rename, casefold):
    result, _, state, client, _ = repair_run(tmp_path,
        repeat=repeated(rename=rename, casefold=casefold))
    assert result.status == "succeeded", result.error
    assert result.total_actions == state.attempts == len(state.tasks) == 2
    assert result.planning_calls == result.model_requests == len(client.calls) == 5
    assert result.unconfirmed_action_attempts == 0 and result.accounting_complete
    assert [entry["status"] for entry in result.goal_outcomes] == ["failed", "succeeded"]
    assert [entry["goal_id"] for entry in result.completed_subgoals] == ["PersistRemaining"]
    assert result.goal_outcomes[0]["expect_text"] == initial()["subgoals"][0]["expect_text"]
    directory = tmp_path / "semantic/semantic_plan_round_002"
    error = json.loads((directory / "attempt_01_error.json").read_text("utf8"))
    assert "unchanged failed/rejected objective" in error["error"]
    assert json.loads((directory / "native_proposal.json").read_text("utf8")) == json.loads(remaining())
    assert json.loads((directory / "attempt_01_response.json").read_text("utf8"))["content"] == repeated(
        rename=rename, casefold=casefold)
    correction = json.loads((directory / "attempt_02_request.json").read_text("utf8"))
    assert correction["context"]["constraint_error"] == error["error"]
    assert correction["context"]["failure"]["failed_goal"] == initial()["subgoals"][0]
    assert correction["image_path"] is None and not correction["image_attached"]
    assert semantic.SEMANTIC_REPEATED_OBJECTIVE_REPAIR in correction["system"]
    assert semantic.SEMANTIC_REPEATED_OBJECTIVE_REPAIR not in json.loads(
        (directory / "attempt_01_request.json").read_text("utf8"))["system"]
    assert json.loads(correction["messages"][-1]["content"])["repair_instruction"] == (
        semantic.SEMANTIC_REPEATED_OBJECTIVE_REPAIR)


def test_exhausted_repair_cap_keeps_blocked_status_and_never_replays(tmp_path):
    runner, request, state, client, _ = lab(tmp_path,
        responses=[json.dumps(initial()), plan(), repeated(), repeated(), repeated()],
        rounds=2, format_retries=2, goal_calls=1)
    result = runner.run(request, options())
    assert result.status == "blocked" and "unchanged" in result.error
    assert state.attempts == result.total_actions == 1
    assert result.model_requests == len(client.calls) == 5
    assert not result.completed_subgoals and result.final_verification is None
    assert result.plans[-1].workflow is None and result.plans[-1].proposal is None


def test_zero_repair_cap_preserves_existing_block_without_extra_request(tmp_path):
    runner, request, state, client, _ = lab(tmp_path,
        responses=[json.dumps(initial()), plan(), repeated()], rounds=2, goal_calls=1)
    result = runner.run(request, options())
    assert result.status == "blocked" and "unchanged" in result.error
    assert state.attempts == 1 and result.model_requests == len(client.calls) == 3
    assert result.final_verification is None


def test_repair_cannot_renew_original_http_budget(tmp_path):
    result, _, state, client, _ = repair_run(tmp_path, max_requests=3)
    assert result.status != "succeeded" and result.final_verification is None
    assert state.attempts == 1 and result.model_requests == len(client.calls) == 3
    assert not result.completed_subgoals


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(7)])
def test_repair_cancellation_never_dispatches_the_rejected_objective(tmp_path, interrupt):
    runner, request, state, client, _ = lab(tmp_path,
        responses=[json.dumps(initial()), plan(), repeated(), interrupt],
        rounds=2, format_retries=1, goal_calls=1)
    if isinstance(interrupt, SystemExit):
        with pytest.raises(SystemExit):
            runner.run(request, options())
    else:
        result = runner.run(request, options())
        assert result.status == "cancelled" and result.final_verification is None
    assert state.attempts == 1 and len(client.calls) == 4


def test_unknown_dispatch_still_stops_before_any_semantic_repair(tmp_path):
    result, _, state, client, _ = repair_run(tmp_path, unknown_at=1)
    assert result.status != "succeeded" and not result.accounting_complete
    assert result.unconfirmed_action_attempts == 1
    assert state.attempts == 1 and len(client.calls) == 2
    assert len(result.plans) == 1 and result.final_verification is None


def test_repeated_hint_is_removed_after_an_ordinary_error(tmp_path):
    runner, request, state, _, _ = lab(tmp_path,
        responses=[json.dumps(initial()), plan(), repeated(), "not JSON", remaining(), plan()],
        rounds=2, format_retries=2, goal_calls=1)
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 2
    directory = tmp_path / "semantic/semantic_plan_round_002"
    systems = [json.loads((directory / f"attempt_{n:02d}_request.json").read_text("utf8"))["system"]
               for n in (1, 2, 3)]
    assert [semantic.SEMANTIC_REPEATED_OBJECTIVE_REPAIR in value for value in systems] == [False, True, False]


def test_ordinary_error_text_cannot_supply_the_typed_repeated_hint(tmp_path, monkeypatch):
    def ordinary(*_args, **_kwargs):
        raise ValueError("semantic replanning repeated an unchanged failed/rejected objective")
    monkeypatch.setattr(semantic.SemanticDecomposer, "_validate", ordinary)
    runner, request, state, _, _ = lab(tmp_path, responses=[proposal(), proposal()], format_retries=1)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0
    directory = tmp_path / "semantic/semantic_plan_round_001"
    assert semantic.SEMANTIC_REPEATED_OBJECTIVE_REPAIR not in json.loads(
        (directory / "attempt_02_request.json").read_text("utf8"))["system"]


def test_all_prior_failed_signatures_remain_blocked_across_rounds(tmp_path):
    other = proposal({"goal_id": "OtherUnfinished", "instruction": "Apply the remaining configuration",
                      "expect_text": ["Another unseen checkpoint"], "forbid_text": []})
    runner, request, state, client, _ = lab(tmp_path,
        responses=[json.dumps(initial()), plan(), other, plan(), repeated()],
        rounds=3, format_retries=0, goal_calls=1)
    request.expect_text.append("Unseen caller final")
    result = runner.run(request, options())
    assert result.status == "blocked" and "unchanged" in result.error
    assert state.attempts == result.total_actions == 2 and len(client.calls) == 5
    assert [entry["status"] for entry in result.goal_outcomes] == ["failed", "failed"]
    assert not result.completed_subgoals and result.final_verification is None


@pytest.fixture(scope="module")
def audit_reader():
    source = Path(__file__).resolve().parents[1] / "scripts/week6_audit_semantic_acceptance.py"
    spec = importlib.util.spec_from_file_location("failed_plan_repair_auditor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_repair_wire_requires_the_actual_prior_failed_contract(tmp_path, audit_reader):
    result, request, _, _, _ = repair_run(tmp_path / "run")
    assert result.status == "succeeded"
    directory = tmp_path / "run/semantic/semantic_plan_round_002"
    load = lambda name: json.loads((directory / name).read_text("utf8"))
    saved = load("attempt_02_request.json")
    protocol = load("protocol.json")
    before = ObservationSnapshot.model_validate(load("snapshot_archive.json"))
    base = load("attempt_01_request.json")["context"]
    goal = initial()["subgoals"][0]
    blocked = ((goal["instruction"].casefold(), tuple(sorted(x.casefold() for x in goal["expect_text"])), ()),)
    evidence = audit_reader.Evidence(tmp_path)
    args = (evidence, directory, 2, saved, protocol, request, before, [], 8, True, base)
    assert audit_reader.audit_semantic_repair_messages(*args, blocked) == saved["messages"]
    with pytest.raises(audit_reader.AuditError, match="after_locally_valid_reply"):
        audit_reader.audit_semantic_repair_messages(*args)


def test_saved_repair_does_not_trust_a_forged_error_label(tmp_path, audit_reader):
    result, request, _, _, _ = repair_run(tmp_path / "run")
    assert result.status == "succeeded"
    directory = tmp_path / "run/semantic/semantic_plan_round_002"
    load = lambda name: json.loads((directory / name).read_text("utf8"))
    saved, protocol = load("attempt_02_request.json"), load("protocol.json")
    before = ObservationSnapshot.model_validate(load("snapshot_archive.json"))
    base = load("attempt_01_request.json")["context"]
    goal = initial()["subgoals"][0]
    blocked = ((goal["instruction"].casefold(), tuple(sorted(x.casefold() for x in goal["expect_text"])), ()),)
    (directory / "attempt_01_error.json").write_text(json.dumps({"error": "forged rejection"}), "utf8")
    with pytest.raises(audit_reader.AuditError):
        audit_reader.audit_semantic_repair_messages(audit_reader.Evidence(tmp_path), directory, 2,
            saved, protocol, request, before, [], 8, True, base, blocked)
