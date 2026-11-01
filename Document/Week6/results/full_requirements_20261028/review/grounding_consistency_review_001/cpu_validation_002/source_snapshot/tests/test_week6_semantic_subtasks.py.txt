"""Semantic objectives are model output; real runtime verifies each before moving on."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_week6_cli as legacy_cli
from test_week6_automatic_decomposition import Clock, ScriptedClient, _frame
from test_week6_rolling_actions import options, plan

from gui_agent.planning.planner import TaskPlanner
from gui_agent.planning.semantic_workflow import SemanticDecomposer, SemanticGoal
from gui_agent.planning.workflow_decomposer import AutomaticWorkflowRunner, WorkflowRequest
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult

cli = legacy_cli.cli


def proposal(*goals):
    return json.dumps({"status": "ready", "reason": "Two requested state changes", "subgoals": list(goals) or [
        {"goal_id": "SetPreference", "instruction": "Set the requested preference",
         "expect_text": ["Verified part 1"], "forbid_text": []},
        {"goal_id": "PersistConfig", "instruction": "Persist the requested configuration",
         "expect_text": ["Verified part 2"], "forbid_text": []},
    ]})


def lab(tmp_path, *, responses=None, unknown_at=0, fail_at=0, model_cost=0,
        factory_delay=0, max_requests=20, initial_fault=None, final_fault=None,
        rounds=1, format_retries=0, goal_calls=3, extra_text_at_stage=None):
    state = SimpleNamespace(stage=0, attempts=0, captures=0, children=[], tasks=[], root=None)
    clock = Clock()
    client = ScriptedClient(responses or [proposal(), *([plan()] * 20)], clock=clock, costs=[model_cost] * 24)
    request = WorkflowRequest(workflow_id="SemanticLab", instruction="Set the requested preference and persist the configuration",
                              expect_text=["Verified part 1", "Verified part 2", "Final accepted"], forbid_text=["ERROR"])

    def frame():
        state.captures += 1
        texts = ["Advance control", *[f"Verified part {i}" for i in range(1, state.stage + 1)]]
        texts.append("Final accepted" if state.stage >= 2 else "Pending work")
        for stage, extra in (extra_text_at_stage or {}).items():
            if state.stage >= stage:
                texts.extend(extra)
        value = _frame(f"obs-{state.captures:04d}", texts)
        value.captured_at = datetime(2026, 10, 28, tzinfo=UTC) + timedelta(seconds=state.captures)
        value.image_path = f"/virtual/semantic/{state.captures}.png"
        return value

    def initial(directory):
        value = frame()
        if directory.name == "initial_observation" and initial_fault:
            initial_fault(value)
        if directory.name.startswith("caller_final") and final_fault:
            final_fault(value)
        return value

    class Observer:
        def observe(self, **_kwargs):
            return frame()

    class Executor:
        def execute(self, action, *, dry_run=False, screen=None):
            if dry_run:
                return ActionResult(action=action, success=True, dry_run=True)
            state.attempts += 1
            if state.attempts == unknown_at:
                state.stage = 2
                raise RuntimeError("receipt unavailable after dispatch")
            if state.attempts == fail_at:
                return ActionResult(action=action, success=False, dry_run=False, error="known transient")
            state.stage += 1
            return ActionResult(action=action, success=True, dry_run=False)

    def factory(task, _index, directory, budgeted):
        clock.advance(factory_delay)
        child = TaskRunner(observer=Observer(), planner=TaskPlanner(budgeted, max_steps=7, max_format_retries=0),
                           executor=Executor(), adapter=ActionAdapter(platform="win32"), verifier=Verifier(),
                           recorder=TaskRecorder(RunSession(session_id=directory.name, directory=directory)),
                           clock=clock, sleep=lambda _seconds: None)
        state.children.append(child)
        state.tasks.append(task)
        return child

    runner = AutomaticWorkflowRunner(tmp_path / "semantic", client, factory, initial_observer=initial,
        protocol_mode="semantic_subtasks", clock=clock, max_model_requests=max_requests,
        max_format_retries=format_retries, max_decomposition_rounds=rounds)
    # Public direct runner exposes the per-goal cap; Automatic uses its safe default.
    if goal_calls != 3:
        from gui_agent.planning.semantic_workflow import SemanticWorkflowRunner
        runner = SemanticWorkflowRunner(tmp_path / "semantic", client, factory, initial_observer=initial,
            clock=clock, max_model_requests=max_requests, max_format_retries=format_retries,
            max_decomposition_rounds=rounds, max_goal_planning_calls=goal_calls)
    return runner, request, state, client, clock


def test_model_sequence_executes_distinct_semantic_objectives_and_independent_caller_final(tmp_path):
    runner, request, state, client, _ = lab(tmp_path)
    result = runner.run(request, options())
    assert result.status == "succeeded", result.error
    assert [x["goal_id"] for x in result.completed_subgoals] == ["SetPreference", "PersistConfig"]
    assert result.total_actions == state.attempts == 2
    assert result.planning_calls == result.model_requests == len(client.calls) == 3
    assert result.final_verification.passed and result.accounting_complete
    assert [x.expect_text for x in state.tasks] == [["Verified part 1"], ["Verified part 2"]]
    assert all(request.instruction in x.instruction for x in state.tasks)
    assert all(x.instruction != request.instruction and not x.gate_on_step_verification for x in state.tasks)
    assert all(c.planner.max_steps == 7 and "plan" not in c.planner.__dict__ for c in state.children)
    directory = tmp_path / "semantic"
    native = json.loads((directory / "semantic_plan_round_001/native_proposal.json").read_text("utf8"))
    assert native == json.loads(proposal())
    compiled = json.loads((directory / "semantic_plan_round_001/compiled_workflow.json").read_text("utf8"))
    assert compiled["workflow"]["final_check"]["expect_text"] == request.expect_text
    contracts = sorted(directory.glob("goal_contract_*.json"))
    assert len(contracts) == 2 and all(json.loads(p.read_text("utf8"))["postconditions_fixed_before_input"] for p in contracts)
    audits = [json.loads(x.read_text("utf8")) for x in directory.glob("goal_*/001_*/semantic_goal_audit.json")]
    assert len(audits) == 2 and all(x["eligible_for_semantic_credit"] for x in audits)
    assert read_run_status(directory)["snapshot"]["task_status"] == "succeeded"


@pytest.mark.parametrize("bad", [
    {"status": "ready", "reason": "", "subgoals": []},
    {"status": "ready", "reason": "", "subgoals": [json.loads(proposal())["subgoals"][0]]},
    {**json.loads(proposal()), "final_check": {"expect_text": ["invented"]}},
    {**json.loads(proposal()), "subgoals": [json.loads(proposal())["subgoals"][0]] * 2},
    {**json.loads(proposal()), "subgoals": [
        {"goal_id": "Already", "instruction": "Set the requested preference", "expect_text": ["Advance control"], "forbid_text": []},
        json.loads(proposal())["subgoals"][1]]},
    {**json.loads(proposal()), "subgoals": [
        {"goal_id": "Conflict", "instruction": "Set the requested preference", "expect_text": [], "forbid_text": ["Final accepted"]},
        json.loads(proposal())["subgoals"][1]]},
])
def test_bad_semantic_proposal_never_creates_executor_child_or_alters_final_rules(tmp_path, bad):
    runner, request, state, client, _ = lab(tmp_path, responses=[json.dumps(bad)])
    original = request.model_dump()
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.children == [] and state.attempts == 0
    assert request.model_dump() == original and len(client.calls) == 1
    assert (tmp_path / "semantic/semantic_plan_round_001/attempt_01_error.json").is_file()


@pytest.mark.parametrize("instruction", ["Click (22, 33)", "Click Continue", "Send a message", "Use obs-0001-e001"])
def test_coordinates_primitive_lists_messages_and_old_frame_references_are_not_semantic_goals(instruction):
    with pytest.raises(ValueError):
        SemanticGoal(goal_id="Bad", instruction=instruction, expect_text=["changed"])


def test_repeated_entire_caller_instruction_is_not_a_semantic_decomposition(tmp_path):
    runner, request, state, _, _ = lab(tmp_path)
    bad = json.loads(proposal())
    bad["subgoals"][0]["instruction"] = request.instruction
    runner.client.responses = [json.dumps(bad)]
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0
    assert "copy" in result.plans[0].error


def test_format_repairs_consume_both_global_planning_and_actual_request_budgets(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, responses=["broken", proposal(), plan(), plan()], format_retries=1)
    result = runner.run(request, options())
    assert result.status == "succeeded" and result.planning_calls == result.model_requests == len(client.calls) == 4
    assert result.plans[0].planning_calls == 2 and state.attempts == 2


@pytest.mark.parametrize("cap", ["actions", "planning", "http"])
def test_all_goals_and_decomposition_share_caps(tmp_path, cap):
    runner, request, state, _, _ = lab(tmp_path, max_requests=2 if cap == "http" else 20)
    opt = options(max_actions=1 if cap == "actions" else 8, max_planning_attempts=2 if cap == "planning" else 8)
    result = runner.run(request, opt)
    assert result.status != "succeeded" and state.attempts == result.total_actions == 1
    assert result.planning_calls <= opt.max_planning_attempts and result.model_requests <= runner.max_model_requests
    assert len(result.completed_subgoals) == 1


def test_unknown_second_goal_preserves_first_goal_but_never_replans_or_final_rescues(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, unknown_at=2, rounds=3)
    result = runner.run(request, options())
    assert result.status != "succeeded" and result.unconfirmed_action_attempts == 1
    assert len(result.completed_subgoals) == 1 and len(result.plans) == 1
    assert len(client.calls) == 3 and state.attempts == 2 and result.final_verification is None


def test_known_transient_recovers_same_goal_without_repeating_completed_goal(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, fail_at=2)
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 3
    assert len(state.tasks) == len(result.completed_subgoals) == 2
    assert result.planning_calls == 4


@pytest.mark.parametrize("fault", ["scope", "blank", "errors", "unstable"])
def test_fresh_independent_caller_final_must_be_readable_and_in_original_scope(tmp_path, fault):
    def mutate(frame):
        if fault == "scope":
            frame.window_id = "other"
        elif fault == "blank":
            frame.elements = []
        elif fault == "errors":
            frame.errors.append("OCR unavailable")
        else:
            frame.foreground_stable = False
    runner, request, state, _, _ = lab(tmp_path, final_fault=mutate)
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == 2
    assert len(result.completed_subgoals) == 2


def test_late_decomposition_or_delayed_factory_cannot_renew_task_time(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, model_cost=6)
    result = runner.run(request, options(task_timeout_seconds=5))
    assert result.status == "timed_out" and len(client.calls) == 1 and state.attempts == 0
    assert result.planning_calls == 1


def test_delayed_factory_keeps_original_absolute_deadline(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, factory_delay=6)
    result = runner.run(request, options(task_timeout_seconds=5))
    assert result.status == "timed_out" and state.attempts == 0
    assert state.children[0].absolute_deadline == 105


def test_cancelled_confirmation_cannot_be_replanned_or_counted_completed(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, rounds=3)
    result = runner.run(request, options(confirm=True), confirm=lambda _p: False)
    assert result.status == "cancelled" and not result.completed_subgoals and len(result.plans) == 1
    assert state.attempts == 0 and state.children[0].planner.max_steps == 7


def test_bounded_semantic_revision_keeps_failed_goal_original_and_does_not_replay_completed(tmp_path):
    original = json.loads(proposal())
    original["subgoals"][1]["expect_text"] = ["A genuine but unsupported future caption"]
    revision = proposal({"goal_id": "PersistRevised", "instruction": "Complete the remaining persistence confirmation",
                         "expect_text": ["Verified part 3"], "forbid_text": []})
    # No caller success can rescue the failed original goal; keep final pending
    # until a new input under a newly predeclared objective completes it.
    runner, request, state, client, _ = lab(tmp_path, responses=[json.dumps(original), plan(), plan(), revision, plan()],
                                           rounds=2, goal_calls=1,
                                           extra_text_at_stage={3: ["A genuine but unsupported future caption"]})
    request.expect_text = ["Verified part 1", "A genuine but unsupported future caption", "Verified part 3"]
    result = runner.run(request, options())
    assert result.status == "succeeded", result.error
    assert [x["status"] for x in result.goal_outcomes] == ["succeeded", "failed", "succeeded"]
    assert [x["goal_id"] for x in result.completed_subgoals] == ["SetPreference", "PersistRevised"]
    assert state.attempts == 3 and result.planning_calls == len(client.calls) == 5
    assert result.plans[1].proposal.subgoals[0].expect_text == ["Verified part 3"]


@pytest.mark.parametrize("kwargs", [{"max_subgoals": 1}, {"max_subgoals": 33}, {"max_format_retries": 3}])
def test_semantic_budget_configuration_is_bounded(kwargs):
    with pytest.raises(ValueError):
        SemanticDecomposer(**kwargs)


def test_raw_step_prediction_failure_is_preserved_while_fixed_semantic_postcondition_passes(tmp_path):
    bad_prediction = json.loads(plan())
    bad_prediction["steps"][0]["expected_result"] = "A model narrative absent from the actual UI"
    runner, request, state, _, _ = lab(tmp_path, responses=[proposal(), json.dumps(bad_prediction), plan()])
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 2
    audit = json.loads(next((tmp_path / "semantic").glob("goal_001_001_*/001_*/semantic_goal_audit.json")).read_text("utf8"))
    assert audit["steps"][0]["verification"]["outcome"] == "failed"
    assert audit["semantic_postcondition_verification"]["outcome"] == "passed"
    assert audit["eligible_for_semantic_credit"]


def test_two_step_plan_cannot_dispatch_and_max_steps_is_restored(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, responses=[proposal(), *([plan(count=2)] * 3)])
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == 0
    assert state.children[0].planner.max_steps == 7 and "plan" not in state.children[0].planner.__dict__


def test_unsupported_schema_provider_fails_closed_without_any_request(tmp_path):
    runner, request, state, client, _ = lab(tmp_path)
    client.supports_strict_json_schema = False
    result = runner.run(request, options())
    assert result.status == "failed" and client.calls == [] and state.children == []
    assert result.model_requests == result.planning_calls == 0
    assert "no fallback" in result.error


def test_shared_budget_client_cannot_be_bypassed_by_custom_factory(tmp_path):
    runner, request, state, client, _ = lab(tmp_path)
    original = runner.runner_factory

    def bypass(task, index, directory, budgeted):
        child = original(task, index, directory, budgeted)
        child.planner.client = client
        return child

    runner.runner_factory = bypass
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0 and len(client.calls) == 1
    error = json.loads(next((tmp_path / "semantic").glob("goal_*/001_*/semantic_child_setup_error.json")).read_text("utf8"))
    assert "BudgetedModelClient" in error["error"]


def test_summary_failure_after_real_runtime_input_retains_known_count_lower_bounds(tmp_path, monkeypatch):
    from gui_agent.runtime.workflow import WorkflowRunner

    original = WorkflowRunner._write_summary

    def fail(self, result):
        if result.total_actions:
            raise PermissionError("injected summary persistence failure")
        return original(self, result)

    monkeypatch.setattr(WorkflowRunner, "_write_summary", fail)
    runner, request, state, _, _ = lab(tmp_path)
    result = runner.run(request, options())
    assert result.status == "failed" and not result.accounting_complete
    assert result.total_actions == state.attempts == 1 and result.planning_calls == 2
    assert result.final_verification is None and result.notes


def test_outer_summary_error_does_not_throw_away_returned_known_actions(tmp_path, monkeypatch):
    import gui_agent.planning.semantic_workflow as module

    original = module._write_json

    def fail(path, value):
        if Path(path).name == "automatic_workflow_summary.json":
            raise PermissionError("injected outer summary error")
        return original(path, value)

    monkeypatch.setattr(module, "_write_json", fail)
    runner, request, state, _, _ = lab(tmp_path)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == result.total_actions == 2
    assert result.planning_calls == 3 and "persistence" in result.error


def test_completed_semantic_goal_cannot_reappear_in_revised_plan(tmp_path):
    original = json.loads(proposal())
    original["subgoals"][1]["expect_text"] = ["Unseen future caption"]
    replay = proposal(original["subgoals"][0])
    runner, request, state, client, _ = lab(tmp_path,
        responses=[json.dumps(original), plan(), plan(), replay], rounds=2, goal_calls=1)
    request.expect_text = ["Verified part 1", "Unseen future caption", "Verified part 3"]
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 2 and len(client.calls) == 4
    assert len(result.completed_subgoals) == 1
    assert "replayed" in result.plans[1].error


def test_reused_plan_with_renamed_id_and_reason_does_not_renew_replanning(tmp_path):
    original = json.loads(proposal())
    original["subgoals"][1]["expect_text"] = ["Unseen future caption"]
    # After a known failure the exact failed objective, with only narrative/ID
    # changed, still cannot become an unbounded semantic planning loop.
    runner, request, state, client, _ = lab(tmp_path,
        responses=[json.dumps(original), plan(), plan(), proposal({**original["subgoals"][1], "goal_id": "Renamed"})],
        rounds=2, goal_calls=1)
    request.expect_text = ["Verified part 1", "Unseen future caption", "Verified part 3"]
    result = runner.run(request, options())
    assert result.status == "blocked" and state.attempts == 2 and len(client.calls) == 4
    assert "unchanged" in result.error


def test_stale_or_title_only_initial_evidence_cannot_dispatch_a_model_request(tmp_path):
    def blank(frame):
        frame.elements = []
        frame.window_title = "Final accepted"
    runner, request, state, client, _ = lab(tmp_path, initial_fault=blank)
    result = runner.run(request, options())
    assert result.status == "failed" and client.calls == [] and state.children == []


def test_semantic_objective_can_require_multiple_fresh_primitive_actions_before_next_goal(tmp_path):
    native = json.loads(proposal())
    native["subgoals"][0]["expect_text"] = ["Verified part 2"]
    native["subgoals"][1]["expect_text"] = ["Verified part 3"]
    runner, request, state, _, _ = lab(tmp_path, responses=[json.dumps(native), plan(), plan(), plan()])
    request.expect_text = ["Verified part 2", "Verified part 3"]
    result = runner.run(request, options())
    assert result.status == "succeeded" and len(result.completed_subgoals) == 2
    assert [x.total_actions for x in result.workflow_results] == [2, 1]
    assert state.attempts == result.total_actions == 3 and result.planning_calls == 4
    assert all(child.planner.max_steps == 7 for child in state.children)


@pytest.mark.parametrize("extra,expected", [([], 0), (["--max-subgoals", "1"], 2), (["--planning-attempts", "1"], 2)])
def test_semantic_cli_validation_is_explicit_and_requires_no_runtime(cli, monkeypatch, tmp_path, extra, expected):
    legacy_cli.forbid_runtime(monkeypatch, cli)
    argv = ["--instruction", "Set preferences and save the local configuration", "--expect-text", "Done",
            "--automatic-protocol", "semantic_subtasks", "--validate-only", "--output-directory", str(tmp_path / "runs"), *extra]
    assert cli.main(argv) == expected
    assert not (tmp_path / "runs").exists()
