"""Whole caller-goal next-action execution uses real bounded runtime, with no selector."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_week6_automatic_decomposition import Clock, ScriptedClient, _frame

from gui_agent.planning.planner import TaskPlanner
from gui_agent.planning.workflow_decomposer import AutomaticWorkflowRunner, WorkflowRequest
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import ExecutionOptions, TaskSpec
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult


def plan(count=1):
    return json.dumps({"task_id": "local", "instruction": "Advance the original caller goal", "steps": [
        {"step_id": f"s{index}", "description": "Choose the currently requested operation",
         "action_type": "click", "target_text": "Advance control", "arguments": {}}
        for index in range(count)
    ]})


def lab(tmp_path, *, required=2, initial_done=False, child_fault=None, initial_fault=None,
        unknown=False, transient=False, model_cost=0, factory_delay=0, max_requests=12, count=1,
        failed_receipt_visible_goal=False):
    state = SimpleNamespace(stage=required if initial_done else 0, attempts=0, captures=0,
                            initial_captures=0, children=[], planners=[], childdir=None, last=None)
    clock = Clock()
    client = ScriptedClient([plan(count)] * 16, clock=clock, costs=[model_cost] * 16)
    request = WorkflowRequest(workflow_id="Rolling", instruction="Complete the requested local sequence",
                              expect_text=["Caller finished"], forbid_text=["ERROR"])
    base = datetime(2026, 10, 25, tzinfo=UTC)

    def frame(identifier, sequence, origin):
        texts = ("Advance control", "Caller finished" if state.stage >= required else f"Current stage {state.stage}")
        value = _frame(identifier, texts)
        value.captured_at = base + timedelta(seconds=sequence)
        value.image_path = f"/virtual/{origin}/{sequence}.png"
        return value

    def initial_observer(_directory):
        state.initial_captures += 1
        value = frame("obs-0001", state.initial_captures - 1, "root")
        if initial_fault:
            initial_fault(value, state)
        return value

    class Observer:
        def observe(self, **_kwargs):
            state.captures += 1
            value = frame(f"obs-{state.captures:04d}", state.captures + 1, "child")
            if child_fault:
                child_fault(value, state)
            state.last = value.model_copy(deep=True)
            return value

    class Executor:
        def execute(self, action, *, dry_run=False, screen=None):
            if dry_run:
                return ActionResult(action=action, success=True, dry_run=True)
            state.attempts += 1
            if unknown:
                state.stage = required  # even an apparently complete screen cannot rescue unknown dispatch
                raise RuntimeError("input receipt unavailable")
            if failed_receipt_visible_goal:
                state.stage = required
                return ActionResult(action=action, success=False, dry_run=False, error="confirmed failed input")
            if transient and state.attempts == 1:
                return ActionResult(action=action, success=False, dry_run=False, error="known transient failure")
            state.stage += 1
            return ActionResult(action=action, success=True, dry_run=False)

    def factory(_task, _index, directory, budgeted):
        clock.advance(factory_delay)
        state.childdir = directory
        planner = TaskPlanner(budgeted, max_steps=6, max_format_retries=0)
        # Deliberately omit absolute_deadline: the automatic adapter must clamp it.
        child = TaskRunner(observer=Observer(), planner=planner, executor=Executor(),
                           adapter=ActionAdapter(platform="win32"), verifier=Verifier(),
                           recorder=TaskRecorder(RunSession(session_id=directory.name, directory=directory)),
                           clock=clock, sleep=lambda _seconds: None)
        state.children.append(child)
        state.planners.append(planner)
        return child

    runner = AutomaticWorkflowRunner(tmp_path / "automatic", client, factory, initial_observer=initial_observer,
                                     clock=clock, max_model_requests=max_requests, protocol_mode="rolling_actions")
    return runner, request, state, client, clock


def options(**changes):
    return ExecutionOptions(**{"execute": True, "confirm": False, "max_actions": 8, "max_planning_attempts": 8,
                               "verification_timeout_seconds": 0, **changes})


def test_real_runtime_rolls_one_action_at_a_time_on_full_goal_without_selector_or_marker_claims(tmp_path):
    runner, request, state, client, _ = lab(tmp_path)
    result = runner.run(request, options())
    assert result.status == "succeeded" and result.completion_basis == "executed_caller_goal"
    assert state.attempts == result.total_actions == result.planning_calls == result.model_requests == 2
    assert result.decomposition.status == "not_requested" and result.decomposition.decomposition_calls == 0
    assert result.decomposition_protocol_mode == "rolling_actions" and result.completed_subgoals == result.progress_verifications == []
    assert result.final_verification.passed and result.accounting_complete
    assert state.planners[0].max_steps == 6 and "plan" not in state.planners[0].__dict__
    assert result.rolling_actions_audit.eligible_for_final_verification
    assert result.rolling_actions_audit.planning_accounting_complete
    assert result.rolling_actions_audit.confirmed_input_count == 2
    assert len(result.rolling_actions_audit.plans) == 2
    assert all(value.max_steps == value.returned_steps == 1 for value in result.rolling_actions_audit.plans)
    assert all(request.instruction in value.instruction for value in result.rolling_actions_audit.plans)
    assert all(value.context["observation_id"] and value.image_path and value.response["raw_response"]
               and value.parsed_plan["steps"] for value in result.rolling_actions_audit.plans)
    final = json.loads(Path(result.final_observation_reference).read_text(encoding="utf-8"))
    assert final["observation_id"] != result.rolling_actions_audit.post_observation_id
    assert final["window_id"] == "test-window" and final["foreground_stable"]
    assert len(client.calls) == 2
    for value in client.calls:
        response_format = value["kwargs"]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "gui_task_plan"
        assert response_format["json_schema"]["strict"] is False
        schema = response_format["json_schema"]["schema"]
        assert "steps" in schema["required"] and schema["properties"]["steps"]["maxItems"] == 1
        assert "subgoals" not in schema["properties"]
        assert not {"disappearing_text", "caller_instruction_anchor", "candidates",
                    "raw_checks", "derived_progress_checks"} & schema["properties"].keys()
    assert read_run_status(tmp_path / "automatic")["snapshot"]["task_status"] == "succeeded"


def test_already_satisfied_is_twice_read_without_any_input_or_model_call_or_action_completion(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, initial_done=True)
    result = runner.run(request, options())
    assert result.status == "succeeded" and result.already_satisfied
    assert result.completion_basis == "preexisting_caller_final" and result.final_verification.passed
    assert state.initial_captures == 2 and state.children == [] and client.calls == []
    assert result.total_actions == result.model_requests == result.planning_calls == 0
    assert result.rolling_actions_audit is None and result.completed_subgoals == []


@pytest.mark.parametrize("fault", ["error", "unstable", "window", "geometry", "title_only", "old_path", "old_time"])
def test_independent_last_frame_must_be_new_readable_and_in_original_window_scope(tmp_path, fault):
    def corrupt(value, state):
        if state.childdir is None or not (state.childdir / "rolling_actions_audit.json").exists():
            return
        if fault == "error":
            value.errors.append("capture error")
        elif fault == "unstable":
            value.foreground_stable = False
        elif fault == "window":
            value.window_id = "different"
        elif fault == "geometry":
            value.window_bounds = None
        elif fault == "title_only":
            value.elements = []
            value.window_title = "Caller finished"
        elif fault == "old_path":
            value.image_path = state.last.image_path
        else:
            value.captured_at = state.last.captured_at

    runner, request, state, _, _ = lab(tmp_path, required=1, child_fault=corrupt)
    result = runner.run(request, options())
    assert result.status != "succeeded" and result.total_actions == state.attempts == 1
    assert result.completion_basis == "" and state.planners[0].max_steps == 6


def test_unknown_input_never_replans_or_uses_apparently_successful_final_screen(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, unknown=True)
    result = runner.run(request, options())
    assert result.status != "succeeded" and result.unconfirmed_action_attempts == 1
    assert state.attempts == len(client.calls) == 1 and result.final_verification is None
    assert not result.rolling_actions_audit.eligible_for_final_verification


def test_known_transient_failure_can_recover_while_original_receipts_remain_in_audit(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, required=1, transient=True)
    result = runner.run(request, options())
    assert result.status == "succeeded" and result.total_actions == state.attempts == 2
    audit = result.rolling_actions_audit
    assert [step.action_result["success"] for step in audit.actions] == [False, True]
    assert audit.confirmed_input_count == 1 and audit.eligible_for_final_verification


def test_declined_confirmation_stops_without_input_or_final_rescue(tmp_path):
    runner, request, state, _, _ = lab(tmp_path)
    result = runner.run(request, options(confirm=True), confirm=lambda _plan: False,
                        high_risk_confirm=lambda _plan: True, countdown=lambda _seconds: None)
    assert result.status == "cancelled" and state.attempts == 0 and result.planning_calls == 1
    assert result.final_verification is None and state.planners[0].max_steps == 6


@pytest.mark.parametrize("budget", ["actions", "planning", "http"])
def test_shared_caps_stop_whole_goal_recovery_without_hidden_selector_spend(tmp_path, budget):
    runner, request, state, _, _ = lab(tmp_path, max_requests=1 if budget == "http" else 12)
    config = options()
    if budget == "actions":
        config.max_actions = 1
    elif budget == "planning":
        config.max_planning_attempts = 1
    result = runner.run(request, config)
    assert result.status != "succeeded" and state.attempts == result.total_actions == 1
    assert result.model_requests == 1 and result.planning_calls <= config.max_planning_attempts
    assert state.planners[0].max_steps == 6


def test_delayed_factory_cannot_observe_or_reassign_time_after_global_deadline(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, factory_delay=6)
    result = runner.run(request, options(task_timeout_seconds=5))
    assert result.status == "timed_out" and state.captures == state.attempts == len(client.calls) == 0
    assert state.children[0].absolute_deadline == 105 and state.planners[0].max_steps == 6


def test_two_step_native_plan_is_refused_before_input_and_limit_restores(tmp_path):
    runner, request, state, _, _ = lab(tmp_path, count=2)
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == 0
    assert state.planners[0].max_steps == 6 and len(result.rolling_actions_audit.plans) == 1
    assert not result.rolling_actions_audit.plans[0].ok


def test_workflow_summary_write_failure_after_input_retains_confirmed_count_lower_bounds(tmp_path, monkeypatch):
    from gui_agent.runtime.workflow import WorkflowRunner

    original = WorkflowRunner._write_summary

    def fail(self, result):
        if result.total_actions:
            raise PermissionError("injected workflow summary persistence failure")
        return original(self, result)

    monkeypatch.setattr(WorkflowRunner, "_write_summary", fail)
    runner, request, state, _, _ = lab(tmp_path, required=1)
    result = runner.run(request, options())
    assert result.status == "failed" and not result.accounting_complete
    assert result.total_actions == state.attempts == 1 and result.planning_calls == 1
    assert "lower bounds" in result.error and state.planners[0].max_steps == 6


@pytest.mark.parametrize("fault", ["same_capture", "old_time", "error", "empty"])
def test_initial_child_capture_cannot_reuse_old_root_pixels_or_unreadable_evidence(tmp_path, fault):
    def corrupt(value, state):
        if fault == "same_capture":
            value.image_path = "/virtual/root/0.png"
        elif fault == "old_time":
            value.captured_at = datetime(2026, 10, 25, tzinfo=UTC)
        elif fault == "error":
            value.errors.append("OCR unavailable")
        else:
            value.elements = []

    runner, request, state, client, _ = lab(tmp_path, child_fault=corrupt)
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == len(client.calls) == 0
    assert result.planning_calls == 0 and state.planners[0].max_steps == 6


def test_late_model_reply_cannot_extend_global_deadline_or_dispatch_an_input(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, model_cost=6)
    result = runner.run(request, options(task_timeout_seconds=5))
    assert result.status == "timed_out" and state.attempts == 0 and len(client.calls) == 1
    assert result.model_requests == 1 and state.planners[0].max_steps == 6


def wrong_model_prediction(client):
    for index, response in enumerate(client.responses):
        value = json.loads(response)
        value["steps"][0]["expected_result"] = "Unobserved confirmation evidence"
        client.responses[index] = json.dumps(value)


@pytest.mark.parametrize("required", [1, 2])
def test_raw_model_prediction_failure_never_overrides_fresh_unchanged_caller_goal(tmp_path, required):
    runner, request, state, client, _ = lab(tmp_path, required=required)
    wrong_model_prediction(client)
    result = runner.run(request, options())
    assert result.status == "succeeded" and result.total_actions == state.attempts == required
    assert result.model_requests == result.planning_calls == len(client.calls) == required
    audit = result.rolling_actions_audit
    assert audit.schema_version == "week6.rolling_actions.v2" and audit.gate_on_step_verification is False
    assert all(step.verification.outcome == "failed" for step in audit.actions)
    assert all(step.verification.evidence["expectation_tokens"] and not step.verification.evidence["expectation_hits"]
               for step in audit.actions)
    assert audit.caller_verification.passed and result.final_verification.passed
    assert result.workflow_result.final_observation_id != audit.post_observation_id
    compiled = json.loads((tmp_path / "automatic/rolling_compilation.json").read_text("utf-8"))
    assert compiled["gate_on_step_verification"] is False
    assert compiled["workflow"]["subgoals"][0]["gate_on_step_verification"] is False
    assert compiled["workflow"]["subgoals"][0]["expect_text"] == request.expect_text
    assert compiled["workflow"]["subgoals"][0]["forbid_text"] == request.forbid_text
    assert result.completed_subgoals == []


def test_confirmed_failed_input_is_not_rescued_by_already_visible_caller_goal(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, required=1, failed_receipt_visible_goal=True)
    wrong_model_prediction(client)
    config = options()
    config.max_planning_attempts = 1
    result = runner.run(request, config)
    assert result.status == "failed" and state.stage == 1 and state.attempts == result.total_actions == 1
    assert result.model_requests == result.planning_calls == 1
    assert result.final_verification is None and not result.rolling_actions_audit.eligible_for_final_verification
    assert result.rolling_actions_audit.actions[0].action_result["success"] is False


def test_manual_task_default_still_stops_on_model_step_check_failure(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, required=1)
    wrong_model_prediction(client)
    directory = tmp_path / "manual"
    directory.mkdir()
    child = runner.runner_factory(None, 0, directory, client)
    task = TaskSpec(case_id="MANUAL", instruction=request.instruction, expect_text=request.expect_text,
                    forbid_text=request.forbid_text)
    assert task.gate_on_step_verification
    config = options()
    config.max_planning_attempts = 1
    result = child.run(task, config)
    assert result.status == "failed" and result.steps[0].verification.outcome == "failed"
    assert state.attempts == len(client.calls) == 1
    assert Verifier().check_task(task, child.observer.observe()).passed
