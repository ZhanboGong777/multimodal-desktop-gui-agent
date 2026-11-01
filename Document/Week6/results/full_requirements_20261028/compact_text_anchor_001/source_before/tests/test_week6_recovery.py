"""Recovery feedback and records, exercised without a desktop or a model."""

from __future__ import annotations

import json

import pytest
from test_runtime_runner import (
    FakeExecutor,
    FakeObserver,
    FakePlanner,
    _frame,
    _ManualRuntimeClock,
    _plan,
    _runner,
)

from gui_agent.planning.planner import PlanResult
from gui_agent.planning.schemas import PlanStep
from gui_agent.runtime.recovery import build_recovery_feedback, failure_reason
from gui_agent.runtime.schemas import (
    ExecutionOptions,
    RecoveryRecord,
    StepRecord,
    TaskRunResult,
    TaskSpec,
    VerificationResult,
)


def _finish_plan():
    return _plan(PlanStep(step_id="finish", description="stop", action_type="finish"))


def _task():
    return TaskSpec(case_id="T", instruction="reach Done", expect_text=["Done"])


def _options(**changes):
    return ExecutionOptions(
        execute=True, confirm=False, max_planning_attempts=3,
        verification_timeout_seconds=0, **changes,
    )


def _feedback(steps, **changes):
    defaults = {
        "previous_pass": 1, "verification": None,
        "actions_used": 0, "max_actions": 20, "remaining_ms": 1000,
    }
    return build_recovery_feedback(steps, **(defaults | changes))


def _written_recovery(recorder):
    path = recorder.directory / "recovery.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


@pytest.mark.parametrize("changes,reason", [
    ({"error": "no element matches a target"}, "target_resolution"),
    ({"error": "visual grounding refused: stale pixels"}, "visual_grounding"),
    ({"error": "observation failed: unavailable"}, "pre_action_observation"),
    ({"error": "post-action observation failed: unavailable"}, "post_action_observation"),
    ({"action_result": {"success": False, "error": "executor failed"}}, "action_execution"),
    ({"verification": VerificationResult(outcome="failed")}, "step_verification"),
])
def test_failure_classification_includes_non_exception_failures(changes, reason):
    step = StepRecord(index=4, step_id="s4", description="step", action_type="click", **changes)
    assert failure_reason([step], None) == (reason, 4)


def test_finish_does_not_hide_the_failed_step_or_claim_task_success():
    failed = StepRecord(
        index=1, step_id="s1", description="click", action_type="click", error="missing",
    )
    finish = StepRecord(index=2, step_id="finish", description="stop", action_type="finish")
    assert failure_reason([failed, finish], None) == ("target_resolution", 1)
    assert failure_reason([finish], VerificationResult(outcome="inconclusive")) == (
        "task_verification", None,
    )


def test_feedback_cannot_copy_text_arguments_errors_or_verifier_evidence():
    secret = "private typed text plus an access token"
    step = StepRecord(
        index=1, step_id=secret, description=secret, action_type="type_text", error=secret,
        action_result={"success": False, "error": secret, "action": {"text": secret}},
        verification=VerificationResult(outcome="failed", detail=secret, evidence={"text": secret}),
    )
    feedback = _feedback([step])
    assert secret not in feedback.model_dump_json()
    assert feedback.reason == "action_execution"
    assert feedback.recent_steps[0].model_dump() == {
        "index": 1, "action_type": "type_text", "attempted": True,
        "succeeded": False, "verification": "failed",
    }


def test_feedback_is_bounded_and_does_not_supply_old_targets():
    steps = [
        StepRecord(
            index=i, step_id=f"s{i}", description="not included", action_type="click",
            observation_id=f"old-frame-{i}", action_result={"success": True},
        ) for i in range(1, 10)
    ]
    feedback = _feedback(steps, actions_used=9, max_actions=12)
    assert [step.index for step in feedback.recent_steps] == [5, 6, 7, 8, 9]
    assert feedback.omitted_steps == 4
    assert feedback.actions_remaining == 3
    assert "old-frame" not in feedback.model_dump_json()
    assert "not included" not in feedback.model_dump_json()


def test_each_replan_receives_only_the_latest_pass_failure(tmp_path):
    finish = _finish_plan()

    class Planner(FakePlanner):
        def plan(self, instruction, *, context=None, **kwargs):
            self.contexts.append(context)
            if len(self.contexts) == 1:
                return PlanResult(plan=_plan(PlanStep(
                    step_id="missing", description="click", action_type="click", target_text="Absent",
                )), attempts=1)
            return PlanResult(plan=finish, attempts=1)

    planner = Planner(finish)
    observer = FakeObserver([
        _frame(f"frame-{i}", ("Done",) if i == 5 else ("Start",)) for i in range(6)
    ])
    executor = FakeExecutor()
    runner, recorder = _runner(tmp_path, observer, planner, executor)
    result = runner.run(_task(), _options())

    assert result.status == "succeeded" and not executor.actions
    assert "recovery" not in planner.contexts[0]
    first, second = [context["recovery"] for context in planner.contexts[1:]]
    assert first["reason"] == "target_resolution" and first["failed_step_index"] == 1
    assert second["previous_pass"] == 2
    assert second["reason"] == "task_verification" and second["failed_step_index"] is None
    assert second["recent_steps"] == []
    assert [record.next_pass for record in result.recovery_records] == [2, 3]
    records = _written_recovery(recorder)
    assert records == [record.model_dump(mode="json") for record in result.recovery_records]
    written = json.loads((recorder.directory / "task_summary.json").read_text("utf-8"))
    assert written["recovery_records"] == records
    assert TaskRunResult.model_validate(result.model_dump(mode="json")) == result
    assert not (recorder.directory / "planning_0001.json").exists()


def test_executor_failure_and_remaining_actions_reach_the_next_plan(tmp_path):
    click = _plan(PlanStep(
        step_id="click", description="click", action_type="click", target_text="Start",
    ))

    class Executor(FakeExecutor):
        def execute(self, *args, **kwargs):
            self.fail_on = "click" if not self.actions else None
            return super().execute(*args, **kwargs)

    observer = FakeObserver([
        _frame(f"frame-{i}", ("Done",) if i >= 4 else ("Start",)) for i in range(6)
    ])
    planner = FakePlanner(click)
    executor = Executor()
    runner, _ = _runner(tmp_path, observer, planner, executor)
    result = runner.run(_task(), _options(max_actions=2))

    assert result.status == "succeeded" and result.action_count == 2
    feedback = planner.contexts[1]["recovery"]
    assert feedback["reason"] == "action_execution"
    assert feedback["actions_used"] == feedback["actions_remaining"] == 1
    assert feedback["recent_steps"][0]["succeeded"] is False


def test_advisory_step_check_does_not_replace_the_task_failure(tmp_path):
    first_plan = _plan(
        PlanStep(
            step_id="click", description="click", action_type="click", target_text="Start",
            expected_result="Missing",
        ),
        PlanStep(step_id="finish", description="stop", action_type="finish"),
    )

    class Planner(FakePlanner):
        def plan(self, instruction, *, context=None, **kwargs):
            self.contexts.append(context)
            return PlanResult(
                plan=first_plan if len(self.contexts) == 1 else _finish_plan(), attempts=1,
            )

    planner = Planner(first_plan)
    observer = FakeObserver([
        _frame(f"frame-{i}", ("Done",) if i == 5 else ("Start",)) for i in range(6)
    ])
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, observer, planner, executor)
    result = runner.run(
        _task().model_copy(update={"gate_on_step_verification": False}), _options(),
    )

    assert result.status == "succeeded" and len(executor.actions) == 1
    assert result.steps[0].verification.outcome == "failed"
    feedback = planner.contexts[1]["recovery"]
    assert feedback["reason"] == "task_verification"
    assert feedback["failed_step_index"] is None
    assert feedback["recent_steps"][0]["verification"] == "failed"


@pytest.mark.parametrize("phase", ["observation", "planning", "execution"])
def test_interruption_keeps_completed_recovery_records(tmp_path, phase):
    finish = _finish_plan()

    class Observer(FakeObserver):
        def observe(self, **kwargs):
            if phase == "observation" and self.calls == 2:
                raise KeyboardInterrupt
            return super().observe(**kwargs)

    class Planner(FakePlanner):
        def plan(self, instruction, *, context=None, **kwargs):
            self.contexts.append(context)
            if len(self.contexts) == 2:
                if phase == "planning":
                    raise KeyboardInterrupt
                return PlanResult(plan=_plan(PlanStep(
                    step_id="click", description="click", action_type="click", target_text="Start",
                )), attempts=1)
            return PlanResult(plan=finish, attempts=1)

    class Executor(FakeExecutor):
        def execute(self, *args, **kwargs):
            raise KeyboardInterrupt

    planner = Planner(finish)
    runner, recorder = _runner(
        tmp_path, Observer([_frame("start", ("Start",))]), planner, Executor(),
    )
    result = runner.run(_task(), _options())
    assert result.status == "cancelled"
    assert len(result.recovery_records) == 1
    assert result.recovery_records[0].outcome == (
        "plan_ready" if phase == "execution" else "cancelled"
    )
    assert result.planning_calls == (1 if phase == "observation" else 2)
    assert _written_recovery(recorder) == [
        record.model_dump(mode="json") for record in result.recovery_records
    ]
    summary = json.loads((recorder.directory / "task_summary.json").read_text("utf-8"))
    assert summary["recovery_records"] == _written_recovery(recorder)


@pytest.mark.parametrize("failure,outcome", [
    ("observation", "observation_failed"),
    ("planning", "planning_failed"),
    ("empty", "no_plan"),
    ("response_error", "no_plan"),
    ("plan_errors", "plan_errors"),
    ("observation_timeout", "timed_out"),
    ("planning_timeout", "timed_out"),
])
def test_every_failed_recovery_attempt_is_recorded_once(tmp_path, failure, outcome):
    clock = _ManualRuntimeClock()
    finish = _finish_plan()

    class Observer(FakeObserver):
        def observe(self, **kwargs):
            if self.calls == 2:
                if failure == "observation":
                    clock.value += 2
                    raise RuntimeError("not readable")
                if failure == "observation_timeout":
                    clock.value = 10
            return super().observe(**kwargs)

    class Planner(FakePlanner):
        def plan(self, instruction, *, context=None, **kwargs):
            self.contexts.append(context)
            if len(self.contexts) == 1:
                return PlanResult(plan=finish, attempts=1)
            if failure == "planning":
                raise RuntimeError("provider unavailable")
            if failure == "planning_timeout":
                clock.value = 10
            if failure == "empty":
                return PlanResult(plan=_plan(), attempts=1)
            if failure == "response_error":
                return PlanResult(error="no usable response", attempts=1)
            if failure == "plan_errors":
                return PlanResult(plan=finish.model_copy(update={"errors": ["cannot plan"]}))
            return PlanResult(plan=finish, attempts=1)

    planner = Planner(finish)
    executor = FakeExecutor()
    runner, recorder = _runner(
        tmp_path, Observer([_frame("start", ("Start",))]), planner, executor, clock=clock,
    )
    result = runner.run(_task(), _options(task_timeout_seconds=5))
    assert result.status == ("timed_out" if outcome == "timed_out" else "failed")
    assert not executor.actions
    assert len(result.recovery_records) == len(_written_recovery(recorder)) == 1
    assert result.recovery_records[0].outcome == outcome
    assert result.planning_calls == (1 if failure.startswith("observation") else 2)
    assert result.planning_attempts == result.planning_calls
    assert result.recovery_records[0].elapsed_ms == (10_000 if "timeout" in failure else (
        2_000 if failure == "observation" else 0
    ))


def test_old_result_json_and_new_runs_do_not_share_recovery_lists(tmp_path):
    payload = {"run_id": "old", "case_id": "T", "instruction": "x", "status": "failed"}
    first, second = TaskRunResult.model_validate(payload), TaskRunResult.model_validate(payload)
    record = RecoveryRecord(next_pass=2, outcome="plan_ready", feedback=_feedback([]), elapsed_ms=0)
    first.recovery_records.append(record)
    assert second.recovery_records == []

    runner, recorder = _runner(
        tmp_path, FakeObserver([_frame("done", ("Done",))]),
        FakePlanner(_finish_plan()), FakeExecutor(),
    )
    runner._recovery_records = [record]
    result = runner.run(_task(), _options())
    assert result.status == "succeeded" and result.recovery_records == []
    assert not (recorder.directory / "recovery.jsonl").exists()


@pytest.mark.parametrize("write_phase", ["before", "partial", "after"])
def test_interruption_while_writing_keeps_only_complete_logged_records(tmp_path, write_phase):
    planner = FakePlanner(_finish_plan())
    executor = FakeExecutor()
    runner, recorder = _runner(
        tmp_path, FakeObserver([_frame("start", ("Start",))]), planner, executor,
    )
    append = recorder.append_recovery
    calls = 0

    def interrupted_append(record):
        nonlocal calls
        calls += 1
        if calls == 1:
            append(record)
            return
        if write_phase == "partial":
            with (recorder.directory / "recovery.jsonl").open("ab") as handle:
                handle.write(record.model_dump_json().encode("utf-8")[:40])
        elif write_phase == "after":
            append(record)
        raise KeyboardInterrupt

    recorder.append_recovery = interrupted_append
    result = runner.run(_task(), _options())
    assert result.status == "cancelled" and not executor.actions
    assert result.planning_calls == 3
    assert len(result.recovery_records) == (2 if write_phase == "after" else 1)
    complete, invalid_lines = recorder.read_recovery()
    assert complete == result.recovery_records
    assert invalid_lines == ([2] if write_phase == "partial" else [])
    summary = json.loads((recorder.directory / "task_summary.json").read_text("utf-8"))
    assert summary["recovery_records"] == [record.model_dump(mode="json") for record in complete]
    if write_phase != "after":
        assert any("rebuilt from completed log" in note for note in result.notes)
    if write_phase == "partial":
        assert any("invalid records on lines 2" in note for note in result.notes)
        # The interrupted bytes remain available for diagnosis.
        assert (recorder.directory / "recovery.jsonl").read_bytes().splitlines()[1].startswith(
            b'{"next_pass":3',
        )


def test_damaged_recovery_lines_do_not_hide_complete_records(tmp_path):
    _, recorder = _runner(
        tmp_path, FakeObserver([_frame("start", ("Start",))]),
        FakePlanner(_finish_plan()), FakeExecutor(),
    )
    assert recorder.read_recovery() == ([], [])
    record = RecoveryRecord(next_pass=2, outcome="plan_ready", feedback=_feedback([]), elapsed_ms=0)
    raw = record.model_dump_json().encode("utf-8")
    path = recorder.directory / "recovery.jsonl"
    payload = raw + b'\n{"next_pass":\n{"outcome":"wrong"}\n' + raw + b'\n\xff'
    path.write_bytes(payload)
    complete, invalid_lines = recorder.read_recovery()
    assert complete == [record, record] and invalid_lines == [2, 3, 5]
    assert path.read_bytes() == payload


def test_cancellation_still_finishes_when_recovery_log_cannot_be_read(tmp_path):
    class Observer(FakeObserver):
        def observe(self, **kwargs):
            raise KeyboardInterrupt

    runner, recorder = _runner(
        tmp_path, Observer([_frame("start", ("Start",))]),
        FakePlanner(_finish_plan()), FakeExecutor(),
    )

    def unreadable():
        raise PermissionError("unavailable")

    recorder.read_recovery = unreadable
    result = runner.run(_task(), _options())
    assert result.status == "cancelled" and not result.recovery_records
    assert any("records are unconfirmed" in note for note in result.notes)
    assert (recorder.directory / "task_summary.json").exists()
