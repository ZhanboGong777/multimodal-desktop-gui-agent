"""V2 integration: honest verdicts, live status and interrupted action evidence."""
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

from gui_agent.planning.schemas import PlanStep
from gui_agent.runtime.schemas import ExecutionOptions, StepRecord, TaskSpec
from gui_agent.runtime.verification import Verifier


def test_natural_language_success_sentence_is_not_an_executable_check(tmp_path):
    task = TaskSpec(case_id="V2", instruction="do work", success_rules=["all done"])
    observer = FakeObserver([_frame("o1", ("Start",))])
    planner = FakePlanner(_plan())
    executor = FakeExecutor()
    runner, recorder = _runner(tmp_path, observer, planner, executor)
    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))
    assert result.status == "blocked"
    assert not planner.contexts and not executor.actions
    assert not Verifier().check_task(task, _frame("o2", ("Done",))).passed
    status = json.loads((recorder.directory / "run_status.json").read_text("utf-8"))
    assert status["task_status"] == "blocked"


@pytest.mark.parametrize("kind", ["ocr_error", "empty_absence", "empty_needle"])
def test_unreadable_or_empty_evidence_cannot_complete_a_task(kind):
    task = TaskSpec(case_id="V2", instruction="close", forbid_text=["Unsaved"])
    frame = _frame("o1", ("Desktop",))
    if kind == "ocr_error":
        frame.errors = ["ocr failed"]
    elif kind == "empty_absence":
        frame.elements = []
    else:
        task.forbid_text = [""]
    result = Verifier().check_task(task, frame)
    assert result.outcome == "inconclusive"


def test_status_is_visible_before_dispatch_and_records_actual_final_verdict(tmp_path):
    observed = []
    runner_ref = []

    class Executor(FakeExecutor):
        def execute(self, *args, **kwargs):
            directory = runner_ref[0].recorder.directory
            observed.append(json.loads((directory / "run_status.json").read_text("utf-8")))
            return super().execute(*args, **kwargs)

    planner = FakePlanner(_plan(PlanStep(
        step_id="s1", description="click", action_type="click", target_text="Start",
    )))
    observer = FakeObserver([
        _frame("o1", ("Start",)), _frame("o2", ("Start",)),
        _frame("o3", ("Done",)),
    ])
    runner, recorder = _runner(tmp_path, observer, planner, Executor())
    runner_ref.append(runner)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="click Start", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False, verification_timeout_seconds=0),
    )
    assert result.status == "succeeded"
    assert observed and observed[0]["stage"] == "executing"
    final = json.loads((recorder.directory / "run_status.json").read_text("utf-8"))
    assert final["stage"] == "finished" and final["task_status"] == result.status
    assert final["actions_used"] == 1


@pytest.mark.parametrize("tail", [b'{"index":', b'\x00\xffbad', b'[]\n'])
def test_cancel_keeps_valid_steps_and_reports_damaged_tail(tmp_path, tail):
    class Observer(FakeObserver):
        def observe(self, **kwargs):
            raise KeyboardInterrupt

    runner, recorder = _runner(tmp_path, Observer([]), FakePlanner(_plan()), FakeExecutor())
    record = StepRecord(
        index=1, step_id="s1", description="completed before interruption", action_type="click",
        action_result={"success": True},
    )
    recorder.append_step(record)
    log = recorder.directory / "steps.jsonl"
    with log.open("ab") as handle:
        handle.write(tail)
    before = log.read_bytes()
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False),
    )
    assert result.status == "cancelled" and result.action_count == 1
    assert any("invalid records on lines 2" in note for note in result.notes)
    assert log.read_bytes() == before
    assert recorder.read_steps_with_errors()[1] == [2]


def test_monitor_write_failure_never_repeats_or_suppresses_dispatch(tmp_path, monkeypatch):
    executor = FakeExecutor()
    runner, _ = _runner(
        tmp_path, FakeObserver([_frame("o1", ("Start",)), _frame("o2", ("Start",)),
                                _frame("o3", ("Done",))]),
        FakePlanner(_plan(PlanStep(step_id="s1", description="click", action_type="click", target_text="Start"))),
        executor,
    )
    def broken(*args, **kwargs):
        raise OSError("status storage unavailable")
    monkeypatch.setattr(runner.monitor, "update", broken)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False, verification_timeout_seconds=0),
    )
    assert result.status == "succeeded" and len(executor.actions) == 1
    assert any("monitoring unavailable" in note for note in result.notes)


def test_window_class_alone_does_not_prove_required_text_disappeared():
    frame = _frame("o1", ())
    frame.window_class = "Chrome_WidgetWin_1"
    task = TaskSpec(case_id="V2", instruction="close", forbid_text=["Unsaved"])
    assert Verifier().check_task(task, frame).outcome == "inconclusive"


@pytest.mark.parametrize("interruption", [False, True])
def test_unknown_executor_outcome_is_recorded_and_never_retried(tmp_path, interruption):
    class Executor(FakeExecutor):
        def execute(self, action, **kwargs):
            self.actions.append(action)
            if interruption:
                raise KeyboardInterrupt
            raise RuntimeError("failure after possible input")
    executor = Executor()
    planner = FakePlanner(_plan(PlanStep(
        step_id="s1", description="click", action_type="click", target_text="Start",
    )))
    runner, recorder = _runner(tmp_path, FakeObserver([_frame("o1", ("Start",))]), planner, executor)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False, max_planning_attempts=4),
    )
    assert result.status == ("cancelled" if interruption else "failed")
    assert result.unconfirmed_action_attempts == 1 and result.action_count == 0
    assert len(executor.actions) == len(planner.contexts) == 1
    assert not result.recovery_records
    intent = json.loads((recorder.directory / "dispatch_intents.jsonl").read_text("utf-8"))
    assert intent["meaning"] == "intent_only_not_proof_of_dispatch_or_success"
    assert "text" not in intent and "coordinates" not in intent


def test_monitor_latency_cannot_dispatch_after_budget_expires(tmp_path, monkeypatch):
    clock = _ManualRuntimeClock()
    executor = FakeExecutor()
    runner, _ = _runner(
        tmp_path, FakeObserver([_frame("o1", ("Start",))]),
        FakePlanner(_plan(PlanStep(step_id="s1", description="click", action_type="click", target_text="Start"))),
        executor, clock=clock,
    )
    original = runner.monitor.update
    def delayed(stage, **kwargs):
        result = original(stage, **kwargs)
        if stage == "executing":
            clock.value = 20
        return result
    monkeypatch.setattr(runner.monitor, "update", delayed)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False, task_timeout_seconds=10),
    )
    assert result.status == "timed_out" and not executor.actions
    assert result.unconfirmed_action_attempts == 0


def test_initial_capture_cannot_start_model_after_task_budget(tmp_path):
    clock = _ManualRuntimeClock()
    class Observer(FakeObserver):
        def observe(self, **kwargs):
            clock.value += 11
            return super().observe(**kwargs)
    planner = FakePlanner(_plan())
    runner, _ = _runner(tmp_path, Observer([_frame("o1", ("Start",))]), planner, FakeExecutor(), clock=clock)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False, task_timeout_seconds=10),
    )
    assert result.status == "timed_out" and not planner.contexts
    assert result.planning_calls == 0


def test_monitor_io_cannot_dispatch_after_foreground_changes(tmp_path, monkeypatch):
    class Observer(FakeObserver):
        focused = True
        def foreground_matches(self, frame):
            return self.focused
    observer = Observer([_frame("o1", ("Start",))])
    executor = FakeExecutor()
    runner, _ = _runner(
        tmp_path, observer,
        FakePlanner(_plan(PlanStep(step_id="s1", description="click", action_type="click", target_text="Start"))),
        executor,
    )
    monkeypatch.setattr(runner, "_ground_step", lambda step, source, current, notes, **kwargs: (step, current, True))
    original = runner.monitor.update
    def switched(stage, **kwargs):
        value = original(stage, **kwargs)
        if stage == "executing":
            observer.focused = False
        return value
    monkeypatch.setattr(runner.monitor, "update", switched)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False),
    )
    assert result.status == "failed" and not executor.actions
    assert result.action_count == result.unconfirmed_action_attempts == 0
    assert "foreground changed before executor entry" in result.steps[-1].error


def test_task_verification_polling_stops_new_observations_at_global_budget(tmp_path):
    clock = _ManualRuntimeClock()
    class Observer(FakeObserver):
        def observe(self, **kwargs):
            assert clock.value <= 1.0
            return super().observe(**kwargs)
    observer = Observer([_frame("o1", ("Start",))])
    runner, _ = _runner(tmp_path, observer, FakePlanner(_plan()), FakeExecutor(), clock=clock)
    runner.sleep = lambda seconds: setattr(clock, "value", clock.value + seconds)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False, task_timeout_seconds=1.0,
                         verification_timeout_seconds=10.0),
    )
    assert clock.value == 1.0
    assert result.status == "timed_out" and not result.ok
    assert observer.calls <= 3


@pytest.mark.parametrize("slow_boundary", ["capture", "record"])
def test_final_capture_or_save_cannot_start_verifier_after_global_budget(
    tmp_path, monkeypatch, slow_boundary,
):
    clock = _ManualRuntimeClock()

    class Observer(FakeObserver):
        def observe(self, **kwargs):
            frame = super().observe(**kwargs)
            if self.calls >= 2 and slow_boundary == "capture":
                clock.value = 11
            return frame

    observer = Observer([_frame("o1", ("Start",)), _frame("o2", ("Done",))])
    runner, recorder = _runner(tmp_path, observer, FakePlanner(_plan()), FakeExecutor(), clock=clock)
    original_save = recorder.save_observation

    def delayed_save(frame):
        value = original_save(frame)
        if frame.observation_id == "o2" and slow_boundary == "record":
            clock.value = 11
        return value

    monkeypatch.setattr(recorder, "save_observation", delayed_save)
    verifier_calls = []
    original_check = runner.verifier.check_task

    def checked(task, frame):
        verifier_calls.append(clock.value)
        return original_check(task, frame)

    monkeypatch.setattr(runner.verifier, "check_task", checked)
    result = runner.run(
        TaskSpec(case_id="V2", instruction="task", expect_text=["Done"]),
        ExecutionOptions(execute=True, confirm=False, task_timeout_seconds=10,
                         verification_timeout_seconds=0),
    )
    assert result.status == "timed_out" and not result.ok
    assert observer.calls == 2 and verifier_calls == []
