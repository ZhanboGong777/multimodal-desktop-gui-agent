"""Live child telemetry is passive and never grants semantic completion credit."""

from __future__ import annotations

import json

import pytest
from test_week6_rolling_actions import options
from test_week6_semantic_subtasks import lab

from gui_agent.planning import semantic_workflow
from gui_agent.runtime import monitoring
from gui_agent.runtime.monitoring import RunMonitor, read_run_status


@pytest.fixture(autouse=True)
def process_identity(monkeypatch):
    monkeypatch.setattr(monitoring, "_process_info", lambda _pid: (True, "windows:1234"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_observer_sees_saved_records_and_cannot_mutate_status_or_consume_clock(tmp_path):
    clock_reads, observed = [], []

    def clock():
        clock_reads.append(1)
        return 50.0

    def observer(status):
        saved = load(tmp_path / "run_status.json")
        journal = [json.loads(line) for line in (tmp_path / "status_events.jsonl").read_text("utf-8").splitlines()]
        assert status == saved == journal[-1]
        observed.append(saved)
        status["actions_used"] = 999
        status["write_failures"].append("not_a_monitor_diagnostic")

    writer = RunMonitor(tmp_path, clock=clock, on_status=observer)
    writer.start("T06", True, 4, 5)
    writer.update("executing", actions_used=1, planning_pass=1)
    final = writer.finish("failed", actions_used=1)
    assert len(observed) == len(clock_reads) == 3
    assert [item["actions_used"] for item in observed] == [0, 1, 1]
    assert final == load(tmp_path / "run_status.json")
    assert final["write_failures"] == [] and set(final) == monitoring._FIELDS
    assert writer.observer_failures == ()
    assert read_run_status(tmp_path)["snapshot"]["task_status"] == "failed"


@pytest.mark.parametrize("exception", [ValueError, RuntimeError])
def test_observer_failure_is_fixed_private_safe_and_never_retries_or_aborts(tmp_path, caplog, exception):
    callbacks = []

    def broken(status):
        callbacks.append(status["seq"])
        raise exception("private raw callback content")

    writer = RunMonitor(tmp_path, clock=lambda: 50.0, on_status=broken)
    writer.start("T06", True, 4, 5)
    writer.update("executing", actions_used=1)
    writer.finish("cancelled", actions_used=1)
    assert callbacks == [1, 2, 3]
    assert writer.observer_failures == ("status_observer_failed",)
    assert writer.write_failures == ()
    assert "status_observer_failed" in caplog.text
    assert "private raw callback content" not in caplog.text
    final = read_run_status(tmp_path)
    assert final["state"] == "finished" and final["snapshot"]["task_status"] == "cancelled"


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_cancellation_and_process_exit_are_not_suppressed_as_observer_faults(tmp_path, exception):
    def cancelled(_status):
        raise exception()

    writer = RunMonitor(tmp_path, clock=lambda: 50.0, on_status=cancelled)
    with pytest.raises(exception):
        writer.start("T06", True, 4, 5)
    assert writer.observer_failures == ()
    assert load(tmp_path / "run_status.json")["event"] == "started"


def test_bad_observer_argument_is_rejected_without_creating_records(tmp_path):
    with pytest.raises(ValueError, match="status observer"):
        RunMonitor(tmp_path, on_status=12)
    assert not list(tmp_path.iterdir())


def test_persistence_failure_does_not_publish_an_unsaved_child_snapshot(tmp_path, monkeypatch):
    callbacks = []
    writer = RunMonitor(tmp_path, clock=lambda: 50.0, on_status=callbacks.append)

    def fail(_status):
        raise PermissionError("private path")

    monkeypatch.setattr(writer, "_replace_latest", fail)
    status = writer.start("T06", True, 4, 5)
    assert status["write_failures"] == ["latest_write_failed"]
    assert callbacks == [] and writer.observer_failures == ()


def telemetry_lab(tmp_path, *, unknown_at=0, broken_observer=False):
    runner, request, state, client, clock = lab(tmp_path, unknown_at=unknown_at)
    factory = runner.runner_factory
    witnessed = []

    def wrap(task, index, directory, budgeted):
        child = factory(task, index, directory, budgeted)

        def previous_observer(snapshot):
            root = load(tmp_path / "semantic/run_status.json")
            witnessed.append({
                "child": snapshot, "root": root,
                "summary_exists": (directory / "task_summary.json").exists(),
            })
            if broken_observer:
                raise RuntimeError("private prior observer exception")

        child.monitor.on_status = previous_observer
        return child

    runner.runner_factory = wrap
    return runner, request, state, client, clock, witnessed


def test_global_counts_advance_before_child_returns_without_granting_goal_credit(tmp_path):
    runner, request, state, client, _, witnessed = telemetry_lab(tmp_path)
    result = runner.run(request, options())
    assert result.status == "succeeded", result.error
    assert state.attempts == result.total_actions == 2
    assert result.planning_calls == result.model_requests == len(client.calls) == 3
    assert len(result.completed_subgoals) == 2
    live = [item for item in witnessed if item["child"]["actions_used"] == 1
            and item["child"]["event"] != "finished" and not item["summary_exists"]]
    assert live
    assert {item["root"]["actions_used"] for item in live} == {1, 2}
    assert {item["root"]["planning_pass"] for item in live} == {2, 3}
    for item in witnessed:
        root, child = item["root"], item["child"]
        assert root["stage"] == "executing" and root["task_status"] is None
        assert root["detail_code"] == f"child_{child['stage']}"
        assert root["subgoals_completed"] == (0 if root["subgoal_id"] == "SetPreference" else 1)
        assert root["max_actions"] == options().max_actions
        assert root["max_planning_attempts"] == options().max_planning_attempts
    root_history = [json.loads(line) for line in (tmp_path / "semantic/status_events.jsonl").read_text("utf-8").splitlines()]
    assert [item["actions_used"] for item in root_history] == sorted(item["actions_used"] for item in root_history)
    assert read_run_status(tmp_path / "semantic")["snapshot"]["subgoals_completed"] == 2


def test_unknown_attempt_stays_visible_without_count_regression_or_final_rescue(tmp_path):
    runner, request, state, client, _, witnessed = telemetry_lab(tmp_path, unknown_at=2)
    result = runner.run(request, options())
    assert result.status != "succeeded" and result.unconfirmed_action_attempts == 1
    assert result.total_actions == 1 and state.attempts == 2
    assert len(result.completed_subgoals) == 1 and result.final_verification is None
    assert len(client.calls) == 3
    root = read_run_status(tmp_path / "semantic")["snapshot"]
    assert root["actions_used"] == 2 and root["subgoals_completed"] == 1
    assert witnessed[-1]["root"]["subgoals_completed"] == 1
    assert "nonmonotonic" not in result.error


def test_existing_observer_failure_does_not_change_execution_or_model_call_counts(tmp_path, caplog):
    runner, request, state, client, _, witnessed = telemetry_lab(tmp_path, broken_observer=True)
    result = runner.run(request, options())
    assert result.status == "succeeded", result.error
    assert state.attempts == result.total_actions == 2 and len(client.calls) == 3
    assert witnessed and len(result.completed_subgoals) == 2
    assert all(child.monitor.observer_failures == ("status_observer_failed",) for child in state.children)
    assert "private prior observer exception" not in caplog.text


def test_parent_relay_failure_is_telemetry_only_and_does_not_repeat_actions(tmp_path, monkeypatch):
    original_update = semantic_workflow.RunMonitor.update

    def fail_relay(self, stage, **kwargs):
        if kwargs.get("detail_code", "").startswith("child_"):
            raise RuntimeError("private relay failure")
        return original_update(self, stage, **kwargs)

    monkeypatch.setattr(semantic_workflow.RunMonitor, "update", fail_relay)
    runner, request, state, client, _, _ = telemetry_lab(tmp_path)
    result = runner.run(request, options())
    assert result.status == "succeeded", result.error
    assert state.attempts == result.total_actions == 2 and len(client.calls) == 3
    assert len(result.completed_subgoals) == 2
    assert all(child.monitor.observer_failures == ("status_observer_failed",) for child in state.children)
    assert read_run_status(tmp_path / "semantic")["snapshot"]["actions_used"] == 2


def test_live_relay_does_not_consume_extra_execution_clock_reads(tmp_path, monkeypatch):
    baseline, request1, state1, client1, clock1 = lab(tmp_path / "baseline")
    observed, request2, state2, client2, clock2 = lab(tmp_path / "observed")
    reads = {id(clock1): 0, id(clock2): 0}

    def advancing_read(self):
        reads[id(self)] += 1
        self.seconds += 0.001
        return self.seconds

    monkeypatch.setattr(type(clock1), "__call__", advancing_read)
    original = RunMonitor.on_status
    # Compare the same production execution with and without child observers.
    # This test does not replace the model, verifier, budgets or clock policy.
    with monkeypatch.context() as context:
        context.setattr(RunMonitor, "on_status", property(
            original.fget, lambda self, _callback: original.fset(self, None)))
        result1 = baseline.run(request1, options())
    result2 = observed.run(request2, options())
    assert result1.status == result2.status == "succeeded"
    assert state1.attempts == state2.attempts == 2
    assert len(client1.calls) == len(client2.calls) == 3
    assert reads[id(clock1)] == reads[id(clock2)]
    assert result1.elapsed_ms == pytest.approx(result2.elapsed_ms)


def test_interrupt_during_child_observer_keeps_cancellation_and_does_not_rescue_final(tmp_path):
    runner, request, state, client, _ = lab(tmp_path)
    factory = runner.runner_factory
    interrupted = []

    def wrap(task, index, directory, budgeted):
        child = factory(task, index, directory, budgeted)

        def interrupt(status):
            if status["actions_used"] == 1 and not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt()

        child.monitor.on_status = interrupt
        return child

    runner.runner_factory = wrap
    result = runner.run(request, options())
    assert result.status == "cancelled", result.error
    assert state.attempts == 1 and len(client.calls) == 2
    assert result.completed_subgoals == [] and result.final_verification is None
    assert state.children[0].monitor.observer_failures == ()


def test_retained_child_relay_keeps_its_own_planning_offset(tmp_path):
    runner, request, state, _, _ = lab(tmp_path)
    factory = runner.runner_factory
    previous_child = []
    previous_planning_status = []
    checked = []

    def wrap(task, index, directory, budgeted):
        child = factory(task, index, directory, budgeted)

        def observer(status):
            if task.case_id == "SetPreference" and status["stage"] == "planning":
                previous_planning_status[:] = [status.copy()]
            if task.case_id == "PersistConfig" and status["event"] == "started":
                # Retaining an earlier relay must not give it this child's base.
                before = load(tmp_path / "semantic/run_status.json")["planning_pass"]
                previous_child[0].monitor.on_status(previous_planning_status[0])
                after = load(tmp_path / "semantic/run_status.json")["planning_pass"]
                checked.append((before, after))

        child.monitor.on_status = observer
        if task.case_id == "SetPreference":
            previous_child.append(child)
        return child

    runner.runner_factory = wrap
    result = runner.run(request, options())
    assert result.status == "succeeded", result.error
    assert checked == [(2, 2)] and state.attempts == 2
