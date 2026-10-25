"""Subgoal execution and independent final checks, with no desktop/model calls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_runtime_runner import FakeExecutor, FakeObserver, FakePlanner, _frame, _plan

from gui_agent.planning.schemas import PlanStep
from gui_agent.recording import RunSession
from gui_agent.runtime import workflow as workflow_module
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import (
    ExecutionOptions,
    StepRecord,
    TaskRunResult,
    TaskSpec,
    VerificationResult,
)
from gui_agent.runtime.verification import Verifier
from gui_agent.runtime.workflow import MAX_SUBGOALS, WorkflowRunner, WorkflowSpec


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _spec(count=3):
    return WorkflowSpec(
        workflow_id="W6_workflow", instruction="Complete the local staged demonstration",
        subgoals=[TaskSpec(case_id=f"G{i}", instruction=f"reach goal {i}", expect_text=[f"Done{i}"])
                  for i in range(count)],
        final_check=TaskSpec(case_id="FINAL", instruction="Read the final state", expect_text=["AllDone"]),
    )


def _options(**updates):
    values = {"execute": True, "confirm": False, "max_actions": 10,
              "max_planning_attempts": 8, "task_timeout_seconds": 20,
              "verification_timeout_seconds": 0}
    return ExecutionOptions(**(values | updates))


def _child_result(task, *, actions=1, calls=1, status="succeeded", passed=True, execute=True,
                  unconfirmed=0):
    return TaskRunResult(
        run_id=f"child_{task.case_id}", case_id=task.case_id, instruction=task.instruction,
        status=status, execute=execute, planning_calls=calls,
        unconfirmed_action_attempts=unconfirmed,
        steps=[StepRecord(index=i, step_id=f"s{i}", description="local action",
                          action_type="wait", action_result={"success": True})
               for i in range(actions)],
        verification=VerificationResult(outcome="passed" if passed else "failed"),
    )


class Factory:
    def __init__(self, clock, behaviors=None):
        self.clock = clock
        self.behaviors = behaviors or {}
        self.created = []
        self.calls = []
        self.observers = []
        self.runs = []

    def __call__(self, task, index, childdir):
        self.created.append((task, index, childdir))
        assert childdir.is_dir() and not list(childdir.iterdir())
        behavior = self.behaviors.get(index, {})
        self.clock.now += behavior.get("setup_seconds", 0)
        if "factory_exception" in behavior:
            raise behavior["factory_exception"]
        frame = _frame(f"final-{index}", ("AllDone",))
        frame.errors = behavior.get("final_errors", [])
        observer = FakeObserver([frame])
        original_observe = observer.observe

        def final_observe():
            self.clock.now += behavior.get("final_seconds", 0)
            if "final_exception" in behavior:
                raise behavior["final_exception"]
            return original_observe()

        observer.observe = final_observe
        self.observers.append(observer)
        recorder = TaskRecorder(RunSession(session_id=childdir.name, directory=childdir))
        verifier = Verifier()

        def final_verify(task, frame):
            self.clock.now += behavior.get("verify_seconds", 0)
            if "verify_exception" in behavior:
                raise behavior["verify_exception"]
            return verifier.check_task(task, frame)

        def run(child_task, options, **callbacks):
            self.calls.append((child_task, options, callbacks))
            self.clock.now += behavior.get("seconds", 0)
            if "exception" in behavior:
                if behavior.get("write_summary"):
                    recorder.write_summary(_child_result(child_task, actions=2, calls=3))
                if behavior.get("write_step"):
                    recorder.append_step(StepRecord(
                        index=0, step_id="written", description="attempt already happened",
                        action_type="wait", action_result={"success": True},
                    ))
                    with (childdir / "steps.jsonl").open("ab") as handle:
                        handle.write(b'{"partial":')
                raise behavior["exception"]
            args = {key: value for key, value in behavior.items()
                    if key in {"actions", "calls", "status", "passed", "execute", "unconfirmed"}}
            result = _child_result(child_task, **args)
            self.runs.append(result)
            recorder.write_summary(result)
            return result

        return SimpleNamespace(run=run, observer=observer,
                               verifier=SimpleNamespace(check_task=final_verify), recorder=recorder)


def _run(tmp_path, behaviors=None, options=None, spec=None, **callbacks):
    clock = Clock()
    factory = Factory(clock, behaviors)
    directory = tmp_path / "fresh_workflow"
    runner = WorkflowRunner(directory, factory, clock)
    result = runner.run(spec or _spec(), options or _options(), **callbacks)
    saved = json.loads((directory / "workflow_summary.json").read_text("utf-8"))
    assert saved == result.model_dump(mode="json")
    return result, factory, directory


def test_subgoals_share_actions_planning_calls_and_monotonic_time(tmp_path):
    result, factory, directory = _run(tmp_path, {
        0: {"actions": 2, "calls": 2, "seconds": 3},
        1: {"actions": 1, "calls": 1, "seconds": 4},
        2: {"actions": 2, "calls": 2, "seconds": 2},
    })
    assert result.status == "succeeded" and result.total_actions == 5
    assert result.planning_calls == 5 and result.elapsed_ms == 9000
    assert [call[1].max_actions for call in factory.calls] == [10, 8, 7]
    assert [call[1].max_planning_attempts for call in factory.calls] == [8, 6, 5]
    assert [call[1].task_timeout_seconds for call in factory.calls] == [20, 17, 13]
    assert [goal.status for goal in result.subgoals] == ["completed"] * 3
    assert len({entry[2] for entry in factory.created}) == 3
    assert result.final_observation_id == "final-2"
    assert factory.observers[-1].calls == 1
    assert Path(result.final_observation_reference).is_file()
    events = [json.loads(line) for line in (directory / "checkpoints.jsonl").read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["completed"] == 3 and events[-1]["status"] == "succeeded"


def test_enclosing_absolute_deadline_cannot_be_renewed_by_child_setup(tmp_path):
    clock = Clock()
    factory = Factory(clock, {0: {"setup_seconds": 4}})
    result = WorkflowRunner(tmp_path / "absolute", factory, clock).run(
        _spec(1), _options(task_timeout_seconds=20), absolute_deadline=103,
    )
    assert result.status == "timed_out"
    assert not factory.calls  # The enclosing budget expired while constructing the child.


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_absolute_deadline_refused_before_evidence_io(tmp_path, deadline):
    directory = tmp_path / "invalid_absolute"
    with pytest.raises(ValueError, match="absolute deadline"):
        WorkflowRunner(directory, Factory(Clock())).run(_spec(1), _options(), absolute_deadline=deadline)
    assert not directory.exists()


@pytest.mark.parametrize("changes,reason", [
    ({"max_actions": 1}, "action"),
    ({"max_planning_attempts": 1}, "planning"),
])
def test_exhausted_global_budgets_do_not_create_or_run_later_goals(tmp_path, changes, reason):
    result, factory, _ = _run(tmp_path, options=_options(**changes))
    assert result.status == "blocked" and reason in result.error
    assert len(factory.created) == len(factory.calls) == 1
    assert result.subgoals[0].status == "completed"
    assert [goal.status for goal in result.subgoals[1:]] == ["not_started", "not_started"]
    assert not any(observer.calls for observer in factory.observers)


def test_setup_time_is_charged_and_expired_setup_dispatches_nothing(tmp_path):
    result, factory, _ = _run(tmp_path, {0: {"setup_seconds": 21}})
    assert result.status == "timed_out" and factory.calls == []
    assert result.total_actions == result.planning_calls == 0


def test_child_setup_reduces_its_actual_time_budget(tmp_path):
    _, factory, _ = _run(tmp_path, {0: {"setup_seconds": 3, "seconds": 2}})
    assert factory.calls[0][1].task_timeout_seconds == 17
    assert factory.calls[1][1].task_timeout_seconds == 15


@pytest.mark.parametrize("status", ["failed", "blocked", "cancelled", "timed_out"])
def test_child_terminal_failure_stops_following_goals_and_keeps_completed_prefix(tmp_path, status):
    result, factory, _ = _run(tmp_path, {1: {"status": status, "passed": False}})
    assert result.status == status
    assert [goal.status for goal in result.subgoals] == ["completed", status, "not_started"]
    assert len(factory.calls) == 2
    assert result.total_actions == result.planning_calls == 2


@pytest.mark.parametrize("updates", [{"passed": False}, {"execute": False}])
def test_child_success_without_executed_verified_evidence_is_not_completed(tmp_path, updates):
    result, factory, _ = _run(tmp_path, {0: updates})
    assert result.status == "failed"
    assert result.subgoals[0].status == "failed" and len(factory.calls) == 1


def test_dry_run_checks_all_plans_without_completing_subgoals_or_final_verification(tmp_path):
    result, factory, _ = _run(
        tmp_path, {i: {"status": "dry_run_completed", "execute": False} for i in range(3)},
        options=_options(execute=False),
    )
    assert result.status == "dry_run_completed" and not result.ok
    assert len(factory.calls) == 3
    assert all(goal.status == "dry_run_completed" for goal in result.subgoals)
    assert result.final_verification is None and result.final_observation_id is None
    assert all(observer.calls == 0 for observer in factory.observers)


def test_a_child_exceeding_budget_cannot_produce_workflow_success(tmp_path):
    result, factory, _ = _run(tmp_path, {0: {"actions": 11}})
    assert result.status == "failed" and "exceeded" in result.error
    assert len(factory.calls) == 1 and result.total_actions == 11


@pytest.mark.parametrize("child_status,workflow_status", [("succeeded", "failed"),
                                                         ("failed", "failed"),
                                                         ("cancelled", "cancelled")])
def test_unconfirmed_dispatch_stops_workflow_and_retains_confirmed_lower_bounds(
    tmp_path, child_status, workflow_status,
):
    result, factory, _ = _run(tmp_path, {1: {"unconfirmed": 1, "status": child_status}})
    assert result.status == workflow_status and not result.accounting_complete
    assert result.total_actions == 2 and result.unconfirmed_action_attempts == 1
    assert result.subgoals[0].status == "completed"
    assert result.subgoals[1].unconfirmed_action_attempts == 1
    assert result.subgoals[2].status == "not_started" and len(factory.calls) == 2
    assert "confirmed lower bound" in " ".join(result.notes)


def test_completion_after_global_deadline_is_not_credited(tmp_path):
    result, factory, _ = _run(tmp_path, {0: {"seconds": 21}})
    assert result.status == "timed_out" and len(factory.calls) == 1
    assert result.subgoals[0].status == "timed_out"


def test_all_subgoals_passing_does_not_replace_overall_fresh_check(tmp_path):
    spec = _spec()
    spec.final_check.expect_text = ["MissingFinalMarker"]
    result, factory, _ = _run(tmp_path, spec=spec)
    assert result.status == "failed" and result.final_verification.outcome == "failed"
    assert all(goal.status == "completed" for goal in result.subgoals)
    assert factory.observers[-1].calls == 1


def test_perception_errors_cannot_credit_a_final_marker(tmp_path):
    result, _, _ = _run(tmp_path, {2: {"final_errors": ["OCR failed"]}})
    assert result.status == "failed" and result.final_verification.outcome == "inconclusive"
    assert all(goal.status == "completed" for goal in result.subgoals)


@pytest.mark.parametrize("behavior", [{"final_seconds": 21}, {"verify_seconds": 21}])
def test_final_readonly_check_shares_original_deadline(tmp_path, behavior):
    result, factory, _ = _run(tmp_path, {2: behavior})
    assert result.status == "timed_out" and len(factory.calls) == 3
    assert all(goal.status == "completed" for goal in result.subgoals)
    assert factory.observers[-1].calls == 1


@pytest.mark.parametrize("behavior,status", [
    ({"final_exception": RuntimeError("observe")}, "failed"),
    ({"verify_exception": RuntimeError("verify")}, "failed"),
    ({"final_exception": KeyboardInterrupt()}, "cancelled"),
])
def test_final_check_failure_or_cancellation_keeps_all_completed_subgoals(tmp_path, behavior, status):
    result, factory, _ = _run(tmp_path, {2: behavior})
    assert result.status == status and len(factory.calls) == 3
    assert all(goal.status == "completed" for goal in result.subgoals)


@pytest.mark.parametrize("exception,status", [(KeyboardInterrupt(), "cancelled"),
                                             (RuntimeError("PRIVATE_TYPED_VALUE"), "failed")])
def test_interrupted_child_retains_prefix_and_confirmed_counts_without_replay(tmp_path, exception, status):
    result, factory, directory = _run(tmp_path, {1: {"exception": exception, "write_step": True}})
    assert result.status == status and not result.accounting_complete
    assert [goal.status for goal in result.subgoals] == ["completed", status, "not_started"]
    assert result.total_actions == 2 and result.planning_calls == 1
    assert len(factory.calls) == 2
    assert "PRIVATE_TYPED_VALUE" not in (directory / "workflow_summary.json").read_text()
    raw_steps = (directory / "002_G1" / "steps.jsonl").read_bytes()
    assert raw_steps.endswith(b'{"partial":')


@pytest.mark.parametrize("exception,status", [(KeyboardInterrupt(), "cancelled"),
                                             (ValueError("setup failed"), "failed")])
def test_factory_cancellation_or_failure_keeps_finished_subgoals(tmp_path, exception, status):
    result, factory, _ = _run(tmp_path, {1: {"factory_exception": exception}})
    assert result.status == status and result.accounting_complete
    assert result.subgoals[0].status == "completed"
    assert len(factory.calls) == 1


def test_interruption_recovers_counts_from_actual_recorder_summary_without_claiming_completion(tmp_path):
    result, _, _ = _run(tmp_path, {1: {"exception": KeyboardInterrupt(), "write_summary": True}})
    assert result.status == "cancelled" and not result.accounting_complete
    assert result.total_actions == 3 and result.planning_calls == 4
    assert result.subgoals[1].status == "cancelled"


def test_checkpoint_callback_cancel_does_not_erase_completed_goal(tmp_path):
    events = []

    def callback(event):
        events.append(event)
        if event["index"] == 0 and event["status"] == "completed":
            raise KeyboardInterrupt

    result, factory, _ = _run(tmp_path, on_checkpoint=callback)
    assert result.status == "cancelled" and result.subgoals[0].status == "completed"
    assert len(factory.calls) == 1
    assert all(set(event) == {"sequence", "case_id", "index", "status", "actions",
                              "planning_calls", "completed", "total", "elapsed_ms"}
               for event in events)


def test_partial_checkpoint_append_preserves_old_bytes_and_final_complete_summary(tmp_path, monkeypatch):
    original = WorkflowRunner._checkpoint
    injected = []

    def partial(self, result, index, callback):
        if index == 0 and result.subgoals[0].status == "completed" and not injected:
            injected.append(True)
            with (self.directory / "checkpoints.jsonl").open("ab") as handle:
                handle.write(b'{"interrupted":')
            raise KeyboardInterrupt
        return original(self, result, index, callback)

    monkeypatch.setattr(WorkflowRunner, "_checkpoint", partial)
    result, factory, directory = _run(tmp_path)
    assert result.status == "cancelled" and result.subgoals[0].status == "completed"
    assert len(factory.calls) == 1
    raw = (directory / "checkpoints.jsonl").read_bytes()
    assert b'{"interrupted":\n' in raw
    terminal = json.loads(raw.splitlines()[-1])
    assert terminal["status"] == "cancelled" and terminal["completed"] == 1


@pytest.mark.parametrize("old", [b'{"status":"succeeded"}\n', b'{"partial":\x00'])
def test_existing_or_damaged_historical_directory_is_never_resumed_or_rewritten(tmp_path, old):
    directory = tmp_path / "old"
    directory.mkdir()
    evidence = directory / "checkpoints.jsonl"
    evidence.write_bytes(old)
    factory = Factory(Clock())
    with pytest.raises(FileExistsError):
        WorkflowRunner(directory, factory).run(_spec(), _options())
    assert factory.created == [] and evidence.read_bytes() == old


def test_second_call_on_same_workflow_is_refused_and_never_replays_completed_inputs(tmp_path):
    result, factory, directory = _run(tmp_path)
    before = (directory / "workflow_summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        WorkflowRunner(directory, factory).run(_spec(), _options())
    assert result.ok and len(factory.calls) == 3
    assert (directory / "workflow_summary.json").read_bytes() == before


def test_transient_summary_replace_denial_retries_same_file_without_replaying_children_or_journal(
    tmp_path, monkeypatch,
):
    original = workflow_module.os.replace
    calls = []
    sleeps = []

    def replace(source, target):
        calls.append((Path(source), Path(source).read_bytes()))
        if len(calls) <= 3:
            raise PermissionError("temporary Windows reader lock")
        return original(source, target)

    monkeypatch.setattr(workflow_module.os, "replace", replace)
    monkeypatch.setattr(workflow_module.time, "sleep", sleeps.append)
    result, factory, directory = _run(tmp_path)
    assert result.status == "succeeded" and len(factory.calls) == 3
    assert sleeps == [0.005, 0.01, 0.02]
    assert len({source for source, _ in calls[:4]}) == 1
    assert len({payload for _, payload in calls[:4]}) == 1
    events = [json.loads(line) for line in (directory / "checkpoints.jsonl").read_text().splitlines()]
    assert len(events) == 8
    assert [event["sequence"] for event in events] == list(range(1, 9))
    assert [event["case_id"] for event in events if event["status"] == "completed"] == ["G0", "G1", "G2"]
    assert not list(directory.glob(".workflow_summary.*.tmp"))


@pytest.mark.parametrize("error,expected_calls,expected_sleeps", [
    (PermissionError("persistent Windows lock"), 4, [0.005, 0.01, 0.02]),
    (OSError("other filesystem failure"), 1, []),
])
def test_persistent_summary_failure_preserves_previous_summary_and_never_reappends_or_executes(
    tmp_path, monkeypatch, error, expected_calls, expected_sleeps,
):
    result, factory, directory = _run(tmp_path)
    previous_summary = (directory / "workflow_summary.json").read_bytes()
    previous_journal = (directory / "checkpoints.jsonl").read_bytes()
    calls = []
    sleeps = []

    def unavailable(source, target):
        calls.append((Path(source), Path(source).read_bytes()))
        assert Path(target).read_bytes() == previous_summary
        raise error

    monkeypatch.setattr(workflow_module.os, "replace", unavailable)
    monkeypatch.setattr(workflow_module.time, "sleep", sleeps.append)
    updated = result.model_copy(update={"status": "failed", "error": "a new terminal result"})
    with pytest.raises(type(error), match=str(error)):
        WorkflowRunner(directory, factory)._write_summary(updated)
    assert len(calls) == expected_calls and sleeps == expected_sleeps
    assert len({source for source, _ in calls}) == 1
    assert len({payload for _, payload in calls}) == 1
    assert json.loads(calls[0][1])["status"] == "failed"
    assert (directory / "workflow_summary.json").read_bytes() == previous_summary
    assert (directory / "checkpoints.jsonl").read_bytes() == previous_journal
    assert len(factory.calls) == 3 and not list(directory.glob(".workflow_summary.*.tmp"))


@pytest.mark.parametrize("bad_id", ["../x", "x/y", "x\\y", ".", "CON", "nul", "C:x", "a b"])
def test_case_ids_cannot_escape_or_alias_directories(bad_id):
    payload = _spec().model_dump()
    payload["subgoals"][0]["case_id"] = bad_id
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate(payload)


@pytest.mark.parametrize("change", ["duplicate", "case_duplicate", "empty", "too_many",
                                    "unverified", "empty_text_check", "message", "empty_message",
                                    "process_final"])
def test_invalid_workflow_checks_are_rejected_before_any_directory_creation(change, tmp_path):
    payload = _spec().model_dump()
    if change == "duplicate":
        payload["subgoals"][1]["case_id"] = "G0"
    elif change == "case_duplicate":
        payload["subgoals"][1]["case_id"] = "g0"
    elif change == "empty":
        payload["subgoals"] = []
    elif change == "too_many":
        payload["subgoals"] = [payload["subgoals"][0]] * (MAX_SUBGOALS + 1)
    elif change == "unverified":
        payload["subgoals"][0].update(expect_text=[], success_rules=["the model says finish"])
    elif change == "empty_text_check":
        payload["subgoals"][0]["expect_text"] = [""]
    elif change == "message":
        payload["subgoals"][0]["message_conversation"] = "A conversation"
    elif change == "empty_message":
        payload["subgoals"][0]["message_conversation"] = ""
    else:
        payload["final_check"]["must_exit_processes"] = ["editor"]
    with pytest.raises(ValidationError):
        WorkflowSpec.model_validate(payload)
    assert list(tmp_path.iterdir()) == []


def test_callbacks_reach_children_unchanged_and_do_not_supply_new_authorization(tmp_path):
    callbacks = {name: (lambda *_: True) for name in (
        "confirm", "countdown", "high_risk_confirm", "on_plan",
    )}
    _, factory, _ = _run(tmp_path, **callbacks)
    assert all(call[2] == callbacks for call in factory.calls)


def test_mutated_spec_is_revalidated_before_directory_creation(tmp_path):
    spec = _spec()
    spec.subgoals[0].case_id = "../outside"
    factory = Factory(Clock())
    with pytest.raises(ValidationError):
        WorkflowRunner(tmp_path / "not_created", factory).run(spec, _options())
    assert not (tmp_path / "not_created").exists() and factory.created == []


def test_unknown_callbacks_and_infinite_deadline_are_rejected_before_start(tmp_path):
    factory = Factory(Clock())
    runner = WorkflowRunner(tmp_path / "not_created", factory)
    with pytest.raises(TypeError):
        runner.run(_spec(), _options(), resume=True)
    with pytest.raises(ValueError):
        runner.run(_spec(), _options(task_timeout_seconds=float("inf")))
    assert not (tmp_path / "not_created").exists() and factory.created == []


def test_example_is_valid_non_message_local_demonstration():
    path = Path(__file__).resolve().parents[1] / "configs" / "week6_example_workflow.json"
    spec = WorkflowSpec.model_validate_json(path.read_bytes())
    assert len(spec.subgoals) == 3
    assert all(goal.message_conversation is None for goal in [*spec.subgoals, spec.final_check])


def test_actual_taskrunner_executes_each_verified_stage_once_then_reads_fresh_final_frame(tmp_path):
    clock = Clock()
    dispatched = []
    planners = []
    observers = []

    def factory(task, index, childdir):
        frames = [_frame(f"g{index}-before", (f"Button{index}",)),
                  _frame(f"g{index}-current", (f"Button{index}",)),
                  _frame(f"g{index}-after", (f"Done{index}",)),
                  _frame(f"g{index}-verified", (f"Done{index}",)),
                  _frame(f"g{index}-final", ("AllDone",))]
        observer = FakeObserver(frames)
        observers.append(observer)
        planner = FakePlanner(_plan(PlanStep(
            step_id="click-local", description="use local demonstration button",
            action_type="click", target_text=f"Button{index}", expected_result=f"Done{index}",
        )))
        planners.append(planner)
        executor = FakeExecutor()
        dispatched.append(executor.actions)
        return TaskRunner(
            observer=observer, planner=planner, adapter=ActionAdapter(platform="darwin"),
            executor=executor, verifier=Verifier(),
            recorder=TaskRecorder(RunSession(session_id=childdir.name, directory=childdir)),
            clock=clock, sleep=lambda _: None,
        )

    result = WorkflowRunner(tmp_path / "integrated", factory, clock).run(_spec(2), _options())
    assert result.status == "succeeded"
    assert result.total_actions == result.planning_calls == 2
    assert [len(actions) for actions in dispatched] == [1, 1]
    assert [len(planner.contexts) for planner in planners] == [1, 1]
    assert observers[-1].calls == 5 and result.final_observation_id == "g1-final"
    assert all(goal.status == "completed" for goal in result.subgoals)
