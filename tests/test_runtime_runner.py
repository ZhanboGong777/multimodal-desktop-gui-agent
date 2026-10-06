"""The closed loop, driven entirely offline.

This is the test that has to pass before any real desktop action is allowed: it
proves the runner observes, plans, resolves against the *current* frame, acts,
looks again and verifies - using injected fakes, so nothing touches a screen,
a network or a model.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gui_agent.planning.planner import PlanResult
from gui_agent.planning.schemas import PlanStep, TaskPlan
from gui_agent.recording import RunSession
from gui_agent.runtime import runner as runner_module
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner, explain_model_failure
from gui_agent.runtime.schemas import (
    ElementRef,
    ExecutionOptions,
    ObservationSnapshot,
    TaskSpec,
)
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult, BoundingBox, Point, ScreenInfo


# ───────────────────────────── fakes ─────────────────────────────
def _frame(observation_id: str, texts: tuple[str, ...]) -> ObservationSnapshot:
    elements = [
        ElementRef(
            element_id=f"{observation_id}-e{index:03d}",
            text=text,
            bounding_box=BoundingBox(
                left=10, top=10 + index * 40, right=200, bottom=40 + index * 40
            ),
            center=Point(x=105, y=25 + index * 40),
            confidence=0.95,
        )
        for index, text in enumerate(texts)
    ]
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=1470, screenshot_height=956, control_width=1470, control_height=956
        ),
        elements=elements,
    )


class FakeObserver:
    """Plays a scripted sequence of frames, repeating the last one forever."""

    def __init__(self, frames: list[ObservationSnapshot]) -> None:
        self.frames = frames
        self.calls = 0

    def observe(self, *, observation_id: str | None = None) -> ObservationSnapshot:
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return frame


class FakePlanner:
    """Returns one canned plan and records what it was asked."""

    def __init__(self, plan: TaskPlan | None, error: str | None = None) -> None:
        # Named with an underscore: ``plan`` is the method the runner calls, and
        # assigning over it turns the planner into a non-callable.
        self._plan = plan
        self.error = error
        self.contexts: list[dict] = []
        self.client = type("C", (), {"model_name": "fake-model", "name": "fake"})()
        self.image_paths: list[str | None] = []

    def plan(self, instruction, *, context=None, image_path=None, task_id="task-1") -> PlanResult:
        self.contexts.append(context or {})
        self.image_paths.append(image_path)
        if self._plan is None:
            return PlanResult(error=self.error or "no plan")
        return PlanResult(plan=self._plan, attempts=1)


class FakeExecutor:
    """Records every action instead of dispatching it."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.actions: list = []
        self.fail_on = fail_on

    def execute(self, action, *, dry_run=None, screen=None) -> ActionResult:
        self.actions.append((action, dry_run))
        if self.fail_on and action.action_type == self.fail_on:
            return ActionResult(action=action, success=False, dry_run=bool(dry_run), error="boom")
        return ActionResult(action=action, success=True, dry_run=bool(dry_run))


def _plan(*steps: PlanStep) -> TaskPlan:
    return TaskPlan(task_id="t1", instruction="open the browser", steps=list(steps))


def _runner(tmp_path: Path, observer, planner, executor, **kwargs):
    session = RunSession.create(tmp_path, session_id="run-1")
    recorder = TaskRecorder(session)
    runner = TaskRunner(
        observer=observer,
        planner=planner,
        adapter=ActionAdapter(platform="darwin"),
        executor=executor,
        verifier=Verifier(),
        recorder=recorder,
        clock=kwargs.pop("clock", None) or __import__("time").monotonic,
        sleep=lambda _s: None,
    )
    return runner, recorder


# ───────────────────────────── tests ─────────────────────────────
def test_the_full_loop_runs_offline_and_verifies(tmp_path: Path) -> None:
    """observe -> plan -> resolve -> act -> observe again -> verify."""
    frames = [
        _frame("obs-0001", ("Start",)),  # initial look
        _frame("obs-0002", ("Search here",)),  # before step 1
        _frame("obs-0003", ("Search here", "Results ready")),  # after step 1
    ]
    plan = _plan(
        PlanStep(
            step_id="s1",
            description="click search",
            action_type="click",
            target_text="Search here",
            expected_result="results appear",
        ),
        PlanStep(step_id="s2", description="stop", action_type="finish"),
    )
    task = TaskSpec(
        case_id="T02",
        instruction="search",
        expect_text=["Results ready"],
        success_rules=["results page loaded"],
    )
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "succeeded"
    assert [a[0].action_type for a in executor.actions] == ["click"]
    assert executor.actions[0][1] is False, "execute=True must dispatch for real"
    assert result.verification is not None and result.verification.passed
    assert result.action_count == 1
    # every step left a record, and the terminal step never became an action
    assert [s.step_id for s in result.steps] == ["s1", "s2"]
    assert len(executor.actions) == 1


def test_the_coordinate_comes_from_the_current_frame_not_the_plan(tmp_path: Path) -> None:
    """A plan made from one screenshot must resolve against the newest one."""
    frames = [
        _frame("obs-0001", ("Start",)),
        _frame("obs-0002", ("Target",)),  # y = 25
        _frame("obs-0003", ("Target",)),  # same text, same place
    ]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)

    runner.run(task, ExecutionOptions(execute=True, confirm=False))

    action, _ = executor.actions[0]
    assert (action.x, action.y) == (105, 25)
    assert runner.recorder.read_steps()[0]["observation_id"] == "obs-0002"


def test_a_dry_run_dispatches_nothing_and_is_not_a_success(tmp_path: Path) -> None:
    frames = [_frame("obs-0001", ("Target",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.status == "dry_run_completed"
    assert result.execute is False
    assert executor.actions[0][1] is True, "dry run must pass dry_run=True to the executor"
    assert not result.ok, "a dry run is never a finished task"


def test_a_task_without_a_success_rule_is_blocked(tmp_path: Path) -> None:
    """No verifiable goal means no run - not a run that quietly 'succeeds'."""
    plan = _plan(PlanStep(step_id="s1", description="click", action_type="click", target_text="X"))
    task = TaskSpec(case_id="T", instruction="do something vague")
    executor = FakeExecutor()
    runner, _ = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("X",))]), FakePlanner(plan), executor
    )

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "blocked"
    assert executor.actions == []


def test_a_run_that_forbids_a_process_is_blocked_before_it_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T01's "no browser window is open", answered by the operating system.

    The screen cannot answer it: a browser behind another window carries none of the
    text T01 looks for, so the text check passed and the run planned against a window
    that already existed. This is the check that stops it, and it has to fire before a
    model call is spent.
    """
    monkeypatch.setattr(runner_module, "match_processes", lambda names: ["msedge.exe"])
    plan = _plan(PlanStep(step_id="s1", description="click", action_type="click",
                          target_text="Microsoft Edge"))
    task = TaskSpec(
        case_id="T01",
        instruction="open the browser",
        success_rules=["a browser window is in the foreground"],
        expect_text=["http"],
        forbids_processes=["browser"],
    )
    planner = FakePlanner(plan)
    executor = FakeExecutor()
    observer = FakeObserver([_frame("obs-0001", ("desktop",))])
    runner, _ = _runner(tmp_path, observer, planner, executor)

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "blocked"
    assert executor.actions == [], "nothing may be dispatched"
    assert planner.contexts == [], "no model call may be spent on a run that cannot count"
    assert observer.calls == 0, (
        "no screenshot may be taken either: the check is about the machine, so "
        "answering it first costs nothing and a capture plus an OCR pass is not free"
    )
    assert "no browser is running" in result.notes[0]
    assert "msedge.exe" in result.notes[0], "the note names what was actually found"


def test_the_same_run_proceeds_when_no_browser_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "match_processes", lambda names: [])
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click",
                 target_text="Microsoft Edge"),
        PlanStep(step_id="s2", description="finish", action_type="finish"),
    )
    task = TaskSpec(
        case_id="T01",
        instruction="open the browser",
        success_rules=["a browser window is in the foreground"],
        expect_text=["http"],
        forbids_processes=["browser"],
    )
    frames = [
        _frame("obs-0001", ("desktop",)),
        _frame("obs-0002", ("Microsoft Edge",)),
        _frame("obs-0003", ("Microsoft Edge", "http search")),
    ]
    planner = FakePlanner(plan)
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), planner, executor)

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert planner.contexts, "the plan was asked for once the precondition held"
    assert result.status == "succeeded"


def test_a_run_that_requires_a_process_is_blocked_when_it_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T02 is meaningless without a browser, and that is a fact about the machine."""
    monkeypatch.setattr(runner_module, "match_processes", lambda names: [])
    task = TaskSpec(
        case_id="T02",
        instruction="search the web",
        success_rules=["a results page is loaded"],
        expect_text=["GUI agent research"],
        requires_processes=["browser"],
    )
    planner = FakePlanner(_plan())
    runner, _ = _runner(tmp_path, FakeObserver([_frame("obs-0001", ("desktop",))]),
                        planner, FakeExecutor())

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "blocked"
    assert "needs a browser to be running" in result.notes[0]


def test_the_process_check_is_skipped_when_preconditions_are_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch that turns the guard off has to turn all of it off."""
    monkeypatch.setattr(runner_module, "match_processes", lambda names: ["msedge.exe"])
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Edge"),
        PlanStep(step_id="s2", description="finish", action_type="finish"),
    )
    task = TaskSpec(
        case_id="T01",
        instruction="open the browser",
        success_rules=["a browser window is in the foreground"],
        expect_text=["http"],
        forbids_processes=["browser"],
    )
    frames = [
        _frame("obs-0001", ("desktop",)),
        _frame("obs-0002", ("Edge",)),
        _frame("obs-0003", ("Edge", "http")),
    ]
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), FakeExecutor())

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False,
                                               require_preconditions=False))

    assert result.status == "succeeded"


def test_an_action_failure_stops_the_run(tmp_path: Path) -> None:
    frames = [_frame("obs-0001", ("Target",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target"),
        PlanStep(
            step_id="s2", description="click again", action_type="click", target_text="Target"
        ),
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    executor = FakeExecutor(fail_on="click")
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "failed"
    assert len(executor.actions) == 1, "the second action must not be attempted"


def test_a_plan_over_the_action_budget_is_refused_before_acting(tmp_path: Path) -> None:
    steps = [
        PlanStep(step_id=f"s{i}", description="click", action_type="click", target_text="Target")
        for i in range(5)
    ]
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    executor = FakeExecutor()
    runner, _ = _runner(
        tmp_path,
        FakeObserver([_frame("obs-0001", ("Target",))]),
        FakePlanner(_plan(*steps)),
        executor,
    )

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False, max_actions=3))

    assert result.status == "blocked"
    assert executor.actions == []


def test_declining_the_confirmation_cancels_without_acting(tmp_path: Path) -> None:
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    executor = FakeExecutor()
    runner, _ = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Target",))]), FakePlanner(plan), executor
    )

    result = runner.run(
        task, ExecutionOptions(execute=True, confirm=True), confirm=lambda _p: False
    )

    assert result.status == "cancelled"
    assert executor.actions == []


def test_a_changed_screen_that_does_not_match_the_rule_fails(tmp_path: Path) -> None:
    """The classic false positive: something happened, but not the right thing."""
    frames = [
        _frame("obs-0001", ("Start",)),
        _frame("obs-0002", ("Target",)),
        _frame("obs-0003", ("An error page",)),  # screen changed
    ]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    task = TaskSpec(
        case_id="T", instruction="search", expect_text=["Results ready"], success_rules=["r"]
    )
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "failed"
    assert result.verification is not None
    assert result.verification.outcome == "failed"


def test_an_unresolvable_step_fails_instead_of_clicking_something(tmp_path: Path) -> None:
    plan = _plan(
        PlanStep(
            step_id="s1", description="click", action_type="click", target_text="Not on screen"
        )
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    executor = FakeExecutor()
    runner, _ = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Target",))]), FakePlanner(plan), executor
    )

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "failed"
    assert executor.actions == []
    assert "no element matches" in (result.steps[0].error or "")


def test_a_failed_planning_run_is_blocked_not_empty(tmp_path: Path) -> None:
    task = TaskSpec(case_id="T", instruction="x", expect_text=["A"], success_rules=["r"])
    runner, _ = _runner(
        tmp_path,
        FakeObserver([_frame("obs-0001", ("A",))]),
        FakePlanner(None, error="backend down"),
        FakeExecutor(),
    )

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "blocked"
    assert "backend down" in (result.notes[0] if result.notes else "")


def test_every_step_is_recorded_and_nothing_is_overwritten(tmp_path: Path) -> None:
    frames = [
        _frame("obs-0001", ("Start",)),
        _frame("obs-0002", ("Target",)),
        _frame("obs-0003", ("Target", "Done")),
    ]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target"),
        PlanStep(step_id="s2", description="stop", action_type="finish"),
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, recorder = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), FakeExecutor())

    runner.run(task, ExecutionOptions(execute=True, confirm=False))

    steps = recorder.read_steps()
    assert [s["step_id"] for s in steps] == ["s1", "s2"]
    summaries = list(tmp_path.glob("run-1/task_summary.json"))
    assert summaries, "the summary must be written"
    observations = sorted(p.name for p in tmp_path.glob("run-1/obs-*.json"))
    assert len(observations) >= 2, "each observation gets its own file"
    assert observations == sorted(set(observations)), "observation ids must not collide"


def test_the_plan_is_built_from_a_real_screenshot_path(tmp_path: Path) -> None:
    """The model must receive the frame, not a description of it."""
    frame = _frame("obs-0001", ("Target",))
    frame = frame.model_copy(update={"image_path": "/tmp/shot.png"})
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    planner = FakePlanner(plan)
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    runner, _ = _runner(tmp_path, FakeObserver([frame]), planner, FakeExecutor())

    runner.run(task, ExecutionOptions(execute=False))

    assert planner.image_paths == ["/tmp/shot.png"]
    assert "Target" in planner.contexts[0]["visible_text"]


def test_a_dry_run_reports_inconclusive_not_failed(tmp_path: Path) -> None:
    """Nothing was dispatched, so the rule cannot have passed - but the run did not
    go wrong either. Calling that "failed" would misreport a clean rehearsal."""
    frames = [_frame("obs-0001", ("Target",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    task = TaskSpec(
        case_id="T", instruction="x", expect_text=["Never on screen"], success_rules=["r"]
    )
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), FakeExecutor())

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.status == "dry_run_completed"
    assert result.verification is not None
    assert result.verification.outcome == "inconclusive"
    assert "nothing was dispatched" in result.verification.detail
    # the rule's own verdict is kept, so the operator can see what would have happened
    assert result.verification.evidence["would_be"] == "failed"


# ─────────────── the confirmation gate, both answers ───────────────
def _approval_case(tmp_path: Path):
    frames = [_frame("obs-0001", ("Target",)), _frame("obs-0002", ("Target",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Target"], success_rules=["r"])
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)
    return runner, executor, task, plan


def test_accepting_the_confirmation_proceeds(tmp_path: Path) -> None:
    """The decline path had a test; the accept path did not.

    Without this, inverting the condition would cancel every approved run and the
    suite would still be green.
    """
    runner, executor, task, plan = _approval_case(tmp_path)
    seen: list[object] = []

    def approve(received) -> bool:
        seen.append(received)
        return True

    result = runner.run(task, ExecutionOptions(execute=True, confirm=True), confirm=approve)

    assert seen == [plan], "the operator must be shown the plan that will run"
    assert len(executor.actions) == 1, "an approved run must act"
    assert result.status != "cancelled"


def test_the_countdown_runs_once_before_the_first_real_action(tmp_path: Path) -> None:
    runner, executor, task, _plan_obj = _approval_case(tmp_path)
    calls: list[int] = []

    runner.run(
        task,
        ExecutionOptions(execute=True, confirm=False),
        countdown=calls.append,
    )

    assert calls == [3], "one countdown, before acting - not one per step"
    assert len(executor.actions) == 1


def test_a_dry_run_never_counts_down_or_asks(tmp_path: Path) -> None:
    """Nothing will be dispatched, so there is nothing to warn about or approve."""
    runner, _executor, task, _plan_obj = _approval_case(tmp_path)
    asked: list[object] = []
    counted: list[int] = []

    result = runner.run(
        task,
        ExecutionOptions(execute=False),
        confirm=lambda p: asked.append(p) or True,
        countdown=counted.append,
    )

    assert asked == []
    assert counted == []
    assert result.status == "dry_run_completed"


def test_a_context_overflow_failure_names_the_deployment_fix() -> None:
    """The Windows review hit this verbatim: `400 ... exceeds the available
    context size (4096 tokens)` for a 7517-token request.

    The provider names the numbers but not the remedy, and the remedy is a server
    setting, not something the code can change.
    """
    message = (
        "ModelError: BadRequestError: Error code: 400 - request (7517 tokens) "
        "exceeds the available context size (4096 tokens)"
    )
    explained = explain_model_failure(message)

    assert message in explained
    assert "OLLAMA_CONTEXT_LENGTH=16384" in explained


def test_a_failure_that_is_not_a_context_overflow_is_left_alone() -> None:
    """The hint must never be attached to an unrelated failure."""
    assert explain_model_failure("APIConnectionError: Connection error.") == (
        "APIConnectionError: Connection error."
    )


# ───────── a run may not be credited with a state it did not create ─────────
def _void_task() -> TaskSpec:
    """A task with T05's shape: the rule asks only that a marker be gone."""
    return TaskSpec(
        case_id="T05",
        instruction="close the week4 test application",
        preconditions=["the week4 test application is open and focused"],
        success_rules=["the application's window is gone"],
        forbid_text=["WEEK4-OPEN-FILE-OK"],
    )


def _void_case(tmp_path: Path, **kwargs):
    observer = FakeObserver([_frame("obs-0001", ("Desktop", "Other window"))])
    executor = FakeExecutor()
    planner = FakePlanner(
        _plan(
            PlanStep(
                step_id="s1", description="click the close box", action_type="click",
                target_text="Desktop",
            )
        )
    )
    runner, _recorder = _runner(tmp_path, observer, planner, executor, **kwargs)
    return runner, executor, planner


def test_a_real_run_is_blocked_when_the_goal_already_holds(tmp_path: Path) -> None:
    """T05's rule is satisfied by an untouched screen, and so is T01's.

    T05 only asks that the marker be gone, so a window that was never opened - or
    was minimised - passes it; T01 looks for text any open browser already carries.
    Without this guard a run that clicked and missed would still be reported as
    succeeded, because the state it claims to have created was already there.
    """
    runner, executor, planner = _void_case(tmp_path)

    result = runner.run(_void_task(), ExecutionOptions(execute=True, confirm=False))

    assert result.status == "blocked"
    assert executor.actions == [], "nothing may be dispatched once the run is void"
    assert planner.contexts == [], "there is no reason to spend a model call either"
    assert "already holds" in result.notes[0]


def test_a_dry_run_still_proceeds_when_the_goal_already_holds(tmp_path: Path) -> None:
    """A dry run dispatches nothing, so there is no state to credit it with.

    Its verdict is forced to inconclusive anyway, which means the guard would only
    stop the pipeline check the dry run exists to perform.
    """
    runner, _executor, _planner = _void_case(tmp_path)

    result = runner.run(_void_task(), ExecutionOptions(execute=False))

    assert result.status == "dry_run_completed"
    assert result.verification is not None
    assert result.verification.outcome == "inconclusive"


def test_the_precondition_guard_can_be_switched_off(tmp_path: Path) -> None:
    """`require_preconditions=False` restores the unguarded behaviour.

    Recorded as a test so the risk is explicit: with the guard off this run reports
    ``succeeded`` while the only screen state it ever saw was the one it started
    with.
    """
    runner, executor, _planner = _void_case(tmp_path)

    result = runner.run(
        _void_task(),
        ExecutionOptions(execute=True, confirm=False, require_preconditions=False),
    )

    assert result.status == "succeeded"
    assert len(executor.actions) == 1


def test_a_task_that_declares_no_preconditions_is_not_guarded(tmp_path: Path) -> None:
    """Declaring preconditions is how a task says it assumes a starting state.

    A task that declares none makes no such assumption, so its run is not refused
    even when its rule happens to hold already.
    """
    runner, executor, _planner = _void_case(tmp_path)
    task = TaskSpec(
        case_id="T99",
        instruction="x",
        success_rules=["r"],
        forbid_text=["WEEK4-OPEN-FILE-OK"],
    )

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "succeeded"
    assert len(executor.actions) == 1


def test_the_planner_is_told_the_budget_it_plans_against(tmp_path: Path) -> None:
    """8.1.7 asks for the execution limits, not just the step cap.

    A plan that overshoots the budget is refused before anything is dispatched, so
    the model is told the budget up front rather than learning it by rejection.
    """
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    planner = FakePlanner(plan)
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Start",))]), planner, FakeExecutor()
    )

    runner.run(task, ExecutionOptions(execute=False, max_actions=7, task_timeout_seconds=42))

    limits = planner.contexts[0]["limits"]
    assert "7 actions" in limits
    assert "42 s" in limits


def test_the_planner_is_given_the_frame_it_is_planning_from(tmp_path: Path) -> None:
    """8.1.4: the observation id, the screen size and the element list."""
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    planner = FakePlanner(plan)
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Start",))]), planner, FakeExecutor()
    )

    runner.run(task, ExecutionOptions(execute=False))

    context = planner.contexts[0]
    assert context["observation_id"] == "obs-0001"
    assert context["screen"] == "1470x956"
    assert "obs-0001-e000" in context["visible_text"]
    assert context["success_rules"] == ["r"]


def test_a_frame_with_no_readable_text_says_so(tmp_path: Path) -> None:
    """The screen was captured but OCR found nothing to aim at.

    That is what a locked, dark or mostly-empty screen looks like from here, and it
    makes every text target unresolvable. The run says so rather than leaving the
    operator to infer it from a bare "no element matches".
    """
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("", ""))]), FakePlanner(plan), FakeExecutor()
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    assert any("no readable text" in note for note in result.notes), result.notes


def test_a_readable_frame_does_not_carry_that_note(tmp_path: Path) -> None:
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path,
        FakeObserver([_frame("obs-0001", ("Start",))]),
        FakePlanner(plan),
        FakeExecutor(),
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    assert not any("no readable text" in note for note in result.notes), result.notes


# ───────── the second confirmation a risky task must get ─────────
def _risky_task() -> TaskSpec:
    return TaskSpec(
        case_id="T04",
        instruction="send the marker to the week4 test conversation",
        target_app="messaging",
        preconditions=["the test conversation is open"],
        success_rules=["the marker appears as a sent message"],
        expect_text=["WEEK4_MESSAGE_CHECK_001"],
        risk="high",
    )


def test_a_risky_task_is_confirmed_a_second_time_before_acting(tmp_path: Path) -> None:
    """8.3.6 and 12.2.6: sending a message gets its own, separate confirmation.

    The CLI built this prompt and never passed it to the runner, so T04 - the one
    task that really sends something - ran on a single confirmation while the help
    text, the usage guide and the troubleshooting guide all said otherwise. Nothing
    caught it because no test asked whether the prompt was ever called.
    """
    runner, executor, _planner = _void_case(tmp_path)
    asked: list[object] = []

    result = runner.run(
        _risky_task(),
        ExecutionOptions(execute=True, confirm=True, verification_timeout_seconds=0),
        confirm=lambda plan: True,
        high_risk_confirm=lambda plan: asked.append(plan) or True,
    )

    assert len(asked) == 1, "the second confirmation was never asked"
    assert len(executor.actions) == 1, "approving both prompts must let the action through"
    # The rule cannot hold on a fixed frame, so the run ends failed. That is the
    # point: it got past the gate instead of being cancelled by it.
    assert result.status != "cancelled"


def test_declining_the_second_confirmation_cancels_without_acting(tmp_path: Path) -> None:
    runner, executor, _planner = _void_case(tmp_path)

    result = runner.run(
        _risky_task(),
        ExecutionOptions(execute=True, confirm=True),
        confirm=lambda plan: True,
        high_risk_confirm=lambda plan: False,
    )

    assert result.status == "cancelled"
    assert executor.actions == []


def test_a_low_risk_task_is_not_asked_twice(tmp_path: Path) -> None:
    """Two prompts for opening a browser would train the operator to click through."""
    runner, executor, _planner = _void_case(tmp_path)
    task = TaskSpec(
        case_id="T01",
        instruction="open the browser",
        preconditions=["no browser window is open"],
        success_rules=["a browser window is in the foreground"],
        expect_text=["Done"],
        risk="low",
    )
    asked: list[object] = []

    result = runner.run(
        task,
        ExecutionOptions(execute=True, confirm=True, verification_timeout_seconds=0),
        confirm=lambda plan: True,
        high_risk_confirm=lambda plan: asked.append(plan) or True,
    )

    assert asked == []
    assert len(executor.actions) == 1
    assert result.status != "cancelled"


def test_a_dry_run_never_asks_for_the_second_confirmation(tmp_path: Path) -> None:
    """Nothing is dispatched, so there is nothing to approve twice."""
    runner, _executor, _planner = _void_case(tmp_path)
    asked: list[object] = []

    result = runner.run(
        _risky_task(),
        ExecutionOptions(execute=False),
        high_risk_confirm=lambda plan: asked.append(plan) or True,
    )

    assert asked == []
    assert result.status == "dry_run_completed"


# ───────── the record has to explain itself away from this machine ─────────
def test_the_summary_carries_the_provenance_the_hand_off_asks_for(tmp_path: Path) -> None:
    """14.2 lists what a summary must contain; most of it was missing.

    Without these the basic task report's environment table has to be filled in
    from memory afterwards, which is how a report ends up quoting a revision the
    run did not use.
    """
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Start",))]), FakePlanner(plan), FakeExecutor()
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.task_id == "T"
    assert result.platform == "darwin"
    assert re.fullmatch(r"\d+\.\d+\.\d+", result.python_version)
    assert result.screen == "screenshot 1470x956, control 1470x956"
    assert result.evidence_directory == str(recorder.directory)
    assert result.started_at is not None and result.finished_at is not None
    assert result.started_at <= result.finished_at
    assert result.planning_attempts == 1


def test_the_transport_count_is_not_the_planning_attempt_count(tmp_path: Path) -> None:
    """14.2.1 keeps the two apart on purpose: one plan can cost several requests."""
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Start",))]), FakePlanner(plan), FakeExecutor()
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    # The fake planner returns a plan without touching a client at all, so the two
    # counters agree only by accident - which is exactly why they are separate.
    assert result.planning_attempts == 1
    assert result.model_requests == 0


def test_the_written_summary_is_the_one_that_was_returned(tmp_path: Path) -> None:
    """The fields have to reach the file, not just the object."""
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Start",))]), FakePlanner(plan), FakeExecutor()
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    written = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))
    assert written["commit"] == result.commit
    assert written["screen"] == result.screen
    assert written["evidence_directory"] == result.evidence_directory
    assert written["planning_attempts"] == result.planning_attempts
    assert written["model_requests"] == result.model_requests


def test_a_stop_reason_is_only_recorded_when_something_stopped_it(tmp_path: Path) -> None:
    """A successful run was not stopped by anything, so it gets no reason."""
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Start",))]), FakePlanner(plan), FakeExecutor()
    )

    done = runner.run(task, ExecutionOptions(execute=False))

    assert done.status == "dry_run_completed"
    assert done.stop_reason == ""


def test_a_free_text_error_is_not_reported_as_a_class_name() -> None:
    """`error_type` holds a type, or nothing - not whatever precedes a colon."""
    from gui_agent.runtime.runner import _error_type

    assert _error_type("ActionResolutionError: no element matches 'x'") == "ActionResolutionError"
    assert _error_type("no element matches 'x' in obs-0002") == ""
    assert _error_type("") == ""
    assert _error_type(None) == ""


def test_a_plan_that_reports_errors_is_not_executed(tmp_path: Path) -> None:
    """8.3.5: `errors` is the model saying it could not work the task out.

    Running such a plan would be reading "I am not sure" as "go ahead". The field
    existed and nothing read it, which is how a plan the model had already flagged
    would have been dispatched anyway.
    """
    plan = TaskPlan(
        task_id="t1",
        instruction="open the browser",
        steps=[
            PlanStep(step_id="s1", description="click", action_type="click", target_text="Desktop")
        ],
        errors=["I cannot tell which window is the test application"],
    )
    executor = FakeExecutor()
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path,
        FakeObserver([_frame("obs-0001", ("Desktop",))]),
        FakePlanner(plan),
        executor,
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.status == "blocked"
    assert executor.actions == []
    assert "reports errors" in result.notes[0]


# ───────── 16.3's negative case ─────────
def test_a_wrong_page_stops_the_run_before_it_types_anything(tmp_path: Path) -> None:
    """16.3: change the second frame to an error page and the type must not happen.

    The plan is click-then-type. When the run re-observes, the search box is gone -
    so the click cannot be resolved, and continuing would type the query into
    whatever happens to be on screen instead. A wrong page is exactly when that is
    worst, so the run has to stop at the step it could not resolve.
    """
    frames = [
        _frame("obs-0001", ("Search box",)),  # the frame the plan is written from
        _frame("obs-0002", ("Service unavailable",)),  # the page went wrong
    ]
    plan = _plan(
        PlanStep(
            step_id="s1", description="click search", action_type="click",
            target_text="Search box",
        ),
        PlanStep(
            step_id="s2", description="type the query", action_type="type_text",
            target_text="Search box", arguments={"text": "GUI agent research"},
        ),
        PlanStep(step_id="s3", description="stop", action_type="finish"),
    )
    task = TaskSpec(
        case_id="T", instruction="search", expect_text=["Results ready"], success_rules=["r"]
    )
    executor = FakeExecutor()
    runner, _recorder = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    assert result.status == "failed"
    assert executor.actions == [], "nothing may be dispatched once the target is gone"
    assert [step.step_id for step in result.steps] == ["s1"], "the later steps were never reached"
    assert "no element matches" in (result.steps[0].error or "")


# ───────── where the wall clock went ─────────
class _StepClock:
    """A monotonic clock that advances a fixed amount on every reading."""

    def __init__(self, step: float = 1.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_a_blocked_run_still_records_how_long_it_took(tmp_path: Path) -> None:
    """`_blocked` used to pass `self.clock()` as the start time.

    Every blocked run therefore recorded 0.0 ms - including one that spent twenty
    seconds inside a model call before giving up, which is the case the Windows
    review actually hit. The phases come from the run's own start now.
    """
    runner, _executor, _planner = _void_case(tmp_path, clock=_StepClock(step=2.5))

    result = runner.run(_void_task(), ExecutionOptions(execute=True, confirm=False))

    assert result.status == "blocked"
    assert result.elapsed_ms == 2500.0, "one reading of a 2.5 s clock, not zero"


def test_the_timing_phases_partition_the_run(tmp_path: Path) -> None:
    """16.5.4's phases have to add up to the whole, or one of them is lying.

    A slow operator must not look like a slow model, which is why the report quotes
    `execution_ms` - the part after the confirmation gate - rather than the total.
    """
    frames = [
        _frame("obs-0001", ("Start",)),
        _frame("obs-0002", ("Target",)),
        _frame("obs-0003", ("Results ready",)),
    ]
    plan = _plan(
        PlanStep(
            step_id="s1", description="click", action_type="click", target_text="Target",
            expected_result="results appear",
        ),
        PlanStep(step_id="s2", description="stop", action_type="finish"),
    )
    task = TaskSpec(
        case_id="T", instruction="search", expect_text=["Results ready"], success_rules=["r"]
    )
    runner, _recorder = _runner(
        tmp_path,
        FakeObserver(frames),
        FakePlanner(plan),
        FakeExecutor(),
        clock=_StepClock(),
    )

    result = runner.run(
        task,
        ExecutionOptions(execute=True, confirm=True),
        confirm=lambda _plan: True,
        countdown=lambda _seconds: None,
    )

    assert result.planning_ms > 0
    assert result.confirmation_ms > 0, "the gate took a reading between plan and action"
    assert result.elapsed_ms == pytest.approx(
        result.planning_ms + result.confirmation_ms + result.execution_ms
    )


def test_a_declined_run_records_the_planning_it_did_and_the_wait_it_caused(tmp_path: Path) -> None:
    """A refusal is still a run, and its record has to say what happened.

    Driving the CLI through a real terminal showed two wrong numbers on the
    cancelled path: `planning_attempts` read 0 although the notes said a plan had
    been made, and the operator's reading time was counted as `execution_ms`. The
    second is the misattribution 16.5.4 exists to prevent - a hesitant person
    looking like a slow model.
    """
    frames = [_frame("obs-0001", ("Start",)), _frame("obs-0002", ("Target",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    executor = FakeExecutor()
    runner, _recorder = _runner(
        tmp_path, FakeObserver(frames), FakePlanner(plan), executor, clock=_StepClock()
    )

    result = runner.run(
        task,
        ExecutionOptions(execute=True, confirm=True),
        confirm=lambda _plan: False,
        countdown=lambda _seconds: None,
    )

    assert result.status == "cancelled"
    assert executor.actions == []
    assert result.planning_attempts == 1, "a plan was made, and the record should say so"
    assert result.confirmation_ms > 0, "the refusal took time, and it was the operator's"
    assert result.elapsed_ms == pytest.approx(
        result.planning_ms + result.confirmation_ms + result.execution_ms
    )


def test_a_refused_plan_is_not_put_to_the_operator_twice(tmp_path: Path) -> None:
    """Declining the first prompt must not lead to the second one."""
    runner, executor, _planner = _void_case(tmp_path)
    asked: list[object] = []

    result = runner.run(
        _risky_task(),
        ExecutionOptions(execute=True, confirm=True),
        confirm=lambda _plan: False,
        high_risk_confirm=lambda plan: asked.append(plan) or True,
    )

    assert result.status == "cancelled"
    assert asked == [], "the second prompt appeared after the first was already refused"
    assert executor.actions == []


def test_a_failed_planning_call_is_still_counted_as_planning_time(tmp_path: Path) -> None:
    """The stamp used to be set only after a plan succeeded.

    So a model call that failed after two seconds reported `planning_ms: 0` and put
    those seconds in `execution_ms` - and a blocked run looked like one that had
    been busy acting. This is the shape the Windows review hit: a 400 after a long
    wait, whose summary would have claimed the time was spent executing.
    """
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path,
        FakeObserver([_frame("obs-0001", ("Start",))]),
        FakePlanner(None, error="BadRequestError: context size exceeded"),
        FakeExecutor(),
        clock=_StepClock(),
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.status == "blocked"
    assert result.planning_ms > 0, "the call took time, and it was planning"
    # Before the fix this was the other way round: planning 0, execution everything.
    assert result.execution_ms <= result.planning_ms, "the wait belongs to planning"
    assert result.elapsed_ms == pytest.approx(result.planning_ms + result.execution_ms)


# ───────── the screen going away mid-run ─────────
class _FailingObserver:
    """Serves prepared frames, then starts raising - the display went to sleep."""

    def __init__(self, frames: list[ObservationSnapshot], fail_from: int) -> None:
        self.frames = frames
        self.fail_from = fail_from
        self.calls = 0

    def observe(self, *, observation_id: str | None = None) -> ObservationSnapshot:
        self.calls += 1
        if self.calls >= self.fail_from:
            raise RuntimeError("monitor_index 1 is out of range (available 1..0)")
        return self.frames[min(self.calls - 1, len(self.frames) - 1)]


def test_a_failed_first_look_blocks_before_planning(tmp_path: Path) -> None:
    """The path a sleeping display takes, and nothing is planned or dispatched.

    A run with no screen is not a run: there is nothing to plan against, so the
    model is not called and no action is considered.
    """
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    planner = FakePlanner(plan)
    executor = FakeExecutor()
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(tmp_path, _FailingObserver([], fail_from=1), planner, executor)

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.status == "blocked"
    assert "initial observation failed" in result.notes[0]
    assert planner.contexts == [], "there is nothing to plan against"
    assert executor.actions == []


def test_the_run_stops_when_the_screen_goes_away_before_a_step(tmp_path: Path) -> None:
    """The fresh look is what resolves the target, so without it the step stops.

    Carrying on would mean acting on the frame the plan was written from, which is
    the one thing the re-observation exists to prevent.
    """
    frames = [_frame("obs-0001", ("Target",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target")
    )
    executor = FakeExecutor()
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(tmp_path, _FailingObserver(frames, fail_from=2), FakePlanner(plan), executor)

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.status == "failed"
    assert "observation failed" in (result.steps[0].error or "")
    assert executor.actions == []


def test_the_screen_going_away_after_an_action_does_not_crash_the_run(tmp_path: Path) -> None:
    """The action was dispatched; only the look afterwards failed.

    The run has to record that and carry on to a verdict. Letting the exception out
    would turn a screen going to sleep into a traceback, and the CLI only catches
    KeyboardInterrupt.
    """
    frames = [
        _frame("obs-0001", ("Target",)),
        _frame("obs-0002", ("Target",)),
    ]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target"),
        PlanStep(step_id="s2", description="stop", action_type="finish"),
    )
    executor = FakeExecutor()
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path, _FailingObserver(frames, fail_from=3), FakePlanner(plan), executor
    )

    result = runner.run(
        task, ExecutionOptions(execute=True, confirm=False, verification_timeout_seconds=0)
    )

    assert len(executor.actions) == 1, "the action itself did succeed"
    assert "post-action observation failed" in (result.steps[0].error or "")
    assert result.status in {"failed", "blocked"}, "it still reaches a verdict"


def test_the_screen_going_away_during_final_verification_does_not_crash_the_run(
    tmp_path: Path,
) -> None:
    """The display can sleep between the last action and the verdict.

    The polling helper calls back into the observer with nothing catching a
    failure, so the exception left `run()` entirely - for the one environmental
    thing that has happened most often in this project.
    """
    frames = [
        _frame("obs-0001", ("Target",)),
        _frame("obs-0002", ("Target",)),
        _frame("obs-0003", ("Target", "Results ready")),
    ]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target"),
        PlanStep(step_id="s2", description="stop", action_type="finish"),
    )
    executor = FakeExecutor()
    task = TaskSpec(
        case_id="T", instruction="x", expect_text=["Results ready"], success_rules=["r"]
    )
    runner, _recorder = _runner(
        tmp_path, _FailingObserver(frames, fail_from=4), FakePlanner(plan), executor
    )

    result = runner.run(
        task, ExecutionOptions(execute=True, confirm=False, verification_timeout_seconds=0)
    )

    assert result.status in {"failed", "blocked"}
    assert result.verification is not None
    assert result.verification.outcome != "passed", "no screen, no verdict"


def test_the_task_budget_stops_the_run_between_steps(tmp_path: Path) -> None:
    """12.2 maps this to exit code 3, and the check is between steps.

    A single long model call is therefore bounded by `model.timeout_seconds` rather
    than by this budget - which is worth knowing before reading a `timed_out` as
    "the whole run took too long".
    """
    frames = [_frame("obs-0001", ("Target",)), _frame("obs-0002", ("Target",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Target"),
        PlanStep(step_id="s2", description="stop", action_type="finish"),
    )
    executor = FakeExecutor()
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, _recorder = _runner(
        tmp_path, FakeObserver(frames), FakePlanner(plan), executor, clock=_StepClock(step=100.0)
    )

    result = runner.run(task, ExecutionOptions(execute=False, task_timeout_seconds=50))

    assert result.status == "timed_out"
    assert "timed_out" == result.status
    assert result.stop_reason == "" or "timed" not in result.stop_reason.lower()
    assert executor.actions == []


def test_the_summary_carries_the_operating_system(tmp_path: Path) -> None:
    """Alongside the commit and the Python version, so a row in the task report
    can be read off the evidence rather than remembered."""
    plan = _plan(PlanStep(step_id="s1", description="stop", action_type="finish"))
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Done"], success_rules=["r"])
    runner, recorder = _runner(
        tmp_path, FakeObserver([_frame("obs-0001", ("Start",))]), FakePlanner(plan), FakeExecutor()
    )

    result = runner.run(task, ExecutionOptions(execute=False))

    assert result.os_version, "the summary has to say which machine this was"
    written = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))
    assert written["os_version"] == result.os_version


def test_an_interrupted_run_still_writes_a_readable_summary(tmp_path: Path) -> None:
    """14.2.3: an interrupt saves a summary too, not only a failure does.

    Ctrl+C arrives wherever the interpreter happens to be, so the in-memory step
    list is whatever the unwinding left behind. The summary is rebuilt from the
    step log, which is appended as the run goes - the interrupted run happened, and
    a report with no row for it is how an attempt disappears from a success rate.
    """
    class InterruptingExecutor(FakeExecutor):
        def execute(self, action, *, dry_run=None, screen=None) -> ActionResult:
            # One action completes - and is therefore on disk - and the next one is
            # interrupted, which is the case that loses in-memory state.
            if self.actions:
                raise KeyboardInterrupt
            return super().execute(action, dry_run=dry_run, screen=screen)

    frames = [_frame("obs-0001", ("Start",)), _frame("obs-0002", ("Start",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Start"),
        PlanStep(step_id="s2", description="click", action_type="click", target_text="Start"),
    )
    executor = InterruptingExecutor()
    runner, recorder = _runner(
        tmp_path, FakeObserver(frames), FakePlanner(plan), executor
    )
    task = TaskSpec(
        case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"]
    )
    result = runner.run(
        task, ExecutionOptions(execute=True, confirm=True), confirm=lambda _plan: True
    )

    assert result.status == "cancelled"
    assert "interrupted by the operator" in result.stop_reason
    assert len(result.steps) == 1, "the step that completed is in the summary"

    written = json.loads(
        (recorder.directory / "task_summary.json").read_text(encoding="utf-8")
    )
    assert written["status"] == "cancelled"
    assert written["run_id"]
    assert written["finished_at"]
    assert written["evidence_directory"]


def test_the_model_is_told_which_keys_it_may_use(tmp_path: Path) -> None:
    """8.1.2 asks for the platform and the allowed key names.

    The prompt said "use this platform's key names" and never listed one, so a
    plan could name a key the adapter then refused - an error the model had no way
    to avoid, on a task whose whole point may be a shortcut.
    """
    frames = [_frame("obs-0001", ("Start",)), _frame("obs-0002", ("Start",))]
    plan = _plan(PlanStep(step_id="s1", description="click", action_type="click", target_text="Start"))
    planner = FakePlanner(plan)
    runner, _ = _runner(tmp_path, FakeObserver(frames), planner, FakeExecutor())

    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])
    runner.run(task, ExecutionOptions(confirm=False))

    sent = planner.contexts[0]
    assert "allowed_keys" in sent, "the whitelist never reached the model"
    assert "enter" in sent["allowed_keys"]["keys"]
    assert "escape" in sent["allowed_keys"]["keys"]
    # The modifier names the adapter accepts. They are the same set on both
    # platforms - what differs is what each one presses, which the adapter
    # resolves. Advertising the names is what stops a plan naming a key that
    # resolves to nothing.
    assert {"command", "ctrl", "shift", "alt"} <= set(sent["allowed_keys"]["modifiers"])
    assert sent["allowed_keys"]["hotkey_separator"] == "+"


def test_the_key_list_is_the_one_the_adapter_enforces(tmp_path: Path) -> None:
    """Two lists that must agree are one list too many, so there is only one.

    A model told about a key the adapter then refuses would fail for a reason the
    prompt invented; a key the adapter accepts but never mentions is one the model
    will not use.
    """
    from gui_agent.runtime.action_adapter import ALLOWED_KEYS, keys_for_platform

    advertised = keys_for_platform("win32")

    assert set(advertised["keys"]) == set(ALLOWED_KEYS)
    assert keys_for_platform(None) == keys_for_platform("win32"), "unknown platforms fall back"
    # The names are shared; the meaning is not. `ctrl` is command on a Mac and
    # ctrl on Windows, which is the difference 9.3.2 is about.
    from gui_agent.runtime.action_adapter import _MODIFIER_ALIASES

    assert _MODIFIER_ALIASES["darwin"]["ctrl"] == "command"
    assert _MODIFIER_ALIASES["win32"]["command"] == "ctrl"


def test_a_step_whose_result_was_not_observed_stops_the_run(tmp_path: Path) -> None:
    """10.1.12: continue when the expectation holds, stop when the result does not match.

    The mismatch was computed and then never read, so a step that plainly had not
    done what it was for was recorded and the plan carried on - typing the rest of
    a query into a window that never took focus, for instance. No test failed,
    because no test asked what happened after a step verification came back
    `failed`.
    """
    frames = [
        _frame("obs-0001", ("Start",)),
        _frame("obs-0002", ("Start",)),
        _frame("obs-0003", ("Start",)),
        _frame("obs-0004", ("Start",)),
    ]
    plan = _plan(
        PlanStep(
            step_id="s1",
            description="type the query",
            action_type="type_text",
            arguments={"text": "hello"},
            expected_result="the query appears in the box",
        ),
        PlanStep(step_id="s2", description="click", action_type="click", target_text="Start"),
    )
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])

    result = runner.run(
        task, ExecutionOptions(execute=True, confirm=True), confirm=lambda _plan: True
    )

    assert result.status == "failed"
    assert len(executor.actions) == 1, "the second step must not be dispatched"
    assert result.steps[0].verification.outcome == "failed"
    assert "stopping rather than continuing" in result.stop_reason


def test_a_dry_run_does_not_stop_on_an_unobserved_expectation(tmp_path: Path) -> None:
    """Nothing is dispatched in a dry run, so an unchanged screen is the expectation.

    Stopping there would truncate the one mode whose purpose is to walk the whole
    plan: the operator would be shown the first step and told the run had failed,
    for the reason that nothing had happened - which is what a dry run is.
    """
    frames = [_frame("obs-0001", ("Start",)), _frame("obs-0002", ("Start",))]
    plan = _plan(
        PlanStep(
            step_id="s1",
            description="type the query",
            action_type="type_text",
            arguments={"text": "hello"},
            expected_result="the query appears in the box",
        ),
        PlanStep(step_id="s2", description="click", action_type="click", target_text="Start"),
    )
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), executor)
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])

    result = runner.run(task, ExecutionOptions(execute=False, confirm=False))

    assert result.status == "dry_run_completed"
    assert len(executor.actions) == 2, "the whole plan is walked through"
    assert result.steps[0].verification.outcome == "failed", "and the mismatch is still recorded"


def test_a_missing_ocr_binary_is_named_in_the_run_notes(tmp_path: Path) -> None:
    """A fresh machine has no `tesseract`, and every task would fail for it.

    The engine wraps the failure into an OcrError, `observe()` records it on the
    frame, and the operate saw only "no readable text" - which reads as a locked or
    dark screen, and the hand-back manual says exactly that. The cause sat in
    obs-0001.json, a file the operator has no reason to open while diagnosing.
    """
    frame = _frame("obs-0001", ("Start",)).model_copy(
        update={
            "elements": [],
            "errors": [
                (
                    "ocr unavailable: Tesseract failed: tesseract is not installed or "
                    "it's not in your PATH"
                )
            ],
        }
    )
    runner, _ = _runner(
        tmp_path, FakeObserver([frame]), FakePlanner(_plan()), FakeExecutor()
    )
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])

    result = runner.run(task, ExecutionOptions(confirm=False))

    notes = " ".join(result.notes)
    assert "not installed or it's not in your PATH" in notes, notes
    assert "no readable text" in notes
    assert "missing `tesseract` binary" in notes, "the note points at the wrong cause"


def test_the_written_summary_carries_the_verification(tmp_path: Path) -> None:
    """`_finish` writes the file; a field attached after it returned was written null.

    Every run's task_summary.json said `"verification": null` while the returned
    object carried the verdict the CLI printed, so the record the reviewer reads -
    and the evidence collector ships to the repository - said the task had no
    verification at all. The test report's result table has a column for it.
    """
    frames = [_frame("obs-0001", ("Results ready",)), _frame("obs-0002", ("Results ready",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Results")
    )
    runner, recorder = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), FakeExecutor())
    task = TaskSpec(
        case_id="T", instruction="x", expect_text=["Results ready"], success_rules=["r"]
    )

    result = runner.run(
        task, ExecutionOptions(execute=True, confirm=True), confirm=lambda _plan: True
    )
    written = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))

    assert result.verification is not None, "the returned object still carries it"
    assert written["verification"] is not None, "and so does the file"
    assert written["verification"]["outcome"] == result.verification.outcome
    assert written["verification"]["method"] == result.verification.method


def test_the_written_summary_of_a_dry_run_carries_it_too(tmp_path: Path) -> None:
    """The dry-run path attached its verdict the same way, and lost it the same way."""
    frames = [_frame("obs-0001", ("Start",)), _frame("obs-0002", ("Start",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Start")
    )
    runner, recorder = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), FakeExecutor())
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])

    runner.run(task, ExecutionOptions(confirm=False))
    written = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))

    assert written["verification"]["outcome"] == "inconclusive"
    assert "dry run" in written["verification"]["detail"]


def test_typed_text_is_masked_in_every_artefact_that_carries_it(tmp_path: Path) -> None:
    """A step names its action twice, and only one copy was masked.

    `resolved` is what the adapter decided; `action_result` is what the executor
    was handed. Only the second went through the redactor, so the raw text was
    written to steps.jsonl - and to the summary that embeds the same records -
    while the usage guide said typed text is never stored. Found by reading the
    files a real run produced instead of the object the code returned.
    """
    secret = "correct-horse-battery-staple"
    frames = [_frame("obs-0001", ("Start",)), _frame("obs-0002", ("Start",))]
    plan = _plan(
        PlanStep(
            step_id="s1",
            description="type the passphrase",
            action_type="type_text",
            arguments={"text": secret},
        )
    )
    runner, recorder = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), FakeExecutor())
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])

    runner.run(
        task, ExecutionOptions(execute=True, confirm=True), confirm=lambda _plan: True
    )

    steps_file = (recorder.directory / "steps.jsonl").read_text(encoding="utf-8")
    summary_file = (recorder.directory / "task_summary.json").read_text(encoding="utf-8")

    assert secret not in steps_file, "the step log stored the typed text"
    assert secret not in summary_file, "the summary stored the typed text"
    record = json.loads(steps_file.splitlines()[0])
    assert record["resolved"]["action"]["text"].startswith("<redacted>")
    assert record["action_result"]["action"]["text"].startswith("<redacted>")
    assert f"({len(secret)} chars)" in record["resolved"]["action"]["text"], (
        "the length is kept so a failed step can still be diagnosed"
    )


def test_a_blocked_run_says_what_it_saw(tmp_path: Path) -> None:
    """The blocked note carries its evidence, not only its conclusion.

    "The success rule already holds" is true both when the application was closed
    and when it is merely behind another window, and those need different things
    from the operator: one is a finished task, the other an unready desktop. The
    Windows round had to parse obs-NNNN.json by hand to tell them apart, because
    the note named neither the elements nor the errors.
    """
    # T05's rule is that the marker is gone; a frame without it satisfies the rule
    # before anything has been done, which is exactly when this guard fires.
    frames = [_frame("obs-0001", ("Desktop", "Notepad", "Files"))]
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(_plan()), FakeExecutor())
    task = TaskSpec(
        case_id="T05",
        instruction="close the app",
        preconditions=["week4_sample.txt is open in the test application"],
        forbid_text=["WEEK4-OPEN-FILE-OK"],
        success_rules=["the application's window is gone"],
    )

    result = runner.run(task, ExecutionOptions(execute=True, confirm=True), confirm=lambda _p: True)

    assert result.status == "blocked"
    note = result.stop_reason
    assert "already holds on the untouched screen" in note
    assert "elements read" in note, note
    assert "Notepad" in note, "the operator needs to see what was on screen"


class SimulatedChat:
    """Prepared pixels and events, with no desktop or network interaction."""

    def __init__(self, directory: Path, *, fault: str = "") -> None:
        self.directory = directory
        self.fault = fault
        self.calls = 0
        self.draft = ""
        self.sent = ""
        self.focused = False
        self.frames: dict[str, tuple[ObservationSnapshot, dict]] = {}
        self.actions: list = []
        self.foreground_checks = 0

    def observe(self, *, observation_id=None) -> ObservationSnapshot:
        from PIL import Image, ImageDraw

        self.calls += 1
        oid = observation_id or f"sim-{self.calls:04d}"
        dx = self.calls * 2
        dy = self.calls
        def box(left, top, right, bottom):
            return BoundingBox(left=left + dx, top=top + dy, right=right + dx, bottom=bottom + dy)
        window = box(20, 20, 620, 420)
        header = box(160, 50, 420, 80)
        composer = box(150, 300, 580, 395)
        entry = box(150, 300, 480, 395)
        send = box(500, 350, 570, 380)
        message = box(200, 180, 550, 235)
        conversation = "Other receiver" if self.fault == "wrong_initial" else "文件传输助手"
        if self.fault == "changed_recipient" and self.draft:
            conversation = "Other receiver"
        image = Image.new("RGB", (800, 500), "#888888")
        draw = ImageDraw.Draw(image)
        def rect(bounds, **kwargs):
            draw.rectangle((bounds.left, bounds.top, bounds.right - 1, bounds.bottom - 1), **kwargs)
        rect(window, fill="white", outline="black", width=2)
        rect(header, fill="white")
        # ASCII pixel identity is intentional; the response transcribes the prepared
        # header, while this pattern proves the recipient cannot change after approval.
        draw.text((header.left + 5, header.top + 5), conversation.encode("ascii", "replace").decode(), fill="black")
        rect(entry, fill="white", outline="black", width=3)
        if self.draft:
            draw.text((entry.left + 10, entry.top + 10), self.draft, fill="black")
        rect(send, fill="#22aa55" if self.draft else "#bbbbbb", outline="black", width=2)
        draw.text((send.left + 8, send.top + 8), "SEND", fill="black")
        if self.sent:
            rect(message, fill="#88ee88", outline="black")
            draw.text((message.left + 5, message.top + 5), self.sent, fill="black")
        path = self.directory / f"{oid}.png"
        image.save(path)
        elements = [
            ElementRef(element_id=f"{oid}-e001", text="", bounding_box=entry,
                       center=entry.center, confidence=0.9, source="contour"),
            ElementRef(element_id=f"{oid}-e002", text="", bounding_box=send,
                       center=send.center, confidence=0.9, source="contour"),
        ]
        if self.fault == "missing_send" and self.draft:
            elements.pop()
        if self.fault == "post_mapping_missing" and self.draft and self.calls >= 7:
            elements.pop()
        if self.fault == "duplicate_send" and self.draft:
            elements.append(elements[-1].model_copy(update={"element_id": f"{oid}-e003"}))
        snapshot = ObservationSnapshot(
            observation_id=oid, captured_at=datetime.now(UTC), image_path=str(path),
            screen_info=ScreenInfo(screenshot_width=800, screenshot_height=500,
                                   control_width=800, control_height=500),
            elements=elements, window_id="simulated:1", window_title="WeChat",
            window_class="SimulatedChat", window_bounds=window, foreground_stable=True,
        )
        assessment = {
            "status": "sent" if self.sent else ("draft" if self.draft else "not_found"),
            "conversation": conversation, "marker": self.sent or self.draft,
            "header_box": [header.left, header.top, header.right, header.bottom],
            "composer_box": [composer.left, composer.top, composer.right, composer.bottom],
            "message_box": [message.left, message.top, message.right, message.bottom] if self.sent else None,
            "composer_empty": not bool(self.draft),
            "outgoing": bool(self.sent),
            "send_state": "sent" if self.sent else "unavailable",
        }
        self.frames[oid] = snapshot, assessment
        return snapshot

    def foreground_matches(self, snapshot) -> bool:
        self.foreground_checks += 1
        return self.fault != "foreground_changed"

    def execute(self, action, *, dry_run=None, screen=None) -> ActionResult:
        self.actions.append((action, dry_run))
        if not dry_run:
            snapshot = next(reversed(self.frames.values()))[0]
            if action.action_type == "click":
                if Point(x=action.x, y=action.y) == snapshot.elements[0].center:
                    self.focused = True
                elif Point(x=action.x, y=action.y) == snapshot.elements[1].center and self.fault != "send_ineffective":
                    self.sent, self.draft = self.draft, ""
                else:
                    assert self.fault == "send_ineffective", "click must use a CURRENT detected candidate"
            elif action.action_type == "type_text":
                assert self.focused, "typing must follow a successful input-box click"
                self.draft = action.text
            else:
                raise AssertionError("T04 must click send rather than press Enter")
        return ActionResult(action=action, success=True, dry_run=bool(dry_run))


def _simulated_t04(tmp_path, *, fault="", clock=None):
    from gui_agent.models.base import ModelClient, ModelResponse
    from gui_agent.planning.planner import TaskPlanner
    from gui_agent.runtime.tasks import get_case_for_run

    task, _ = get_case_for_run("T04", now=datetime(2026, 10, 6, tzinfo=UTC))
    assert task is not None
    chat = SimulatedChat(tmp_path, fault=fault)

    class ChatModel(ModelClient):
        name = "mock"

        def __init__(self):
            super().__init__(model_name="prepared-T04", max_retries=0)
            self.mapping_calls = 0
            self.approved_plan = None
            self.planning_contexts = []
            self.assessment_contexts = []

        def complete(self, messages, **kwargs):
            system = messages[0]["content"]
            context = json.loads(messages[-1]["content"])["context"]
            if system.startswith("Assess the attached foreground-window crop"):
                self.assessment_contexts.append(context)
                payload = dict(chat.frames[context["observation_id"]][1])
                # The verifier now sends only this frame's foreground crop.
                # Keep simulated evidence in that actual image's coordinate
                # space, including when the window translates between frames.
                window = chat.frames[context["observation_id"]][0].window_bounds
                assert window is not None
                for name in ("header_box", "message_box", "composer_box"):
                    region = payload[name]
                    if region is not None:
                        payload[name] = [region[0] - window.left, region[1] - window.top,
                                         region[2] - window.left, region[3] - window.top]
                if fault == "uncertain_after_send" and chat.sent:
                    payload["status"] = "uncertain"
            elif system.startswith("Map the target"):
                self.mapping_calls += 1
                if fault.startswith("mapping_timeout"):
                    clock.value += 100
                candidates = [item["element_id"] for item in context["current_candidates"] if item["element_id"].endswith("e002")]
                if fault == "duplicate_send":
                    candidates += [item["element_id"] for item in context["current_candidates"] if item["element_id"].endswith("e003")]
                payload = {"candidate_ids": candidates}
                if fault == "mapping_timeout_invalid":
                    payload = {"candidate_ids": []}
            else:
                self.planning_contexts.append(context)
                oid = context["observation_id"]
                if fault == "marker_in_description_only":
                    # The shape measured on T04_20261006_225340: the model names the marker
                    # in its own sentence and leaves both text fields empty. `target_text` is
                    # None and `arguments` has no "text", so `ActionAdapter`'s fallback cannot
                    # help and the runner's marker check refuses.
                    steps = [
                        PlanStep(step_id="focus", description="click message input", action_type="click", arguments={"element_id": f"{oid}-e001"}),
                        PlanStep(step_id="type", description=f"Type {task.expect_text[0]}", action_type="type_text"),
                        PlanStep(step_id="send", description="click send button", action_type="click", arguments={"element_id": f"{oid}-e002"}),
                    ]
                elif fault == "marker_never_stated":
                    # The guard on the other side: a description that talks about typing
                    # without containing the marker must still be refused.
                    steps = [
                        PlanStep(step_id="focus", description="click message input", action_type="click", arguments={"element_id": f"{oid}-e001"}),
                        PlanStep(step_id="type", description="type the run's marker into the box", action_type="type_text"),
                        PlanStep(step_id="send", description="click send button", action_type="click", arguments={"element_id": f"{oid}-e002"}),
                    ]
                else:
                    steps = [
                        PlanStep(step_id="focus", description="click message input", action_type="click", arguments={"element_id": f"{oid}-e001"}),
                        PlanStep(step_id="type", description="type this run's marker", action_type="type_text", arguments={"text": task.expect_text[0]}),
                        PlanStep(step_id="send", description="click send button", action_type="click", arguments={"element_id": f"{oid}-e002"}),
                    ]
                if fault == "draft_only":
                    steps.pop()
                if fault in {"extra_send", "repeat_message"}:
                    if fault == "repeat_message":
                        steps.append(steps[1].model_copy(update={"step_id": "type-again"}))
                    steps.append(steps[2].model_copy(update={"step_id": "send-again"}))
                self.approved_plan = TaskPlan(task_id="T04", instruction=task.instruction, steps=steps)
                payload = self.approved_plan.model_dump(mode="json")
            return ModelResponse(content=json.dumps(payload), model_name=self.model_name, provider=self.name)

    client = ChatModel()
    recorder = TaskRecorder(RunSession.create(tmp_path / "records", session_id="simulated-T04"))
    runner = TaskRunner(observer=chat, planner=TaskPlanner(client, max_format_retries=0),
                        adapter=ActionAdapter(platform="win32"), executor=chat,
                        verifier=Verifier(), recorder=recorder, sleep=lambda _: None,
                        clock=clock or __import__("time").monotonic)
    options = ExecutionOptions(execute=True, confirm=True, max_planning_attempts=1,
                               verification_timeout_seconds=0, task_timeout_seconds=60)
    return runner, task, options, chat, client, recorder


def test_t04_complete_send_uses_fresh_candidates_and_strict_message_evidence(tmp_path):
    runner, task, options, chat, client, recorder = _simulated_t04(tmp_path)
    confirmations = []
    def approve(kind):
        def callback(plan):
            assert not chat.actions
            confirmations.append(kind)
            return True
        return callback
    result = runner.run(task, options, confirm=approve("plan"), high_risk_confirm=approve("send"))
    assert result.status == "succeeded", result.stop_reason + str(result.notes)
    assert confirmations == ["plan", "send"]
    assert [action.action_type for action, _ in chat.actions] == ["click", "type_text", "click"]
    assert chat.sent == task.expect_text[0] and not chat.draft
    assert client.mapping_calls == 1, "enabled send button needs a bounded visual refresh"
    assert chat.foreground_checks == 3
    assert client.approved_plan.steps[-1].arguments["element_id"] == "sim-0001-e002"
    assert result.steps[-1].observation_id != "sim-0001"
    assert result.verification.evidence["assessment_method"] == "vision_model"
    written = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))
    assert written["status"] == "succeeded" and written["provider"] == "mock"
    context = json.loads((recorder.directory / "message_context.json").read_text(encoding="utf-8"))
    assert context["outcome"] == "passed"
    assert context["evidence"]["vision_assessment"]["conversation"] == task.message_conversation


@pytest.mark.parametrize("fault,actions", [
    ("wrong_initial", 0), ("foreground_changed", 0), ("changed_recipient", 2),
    ("missing_send", 2), ("duplicate_send", 2), ("draft_only", 2), ("send_ineffective", 3),
])
def test_t04_refuses_wrong_recipient_missing_target_and_draft_success(tmp_path, fault, actions):
    runner, task, options, chat, _, _ = _simulated_t04(tmp_path, fault=fault)
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status in {"failed", "blocked"}
    assert len(chat.actions) == actions
    assert not chat.sent
    if result.verification is not None:
        assert not result.verification.passed


def test_t04_takes_the_marker_from_the_description_when_the_argument_is_empty(tmp_path):
    """The failure measured on the real machine, and the one thing the fix accepts.

    `T04_20261006_225340`, two passes running:

        step_id     = s2
        description = "Type WEEK4_MESSAGE_CHECK_20261006_225340"
        target_text = None
        arguments   = {}          (no "text")
        error       = visual grounding refused: typed message must exactly match this
                      run's marker

    The `click` before it resolved to `(1705,919)` - the editor - so the plan was right
    and only the field was wrong. `ActionAdapter` already falls back to `target_text`,
    but that cannot help when the model leaves *both* text fields empty, and it cannot
    know the marker either because `resolve()` receives no task.

    The narrowed acceptance is the point: the marker must appear **verbatim** in the
    step's own description. Nothing is inferred from free prose.
    """
    runner, task, options, chat, _, _ = _simulated_t04(
        tmp_path, fault="marker_in_description_only"
    )
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)

    assert result.status == "succeeded", result.stop_reason + str(result.notes)
    assert chat.sent == task.expect_text[0], "the marker reached the client despite the empty argument"
    assert not chat.draft
    assert [action.action_type for action, _ in chat.actions] == ["click", "type_text", "click"]
    assert any("took it from the description" in note for note in result.notes), result.notes


def test_t04_still_refuses_a_description_that_never_states_the_marker(tmp_path):
    """The guard on the other side: a description about typing is not the marker.

    If the fix read free prose instead of looking for the marker verbatim, this shape
    would type the sentence into the box. It must keep refusing.
    """
    runner, task, options, chat, _, _ = _simulated_t04(tmp_path, fault="marker_never_stated")
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)

    assert result.status in {"failed", "blocked"}
    assert not chat.sent, "nothing may be typed when the marker was never stated"
    assert any(
        "must exactly match this run's marker" in (step.error or "") for step in result.steps
    ), [step.error for step in result.steps]


def test_t04_second_confirmation_declined_dispatches_no_actions(tmp_path):
    runner, task, options, chat, _, _ = _simulated_t04(tmp_path)
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: False)
    assert result.status == "cancelled" and not chat.actions


def test_t04_slow_mapping_never_dispatches_after_task_deadline(tmp_path):
    class ManualClock:
        value = 0.0
        def __call__(self):
            return self.value
    for fault in ("mapping_timeout", "mapping_timeout_invalid"):
        directory = tmp_path / fault
        directory.mkdir()
        clock = ManualClock()
        runner, task, options, chat, _, _ = _simulated_t04(directory, fault=fault, clock=clock)
        result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
        assert result.status == "timed_out"
        assert len(chat.actions) == 2 and not chat.sent
        assert chat.calls == 6, "expired mapping response must not start another capture"


def test_t04_post_mapping_target_disappearance_refuses_send(tmp_path):
    runner, task, options, chat, client, _ = _simulated_t04(tmp_path, fault="post_mapping_missing")
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "failed" and len(chat.actions) == 2 and not chat.sent
    assert client.mapping_calls == 1


def test_t04_uncertain_sent_verification_never_retries_the_message(tmp_path):
    runner, task, options, chat, client, _ = _simulated_t04(tmp_path, fault="uncertain_after_send")
    options.max_planning_attempts = 4
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "failed" and chat.sent == task.expect_text[0]
    assert len(chat.actions) == 3 and client.mapping_calls == 1
    assert result.planning_attempts == 1
    assert result.verification.outcome == "inconclusive"


@pytest.mark.parametrize("fault", ["extra_send", "repeat_message"])
def test_t04_stops_the_approved_plan_after_one_send_attempt(tmp_path, fault):
    runner, task, options, chat, _, _ = _simulated_t04(tmp_path, fault=fault)
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "succeeded" and chat.sent == task.expect_text[0]
    assert len(chat.actions) == 3 and len(result.steps) == 3


def test_t04_preaction_observation_expiry_starts_no_mapping_or_action(tmp_path):
    class ManualClock:
        value = 0.0
        def __call__(self):
            return self.value
    clock = ManualClock()
    runner, task, options, chat, client, _ = _simulated_t04(tmp_path, clock=clock)
    original = chat.observe
    def observe(**kwargs):
        frame = original(**kwargs)
        if chat.calls == 2:
            clock.value = 100
        return frame
    chat.observe = observe
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "timed_out" and not chat.actions and client.mapping_calls == 0


def test_anonymous_same_id_cannot_bypass_window_validation(tmp_path):
    runner, _, _, chat, _, _ = _simulated_t04(tmp_path)
    source = chat.observe()
    current = source.model_copy(update={"window_id": "different-window"})
    step = PlanStep(step_id="click", description="click", action_type="click",
                    arguments={"element_id": source.elements[0].element_id})
    from gui_agent.runtime.visual_grounding import GroundingError
    with pytest.raises(GroundingError, match="foreground window changed"):
        runner._ground_step(step, source, current, [])
    current = source.model_copy(update={
        "elements": [source.elements[1].model_copy(update={"element_id": source.elements[0].element_id})],
    })
    with pytest.raises(GroundingError):
        runner._ground_step(step, source, current, [])


def test_replanning_grounds_from_the_actual_retry_planning_frame(tmp_path):
    chat = SimulatedChat(tmp_path)
    class RetryPlanner(FakePlanner):
        def plan(self, instruction, *, context=None, image_path=None, task_id="task-1"):
            self.contexts.append(context)
            oid = context["observation_id"]
            target = "missing" if len(self.contexts) == 1 else f"{oid}-e001"
            return PlanResult(plan=_plan(PlanStep(step_id="click", description="click input",
                                                 action_type="click", arguments={"element_id": target})), attempts=1)
    planner = RetryPlanner(_plan())
    runner, _ = _runner(tmp_path / "records", chat, planner, chat)
    received = []
    original = runner._ground_step
    def grounding(step, source, current, notes, **kwargs):
        received.append(source.observation_id)
        return original(step, source, current, notes, **kwargs)
    runner._ground_step = grounding
    result = runner.run(TaskSpec(case_id="T", instruction="click", expect_text=["unseen"]),
                        ExecutionOptions(execute=True, confirm=False, max_planning_attempts=2,
                                         verification_timeout_seconds=0))
    assert result.status == "failed" and len(chat.actions) == 1
    assert received == [context["observation_id"] for context in planner.contexts]
    assert received[0] != received[1]


def test_replanning_obeys_cumulative_action_budget(tmp_path):
    planner = FakePlanner(_plan(
        PlanStep(step_id="click", description="click", action_type="click", target_text="Start")
    ))
    executor = FakeExecutor()
    runner, _ = _runner(tmp_path, FakeObserver([_frame("obs-1", ("Start",))]), planner, executor)
    result = runner.run(TaskSpec(case_id="T", instruction="click", expect_text=["unseen"]),
                        ExecutionOptions(execute=True, confirm=False, max_actions=1,
                                         max_planning_attempts=4, verification_timeout_seconds=0))
    assert result.status == "failed" and len(executor.actions) == 1
    assert len(planner.contexts) == 1


def _chat_controls(chat, frame):
    """Detected editor border and OCR send label, plus distracting real frame ids."""
    assessment = chat.frames[frame.observation_id][1]
    composer = BoundingBox(
        left=assessment["composer_box"][0], top=assessment["composer_box"][1],
        right=assessment["composer_box"][2], bottom=assessment["composer_box"][3],
    )

    def candidate(number, box, text="", source="contour", confidence=0.9):
        return ElementRef(
            element_id=f"{frame.observation_id}-e{number:03d}", text=text,
            bounding_box=box, center=box.center, confidence=confidence, source=source,
        )

    frame.elements.extend([
        candidate(90, composer),
        candidate(91, frame.elements[1].bounding_box, text="发送", source="ocr"),
        candidate(92, BoundingBox(left=40, top=100, right=100, bottom=120), text="Sidebar", source="ocr"),
        candidate(93, BoundingBox(left=200, top=180, right=450, bottom=210), text="Transcript", source="ocr"),
        candidate(94, BoundingBox(left=700, top=100, right=750, bottom=120), text="Other app", source="ocr"),
        candidate(95, BoundingBox(left=composer.left - 1, top=composer.top + 10,
                                  right=composer.left + 20, bottom=composer.top + 40), text="Partial", source="ocr"),
        candidate(96, frame.elements[1].bounding_box, text="Weak", source="ocr", confidence=0.2),
        candidate(97, frame.elements[1].bounding_box, text="Manual", source="manual"),
    ])
    return composer


def _approved_message_context(tmp_path):
    runner, task, options, chat, client, _ = _simulated_t04(tmp_path)
    source = chat.observe()
    composer = _chat_controls(chat, source)
    approved = runner.verifier.check_message_context(task, source, require_empty=True)
    assert approved.passed
    header = approved.evidence["vision_assessment"]["header_box"]
    runner._message_header = (source, BoundingBox(left=header[0], top=header[1], right=header[2], bottom=header[3]))
    runner._message_composer = (source, composer)
    return runner, task, options, chat, client, source


def test_message_planning_context_contains_only_authorised_current_editor_controls(tmp_path):
    runner, task, options, _, _, source = _approved_message_context(tmp_path)
    context = runner._context(source, task, options)
    message = context["message_task"]
    expected = [f"{source.observation_id}-e{number:03d}" for number in (1, 2, 90, 91)]
    assert message["candidate_ids"] == expected
    assert message["editor_candidate_id"] == f"{source.observation_id}-e090"
    assert message["send_candidate_id"] == f"{source.observation_id}-e091"
    assert message["type_text_arguments"] == {"text": task.expect_text[0]}
    assert message["text_already_typed"] is False
    assert "no element_id/target_text" in message["rules"]
    assert "Never press Enter" in message["rules"]
    assert all(element_id in context["visible_text"] for element_id in expected)
    assert all(label not in context["visible_text"] for label in ("Sidebar", "Transcript", "Other app", "Partial", "Weak", "Manual"))
    assert len(source.elements) == 10, "planning context must not remove full-frame audit evidence"


def test_message_editor_context_tracks_window_translation_and_uses_only_new_frame_ids(tmp_path):
    runner, task, options, chat, _, source = _approved_message_context(tmp_path)
    current = chat.observe()
    composer = _chat_controls(chat, current)
    context = runner._context(current, task, options)
    message = context["message_task"]
    assert message["source_observation_id"] == source.observation_id
    assert message["composer_bounds"] == composer.model_dump()
    assert all(element_id.startswith(current.observation_id + "-e") for element_id in message["candidate_ids"])
    assert source.observation_id not in context["visible_text"]
    assert message["editor_candidate_id"] == f"{current.observation_id}-e090"
    assert message["send_candidate_id"] == f"{current.observation_id}-e091"


@pytest.mark.parametrize("problem", ["unknown_editor", "window", "unstable", "resize", "header"])
def test_message_context_with_unusable_authorisation_offers_no_click_candidates(tmp_path, problem):
    from PIL import Image, ImageDraw
    runner, task, options, chat, _, _ = _approved_message_context(tmp_path)
    current = chat.observe()
    _chat_controls(chat, current)
    if problem == "unknown_editor":
        runner._message_composer = None
    elif problem == "window":
        current.window_id = "another-window"
    elif problem == "unstable":
        current.foreground_stable = False
    elif problem == "resize":
        current.window_bounds = current.window_bounds.model_copy(update={"right": current.window_bounds.right + 1})
    else:
        with Image.open(current.image_path) as image:
            altered = image.copy()
        header = chat.frames[current.observation_id][1]["header_box"]
        ImageDraw.Draw(altered).rectangle(tuple(header), fill="red")
        altered.save(current.image_path)
    context = runner._context(current, task, options)
    assert context["visible_text"] == ""
    assert context["message_task"]["region_status"] == "unavailable"
    assert "candidate_ids" not in context["message_task"]
    assert "Return no steps" in context["message_task"]["rules"]


@pytest.mark.parametrize("ambiguity", ["editor", "send", "no_send_ocr"])
def test_message_context_never_guesses_ambiguous_role_ids_or_drops_unlabelled_send(tmp_path, ambiguity):
    runner, task, options, _, _, source = _approved_message_context(tmp_path)
    editor = source.element(f"{source.observation_id}-e090")
    send = source.element(f"{source.observation_id}-e091")
    if ambiguity == "editor":
        source.elements.append(editor.model_copy(update={"element_id": f"{source.observation_id}-e098"}))
    elif ambiguity == "send":
        source.elements.append(send.model_copy(update={"element_id": f"{source.observation_id}-e098", "text": "Send"}))
    else:
        source.elements.remove(send)
    context = runner._context(source, task, options)
    message = context["message_task"]
    if ambiguity == "editor":
        assert "editor_candidate_id" not in message
    else:
        assert "send_candidate_id" not in message
    assert f"{source.observation_id}-e002" in message["candidate_ids"]
    assert "<unlabelled box>" in context["visible_text"]
    assert len(message["candidate_ids"]) >= 3, "role hints do not replace the actual element list"


def test_non_message_planning_context_remains_the_complete_observation(tmp_path):
    from gui_agent.runtime.observation import describe_elements
    runner, _, options, _, _, source = _approved_message_context(tmp_path)
    task = TaskSpec(case_id="T02", instruction="search", target_app="browser", success_rules=["results"])
    context = runner._context(source, task, options)
    assert set(context) == {"platform", "observation_id", "screen", "target_app", "allowed_keys", "limits", "visible_text", "success_rules"}
    assert context["visible_text"] == describe_elements(source)
    assert "Sidebar" in context["visible_text"] and "Other app" in context["visible_text"]


def test_exact_message_literal_is_given_to_planning_but_withheld_from_visual_assessment(tmp_path):
    runner, task, options, chat, client, _ = _simulated_t04(tmp_path)
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "succeeded"
    message = client.planning_contexts[0]["message_task"]
    assert message["type_text_arguments"] == {"text": task.expect_text[0]}
    assert all(task.expect_text[0] not in json.dumps(context) for context in client.assessment_contexts)
    assert runner._message_composer[0].observation_id == next(iter(chat.frames))
    assert runner._message_text_typed is True


def test_successful_typing_retry_plans_only_the_remaining_send_action(tmp_path):
    runner, task, options, chat, _, _ = _simulated_t04(tmp_path, fault="draft_only")
    original = runner.planner.plan
    contexts = []
    def retry_plan(instruction, *, context=None, **kwargs):
        contexts.append(context)
        if len(contexts) == 1:
            return original(instruction, context=context, **kwargs)
        oid = context["observation_id"]
        return PlanResult(plan=_plan(
            PlanStep(step_id="send", description="click current send", action_type="click", arguments={"element_id": f"{oid}-e002"}),
            PlanStep(step_id="finish", description="finish", action_type="finish"),
        ))
    runner.planner.plan = retry_plan
    options = options.model_copy(update={"max_planning_attempts": 2})
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "succeeded"
    assert len(contexts) == 2
    retry = contexts[1]["message_task"]
    assert retry["text_already_typed"] is True and "type_text_arguments" not in retry
    assert "Only click" in retry["rules"] and "Do not type again" in retry["rules"]
    assert "refocus" in retry["rules"]
    assert [action.action_type for action, _ in chat.actions] == ["click", "type_text", "click"]


def test_new_run_resets_message_typing_and_authorised_region_state(tmp_path):
    runner, task, options, chat, client, source = _approved_message_context(tmp_path)
    runner._message_text_typed = True
    result = runner.run(task, options, confirm=lambda _: False)
    assert result.status == "cancelled" and not chat.actions
    assert runner._message_text_typed is False
    assert runner._message_composer[0].observation_id != source.observation_id
    assert client.planning_contexts[-1]["message_task"]["text_already_typed"] is False


@pytest.mark.parametrize("target", ["sidebar", "transcript", "other_app", "partial", "coordinates"])
def test_actual_message_dispatch_refuses_targets_outside_the_authorised_current_candidate_list(tmp_path, target):
    runner, task, options, chat, _, _ = _simulated_t04(tmp_path)
    observe = chat.observe
    def observed(**kwargs):
        frame = observe(**kwargs)
        _chat_controls(chat, frame)
        return frame
    chat.observe = observed
    target_number = {"sidebar": 92, "transcript": 93, "other_app": 94, "partial": 95, "coordinates": 1}[target]
    target_text = {"sidebar": "Sidebar", "transcript": "Transcript", "other_app": "Other app", "partial": "Partial", "coordinates": None}[target]
    def bad_plan(instruction, *, context=None, **kwargs):
        arguments = {"element_id": f"{context['observation_id']}-e{target_number:03d}"}
        if target == "coordinates":
            arguments.update(x=5, y=5)
        return PlanResult(plan=_plan(PlanStep(
            step_id="unsafe", description="click", action_type="click", target_text=target_text, arguments=arguments,
        )))
    runner.planner.plan = bad_plan
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "failed" and not chat.actions
    assert any("visual grounding refused" in (step.error or "") for step in result.steps)
    if target == "coordinates":
        assert "not coordinates" in result.steps[-1].error
    else:
        assert "inside the authorised composer" in result.steps[-1].error


@pytest.mark.parametrize("candidate_number", [1, 2, 90, 91])
def test_message_click_guard_accepts_real_current_editor_icon_and_ocr_candidates(tmp_path, candidate_number):
    runner, _, _, _, _, source = _approved_message_context(tmp_path)
    step = PlanStep(step_id="click", description="click", action_type="click",
                    arguments={"element_id": f"{source.observation_id}-e{candidate_number:03d}"})
    resolved = runner.adapter.resolve(step, source)
    runner._guard_message_click(step, resolved, source)
    assert resolved.element_id in {item.element_id for item in runner._message_controls(source)[1]}
    assert resolved.screenshot_point == source.element(resolved.element_id).center
