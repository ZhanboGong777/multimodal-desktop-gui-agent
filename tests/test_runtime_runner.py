"""The closed loop, driven entirely offline.

This is the test that has to pass before any real desktop action is allowed: it
proves the runner observes, plans, resolves against the *current* frame, acts,
looks again and verifies - using injected fakes, so nothing touches a screen,
a network or a model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gui_agent.planning.planner import PlanResult
from gui_agent.planning.schemas import PlanStep, TaskPlan
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
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
