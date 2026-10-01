"""Run the Week 4 closed loop end to end without a desktop or a model.

    python scripts/week4_offline_demo.py
    python scripts/week4_offline_demo.py --fail-at 2   # second action fails

The rule-based mock planner produces plans that name elements which are not on
the real screen, so a live dry run stops at the first unresolved step. That is
correct behaviour and useless as a demonstration. This script supplies a scripted
observation sequence and a scripted plan instead, so the whole path - observe,
plan, resolve against the current frame, act, look again, verify - is visible and
reproducible on any machine.

Nothing here touches a screen, a network or a model. It is a demonstration of the
loop's mechanics, not evidence that any real task succeeds.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.planning.planner import PlanResult
from gui_agent.planning.schemas import PlanStep, TaskPlan
from gui_agent.recording import RunSession
from gui_agent.runtime import (
    ActionAdapter,
    ExecutionOptions,
    ObservationSnapshot,
    TaskRecorder,
    TaskRunner,
    TaskSpec,
)
from gui_agent.runtime.schemas import ElementRef
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult, BoundingBox, Point, ScreenInfo

BASE = datetime(2026, 10, 1, 9, 0, 0, tzinfo=UTC)


def frame(observation_id: str, rows: list[str], *, image: str | None = None) -> ObservationSnapshot:
    """Build a synthetic frame whose elements sit on a tidy grid."""
    elements = [
        ElementRef(
            element_id=f"{observation_id}-e{index:03d}",
            text=text,
            bounding_box=BoundingBox(
                left=40, top=60 + index * 44, right=520, bottom=96 + index * 44
            ),
            center=Point(x=280, y=78 + index * 44),
            confidence=0.94,
        )
        for index, text in enumerate(rows)
    ]
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=BASE + timedelta(seconds=len(observation_id)),
        image_path=image,
        screen_info=ScreenInfo(
            screenshot_width=1920,
            screenshot_height=1080,
            control_width=1920,
            control_height=1080,
        ),
        elements=elements,
        ocr_engine="scripted",
        processing_time_ms=1.0,
    )


class ScriptedObserver:
    """Hands out the next frame on each call, then repeats the last one."""

    def __init__(self, frames: list[ObservationSnapshot]) -> None:
        self.frames = frames
        self.index = 0

    def observe(self, *, observation_id: str | None = None) -> ObservationSnapshot:
        current = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return current


class ScriptedPlanner:
    """Returns a fixed plan, as if a model had produced it."""

    def __init__(self, plan: TaskPlan) -> None:
        self._plan = plan
        self.client = type("C", (), {"model_name": "scripted-planner", "name": "scripted"})()

    def plan(self, instruction, *, context=None, image_path=None, task_id="task-1") -> PlanResult:
        return PlanResult(plan=self._plan, attempts=1)


class RecordingExecutor:
    """Dispatches nothing; prints what a real executor would have done."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.seen: list[str] = []
        self.fail_at = fail_at

    def execute(self, action, *, dry_run=None, screen=None) -> ActionResult:
        self.seen.append(action.action_type)
        marker = f"  executor: {action.action_type}"
        if action.x is not None:
            marker += f" at ({action.x},{action.y})"
        if action.text is not None:
            marker += f" text={action.text!r}"
        if action.key is not None:
            marker += f" key={action.key!r}"
        print(marker + ("   [would dispatch]" if dry_run else "   [DISPATCHED]"))

        if self.fail_at is not None and len(self.seen) == self.fail_at:
            return ActionResult(
                action=action, success=False, dry_run=bool(dry_run), error="scripted failure"
            )
        return ActionResult(action=action, success=True, dry_run=bool(dry_run))


def build_plan() -> TaskPlan:
    return TaskPlan(
        task_id="T02",
        instruction="Search the web for GUI agent research",
        summary="Open the browser, search, confirm the results page.",
        steps=[
            PlanStep(
                step_id="step-1",
                description="Click the browser icon",
                action_type="click",
                target_text="Browser",
                expected_result="the browser window opens",
            ),
            PlanStep(
                step_id="step-2",
                description="Click the search field",
                action_type="click",
                target_text="Search the web",
                expected_result="the field is focused",
            ),
            PlanStep(
                step_id="step-3",
                description="Type the query",
                action_type="type_text",
                arguments={"text": "GUI agent research"},
                target_text=None,
            ),
            PlanStep(
                step_id="step-4",
                description="Press Enter",
                action_type="key_press",
                arguments={"key": "enter"},
            ),
            PlanStep(step_id="step-5", description="Report the result", action_type="finish"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-at", type=int, default=None, help="make the Nth dispatched action fail"
    )
    parser.add_argument("--output-directory", default=str(REPO_ROOT / "outputs" / "week4_offline"))
    args = parser.parse_args()

    frames = [
        frame("obs-0001", ["Desktop", "Browser", "Files"]),  # initial look
        frame("obs-0002", ["Browser", "Search the web", "Bookmarks"]),  # before step 1
        frame("obs-0003", ["Browser", "Search the web", "Bookmarks"]),  # after step 1
        frame("obs-0004", ["Browser", "Search the web", "Bookmarks"]),  # before step 2
        frame("obs-0005", ["Browser", "Search the web", "Bookmarks"]),  # after step 2
        frame("obs-0006", ["Browser", "Search the web", "Bookmarks"]),  # steps 3 and 4
        frame("obs-0007", ["GUI agent research - Results", "Papers", "Videos"]),  # the goal appears
    ]

    task = TaskSpec(
        case_id="T02",
        instruction="Search the web for GUI agent research",
        success_rules=["a results page is loaded and the query is visible"],
        expect_text=["GUI agent research"],
    )

    session = RunSession.create(args.output_directory, session_id="offline_demo")
    executor = RecordingExecutor(fail_at=args.fail_at)
    runner = TaskRunner(
        observer=ScriptedObserver(frames),
        planner=ScriptedPlanner(build_plan()),
        adapter=ActionAdapter(platform=sys.platform if sys.platform == "darwin" else "win32"),
        executor=executor,
        verifier=Verifier(),
        recorder=TaskRecorder(session),
        sleep=lambda _seconds: None,
    )

    print("  offline closed-loop demonstration")
    print(f"  task   : {task.case_id}  {task.instruction}")
    print(f"  rule   : {task.success_rules[0]}")
    print(f"  records: {session.directory}")
    print()

    result = runner.run(task, ExecutionOptions(execute=True, confirm=False))

    print()
    print(f"  status      : {result.status}")
    print(f"  actions     : {result.action_count} dispatched")
    if result.verification is not None:
        print(f"  verification: {result.verification.outcome} - {result.verification.detail}")
    for note in result.notes:
        print(f"  note        : {note}")
    print(f"  summary     : {Path(session.directory) / 'task_summary.json'}")
    print()
    print("  This ran against scripted frames: it demonstrates the loop, not a real task.")
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
