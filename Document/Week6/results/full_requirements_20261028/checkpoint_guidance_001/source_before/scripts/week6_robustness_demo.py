"""Run the real Week 6 runtime against deterministic, explicitly simulated backends.

This is a no-GUI integration demonstration, not model or desktop acceptance.
Every invocation creates a new batch; an existing --output-dir is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pydantic import ValidationError

from gui_agent.planning import PlanResult, PlanStep, TaskPlan
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import ElementRef, ExecutionOptions, ObservationSnapshot, TaskSpec
from gui_agent.runtime.verification import Verifier
from gui_agent.runtime.workflow import WorkflowRunner, WorkflowSpec
from gui_agent.schemas import ActionResult, BoundingBox, DesktopAction, ScreenInfo

WORDS = ("One", "Two", "Three")
SOURCE_PATHS = (
    "src/gui_agent/runtime/runner.py", "src/gui_agent/runtime/action_adapter.py",
    "src/gui_agent/runtime/verification.py", "src/gui_agent/runtime/recorder.py",
    "src/gui_agent/runtime/schemas.py", "src/gui_agent/runtime/workflow.py",
    "src/gui_agent/runtime/monitoring.py", "scripts/week6_robustness_demo.py",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_hashes() -> dict[str, str]:
    return {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in SOURCE_PATHS}


@dataclass
class VirtualClock:
    """Logical time for deterministic deadlines; no real sleeping is performed."""

    seconds: float = 1000.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


@dataclass
class SimulatedApplication:
    scenario: str
    directory: Path
    clock: VirtualClock
    completed: set[int] = field(default_factory=set)
    observation_counter: int = 0
    dispatches: list[dict[str, Any]] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)
    target_omitted: bool = False
    moved: bool = False

    def target_x(self, goal: int) -> int:
        return 350 if goal == 1 and self.moved else 130

    def event(self, kind: str, **value: Any) -> None:
        append_json(self.directory / "simulator_events.jsonl", {
            "event": kind, "virtual_seconds": self.clock(), **value,
        })


class SimulatedObserver:
    def __init__(self, app: SimulatedApplication, goal: int) -> None:
        self.app, self.goal = app, goal
        self.calls = 0

    def observe(self, *, observation_id: str | None = None) -> ObservationSnapshot:
        self.calls += 1
        self.app.clock.advance(.01)
        self.app.observation_counter += 1
        oid = observation_id or f"sim-obs-{self.app.observation_counter:04d}"
        omitted = bool(self.app.scenario == "target_missing_once" and self.goal == 1
                       and self.calls == 2 and not self.app.target_omitted)
        if omitted:
            self.app.target_omitted, self.app.moved = True, True
        labels = [f"Result{WORDS[index]}Ready" for index in sorted(self.app.completed)]
        if len(self.app.completed) == 3 and self.app.scenario != "fresh_final_check_fails":
            labels.append("WorkflowComplete")
        elements = []
        if not omitted:
            x = self.app.target_x(self.goal)
            box = BoundingBox(left=x - 55, top=70, right=x + 55, bottom=110)
            elements.append(ElementRef(element_id=f"{oid}-e000", text=f"Apply {WORDS[self.goal]}",
                                       bounding_box=box, center=box.center, confidence=.99))
        for index, label in enumerate(labels, start=1):
            box = BoundingBox(left=40, top=150 + index * 40, right=430, bottom=180 + index * 40)
            elements.append(ElementRef(element_id=f"{oid}-e{index:03d}", text=label,
                                       bounding_box=box, center=box.center, confidence=.99))
        result = ObservationSnapshot(
            observation_id=oid, captured_at=datetime.now(UTC),
            screen_info=ScreenInfo(screenshot_width=640, screenshot_height=480,
                                   control_width=640, control_height=480),
            elements=elements, ocr_engine="simulated_state_machine", image_path=None,
            notices=["SYNTHETIC backend: no screenshot capture or OCR occurred."],
        )
        self.app.event("observation", goal=self.goal + 1, goal_observe_call=self.calls,
                       snapshot=result.model_dump(mode="json"), target_omitted=omitted)
        return result


class SimulatedPlanner:
    def __init__(self, app: SimulatedApplication, goal: int) -> None:
        self.app, self.goal = app, goal
        self.calls = 0

    def plan(self, instruction: str, *, context: dict[str, Any] | None = None,
             image_path: str | None = None, task_id: str = "demo") -> PlanResult:
        self.calls += 1
        self.app.clock.advance(.02)
        request = {"goal": self.goal + 1, "goal_planning_call": self.calls,
                   "instruction": instruction, "context": context, "image_path": image_path,
                   "task_id": task_id, "backend": "deterministic_simulator"}
        self.app.plans.append(request)
        append_json(self.app.directory / "simulated_planning.jsonl", request)
        if self.app.scenario == "cancel_during_planning" and self.goal == 1:
            self.app.event("injected_keyboard_interrupt", goal=2)
            raise KeyboardInterrupt("deterministic cancellation, no input entered")
        plan = TaskPlan(
            task_id=task_id, instruction=instruction, summary="Synthetic one-button subgoal",
            requires_confirmation=False,
            steps=[PlanStep(step_id=f"apply-{self.calls}", action_type="click",
                            description=f"Apply synthetic phase {WORDS[self.goal]}",
                            target_text=f"Apply {WORDS[self.goal]}",
                            expected_result=f"Result{WORDS[self.goal]}Ready")],
        )
        append_json(self.app.directory / "simulated_plan_outputs.jsonl", {
            "goal": self.goal + 1, "goal_planning_call": self.calls,
            "plan": plan.model_dump(mode="json"),
        })
        return PlanResult(plan=plan, attempts=1, elapsed_ms=20,
                          metadata={"backend": "deterministic_simulator", "model_called": False})


class SimulatedExecutor:
    """It updates only this in-memory application; it has no GUI implementation."""

    def __init__(self, app: SimulatedApplication, goal: int) -> None:
        self.app, self.goal = app, goal
        self.calls = 0

    def execute(self, action: DesktopAction, *, dry_run: bool, screen: ScreenInfo) -> ActionResult:
        self.calls += 1
        self.app.clock.advance(.03)
        valid = (action.action_type == "click" and action.x == self.app.target_x(self.goal)
                 and action.y == 90 and screen.screenshot_width == 640)
        failed = bool(not dry_run and self.goal == 1 and (
            self.app.scenario == "persistent_execution_failure"
            or self.app.scenario == "execution_failure_once" and self.calls == 1
        ))
        uncertain = bool(not dry_run and self.goal == 1
                         and self.app.scenario == "executor_result_unknown")
        event = {"goal": self.goal + 1, "goal_executor_call": self.calls,
                 "action": action.model_dump(mode="json"), "dry_run": dry_run,
                 "coordinates_match_current_frame": valid,
                 "success": bool(valid and not failed and not uncertain),
                 "outcome_unconfirmed": uncertain}
        self.app.dispatches.append(event)
        append_json(self.app.directory / "simulated_executor.jsonl", event)
        if not valid:
            return ActionResult(success=False, dry_run=dry_run, action=action,
                                error="simulator rejected a stale or unexpected action")
        if uncertain:
            # Demonstrate why an exception cannot authorize automatic replay:
            # the simulated state changed, but no ActionResult reached the runner.
            self.app.completed.add(self.goal)
            self.app.event("simulated_state_changed_before_executor_exception", goal=self.goal + 1)
            raise RuntimeError("injected after simulated state change; outcome unconfirmed")
        if failed:
            self.app.moved = True
            return ActionResult(success=False, dry_run=False, action=action,
                                error="injected confirmed failure; simulated state unchanged")
        if not dry_run:
            self.app.completed.add(self.goal)
            if self.app.scenario == "task_time_exhausted" and self.goal == 0:
                self.app.clock.advance(2.5)
                self.app.event("injected_slow_executor_return", goal=1)
        return ActionResult(success=True, dry_run=dry_run, action=action,
                            metadata={"backend": "deterministic_simulator", "gui_input": False})


def specification(identifier: str, *, message: bool = False) -> WorkflowSpec:
    goals = [TaskSpec(case_id=f"phase_{word.casefold()}",
                      instruction=f"Apply synthetic phase {word} once",
                      expect_text=[f"Result{word}Ready"],
                      message_conversation="Synthetic Recipient" if message and index == 0 else None)
             for index, word in enumerate(WORDS)]
    return WorkflowSpec(workflow_id=identifier, instruction="Complete three synthetic phases",
                        subgoals=goals,
                        final_check=TaskSpec(case_id="final", instruction="Check fresh synthetic state",
                                             expect_text=["WorkflowComplete"]))


def expected_scenarios() -> dict[str, dict[str, Any]]:
    common = {"status": "succeeded", "total_actions": 3, "planning_calls": 3,
              "recovery_count": 0, "unconfirmed_action_attempts": 0,
              "subgoal_statuses": ["completed"] * 3, "final_verification": "passed",
              "simulated_successful_dispatches": 3, "simulated_executor_calls": 3,
              "first_goal_executor_calls": 1, "accounting_complete": True,
              "callback_completed_progress": [0, 0, 1, 1, 2, 2, 3]}
    failed_goals = ["completed", "failed", "not_started"]
    failure_progress = [0, 0, 1, 1, 1]
    return {
        "no_fault_three_subgoals": dict(common),
        "target_missing_once": {**common, "planning_calls": 4, "recovery_count": 1},
        "execution_failure_once": {**common, "total_actions": 4, "planning_calls": 4,
                                    "recovery_count": 1, "simulated_executor_calls": 4},
        "persistent_execution_failure": {
            **common, "status": "failed", "total_actions": 3, "planning_calls": 3,
            "recovery_count": 1, "subgoal_statuses": failed_goals,
            "final_verification": None, "simulated_successful_dispatches": 1,
            "callback_completed_progress": failure_progress},
        "shared_action_budget_exhausted": {
            **common, "status": "blocked", "total_actions": 2, "planning_calls": 2,
            "subgoal_statuses": ["completed", "completed", "not_started"],
            "final_verification": None, "simulated_successful_dispatches": 2,
            "simulated_executor_calls": 2, "callback_completed_progress": [0, 0, 1, 1, 2]},
        "task_time_exhausted": {
            **common, "status": "timed_out", "total_actions": 1, "planning_calls": 1,
            "subgoal_statuses": ["timed_out", "not_started", "not_started"],
            "final_verification": None, "simulated_successful_dispatches": 1,
            "simulated_executor_calls": 1, "callback_completed_progress": [0, 0, 0]},
        "cancel_during_planning": {
            **common, "status": "cancelled", "total_actions": 1, "planning_calls": 2,
            "subgoal_statuses": ["completed", "cancelled", "not_started"],
            "final_verification": None, "simulated_successful_dispatches": 1,
            "simulated_executor_calls": 1, "callback_completed_progress": failure_progress},
        "fresh_final_check_fails": {**common, "status": "failed", "final_verification": "failed"},
        "dry_run_is_not_success": {
            **common, "status": "dry_run_completed", "subgoal_statuses": ["dry_run_completed"] * 3,
            "final_verification": None, "simulated_successful_dispatches": 0,
            "callback_completed_progress": [0] * 7},
        "message_workflow_rejected": {
            **common, "status": "rejected", "total_actions": 0, "planning_calls": 0,
            "subgoal_statuses": ["not_started"] * 3, "final_verification": None,
            "simulated_successful_dispatches": 0, "simulated_executor_calls": 0,
            "first_goal_executor_calls": 0, "callback_completed_progress": []},
        "executor_result_unknown": {
            **common, "status": "failed", "total_actions": 1, "planning_calls": 2,
            "unconfirmed_action_attempts": 1, "subgoal_statuses": failed_goals,
            "final_verification": None, "simulated_successful_dispatches": 1,
            "simulated_executor_calls": 2, "accounting_complete": False,
            "callback_completed_progress": failure_progress},
    }


def run_scenario(batch: Path, identifier: str, expected: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    directory = batch / identifier
    directory.mkdir(exist_ok=False)
    clock = VirtualClock()
    app = SimulatedApplication(identifier, directory, clock)
    callback_events = []
    callback_visibility = []
    children: list[TaskRunner] = []
    result = None
    validation_error = None
    options = ExecutionOptions(
        execute=identifier != "dry_run_is_not_success", confirm=False,
        max_actions=2 if identifier == "shared_action_budget_exhausted" else 8,
        max_planning_attempts=3 if identifier == "persistent_execution_failure" else 8,
        task_timeout_seconds=2 if identifier == "task_time_exhausted" else 30,
        verification_timeout_seconds=0,
    )

    def factory(task: TaskSpec, index: int, child_directory: Path) -> TaskRunner:
        recorder = TaskRecorder(RunSession(child_directory.name, child_directory))
        runner = TaskRunner(observer=SimulatedObserver(app, index),
                            planner=SimulatedPlanner(app, index),
                            adapter=ActionAdapter(platform="win32"),
                            executor=SimulatedExecutor(app, index),
                            verifier=Verifier(poll_interval_seconds=0), recorder=recorder,
                            clock=clock, sleep=clock.advance)
        children.append(runner)
        return runner

    def on_checkpoint(event: dict[str, Any]) -> None:
        callback_events.append(event)
        visible = json.loads((directory / "workflow/workflow_summary.json").read_text(
            encoding="utf-8"))
        callback_visibility.append({"event": event, "visible_summary": visible,
                                    "counts_agree": event["completed"] == sum(
                                        item["status"] == "completed" for item in visible["subgoals"])
                                    and event["actions"] == visible["total_actions"]
                                    and event["planning_calls"] == visible["planning_calls"]})

    try:
        spec = specification(identifier, message=identifier == "message_workflow_rejected")
        write_json(directory / "specification.json", spec.model_dump(mode="json"))
        write_json(directory / "execution_options.json", options.model_dump(mode="json"))
        result = WorkflowRunner(directory / "workflow", factory, clock=clock).run(
            spec, options, on_checkpoint=on_checkpoint)
    except ValidationError as exc:
        if identifier != "message_workflow_rejected":
            raise
        validation_error = str(exc)
        write_json(directory / "rejected_specification.json", {
            "message_conversation": "Synthetic Recipient", "reason": validation_error,
            "executor_created": bool(children), "external_message_sent": False,
        })
    write_json(directory / "checkpoint_callbacks.json", callback_visibility)
    persisted_events = read_jsonl(directory / "workflow/checkpoints.jsonl")
    summaries = []
    steps = []
    frame_checks = []
    for child in children:
        summaries.append(json.loads((child.recorder.directory / "task_summary.json").read_text(
            encoding="utf-8")))
        for step in read_jsonl(child.recorder.directory / "steps.jsonl"):
            steps.append(step)
            resolved = step.get("resolved")
            if resolved and step.get("action_result") is not None:
                oid = step["observation_id"]
                frame_path = child.recorder.directory / f"{oid}.json"
                payload = json.loads(frame_path.read_text(encoding="utf-8"))
                frame_checks.append(resolved["element_id"] in {
                    item["element_id"] for item in payload["elements"]})
    recovery = [entry for summary in summaries for entry in summary["recovery_records"]]
    actual = {
        "status": result.status if result else "rejected",
        "total_actions": result.total_actions if result else 0,
        "planning_calls": result.planning_calls if result else 0,
        "recovery_count": len(recovery),
        "unconfirmed_action_attempts": result.unconfirmed_action_attempts if result else 0,
        "subgoal_statuses": [goal.status for goal in result.subgoals] if result else ["not_started"] * 3,
        "final_verification": result.final_verification.outcome if result and result.final_verification else None,
        "simulated_successful_dispatches": sum(
            row["success"] and not row["dry_run"] for row in app.dispatches),
        "simulated_executor_calls": len(app.dispatches),
        "first_goal_executor_calls": sum(row["goal"] == 1 for row in app.dispatches),
        "accounting_complete": result.accounting_complete if result else True,
        "callback_completed_progress": [row["completed"] for row in callback_events],
    }
    checks = [{"name": name, "expected": value, "actual": actual[name],
               "passed": value == actual[name]} for name, value in expected.items()]

    def check(name: str, value: bool, detail: Any = None) -> None:
        checks.append({"name": name, "expected": True, "actual": value,
                       "passed": value, "detail": detail})

    check("every_executor_coordinate_matches_current_frame", all(
        row["coordinates_match_current_frame"] for row in app.dispatches))
    check("resolved_ids_belong_to_recorded_dispatch_frame", all(frame_checks),
          {"resolved_frames_checked": len(frame_checks)})
    check("callback_observes_already_written_progress", all(
        row["counts_agree"] for row in callback_visibility))
    check("callback_events_match_durable_journal_prefix",
          callback_events == persisted_events[:-1] if result else not persisted_events)
    check("checkpoint_sequences_are_contiguous", [row["sequence"] for row in persisted_events]
          == list(range(1, len(persisted_events) + 1)))
    if result:
        saved = json.loads((directory / "workflow/workflow_summary.json").read_text(encoding="utf-8"))
        check("returned_result_equals_durable_summary", saved == result.model_dump(mode="json"))
        check("terminal_checkpoint_matches_result", bool(persisted_events)
              and persisted_events[-1]["status"] == result.status)
    if result and result.final_observation_id:
        final_number = int(result.final_observation_id.rsplit("-", 1)[1])
        earlier_numbers = [int(summary["verification"]["evidence"]["observation_id"].rsplit("-", 1)[1])
                           for summary in summaries if summary.get("verification")
                           and "observation_id" in summary["verification"]["evidence"]]
        check("overall_check_uses_new_frame_after_all_child_checks",
              bool(earlier_numbers) and final_number > max(earlier_numbers))
    if identifier == "target_missing_once":
        check("missing_target_refused_without_dispatch", any(
            step.get("action_result") is None and "no element matches" in (step.get("error") or "")
            for step in steps))
        check("recovery_feedback_cites_target_resolution", len(recovery) == 1
              and recovery[0]["feedback"]["reason"] == "target_resolution")
        check("recovered_click_uses_moved_target", any(
            row["goal"] == 2 and row["action"]["x"] == 350 for row in app.dispatches))
    if identifier == "persistent_execution_failure":
        check("shared_planning_budget_reached", actual["planning_calls"] == options.max_planning_attempts)
    if identifier == "executor_result_unknown":
        check("unknown_dispatch_not_replanned_or_replayed", actual["recovery_count"] == 0
              and len([row for row in app.dispatches if row["goal"] == 2]) == 1)
    if identifier == "message_workflow_rejected":
        check("message_rejected_before_any_child_created", bool(validation_error)
              and "message sending is not supported" in validation_error and not children)
    if identifier == "dry_run_is_not_success":
        check("dry_run_leaves_simulated_state_untouched", not app.completed)
    report = {"scenario": identifier, "expected": expected, "actual": actual,
              "checks": checks, "passed": all(row["passed"] for row in checks),
              "wall_elapsed_ms": (time.perf_counter() - started) * 1000,
              "virtual_elapsed_ms": (clock() - 1000) * 1000,
              "evidence_directory": str(directory.resolve()),
              "recovery_records": recovery, "workflow_result": result.model_dump(mode="json") if result else None}
    write_json(directory / "scenario_verdict.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="New batch directory; an existing directory is refused")
    args = parser.parse_args()
    destination = args.output_dir or (ROOT / "Document/Week6/results" / (
        "robustness_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]))
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before = source_hashes()
    # Preserve the actual files used for this batch, including the uncommitted
    # demonstration driver. A later edit must not erase a failed run's provenance.
    for relative in SOURCE_PATHS:
        copied = destination / "source_snapshot" / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((ROOT / relative).read_bytes())
    cases = []
    for identifier, expected in expected_scenarios().items():
        try:
            cases.append(run_scenario(destination, identifier, expected))
        except Exception as exc:  # noqa: BLE001 - preserve each failure and continue other cases
            failed = {"scenario": identifier, "passed": False,
                      "unexpected_exception": f"{type(exc).__name__}: {exc}"}
            write_json(destination / identifier / "unexpected_failure.json", failed)
            cases.append(failed)
    after = source_hashes()
    summary = {
        "schema": "week6.robustness.integration_demo.v1",
        "created_at_utc": datetime.now(UTC).isoformat(), "batch_directory": str(destination),
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform()},
        "production_components": ["TaskRunner", "ActionAdapter", "Verifier", "TaskRecorder", "WorkflowRunner"],
        "injected_backends": ["SimulatedObserver", "SimulatedPlanner", "SimulatedExecutor", "VirtualClock"],
        "gpu_ml_modules_loaded": [name for name in ("torch", "paddle", "transformers")
                                  if sys.modules.get(name) is not None],
        "limits": ["Deterministic simulated application, not real desktop or model acceptance.",
                   "No capture, OCR, GPU inference, GUI input, external messages, or real waiting.",
                   "Virtual durations exercise deadlines; wall durations measure only this demonstration.",
                   "total_actions counts confirmed ActionResults, including failed/dry-run validation calls.",
                   "Unconfirmed executor outcomes are separate and never credited as confirmed actions.",
                   "Every batch is new; histories are neither resumed nor overwritten."],
        "source_hashes_before": before, "source_hashes_after": after,
        "source_unchanged_during_run": before == after,
        "scenarios": cases, "scenario_count": len(cases),
        "scenario_passed": sum(row["passed"] for row in cases),
        "all_passed": all(row["passed"] for row in cases) and before == after,
        "wall_elapsed_ms": (time.perf_counter() - started) * 1000,
    }
    write_json(destination / "summary.json", summary)
    print(json.dumps({"summary": str(destination / "summary.json"),
                      "scenario_passed": summary["scenario_passed"],
                      "scenario_count": summary["scenario_count"],
                      "source_unchanged_during_run": before == after,
                      "all_passed": summary["all_passed"]}, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
