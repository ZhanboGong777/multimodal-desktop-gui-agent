"""Verified subgoals sharing one execution budget, with no log-driven resume.

The caller supplies task checks, not a model-generated claim of completion. Each
subgoal uses the existing TaskRunner in a new directory. A workflow never replays
an earlier subgoal, and a final fresh, read-only task check decides overall success.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from ..schemas import SchemaModel
from .schemas import ExecutionOptions, TaskRunResult, TaskSpec, VerificationResult

MAX_SUBGOALS = 32
_REPLACE_RETRY_DELAYS = (0.005, 0.01, 0.02)
_SAFE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)),
                       *(f"lpt{i}" for i in range(10))})


def _safe_id(value: str) -> str:
    if not _SAFE_ID.fullmatch(value) or value.casefold() in _RESERVED:
        raise ValueError("ids must be safe ASCII names, not paths or reserved device names")
    return value


def _check_task(task: TaskSpec) -> None:
    _safe_id(task.case_id)
    if not task.instruction.strip():
        raise ValueError("a subgoal instruction must not be empty")
    checks = (*task.expect_text, *task.forbid_text)
    if not any(value.strip() for value in checks):
        raise ValueError("each subgoal and final_check needs an explicit expect/forbid text check")
    if any(not value.strip() for value in checks):
        raise ValueError("empty text checks cannot verify a workflow")
    # Existing message tasks have separate authorization and no-retry rules.
    # They cannot be introduced indirectly through this non-message workflow.
    if task.message_conversation is not None:
        raise ValueError("message sending is not supported by workflows")


class WorkflowSpec(SchemaModel):
    workflow_id: str
    instruction: str
    subgoals: list[TaskSpec] = Field(min_length=1, max_length=MAX_SUBGOALS)
    final_check: TaskSpec

    @field_validator("workflow_id")
    @classmethod
    def _identifier(cls, value: str) -> str:
        return _safe_id(value)

    @field_validator("instruction")
    @classmethod
    def _instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("workflow instruction must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def _tasks(self) -> WorkflowSpec:
        ids = set()
        for task in self.subgoals:
            _check_task(task)
            normalized = task.case_id.casefold()
            if normalized in ids:
                raise ValueError("subgoal case_id values must be unique ignoring case")
            ids.add(normalized)
        _check_task(self.final_check)
        if (self.final_check.must_exit_processes or self.final_check.requires_processes
                or self.final_check.forbids_processes):
            raise ValueError("final_check supports text checks only, not process checks")
        return self


GoalStatus = Literal[
    "not_started", "running", "completed", "dry_run_completed", "failed",
    "blocked", "cancelled", "timed_out",
]
WorkflowStatus = Literal[
    "running", "succeeded", "dry_run_completed", "failed", "blocked", "cancelled", "timed_out",
]


class SubgoalOutcome(SchemaModel):
    index: int = Field(ge=0)
    case_id: str
    status: GoalStatus = "not_started"
    child_status: str | None = None
    actions: int = Field(default=0, ge=0)
    unconfirmed_action_attempts: int = Field(default=0, ge=0)
    planning_calls: int = Field(default=0, ge=0)
    verification: VerificationResult | None = None
    elapsed_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    evidence_directory: str = ""
    summary_reference: str = ""
    error: str | None = None


class WorkflowRunResult(SchemaModel):
    schema_version: Literal["week6.workflow.v1"] = "week6.workflow.v1"
    workflow_id: str
    status: WorkflowStatus
    execute: bool
    subgoals: list[SubgoalOutcome]
    total_actions: int = Field(default=0, ge=0)
    unconfirmed_action_attempts: int = Field(default=0, ge=0)
    planning_calls: int = Field(default=0, ge=0)
    accounting_complete: bool = True
    elapsed_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    final_verification: VerificationResult | None = None
    final_observation_id: str | None = None
    final_observation_reference: str = ""
    evidence_directory: str
    summary_reference: str
    checkpoint_reference: str
    error: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


class ChildRunner(Protocol):
    observer: Any
    verifier: Any
    recorder: Any

    def run(self, task: TaskSpec, options: ExecutionOptions, **callbacks: Any) -> TaskRunResult: ...


class WorkflowRunner:
    """Run once in an exclusively created directory.

    ``runner_factory(task, zero_based_index, child_directory)`` receives an already
    created, empty directory. ``on_checkpoint`` receives only state/count fields;
    the other callbacks are forwarded to the existing child's ``run`` unchanged.
    Existing directories are refused, even if their logs are complete or damaged.
    """

    def __init__(
        self,
        directory: str | Path,
        runner_factory: Callable[[TaskSpec, int, Path], ChildRunner],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = Path(directory)
        self.runner_factory = runner_factory
        self.clock = clock
        self._sequence = 0

    def _elapsed(self, started: float) -> float:
        elapsed = (self.clock() - started) * 1000
        if not math.isfinite(elapsed):
            raise ValueError("workflow clock must be finite")
        return max(0.0, elapsed)

    def _write_summary(self, result: WorkflowRunResult) -> None:
        target = self.directory / "workflow_summary.json"
        temporary = self.directory / f".workflow_summary.{uuid4().hex}.tmp"
        try:
            temporary.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
            # Windows readers or scanners can briefly deny atomic replacement.
            # Retry only this file operation with the same complete temporary;
            # never append another checkpoint or repeat a child action.
            for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
                try:
                    os.replace(temporary, target)
                    break
                except PermissionError:
                    if attempt == len(_REPLACE_RETRY_DELAYS):
                        raise
                    time.sleep(_REPLACE_RETRY_DELAYS[attempt])
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A still-locked temporary stays intact; cleanup must not mask
                # the original replacement failure or damage the old summary.
                pass

    def _checkpoint(
        self, result: WorkflowRunResult, index: int | None, callback: Callable | None,
    ) -> None:
        self._sequence += 1
        event = {
            "sequence": self._sequence,
            "case_id": result.subgoals[index].case_id if index is not None else None,
            "index": index,
            "status": result.subgoals[index].status if index is not None else result.status,
            "actions": result.total_actions,
            "planning_calls": result.planning_calls,
            "completed": sum(goal.status == "completed" for goal in result.subgoals),
            "total": len(result.subgoals),
            "elapsed_ms": result.elapsed_ms,
        }
        with (self.directory / "checkpoints.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            path = self.directory / "checkpoints.jsonl"
            if path.stat().st_size:
                with path.open("rb") as previous:
                    previous.seek(-1, os.SEEK_END)
                    if previous.read(1) != b"\n":
                        # Preserve partial bytes while separating a new complete
                        # event after an interrupted append.
                        handle.write("\n")
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._write_summary(result)
        if callback is not None:
            callback(dict(event))

    @staticmethod
    def _read_interrupted_counts(childdir: Path, case_id: str) -> tuple[int, int, int]:
        """Recover confirmed counts only; this never resumes a child or trusts success."""
        actions = calls = unconfirmed = 0
        summary = childdir / "task_summary.json"
        if summary.exists():
            try:
                payload = json.loads(summary.read_bytes())
                if not isinstance(payload, dict):
                    raise TypeError("child summary must be an object")
                payload.pop("action_count", None)  # recorder adds this derived display field
                child = TaskRunResult.model_validate(payload)
                if child.case_id == case_id:
                    actions, calls = child.action_count, child.planning_calls
                    unconfirmed = child.unconfirmed_action_attempts
            except (OSError, ValueError, TypeError):
                pass
        steps = childdir / "steps.jsonl"
        if steps.exists():
            try:
                complete = []
                for line in steps.read_bytes().splitlines():
                    try:
                        value = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(value, dict) and value.get("action_result") is not None:
                        complete.append(value)
                actions = max(actions, len(complete))
            except OSError:
                pass
        return max(0, actions), max(0, calls), max(0, unconfirmed)

    def run(
        self, spec: WorkflowSpec, options: ExecutionOptions, *,
        absolute_deadline: float | None = None, **callbacks: Any,
    ) -> WorkflowRunResult:
        spec = WorkflowSpec.model_validate(spec.model_dump())
        options = ExecutionOptions.model_validate(options.model_dump())
        if not math.isfinite(options.task_timeout_seconds):
            raise ValueError("workflow time budget must be finite")
        if absolute_deadline is not None and not math.isfinite(absolute_deadline):
            raise ValueError("workflow absolute deadline must be finite")
        child_callbacks = dict(callbacks)
        checkpoint_callback = child_callbacks.pop("on_checkpoint", None)
        unknown = child_callbacks.keys() - {"confirm", "countdown", "high_risk_confirm", "on_plan"}
        if unknown:
            raise TypeError(f"unknown workflow callbacks: {', '.join(sorted(unknown))}")
        started = self.clock()
        if not math.isfinite(started):
            raise ValueError("workflow clock must be finite")
        deadline = started + options.task_timeout_seconds
        if absolute_deadline is not None:
            # An enclosing automatic workflow owns its original deadline. Time
            # spent scheduling or preparing this child cannot renew that budget.
            deadline = min(deadline, absolute_deadline)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._sequence = 0
        result = WorkflowRunResult(
            workflow_id=spec.workflow_id, status="running", execute=options.execute,
            subgoals=[SubgoalOutcome(index=i, case_id=goal.case_id)
                      for i, goal in enumerate(spec.subgoals)],
            evidence_directory=str(self.directory.resolve()),
            summary_reference=str((self.directory / "workflow_summary.json").resolve()),
            checkpoint_reference=str((self.directory / "checkpoints.jsonl").resolve()),
        )
        active: int | None = None
        childdir: Path | None = None
        child_inflight = False
        last_runner: ChildRunner | None = None
        try:
            self._checkpoint(result, None, checkpoint_callback)
            for index, task in enumerate(spec.subgoals):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    result.status, result.error = "timed_out", "workflow deadline exhausted"
                    break
                if result.total_actions >= options.max_actions:
                    result.status, result.error = "blocked", "workflow action budget exhausted"
                    break
                if result.planning_calls >= options.max_planning_attempts:
                    result.status, result.error = "blocked", "workflow planning budget exhausted"
                    break
                active = index
                goal = result.subgoals[index]
                goal.status = "running"
                childdir = self.directory / f"{index + 1:03d}_{task.case_id}"
                childdir.mkdir(exist_ok=False)
                goal.evidence_directory = str(childdir.resolve())
                goal.summary_reference = str((childdir / "task_summary.json").resolve())
                result.elapsed_ms = self._elapsed(started)
                self._checkpoint(result, index, checkpoint_callback)
                child = self.runner_factory(task.model_copy(deep=True), index, childdir)
                last_runner = child
                remaining = deadline - self.clock()
                if remaining <= 0:
                    goal.status = "timed_out"
                    result.status, result.error = "timed_out", "deadline exhausted during child setup"
                    break
                child_options = options.model_copy(update={
                    "max_actions": options.max_actions - result.total_actions,
                    "max_planning_attempts": options.max_planning_attempts - result.planning_calls,
                    "task_timeout_seconds": remaining,
                })
                child_inflight = True
                outcome = child.run(task.model_copy(deep=True), child_options, **child_callbacks)
                if not isinstance(outcome, TaskRunResult) or outcome.case_id != task.case_id:
                    raise ValueError("child returned an invalid or unrelated task result")
                if outcome.planning_calls < 0:
                    raise ValueError("child planning counts cannot be negative")
                child_inflight = False
                goal.child_status = outcome.status
                goal.actions = outcome.action_count
                goal.unconfirmed_action_attempts = outcome.unconfirmed_action_attempts
                goal.planning_calls = outcome.planning_calls
                goal.verification = outcome.verification
                goal.elapsed_ms = max(0.0, outcome.elapsed_ms)
                result.total_actions += goal.actions
                result.unconfirmed_action_attempts += goal.unconfirmed_action_attempts
                result.planning_calls += goal.planning_calls
                result.elapsed_ms = self._elapsed(started)
                if goal.unconfirmed_action_attempts:
                    result.accounting_complete = False
                    result.notes.append(
                        "A child reports unconfirmed dispatch attempts; total_actions is a confirmed "
                        "lower bound. Later subgoals will not run."
                    )
                if (result.total_actions > options.max_actions
                        or result.planning_calls > options.max_planning_attempts):
                    goal.status = "failed"
                    result.status, result.error = "failed", "child exceeded the shared workflow budget"
                elif self.clock() > deadline:
                    goal.status = "timed_out"
                    result.status, result.error = "timed_out", "child returned after workflow deadline"
                elif goal.unconfirmed_action_attempts:
                    goal.status = outcome.status if outcome.status in {
                        "blocked", "cancelled", "timed_out",
                    } else "failed"
                    result.status = goal.status
                    result.error = "child dispatch outcome is unconfirmed; no further execution"
                elif (options.execute and outcome.execute and outcome.status == "succeeded"
                      and outcome.verification is not None and outcome.verification.passed):
                    goal.status = "completed"
                elif not options.execute and outcome.status == "dry_run_completed":
                    goal.status = "dry_run_completed"
                else:
                    goal.status = outcome.status if outcome.status in {
                        "blocked", "cancelled", "timed_out",
                    } else "failed"
                    result.status = goal.status
                    result.error = "subgoal did not finish with the required task verification"
                self._checkpoint(result, index, checkpoint_callback)
                if goal.status not in {"completed", "dry_run_completed"}:
                    break
                active = None
            else:
                if not options.execute:
                    result.status = "dry_run_completed"
                elif self.clock() >= deadline:
                    result.status, result.error = "timed_out", "no time remains for final verification"
                else:
                    assert last_runner is not None
                    # No planner/executor call is made after the last child. A new
                    # frame is mandatory even if all subgoal checks just passed.
                    frame = last_runner.observer.observe()
                    result.final_observation_id = frame.observation_id
                    path = last_runner.recorder.save_observation(frame)
                    result.final_observation_reference = str(Path(path).resolve())
                    if self.clock() >= deadline:
                        result.status, result.error = "timed_out", "deadline expired during final observation"
                    else:
                        result.final_verification = (
                            VerificationResult(
                                outcome="inconclusive", detail="final observation has perception errors",
                                evidence={"observation_id": frame.observation_id},
                            ) if frame.errors else last_runner.verifier.check_task(spec.final_check, frame)
                        )
                        if self.clock() > deadline:
                            result.status, result.error = "timed_out", "deadline expired during final verification"
                        elif result.final_verification.passed:
                            result.status = "succeeded"
                        else:
                            result.status, result.error = "failed", "final workflow verification did not pass"
        except (KeyboardInterrupt, Exception) as exc:  # noqa: BLE001 - preserve completed evidence
            result.status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            # Free-form exceptions can contain typed text or model output. The
            # workflow summary retains the class, while child evidence stays local.
            result.error = f"workflow interrupted by {type(exc).__name__}"
            if active is not None:
                goal = result.subgoals[active]
                if goal.status == "running":
                    goal.status = result.status
                if child_inflight and childdir is not None:
                    goal.actions, goal.planning_calls, goal.unconfirmed_action_attempts = (
                        self._read_interrupted_counts(childdir, goal.case_id)
                    )
                    result.total_actions += goal.actions
                    result.planning_calls += goal.planning_calls
                    result.unconfirmed_action_attempts += goal.unconfirmed_action_attempts
                    result.accounting_complete = False
                    result.notes.append(
                        "Interrupted child counts are confirmed lower bounds; complete counts are unknown."
                    )
        finally:
            result.elapsed_ms = self._elapsed(started)
            # A final snapshot retains completed subgoals even if cancellation
            # arrived while writing the previous checkpoint or invoking a callback.
            self._write_summary(result)
            self._checkpoint(result, None, None)
        return result
