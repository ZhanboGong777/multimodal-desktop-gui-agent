"""The finite task loop: observe, plan, act, verify, repeat — within limits.

Design decisions worth stating, because each rules out an easier wrong version:

* **One plan, then step-by-step re-observation.** The model is not called once per
  click. Re-observing is cheap; re-planning is not, and a plan that is re-derived
  every step cannot be shown to the user before it runs.
* **No step is trusted because it is next in the plan.** Before each action the
  target is resolved against the *current* frame, so a plan made from an old
  screenshot cannot click through to a page that has since changed.
* **Screenshots changing is not success.** Only the task verifier can return
  ``succeeded``, and only when the task's own success rules match the screen.
* **Dry-run reports what it would do and stops.** A dry run that resolves every
  step but dispatches nothing must not be summarised as a finished task.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from ..control.executor import ActionExecutor
from ..planning import PlanResult, TaskPlan, TaskPlanner
from .action_adapter import ActionAdapter, ActionResolutionError
from .recorder import TaskRecorder, redact
from .schemas import (
    ExecutionOptions,
    ObservationSnapshot,
    StepRecord,
    TaskRunResult,
    TaskSpec,
    VerificationResult,
)
from .verification import Verifier


class Observer(Protocol):
    def observe(self, *, observation_id: str | None = None) -> ObservationSnapshot: ...


class Planner(Protocol):
    def plan(
        self,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
        image_path: str | None = None,
        task_id: str = ...,
    ) -> PlanResult: ...


#: Providers each describe an over-long prompt in their own words; there is no
#: shared error code for it. Matching on the wording only ever *adds* advice to a
#: failure that has already happened, so a miss costs nothing.
_CONTEXT_OVERFLOW = re.compile(
    r"context (?:size|length|window)|context_length_exceeded|maximum context", re.IGNORECASE
)


def explain_model_failure(message: str) -> str:
    """Append the deployment fix when a failure looks like a context overflow.

    A full-screen screenshot plus the element list measured 7 517 tokens on a
    2560x1600 display, and Ollama serves 4096 by default. The provider's error
    names the numbers but not what to do about them.
    """
    if not _CONTEXT_OVERFLOW.search(message):
        return message
    return (
        f"{message}\n  hint: the screenshot plus the element list did not fit the "
        "server's context window. Raise it before retrying "
        "(Ollama: OLLAMA_CONTEXT_LENGTH=16384), or lower agent.max_elements."
    )


#: Risk levels that ask for a second, separate confirmation before the first real
#: action. They are not reversible by looking at the screen afterwards: a message
#: has been sent, or a window with unsaved content has been closed.
_RISKY_RISKS = frozenset({"medium", "high"})


class RunnerError(RuntimeError):
    """Raised when the run cannot even start."""


class TaskRunner:
    """Executes one task under explicit limits."""

    def __init__(
        self,
        *,
        observer: Observer,
        planner: TaskPlanner | Planner,
        adapter: ActionAdapter,
        executor: ActionExecutor,
        verifier: Verifier,
        recorder: TaskRecorder,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.observer = observer
        self.planner = planner
        self.adapter = adapter
        self.executor = executor
        self.verifier = verifier
        self.recorder = recorder
        self.clock = clock
        self.sleep = sleep

    # ── entry point ────────────────────────────────────────────────────
    def run(
        self,
        task: TaskSpec,
        options: ExecutionOptions,
        *,
        confirm: Callable[[TaskPlan], bool] | None = None,
        countdown: Callable[[int], None] | None = None,
        high_risk_confirm: Callable[[TaskPlan], bool] | None = None,
    ) -> TaskRunResult:
        started = self.clock()
        notes: list[str] = []

        if not task.success_rules and not task.expect_text and not task.forbid_text:
            return self._blocked(task, options, "the task defines no verifiable success rule")

        # 1. first look at the screen
        try:
            initial = self.observer.observe()
        except Exception as exc:  # noqa: BLE001 - a failed first look ends the run
            return self._blocked(task, options, f"initial observation failed: {exc}")
        self.recorder.save_observation(initial)

        # A frame with no readable text is not an error - OCR cannot read a locked,
        # dark or mostly-empty screen - but it makes every text target unresolvable.
        # Saying so here turns a bare "no element matches '...'" into something the
        # operator can act on.
        if not any(item.text.strip() for item in initial.elements):
            notes.append(
                "the first frame had no readable text: OCR returned no labels, so no "
                "text target can resolve against it"
            )

        # 1b. A real run must not start from a screen where the goal already holds.
        # T01 and T05 are both satisfiable by doing nothing: a browser that was
        # already open carries the text T01 looks for, and T05's rule only asks that
        # the marker be gone, so a window that was never opened - or was minimised -
        # satisfies it. A run cannot be credited with a state it did not create.
        #
        # A dry run is exempt on purpose: it dispatches nothing and its verdict is
        # already forced to inconclusive, so the guard would only stop the pipeline
        # check it exists to perform.
        #
        # Only a task that declares preconditions is checked. Declaring them is how
        # the spec says "this task assumes a starting state"; a task that declares
        # none is making no such assumption.
        if options.execute and options.require_preconditions and task.preconditions:
            precondition = self.verifier.check_task(task, initial)
            if precondition.outcome == "passed":
                return self._blocked(
                    task,
                    options,
                    "the success rule already holds on the untouched screen "
                    f"({precondition.detail}); this run cannot be credited with it",
                )

        # 2. plan from that observation
        plan_result = self.planner.plan(
            task.instruction,
            context=self._context(initial, task, options),
            image_path=initial.image_path,
            task_id=task.case_id,
        )
        if not plan_result.ok or plan_result.plan is None:
            return self._blocked(
                task,
                options,
                f"planning failed: {explain_model_failure(plan_result.error or 'no plan produced')}",
            )
        plan = plan_result.plan
        notes.append(f"planned {len(plan.steps)} steps from {initial.observation_id}")

        # 3. budget check before anything is dispatched
        if len(plan.executable_steps) > options.max_actions:
            return self._blocked(
                task,
                options,
                f"the plan wants {len(plan.executable_steps)} actions, above the "
                f"{options.max_actions} allowed",
            )

        # 4. confirmation gate — before any real input event
        if options.execute:
            if options.confirm and confirm is not None and not confirm(plan):
                result = self._finish(task, options, "cancelled", started, notes)
                result.verification = None
                return result
            # A risky task gets a second, separate confirmation. The policy lives
            # here, next to the risk it is about, rather than in each caller: a
            # caller that forgets to ask is the failure this gate exists to stop,
            # and one already did - the CLI built the prompt and never passed it.
            if (
                options.confirm
                and high_risk_confirm is not None
                and task.risk in _RISKY_RISKS
                and not high_risk_confirm(plan)
            ):
                result = self._finish(task, options, "cancelled", started, notes)
                result.verification = None
                return result
            if countdown is not None:
                countdown(3)

        return self._execute_plan(task, plan, options, initial, started, notes)

    # ── the loop ───────────────────────────────────────────────────────
    def _execute_plan(
        self,
        task: TaskSpec,
        plan: TaskPlan,
        options: ExecutionOptions,
        initial: ObservationSnapshot,
        started: float,
        notes: list[str],
    ) -> TaskRunResult:
        steps: list[StepRecord] = []
        current = initial

        for index, step in enumerate(plan.steps, start=1):
            if self.clock() - started > options.task_timeout_seconds:
                return self._finish(task, options, "timed_out", started, notes, steps)

            record = StepRecord(
                index=index,
                step_id=step.step_id,
                description=step.description,
                action_type=step.action_type,
                observation_id=current.observation_id,
            )
            step_started = self.clock()

            # finish is a planning verb: it closes the plan, it never dispatches
            if step.action_type == "finish":
                record.error = None
                record.elapsed_ms = (self.clock() - step_started) * 1000
                steps.append(record)
                self.recorder.append_step(record)
                notes.append(f"{step.step_id}: finish reached")
                break

            # fresh look before acting; the plan may be minutes old
            try:
                before = self.observer.observe()
                self.recorder.save_observation(before)
            except Exception as exc:  # noqa: BLE001
                record.error = f"observation failed: {exc}"
                record.elapsed_ms = (self.clock() - step_started) * 1000
                steps.append(record)
                self.recorder.append_step(record)
                return self._finish(task, options, "failed", started, notes, steps)

            record.observation_id = before.observation_id

            try:
                resolved = self.adapter.resolve(step, before)
            except ActionResolutionError as exc:
                record.error = str(exc)
                record.elapsed_ms = (self.clock() - step_started) * 1000
                steps.append(record)
                self.recorder.append_step(record)
                notes.append(f"{step.step_id}: {exc}")
                return self._finish(task, options, "failed", started, notes, steps)

            record.resolved = resolved
            action_result = self.executor.execute(
                resolved.action, dry_run=not options.execute, screen=before.screen_info
            )
            record.action_result = {
                "success": action_result.success,
                "dry_run": action_result.dry_run,
                "error": action_result.error,
                "action": redact(resolved.action.model_dump(mode="json")),
            }

            # re-observe only when something was actually dispatched
            after: ObservationSnapshot | None = None
            if options.execute and action_result.success:
                self.sleep(0.2)
                try:
                    after = self.observer.observe()
                    self.recorder.save_observation(after)
                    record.after_observation_id = after.observation_id
                except Exception as exc:  # noqa: BLE001
                    record.error = f"post-action observation failed: {exc}"

            record.verification = self.verifier.check_step(
                expected_result=step.expected_result,
                before=before,
                after=after if after is not None else (before if not options.execute else None),
                action_result=action_result,
            )
            record.elapsed_ms = (self.clock() - step_started) * 1000
            steps.append(record)
            self.recorder.append_step(record)

            if not action_result.success:
                notes.append(f"{step.step_id}: action failed")
                return self._finish(task, options, "failed", started, notes, steps)

            if after is not None:
                current = after

        # ── the plan is done; only the task verifier may call it a success ──
        if not options.execute:
            result = self._finish(task, options, "dry_run_completed", started, notes, steps)
            # A dry run dispatched nothing, so the task rule cannot have been met -
            # but reporting it as "failed" would say the run went wrong, and it did
            # not. The rule's own verdict is kept as evidence, not as the outcome.
            would_be = self.verifier.check_task(task, current)
            result.verification = VerificationResult(
                outcome="inconclusive",
                method=task.verification,
                detail=(
                    "dry run: actions were resolved and validated but nothing was dispatched, "
                    f"so the task goal was not attempted (the rule would read {would_be.outcome})"
                ),
                evidence={"would_be": would_be.outcome, "rule_detail": would_be.detail},
            )
            return result

        verification, final = self.verifier.check_task_with_polling(
            task,
            self._observing,
            deadline_seconds=options.verification_timeout_seconds,
            sleep=self.sleep,
            clock=self.clock,
        )
        status = "succeeded" if verification.passed else "failed"
        if not verification.passed:
            notes.append(f"task verification: {verification.detail}")
        result = self._finish(task, options, status, started, notes, steps)
        result.verification = verification
        if final is not None:
            self.recorder.save_observation(final)
        return result

    # ── helpers ────────────────────────────────────────────────────────
    def _observing(self) -> ObservationSnapshot:
        snapshot = self.observer.observe()
        self.recorder.save_observation(snapshot)
        return snapshot

    def _context(
        self, snapshot: ObservationSnapshot, task: TaskSpec, options: ExecutionOptions
    ) -> dict[str, Any]:
        from .observation import describe_elements

        return {
            "platform": self.adapter.platform,
            "observation_id": snapshot.observation_id,
            "screen": (
                f"{snapshot.screen_info.screenshot_width}x{snapshot.screen_info.screenshot_height}"
            ),
            "target_app": task.target_app,
            # A plan that overshoots the budget is refused before anything is
            # dispatched, so the model is told the budget it is planning against
            # rather than discovering it by having its plan rejected.
            "limits": (
                f"at most {options.max_actions} actions and "
                f"{options.task_timeout_seconds:g} s for the whole task"
            ),
            "visible_text": describe_elements(snapshot),
            "success_rules": task.success_rules,
        }

    def _finish(
        self,
        task: TaskSpec,
        options: ExecutionOptions,
        status: str,
        started: float,
        notes: list[str],
        steps: list[StepRecord] | None = None,
    ) -> TaskRunResult:
        result = TaskRunResult(
            run_id=self.recorder.run_id,
            case_id=task.case_id,
            instruction=task.instruction,
            status=status,  # type: ignore[arg-type]
            steps=steps or [],
            elapsed_ms=(self.clock() - started) * 1000.0,
            execute=options.execute,
            model_name=getattr(getattr(self.planner, "client", None), "model_name", ""),
            provider=getattr(getattr(self.planner, "client", None), "name", ""),
            notes=notes,
        )
        self.recorder.write_summary(result)
        return result

    def _blocked(self, task: TaskSpec, options: ExecutionOptions, reason: str) -> TaskRunResult:
        return self._finish(task, options, "blocked", self.clock(), [reason])
