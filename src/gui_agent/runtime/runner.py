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

import contextlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from PIL import Image

from ..control.executor import ActionExecutor
from ..planning import PlanResult, TaskPlan, TaskPlanner
from ..planning.schemas import PlanStep
from ..schemas import ActionResult, BoundingBox
from . import provenance
from .action_adapter import ActionAdapter, ActionResolutionError, keys_for_platform
from .cursor_parking import plan_cursor_parking, verify_parking_frame
from .processes import known as known_processes
from .processes import match as match_processes
from .recorder import TaskRecorder, redact
from .recovery import build_recovery_feedback, remaining_milliseconds
from .schemas import (
    ElementRef,
    ExecutionOptions,
    ObservationSnapshot,
    RecoveryFeedback,
    RecoveryRecord,
    ResolvedAction,
    StepRecord,
    TaskRunResult,
    TaskSpec,
    VerificationResult,
)
from .screenshot_quality import assess_screenshot_quality, window_identity
from .target_grounding import TargetGrounder
from .verification import Verifier
from .visual_grounding import GroundingError, match_candidate, validate_frames


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
        "(Ollama: OLLAMA_CONTEXT_LENGTH=16384), or lower execution.max_elements."
    )


#: Risk levels that ask for a second, separate confirmation before the first real
#: action. They are not reversible by looking at the screen afterwards: a message
#: has been sent, or a window with unsaved content has been closed.
_RISKY_RISKS = frozenset({"medium", "high"})

# First capture plus four bounded retries. Staggering only changes the capture
# phase; it never relaxes the required exact draft pixels or authorises input.
_MESSAGE_RECHECK_DELAYS = (0.15, 0.25, 0.35, 0.45)

# One initial frame plus two read-only captures; this does not retry a verifier,
# change focus, move the pointer, type text, or authorise a Send.
_MESSAGE_QUALITY_MAX_CAPTURES = 3


class _TaskDeadlineExceeded(RuntimeError):
    """An expensive observation or visual request consumed the remaining budget."""


@dataclass
class Timings:
    """Runtime entry to final verdict, excluding CLI setup and final serialization.

    The original phases retain their boundaries: initial observation/planning,
    all confirmation gates (including countdowns), then execution/re-planning.
    ``planner_ms`` separately totals *every* planner call, so it overlaps those
    phases and must not be added to them. ``system_elapsed_ms`` removes gate time.
    """

    started: float
    planned: float | None = None
    confirmation_seconds: float = 0.0
    confirmation_started: float | None = None
    planner_seconds: float = 0.0
    planning_calls: int = 0
    planning_attempts: int = 0

    def begin_confirmation(self, started: float) -> None:
        self.confirmation_started = started

    def end_confirmation(self, ended: float) -> None:
        if self.confirmation_started is not None:
            self.confirmation_seconds += ended - self.confirmation_started
            self.confirmation_started = None

    def breakdown(self, ended: float) -> dict[str, float]:
        planned = self.planned if self.planned is not None else self.started
        confirmation = self.confirmation_seconds
        if self.confirmation_started is not None:
            confirmation += ended - self.confirmation_started
        elapsed = ended - self.started
        planning = planned - self.started
        return {
            "elapsed_ms": elapsed * 1000.0,
            "planning_ms": planning * 1000.0,
            "confirmation_ms": confirmation * 1000.0,
            "execution_ms": (elapsed - planning - confirmation) * 1000.0,
            "planner_ms": self.planner_seconds * 1000.0,
            "system_elapsed_ms": (elapsed - confirmation) * 1000.0,
        }


def _error_type(message: str | None) -> str:
    """The exception class a step recorded, when its message starts with one.

    Step errors are free text, and "no element matches 'x'" is as common as
    "ActionResolutionError: ...". Only a leading single word is treated as a class
    name; reporting a whole sentence as one would be worse than reporting nothing.
    """
    head, separator, _ = (message or "").partition(":")
    head = head.strip()
    return head if separator and head.isidentifier() else ""


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
        client = getattr(planner, "client", None)
        self.target_grounder = (
            TargetGrounder(client) if callable(getattr(client, "generate_multimodal", None)) else None
        )
        if verifier.message_client is None and self.target_grounder is not None:
            verifier.message_client = client
        self._message_header: tuple[ObservationSnapshot, BoundingBox] | None = None
        self._message_composer: tuple[ObservationSnapshot, BoundingBox] | None = None
        self._message_text_typed = False
        self._recovery_records: list[RecoveryRecord] = []

    # ── entry point ────────────────────────────────────────────────────
    def run(
        self,
        task: TaskSpec,
        options: ExecutionOptions,
        *,
        confirm: Callable[[TaskPlan], bool] | None = None,
        countdown: Callable[[int], None] | None = None,
        high_risk_confirm: Callable[[TaskPlan], bool] | None = None,
        on_plan: Callable[[TaskPlan], None] | None = None,
    ) -> TaskRunResult:
        """Run one task, and leave a readable record whatever happens.

        14.2.3 asks for a summary on interruption as well as on failure. Ctrl+C
        arrives as a KeyboardInterrupt at whatever line is executing, which is
        usually inside a step or a model call, so the in-memory step list is
        whatever the unwinding happened to leave behind. What is trustworthy is
        what was already appended to the log, and that is what the summary is
        rebuilt from.
        """
        self._recovery_records = []
        started = self.clock()
        self._timings = Timings(started=started)
        self._notes: list[str] = []
        try:
            return self._run(
                task, options, started, confirm, countdown, high_risk_confirm, on_plan
            )
        except KeyboardInterrupt:
            return self._abandon(task, options, started)

    def _run(
        self,
        task: TaskSpec,
        options: ExecutionOptions,
        started: float,
        confirm: Callable[[TaskPlan], bool] | None,
        countdown: Callable[[int], None] | None,
        high_risk_confirm: Callable[[TaskPlan], bool] | None,
        on_plan: Callable[[TaskPlan], None] | None,
    ) -> TaskRunResult:
        timings = self._timings
        notes = self._notes
        self._message_header = None
        self._message_composer = None
        self._message_text_typed = False
        self._message_send_attempted = False
        self._message_send_blocked = False
        self._message_cursor_parking_attempted = False

        if not task.success_rules and not task.expect_text and not task.forbid_text:
            return self._blocked(
                task, options, "the task defines no verifiable success rule", timings=timings
            )

        # 0. Application state, which text on screen cannot express - and which is cheap
        # enough to check before a screenshot is taken.
        #
        # T01 says "no browser window is open" and T02 says "a browser window is open".
        # Both are facts about the machine, and reading them off the screen gets them
        # wrong in the same way: a browser that is behind another window or minimised
        # contributes none of the text either rule looks for. T01 therefore planned
        # against a window that already existed and was failed for not opening anything,
        # and T05's "the marker is gone" cannot be told from "the window is not visible".
        # These two lists are the machine-checkable half of those preconditions.
        #
        # Checked first on purpose: a run that cannot count should not spend a capture,
        # an OCR pass and a model call finding that out.
        if options.execute and options.require_preconditions:
            for group in task.forbids_processes:
                found = match_processes(known_processes(group))
                if found:
                    return self._blocked(
                        task,
                        options,
                        f"the run assumes no {group} is running, but these are: "
                        f"{', '.join(found)}. Text on screen cannot show this - a "
                        f"window behind another one carries no text - so the plan would "
                        f"be written against a window that already exists and the run "
                        f"could not be credited with opening it. Close them first.",
                        timings=timings,
                    )
            for group in task.requires_processes:
                if not match_processes(known_processes(group)):
                    return self._blocked(
                        task,
                        options,
                        f"the run needs a {group} to be running, and none of "
                        f"{', '.join(known_processes(group))} is. Start one, put it in "
                        f"the foreground, and re-run.",
                        timings=timings,
                    )

        # 1. first look at the screen
        message_deadline = (
            started + options.task_timeout_seconds
            if options.execute and task.message_conversation else None
        )
        if message_deadline is not None and self.clock() >= message_deadline:
            self.recorder.session.save_json("screenshot_quality_initial.json", {
                "stage": "initial_message_preflight", "outcome": "timed_out",
                "code": "task_budget_exhausted_before_capture",
                "detail": "task budget expired before the first message capture",
                "max_captures": _MESSAGE_QUALITY_MAX_CAPTURES, "capture_attempts": 0,
                "samples": [], "gui_actions": 0, "model_calls": 0,
                "semantic_authorization": False, "send_authorization": False,
            })
            return self._finish(
                task, options, "timed_out", started,
                ["screenshot quality: task budget expired before initial capture"], timings=timings,
            )
        try:
            initial = self.observer.observe()
        except Exception as exc:  # noqa: BLE001 - a failed first look ends the run
            if message_deadline is not None:
                expired = self.clock() >= message_deadline
                self.recorder.session.save_json("screenshot_quality_initial.json", {
                    "stage": "initial_message_preflight",
                    "outcome": "timed_out" if expired else "blocked",
                    "code": "initial_message_observation_failed",
                    "detail": f"initial observation failed: {type(exc).__name__}: {exc}",
                    "max_captures": _MESSAGE_QUALITY_MAX_CAPTURES, "capture_attempts": 1,
                    "samples": [{"capture_index": 1, "capture_error": f"{type(exc).__name__}: {exc}"}],
                    "gui_actions": 0, "model_calls": 0,
                    "semantic_authorization": False, "send_authorization": False,
                })
                if expired:
                    return self._finish(
                        task, options, "timed_out", started,
                        [f"initial observation failed after task budget expired: {exc}"], timings=timings,
                    )
            return self._blocked(
                task, options, f"initial observation failed: {exc}", timings=timings
            )
        self.recorder.save_observation(initial)
        # 7.1.4: the engine's own account of itself belongs in the run, not only in
        # the frame file. If OCR fell back, that explains a slow frame or a poorer
        # text pass, and the operator should hear it while the run is happening
        # rather than by reading obs-0001.json afterwards.
        for notice in getattr(initial, "notices", []):
            notes.append(f"ocr: {notice}")

        if options.execute and task.message_conversation:
            initial, quality = self._message_quality_preflight(initial, message_deadline)
            reason = f"screenshot quality: {quality['code']}: {quality['detail']}"
            if quality["outcome"] == "timed_out":
                return self._finish(
                    task, options, "timed_out", started, notes + [reason],
                    snapshot=initial, timings=timings,
                )
            if quality["outcome"] == "blocked":
                return self._blocked(task, options, reason, snapshot=initial, timings=timings)
            notes.append(reason)
            context = self.verifier.check_message_context(task, initial, require_empty=True)
            # Keep the assessment that authorised the recipient, including a
            # refusal; later header matching must be auditable from the run.
            self.recorder.session.save_json("message_context.json", context.model_dump(mode="json"))
            if not context.passed:
                return self._blocked(
                    task, options, f"message context: {context.detail}",
                    snapshot=initial, timings=timings,
                )
            header = context.evidence["vision_assessment"]["header_box"]
            self._message_header = (
                initial, BoundingBox(left=header[0], top=header[1], right=header[2], bottom=header[3])
            )
            composer = context.evidence["vision_assessment"]["composer_box"]
            self._message_composer = (
                initial, BoundingBox(left=composer[0], top=composer[1], right=composer[2], bottom=composer[3])
            )

        # What the frame said went wrong, in the run's own notes. The OCR engine's
        # account of a failure was written to obs-0001.json and nowhere else, so the
        # operator saw "no readable text" - which reads as "wake the screen" - while
        # the actual cause, a missing `tesseract` binary on a fresh machine, sat in a
        # file they had no reason to open.
        for problem in getattr(initial, "errors", [])[:3]:
            notes.append(f"observation: {problem}")

        # A frame with no readable text is not an error - OCR cannot read a locked,
        # dark or mostly-empty screen - but it makes every text target unresolvable.
        # Saying so here turns a bare "no element matches '...'" into something the
        # operator can act on.
        if not any(item.text.strip() for item in initial.elements):
            hint = (
                " (the observation above names the cause; a missing `tesseract` binary "
                "reads the same way as a locked screen)"
                if initial.errors
                else ""
            )
            notes.append(
                "the first frame had no readable text: OCR returned no labels, so no "
                f"text target can resolve against it{hint}"
            )

        # 1b. A real run must not start from a screen where the goal already holds.
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
                # The note carries what was actually seen, not only the conclusion.
                # "The rule already holds" is true both when the application was
                # closed and when it is merely behind another window (or its text
                # was not read), and those need different things from the operator -
                # one is a finished task, the other is an unready desktop. Without
                # the evidence the reviewer has to parse obs-NNNN.json to tell them
                # apart, which is what the Windows round had to do.
                seen = [item.text.strip() for item in initial.elements if item.text.strip()][:8]
                return self._blocked(
                    task,
                    options,
                    "the success rule already holds on the untouched screen "
                    f"({precondition.detail}); this run cannot be credited with it. "
                    f"Evidence: {len(initial.elements)} elements read, "
                    f"{len(initial.errors)} observation error(s); first text seen: {seen}",
                    timings=timings,
                )

        # 2. plan from that observation
        try:
            plan_result = self._plan_from(initial, task, options, timings)
        except Exception as exc:  # noqa: BLE001 - failed providers still get a summary
            return self._blocked(
                task, options, f"planning failed: {type(exc).__name__}: {exc}",
                timings=timings, snapshot=initial,
            )
        # Stamped whether or not the plan worked out. A planning call that fails
        # after two seconds still spent those two seconds planning, and leaving the
        # stamp unset reported them as `execution_ms` - which is the direction that
        # misleads, because a blocked run then looks like one that was busy acting.
        if not plan_result.ok or plan_result.plan is None:
            return self._blocked(
                task,
                options,
                f"planning failed: {explain_model_failure(plan_result.error or 'no plan produced')}",
                timings=timings,
            )
        plan = plan_result.plan
        planning_attempts = timings.planning_attempts
        notes.append(f"planned {len(plan.steps)} steps from {initial.observation_id}")

        # 8.3.5: a plan that reports errors is not executed. The model uses this
        # field to say it could not work the task out, and running it anyway would
        # be reading "I am not sure" as "go ahead".
        if plan.errors:
            return self._blocked(
                task,
                options,
                "the plan reports errors and will not be executed: "
                + "; ".join(plan.errors[:3]),
                snapshot=initial,
                timings=timings,
            )

        # 3. budget check before anything is dispatched
        if len(plan.executable_steps) > options.max_actions:
            return self._blocked(
                task,
                options,
                f"the plan wants {len(plan.executable_steps)} actions, above the "
                f"{options.max_actions} allowed",
                timings=timings,
            )

        # 12.2.2 wants the plan shown with its steps, the text it would type, the
        # targets and the risk. It was shown only inside the execute-mode
        # confirmation, so the default mode - a dry run, whose entire purpose is to
        # let the operator see what would happen - printed none of it.
        if on_plan is not None:
            on_plan(plan)

        # 4. confirmation gate — before any real input event
        if options.execute:
            timings.begin_confirmation(timings.planned if timings.planned is not None else self.clock())
            # Both prompts are asked in order, and a refusal at either ends the run.
            # The short-circuit is the point: a plan the operator just declined must
            # not then be put to them a second time.
            #
            # A risky task gets the second, separate confirmation, and the policy
            # lives here rather than in each caller - a caller that forgets to ask is
            # the failure this gate exists to stop, and one already did: the CLI
            # built the prompt and never passed it on.
            declined = (
                options.confirm and confirm is not None and not confirm(plan)
            ) or (
                options.confirm
                and high_risk_confirm is not None
                and task.risk in _RISKY_RISKS
                and not high_risk_confirm(plan)
            )
            if declined:
                # The gate is over, so time a person spent reading the plan belongs
                # to confirmation_ms. Without this the wait landed in execution_ms,
                # which made a hesitant operator look like a slow system.
                timings.end_confirmation(self.clock())
                result = self._finish(
                    task,
                    options,
                    "cancelled",
                    started,
                    notes,
                    snapshot=initial,
                    planning_attempts=planning_attempts,
                    timings=timings,
                )
                return result
            if countdown is not None:
                countdown(3)
            # Everything from here is the system working, not a person reading - the
            # countdown included, since it is a warning addressed to the operator.
            timings.end_confirmation(self.clock())

        return self._execute_plan(
            task, plan, options, initial, started, notes, planning_attempts, timings=timings
        )

    def _plan_from(
        self, snapshot: ObservationSnapshot, task: TaskSpec,
        options: ExecutionOptions, timings: Timings,
        *, recovery: RecoveryFeedback | None = None,
    ) -> PlanResult:
        """Account for a call even if it raises, returns errors or returns no steps."""
        call_started = self.clock()
        timings.planning_calls += 1
        result: PlanResult | None = None
        planner_started: float | None = None
        message_call = task.case_id == "T04" and bool(task.message_conversation)
        instruction = task.instruction
        context: dict[str, Any] | None = None
        error = ""
        record_name = f"planning_{timings.planning_calls:04d}.json"
        planning_record: dict[str, Any] = {
            "schema": "week4.t04_planning.v1", "call_index": timings.planning_calls,
            "observation_id": snapshot.observation_id, "image_path": snapshot.image_path,
            "original_instruction": task.instruction, "instruction": instruction,
            "context": None, "raw_response": None, "parsed_plan": None,
            "error": None, "planner_invoked": False,
        }
        try:
            context = self._context(snapshot, task, options)
            if recovery is not None:
                context["recovery"] = recovery.model_dump(mode="json")
            if message_call:
                if isinstance(self.planner, TaskPlanner):
                    # TaskPlanner adds this to its model context. Include it in
                    # the saved request as well, with the same actual value.
                    context = {**context, "max_steps": self.planner.max_steps}
                if options.execute and self._message_text_typed:
                    # The event was dispatched, not verified as committed. The
                    # original task still controls independent draft/sent checks.
                    instruction = (
                        "The text-entry event for this task was already dispatched. "
                        "The draft has not yet been independently verified as committed. "
                        "Continue from the current screen: only click the visible send control "
                        "using a current listed element_id, then finish. For an OCR send candidate, "
                        "the click must include both arguments.element_id and target_text copied "
                        "exactly from that same candidate's actual listed text; do not invent a label. "
                        "Do not type again, "
                        "refocus the editor, clear text, press Enter or use keyboard shortcuts. "
                        "If no safe send control is visible, return finish with errors."
                    )
                elif len(task.expect_text) == 1:
                    message = context["message_task"]
                    editor_id = message.get("editor_candidate_id")
                    send_id = message.get("send_candidate_id")
                    editor = snapshot.element(editor_id) if isinstance(editor_id, str) else None
                    send = snapshot.element(send_id) if isinstance(send_id, str) else None
                    candidate_ids = message.get("candidate_ids", [])
                    if (
                        message.get("region_status") == "visually_authorised"
                        and editor is not None and editor.element_id in candidate_ids
                        and editor.source == "contour" and not editor.text.strip()
                        and send is not None and send.element_id in candidate_ids
                        and send.source == "ocr" and send.text.strip()
                    ):
                        # A schema illustration, not an executable plan. Values
                        # come from the current authorised context/frame; the
                        # model still has to produce every executable argument.
                        example = json.dumps({
                            "task_id": task.case_id, "instruction": task.instruction,
                            "summary": "Focus the editor, type the marker, and send",
                            "steps": [
                                {"step_id": "focus", "description": "click the authorised editor",
                                 "action_type": "click", "target_text": None,
                                 "arguments": {"element_id": editor.element_id}},
                                {"step_id": "type", "description": "type this run's complete marker",
                                 "action_type": "type_text", "target_text": None,
                                 "arguments": {"text": task.expect_text[0]}},
                                {"step_id": "send", "description": "click the visible send control",
                                 "action_type": "click", "target_text": send.text,
                                 "arguments": {"element_id": send.element_id}},
                                {"step_id": "finish", "description": "end the plan",
                                 "action_type": "finish", "target_text": None, "arguments": {}},
                            ], "errors": [],
                        }, ensure_ascii=False)
                        instruction = (
                            task.instruction + "\nPlan all four steps in order: click the authorised "
                            "current editor, type_text, click the visible send control, finish. "
                            "Include the editor click even if it appears focused. Put the complete "
                            "marker in arguments.text; the type_text step must have target_text null "
                            "and no element_id. A marker only in target_text or description does not "
                            "authorise typing. For the OCR send click, include both arguments.element_id "
                            "and target_text copied exactly from that same candidate's actual listed "
                            "text: an ID alone may be stale in the fresh dispatch frame. Use only "
                            "current authorised listed candidates; do not invent target labels or "
                            "press Enter. Typing is a dispatched event, not proof of a committed draft. "
                            "If a required candidate is unavailable or ambiguous, return no executable "
                            "steps and explain in errors.\nPlan JSON example: " + example
                        )
                    else:
                        example = json.dumps({"steps": [{
                            "step_id": "type", "description": "type this run's complete marker",
                            "action_type": "type_text", "target_text": None,
                            "arguments": {"text": task.expect_text[0]},
                        }]}, ensure_ascii=False)
                        instruction = (
                            task.instruction + "\nThe typing step must use the nested JSON shape "
                            + example + ". Put the complete marker in arguments.text. A marker only "
                            "in target_text or description does not authorise typing. This illustrates "
                            "only the typing fields: a complete click example cannot be built from "
                            "the current unique authorised editor/send roles. Use only actual current "
                            "authorised candidates; do not invent IDs or target labels. If a required "
                            "candidate is unavailable or ambiguous, return no executable steps and "
                            "explain in errors."
                        )
                planning_record.update(instruction=instruction, context=context, planner_invoked=True)
                # Keep the actual request even if the model call is interrupted.
                self.recorder.session.save_json(record_name, planning_record)
            planner_started = self.clock()
            result = self.planner.plan(
                instruction,
                context=context,
                image_path=snapshot.image_path,
                task_id=task.case_id,
            )
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended = self.clock()
            planner_ms = (ended - planner_started) * 1000.0 if planner_started is not None else 0.0
            timings.planner_seconds += planner_ms / 1000.0
            # A provider exception has no PlanResult, but a planning attempt did
            # happen. Returned parse/format attempts are summed across all calls.
            timings.planning_attempts += max(1, int(result.attempts or 0)) if result else 1
            if message_call:
                planning_record.update(
                    instruction=instruction, context=context,
                    elapsed_ms=(ended - call_started) * 1000.0,
                    planner_ms=planner_ms,
                    attempts=max(1, int(result.attempts or 0)) if result else 1,
                    response_scope="final_response_of_planner_call",
                    raw_response=result.response.content if result and result.response else None,
                    response=result.response.as_dict() if result and result.response else None,
                    parsed_plan=result.plan.model_dump(mode="json") if result and result.plan else None,
                    error=error or (result.error if result else None),
                )
                self.recorder.session.save_json(record_name, planning_record)
            if timings.planned is None:
                # Automated request/response evidence writes belong to initial
                # planning, never to the subsequent human confirmation gate.
                timings.planned = self.clock() if message_call else ended

    # ── the loop ───────────────────────────────────────────────────────
    def _execute_plan(
        self,
        task: TaskSpec,
        plan: TaskPlan,
        options: ExecutionOptions,
        initial: ObservationSnapshot,
        started: float,
        notes: list[str],
        planning_attempts: int = 0,
        *,
        timings: Timings | None = None,
    ) -> TaskRunResult:
        """Run a plan, and plan again from what the screen shows when it falls short.

        Measured on T02, which needs two actions - enter a query, submit it - and got a
        one-step plan three runs in a row. `planner.plan` was called exactly once per run
        and the returned list was executed in order, so a one-step plan ended the run after
        one action with the goal unmet, and no amount of prompt wording moved it: the
        identical prompt and frame produced 1, 2, 8 and 9 steps across runs.

        So the retry lives here instead. When a pass ends without the success rules holding,
        and the run's own budget still allows it, the model is asked again with the screen
        as it now stands - which is the difference between "the model has to answer the
        whole task in one shot" and "the model has to answer the next step". The whole loop
        is bounded twice over by what already existed: `task_timeout_seconds` is re-checked
        before each pass, and `max_planning_attempts` caps the number of passes.
        """
        # One list for the whole run, not one per pass. A step that could not be resolved is
        # recorded here and its pass ends; without carrying this across passes, the record of
        # *why* the run failed would be discarded by the very retry that follows it, and the
        # result would report no steps at all for a run that plainly took some.
        steps: list[StepRecord] = []
        pass_start = 0
        result = self._execute_plan_once(
            task,
            plan,
            options,
            initial,
            started,
            notes,
            planning_attempts,
            steps=steps,
            timings=timings,
        )
        passes = 1
        while (
            (result is None or result.status == "failed")
            and passes < options.max_planning_attempts
            and self.clock() - started <= options.task_timeout_seconds
            and not self._message_send_attempted
            and not self._message_send_blocked
            and sum(record.action_result is not None for record in steps) < options.max_actions
        ):
            prepared = self._prepare_recovery(
                task, options, started, notes, steps[pass_start:], result,
                next_pass=passes + 1,
                actions_used=sum(record.action_result is not None for record in steps),
                timings=timings or self._timings,
            )
            planning_attempts = (timings or self._timings).planning_attempts
            if prepared is None:
                break
            retry_plan, after = prepared
            passes += 1
            notes.append(
                f"re-planned from {after.observation_id}: attempt {passes} of at most "
                f"{options.max_planning_attempts}"
            )
            pass_start = len(steps)
            result = self._execute_plan_once(
                task,
                retry_plan,
                options,
                initial,
                started,
                notes,
                planning_attempts,
                steps=steps,
                timings=timings,
                planning_snapshot=after,
            )
        status = result.status if result is not None else "failed"
        if status == "failed" and self.clock() - started > options.task_timeout_seconds:
            status = "timed_out"
        # A pass may already have written a provisional result. Finalize again
        # after *all* retries: an empty/error retry still spent time, and a later
        # unresolved step still belongs beside the earlier steps in the summary.
        return self._finish(
            task,
            options,
            status,
            started,
            notes,
            steps,
            snapshot=initial,
            planning_attempts=planning_attempts,
            timings=timings,
            verification=result.verification if result is not None else None,
        )

    def _prepare_recovery(
        self,
        task: TaskSpec,
        options: ExecutionOptions,
        started: float,
        notes: list[str],
        pass_steps: list[StepRecord],
        result: TaskRunResult | None,
        *,
        next_pass: int,
        actions_used: int,
        timings: Timings,
    ) -> tuple[TaskPlan, ObservationSnapshot] | None:
        """Observe, re-plan and record the attempt without dispatching input."""
        attempt_started = self.clock()
        deadline = started + options.task_timeout_seconds
        feedback = build_recovery_feedback(
            pass_steps, previous_pass=next_pass - 1,
            verification=result.verification if result else None,
            actions_used=actions_used, max_actions=options.max_actions,
            remaining_ms=remaining_milliseconds(deadline, attempt_started),
        )
        observation_id: str | None = None
        outcome = "observation_failed"
        try:
            try:
                after = self._observing()
                observation_id = after.observation_id
            except Exception as exc:  # noqa: BLE001 - preserve the completed pass
                notes.append(f"re-planning skipped: observation failed: {exc}")
                return None
            now = self.clock()
            feedback = feedback.model_copy(update={
                "remaining_ms": remaining_milliseconds(deadline, now),
            })
            if now > deadline:
                outcome = "timed_out"
                notes.append("task budget exhausted while observing for re-planning")
                return None
            outcome = "planning_failed"
            try:
                retry = self._plan_from(after, task, options, timings, recovery=feedback)
            except Exception as exc:  # noqa: BLE001 - preserve count and duration
                notes.append(f"re-planning failed: {type(exc).__name__}: {exc}")
                return None
            if self.clock() > deadline:
                outcome = "timed_out"
                notes.append("task budget exhausted during re-planning; no further actions")
                return None
            if not retry.ok or retry.plan is None or not retry.plan.steps:
                outcome = "no_plan"
                notes.append(f"re-planning attempt {next_pass} produced no steps; stopping")
                return None
            if retry.plan.errors:
                outcome = "plan_errors"
                notes.append(
                    f"re-planning attempt {next_pass} reports errors and will not be "
                    "executed: " + "; ".join(retry.plan.errors[:3])
                )
                return None
            outcome = "plan_ready"
            return retry.plan, after
        except KeyboardInterrupt:
            outcome = "cancelled"
            raise
        finally:
            record = RecoveryRecord(
                next_pass=next_pass, outcome=outcome,  # type: ignore[arg-type]
                observation_id=observation_id, feedback=feedback,
                elapsed_ms=max(0.0, (self.clock() - attempt_started) * 1000),
            )
            self._recovery_records.append(record)
            self.recorder.append_recovery(record)

    def _execute_plan_once(
        self,
        task: TaskSpec,
        plan: TaskPlan,
        options: ExecutionOptions,
        initial: ObservationSnapshot,
        started: float,
        notes: list[str],
        planning_attempts: int = 0,
        *,
        steps: list[StepRecord] | None = None,
        timings: Timings | None = None,
        planning_snapshot: ObservationSnapshot | None = None,
    ) -> TaskRunResult | None:
        steps = [] if steps is None else steps
        source = planning_snapshot or initial
        current = source

        for step in plan.steps:
            if self.clock() - started > options.task_timeout_seconds:
                return self._finish(
                    task,
                    options,
                    "timed_out",
                    started,
                    notes,
                    steps,
                    snapshot=initial,
                    planning_attempts=planning_attempts,
                    timings=timings,
                )

            record = StepRecord(
                index=len(steps) + 1,
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

            if sum(item.action_result is not None for item in steps) >= options.max_actions:
                notes.append("the cumulative action budget is exhausted")
                return self._finish(
                    task, options, "failed", started, notes, steps, snapshot=initial,
                    planning_attempts=planning_attempts, timings=timings,
                )

            # fresh look before acting; the plan may be minutes old
            try:
                before = self.observer.observe()
                self.recorder.save_observation(before)
            except Exception as exc:  # noqa: BLE001
                record.error = f"observation failed: {exc}"
                record.elapsed_ms = (self.clock() - step_started) * 1000
                steps.append(record)
                self.recorder.append_step(record)
                return self._finish(
                    task,
                    options,
                    "failed",
                    started,
                    notes,
                    steps,
                    snapshot=initial,
                    planning_attempts=planning_attempts,
                    timings=timings,
                )

            record.observation_id = before.observation_id

            try:
                deadline = started + options.task_timeout_seconds
                self._check_deadline(deadline)
                fresh_step, before, anonymous = self._ground_step(
                    step, source, before, notes, deadline=deadline
                )
                record.observation_id = before.observation_id
                if options.execute and task.message_conversation:
                    if step.action_type in {"key_press", "hotkey"}:
                        raise GroundingError("message tasks require the approved send control, not keys")
                    # The plan must explicitly supply the executable literal.
                    # A description/target label cannot fill missing arguments.
                    if step.action_type == "type_text" and (
                        len(task.expect_text) != 1 or step.arguments.get("text") != task.expect_text[0]
                    ):
                        raise GroundingError("typed message must exactly match this run's marker")
                    if step.action_type == "type_text" and any(
                        item.action_type == "type_text" and item.action_result is not None
                        for item in steps
                    ):
                        raise GroundingError("this run's marker was already typed; refusing repetition")
                    self._guard_message_header(before)
                if options.execute and (anonymous or task.message_conversation):
                    check_foreground = getattr(self.observer, "foreground_matches", None)
                    if not callable(check_foreground) or not check_foreground(before):
                        raise GroundingError("foreground changed immediately before dispatch")
                if self.clock() - started > options.task_timeout_seconds:
                    return self._finish(
                        task, options, "timed_out", started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings,
                    )
                self.adapter.resolution_region = None
                if options.execute and task.message_conversation:
                    # Scope text resolution to the composer this task has already authorised,
                    # so a word the plan names cannot match the same word that another
                    # application happens to be rendering. Measured on
                    # T04_20261006_233621: the frame held both the send control at
                    # (2072,986) and this harness's own activity log line
                    # `'已点击发送（2072,986）'` at (654,315), and the adapter refused the
                    # step for ambiguity before the guard that would have rejected the
                    # outside match could run.
                    #
                    # Only for an executing message task: a dry run dispatches nothing, and
                    # authorising a region is part of what a real send requires. A composer
                    # that is not authorised leaves the region unset and the old behaviour
                    # intact.
                    try:
                        composer, _ = self._message_controls(before)
                    except GroundingError:
                        composer = None
                    if composer is not None:
                        self.adapter.resolution_region = composer
                resolved = self.adapter.resolve(fresh_step, before)
                self.adapter.resolution_region = None
                if options.execute and task.message_conversation and step.action_type == "click":
                    self._guard_message_click(fresh_step, resolved, before)
            except _TaskDeadlineExceeded:
                return self._finish(
                    task, options, "timed_out", started, notes, steps, snapshot=initial,
                    planning_attempts=planning_attempts, timings=timings,
                )
            except GroundingError as exc:
                record.error = f"visual grounding refused: {exc}"
                record.elapsed_ms = (self.clock() - step_started) * 1000
                steps.append(record)
                self.recorder.append_step(record)
                notes.append(f"{step.step_id}: {record.error}")
                if self.clock() - started > options.task_timeout_seconds:
                    return self._finish(
                        task, options, "timed_out", started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings,
                    )
                return None
            except ActionResolutionError as exc:
                record.error = str(exc)
                record.elapsed_ms = (self.clock() - step_started) * 1000
                steps.append(record)
                self.recorder.append_step(record)
                notes.append(f"{step.step_id}: {exc}")
                # The pass ends here, not the run. Measured on T04: the model emitted
                # `type_text` with no `text`, the adapter refused it, and the whole run was
                # recorded `failed` with zero actions - when the case's own next attempt
                # needed only for the model to be asked again. A step that cannot be resolved
                # says this *plan* cannot be carried out; it does not say the task cannot, and
                # the caller has a re-planning loop for exactly the difference.
                #
                # `exhausted` is how that is signalled: the caller sees no verification result
                # and decides between another pass and a final failure, so a resolution error
                # it cannot improve on - and a run with no passes left - still ends properly.
                return None

            record.resolved = resolved
            if options.execute and task.message_conversation and step.action_type == "click" and any(
                item.action_type == "type_text" and item.action_result is not None for item in steps
            ):
                # Dispatch success only says the typing event was accepted. Read
                # the current draft independently before an irreversible send.
                try:
                    draft = self.verifier.check_message_draft(task, before)
                except Exception as exc:  # noqa: BLE001 - uncertainty cannot authorise a send
                    draft = VerificationResult(
                        outcome="inconclusive",
                        detail=f"draft verification failed: {type(exc).__name__}: {exc}",
                    )
                self.recorder.session.save_json(
                    f"message_draft_{record.index:04d}.json", draft.model_dump(mode="json")
                )
                if not draft.passed:
                    self._message_send_blocked = True
                    record.verification = draft
                    record.error = f"message draft verification refused send: {draft.detail}"
                    record.elapsed_ms = (self.clock() - step_started) * 1000
                    steps.append(record)
                    self.recorder.append_step(record)
                    notes.append(record.error)
                    return self._finish(
                        task, options, "failed", started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings, verification=draft,
                    )
                # Draft verification is another slow visual request: the run's
                # budget and foreground must still hold when it returns.
                if self.clock() - started > options.task_timeout_seconds:
                    self._message_send_blocked = True
                    record.verification = draft
                    record.error = "task budget exhausted while verifying the message draft"
                    record.elapsed_ms = (self.clock() - step_started) * 1000
                    steps.append(record)
                    self.recorder.append_step(record)
                    notes.append(record.error)
                    return self._finish(
                        task, options, "timed_out", started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings,
                    )
                verified = before
                recheck: dict[str, Any] = {
                    "verified_observation_id": verified.observation_id,
                    "outcome": "inconclusive",
                }
                draft.evidence["dispatch_recheck"] = recheck
                try:
                    fresh_step, resolved, before = self._refresh_verified_message_send(
                        fresh_step, resolved, verified, recheck,
                        deadline=started + options.task_timeout_seconds,
                    )
                    record.observation_id = before.observation_id
                    record.resolved = resolved
                    recheck.update(
                        outcome="passed", composer_pixels="exact_match",
                        send_candidate_pixels="exact_match", send_candidate_id=resolved.element_id,
                    )
                    self.recorder.session.save_json(
                        f"message_draft_{record.index:04d}.json", draft.model_dump(mode="json")
                    )
                except KeyboardInterrupt:
                    recheck["outcome"] = "cancelled"
                    self.recorder.session.save_json(
                        f"message_draft_{record.index:04d}.json", draft.model_dump(mode="json")
                    )
                    raise
                except Exception as exc:  # noqa: BLE001 - a changed/unreadable draft cannot be sent
                    self._message_send_blocked = True
                    stopped_status = "timed_out" if isinstance(exc, _TaskDeadlineExceeded) else "failed"
                    recheck.update(outcome=stopped_status, error=f"{type(exc).__name__}: {exc}")
                    self.recorder.session.save_json(
                        f"message_draft_{record.index:04d}.json", draft.model_dump(mode="json")
                    )
                    record.verification = VerificationResult(
                        outcome="failed", detail="the accepted draft changed before dispatch",
                        evidence=draft.evidence,
                    )
                    record.error = f"message changed after draft verification: {type(exc).__name__}: {exc}"
                    record.elapsed_ms = (self.clock() - step_started) * 1000
                    steps.append(record)
                    self.recorder.append_step(record)
                    notes.append(record.error)
                    return self._finish(
                        task, options, stopped_status, started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings,
                    )
                if self.clock() - started > options.task_timeout_seconds:
                    self._message_send_blocked = True
                    record.verification = draft
                    record.error = "task budget exhausted while rechecking the verified message draft"
                    record.elapsed_ms = (self.clock() - step_started) * 1000
                    steps.append(record)
                    self.recorder.append_step(record)
                    notes.append(record.error)
                    return self._finish(
                        task, options, "timed_out", started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings,
                    )
                check_foreground = getattr(self.observer, "foreground_matches", None)
                if not callable(check_foreground) or not check_foreground(before):
                    self._message_send_blocked = True
                    record.error = "foreground changed while verifying the message draft"
                    record.elapsed_ms = (self.clock() - step_started) * 1000
                    steps.append(record)
                    self.recorder.append_step(record)
                    notes.append(record.error)
                    return self._finish(
                        task, options, "failed", started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings,
                    )
                # Dispatch may have sent the message even if the next screenshot
                # or visual verdict fails. A retry must never send it again.
                self._message_send_attempted = True
            action_result = self.executor.execute(
                resolved.action, dry_run=not options.execute, screen=before.screen_info
            )
            record.action_result = {
                "success": action_result.success,
                "dry_run": action_result.dry_run,
                "error": action_result.error,
                "action": redact(resolved.action.model_dump(mode="json")),
            }
            if options.execute and task.message_conversation and step.action_type == "type_text" and action_result.success:
                self._message_text_typed = True

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
                action_type=step.action_type,
                before=before,
                after=after if after is not None else (before if not options.execute else None),
                action_result=action_result,
            )
            record.elapsed_ms = (self.clock() - step_started) * 1000
            steps.append(record)
            self.recorder.append_step(record)

            if not action_result.success:
                notes.append(f"{step.step_id}: action failed")
                return self._finish(
                    task,
                    options,
                    "failed",
                    started,
                    notes,
                    steps,
                    snapshot=initial,
                    planning_attempts=planning_attempts,
                    timings=timings,
                )

            # 10.1.12: continue when the expectation is met, and stop when the
            # result does not match it. The mismatch was computed and then never
            # read, so a step that plainly did not do what it was for was recorded
            # and the plan carried on typing into a screen that had not responded.
            #
            # Only for a real run. Nothing is dispatched in a dry run, so a screen
            # that did not change is the expected outcome rather than a mismatch -
            # stopping there would truncate the one mode whose purpose is to walk
            # the whole plan and show it.
            if (
                options.execute
                and record.verification.outcome == "failed"
                # A case may declare that its steps cannot be judged this way; see
                # TaskSpec.gate_on_step_verification for the measurement that made T04
                # need it. The task rule still decides the outcome at the end.
                and task.gate_on_step_verification
            ):
                notes.append(
                    f"{step.step_id}: {record.verification.detail}; stopping rather than "
                    "continuing from a step whose result was not observed"
                )
                return self._finish(
                    task,
                    options,
                    "failed",
                    started,
                    notes,
                    steps,
                    snapshot=initial,
                    planning_attempts=planning_attempts,
                    timings=timings,
                )

            if after is not None:
                current = after
            if (options.execute and task.message_conversation
                    and step.action_type == "type_text" and action_result.success):
                try:
                    current = self._park_message_cursor(
                        before, after, options, started, steps,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - uncertain parking cannot precede a Send
                    self._message_send_blocked = True
                    status = "timed_out" if isinstance(exc, _TaskDeadlineExceeded) else "failed"
                    notes.append(f"cursor parking refused further input: {type(exc).__name__}: {exc}")
                    return self._finish(
                        task, options, status, started, notes, steps, snapshot=initial,
                        planning_attempts=planning_attempts, timings=timings,
                    )
            if self._message_send_attempted:
                notes.append("message send was attempted; verifying before any further input")
                break

        # ── the plan is done; only the task verifier may call it a success ──
        if not options.execute:
            # A dry run dispatched nothing, so the task rule cannot have been met -
            # but reporting it as "failed" would say the run went wrong, and it did
            # not. The rule's own verdict is kept as evidence, not as the outcome.
            would_be = self.verifier.check_task(task, current)
            return self._finish(
                task,
                options,
                "dry_run_completed",
                started,
                notes,
                steps,
                snapshot=initial,
                planning_attempts=planning_attempts,
                timings=timings,
                verification=VerificationResult(
                    outcome="inconclusive",
                    method=task.verification,
                    detail=(
                        "dry run: actions were resolved and validated but nothing was "
                        f"dispatched, so the task goal was not attempted "
                        f"(the rule would read {would_be.outcome})"
                    ),
                    evidence={"would_be": would_be.outcome, "rule_detail": would_be.detail},
                ),
            )

        verification, final = self.verifier.check_task_with_polling(
            task,
            self._observing,
            deadline_seconds=options.verification_timeout_seconds,
            sleep=self.sleep,
            clock=self.clock,
        )
        # A missing marker string is not evidence that a window closed. T05's rule asks for
        # "the application's window is gone", and a text-only rule cannot tell a closed
        # application from an unreadable frame, a frame whose element budget filled with
        # other text, or a miss by the OCR engine. Asked of the machine instead, which is
        # the one place the question has an answer.
        if verification.passed and task.must_exit_processes:
            still_running: list[str] = []
            for group in task.must_exit_processes:
                names = known_processes(group)
                if names and match_processes(names):
                    still_running.append(group)
            if still_running:
                notes.append(
                    "the screen rule matched, but the process the task was to close is "
                    f"still running ({', '.join(still_running)}); a marker that cannot be "
                    "read is not a window that closed"
                )
                verification = VerificationResult(
                    outcome="failed",
                    method=task.verification,
                    detail=(
                        "the text rule matched but the application was never closed: "
                        + ", ".join(still_running)
                        + " still running"
                    ),
                    evidence={
                        **(verification.evidence or {}),
                        "still_running": still_running,
                        "screen_rule_outcome": verification.outcome,
                    },
                )
        status = "succeeded" if verification.passed else "failed"
        if self.clock() - started > options.task_timeout_seconds:
            status = "timed_out"
        if not verification.passed:
            notes.append(f"task verification: {verification.detail}")
        result = self._finish(
            task,
            options,
            status,
            started,
            notes,
            steps,
            snapshot=initial,
            planning_attempts=planning_attempts,
            timings=timings,
            verification=verification,
        )
        if final is not None:
            self.recorder.save_observation(final)
        return result

    def _ground_step(
        self, step: PlanStep, source: ObservationSnapshot, current: ObservationSnapshot,
        notes: list[str], *, deadline: float | None = None,
    ) -> tuple[PlanStep, ObservationSnapshot, bool]:
        """Refresh only one anonymous target, preserving every approved action field."""
        old_id = step.arguments.get("element_id") or step.arguments.get("target_element")
        old = source.element(old_id) if isinstance(old_id, str) else None
        anonymous = old is not None and old.source == "contour" and not old.text.strip()
        if not anonymous:
            return step, current, anonymous
        # Invalid frame identity is terminal for grounding, not a reason to ask
        # the model to guess across a different window or degraded capture.
        validate_frames(source, current)
        editor_target = False
        allowed_candidate_ids = None
        if self._message_composer is not None:
            source_composer, _ = self._message_controls(source)
            if old.bounding_box == source_composer:
                # T04_20261007_132911_213511_fb82db6e mapped the approved
                # editor to eligible OCR Send. Fresh RGB checks proved that
                # Send stayed unchanged, not that it was the old editor.
                # Bind the role to the actual approved source candidate and
                # independent composer evidence, not step_id/description.
                source_editor = self._message_editor_candidate(source)
                if source_editor.element_id != old_id:
                    raise GroundingError("approved source target is not the authorised message editor")
                editor_target = True
                current_editor = self._message_editor_candidate(current)
                allowed_candidate_ids = frozenset({current_editor.element_id})
        method = "pixel_match"
        try:
            chosen = match_candidate(source, current, old_id)
        except GroundingError:
            if self.target_grounder is None:
                raise GroundingError("no model client can refresh the changed anonymous target") from None
            self._check_deadline(deadline)
            if editor_target:
                mapped = self.target_grounder.ground(
                    step, source, current, self.recorder.directory,
                    allowed_candidate_ids=allowed_candidate_ids,
                )
            else:
                mapped = self.target_grounder.ground(step, source, current, self.recorder.directory)
            self._check_deadline(deadline)
            chosen_id = mapped.arguments.get("element_id") or mapped.arguments.get("target_element")
            if editor_target and chosen_id not in allowed_candidate_ids:
                raise GroundingError("message editor mapping changed the approved target role")
            mapping_frame = current
            current = self._observing()
            if editor_target:
                # A model may take minutes. Preserve the role/uniqueness in the
                # fresh frame as well as the original RGB identity check.
                current_editor = self._message_editor_candidate(current)
            chosen = match_candidate(mapping_frame, current, chosen_id)
            method = "vision_model_then_pixel_match"
        if editor_target and chosen.element_id != current_editor.element_id:
            raise GroundingError("message editor grounding changed the approved target role")
        arguments = dict(step.arguments)
        for key in ("element_id", "target_element"):
            if key in arguments:
                arguments[key] = chosen.element_id
        notes.append(
            f"{step.step_id}: grounded {old_id} from {source.observation_id} "
            f"to {chosen.element_id} in {current.observation_id} ({method})"
        )
        return step.model_copy(update={"arguments": arguments}, deep=True), current, True

    def _park_message_cursor(
        self, typing_before: ObservationSnapshot, typing_after: ObservationSnapshot | None,
        options: ExecutionOptions, started: float, steps: list[StepRecord],
    ) -> ObservationSnapshot:
        """One counted move away from draft pixels; strict Send guards stay downstream."""
        current = typing_after or typing_before
        if self._message_cursor_parking_attempted:
            return current
        self._message_cursor_parking_attempted = True
        parking_started = self.clock()
        deadline = started + options.task_timeout_seconds
        ledger: dict[str, Any] = {
            "scope": "one runtime geometry-only cursor move after successful typing",
            "typing_before_observation_id": typing_before.observation_id,
            "typing_after_observation_id": typing_after.observation_id if typing_after else None,
            "dispatch_attempted": False, "outcome": "inconclusive",
            "semantic_authorization": False, "send_authorization": False,
            "max_actions": options.max_actions,
            "modifier_cleanup": "existing executor key_up cleanup retained; no click/type/press/send",
            "stop_mechanism": "KeyboardInterrupt and existing executor fail-safe; no global Esc listener",
        }
        record: StepRecord | None = None

        def persist() -> None:
            ledger["elapsed_ms"] = (self.clock() - parking_started) * 1000
            self.recorder.session.save_json("cursor_parking.json", ledger)

        try:
            self._check_deadline(deadline)
            used = sum(item.action_result is not None for item in steps)
            ledger["actions_used_before"] = used
            if options.max_actions - used < 2:
                ledger.update(outcome="skipped", code="reserve_one_send_action")
                persist()
                self._check_deadline(deadline)
                return current
            # check_step may itself be slow. Take a new capture for this internal
            # move rather than treating the earlier post-type RGB as current.
            current = self.observer.observe()
            self.recorder.save_observation(current)
            self._check_deadline(deadline)
            validate_frames(typing_before, current)
            if (current.observation_id == typing_before.observation_id
                    or current.captured_at < typing_before.captured_at
                    or (typing_after is not None and (
                        current.observation_id == typing_after.observation_id
                        or current.captured_at < typing_after.captured_at
                    ))):
                raise GroundingError("cursor parking requires a fresh post-type observation")
            self._guard_message_header(current)
            decision = plan_cursor_parking(current, step_id="runtime_cursor_parking")
            ledger["before"] = decision.evidence["source"]
            ledger["decision"] = {"code": decision.code, **decision.evidence}
            self._check_deadline(deadline)
            if decision.outcome == "skipped":
                ledger.update(outcome="skipped", code=decision.code)
                persist()
                self._check_deadline(deadline)
                return current
            if decision.outcome != "planned" or decision.resolved is None:
                raise GroundingError(f"cursor parking geometry refused: {decision.code}")
            record = StepRecord(
                index=len(steps) + 1, step_id=decision.resolved.step_id,
                description="move cursor outside current message window before strict draft reading",
                action_type="move", observation_id=current.observation_id, resolved=decision.resolved,
            )
            ledger.update(outcome="ready_to_dispatch", resolved=decision.resolved.model_dump(mode="json"))
            persist()
            self._check_deadline(deadline)
            check_foreground = getattr(self.observer, "foreground_matches", None)
            if not callable(check_foreground) or not check_foreground(current):
                raise GroundingError("foreground changed immediately before cursor parking")
            self._check_deadline(deadline)
            ledger["dispatch_attempted"] = True
            record.action_result = {
                "success": False, "dry_run": False, "error": "cursor move completion unknown",
                "action": decision.resolved.action.model_dump(mode="json"),
            }
            result = self.executor.execute(
                decision.resolved.action, dry_run=False, screen=current.screen_info,
            )
            if (not isinstance(result, ActionResult) or type(result.success) is not bool
                    or result.dry_run is not False or result.action != decision.resolved.action):
                raise GroundingError("cursor parking returned an unknown execution result")
            record.action_result.update(success=result.success, error=result.error)
            ledger["action_result"] = record.action_result
            if not result.success:
                raise GroundingError(f"cursor parking move failed: {result.error}")
            self._check_deadline(deadline)
            ledger["post_capture_attempted"] = True
            moved = self.observer.observe()
            self.recorder.save_observation(moved)
            record.after_observation_id = moved.observation_id
            ledger["after_observation_id"] = moved.observation_id
            ledger["after_geometry"] = {
                "window_identity": list(window_identity(moved)),
                "screen_info": moved.screen_info.model_dump(mode="json"),
                "image_path": moved.image_path, "errors": moved.errors,
            }
            ledger["after"] = verify_parking_frame(current, moved)
            self._guard_message_header(moved)
            if not check_foreground(moved):
                raise GroundingError("foreground changed after cursor parking")
            self._check_deadline(deadline)
            record.verification = VerificationResult(
                outcome="passed", detail="cursor move completed with unchanged foreground geometry; no Send authorization",
                evidence={"semantic_authorization": False, "send_authorization": False},
            )
            ledger.update(outcome="completed", code="move_completed_geometry_unchanged")
            persist()
            self._check_deadline(deadline)
            return moved
        except KeyboardInterrupt:
            self._message_send_blocked = True
            ledger.update(outcome="cancelled", error="KeyboardInterrupt")
            if record is not None:
                record.error = "cursor parking interrupted; no further input authorised"
            raise
        except Exception as exc:
            self._message_send_blocked = True
            ledger.update(outcome="timed_out" if isinstance(exc, _TaskDeadlineExceeded) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            if record is not None:
                record.error = ledger["error"]
                record.verification = VerificationResult(outcome="failed", detail=record.error)
            raise
        finally:
            if record is not None:
                ledger["action_result"] = record.action_result
            try:
                persist()
                if ledger["outcome"] in {"completed", "skipped"}:
                    self._check_deadline(deadline)
            except KeyboardInterrupt:
                self._message_send_blocked = True
                ledger.update(outcome="cancelled", error="KeyboardInterrupt while saving parking evidence")
                if record is not None:
                    record.error = ledger["error"]
                    record.verification = VerificationResult(outcome="failed", detail=record.error)
                persist()
                raise
            except _TaskDeadlineExceeded:
                self._message_send_blocked = True
                ledger.update(outcome="timed_out", error="task budget exhausted while saving parking evidence")
                if record is not None:
                    record.error = ledger["error"]
                    record.verification = VerificationResult(outcome="failed", detail=record.error)
                persist()
                raise
            except Exception as exc:
                self._message_send_blocked = True
                ledger.update(outcome="failed", error=f"parking evidence write failed: {type(exc).__name__}: {exc}")
                if record is not None:
                    record.error = ledger["error"]
                    record.verification = VerificationResult(outcome="failed", detail=record.error)
                persist()
                raise
            finally:
                if record is not None:
                    record.elapsed_ms = (self.clock() - parking_started) * 1000
                    steps.append(record)
                    self.recorder.append_step(record)
                    try:
                        if ledger["outcome"] in {"completed", "skipped"}:
                            self._check_deadline(deadline)
                    except _TaskDeadlineExceeded:
                        self._message_send_blocked = True
                        ledger.update(outcome="timed_out", code="parking_step_log_budget_exhausted",
                                      error="move already recorded; step-log IO exhausted task budget")
                        persist()
                        # The move log is append-only. Record the subsequent IO
                        # failure separately, without counting another action.
                        failure = StepRecord(
                            index=len(steps) + 1, step_id="runtime_cursor_parking_evidence",
                            description="cursor move recorded; evidence IO consumed remaining task budget",
                            action_type="evidence", error=ledger["error"],
                            verification=VerificationResult(outcome="failed", detail=ledger["error"]),
                        )
                        steps.append(failure)
                        self.recorder.append_step(failure)
                        raise

    def _check_deadline(self, deadline: float | None) -> None:
        if deadline is not None and self.clock() > deadline:
            raise _TaskDeadlineExceeded

    def _guard_message_header(self, current: ObservationSnapshot) -> None:
        """Keep the visually authorised recipient fixed through all input events.

        Header pixels are verification evidence only. Coordinates dispatched by
        the executor still come from a real current candidate through the adapter.
        Exact crops deliberately refuse unreadable or changed recipient headers.
        """
        if self._message_header is None:
            raise GroundingError("message recipient was not visually authorised")
        source, box = self._message_header
        validate_frames(source, current)
        assert source.window_bounds is not None and current.window_bounds is not None
        dx = current.window_bounds.left - source.window_bounds.left
        dy = current.window_bounds.top - source.window_bounds.top
        translated = (box.left + dx, box.top + dy, box.right + dx, box.bottom + dy)
        bounds = current.window_bounds
        if not (
            bounds.left <= translated[0] < translated[2] <= bounds.right
            and bounds.top <= translated[1] < translated[3] <= bounds.bottom
        ):
            raise GroundingError("message header left the active window")
        try:
            with Image.open(source.image_path) as image:
                old = image.convert("RGB").crop((box.left, box.top, box.right, box.bottom))
            with Image.open(current.image_path) as image:
                new = image.convert("RGB").crop(translated)
            low, high = old.convert("L").getextrema()
            if high - low < 10 or old.tobytes() != new.tobytes():
                raise GroundingError("active conversation header changed or is unreadable")
        except (OSError, ValueError) as exc:
            raise GroundingError(f"cannot validate the active conversation header: {exc}") from exc

    def _message_controls(self, current: ObservationSnapshot) -> tuple[BoundingBox, list[ElementRef]]:
        """Current detected controls in the independently authorised editor region.

        Planning and dispatch share this boundary. Window translation transports
        the approved region, never a click point or an old candidate id.
        """
        if self._message_composer is None:
            raise GroundingError("message editor was not visually authorised")
        source, authorised = self._message_composer
        validate_frames(source, current)
        self._guard_message_header(current)
        assert source.window_bounds is not None and current.window_bounds is not None
        dx = current.window_bounds.left - source.window_bounds.left
        dy = current.window_bounds.top - source.window_bounds.top
        composer = BoundingBox(
            left=authorised.left + dx, top=authorised.top + dy,
            right=authorised.right + dx, bottom=authorised.bottom + dy,
        )
        bounds = current.window_bounds
        if not (
            bounds.left <= composer.left < composer.right <= bounds.right
            and bounds.top <= composer.top < composer.bottom <= bounds.bottom
        ):
            raise GroundingError("message composer left the active window")
        candidates = [
            item for item in current.elements
            if item.source in {"ocr", "contour"} and item.confidence >= 0.35
            and item.center == item.bounding_box.center
            and composer.left <= item.bounding_box.left < item.bounding_box.right <= composer.right
            and composer.top <= item.bounding_box.top < item.bounding_box.bottom <= composer.bottom
        ]
        return composer, candidates

    def _message_editor_candidate(self, current: ObservationSnapshot) -> ElementRef:
        """One observed editor contour, exactly matching the authorised region."""
        composer, candidates = self._message_controls(current)
        editors = [
            item for item in candidates
            if item.source == "contour" and not item.text.strip() and item.bounding_box == composer
        ]
        if len(editors) != 1:
            raise GroundingError("authorised message editor is not one unique current contour")
        return editors[0]

    def _guard_message_click(
        self, step: PlanStep, resolved: ResolvedAction, current: ObservationSnapshot,
    ) -> None:
        """Refuse a target outside the exact current candidate list given to planning."""
        if any(key in step.arguments for key in ("x", "y", "coordinates", "position", "point")):
            raise GroundingError("message clicks require a current detected candidate, not coordinates")
        _, candidates = self._message_controls(current)
        matches = [item for item in candidates if item.element_id == resolved.element_id]
        if len(matches) != 1 or resolved.screenshot_point != matches[0].center:
            raise GroundingError("message click is not a current candidate inside the authorised composer")

    def _refresh_verified_message_send(
        self, step: PlanStep, resolved: ResolvedAction, verified: ObservationSnapshot,
        evidence: dict[str, Any],
        *, deadline: float,
    ) -> tuple[PlanStep, ResolvedAction, ObservationSnapshot]:
        """Recheck the exact accepted draft after the slow visual request returns.

        Window translation is allowed. A blinking caret can return to the accepted
        pixels within at most five fresh captures; only composer mismatch permits
        another capture, never altered headers, candidates or degraded frames.
        Even one persistently changed text pixel refuses the send. No similarity
        threshold or mask can distinguish a caret from an edited character safely.
        """
        old_composer, _ = self._message_controls(verified)
        previous = verified.element(resolved.element_id or "")
        if previous is None:
            raise GroundingError("the verified send candidate is missing")

        def crop(image: Image.Image, box: BoundingBox) -> Image.Image:
            return image.convert("RGB").crop((box.left, box.top, box.right, box.bottom))

        def exact(old: Image.Image, new: Image.Image) -> bool:
            return old.size == new.size and old.tobytes() == new.tobytes()

        def ensure_budget() -> None:
            if self.clock() >= deadline:
                raise _TaskDeadlineExceeded("task budget exhausted during message pixel recheck")

        def ensure_foreground(snapshot: ObservationSnapshot) -> None:
            check = getattr(self.observer, "foreground_matches", None)
            if not callable(check) or not check(snapshot):
                raise GroundingError("foreground changed during message pixel recheck")

        with Image.open(verified.image_path) as old_image:
            accepted_composer = crop(old_image, old_composer)
            accepted_send = crop(old_image, previous.bounding_box)
        samples: list[dict[str, Any]] = []
        evidence.update(max_capture_attempts=5, samples=samples)
        latest = verified
        for index in range(5):
            sample: dict[str, Any] = {"attempt": index + 1, "outcome": "inconclusive"}
            samples.append(sample)
            try:
                ensure_budget()
                ensure_foreground(latest)
                if index:
                    delay = _MESSAGE_RECHECK_DELAYS[index - 1]
                    if self.clock() + delay >= deadline:
                        raise _TaskDeadlineExceeded("not enough task budget for another pixel recheck")
                    sample["wait_seconds"] = delay
                    self.sleep(delay)
                    ensure_budget()
                    ensure_foreground(latest)
                current = self._observing()
                latest = current
                sample.update(observation_id=current.observation_id, image_path=current.image_path)
                evidence.update(
                    dispatch_observation_id=current.observation_id,
                    dispatch_image_path=current.image_path,
                )
                ensure_budget()
                new_composer, _ = self._message_controls(current)
                ensure_foreground(current)
                chosen = match_candidate(verified, current, resolved.element_id or "")
                with Image.open(current.image_path) as new_image:
                    if not exact(accepted_send, crop(new_image, chosen.bounding_box)):
                        raise GroundingError("send candidate pixels changed after draft verification")
                    sample["send_candidate_pixels"] = "exact_match"
                    sample["send_candidate_id"] = chosen.element_id
                    if not exact(accepted_composer, crop(new_image, new_composer)):
                        sample.update(outcome="composer_mismatch", composer_pixels="mismatch")
                        ensure_budget()
                        continue
                sample["composer_pixels"] = "exact_match"
                arguments = {**step.arguments, "element_id": chosen.element_id}
                refreshed = step.model_copy(update={"arguments": arguments}, deep=True)
                fresh_resolved = self.adapter.resolve(refreshed, current)
                self._guard_message_click(refreshed, fresh_resolved, current)
                ensure_budget()
                sample["outcome"] = "passed"
                return refreshed, fresh_resolved, current
            except Exception as exc:
                sample.update(
                    outcome="timed_out" if isinstance(exc, _TaskDeadlineExceeded) else "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            except KeyboardInterrupt:
                sample["outcome"] = "cancelled"
                raise
        raise GroundingError("message composer pixels did not match in five bounded captures")

    # ── helpers ────────────────────────────────────────────────────────
    def _message_quality_preflight(
        self, initial: ObservationSnapshot, deadline: float | None,
    ) -> tuple[ObservationSnapshot, dict[str, Any]]:
        """Bounded source-frame checks before the first semantic model request.

        Only known transient capture conditions may cause another observation.
        A different window/title/class/bounds or monitor geometry is a refusal,
        not a reason to focus an app. Every sampled frame stays in the ledger.
        """
        samples: list[dict[str, Any]] = []
        ledger: dict[str, Any] = {
            "stage": "initial_message_preflight", "outcome": "pending",
            "max_captures": _MESSAGE_QUALITY_MAX_CAPTURES, "samples": samples,
            "gui_actions": 0, "model_calls": 0,
            "semantic_authorization": False, "send_authorization": False,
            "scope": "basic source-frame integrity; arbitrary overlays and cursor state are unknown",
        }
        current = initial
        original_identity = window_identity(initial)
        original_screen = initial.screen_info.model_dump()

        def conclude(outcome: str, code: str, detail: str):
            ledger.update(outcome=outcome, code=code, detail=detail)
            return current, ledger

        try:
            for index in range(1, _MESSAGE_QUALITY_MAX_CAPTURES + 1):
                sample: dict[str, Any] = {
                    "capture_index": index, "observation_id": current.observation_id,
                    "image_path": current.image_path, "captured_at": current.captured_at.isoformat(),
                    "window_id": current.window_id, "window_title": current.window_title,
                    "window_class": current.window_class,
                    "window_bounds": current.window_bounds.model_dump() if current.window_bounds else None,
                    "foreground_stable": current.foreground_stable,
                    "screen_info": current.screen_info.model_dump(),
                }
                samples.append(sample)
                self._check_deadline(deadline)
                if (window_identity(current) != original_identity
                        or current.screen_info.model_dump() != original_screen):
                    sample["outcome"] = "blocked"
                    return conclude(
                        "blocked", "foreground_changed_during_quality_recheck",
                        "window identity, bounds or monitor geometry changed during quality captures",
                    )
                assessment = assess_screenshot_quality(current)
                sample["quality"] = assessment.to_dict()
                self._check_deadline(deadline)
                sample["outcome"] = assessment.outcome
                if assessment.outcome != "retryable":
                    return conclude(assessment.outcome, assessment.code, assessment.detail)
                if index == _MESSAGE_QUALITY_MAX_CAPTURES:
                    return conclude(
                        "blocked", "quality_capture_limit_reached",
                        f"{assessment.code} persisted for {_MESSAGE_QUALITY_MAX_CAPTURES} captures",
                    )
                self._check_deadline(deadline)
                try:
                    current = self._observing()
                except Exception as exc:  # noqa: BLE001 - capture failure cannot authorize an action
                    expired = deadline is not None and self.clock() >= deadline
                    samples.append({
                        "capture_index": index + 1, "outcome": "timed_out" if expired else "blocked",
                        "capture_error": f"{type(exc).__name__}: {exc}",
                    })
                    return conclude(
                        "timed_out" if expired else "blocked", "quality_reobservation_failed",
                        f"read-only quality reobservation failed: {type(exc).__name__}: {exc}",
                    )
                except KeyboardInterrupt:
                    samples.append({"capture_index": index + 1, "outcome": "cancelled",
                                    "capture_error": "KeyboardInterrupt"})
                    raise
        except _TaskDeadlineExceeded as exc:
            if samples:
                samples[-1].update(outcome="timed_out", error=str(exc))
            return conclude("timed_out", "task_budget_exhausted_during_quality", str(exc))
        except KeyboardInterrupt:
            if samples:
                samples[-1]["outcome"] = "cancelled"
            ledger.update(outcome="cancelled", code="quality_preflight_interrupted",
                          detail="operator interrupted read-only quality preflight")
            raise
        finally:
            ledger["capture_attempts"] = len(samples)
            self.recorder.session.save_json("screenshot_quality_initial.json", ledger)
            # Automatic evidence IO is part of the task budget, too. Do not
            # start a semantic request if the apparently passed frame used it up.
            if (ledger["outcome"] == "passed" and deadline is not None
                    and self.clock() >= deadline):
                ledger.update(outcome="timed_out", code="task_budget_exhausted_during_quality_recording",
                              detail="task budget expired while recording quality evidence")
                samples[-1].update(outcome="timed_out", error=ledger["detail"])
                self.recorder.session.save_json("screenshot_quality_initial.json", ledger)
        raise AssertionError("quality capture loop ended without a verdict")

    def _observing(self) -> ObservationSnapshot:
        snapshot = self.observer.observe()
        self.recorder.save_observation(snapshot)
        return snapshot

    def _context(
        self, snapshot: ObservationSnapshot, task: TaskSpec, options: ExecutionOptions
    ) -> dict[str, Any]:
        from .observation import describe_elements

        context = {
            "platform": self.adapter.platform,
            "observation_id": snapshot.observation_id,
            "screen": (
                f"{snapshot.screen_info.screenshot_width}x{snapshot.screen_info.screenshot_height}"
            ),
            "target_app": task.target_app,
            # A plan that overshoots the budget is refused before anything is
            # dispatched, so the model is told the budget it is planning against
            # rather than discovering it by having its plan rejected.
            # 8.1.2: the whitelist itself, not just the advice to respect it.
            "allowed_keys": keys_for_platform(self.adapter.platform),
            "limits": (
                f"at most {options.max_actions} actions and "
                f"{options.task_timeout_seconds:g} s for the whole task"
            ),
            "visible_text": describe_elements(snapshot),
            "success_rules": task.success_rules,
        }
        if not task.message_conversation:
            return context
        # The recorded static T04 probe sent 300 whole-desktop targets (21806
        # tokens). The model bound typing to a sidebar/Codex target and selected
        # a sidebar click as send. Keep only observed controls inside the editor
        # region that the independent initial visual assessment authorised.
        context["visible_text"] = ""
        message_context: dict[str, Any] = {
            "text_already_typed": self._message_text_typed,
            "rules": (
                "Use only current listed candidate ids for clicks. "
                "Click the message editor, then type_text with arguments containing only text "
                "(no element_id/target_text), then click the visible send control. "
                "Never press Enter or use a keyboard shortcut to send. "
                "Do not click the sidebar, transcript or other applications."
            ),
        }
        context["message_task"] = message_context
        if self._message_composer is None:
            message_context.update(
                region_status="unavailable",
                rules="The message editor region has not been visually authorised. Return no steps and explain in errors.",
            )
            return context
        try:
            composer, candidates = self._message_controls(snapshot)
        except GroundingError as exc:
            message_context.update(
                region_status="unavailable", region_error=str(exc),
                rules="Current pixels cannot validate the authorised message editor. Return no steps and explain in errors.",
            )
            return context
        source = self._message_composer[0]
        visible = snapshot.model_copy(update={"elements": candidates})
        context["visible_text"] = describe_elements(visible)
        editors = [
            item for item in candidates
            if item.source == "contour" and not item.text.strip() and item.bounding_box == composer
        ]
        send_labels = [
            item for item in candidates
            if item.source == "ocr" and item.text.strip().casefold() in {"发送", "send"}
        ]
        message_context.update(
            region_status="visually_authorised", source_observation_id=source.observation_id,
            composer_bounds=composer.model_dump(),
            candidate_ids=[item.element_id for item in candidates],
        )
        if len(editors) == 1:
            message_context["editor_candidate_id"] = editors[0].element_id
        if len(send_labels) == 1:
            message_context["send_candidate_id"] = send_labels[0].element_id
        if self._message_text_typed:
            message_context["rules"] = (
                "A text-entry event for this run's marker was successfully dispatched; the "
                "draft is not yet verified as committed. Only click the "
                "current visible send control, then finish. Do not type again, click the editor "
                "to refocus, clear text, or press Enter. If send is unavailable return no steps "
                "and explain in errors."
            )
        elif len(task.expect_text) == 1:
            message_context["type_text_arguments"] = {"text": task.expect_text[0]}
        return context

    def _finish(
        self,
        task: TaskSpec,
        options: ExecutionOptions,
        status: str,
        started: float,
        notes: list[str],
        steps: list[StepRecord] | None = None,
        *,
        snapshot: ObservationSnapshot | None = None,
        planning_attempts: int = 0,
        timings: Timings | None = None,
        verification: VerificationResult | None = None,
    ) -> TaskRunResult:
        client = getattr(self.planner, "client", None)
        records = steps or []
        failed = next((record for record in records if record.error), None)
        # 16.5.4 wants the phases apart, so the clock is read once and split rather
        # than sampled twice with a different meaning each time.
        phases = (timings or Timings(started=started)).breakdown(self.clock())
        elapsed_ms = phases["elapsed_ms"]
        finished_at = datetime.now(UTC)
        result = TaskRunResult(
            run_id=self.recorder.run_id,
            case_id=task.case_id,
            instruction=task.instruction,
            status=status,  # type: ignore[arg-type]
            steps=records,
            # Passed in rather than attached to the returned object afterwards:
            # `_finish` is what writes task_summary.json, so a field set after it
            # returned was written as null. Every run's record on disk - the one the
            # reviewer reads and the evidence collector ships - said the task had no
            # verification at all.
            verification=verification,
            elapsed_ms=elapsed_ms,
            execute=options.execute,
            model_name=getattr(client, "model_name", ""),
            provider=getattr(client, "name", ""),
            notes=notes,
            # Provenance, because the record has to explain itself on a machine
            # that did not produce it - that is the point of shipping it as
            # evidence. `started_at` is derived rather than captured so the run
            # needs no extra state: `elapsed_ms` comes from the monotonic clock,
            # which is what a duration should be measured with anyway.
            task_id=task.case_id,
            commit=provenance.git_commit(),
            platform=self.adapter.platform,
            os_version=provenance.os_description(),
            python_version=provenance.python_version(),
            screen=provenance.screen_description(snapshot),
            started_at=finished_at - timedelta(milliseconds=elapsed_ms),
            finished_at=finished_at,
            planning_attempts=(timings.planning_attempts if timings is not None else planning_attempts),
            model_requests=int(getattr(client, "request_count", 0)),
            planning_ms=phases["planning_ms"],
            confirmation_ms=phases["confirmation_ms"],
            execution_ms=phases["execution_ms"],
            planner_ms=phases["planner_ms"],
            planning_calls=(timings.planning_calls if timings is not None else 0),
            system_elapsed_ms=phases["system_elapsed_ms"],
            evidence_directory=str(self.recorder.directory),
            stop_reason=(
                notes[-1] if notes and status not in {"succeeded", "dry_run_completed"} else ""
            ),
            error_type=_error_type(failed.error if failed else None),
            failed_step_id=failed.step_id if failed else None,
            recovery_records=self._recovery_records,
        )
        self.recorder.write_summary(result)
        return result

    def _abandon(self, task: TaskSpec, options: ExecutionOptions, started: float) -> TaskRunResult:
        """Close out a run the operator interrupted.

        The steps are read back from the step log rather than from memory: they are
        appended as they happen, so the file is the record of what was actually
        done. An interrupted run is a run that happened, and a report with no row
        for it is how a failed attempt disappears from a success rate.
        """
        records: list[StepRecord] = []
        for entry in self.recorder.read_steps():
            # A torn last line is possible when the interrupt landed mid-write, and
            # it must not cost the summary every step before it.
            with contextlib.suppress(Exception):
                records.append(StepRecord.model_validate(entry))
        recovery_notes: list[str] = []
        try:
            completed, invalid_lines = self.recorder.read_recovery()
            if invalid_lines:
                recovery_notes.append(
                    "recovery log contains incomplete or invalid records on lines "
                    + ", ".join(map(str, invalid_lines))
                    + "; retained complete records only"
                )
            if completed != self._recovery_records:
                recovery_notes.append(
                    "recovery summary rebuilt from completed log records after interruption"
                )
            self._recovery_records = completed
        except OSError:
            self._recovery_records = []
            recovery_notes.append(
                "recovery log could not be read; completed recovery records are unconfirmed"
            )
        return self._finish(
            task,
            options,
            "cancelled",
            started,
            self._notes + recovery_notes + [
                "interrupted by the operator; nothing further was executed",
            ],
            records,
            planning_attempts=self._timings.planning_attempts,
            timings=self._timings,
        )

    def _blocked(
        self,
        task: TaskSpec,
        options: ExecutionOptions,
        reason: str,
        *,
        timings: Timings,
        snapshot: ObservationSnapshot | None = None,
    ) -> TaskRunResult:
        # `timings.started`, not `self.clock()`: a blocked run that spent twenty
        # seconds inside a model call before giving up used to record 0.0 ms.
        return self._finish(
            task, options, "blocked", timings.started, [reason], snapshot=snapshot, timings=timings,
            planning_attempts=timings.planning_attempts,
        )
