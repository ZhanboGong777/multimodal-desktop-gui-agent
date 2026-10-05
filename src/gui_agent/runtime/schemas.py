"""Data structures for the Week 4 runtime layer.

The observation, the resolved action, the verification outcome and the task
result are separate objects on purpose. Week 3 could report "the action did not
raise" and stop there; a closed loop has to keep those apart, because an action
that succeeded on a screen that did not move is not a finished task.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from ..schemas import BoundingBox, DesktopAction, Point, SchemaModel, ScreenInfo

#: How the outcome of a step or a task was decided.
VerificationMethod = Literal["automatic", "manual"]

#: Terminal states of one task run. Only ``succeeded`` means the task finished.
TaskStatus = Literal[
    "planned",
    "dry_run_completed",
    "succeeded",
    "failed",
    "blocked",
    "cancelled",
    "timed_out",
]

#: Outcome of one verification, kept separate from the action's own result.
VerificationOutcome = Literal["passed", "failed", "inconclusive"]


class ElementRef(SchemaModel):
    """One element of one observation, addressable by a frame-local id.

    ``element_id`` is deliberately not a property of :class:`~gui_agent.schemas.UIElement`:
    the id is only meaningful inside its own frame. Reusing an id from an older
    frame is the mistake this wrapper exists to make visible.
    """

    element_id: str
    text: str
    bounding_box: BoundingBox
    center: Point
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "ocr"


class ObservationSnapshot(SchemaModel):
    """One frame: image, geometry, elements and how long it took to produce."""

    observation_id: str
    captured_at: datetime
    image_path: str | None = None
    screen_info: ScreenInfo
    elements: list[ElementRef] = Field(default_factory=list)
    ocr_engine: str = "none"
    processing_time_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)
    #: The window in front when the frame was taken, read from the window rather than from
    #: the pixels, and the class name that identifies which application it belongs to.
    #:
    #: Kept because browser chrome is where a class of evidence lives - a results page's URL
    #: and its tab title - and the OCR engine reads that region unreliably. Measured on T02:
    #: a real Google results page carried `google.com/search?q=...` in its address field, the
    #: element list for that frame did not contain it, and the window title was read on one
    #: frame of the run and not on the next. The title needs no image work, and the class is
    #: what answers "is a browser in front at all", which is the fact a precondition like
    #: T02's actually wants.
    window_title: str = ""
    window_class: str = ""
    #: Anonymous targets need a window identity, not merely a reusable title.
    #: The capture samples it on both sides of the screenshot; changed or unknown
    #: foreground state cannot authorize cross-frame visual re-grounding.
    window_id: str = ""
    window_bounds: BoundingBox | None = None
    foreground_stable: bool = False
    #: What the OCR engine said about itself: a fallback to another engine, a
    #: missing cache directory, a suppressed Windows workaround. Kept separate
    #: from ``errors`` on purpose - 7.1.4 asks for the fallback engine to be
    #: recorded, and a frame produced by a working fallback is not a degraded
    #: frame, so putting this in ``errors`` would make the verifier refuse to
    #: judge any observation taken after one.
    notices: list[str] = Field(default_factory=list)

    def element(self, element_id: str) -> ElementRef | None:
        """Return the element with this id, or None when the id is not from this frame."""
        for item in self.elements:
            if item.element_id == element_id:
                return item
        return None


class ResolvedAction(SchemaModel):
    """A plan step turned into something the executor can run.

    Both coordinate spaces are kept. The screenshot point is what the model and
    the elements talk about; the control point is what PyAutoGUI receives, and the
    two differ whenever display scaling is not 1.0.
    """

    step_id: str
    action: DesktopAction
    element_id: str | None = None
    screenshot_point: Point | None = None
    control_point: Point | None = None
    note: str = ""


class VerificationResult(SchemaModel):
    """What a check concluded, and how confident the conclusion is."""

    outcome: VerificationOutcome
    method: VerificationMethod = "automatic"
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"


class TaskSpec(SchemaModel):
    """What the run is supposed to achieve, and how that will be judged.

    The model does not get to invent the success condition: a task with no
    verifiable rule returns ``blocked`` rather than being declared finished when
    the plan runs out.
    """

    case_id: str
    instruction: str
    target_app: str | None = None
    preconditions: list[str] = Field(default_factory=list)
    success_rules: list[str] = Field(default_factory=list)
    verification: VerificationMethod = "automatic"
    #: Text that must appear on screen for the task to count as complete.
    expect_text: list[str] = Field(default_factory=list)
    #: Text that must be gone for the task to count as complete.
    forbid_text: list[str] = Field(default_factory=list)
    #: Process groups (see runtime.processes.known) that must be running before the
    #: run starts, and groups that must not be. These are the machine-checkable half of
    #: preconditions that are about application state rather than screen text: a browser
    #: that is open but behind another window carries no text, so a text check calls it
    #: absent and T01 then plans against a window that already exists.
    requires_processes: list[str] = Field(default_factory=list)
    forbids_processes: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"] = "low"
    #: A send is not proved by finding its input marker anywhere on screen.
    #: T04 requires an independent visual assessment of the correct conversation,
    #: sent bubble and empty composer; ordinary text rules retain their behavior.
    message_conversation: str | None = None

    #: Whether a step whose own verification fails should stop the run.
    #:
    #: True everywhere by default, and it is the right rule: a step whose result was not
    #: observed is not something to build on, and continuing would act on a guess.
    #:
    #: Set False for a case whose *steps* cannot be verified that way but whose *task* can.
    #: T04 is the one, and the measurement is in the troubleshooting guide: its typing step asks
    #: the model to describe what it just did - `expected_result='The message is typed into the
    #: message box'` - and that sentence cannot be matched against a screen, because the box it
    #: names is invisible to the OCR. The same sentence came back on four passes of four separate
    #: runs, including runs whose instruction explicitly forbade mentioning the box. So the gate
    #: was not measuring the step; it was measuring whether the model happened to phrase its own
    #: expectation in words that appear on screen.
    #:
    #: Turning it off does not remove verification. The task-level rule still runs at the end
    #: against the whole screen - for T04, the marker as a sent message in the conversation - and
    #: every step's own outcome is still recorded. What goes away is a mid-plan stop caused by a
    #: description rather than by a fact.
    gate_on_step_verification: bool = True


class ExecutionOptions(SchemaModel):
    """Limits and switches for one run, all of them explicit."""

    execute: bool = False
    max_actions: int = Field(default=20, gt=0)
    task_timeout_seconds: float = Field(default=240.0, gt=0.0)
    verification_timeout_seconds: float = Field(default=10.0, ge=0.0)
    verification_poll_interval_seconds: float = Field(default=0.5, ge=0.0)
    confirm: bool = True
    #: When True, a real run checks the screen before planning and refuses to start
    #: when the success rule already holds there. The prose preconditions on
    #: TaskSpec say the same thing in words; this is the machine-checkable half, and
    #: it is what stops T01 or T05 being credited for a state that was already true.
    #: Only a task that declares preconditions is checked - declaring them is how a
    #: task says it assumes a starting state. Dry runs are never checked: they
    #: dispatch nothing, and the pipeline check is what they exist to perform.
    #:
    #: The same switch also gates `requires_processes`/`forbids_processes`, which answer
    #: the part of a precondition that screen text cannot: whether an application is
    #: running at all.
    require_preconditions: bool = True
    #: How many times the model may be asked for a plan in one run.
    #:
    #: 1 keeps the behaviour this runner had: plan once, execute the list, stop. Measured
    #: on T02 - which needs two actions and got a one-step plan three runs running - the
    #: ceiling of one pass is what ended the run with the goal unmet, and the identical
    #: prompt and frame produced 1, 2, 8 and 9 steps across runs, so more passes rather
    #: than better wording is the lever. Raising it lets a run that falls short ask again
    #: from the screen as it now stands, which turns "answer the whole task at once" into
    #: "answer the next step".
    #:
    #: Every pass re-checks `task_timeout_seconds`, so this bounds the count of attempts
    #: and the clock still bounds the run. Set it to 1 to plan exactly once.
    max_planning_attempts: int = Field(default=1, gt=0)


class StepRecord(SchemaModel):
    """Everything one step produced, kept so a failure can be reconstructed."""

    index: int
    step_id: str
    description: str
    action_type: str
    observation_id: str | None = None
    resolved: ResolvedAction | None = None
    action_result: dict[str, Any] | None = None
    verification: VerificationResult | None = None
    error: str | None = None
    elapsed_ms: float = 0.0
    after_observation_id: str | None = None


class TaskRunResult(SchemaModel):
    """The outcome of one task run."""

    run_id: str
    case_id: str
    instruction: str
    status: TaskStatus
    steps: list[StepRecord] = Field(default_factory=list)
    verification: VerificationResult | None = None
    error: str | None = None
    elapsed_ms: float = 0.0
    execute: bool = False
    model_name: str = ""
    provider: str = ""
    notes: list[str] = Field(default_factory=list)

    # ── provenance, so the record explains itself away from this machine ──
    # 14.2 asks the summary to carry these rather than leaving a reader to
    # reconstruct them from the environment they were typed into.
    task_id: str = ""
    commit: str = ""
    platform: str = ""
    #: `platform.platform()` - the OS *and* its build. The task report's
    #: environment table asks for the OS, and this is where it can come from
    #: instead of the operator typing it in afterwards.
    os_version: str = ""
    python_version: str = ""
    screen: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: How often the planner re-parsed or retried the plan format.
    planning_attempts: int = 0
    #: Transport attempts actually sent, retries included. Deliberately separate
    #: from planning_attempts: one plan can cost several requests, and one request
    #: can produce several parse attempts.
    model_requests: int = 0
    #: Wall clock, split so a slow operator cannot look like a slow model. 16.5.4
    #: asks for the phases separately and quotes `execution_ms` as the primary
    #: measure: from the moment the run was authorised to the final verdict.
    #: `elapsed_ms` stays as the whole run, confirmation prompt included.
    planning_ms: float = 0.0
    confirmation_ms: float = 0.0
    execution_ms: float = 0.0
    evidence_directory: str = ""
    #: Why the run stopped, in one line, when it did not simply succeed.
    stop_reason: str = ""
    error_type: str = ""
    failed_step_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"

    @property
    def action_count(self) -> int:
        """Steps that actually produced a desktop action, excluding ``finish``."""
        return sum(1 for step in self.steps if step.resolved is not None)
