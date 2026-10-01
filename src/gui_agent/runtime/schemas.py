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

    def element(self, element_id: str) -> ElementRef | None:
        """Return the element with this id, or None when the id is not from this frame."""
        for item in self.elements:
            if item.element_id == element_id:
                return item
        return None

    @property
    def texts(self) -> set[str]:
        return {item.text.casefold() for item in self.elements if item.text}


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
    risk: Literal["low", "medium", "high"] = "low"


class ExecutionOptions(SchemaModel):
    """Limits and switches for one run, all of them explicit."""

    execute: bool = False
    max_actions: int = Field(default=20, gt=0)
    task_timeout_seconds: float = Field(default=240.0, gt=0.0)
    verification_timeout_seconds: float = Field(default=10.0, ge=0.0)
    verification_poll_interval_seconds: float = Field(default=0.5, ge=0.0)
    confirm: bool = True
    require_preconditions: bool = True


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

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"

    @property
    def action_count(self) -> int:
        """Steps that actually produced a desktop action, excluding ``finish``."""
        return sum(1 for step in self.steps if step.resolved is not None)
