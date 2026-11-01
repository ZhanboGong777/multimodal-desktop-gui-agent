"""Structured task plan produced by the model and validated by Pydantic.

The action vocabulary is the control layer's, extended by exactly one verb:
``finish``. It marks the end of a plan and is never handed to the executor, so it
belongs to the planning layer rather than to ``DesktopAction``.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import Field, field_validator, model_validator

from ..schemas import ActionType, SchemaModel

#: Verbs a plan step may use: everything the control layer can execute, plus finish.
PLAN_ACTION_TYPES: tuple[str, ...] = (*get_args(ActionType), "finish")

#: Verbs that are executed on the desktop; ``finish`` is deliberately excluded.
EXECUTABLE_ACTION_TYPES: frozenset[str] = frozenset(get_args(ActionType))

#: ``pending_runtime_resolution`` is the hand-off's name for a step whose target is
#: not on screen yet: the element cannot be named from this frame and has to be
#: resolved against a fresh observation when the step is reached. This runner
#: resolves every step that way, but the value still has to be accepted - a plan
#: written in the hand-off's own vocabulary must not be rejected as invalid.
StepStatus = Literal[
    "pending",
    "planned",
    "skipped",
    "done",
    "failed",
    "pending_runtime_resolution",
]


class PlanStep(SchemaModel):
    """One step of a plan."""

    step_id: str
    description: str
    action_type: str
    target_text: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_result: str | None = None
    status: StepStatus = "pending"

    @field_validator("action_type")
    @classmethod
    def _known_action(cls, value: str) -> str:
        cleaned = value.strip().casefold()
        if cleaned not in PLAN_ACTION_TYPES:
            known = ", ".join(PLAN_ACTION_TYPES)
            raise ValueError(f"unknown action_type {value!r}; allowed: {known}")
        return cleaned

    @field_validator("step_id", "description")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @property
    def is_terminal(self) -> bool:
        return self.action_type == "finish"

    @property
    def is_executable(self) -> bool:
        """False for ``finish``: that verb never reaches the control layer."""
        return self.action_type in EXECUTABLE_ACTION_TYPES


class TaskPlan(SchemaModel):
    """A validated plan: the only thing the planner is allowed to emit."""

    task_id: str
    instruction: str
    summary: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    requires_confirmation: bool = True
    errors: list[str] = Field(default_factory=list)

    @field_validator("task_id", "instruction")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _check_step_order(self) -> TaskPlan:
        """8.3.2 and 8.3.3: unique ids, and nothing after the terminal step.

        A step after ``finish`` is the dangerous one: ``is_executable`` looks only
        at the verb, so a plan that says "stop" and then adds one more click would
        have that click executed. Refusing the plan is the only safe reading of
        "nothing may run after the terminal step".
        """
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"duplicate step_id {step.step_id!r}")
            seen.add(step.step_id)

        endings = [
            index for index, step in enumerate(self.steps) if step.action_type == "finish"
        ]
        if len(endings) > 1:
            raise ValueError(f"{len(endings)} finish steps; a plan ends once")
        if endings and endings[0] != len(self.steps) - 1:
            trailing = len(self.steps) - endings[0] - 1
            raise ValueError(
                f"step {self.steps[endings[0]].step_id!r} is finish but {trailing} "
                "step(s) follow it"
            )
        return self

    @property
    def executable_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.is_executable]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
