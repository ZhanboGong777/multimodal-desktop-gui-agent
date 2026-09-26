"""The planner: instruction plus screen context in, validated TaskPlan out.

It never executes anything. ``allow_real_execution`` is recorded on the result and
nothing else in this module touches the control layer - Week 4 wires the executor
in, Week 3 stops at the plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..models.base import ModelClient, ModelResponse
from .parser import PlanParseError, parse_plan
from .prompts import SYSTEM_PROMPT
from .schemas import TaskPlan


@dataclass
class PlanResult:
    """Everything one planning attempt produced, successful or not."""

    plan: TaskPlan | None = None
    response: ModelResponse | None = None
    error: str | None = None
    elapsed_ms: float = 0.0
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "attempts": self.attempts,
            "error": self.error,
            "plan": self.plan.as_dict() if self.plan else None,
            "model": self.response.as_dict() if self.response else None,
        }


class TaskPlanner:
    """Produces validated plans from a model backend."""

    def __init__(
        self,
        client: ModelClient,
        *,
        max_steps: int = 10,
        require_structured_output: bool = True,
        allow_real_execution: bool = False,
        max_format_retries: int = 1,
    ) -> None:
        self.client = client
        self.max_steps = max_steps
        self.require_structured_output = require_structured_output
        self.allow_real_execution = allow_real_execution
        self.max_format_retries = max(0, max_format_retries)

    def plan(
        self,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
        image_path: str | None = None,
        task_id: str = "task-1",
    ) -> PlanResult:
        """Ask the model for a plan and validate it. Never raises for a bad reply."""
        started = time.perf_counter()
        # The rules go in the system turn and the task stays a task. Passing the
        # whole rendered prompt as the instruction made an echoing backend reply
        # with the template instead of the instruction.
        screen_context = {**(context or {}), "max_steps": self.max_steps}
        attempts = 0
        last_error = "no attempt was made"
        response: ModelResponse | None = None

        for _ in range(self.max_format_retries + 1):
            attempts += 1
            response = self.client.generate_multimodal(
                instruction,
                image_path=image_path,
                context=screen_context,
                system=SYSTEM_PROMPT,
            )
            if not response.ok:
                last_error = response.error or "the model returned an unusable response"
                break
            try:
                plan = parse_plan(
                    response.content,
                    instruction=instruction,
                    max_steps=self.max_steps,
                    task_id=task_id,
                )
            except PlanParseError as exc:
                last_error = str(exc)
                continue  # one controlled retry, then give up
            return PlanResult(
                plan=plan,
                response=response,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                attempts=attempts,
                metadata={
                    "provider": response.provider,
                    "model_name": response.model_name,
                    "allow_real_execution": self.allow_real_execution,
                },
            )

        return PlanResult(
            response=response,
            error=last_error,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            attempts=attempts,
            metadata={"allow_real_execution": self.allow_real_execution},
        )

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
