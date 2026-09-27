"""Turn raw model text into a validated :class:`TaskPlan`.

Models wrap JSON in prose or markdown fences even when told not to, so the parser
recovers the object first and then lets Pydantic be the judge. Anything that still
does not validate is reported, never repaired silently.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .schemas import TaskPlan

FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class PlanParseError(RuntimeError):
    """Raised when model output cannot become a TaskPlan."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Recover a JSON object from raw model output."""
    if not text or not text.strip():
        raise PlanParseError("the model returned an empty response")

    candidates: list[str] = []
    fenced = FENCED.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    # Last resort: the outermost braces in the response.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text.strip().replace("\n", " ")[:120]
    raise PlanParseError(f"no JSON object found in the response: {preview!r}")


def parse_plan(
    text: str, *, instruction: str, max_steps: int = 10, task_id: str = "task-1"
) -> TaskPlan:
    """Validate model output into a plan, filling in the caller's instruction."""
    payload = extract_json_object(text)
    payload.setdefault("instruction", instruction)
    payload.setdefault("task_id", task_id)

    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanParseError("the plan contains no steps")

    # Give every step an id before validation, so a model that omits them still
    # produces a usable plan while anything else stays strict.
    for index, step in enumerate(steps):
        if isinstance(step, dict):
            step.setdefault("step_id", f"step-{index + 1}")
            step.setdefault("description", step.get("action_type", f"step {index + 1}"))

    try:
        plan = TaskPlan.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}"
            for error in exc.errors()[:5]
        )
        raise PlanParseError(f"the plan failed validation: {problems}") from exc

    if plan.step_count > max_steps:
        raise PlanParseError(
            f"the plan has {plan.step_count} steps, above the limit of {max_steps}"
        )
    return plan
