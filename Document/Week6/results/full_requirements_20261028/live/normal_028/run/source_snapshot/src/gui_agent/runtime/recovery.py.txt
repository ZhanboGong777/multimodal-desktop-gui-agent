"""Failure feedback for the existing bounded re-planning loop.

Only outcomes and counts cross this boundary. Arguments, raw model text,
verification evidence and free-form errors stay out of the recovery prompt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import get_args

from ..schemas import ActionType
from .schemas import (
    RecoveryFeedback,
    RecoveryReason,
    RecoveryStep,
    StepRecord,
    VerificationResult,
)

_ACTIONS = frozenset(get_args(ActionType))


def remaining_milliseconds(deadline: float, now: float) -> float | None:
    """None represents an unbounded budget; exhausted finite budgets are zero."""
    remaining = (deadline - now) * 1000
    return max(0.0, remaining) if math.isfinite(remaining) else None


def failure_reason(
    steps: Sequence[StepRecord], verification: VerificationResult | None,
) -> tuple[RecoveryReason, int | None]:
    """Classify the latest pass, never an error retained from an older pass."""
    # A task verdict means execution reached the task check. Earlier step checks
    # may be advisory (gate_on_step_verification=False), so they did not stop it.
    if verification is not None and not verification.passed:
        return "task_verification", None
    for step in reversed(steps):
        if step.action_type == "finish":
            continue
        if step.action_result is not None and step.action_result.get("success") is False:
            return "action_execution", step.index
        if step.error:
            if step.error.startswith("visual grounding refused:"):
                return "visual_grounding", step.index
            if step.error.startswith("observation failed:"):
                return "pre_action_observation", step.index
            if step.error.startswith("post-action observation failed:"):
                return "post_action_observation", step.index
            return "target_resolution", step.index
        if step.verification is not None and step.verification.outcome == "failed":
            return "step_verification", step.index
    return "plan_exhausted", None


def build_recovery_feedback(
    steps: Sequence[StepRecord],
    *,
    previous_pass: int,
    verification: VerificationResult | None,
    actions_used: int,
    max_actions: int,
    remaining_ms: float | None,
) -> RecoveryFeedback:
    """Summarize at most five executable steps from this pass only."""
    reason, failed_index = failure_reason(steps, verification)
    executable = [step for step in steps if step.action_type in _ACTIONS]
    recent = [
        RecoveryStep(
            index=step.index,
            action_type=step.action_type,  # type: ignore[arg-type]
            attempted=step.action_result is not None,
            succeeded=(
                step.action_result.get("success") is True
                if step.action_result is not None else None
            ),
            verification=step.verification.outcome if step.verification else None,
        )
        for step in executable[-5:]
    ]
    return RecoveryFeedback(
        previous_pass=previous_pass,
        reason=reason,
        failed_step_index=failed_index,
        actions_used=actions_used,
        actions_remaining=max(0, max_actions - actions_used),
        remaining_ms=remaining_ms,
        recent_steps=recent,
        omitted_steps=max(0, len(executable) - len(recent)),
    )
