"""Week 4 runtime: the closed loop that actually drives the desktop.

This layer owns the order of operations — observe, plan, resolve, act, verify —
and the limits under which it is allowed to do so. It does not reimplement
perception or control: those come from Week 2, and planning from Week 3.
"""

from .action_adapter import ActionAdapter, ActionResolutionError
from .observation import ObservationError, ObservationService, describe_elements
from .recorder import TaskRecorder
from .runner import TaskRunner
from .schemas import (
    ElementRef,
    ExecutionOptions,
    ObservationSnapshot,
    ResolvedAction,
    StepRecord,
    TaskRunResult,
    TaskSpec,
    TaskStatus,
    VerificationResult,
)
from .verification import Verifier

__all__ = [
    "ActionAdapter",
    "ActionResolutionError",
    "ElementRef",
    "ExecutionOptions",
    "ObservationError",
    "ObservationService",
    "ObservationSnapshot",
    "ResolvedAction",
    "StepRecord",
    "TaskRecorder",
    "TaskRunResult",
    "TaskRunner",
    "TaskSpec",
    "TaskStatus",
    "VerificationResult",
    "Verifier",
    "describe_elements",
]
