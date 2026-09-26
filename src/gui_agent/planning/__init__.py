"""Task decomposition: instruction in, validated plan out.

The planner never executes anything. It turns a natural-language instruction plus
whatever the perception layer saw into a :class:`TaskPlan` that Pydantic has
accepted, and stops there.
"""

from .parser import PlanParseError, extract_json_object, parse_plan
from .planner import PlanResult, TaskPlanner
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import EXECUTABLE_ACTION_TYPES, PLAN_ACTION_TYPES, PlanStep, TaskPlan

__all__ = [
    "EXECUTABLE_ACTION_TYPES",
    "PLAN_ACTION_TYPES",
    "SYSTEM_PROMPT",
    "PlanParseError",
    "PlanResult",
    "PlanStep",
    "TaskPlan",
    "TaskPlanner",
    "build_user_prompt",
    "extract_json_object",
    "parse_plan",
]
