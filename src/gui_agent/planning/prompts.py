"""Prompts for task decomposition.

The prompt states the allowed verbs and the exact JSON shape, because the parser
downstream is strict: a plan that does not validate is rejected rather than
patched up.

The example is kept as a plain string and the prompt is assembled by
concatenation; ``str.format`` would treat the braces of that JSON as placeholders.
"""

from __future__ import annotations

import json
from typing import Any

from .schemas import PLAN_ACTION_TYPES

# Kept deliberately short. A small local vision model rambles when the prompt is
# long - and a rambling answer gets truncated, which loses the JSON entirely. The
# full field list is still stated; only the prose around it was cut.
_JSON_EXAMPLE = (
    '{"task_id":"t1","instruction":"...","summary":"...",'
    '"steps":[{"step_id":"step-1","description":"...","action_type":"click",'
    '"target_text":"...","arguments":{},"expected_result":"...","status":"pending"}],'
    '"assumptions":[],"requires_confirmation":true,"errors":[]}'
)

SYSTEM_PROMPT = "\n".join(
    [
        "You plan desktop GUI actions. Reply with ONE JSON object: no prose, no markdown fence.",
        "",
        _JSON_EXAMPLE,
        "",
        "Rules:",
        "- action_type must be one of: " + ", ".join(PLAN_ACTION_TYPES),
        '- The LAST step must use action_type "finish".',
        "- Use at most 3 steps unless the task clearly needs more.",
        "- Keep every string under 60 characters.",
        "- Never invent screen coordinates; name targets by the text visible on screen.",
        "- If the instruction is ambiguous, note it in assumptions and still return a plan.",
    ]
)


def build_user_prompt(
    instruction: str,
    *,
    context: dict[str, Any] | None = None,
    image_path: str | None = None,
    max_steps: int = 10,
) -> str:
    """Assemble the user turn: instruction, screen context and the hard limits."""
    parts = [f"Instruction: {instruction}", f"Maximum steps: {max_steps}"]
    if image_path:
        parts.append(f"Screenshot: {image_path}")
    if context:
        rendered = json.dumps(context, ensure_ascii=False, indent=2)[:4000]
        parts.append(f"Screen context:\n{rendered}")
    return "\n\n".join(parts)
