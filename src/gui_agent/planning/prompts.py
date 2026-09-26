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

_JSON_EXAMPLE = """{
  "task_id": "task-1",
  "instruction": "the original instruction",
  "summary": "one sentence describing the approach",
  "steps": [
    {
      "step_id": "step-1",
      "description": "what this step does, in plain language",
      "action_type": "one of the allowed verbs",
      "target_text": "text visible on screen to act on, or null",
      "arguments": {},
      "expected_result": "what should be true afterwards",
      "status": "pending"
    }
  ],
  "assumptions": ["anything you had to assume"],
  "requires_confirmation": true,
  "errors": []
}"""

SYSTEM_PROMPT = "\n".join(
    [
        "You are the planning module of a desktop GUI agent.",
        "",
        "You receive a user instruction plus optional context about what is currently",
        "on screen. You reply with ONE JSON object and nothing else: no prose, no",
        "markdown fence.",
        "",
        "Required shape:",
        _JSON_EXAMPLE,
        "",
        "Rules:",
        "- Allowed action_type values: " + ", ".join(PLAN_ACTION_TYPES),
        '- Use "finish" as the LAST step to end the plan.',
        "- Keep the plan under 10 steps.",
        "- Never invent screen coordinates. Identify targets by the text visible on",
        "  screen.",
        "- If the instruction is ambiguous, record the ambiguity in assumptions and",
        "  still return a plan.",
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
