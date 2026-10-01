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
        "- action_type is one of: " + ", ".join(PLAN_ACTION_TYPES),
        '- Target an element id from the screen list: {"element_id":"obs-0001-e003"}.',
        "- target_text must be copied exactly from the screen; never invent coordinates.",
        '- Arguments: type_text {"text"}; key_press {"key"}; hotkey {"keys":[...]}; '
        + 'scroll {"scroll_amount"}; wait {"duration"}.',
        "- Use this platform's key names.",
        '- The LAST step is "finish", only once the goal is reached.',
        "- description and summary under 60 chars; copy text exactly.",
        "- If it is ambiguous, say so in assumptions and still return a plan.",
    ]
)

#: The element list is the one part of the context that runs to thousands of
#: characters on a text-heavy screen, and it is the part the model aims with. It is
#: trimmed by whole element lines: cutting the serialized JSON instead leaves an
#: element id with half its text, which is worse than not listing that element.
ELEMENT_BUDGET_CHARS = 4000


def _trim_element_lines(block: str, budget: int) -> str:
    """Keep whole element lines, up to ``budget`` characters, and say what went."""
    lines = [line for line in block.splitlines() if line.strip()]
    kept: list[str] = []
    used = 0
    for line in lines:
        if kept and used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    dropped = len(lines) - len(kept)
    if dropped:
        kept.append(f"... {dropped} further elements omitted to fit the prompt")
    return "\n".join(kept)


def build_user_prompt(
    instruction: str,
    *,
    context: dict[str, Any] | None = None,
    image_path: str | None = None,
    max_steps: int = 10,
) -> str:
    """Assemble the user turn: instruction, screen context and the hard limits.

    The context is split rather than dumped as one blob: the element list is
    trimmed by whole lines, and the rest - short and bounded - is serialized whole.
    """
    data = dict(context or {})
    parts = [f"Instruction: {instruction}", f"Maximum steps: {max_steps}"]
    platform = data.pop("platform", None)
    if platform:
        parts.append(f"Platform: {platform}")
    if image_path:
        parts.append(f"Screenshot: {image_path}")
    elements = str(data.pop("visible_text", "") or "")
    if data:
        parts.append("Screen context:\n" + json.dumps(data, ensure_ascii=False, indent=2))
    if elements.strip():
        parts.append("Screen elements:\n" + _trim_element_lines(elements, ELEMENT_BUDGET_CHARS))
    return "\n\n".join(parts)
