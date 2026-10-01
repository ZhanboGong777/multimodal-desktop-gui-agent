"""Prompts for task decomposition.

The prompt states the allowed verbs and the exact JSON shape, because the parser
downstream is strict: a plan that does not validate is rejected rather than
patched up.

The example is kept as a plain string and the system prompt is assembled by
concatenation; ``str.format`` would treat the braces of that JSON as placeholders.

The user turn is not built here. It is the JSON envelope ``ModelRequest.to_messages``
produces from the instruction, the screen context and the image path, so that a
backend sees one object rather than a rendered template.
"""

from __future__ import annotations

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
