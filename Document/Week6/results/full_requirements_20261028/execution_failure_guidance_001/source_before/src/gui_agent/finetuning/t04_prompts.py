"""Final C17/C19 prompt text only; this module never performs desktop input."""
from __future__ import annotations

import json

FOCUS_INTERIOR = (
    "Click inside the large blank message typing area below the chat history and above the "
    "row of emoji and attachment icons. Click the central interior of the typing area. "
    "Do not click the bottom toolbar, the Send button, or a chat bubble. Return only one click action."
)


def final_prompt(marker: str, phase: str) -> tuple[str, str]:
    """Preserve the exact literal and phase schemas from the final prompt version.

    Historical markers and the consumed authorization are deliberately absent.
    Returning text provides no authorization or facility to execute it.
    """
    if not isinstance(marker, str) or len(marker) != 44:
        raise ValueError("exactly44 characters required")
    literal = json.dumps(marker, ensure_ascii=False)
    instruction = (
        f'Send exactly this 44-character message to 文件传输助手:\n{literal}\n'
        'The quotation marks delimit the message and are not part of its text. '
        'Use the Send button, never Enter.'
    )
    steps = {
        'focus': FOCUS_INTERIOR,
        'type': (
            f'Type exactly these 44 characters into the focused input: {literal}\n'
            'Do not type the surrounding quotation marks. Return only a one-object JSON array '
            'with exactly two keys: action_type equal to type_text, and text equal to the '
            '44-character message above. The type_text action has a text field; '
            'it has no x or y coordinate fields.'
        ),
        'send': 'Click the Send button once. Return only one click action. Do not press Enter.',
        'verify': 'Observe the sent message.',
    }
    if phase not in steps:
        raise ValueError("unknown phase")
    return instruction, steps[phase]
