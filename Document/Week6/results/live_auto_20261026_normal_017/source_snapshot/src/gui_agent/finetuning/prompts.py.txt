"""Pre-registered Week 5 prompts and label serialization.

All variants have the same action schema. Labels, paths, source metadata and
sample identifiers are deliberately excluded from the user prompt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

PROMPT_VARIANTS = ("original", "improved", "targeted_sequence", "current_step", "conditional_actor")
TRAINING_PROMPT_VARIANTS = ("original", "targeted_sequence", "current_step", "conditional_actor")
PROMPT_VERSION = "week5.action-array.v1"

ACTION_SCHEMA = (
    'Output a JSON array of desktop actions. Each object has "action_type" and '
    'only the fields needed for that action: click/double_click/right_click/move '
    'use integer "x" and "y"; type_text uses "text"; key_press uses "key"; '
    'hotkey uses "keys" (an array of strings); scroll uses integer '
    '"scroll_amount", "x" and "y"; drag uses "start" and "end" '
    '(each an object with integer "x","y"); wait uses "duration" in seconds. '
    'Use screenshot pixel coordinates, with (0,0) at the top left. '
    'Return the array alone, with no markdown, explanation or additional fields.'
)

_RULES = {
    "conditional_actor": (
        "You are executing the supplied current subtask of the overall goal. "
        "The subtask was supplied before this action request; do not replace it with "
        "another stage of the overall workflow. Use the current screenshot to produce "
        "the complete ordered executable action chunk for this subtask. Include necessary "
        "focus, text-entry and key actions in order, but do not continue into later subtasks. "
        "Copy requested input literals exactly, preserving language, spelling, whitespace "
        "and punctuation; never translate or paraphrase them. Do not invent values or "
        "actions on unseen controls. Coordinates use the original screenshot dimensions. "
    )
    + ACTION_SCHEMA,
    "original": "Choose the next executable action(s) for the task in this screenshot. "
    + ACTION_SCHEMA,
    "improved": (
        "Read the visible screenshot and choose the next executable action(s) for the task. "
        "Locate the intended control before choosing coordinates. Coordinates refer to the "
        "original screenshot size given below, even if the image is resized internally. "
        "Preserve the requested input text exactly, including punctuation and spaces. "
        "A key_press is one key; use hotkey for a simultaneous combination. "
        "Do not invent actions for controls that are not visible. "
    )
    + ACTION_SCHEMA,
    "targeted_sequence": (
        "Produce the complete ordered executable action sequence for the requested task "
        "in this screenshot. Include every required mouse, text-entry, key, hotkey and "
        "waiting step in execution order. A sequence can contain multiple actions; "
        "include necessary follow-up actions after clicking or entering text. "
        "When the task specifies text to enter, copy that literal character for character, "
        "preserving its spelling, case, punctuation, spaces and newlines. "
        "Do not invent an input value that the task does not specify or support. "
        "Locate controls in the visible screenshot and use its original pixel dimensions. "
        "Do not invent coordinates or additional tasks. "
    )
    + ACTION_SCHEMA,
    "current_step": (
        "The task is the overall goal, not a request to enumerate its entire future workflow. "
        "Use the current screenshot to choose the next bounded executable action chunk "
        "that makes immediate progress toward this goal. Include locally necessary focus, "
        "click, text-entry, key or confirmation actions only when justified by the goal "
        "and controls visible in this screenshot. The chunk can contain multiple actions "
        "on the same visible control; do not force it to a single action. "
        "Stop immediately after an action that opens a new screen, navigates, opens a menu "
        "or dialog, submits the current input, or otherwise requires fresh visual evidence "
        "before deciding the next action. Re-observe the screen before planning further actions. "
        "Do not plan actions on future or unseen controls, or assume earlier actions succeeded. "
        "When the goal specifies a literal to enter, copy that original literal character "
        "for character, preserving its language, spelling, case, punctuation, spaces and "
        "newlines. Do not translate, rewrite or invent the input value. "
        "Do not add saving or other post-processing that is not justified by the original "
        "goal and the current screenshot. Use original screenshot pixel dimensions. "
    )
    + ACTION_SCHEMA,
}


def prompt_rules(prompt_variant: str = "original") -> str:
    try:
        return _RULES[prompt_variant]
    except KeyError as exc:
        raise ValueError(f"unknown prompt variant: {prompt_variant}") from exc


def prompt_sha256(prompt_variant: str = "original") -> str:
    return hashlib.sha256(prompt_rules(prompt_variant).encode("utf-8")).hexdigest()


def prompt_input(sample: Mapping[str, Any], prompt_variant: str = "original") -> dict[str, Any]:
    """Whitelist request inputs, with an explicit opt-in to conditional acting.

    An actor_step must come from a separately verified pre-request context. This
    function neither extracts it from labels nor silently reads raw current_task.
    Existing task-only profiles keep their original three-field contract.
    """
    prompt_rules(prompt_variant)
    names = ("instruction", "image_width", "image_height")
    if prompt_variant == "conditional_actor":
        names += ("actor_step",)
    return {name: sample[name] for name in names}


def build_user_prompt(
    sample: Mapping[str, Any], prompt_variant: str = "original"
) -> str:
    instruction = sample.get("instruction")
    width, height = sample.get("image_width"), sample.get("image_height")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("a nonempty instruction is required")
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
    ):
        raise ValueError("positive integer screenshot dimensions are required")
    prefix = (
        f"{prompt_rules(prompt_variant)}\n\n"
        f"Screenshot size: {width} x {height} pixels.\nTask: {instruction}"
    )
    if prompt_variant == "conditional_actor":
        step = sample.get("actor_step")
        if not isinstance(step, str) or not step.strip():
            raise ValueError("conditional acting requires a nonempty pre-request actor_step")
        return prefix + f"\nCurrent subtask: {step}"
    return prefix


def action_target_json(actions: Sequence[Mapping[str, Any]]) -> str:
    if isinstance(actions, (str, bytes)) or not actions:
        raise ValueError("at least one action is required")
    return json.dumps(list(actions), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
