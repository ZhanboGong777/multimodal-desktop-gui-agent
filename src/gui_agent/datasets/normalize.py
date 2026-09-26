"""Helpers the adapters share: action vocabulary and text cleanup.

Each source dataset invents its own action verbs. ``normalize_action`` folds the
common ones onto the vocabulary this project implements; anything unrecognised is
passed through lower-cased and kept verbatim in ``GUIActionStep.raw_action`` so no
information is lost.
"""

from __future__ import annotations

import re
from typing import Any

WHITESPACE = re.compile(r"\s+")

# Source verb -> project verb. Keys are already casefolded with single spaces.
ACTION_ALIASES: dict[str, str] = {
    "click": "click",
    "left_click": "click",
    "left click": "click",
    "tap": "click",
    "select": "click",
    "select_option": "click",
    "select option": "click",
    "goto": "click",
    "go to": "click",
    "navigate": "click",
    "press_button": "click",
    "double_click": "double_click",
    "double click": "double_click",
    "right_click": "right_click",
    "right click": "right_click",
    "hover": "move",
    "move": "move",
    "mousemove": "move",
    "mouse_move": "move",
    "type": "type_text",
    "type_text": "type_text",
    "type text": "type_text",
    "input": "type_text",
    "fill": "type_text",
    "enter_text": "type_text",
    "enter text": "type_text",
    "press": "key_press",
    "keypress": "key_press",
    "key_press": "key_press",
    "key": "key_press",
    "hotkey": "hotkey",
    "scroll": "scroll",
    "scroll_down": "scroll",
    "swipe": "scroll",
    "drag": "drag",
    "drag_and_drop": "drag",
    "wait": "wait",
    "waitaction": "wait",
    "sleep": "wait",
    "delay": "wait",
    "finish": "finish",
    "done": "finish",
    "stop": "finish",
    "answer": "finish",
    "terminate": "finish",
}


def clean_text(value: Any) -> str:
    """Collapse whitespace and drop non-string values, matching the OCR layer."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return WHITESPACE.sub(" ", value).strip()


def normalize_action(verb: Any) -> str:
    """Fold a source action verb onto the project's vocabulary.

    Unknown verbs are returned lower-cased with underscores rather than rejected:
    the sample is still useful, and the planner is the layer that decides which
    verbs it can actually execute.
    """
    key = clean_text(verb).casefold().replace("-", "_")
    if key in ACTION_ALIASES:
        return ACTION_ALIASES[key]
    return key or "unknown"


def first_present(record: dict[str, Any], *keys: str) -> Any:
    """Return the first key whose value is neither None nor an empty string."""
    for key in keys:
        if key in record:
            value = record[key]
            if value is not None and value != "":
                return value
    return None


def as_int(value: Any) -> int | None:
    """Best-effort integer conversion; returns None when it cannot be done."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
