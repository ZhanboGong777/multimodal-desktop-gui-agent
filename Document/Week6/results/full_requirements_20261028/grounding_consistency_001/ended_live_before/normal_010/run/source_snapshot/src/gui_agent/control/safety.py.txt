"""Safety rules that run before any desktop action is executed."""

from __future__ import annotations

from typing import Any

from ..coordinates import is_inside_screen
from ..schemas import ActionResult, DesktopAction, ScreenInfo

# Modifier keys that would leak into other applications if they stayed pressed.
STICKY_KEYS = ("shift", "ctrl", "control", "alt", "option", "cmd", "command", "win", "super")

# Substrings that mark input which must never reach the log files.
SENSITIVE_HINTS = ("password", "passwd", "passphrase", "token", "secret", "api_key", "apikey")

REDACTED = "<redacted>"


class SafetyError(RuntimeError):
    """Raised when an action is refused before it can run."""


def is_sensitive_text(text: str | None) -> bool:
    """True when typed text looks like a credential."""
    if not text:
        return False
    lowered = text.casefold()
    return any(hint in lowered for hint in SENSITIVE_HINTS)


def validate_action(action: DesktopAction, screen: ScreenInfo) -> None:
    """Raise :class:`SafetyError` when the action must not be executed.

    Coordinates are checked for **every** point the action touches, so a drag
    validates both its start and its end.
    """
    for point in action.points:
        if not is_inside_screen(point, screen):
            raise SafetyError(
                f"point ({point.x}, {point.y}) is outside the captured monitor "
                f"[{screen.monitor_left}..{screen.monitor_left + screen.control_width - 1}] x "
                f"[{screen.monitor_top}..{screen.monitor_top + screen.control_height - 1}]"
            )
    if action.action_type == "drag" and action.start == action.end:
        raise SafetyError("drag start and end are identical")
    if action.duration is not None and action.duration < 0:
        raise SafetyError(f"duration must not be negative, got {action.duration}")


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Blank out typed or credential-like text in a serialised action.

    Returns a copy, so the action object the caller still holds is untouched.
    Every ``type_text`` payload is masked because a bare password or token value
    usually contains no reliable keyword that would identify it as sensitive.
    """
    text = payload.get("text")
    if isinstance(text, str) and (
        payload.get("action_type") == "type_text" or is_sensitive_text(text)
    ):
        return {**payload, "text": REDACTED}
    return dict(payload)


def redact_action(action: DesktopAction) -> dict[str, Any]:
    """Serialise an action for logging without leaking credentials."""
    return redact_payload(action.model_dump(mode="json", exclude_none=True))


def redact_result(result: ActionResult) -> dict[str, Any]:
    """Serialise an action result, redacting the action nested inside it.

    ``ActionResult`` carries its own ``DesktopAction``, so redacting only the
    top-level action would still write the credential to disk.
    """
    payload = result.model_dump(mode="json", exclude_none=True)
    nested = payload.get("action")
    if isinstance(nested, dict):
        payload["action"] = redact_payload(nested)
    return payload


def describe_action(action: DesktopAction) -> str:
    """Short, log-safe description of a planned action."""
    parts: list[str] = [action.action_type]
    if action.x is not None and action.y is not None:
        parts.append(f"at=({action.x},{action.y})")
    if action.start is not None and action.end is not None:
        parts.append(f"from=({action.start.x},{action.start.y})")
        parts.append(f"to=({action.end.x},{action.end.y})")
    if action.text is not None:
        # The console description also ends up in run.log, so mask typed text here too.
        typed = action.action_type == "type_text"
        parts.append(
            f"text={REDACTED if typed or is_sensitive_text(action.text) else action.text!r}"
        )
    if action.key:
        parts.append(f"key={action.key}")
    if action.keys:
        parts.append(f"keys={'+'.join(action.keys)}")
    if action.scroll_amount is not None:
        parts.append(f"amount={action.scroll_amount}")
    if action.duration is not None:
        parts.append(f"duration={action.duration}")
    return " ".join(parts)
