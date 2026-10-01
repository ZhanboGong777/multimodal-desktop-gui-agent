"""Turn a validated plan step into an executable desktop action.

The step is a description ("click the search box"); the action needs a coordinate
on the current screen. Everything in this module exists to make that translation
refuse rather than guess.

Two rules carry most of the weight:

* an element id only means something inside the observation it came from, so a
  step that only names an id from an older frame is rejected, not re-resolved;
* more than one matching candidate is an ambiguity to report, never a reason to
  take the first hit.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from typing import Any

from ..coordinates import is_inside_screen, screenshot_to_control
from ..perception.grounding import find_text
from ..planning import PlanStep
from ..schemas import BoundingBox, DesktopAction, Point
from .schemas import ElementRef, ObservationSnapshot, ResolvedAction


class ActionResolutionError(RuntimeError):
    """Raised when a step cannot be turned into a safe desktop action."""


#: Keys the adapter will pass through. Anything else is refused rather than
#: forwarded to PyAutoGUI, which accepts a much wider vocabulary than this
#: project has any business pressing.
ALLOWED_KEYS = frozenset(
    {
        "enter",
        "return",
        "tab",
        "esc",
        "escape",
        "space",
        "backspace",
        "delete",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        *"abcdefghijklmnopqrstuvwxyz",
        *"0123456789",
    }
)

#: Modifier names differ per platform; the plan may say either, the adapter
#: translates to the one the running machine actually has.
_MODIFIER_ALIASES = {
    "darwin": {
        "ctrl": "command",
        "control": "command",
        "cmd": "command",
        "command": "command",
        "alt": "option",
        "option": "option",
        "shift": "shift",
        "win": "command",
        "super": "command",
    },
    "win32": {
        "cmd": "ctrl",
        "command": "ctrl",
        "control": "ctrl",
        "ctrl": "ctrl",
        "win": "win",
        "super": "win",
        "alt": "alt",
        "shift": "shift",
    },
}
_DEFAULT_MODIFIERS = _MODIFIER_ALIASES["win32"]

#: Actions whose target is a point on screen.
_POINT_ACTIONS = frozenset({"move", "click", "double_click", "right_click", "scroll"})

_TOKEN = re.compile(r"[0-9a-z_]+")

#: A one-word query is not enough to anchor a run ("to" appears everywhere), and
#: the run search exists for targets that span more than one element.
_MIN_RUN_TOKENS = 2


def _tokens(text: str) -> list[str]:
    """Word tokens of a label, so 'Summary (required)' -> ['summary', 'required']."""
    return _TOKEN.findall(text.casefold())


def _reading_order(elements: Sequence[ElementRef]) -> list[ElementRef]:
    """Top to bottom, then left to right.

    The observation stores elements ranked by confidence, which is the right order
    for the prompt's truncation cap and the wrong one for "next to each other".
    """
    return sorted(elements, key=lambda item: (item.bounding_box.top, item.bounding_box.left))


def _join_run(run: Sequence[ElementRef]) -> ElementRef:
    """Present a run of elements as the single target the step asked for."""
    box = BoundingBox(
        left=min(item.bounding_box.left for item in run),
        top=min(item.bounding_box.top for item in run),
        right=max(item.bounding_box.right for item in run),
        bottom=max(item.bounding_box.bottom for item in run),
    )
    return ElementRef(
        element_id="+".join(item.element_id for item in run),
        text=" ".join(item.text for item in run),
        bounding_box=box,
        center=box.center,
        confidence=min(item.confidence for item in run),
        source=run[0].source,
    )


def _match_adjacent_runs(
    elements: Sequence[ElementRef], query: str, min_confidence: float
) -> list[list[ElementRef]]:
    """Find runs of neighbouring elements that together spell out ``query``.

    A word-level OCR backend splits "Summary (required)" into two elements, and a
    plan may also name two adjacent labels as one target. Both are invisible to an
    exact-text search, so the query's tokens are matched against a run of
    consecutive elements. The run has to consume the query exactly - no partial or
    loose matches - and every match is returned so the caller can still refuse an
    ambiguous one.
    """
    wanted = _tokens(query)
    if len(wanted) < _MIN_RUN_TOKENS:
        return []
    ordered = [
        item
        for item in _reading_order(elements)
        if item.text.strip() and item.confidence >= min_confidence
    ]
    runs: list[list[ElementRef]] = []
    for start in range(len(ordered)):
        collected: list[str] = []
        for end in range(start, len(ordered)):
            collected.extend(_tokens(ordered[end].text))
            if len(collected) > len(wanted):
                break
            if collected == wanted:
                runs.append(list(ordered[start : end + 1]))
                break
    return runs


def platform_name() -> str:
    return "darwin" if sys.platform == "darwin" else "win32"


def _as_ref(element: Any, observation: ObservationSnapshot) -> ElementRef:
    """Find the frame-local ref for an element returned by ``find_text``."""
    for item in observation.elements:
        if item.element_id == getattr(element, "element_id", None):
            return item
    # find_text may return its own wrapper; fall back to matching on geometry.
    box = getattr(element, "bounding_box", None)
    for item in observation.elements:
        if box is not None and item.bounding_box == box:
            return item
    raise ActionResolutionError("matched element is not part of this observation")


class ActionAdapter:
    """Resolves plan steps against the current observation."""

    def __init__(self, *, platform: str | None = None, min_confidence: float = 0.35) -> None:
        self.platform = platform or platform_name()
        self.min_confidence = min_confidence

    # ── public entry point ─────────────────────────────────────────────
    def resolve(self, step: PlanStep, observation: ObservationSnapshot) -> ResolvedAction:
        """Return the action for ``step``, or raise if it cannot be resolved safely."""
        kind = step.action_type
        if kind == "finish":
            raise ActionResolutionError(
                "finish is a planning verb and must be handled by the runner, never executed"
            )
        if kind in _POINT_ACTIONS:
            return self._resolve_point(step, observation)
        if kind == "drag":
            return self._resolve_drag(step, observation)
        if kind == "type_text":
            return self._resolve_type(step)
        if kind == "key_press":
            return self._resolve_key(step)
        if kind == "hotkey":
            return self._resolve_hotkey(step)
        if kind == "wait":
            return self._resolve_wait(step)
        raise ActionResolutionError(f"unsupported action type {kind!r}")

    # ── target location ────────────────────────────────────────────────
    def _locate(
        self, step: PlanStep, observation: ObservationSnapshot
    ) -> tuple[Point, ElementRef | None, str]:
        """Find the point this step refers to.

        Preference order: an element id from *this* frame, then a unique text match
        in this frame. An id from another frame is an error even when the text
        would match - silently rebinding it is how a plan ends up clicking
        yesterday's coordinates.
        """
        arguments: dict[str, Any] = dict(step.arguments or {})
        element_id = arguments.get("element_id") or arguments.get("target_element")
        query = (step.target_text or arguments.get("target") or "").strip()
        rebound_from: str | None = None

        if isinstance(element_id, str) and element_id:
            found = observation.element(element_id)
            if found is not None:
                return found.center, found, f"element {element_id}"
            # The id belongs to an earlier frame. A plan is written from one
            # observation and executed after another, so this is the normal case,
            # not an error - as long as the step also said what it is aiming at in
            # words. A bare stale id has nothing to re-locate and is refused.
            if not query:
                raise ActionResolutionError(
                    f"element id {element_id!r} is not from observation "
                    f"{observation.observation_id} and the step names no text target "
                    "to re-locate"
                )
            rebound_from = element_id

        if not query:
            raise ActionResolutionError(
                f"{step.action_type} needs target_text or arguments.element_id to locate a target"
            )

        result = find_text(observation.elements, query, min_confidence=self.min_confidence)
        # find_text orders by confidence but also marks a selected_index; taking
        # the first candidate without checking the count is how an ambiguous query
        # silently becomes a click on the wrong control.
        matches = list(result.candidates)
        if len(matches) == 1:
            chosen = observation.element(matches[0].element.element_id) or matches[0].element
            ref = _as_ref(chosen, observation)
            note = f"text {query!r} -> {ref.element_id}"
            if rebound_from:
                note += f" (re-bound from {rebound_from} in an earlier frame)"
            return ref.center, ref, note
        if not matches:
            # A word-level backend, or a plan that names two adjacent labels as one
            # target, reaches here with a query that is genuinely on screen but is
            # not any single element's text.
            runs = _match_adjacent_runs(observation.elements, query, self.min_confidence)
            if not runs:
                raise ActionResolutionError(
                    f"no element matches {query!r} in {observation.observation_id}"
                )
            if len(runs) > 1:
                listed = ", ".join("+".join(item.element_id for item in run) for run in runs[:5])
                raise ActionResolutionError(
                    f"{len(runs)} element runs match {query!r} ({listed}); "
                    "refusing to pick one arbitrarily"
                )
            ref = _join_run(runs[0])
            note = f"text {query!r} -> {ref.element_id}"
            if len(runs[0]) > 1:
                note += f" (joined from {len(runs[0])} adjacent elements)"
            if rebound_from:
                note += f" (re-bound from {rebound_from} in an earlier frame)"
            return ref.center, ref, note
        listed = ", ".join(f"{m.element.element_id}:{m.element.text!r}" for m in matches[:5])
        raise ActionResolutionError(
            f"{len(matches)} elements match {query!r} ({listed}); refusing to pick one arbitrarily"
        )

    def _to_control(self, point: Point, observation: ObservationSnapshot) -> Point:
        control = screenshot_to_control(point, observation.screen_info)
        if not is_inside_screen(control, observation.screen_info):
            raise ActionResolutionError(
                f"screenshot point ({point.x},{point.y}) maps to control "
                f"({control.x},{control.y}), outside the captured monitor"
            )
        return control

    # ── per-action resolution ──────────────────────────────────────────
    def _resolve_point(self, step: PlanStep, observation: ObservationSnapshot) -> ResolvedAction:
        point, element, note = self._locate(step, observation)
        control = self._to_control(point, observation)
        kwargs: dict[str, Any] = {"x": control.x, "y": control.y}
        if step.action_type == "scroll":
            amount = (step.arguments or {}).get(
                "scroll_amount", (step.arguments or {}).get("amount")
            )
            if amount is None:
                raise ActionResolutionError("scroll requires arguments.scroll_amount")
            # A wrong-typed amount has to leave as the module's own error: the
            # runner catches ActionResolutionError and records it, so a bare
            # ValueError from int() would escape the run as a traceback instead.
            try:
                kwargs["scroll_amount"] = int(amount)
            except (TypeError, ValueError) as exc:
                raise ActionResolutionError(
                    f"scroll_amount must be a number, got {amount!r}"
                ) from exc
        action = DesktopAction(
            action_type=step.action_type,  # type: ignore[arg-type]
            target_description=step.target_text,
            **kwargs,
        )
        return ResolvedAction(
            step_id=step.step_id,
            action=action,
            element_id=element.element_id if element else None,
            screenshot_point=point,
            control_point=control,
            note=note,
        )

    def _resolve_drag(self, step: PlanStep, observation: ObservationSnapshot) -> ResolvedAction:
        arguments = dict(step.arguments or {})
        start_id = arguments.get("start_element_id") or arguments.get("from_element_id")
        end_id = arguments.get("end_element_id") or arguments.get("to_element_id")
        if not start_id or not end_id:
            raise ActionResolutionError(
                "drag needs arguments.start_element_id and arguments.end_element_id"
            )
        start = observation.element(str(start_id))
        end = observation.element(str(end_id))
        if start is None or end is None:
            missing = start_id if start is None else end_id
            raise ActionResolutionError(
                f"drag element {missing!r} is not from observation {observation.observation_id}"
            )
        control_start = self._to_control(start.center, observation)
        control_end = self._to_control(end.center, observation)
        try:
            duration = float(arguments.get("duration", 0.5))
        except (TypeError, ValueError) as exc:
            raise ActionResolutionError(
                f"drag duration must be a number, got {arguments.get('duration')!r}"
            ) from exc
        action = DesktopAction(
            action_type="drag",
            start=control_start,
            end=control_end,
            duration=duration,
            target_description=step.target_text,
        )
        return ResolvedAction(
            step_id=step.step_id,
            action=action,
            screenshot_point=start.center,
            control_point=control_start,
            note=f"drag {start_id} -> {end_id}",
        )

    def _resolve_type(self, step: PlanStep) -> ResolvedAction:
        text = (step.arguments or {}).get("text")
        if not (text or "").strip():
            raise ActionResolutionError("type_text requires a non-empty arguments.text")
        action = DesktopAction(
            action_type="type_text", text=text, target_description=step.target_text
        )
        return ResolvedAction(step_id=step.step_id, action=action, note=f"type {len(text)} chars")

    def _resolve_key(self, step: PlanStep) -> ResolvedAction:
        key = str((step.arguments or {}).get("key", "")).strip().casefold()
        if not key:
            raise ActionResolutionError("key_press requires arguments.key")
        if key not in ALLOWED_KEYS:
            raise ActionResolutionError(f"key {key!r} is not in the allowed key set")
        action = DesktopAction(
            action_type="key_press", key=key, target_description=step.target_text
        )
        return ResolvedAction(step_id=step.step_id, action=action, note=f"press {key}")

    def _resolve_hotkey(self, step: PlanStep) -> ResolvedAction:
        raw = (step.arguments or {}).get("keys")
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ActionResolutionError("hotkey requires a non-empty arguments.keys list")
        aliases = _MODIFIER_ALIASES.get(self.platform, _DEFAULT_MODIFIERS)
        resolved: list[str] = []
        for item in raw:
            token = str(item).strip().casefold()
            if not token:
                continue
            mapped = aliases.get(token, token)
            if mapped not in ALLOWED_KEYS and mapped not in {
                "command",
                "option",
                "ctrl",
                "alt",
                "shift",
                "win",
            }:
                raise ActionResolutionError(f"hotkey token {token!r} is not allowed")
            resolved.append(mapped)
        if not resolved:
            raise ActionResolutionError("hotkey resolved to an empty key list")
        action = DesktopAction(
            action_type="hotkey", keys=resolved, target_description=step.target_text
        )
        return ResolvedAction(step_id=step.step_id, action=action, note="+".join(resolved))

    def _resolve_wait(self, step: PlanStep) -> ResolvedAction:
        duration = (step.arguments or {}).get("duration", 1.0)
        try:
            seconds = float(duration)
        except (TypeError, ValueError) as exc:
            raise ActionResolutionError(
                f"wait duration must be a number, got {duration!r}"
            ) from exc
        if not 0 < seconds <= 10:
            raise ActionResolutionError(
                f"wait duration {seconds} is outside the allowed 0-10 s range"
            )
        action = DesktopAction(
            action_type="wait", duration=seconds, target_description=step.target_text
        )
        return ResolvedAction(step_id=step.step_id, action=action, note=f"wait {seconds}s")
