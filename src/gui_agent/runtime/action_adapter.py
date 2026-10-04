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

def keys_for_platform(platform: str | None = None) -> dict[str, list[str]]:
    """The key names a plan may use on this platform, in the form the prompt sends.

    8.1.2 asks for the platform *and the allowed key names* to reach the model. The
    prompt told it to "use this platform's key names" and never said which, so a
    plan could name a key the adapter then refused - a resolution error the model
    had no way to avoid. Read from the same constants the adapter checks against,
    so the two cannot disagree.
    """
    modifiers = sorted(_MODIFIER_ALIASES.get(platform or "", _DEFAULT_MODIFIERS))
    return {
        "keys": sorted(ALLOWED_KEYS),
        "modifiers": modifiers,
        "hotkey_separator": "+",
    }


#: Actions whose target is a point on screen.
_POINT_ACTIONS = frozenset({"move", "click", "double_click", "right_click", "scroll"})

_TOKEN = re.compile(r"[0-9a-z_]+")

#: A one-word query is not enough to anchor a run ("to" appears everywhere), and
#: the run search exists for targets that span more than one element.
_MIN_RUN_TOKENS = 2

#: Longest single ``wait`` step accepted by default. The run's own limit comes from
#: ``execution.max_wait_seconds``; this is what a directly constructed adapter uses.
DEFAULT_MAX_WAIT_SECONDS = 5.0

#: Window classes that mean "this pixel belongs to the desktop". ``FolderView`` is the
#: shell's icon host (SysListView32 inside it on older builds); ``Progman`` is the
#: desktop window itself. Both were observed on the Windows node while probing clicks.
_DESKTOP_CLASSES = ("FolderView", "SysListView32", "Progman", "WorkerW")

#: A desktop shortcut's name, not a sentence. Longer text is a label in a window.
_MAX_SHORTCUT_LABEL = 40

#: How far above the label the icon starts, as a fraction of the label's width. Measured
#: on the review machine: 'Microsoft' spans 80 px at y=379, and the shortcut answers a
#: double-click at y=300, which is 79 px above the label's top - so 0.98 of its width.
#: The fraction is used rather than a pixel constant because a desktop scaled differently
#: scales the icon and its label together.
_ICON_LIFT = 0.98

#: Fraction of the screen width treated as an icon column at each edge. A desktop keeps
#: its shortcuts in narrow strips; 0.12 of 2560 px is 307 px, which covers the two
#: columns seen on the review machine (labels ending near x=2513) without reaching into
#: the middle of the screen where windows live.
_ICON_COLUMN_FRACTION = 0.12

#: Below this frame width the icon columns are not applied at all. See
#: :func:`_in_icon_column`: a fraction of a small frame is not a strip at the edge of a
#: desktop, it is most of the frame.
_MIN_DESKTOP_WIDTH = 1920


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


#: Greatest gap, in pixels, between two elements that can still be one target. A
#: word-level backend leaves a few pixels between the words of a label; a desktop
#: shortcut stacks its lines closely. Both are far below this.
_MAX_JOIN_GAP = 40.0

#: How much two elements have to overlap horizontally to count as stacked. A shortcut's
#: first line ("Microsoft") is wider than its second ("Edge"), so the overlap is real but
#: neither contains the other.
_MIN_JOIN_OVERLAP = 0.30


def _horizontal_overlap(first: ElementRef, second: ElementRef) -> float:
    """Overlap of two boxes along x, as a fraction of the narrower one."""
    left = max(first.bounding_box.left, second.bounding_box.left)
    right = min(first.bounding_box.right, second.bounding_box.right)
    if right <= left:
        return 0.0
    narrow = min(
        first.bounding_box.right - first.bounding_box.left,
        second.bounding_box.right - second.bounding_box.left,
    )
    return (right - left) / narrow if narrow > 0 else 0.0


def _is_next_in_label(first: ElementRef, second: ElementRef) -> bool:
    """True when ``second`` reads as the next part of ``first``'s label.

    Two shapes have to work, and they pull in different directions:

    * inline - "Current" then "repository" on one line, so ``second`` starts to the
      right of ``first`` with a word-sized gap;
    * stacked - a desktop shortcut's two lines, so ``second`` sits below ``first`` with
      its box overlapping horizontally.

    The stacked case is why reading order alone is not enough. On a desktop grid,
    neighbouring columns interleave: with 'Microsoft' at (2473, 386) and 'Edge' at
    (2473, 412), the element at (403, 412) sorts between them, so a search that steps
    through the sorted list never sees the two halves of the icon as neighbours.
    """
    first_box, second_box = first.bounding_box, second.bounding_box

    # inline: same line, second to the right
    vertical_gap = abs(second_box.top - first_box.top)
    horizontal_gap = second_box.left - first_box.right
    if vertical_gap <= _MAX_JOIN_GAP and 0 <= horizontal_gap <= _MAX_JOIN_GAP:
        return True

    # stacked: second below, boxes overlapping horizontally
    if _horizontal_overlap(first, second) >= _MIN_JOIN_OVERLAP:
        down_gap = second_box.top - first_box.bottom
        if -_MAX_JOIN_GAP <= down_gap <= _MAX_JOIN_GAP:
            return True
    return False


def _spatial_runs(
    elements: Sequence[ElementRef], wanted: Sequence[str], min_confidence: float
) -> list[list[ElementRef]]:
    """Runs of elements that are geometric neighbours and spell out ``wanted``.

    Searched as a path rather than a slice: from each element that matches the first
    token, walk to any neighbour that continues the query. This finds the two lines of
    a desktop shortcut even when unrelated elements sort between them.
    """
    usable = [
        item
        for item in elements
        if item.text.strip() and item.confidence >= min_confidence
    ]
    runs: list[list[ElementRef]] = []

    def walk(path: list[ElementRef], position: int) -> None:
        if position == len(wanted):
            runs.append(list(path))
            return
        tail = path[-1]
        for candidate in usable:
            if any(candidate.element_id == seen.element_id for seen in path):
                continue
            if not _is_next_in_label(tail, candidate):
                continue
            if _tokens(candidate.text) != [wanted[position]]:
                continue
            path.append(candidate)
            walk(path, position + 1)
            path.pop()

    for item in usable:
        if _tokens(item.text) == [wanted[0]]:
            walk([item], 1)
    return runs


def _desktop_icon_point(ref: ElementRef, screen_width: int | None) -> tuple[Point, str]:
    """Where to click for ``ref``, lifting a desktop shortcut's label onto its icon.

    A desktop shortcut is drawn as an icon with its name underneath, and OCR sees only
    the name. The icon's own pixels are therefore *above* the element, and the element's
    centre is a click on the label: that selects the shortcut and opens nothing.

    Measured on the Windows node, which is what identified this: the Microsoft Edge
    shortcut's label sits at (2433,379)-(2513,393) with 'Edge' beneath it at
    (2452,403)-(2494,421), and double-clicking the label's centre at three intervals
    (0.15 s, 0.30 s, 0.45 s) left the process count at zero every time, while a
    double-click at y=300 - inside the icon - reached the shortcut. Synthetic input was
    never the problem: typing into Notepad and the Win key both worked throughout.

    Two guards keep this off ordinary text. The label has to be short, and it has to sit
    in the left or right strip where a desktop keeps its icon columns - the second guard
    is what stops the check from firing on a fixture whose coordinates happen to land on
    a desktop pixel, which it did before the guard existed and which moved a click from
    y=30 to y=1 in two existing tests. The point is then confirmed to be the shell's own
    ``FolderView`` rather than an application.
    """
    if not ref.text.strip() or len(ref.text) > _MAX_SHORTCUT_LABEL:
        return ref.center, ""
    if not screen_width or not _in_icon_column(ref.center.x, screen_width):
        return ref.center, ""
    if not _point_is_desktop(ref.center):
        return ref.center, ""
    box = ref.bounding_box
    # The icon is about as tall as the label is wide, and sits directly above it.
    icon_size = box.right - box.left
    lifted = Point(x=ref.center.x, y=max(box.top - int(icon_size * _ICON_LIFT), 1))
    return lifted, (
        f" (click lifted from the label at y={ref.center.y} to the icon at y={lifted.y})"
    )


def _in_icon_column(x: int, screen_width: int) -> bool:
    """True when ``x`` is in the left or right strip where a desktop keeps its icons.

    Gated on the frame being a real desktop size. This is the one assumption in the
    lift, and it is stated rather than hidden: a 2560x1600 desktop keeps shortcuts in
    narrow edge strips, while a frame whose coordinates are a few hundred pixels across
    is a fixture or a scaled-down capture, and in those the strip fractions would swallow
    most of the width. Two existing tests failed exactly that way - a 1470-wide fixture
    put x=145 inside a 176-pixel strip and moved a click from y=30 to y=1.
    """
    if screen_width < _MIN_DESKTOP_WIDTH:
        return False
    margin = int(screen_width * _ICON_COLUMN_FRACTION)
    return x <= margin or x >= screen_width - margin


def _screen_width(observation: ObservationSnapshot) -> int | None:
    """The captured frame's width, or None when the frame does not say.

    The screenshot's own width is used rather than the control width: the element
    coordinates are in the screenshot's space, so the icon strips have to be measured
    against the same space.
    """
    info = getattr(observation, "screen_info", None)
    width = getattr(info, "screenshot_width", None)
    return int(width) if width else None


def _point_is_desktop(point: Point) -> bool:
    """True when the window under ``point`` is the desktop, not an application.

    Best-effort: a non-Windows host, or a failure to load user32, answers False, which
    leaves the click where it was. Guessing wrong in the other direction would move
    clicks inside ordinary windows, so the uncertain answer is the conservative one.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        user32 = ctypes.windll.user32
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.WindowFromPoint.argtypes = [POINT]
        user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        hwnd = user32.WindowFromPoint(POINT(int(point.x), int(point.y)))
        if not hwnd:
            return False
        buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buffer, 256)
        return buffer.value in _DESKTOP_CLASSES
    except Exception:  # noqa: BLE001 - any failure means "not known to be the desktop"
        return False


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

    Reading order is tried first because it is the stricter reading of "consecutive".
    Failing that, geometric neighbours are tried: the two lines of a desktop shortcut
    are adjacent on screen but are not necessarily adjacent in a top-to-bottom sort.
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
    if runs:
        return runs
    return _spatial_runs(elements, wanted, min_confidence)


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

    def __init__(
        self,
        *,
        platform: str | None = None,
        min_confidence: float = 0.35,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    ) -> None:
        self.platform = platform or platform_name()
        self.min_confidence = min_confidence
        self.max_wait_seconds = float(max_wait_seconds)

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
            point, offset_note = _desktop_icon_point(ref, _screen_width(observation))
            note = f"text {query!r} -> {ref.element_id}{offset_note}"
            if rebound_from:
                note += f" (re-bound from {rebound_from} in an earlier frame)"
            return point, ref, note
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
            point, offset_note = _desktop_icon_point(ref, _screen_width(observation))
            note = f"text {query!r} -> {ref.element_id}{offset_note}"
            if len(runs[0]) > 1:
                note += f" (joined from {len(runs[0])} adjacent elements)"
            if rebound_from:
                note += f" (re-bound from {rebound_from} in an earlier frame)"
            return point, ref, note
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

    #: Longest `target_text` that will be used as the text of a `type_text` step.
    #:
    #: The fallback below turns a label into content, which is only right while the label looks
    #: like content. Together with the no-space rule this is what separates a literal from a
    #: description: measured against the two shapes T04 actually produced, `'WEEK4_MESSAGE_CHECK_'`
    #: is accepted and `'the message box in the week4 test conversation'` is refused, which is
    #: the behaviour wanted - typing a field's description into the field is worse than stopping.
    MAX_INFERRED_TEXT = 64

    def _resolve_type(self, step: PlanStep) -> ResolvedAction:
        text = (step.arguments or {}).get("text")
        note = ""
        if not (text or "").strip():
            # Fall back to the step's own `target_text` before refusing.
            #
            # Measured on T04, four passes running: the model produced
            # `type_text text=None target='WEEK4_MESSAGE_CHECK_'` - the right verb, the right
            # string, in the wrong field of the step. `target_text` is where a step names the
            # literal it is about, and the prompt already requires it to be copied verbatim
            # from the screen list, so using it when `arguments.text` is empty costs nothing
            # and rescues a plan that is otherwise correct.
            #
            # Guarded rather than automatic, because the two fields do different jobs: a
            # `type_text` whose `target_text` is a description of a field would otherwise type
            # that description into it. Whitespace-only, multi-line and over-long targets are
            # refused as before, and the note says when the value was inferred so a reader can
            # tell it from one the model supplied.
            candidate = (step.target_text or "").strip()
            # No spaces, one line, and short. `target_text` is allowed to be prose - the prompt
            # asks for a copied screen text, and a step may name a window or a field - so the
            # guard is what keeps this from typing a description into a box. Content a person
            # asks a program to type is very often space-free, and the marker this exists for
            # is; a description with no spaces at all is not a thing anyone writes.
            usable = (
                candidate
                and len(candidate) <= self.MAX_INFERRED_TEXT
                and not any(ch.isspace() for ch in candidate)
            )
            if not usable:
                raise ActionResolutionError("type_text requires a non-empty arguments.text")
            text = candidate
            note = " (text taken from target_text; arguments.text was empty)"
        action = DesktopAction(
            action_type="type_text", text=text, target_description=step.target_text
        )
        return ResolvedAction(
            step_id=step.step_id, action=action, note=f"type {len(text)} chars{note}"
        )

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
        # Clamped to the ceiling rather than refused, and the note says so.
        #
        # Measured on T02: the model wrote `wait 10.0` against a 5 s ceiling, and because an
        # ActionResolutionError ends the whole run, a nine-step plan that had already
        # clicked, typed and submitted the search stopped on a wait. The search did land -
        # obs-0007 onwards show the results page and its URL - and the run was recorded as
        # failed with three good actions behind it.
        #
        # The other resolution failures are refusals for a reason: a target that is not on
        # screen, or an argument that is missing, means the step cannot be carried out at
        # all. Asking to wait *longer* is not that. Waiting longer cannot do the wrong thing
        # to the screen, so the ceiling is honoured by taking it and saying that is what
        # happened, instead of turning a harmless overshoot into a dead run.
        if not 0 < seconds:
            raise ActionResolutionError(
                f"wait duration must be positive, got {seconds}"
            )
        note = f"wait {seconds}s"
        if seconds > self.max_wait_seconds:
            note = (
                f"wait {seconds}s clamped to the {self.max_wait_seconds:g} s ceiling"
            )
            seconds = self.max_wait_seconds
        action = DesktopAction(
            action_type="wait", duration=seconds, target_description=step.target_text
        )
        return ResolvedAction(step_id=step.step_id, action=action, note=note)
