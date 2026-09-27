"""ScreenAgent adapter.

ScreenAgent is the closest source to this project: desktop screenshots with an
action trajectory. The real archive is a zip of ``test/<session>/<timestamp>.json``
files, and each file is **one step** of a session, not a whole task.

Verified against ``data/ScreenAgent/test.zip`` (898 records). Every record has:

``task_prompt``       the overall task -> ``instruction``
``current_task``      the sub-task being attempted -> ``metadata``
``saved_image_name``  the screenshot file -> ``image_path``
``video_width/height`` the desktop size -> ``metadata``
``actions[]``         one entry per attempted action, tagged by ``action_type``

Four ``action_type`` values appear in practice and they are not the same kind of
thing:

``MouseAction``              real coordinates in ``mouse_position``
``KeyboardAction``           ``keyboard_text`` / ``keyboard_key``
``PlanAction``               a plan step, target named by ``element``
``EvaluateSubTaskAction``    an *evaluation* of the previous step, not an action

The last one is deliberately dropped from ``actions`` and kept verbatim in
``metadata.evaluations``: counting it as a step would inflate every trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..schemas import BoundingBox, Point
from .base import DatasetAdapter
from .normalize import as_int, clean_text, first_present, normalize_action
from .schemas import GUIActionStep, GUITaskSample

#: action_type values that describe an evaluation rather than an action.
EVALUATION_ACTION_TYPES = frozenset({"evaluatesubtaskaction"})

#: ScreenAgent verb -> project verb, within each action family.
MOUSE_VERBS = {
    "click": "click",
    "left_click": "click",
    "leftclick": "click",
    "double_click": "double_click",
    "doubleclick": "double_click",
    "right_click": "right_click",
    "rightclick": "right_click",
    "move": "move",
    "drag": "drag",
    "scroll": "scroll",
    "scroll_up": "scroll",
    "scroll_down": "scroll",
}
KEYBOARD_VERBS = {
    "type": "type_text",
    "write": "type_text",
    "input": "type_text",
    "press": "key_press",
    "key": "key_press",
    "hotkey": "hotkey",
    "combination": "hotkey",
}


class ScreenAgentAdapter(DatasetAdapter):
    name = "screenagent"

    def to_sample(self, record: Mapping[str, Any], *, split: str | None = None) -> GUITaskSample:
        payload = dict(record)
        instruction = clean_text(
            first_present(payload, "task_prompt", "instruction", "task", "goal", "task_description")
        )
        if not instruction:
            raise ValueError("ScreenAgent record has no task_prompt")

        sample_id = clean_text(
            first_present(payload, "session_id", "task_id", "id", "sample_id", "uid")
        )
        if not sample_id:
            image = clean_text(first_present(payload, "saved_image_name", "image"))
            sample_id = image.rsplit(".", 1)[0] or f"screenagent-{abs(hash(instruction)) % 10**8}"

        image_path = clean_text(
            first_present(payload, "saved_image_name", "image", "image_path", "screenshot")
        )

        raw_actions = self._raw_actions(payload)
        actions: list[GUIActionStep] = []
        evaluations: list[dict[str, Any]] = []
        skipped = 0
        for raw in raw_actions:
            if not isinstance(raw, Mapping):
                skipped += 1
                continue
            if clean_text(raw.get("action_type")).casefold() in EVALUATION_ACTION_TYPES:
                evaluations.append(dict(raw))
                continue
            step = self._to_step(len(actions), raw)
            if step is None:
                skipped += 1
                continue
            actions.append(step)

        metadata: dict[str, Any] = {
            "current_task": clean_text(payload.get("current_task")) or None,
            "status": clean_text(payload.get("status")) or None,
            "video_width": as_int(payload.get("video_width")),
            "video_height": as_int(payload.get("video_height")),
            "evaluations": evaluations,
            "skipped_actions": skipped,
        }

        return GUITaskSample(
            sample_id=sample_id,
            dataset_name=self.name,
            instruction=instruction,
            image_path=image_path or None,
            observation=clean_text(payload.get("current_task")) or None,
            actions=actions,
            source_split=split,
            metadata={k: v for k, v in metadata.items() if v not in (None, [], 0)},
        )

    @staticmethod
    def _raw_actions(record: Mapping[str, Any]) -> Sequence[Any]:
        for key in ("actions", "action_list", "trajectory", "steps", "operations"):
            value = record.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return value
        return []

    def _to_step(self, index: int, raw: Mapping[str, Any]) -> GUIActionStep | None:
        payload = dict(raw)
        family = clean_text(payload.get("action_type")).casefold()

        # A record with no verb at all is not a step.
        if not family:
            return None

        coordinates = self._coordinates(payload)
        input_text = clean_text(
            first_present(payload, "keyboard_text", "value", "text_input", "input", "content")
        )
        target = clean_text(
            first_present(payload, "element", "target", "label", "text", "clickable_area")
        )
        bbox = self._bbox(payload.get("clickable_area"))

        if family == "mouseaction":
            verb = MOUSE_VERBS.get(
                clean_text(payload.get("mouse_action_type")).casefold().replace(" ", "_"),
                "click",
            )
        elif family == "keyboardaction":
            verb = KEYBOARD_VERBS.get(
                clean_text(payload.get("keyboard_action_type")).casefold().replace(" ", "_"),
                "type_text" if input_text else "key_press",
            )
            # A single key lives in keyboard_key; free text in keyboard_text.
            if verb == "key_press" and not input_text:
                input_text = clean_text(payload.get("keyboard_key"))
        else:
            # PlanAction and anything unseen: keep the verb, let the planner judge.
            verb = normalize_action(family)

        return GUIActionStep(
            step_index=index,
            action_type=verb,
            target_text=target or None,
            target_bbox=bbox,
            coordinates=coordinates,
            input_text=input_text or None,
            raw_action=payload,
        )

    @staticmethod
    def _coordinates(payload: Mapping[str, Any]) -> Point | None:
        position = payload.get("mouse_position")
        x = y = None
        if isinstance(position, Mapping):
            x = as_int(first_present(dict(position), "x", "left"))
            y = as_int(first_present(dict(position), "y", "top"))
        elif isinstance(position, Sequence) and not isinstance(position, (str, bytes)):
            if len(position) >= 2:
                x, y = as_int(position[0]), as_int(position[1])
        if x is None:
            x = as_int(first_present(dict(payload), "x", "coord_x", "position_x"))
        if y is None:
            y = as_int(first_present(dict(payload), "y", "coord_y", "position_y"))
        return Point(x=x, y=y) if x is not None and y is not None else None

    @staticmethod
    def _bbox(value: Any) -> BoundingBox | None:
        box = value
        if isinstance(box, Mapping):
            left = as_int(first_present(dict(box), "left", "x", "xmin"))
            top = as_int(first_present(dict(box), "top", "y", "ymin"))
            right = as_int(first_present(dict(box), "right", "xmax"))
            bottom = as_int(first_present(dict(box), "bottom", "ymax"))
            box = [left, top, right, bottom]
        if isinstance(box, Sequence) and not isinstance(box, (str, bytes)) and len(box) == 4:
            left, top, right, bottom = (as_int(v) for v in box)
            if None not in (left, top, right, bottom):
                try:
                    return BoundingBox(left=left, top=top, right=right, bottom=bottom)
                except ValueError:
                    return None
        return None
