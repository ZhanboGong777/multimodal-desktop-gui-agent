"""Mind2Web adapter.

Mind2Web describes web tasks: an instruction, a website, and steps whose targets
are HTML elements rather than screen coordinates. The element description is the
closest thing to a ``target_text``, so it is promoted there and the full element
record stays in ``raw_action``.

Two properties of the real archive are easy to get wrong, and both were wrong in
the first version of this adapter - which the fixture could not detect, because the
fixture was written from the same wrong assumption:

1. **One row is one step, not one task.** ``action_reprs`` holds the whole task and
   ``target_action_index`` says which of those steps this row is. Treating
   ``action_reprs`` as the trajectory multiplied every task by the length of its own
   action list.
2. **The verb is at the end.** A repr reads
   ``"[textbox]  US City,State or Zip Code -> TYPE: 08817"``: tag, element text,
   then the operation. Splitting on the first space made the HTML tag the verb and
   the element text plus its operation the target, so every ``action_type`` came out
   as ``[button]``, ``[link]``, ``[textbox]``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .base import DatasetAdapter
from .normalize import as_int, clean_text, first_present, normalize_action
from .schemas import GUIActionStep, GUITaskSample

#: ``"[textbox]  US City,State or Zip Code -> TYPE: 08817"``
_ACTION_REPR = re.compile(
    r"^\s*(?:\[(?P<tag>[^\]]+)\]\s*)?(?P<text>.*?)\s*->\s*"
    r"(?P<operation>[A-Za-z_]+)\s*(?::\s*(?P<value>.*))?$"
)


class Mind2WebAdapter(DatasetAdapter):
    name = "mind2web"

    def to_sample(self, record: Mapping[str, Any], *, split: str | None = None) -> GUITaskSample:
        payload = dict(record)
        instruction = clean_text(
            first_present(payload, "instruction", "confirmed_task", "task", "goal")
        )
        if not instruction:
            raise ValueError("Mind2Web record has no instruction")

        # Every row of a task carries the same ``annotation_id``, so the step index is
        # what makes a sample identifiable.
        step_index = as_int(payload.get("target_action_index"))
        base_id = (
            clean_text(first_present(payload, "annotation_id", "task_id", "id", "sample_id"))
            or f"mind2web-{abs(hash(instruction)) % 10**8}"
        )
        sample_id = base_id if step_index is None else f"{base_id}-step{step_index}"

        actions = [
            step
            for step in (
                self._to_step(index, raw) for index, raw in enumerate(self._raw_actions(payload))
            )
            if step is not None
        ]

        image_path, screenshot_bytes = self._screenshot(payload)
        metadata: dict[str, Any] = {
            "website": clean_text(payload.get("website")) or None,
            "domain": clean_text(payload.get("domain")) or None,
            "subdomain": clean_text(payload.get("subdomain")) or None,
            "target_action_index": step_index,
            "screenshot_bytes": screenshot_bytes,
        }

        return GUITaskSample(
            sample_id=sample_id,
            dataset_name=self.name,
            instruction=instruction,
            image_path=image_path,
            actions=actions,
            source_split=split,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    @staticmethod
    def _screenshot(payload: Mapping[str, Any]) -> tuple[str | None, int | None]:
        """Return the screenshot path and, when the bytes are embedded, their size."""
        shot = payload.get("screenshot")
        if isinstance(shot, Mapping):
            path = clean_text(shot.get("path")) or None
            data = shot.get("bytes")
            return path, (len(data) if isinstance(data, bytes) else None)
        return clean_text(first_present(payload, "screenshot", "image_path", "image")) or None, None

    @staticmethod
    def _raw_actions(record: Mapping[str, Any]) -> Sequence[Any]:
        """The action(s) this row represents.

        ``action_reprs`` lists the whole task; ``target_action_index`` selects the one
        step the row is actually about. Without that selection a 6-step task yields 6
        copies of a 6-action trajectory.
        """
        reprs = record.get("action_reprs")
        if isinstance(reprs, Sequence) and not isinstance(reprs, (str, bytes)) and reprs:
            index = as_int(record.get("target_action_index"))
            if index is not None and 0 <= index < len(reprs):
                return [reprs[index]]
            return list(reprs)
        for key in ("actions", "steps", "pos_candidates"):
            value = record.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return value
        return []

    def _to_step(self, index: int, raw: Any) -> GUIActionStep | None:
        if isinstance(raw, str):
            parsed = self._parse_action_repr(raw)
            return GUIActionStep(
                step_index=index,
                action_type=normalize_action(parsed["operation"] or ""),
                target_text=parsed["text"],
                input_text=parsed["value"],
                raw_action={
                    "repr": raw,
                    "element_tag": parsed["tag"],
                    "operation": parsed["operation"],
                },
            )
        if not isinstance(raw, Mapping):
            return None

        payload = dict(raw)
        verb = first_present(payload, "operation", "action", "action_type", "type")
        element = first_present(payload, "element", "target", "node")
        target_text = None
        if isinstance(element, Mapping):
            target_text = (
                clean_text(
                    first_present(dict(element), "text", "inner_text", "label", "title", "value")
                )
                or None
            )
        else:
            target_text = clean_text(element) or None

        return GUIActionStep(
            step_index=index,
            action_type=normalize_action(verb),
            target_text=target_text,
            input_text=clean_text(first_present(payload, "value", "input", "typed")) or None,
            raw_action=payload,
        )

    @staticmethod
    def _parse_action_repr(raw: str) -> dict[str, str | None]:
        """Split ``"[textbox]  US City,State or Zip Code -> TYPE: 08817"``.

        The element text is matched lazily so that a ``->`` inside it does not swallow
        the operation: the whole string only matches when the final ``-> VERB`` is
        reached.
        """
        match = _ACTION_REPR.match(raw)
        if match is None:
            # No "-> OPERATION" tail. Keep the text rather than dropping the step.
            return {"tag": None, "text": clean_text(raw) or None, "operation": None, "value": None}
        return {
            "tag": (match.group("tag") or "").strip() or None,
            "text": (match.group("text") or "").strip() or None,
            "operation": (match.group("operation") or "").strip() or None,
            "value": (match.group("value") or "").strip() or None,
        }
