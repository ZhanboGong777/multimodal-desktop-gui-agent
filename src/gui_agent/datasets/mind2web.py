"""Mind2Web adapter.

Mind2Web describes web tasks: an instruction, a website, and a list of steps
whose targets are HTML elements rather than screen coordinates. The element
description is the closest thing to a ``target_text``, so it is promoted there and
the full element record stays in ``raw_action``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import DatasetAdapter
from .normalize import clean_text, first_present, normalize_action
from .schemas import GUIActionStep, GUITaskSample


class Mind2WebAdapter(DatasetAdapter):
    name = "mind2web"

    def to_sample(self, record: Mapping[str, Any], *, split: str | None = None) -> GUITaskSample:
        payload = dict(record)
        instruction = clean_text(
            first_present(payload, "instruction", "confirmed_task", "task", "goal")
        )
        if not instruction:
            raise ValueError("Mind2Web record has no instruction")

        sample_id = (
            clean_text(first_present(payload, "annotation_id", "task_id", "id", "sample_id"))
            or f"mind2web-{abs(hash(instruction)) % 10**8}"
        )

        actions = [
            step
            for step in (
                self._to_step(index, raw) for index, raw in enumerate(self._raw_actions(payload))
            )
            if step is not None
        ]

        metadata: dict[str, Any] = {
            "website": clean_text(payload.get("website")) or None,
            "domain": clean_text(payload.get("domain")) or None,
            "subdomain": clean_text(payload.get("subdomain")) or None,
        }

        return GUITaskSample(
            sample_id=sample_id,
            dataset_name=self.name,
            instruction=instruction,
            image_path=clean_text(first_present(payload, "screenshot", "image_path", "image"))
            or None,
            actions=actions,
            source_split=split,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    @staticmethod
    def _raw_actions(record: Mapping[str, Any]) -> Sequence[Any]:
        for key in ("actions", "action_reprs", "steps", "pos_candidates"):
            value = record.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return value
        return []

    def _to_step(self, index: int, raw: Any) -> GUIActionStep | None:
        if isinstance(raw, str):
            # ``action_reprs`` is a plain list of strings such as "CLICK [Submit]".
            verb, _, rest = raw.partition(" ")
            return GUIActionStep(
                step_index=index,
                action_type=normalize_action(verb),
                target_text=clean_text(rest.strip("[]")) or None,
                raw_action={"repr": raw},
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
