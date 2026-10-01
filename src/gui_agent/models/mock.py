"""Offline backend used by the tests and by both demonstration scripts.

It is deliberately rule-based rather than random: the same instruction always
produces the same plan, which is what makes it usable as a test oracle and as the
default when no API key is configured.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .base import ModelClient, ModelResponse

#: Verb the instruction contains -> the step that should open the plan.
_INTENT_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("browser", "browse", "website", "search", "url"), "click", "search"),
    (("open", "launch", "start"), "click", "open"),
    (("type", "enter", "write", "input"), "type_text", "type"),
    (("close", "quit", "exit"), "click", "close"),
    (("scroll",), "scroll", "scroll"),
    (("drag", "move"), "drag", "drag"),
    (("wait", "sleep"), "wait", "wait"),
)

_SPLIT = re.compile(r"\s*(?:,|;|then|and then|然后|接着|再)\s*", re.IGNORECASE)

#: One line of the element list the runtime puts in the user turn:
#:   obs-0002-e001  'Browser'  conf=0.90  center=(250,78)  box=(100,60,400,96)
_ELEMENT_LINE = re.compile(r"(obs-\d+-e\d+)\s+'([^']*)'")

#: OCR on a busy or low-contrast screen returns fragments like "@" or "HO". They
#: are real elements, but aiming at one makes the dry run die on a token that is
#: gone by the next frame, which says nothing about the pipeline.
_WORDLIKE = re.compile(r"[A-Za-z]{3,}")


def _unambiguous(elements: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """The elements whose text is contained in no other element's text.

    The adapter looks a target up with a substring match, so "main" is not a
    unique target when "Commit to main" is also on screen. When nothing is
    unambiguous the original list is returned: a dry run that stops on a genuinely
    ambiguous screen is the correct outcome, and the adapter's own tests cover it.
    """
    lowered = [(element_id, text.casefold()) for element_id, text in elements]
    unique = [
        (element_id, text)
        for element_id, text in elements
        if text.strip()
        and sum(1 for _, other in lowered if text.casefold() in other) == 1
    ]
    return unique or list(elements)


class MockModelClient(ModelClient):
    """Deterministic, network-free, free-of-charge."""

    name = "mock"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", "mock-vision-model")
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ModelResponse:
        instruction = self._instruction_from(messages)
        elements = self._elements_from(messages)
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "instruction": instruction,
                "elements": elements,
            }
        )
        plan = self._plan_for(instruction, kwargs.get("image_path"), elements)
        return ModelResponse(
            content=json.dumps(plan, ensure_ascii=False),
            model_name=self.model_name,
            provider=self.name,
            raw_response={"mock": True, "step_count": len(plan["steps"])},
        )

    @staticmethod
    def _instruction_from(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in reversed(list(messages)):
            if message.get("role") != "user":
                continue
            raw = str(message.get("content", ""))
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                # build_user_prompt emits plain text, not JSON: pull the
                # instruction line out rather than feeding the whole prompt back.
                for line in raw.splitlines():
                    if line.strip().casefold().startswith("instruction:"):
                        return line.split(":", 1)[1].strip()
                return raw.strip()
            if isinstance(payload, Mapping):
                return str(payload.get("instruction", "")).strip()
            return raw.strip()
        return ""

    @staticmethod
    def _elements_from(messages: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
        """Pull the frame's ``(element_id, text)`` pairs out of the user turn.

        The runtime lists the current observation there so a model can aim at
        something that exists. A rule-based backend that ignored that list produced
        plans naming elements which were not on screen, which the adapter then
        refused - correct, but it made every dry run stop at step one.
        """
        found: list[tuple[str, str]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            for element_id, text in _ELEMENT_LINE.findall(content):
                if text.strip():
                    found.append((element_id, text.strip()))
        return found

    def _pick_target(
        self, clause: str, elements: Sequence[tuple[str, str]]
    ) -> tuple[str, str] | None:
        """The element this clause is most likely about, or the first labelled one.

        The text has to resolve uniquely later: the adapter refuses an ambiguous
        target, so choosing a word like "the" - which also occurs inside half the
        other elements - stops the run for a reason that has nothing to do with the
        pipeline. Candidates are therefore narrowed to the elements whose text
        occurs in no other element before anything is picked from them.
        """
        if not elements:
            return None
        lowered = clause.casefold()
        candidates = _unambiguous(elements)
        readable = [item for item in candidates if _WORDLIKE.search(item[1])]
        candidates = readable or candidates
        for element_id, text in candidates:
            if text.casefold() in lowered:
                return element_id, text
        head = re.findall(r"[A-Za-z]{3,}", clause)
        for element_id, text in candidates:
            if any(word.casefold() in text.casefold() for word in head):
                return element_id, text
        return candidates[0]

    def _plan_for(
        self,
        instruction: str,
        image_path: str | None,
        elements: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        # Only keys TaskPlan declares: the schema forbids extras on purpose, so
        # a backend that invents fields is rejected rather than quietly trusted.
        del image_path
        clauses = [part.strip() for part in _SPLIT.split(instruction) if part.strip()]
        if not clauses:
            clauses = [instruction] if instruction else ["do nothing"]

        steps: list[dict[str, Any]] = []
        for index, clause in enumerate(clauses):
            lowered = clause.casefold()
            action_type, verb = "click", "act"
            for words, candidate_action, candidate_verb in _INTENT_RULES:
                if any(word in lowered for word in words):
                    action_type, verb = candidate_action, candidate_verb
                    break
            # Aim at something that is actually on screen. When the runtime sent
            # no element list there is nothing honest to aim at, so fall back to
            # the clause's own words and let the adapter refuse them.
            picked = self._pick_target(clause, elements)
            arguments: dict[str, Any] = {}
            if picked is not None:
                target_text = picked[1]
                arguments["element_id"] = picked[0]
            else:
                target_text = self._target_for(clause)
            if action_type == "type_text":
                arguments["text"] = clause

            steps.append(
                {
                    "step_id": f"step-{index + 1}",
                    "description": clause,
                    "action_type": action_type,
                    "target_text": target_text,
                    "arguments": arguments,
                    "expected_result": f"{verb} step completed",
                    "status": "pending",
                }
            )

        steps.append(
            {
                "step_id": f"step-{len(steps) + 1}",
                "description": "report the result and stop",
                "action_type": "finish",
                "target_text": None,
                "arguments": {},
                "expected_result": "task finished",
                "status": "pending",
            }
        )

        return {
            "task_id": "mock-task",
            "instruction": instruction,
            "summary": f"Mock plan with {len(steps)} steps",
            "steps": steps,
            "assumptions": [
                "produced by the offline mock backend",
                "no screen was inspected",
            ],
            "requires_confirmation": True,
            "errors": [],
        }

    @staticmethod
    def _target_for(clause: str) -> str | None:
        quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{1,40})[\"'“”‘’]", clause)
        if quoted:
            return quoted[0]
        words = [w for w in re.findall(r"[A-Za-z]{3,}", clause)]
        return words[-1] if words else None
