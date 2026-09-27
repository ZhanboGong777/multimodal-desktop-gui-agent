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


class MockModelClient(ModelClient):
    """Deterministic, network-free, free-of-charge."""

    name = "mock"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", "mock-vision-model")
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ModelResponse:
        instruction = self._instruction_from(messages)
        self.calls.append({"messages": [dict(m) for m in messages], "instruction": instruction})
        plan = self._plan_for(instruction, kwargs.get("image_path"))
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

    def _plan_for(self, instruction: str, image_path: str | None) -> dict[str, Any]:
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
            steps.append(
                {
                    "step_id": f"step-{index + 1}",
                    "description": clause,
                    "action_type": action_type,
                    "target_text": self._target_for(clause),
                    "arguments": {} if action_type != "type_text" else {"text": clause},
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
