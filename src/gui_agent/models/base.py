"""Provider-independent multimodal model interface.

Business code talks to :class:`ModelClient` and never to a vendor SDK. Every call
returns a :class:`ModelResponse` carrying the text, the provider, the latency and
whatever usage information the backend reported, so the planner, the logs and the
experiment report all read the same shape.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar


class ModelError(RuntimeError):
    """Raised by a backend when a call cannot be completed."""


class ModelConfigError(ModelError):
    """Raised when a backend is missing configuration it cannot work without."""


@dataclass
class ModelResponse:
    """One completion, normalised across providers."""

    content: str
    model_name: str
    provider: str
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content.strip())

    def as_dict(self) -> dict[str, Any]:
        """Log-safe view: the raw provider payload is deliberately omitted."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "latency_ms": round(self.latency_ms, 3),
            "usage": self.usage,
            "error": self.error,
            "content_length": len(self.content),
        }


@dataclass
class ModelRequest:
    """What a caller wants the model to do."""

    instruction: str
    context: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None

    def to_messages(self) -> list[dict[str, Any]]:
        """Flatten to chat messages; the image travels as a path in the context."""
        payload: dict[str, Any] = {"instruction": self.instruction}
        if self.context:
            payload["context"] = self.context
        if self.image_path:
            payload["image_path"] = self.image_path
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


class ModelClient(ABC):
    """Base class for every backend."""

    #: Value written to :attr:`ModelResponse.provider`.
    name: ClassVar[str] = "base"

    def __init__(
        self,
        *,
        model_name: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.temperature = temperature

    @abstractmethod
    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ModelResponse:
        """Send already-built messages. Subclasses implement this one method."""

    def generate_text(self, prompt: str, *, system: str | None = None) -> ModelResponse:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._with_retries(messages, {})

    def generate_multimodal(
        self,
        instruction: str,
        *,
        image_path: str | None = None,
        context: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> ModelResponse:
        """Send an instruction with screen context, and optionally a system prompt.

        The instruction stays separate from the rules on purpose: a backend that
        echoes the instruction back (the mock does) must see the task, not the
        whole prompt template.
        """
        request = ModelRequest(
            instruction=instruction, context=context or {}, image_path=image_path
        )
        messages = request.to_messages()
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return self._with_retries(messages, {"image_path": image_path})

    def health_check(self) -> bool:
        """True when the backend answers at all. Never raises."""
        try:
            return self.generate_text("ping").ok
        except Exception:  # noqa: BLE001 - a health check reports, never propagates
            return False

    def _with_retries(
        self, messages: Sequence[Mapping[str, Any]], kwargs: dict[str, Any]
    ) -> ModelResponse:
        started = time.perf_counter()
        last_error = "no attempt was made"
        for _ in range(self.max_retries + 1):
            try:
                response = self.complete(messages, **kwargs)
                response.latency_ms = (time.perf_counter() - started) * 1000.0
                return response
            except ModelError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        return ModelResponse(
            content="",
            model_name=self.model_name,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=last_error,
        )
