"""Backend for any OpenAI-compatible chat endpoint.

The same class covers a hosted API and a local server (vLLM, LM Studio, Ollama's
OpenAI shim): only ``base_url`` changes. Credentials are read from the environment
and never logged - :meth:`ModelResponse.as_dict` omits the raw payload for exactly
that reason.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .base import ModelClient, ModelConfigError, ModelError, ModelResponse

#: Environment variables the client reads, in preference order per setting.
API_KEY_ENV = "GUI_AGENT_API_KEY"
BASE_URL_ENV = "GUI_AGENT_BASE_URL"
MODEL_ENV = "GUI_AGENT_MODEL"
DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatibleClient(ModelClient):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        temperature: float = 0.0,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        self.api_key = api_key or env.get(API_KEY_ENV) or None
        self.base_url = base_url or env.get(BASE_URL_ENV) or None
        super().__init__(
            model_name=model_name or env.get(MODEL_ENV) or DEFAULT_MODEL,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            temperature=temperature,
        )
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ModelConfigError(
                f"{API_KEY_ENV} is not set. Export it, or use --provider mock to run offline."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError(
                f"the openai package is required for this backend ({exc}). "
                "Install it with: pip install -r requirements-agent.txt"
            ) from exc
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )
        return self._client

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ModelResponse:
        client = self._ensure_client()
        payload = [dict(message) for message in messages]
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                temperature=self.temperature,
            )
        except Exception as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc

        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise ModelError("the model returned no choices")
        content = (getattr(choices[0].message, "content", "") or "").strip()
        if not content:
            raise ModelError("the model returned an empty response")

        usage_obj = getattr(completion, "usage", None)
        usage: dict[str, Any] = {}
        if usage_obj is not None:
            usage = {
                key: getattr(usage_obj, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if getattr(usage_obj, key, None) is not None
            }

        return ModelResponse(
            content=content,
            model_name=getattr(completion, "model", self.model_name),
            provider=self.name,
            usage=usage,
            raw_response={"finish_reason": getattr(choices[0], "finish_reason", None)},
        )
