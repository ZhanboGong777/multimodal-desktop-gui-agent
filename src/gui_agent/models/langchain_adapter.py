"""LangChain-backed model client.

The project's own :class:`ModelClient` stays the contract. LangChain is one more
backend behind it, not a dependency of the planner: nothing outside this module
imports it, so a LangChain upgrade can never reach the rest of the codebase.

It is registered under the provider name ``langchain`` and is a drop-in
replacement for ``openai_compatible`` - same credentials, same environment
variables, same vision payload.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from .base import ModelClient, ModelConfigError, ModelError, ModelResponse
from .openai_compatible import (
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_MODEL,
    MODEL_ENV,
    OpenAICompatibleClient,
)


class LangChainClient(ModelClient):
    """Sends the same messages through LangChain's chat model wrappers."""

    name = "langchain"

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
        self.base_url = base_url or env.get(BASE_URL_ENV) or "https://api.openai.com/v1"
        super().__init__(
            model_name=model_name or env.get(MODEL_ENV) or DEFAULT_MODEL,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            temperature=temperature,
        )
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.api_key:
            raise ModelConfigError(
                f"{API_KEY_ENV} is not set. Export it, or use --provider mock to run offline."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError(
                f"langchain-openai is required for this backend ({exc}). "
                "Install it with: pip install -r requirements-agent.txt"
            ) from exc
        self._model = ChatOpenAI(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            timeout=self.timeout_seconds,
            max_retries=0,  # ModelClient owns the retry policy.
        )
        return self._model

    @staticmethod
    def to_langchain_messages(
        messages: Sequence[Mapping[str, Any]], image_path: str | None
    ) -> list[Any]:
        """Convert OpenAI-shaped messages into LangChain message objects.

        The vision block is built by the OpenAI client first so that both backends
        encode images identically - the format is the same, only the wrapper
        differs.
        """
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ModelError(f"langchain-core is required for this backend ({exc})") from exc

        prepared = OpenAICompatibleClient.to_vision_messages(messages, image_path)
        converted: list[Any] = []
        for message in prepared:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ModelResponse:
        model = self._ensure_model()
        payload = self.to_langchain_messages(messages, kwargs.get("image_path"))
        try:
            result = model.invoke(payload)
        except Exception as exc:
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc

        content = getattr(result, "content", "")
        if isinstance(content, list):
            # Some providers return content blocks rather than a plain string.
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        content = (content or "").strip()
        if not content:
            raise ModelError("the model returned an empty response")

        usage: dict[str, Any] = {}
        metadata = getattr(result, "usage_metadata", None)
        if isinstance(metadata, dict):
            usage = {k: v for k, v in metadata.items() if isinstance(v, int)}

        return ModelResponse(
            content=content,
            model_name=self.model_name,
            provider=self.name,
            usage=usage,
            raw_response={"backend": "langchain"},
        )
