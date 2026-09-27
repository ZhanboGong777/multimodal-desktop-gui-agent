"""Backend for any OpenAI-compatible chat endpoint.

The same class covers a hosted API and a local server (vLLM, LM Studio, Ollama's
OpenAI shim): only ``base_url`` changes. Credentials are read from the environment
and never logged - :meth:`ModelResponse.as_dict` omits the raw payload for exactly
that reason.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .base import ModelClient, ModelConfigError, ModelError, ModelResponse

#: Environment variables the client reads, in preference order per setting.
API_KEY_ENV = "GUI_AGENT_API_KEY"
BASE_URL_ENV = "GUI_AGENT_BASE_URL"
MODEL_ENV = "GUI_AGENT_MODEL"
DEFAULT_MODEL = "gpt-4o-mini"

#: Suffix -> MIME type. Anything else is refused rather than guessed.
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

#: Base64 inflates by a third, and a full-screen PNG is already a megabyte or two.
#: 20 MB of source bytes stays well inside what the common endpoints accept.
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def encode_image_data_url(path: str | Path) -> str:
    """Read an image and return it as a ``data:`` URL for the vision API.

    Raises :class:`ModelConfigError` when the file is missing and
    :class:`ModelError` when it is too large or of an unknown type - a screenshot
    silently dropped from the request would make the model answer about nothing.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise ModelConfigError(f"image not found: {image_path}")

    mime = IMAGE_MIME_TYPES.get(image_path.suffix.casefold())
    if mime is None:
        known = ", ".join(sorted(IMAGE_MIME_TYPES))
        raise ModelError(f"unsupported image type {image_path.suffix!r}; known: {known}")

    payload = image_path.read_bytes()
    if not payload:
        raise ModelError(f"image is empty: {image_path}")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ModelError(
            f"image is {len(payload) / 1e6:.1f} MB, above the {MAX_IMAGE_BYTES / 1e6:.0f} MB limit: "
            f"{image_path}"
        )
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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

    @staticmethod
    def to_vision_messages(
        messages: Sequence[Mapping[str, Any]], image_path: str | Path | None
    ) -> list[dict[str, Any]]:
        """Attach an image to the last user turn in the OpenAI vision format.

        Without this the image travelled as a bare path inside the text payload and
        the model could not see it at all.
        """
        payload = [dict(message) for message in messages]
        if not image_path:
            return payload

        data_url = encode_image_data_url(image_path)
        for message in reversed(payload):
            if message.get("role") != "user":
                continue
            text = message.get("content")
            message["content"] = [
                {"type": "text", "text": text if isinstance(text, str) else str(text)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            break
        return payload

    def complete(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ModelResponse:
        client = self._ensure_client()
        payload = self.to_vision_messages(messages, kwargs.get("image_path"))
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
