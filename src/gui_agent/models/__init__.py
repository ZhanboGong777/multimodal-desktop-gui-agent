"""Model clients: one interface, several backends."""

from __future__ import annotations

from ..config import ModelConfig
from .base import ModelClient, ModelConfigError, ModelError, ModelRequest, ModelResponse
from .mock import MockModelClient
from .openai_compatible import OpenAICompatibleClient

#: Backends by the name used in ``configs/default.yaml`` and ``--provider``.
CLIENTS: dict[str, type[ModelClient]] = {
    MockModelClient.name: MockModelClient,
    OpenAICompatibleClient.name: OpenAICompatibleClient,
}


def create_model_client(config: ModelConfig) -> ModelClient:
    """Build the backend named in the configuration."""
    try:
        factory = CLIENTS[config.provider]
    except KeyError:
        known = ", ".join(sorted(CLIENTS))
        raise ModelConfigError(f"unknown provider {config.provider!r}; known: {known}") from None

    common = {
        "model_name": config.model_name,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "temperature": config.temperature,
    }
    if factory is MockModelClient:
        return MockModelClient(**common)
    return OpenAICompatibleClient(**common, base_url=config.base_url)


__all__ = [
    "CLIENTS",
    "MockModelClient",
    "ModelClient",
    "ModelConfigError",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleClient",
    "create_model_client",
]
