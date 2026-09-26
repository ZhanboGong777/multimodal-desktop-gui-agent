"""Tests for provider selection and credential handling."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gui_agent.config import ModelConfig
from gui_agent.models import MockModelClient, OpenAICompatibleClient, create_model_client
from gui_agent.models.base import ModelConfigError


def test_mock_provider_is_the_default() -> None:
    assert ModelConfig().provider == "mock"


def test_create_model_client_for_mock() -> None:
    client = create_model_client(ModelConfig(provider="mock"))
    assert isinstance(client, MockModelClient)
    assert client.model_name == "mock-vision-model"


def test_create_model_client_for_openai_compatible() -> None:
    client = create_model_client(
        ModelConfig(
            provider="openai_compatible", model_name="qwen-vl", base_url="http://localhost:8000/v1"
        )
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == "http://localhost:8000/v1"
    assert client.model_name == "qwen-vl"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="anthropic")  # type: ignore[arg-type]


def test_missing_api_key_gives_an_actionable_message() -> None:
    client = OpenAICompatibleClient(environ={})
    # health_check reports rather than raises, which is what makes it usable as a
    # readiness probe; the guarded path below is where the error surfaces.
    assert client.health_check() is False
    assert client.api_key is None
    with pytest.raises(ModelConfigError) as raised:
        client._ensure_client()
    assert "GUI_AGENT_API_KEY" in str(raised.value)
    assert "mock" in str(raised.value)


def test_credentials_are_read_from_the_environment() -> None:
    client = OpenAICompatibleClient(
        environ={
            "GUI_AGENT_API_KEY": "sk-test",
            "GUI_AGENT_BASE_URL": "http://localhost:11434/v1",
            "GUI_AGENT_MODEL": "qwen2.5-vl",
        }
    )
    assert client.api_key == "sk-test"
    assert client.base_url == "http://localhost:11434/v1"
    assert client.model_name == "qwen2.5-vl"


def test_a_failing_backend_returns_an_error_response_not_an_exception() -> None:
    # Point at a closed local port so the failure is immediate: a real base URL
    # would make this test wait for a network timeout.
    client = OpenAICompatibleClient(
        environ={
            "GUI_AGENT_API_KEY": "sk-test",
            "GUI_AGENT_BASE_URL": "http://127.0.0.1:9/v1",
        },
        max_retries=0,
        timeout_seconds=0.5,
    )
    response = client.generate_text("hello")
    assert response.ok is False
    # No network in tests: the SDK import or the connection fails, either way the
    # caller receives a ModelResponse rather than a traceback.
    assert response.error
