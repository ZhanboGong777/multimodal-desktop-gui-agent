"""The LangChain backend must behave exactly like the other model clients."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from gui_agent.config import ModelConfig
from gui_agent.models import CLIENTS, LangChainClient, create_model_client
from gui_agent.models.base import ModelConfigError, ModelError
from gui_agent.models.openai_compatible import API_KEY_ENV

#: Smallest valid PNG (1x1 pixel), so the encoder has a real file to read.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

TEXT_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a GUI agent."},
    {"role": "user", "content": "What is on screen?"},
]


class StubResult:
    """Stands in for a LangChain AIMessage."""

    def __init__(self, content: Any, usage: dict[str, int] | None = None) -> None:
        self.content = content
        self.usage_metadata = usage


class StubModel:
    """Records what it was invoked with and replays a canned answer."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result if result is not None else StubResult('{"action": "click"}')
        self.error = error
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.result


def make_client(**overrides: Any) -> LangChainClient:
    kwargs: dict[str, Any] = {
        "model_name": "qwen2.5vl:7b",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key": "test-key",
        "timeout_seconds": 0.5,
        "environ": {},
    }
    kwargs.update(overrides)
    return LangChainClient(**kwargs)


def test_langchain_is_a_registered_backend() -> None:
    assert LangChainClient.name == "langchain"
    assert CLIENTS["langchain"] is LangChainClient


def test_factory_builds_the_langchain_backend() -> None:
    config = ModelConfig(provider="langchain", model_name="qwen2.5vl:7b")
    client = create_model_client(config)
    assert isinstance(client, LangChainClient)
    assert client.model_name == "qwen2.5vl:7b"


def test_credentials_fall_back_to_the_environment() -> None:
    client = make_client(api_key=None, environ={API_KEY_ENV: "from-env"})
    assert client.api_key == "from-env"


def test_missing_api_key_is_reported_before_any_network_call() -> None:
    client = make_client(api_key=None)
    with pytest.raises(ModelConfigError, match=API_KEY_ENV):
        client.complete(TEXT_MESSAGES)


def test_roles_survive_the_conversion() -> None:
    converted = LangChainClient.to_langchain_messages(TEXT_MESSAGES, None)
    assert [type(m) for m in converted] == [SystemMessage, HumanMessage]
    assert converted[0].content == "You are a GUI agent."


def test_assistant_turns_are_preserved() -> None:
    messages = [*TEXT_MESSAGES, {"role": "assistant", "content": "I see a browser."}]
    converted = LangChainClient.to_langchain_messages(messages, None)
    assert isinstance(converted[-1], AIMessage)


def test_image_becomes_a_vision_block_not_a_path(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(PNG_BYTES)

    converted = LangChainClient.to_langchain_messages(TEXT_MESSAGES, str(image))
    blocks = converted[-1].content

    assert isinstance(blocks, list)
    assert blocks[0] == {"type": "text", "text": "What is on screen?"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # The path itself must never reach the model.
    assert str(image) not in str(blocks)


def test_image_is_attached_only_to_the_last_user_turn(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(PNG_BYTES)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    converted = LangChainClient.to_langchain_messages(messages, str(image))
    assert isinstance(converted[0].content, str)
    assert isinstance(converted[-1].content, list)


def test_complete_returns_the_model_answer() -> None:
    client = make_client()
    client._model = StubModel()
    response = client.complete(TEXT_MESSAGES)

    assert response.ok
    assert response.content == '{"action": "click"}'
    assert response.provider == "langchain"
    assert response.model_name == "qwen2.5vl:7b"


def test_complete_forwards_the_converted_messages() -> None:
    client = make_client()
    stub = StubModel()
    client._model = stub

    client.complete(TEXT_MESSAGES)
    assert isinstance(stub.calls[0][0], SystemMessage)


def test_content_blocks_are_joined_into_text() -> None:
    """Some providers answer with content blocks instead of a plain string."""
    client = make_client()
    client._model = StubModel(
        StubResult([{"type": "text", "text": '{"action": '}, {"type": "text", "text": '"click"}'}])
    )
    assert client.complete(TEXT_MESSAGES).content == '{"action": "click"}'


def test_usage_metadata_is_carried_over() -> None:
    client = make_client()
    client._model = StubModel(StubResult("ok", {"input_tokens": 11, "output_tokens": 3}))
    assert client.complete(TEXT_MESSAGES).usage == {"input_tokens": 11, "output_tokens": 3}


def test_empty_answer_is_an_error_not_a_success() -> None:
    client = make_client()
    client._model = StubModel(StubResult("   "))
    with pytest.raises(ModelError, match="empty"):
        client.complete(TEXT_MESSAGES)


def test_backend_failures_become_model_errors() -> None:
    client = make_client()
    client._model = StubModel(error=RuntimeError("connection refused"))
    with pytest.raises(ModelError, match="connection refused"):
        client.complete(TEXT_MESSAGES)


def test_complete_can_be_called_repeatedly() -> None:
    client = make_client()
    stub = StubModel()
    client._model = stub
    client.complete(TEXT_MESSAGES)
    client.complete(TEXT_MESSAGES)
    assert len(stub.calls) == 2
