"""Tests for provider selection and credential handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from gui_agent.config import ModelConfig
from gui_agent.models import MockModelClient, OpenAICompatibleClient, create_model_client
from gui_agent.models.base import ModelConfigError, ModelError


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


# ───────────────────── multimodal payloads ─────────────────────
def _png(path: Path, size: tuple[int, int] = (16, 12)) -> Path:
    Image.new("RGB", size, "white").save(path)
    return path


def test_an_image_is_attached_as_a_vision_block(tmp_path: Path) -> None:
    """Regression: the image used to travel as a bare path inside the text."""
    image = _png(tmp_path / "shot.png")
    messages = [{"role": "user", "content": '{"instruction": "close it"}'}]

    payload = OpenAICompatibleClient.to_vision_messages(messages, image)

    content = payload[0]["content"]
    assert isinstance(content, list) and len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_the_original_messages_are_not_mutated(tmp_path: Path) -> None:
    image = _png(tmp_path / "shot.png")
    messages = [{"role": "user", "content": "hi"}]
    OpenAICompatibleClient.to_vision_messages(messages, image)
    assert messages[0]["content"] == "hi"


def test_without_an_image_the_messages_pass_through() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert OpenAICompatibleClient.to_vision_messages(messages, None) == messages


def test_the_image_lands_on_the_last_user_turn(tmp_path: Path) -> None:
    image = _png(tmp_path / "shot.png")
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    payload = OpenAICompatibleClient.to_vision_messages(messages, image)
    assert payload[1]["content"] == "first"
    assert isinstance(payload[3]["content"], list)
    assert payload[0]["content"] == "rules"


def test_a_missing_image_is_reported() -> None:
    with pytest.raises(ModelConfigError):
        OpenAICompatibleClient.to_vision_messages(
            [{"role": "user", "content": "x"}], "/no/such.png"
        )


def test_an_unknown_image_type_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image", encoding="utf-8")
    with pytest.raises(ModelError) as excinfo:
        OpenAICompatibleClient.to_vision_messages([{"role": "user", "content": "x"}], bad)
    assert "unsupported image type" in str(excinfo.value)


def test_an_empty_image_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ModelError):
        OpenAICompatibleClient.to_vision_messages([{"role": "user", "content": "x"}], empty)


def test_an_oversized_image_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = _png(tmp_path / "shot.png")
    monkeypatch.setattr(
        ocr_modules := __import__("gui_agent.models.openai_compatible", fromlist=["x"]),
        "MAX_IMAGE_BYTES",
        4,
    )
    with pytest.raises(ModelError) as excinfo:
        OpenAICompatibleClient.to_vision_messages([{"role": "user", "content": "x"}], image)
    assert "above the" in str(excinfo.value)
    assert ocr_modules is not None


def test_each_supported_suffix_maps_to_a_mime_type(tmp_path: Path) -> None:
    from gui_agent.models.openai_compatible import IMAGE_MIME_TYPES

    for suffix, mime in IMAGE_MIME_TYPES.items():
        path = tmp_path / f"shot{suffix}"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 8)
        url = OpenAICompatibleClient.to_vision_messages([{"role": "user", "content": "x"}], path)[
            0
        ]["content"][1]["image_url"]["url"]
        assert url.startswith(f"data:{mime};base64,")
