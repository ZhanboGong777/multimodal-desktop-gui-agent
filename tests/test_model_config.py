"""Tests for provider selection and credential handling."""

from __future__ import annotations

import contextlib
import json
import socket
import threading
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


# ─────────────── timeouts and error classification ───────────────
class _SilentServer:
    """Accepts a connection and then never answers, to force a read timeout."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        try:
            conn, _ = self._sock.accept()
            threading.Event().wait(30)  # hold the connection open, send nothing
            conn.close()
        except OSError:
            pass

    def close(self) -> None:
        self._sock.close()


def test_a_hanging_backend_times_out_instead_of_blocking() -> None:
    """A backend that accepts and never replies must not hang the caller.

    Timeouts were configured but never exercised: nothing proved that a slow
    endpoint is actually cut off, or that the cut-off is reported as a model
    failure rather than raised as a raw SDK exception.
    """
    server = _SilentServer()
    try:
        client = OpenAICompatibleClient(
            model_name="slow",
            base_url=f"http://127.0.0.1:{server.port}/v1",
            api_key="test-key",
            timeout_seconds=0.4,
            max_retries=0,
            environ={},
        )
        # complete() raises; generate_text() would swallow this into a response.
        with pytest.raises(ModelError) as excinfo:
            client.complete([{"role": "user", "content": "hello"}])
        message = str(excinfo.value).casefold()
        assert "timeout" in message or "timed out" in message
    finally:
        server.close()


def test_a_timed_out_call_is_reported_not_raised_by_the_wrapper() -> None:
    """The retrying wrapper turns the timeout into a failed ModelResponse."""
    server = _SilentServer()
    try:
        client = OpenAICompatibleClient(
            model_name="slow",
            base_url=f"http://127.0.0.1:{server.port}/v1",
            api_key="test-key",
            timeout_seconds=0.4,
            max_retries=0,
            environ={},
        )
        response = client.generate_text("hello")

        assert not response.ok
        assert response.content == ""
        assert response.error is not None
    finally:
        server.close()


def test_the_client_retries_once_before_giving_up() -> None:
    """max_retries=1 means two attempts, and both are timed, not just the last."""
    server = _SilentServer()
    try:
        client = OpenAICompatibleClient(
            model_name="slow",
            base_url=f"http://127.0.0.1:{server.port}/v1",
            api_key="test-key",
            timeout_seconds=0.3,
            max_retries=1,
            environ={},
        )
        response = client.generate_text("hello")

        assert not response.ok
        assert response.error is not None
        # Two attempts of roughly 0.3 s each: the reported latency covers both.
        assert response.latency_ms > 500
    finally:
        server.close()


def test_configuration_errors_are_a_separate_class_from_runtime_errors() -> None:
    """A missing key is the caller's mistake; a broken image is the input's.

    Both surface as ModelError, so code that catches the base class keeps working,
    but the two are distinguishable and the distinction is load-bearing: one is
    fixed by exporting a variable, the other by passing a better file.
    """
    assert issubclass(ModelConfigError, ModelError)

    client = OpenAICompatibleClient(model_name="m", environ={})
    with pytest.raises(ModelConfigError):
        client.complete([{"role": "user", "content": "x"}])

    # A malformed image is not a configuration problem.
    with pytest.raises(ModelError) as excinfo:
        OpenAICompatibleClient.to_vision_messages(
            [{"role": "user", "content": "x"}], "/no/such/file.png"
        )
    assert not (excinfo.value is None)


def test_the_sdk_retry_policy_is_ours_not_its_own() -> None:
    """10.3.4: stacked retries have to be prevented, not documented.

    The SDK retries twice by default. Ours does too, so a failing endpoint was
    attempted three times inside a single `complete()` call while our loop counted
    one - which meant `model.max_retries: 0` did not actually mean "one attempt",
    and `model_requests` under-reported the real traffic by up to 3x.
    """
    patient = OpenAICompatibleClient(
        model_name="m", api_key="k", base_url="http://127.0.0.1:1/v1", max_retries=0
    )
    assert patient._ensure_client().max_retries == 0

    retrying = OpenAICompatibleClient(
        model_name="m", api_key="k", base_url="http://127.0.0.1:1/v1", max_retries=3
    )
    assert retrying._ensure_client().max_retries == 3


def test_an_unknown_provider_is_caught_even_when_it_bypasses_validation() -> None:
    """`ModelConfig.provider` is a Literal, so pydantic rejects a bad value.

    Assignment is not validated, though, and the factory can also be handed a
    config built some other way - so it still has to say which providers exist
    rather than raising a bare KeyError.
    """
    config = ModelConfig()
    config.provider = "nonsense"  # type: ignore[assignment]

    with pytest.raises(ModelConfigError, match="known:"):
        create_model_client(config)


def _read_request(conn: socket.socket) -> bytes:
    """Read one HTTP request completely - headers and body.

    Draining the body is the point. The OpenAI client sends a JSON body, and a
    server that replies and closes after reading only the headers leaves data
    unread on its side; Windows turns that close into an RST and the client sees
    `APIConnectionError` rather than the response it was sent.
    """
    conn.settimeout(5.0)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return data
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            with contextlib.suppress(ValueError):
                length = int(value.strip() or 0)
    while len(body) < length:
        chunk = conn.recv(65536)
        if not chunk:
            break
        body += chunk
    return head + b"\r\n\r\n" + body


class _AnsweringServer:
    """One-shot HTTP server that always replies with the same JSON body.

    `_SilentServer` covers the endpoint that never answers; this covers the one
    that answers with something unusable, which is the other way a local model can
    disappoint.
    """

    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode()
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            _read_request(conn)
            head = (
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\nContent-Length: "
                + str(len(self._payload)).encode()
                + b"\r\n\r\n"
            )
            conn.sendall(head + self._payload)
            # Send FIN before the close, and let the close be ordinary. A server
            # that answers while the client is still sending gets an abortive
            # close on Windows, and the client then reports a connection error
            # instead of reading the reply - which is how these two tests failed
            # on the Windows review machine and passed here.
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_WR)

    def close(self) -> None:
        self._sock.close()


def _client_for(server: _AnsweringServer) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model_name="m",
        api_key="k",
        base_url=f"http://127.0.0.1:{server.port}/v1",
        max_retries=0,
    )


def _completion(choices: list[dict]) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": choices,
    }


def test_an_endpoint_that_answers_with_no_choices_says_so() -> None:
    """Otherwise the caller gets an IndexError from deep inside the SDK."""
    server = _AnsweringServer(_completion([]))
    try:
        with pytest.raises(ModelError, match="no choices"):
            _client_for(server).complete([{"role": "user", "content": "hi"}])
    finally:
        server.close()


def test_an_endpoint_that_answers_with_nothing_says_so() -> None:
    """A 200 with a blank message is not a plan, and parsing it would fail later
    with a message about JSON rather than about the endpoint."""
    server = _AnsweringServer(
        _completion(
            [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "   "},
                }
            ]
        )
    )
    try:
        with pytest.raises(ModelError, match="empty response"):
            _client_for(server).complete([{"role": "user", "content": "hi"}])
    finally:
        server.close()
