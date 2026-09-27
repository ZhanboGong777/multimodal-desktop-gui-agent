"""End-to-end test of the LangChain backend over real HTTP.

Unit tests inject a stub model, which cannot prove that LangChain itself builds a
correct request. This test stands up a real OpenAI-compatible server on localhost
and lets the real ``langchain-openai`` client talk to it, so the assertion is
about bytes that actually crossed a socket.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from gui_agent.models.langchain_adapter import LangChainClient
from gui_agent.planning import TaskPlanner

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

#: A single action, enough for the transport-level assertions.
REPLY: dict[str, Any] = {
    "action": "click",
    "target": "browser icon",
    "reasoning": "The task starts by opening the browser.",
}

#: A complete plan, because the planner validates against the strict TaskPlan schema.
PLAN_REPLY: dict[str, Any] = {
    "task_id": "task-1",
    "instruction": "Open the browser",
    "summary": "Open the browser from the desktop.",
    "steps": [
        {
            "step_id": "step-1",
            "description": "Click the browser icon on the desktop",
            "action_type": "click",
            "target_text": "browser icon",
            "expected_result": "The browser window opens.",
        }
    ],
    "requires_confirmation": True,
}


class _Handler(BaseHTTPRequestHandler):
    """Records the request body and replays a canned completion."""

    received: ClassVar[list[dict[str, Any]]] = []
    reply: ClassVar[dict[str, Any]] = REPLY

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.append(body)

        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "test"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(type(self).reply)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr logging."""


@pytest.fixture
def fake_server() -> Iterator[str]:
    _Handler.received = []
    _Handler.reply = REPLY
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_langchain_sends_a_real_vision_request(fake_server: str, tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(PNG_BYTES)

    client = LangChainClient(
        model_name="qwen2.5vl:7b",
        base_url=fake_server,
        api_key="test-key",
        timeout_seconds=10.0,
        environ={},
    )
    response = client.complete(
        [
            {"role": "system", "content": "You are a GUI agent."},
            {"role": "user", "content": "What should I click?"},
        ],
        image_path=str(image),
    )

    assert response.ok
    assert json.loads(response.content)["action"] == "click"
    assert response.provider == "langchain"

    sent = _Handler.received[-1]
    assert sent["model"] == "qwen2.5vl:7b"
    assert sent["messages"][0]["role"] == "system"

    blocks = sent["messages"][-1]["content"]
    assert isinstance(blocks, list), "the image never left as a vision block"
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert str(image) not in json.dumps(sent), "the raw path leaked into the payload"


def test_usage_is_reported_back(fake_server: str) -> None:
    client = LangChainClient(
        model_name="qwen2.5vl:7b",
        base_url=fake_server,
        api_key="test-key",
        timeout_seconds=10.0,
        environ={},
    )
    response = client.complete([{"role": "user", "content": "hello"}])
    assert response.usage.get("input_tokens") == 7
    assert response.usage.get("output_tokens") == 5


def test_the_planner_runs_unchanged_on_the_langchain_backend(fake_server: str) -> None:
    """The planner must not know or care which backend it is talking to."""
    _Handler.reply = PLAN_REPLY
    client = LangChainClient(
        model_name="qwen2.5vl:7b",
        base_url=fake_server,
        api_key="test-key",
        timeout_seconds=10.0,
        environ={},
    )
    planner = TaskPlanner(client)
    outcome = planner.plan(instruction="Open the browser", context={"observation": "Desktop"})

    assert outcome.ok, outcome.error
    assert outcome.plan is not None
    assert outcome.plan.steps[0].action_type == "click"
    assert outcome.plan.requires_confirmation is True
    assert outcome.metadata["provider"] == "langchain"
    assert _Handler.received[-1]["messages"][-1]["role"] == "user"
