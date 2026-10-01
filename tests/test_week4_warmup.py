"""The warmup probe, against a real server over a real socket.

13.4 asks for a warmup before the week's tasks and for five things to be recorded:
cold/warm state, available memory, the request time, a failure classification and
the retry count. None of it existed, so the first task run on a cold server would
have recorded a model load as its own planning time - the failure mode the section
exists to prevent.

The point of these tests is not that the script prints something. It is that the
probe goes through the project's client (13.4.6), that the second probe really
carries image data rather than a path, and that a backend which is not ready makes
the command say so with a non-zero exit instead of leaving the operator to decide.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "week4_warmup.py"


class _Handler(BaseHTTPRequestHandler):
    """A stand-in vision endpoint that records what it was asked."""

    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(body)

        payload = {
            "id": "chatcmpl-warmup",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ready, and the background is grey"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr logging."""


class _RefusingHandler(_Handler):
    """Answers the way a server with no context left does."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(
            {
                "error": {
                    "message": "the request exceeds the available context size",
                    "type": "invalid_request_error",
                }
            }
        ).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _env(**extra: str) -> dict[str, str]:
    """The ambient environment, with anything GUI_AGENT_* taken out of it.

    Copied rather than built from nothing. A stripped environment is not a portable
    one: Windows needs SystemRoot for socket setup, and the interpreter looks for
    TEMP under names this test has no business guessing. What the test actually
    needs is that a stray GUI_AGENT_BASE_URL or a real key cannot reach the child.

    `GUI_AGENT_DISABLE_DOTENV` used to be set here. Nothing reads it - it appears
    nowhere in src/ or scripts/ - so it did nothing at all while looking like it
    made the child hermetic. What does that is the working directory: both entry
    points load `./.env` relative to the process's cwd, so running the child in a
    temporary directory is what actually keeps a real `.env` out.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("GUI_AGENT_")}
    env.update(extra)
    return env


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _run_warmup(server_url: str, tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provider",
            "openai_compatible",
            "--model",
            "stub-vision",
            "--base-url",
            server_url,
            "--image",
            str(tmp_path / "shot.png"),
            "--json",
            str(tmp_path / "warmup.json"),
            *extra,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=_env(GUI_AGENT_API_KEY="test-key"),
    )


@pytest.fixture
def shot(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "shot.png"
    Image.new("RGB", (12, 12), (10, 20, 30)).save(path)
    return path


def test_the_warmup_probes_text_then_image_through_the_client(
    server: str, tmp_path: Path, shot: Path
) -> None:
    """Both probes reach the server, and the second one carries the pixels."""
    result = _run_warmup(server, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_Handler.requests) == 2, "13.4.2 wants a text probe and an image probe"

    text_blocks = _Handler.requests[0]["messages"][-1]["content"]
    assert isinstance(text_blocks, str), "the first probe is the text call"

    blocks = _Handler.requests[1]["messages"][-1]["content"]
    assert isinstance(blocks, list), "the image probe must send vision content"
    images = [block for block in blocks if block.get("type") == "image_url"]
    assert len(images) == 1
    url = images[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == shot.read_bytes()

    assert "text" in result.stdout and "image" in result.stdout


def test_the_record_carries_everything_13_4_asks_to_record(
    server: str, tmp_path: Path, shot: Path
) -> None:
    """Cold/warm state, memory, time, classification and retry count."""
    _run_warmup(server, tmp_path)
    record = json.loads((tmp_path / "warmup.json").read_text(encoding="utf-8"))

    assert record["ready"] is True
    assert record["provider"] == "openai_compatible"
    assert record["model_name"] == "stub-vision"
    assert record["base_url"].startswith("http://127.0.0.1:")
    assert record["timeout_seconds"] == 120.0
    assert record["max_retries_configured"] == 0
    assert "memory_available_mb_before" in record
    assert record["screenshot"].endswith("shot.png")
    assert record["screenshot_captured_here"] is False

    kinds = [entry["kind"] for entry in record["probes"]]
    states = [entry["state"] for entry in record["probes"]]
    assert kinds == ["text", "image"]
    assert states[0] == "cold-or-idle", "the first request of the session is the cold one"
    assert states[1] == "vision-path"
    for entry in record["probes"]:
        assert entry["ok"] is True
        assert entry["latency_ms"] >= 0.0
        assert entry["usage"], "the token counts come from the response, not from a guess"


def test_a_repeat_shows_the_warm_steady_state(server: str, tmp_path: Path, shot: Path) -> None:
    """`--repeat` is how the operator sees the cold number next to the warm one."""
    _run_warmup(server, tmp_path, "--repeat", "2")
    record = json.loads((tmp_path / "warmup.json").read_text(encoding="utf-8"))

    assert [entry["state"] for entry in record["probes"]] == [
        "cold-or-idle",
        "vision-path",
        "warm-repeat",
        "warm",
    ]
    assert len(_Handler.requests) == 4


def test_a_context_overflow_is_classified_not_just_reported(tmp_path: Path) -> None:
    """The failure reads like a run failure, because it is the classifier runs use.

    13.4.3 asks for a failure classification. Reusing the runtime's means a warmup
    that fails on the context window gives the operator the same hint a task run
    would - including the one the review needed and did not get.
    """
    _RefusingHandler.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _RefusingHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/v1"
        result = _run_warmup(url, tmp_path)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert result.returncode == 1, result.stdout
    record = json.loads((tmp_path / "warmup.json").read_text(encoding="utf-8"))
    assert record["ready"] is False

    failed = [entry for entry in record["probes"] if not entry["ok"]]
    assert failed, "a refusing backend must not be recorded as ready"
    assert "OLLAMA_CONTEXT_LENGTH" in failed[0]["failure_class"]
    assert "not ready" in result.stdout


def test_an_unreachable_backend_is_reported_rather_than_raised(tmp_path: Path) -> None:
    """Nothing to talk to is exit 1 with the reason, never a traceback.

    Port 9 is the discard port: open on the loopback of a machine that has
    inetd, closed on one that does not, and refused either way for our purposes.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provider",
            "openai_compatible",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--image",
            str(tmp_path / "missing.png"),
            "--json",
            str(tmp_path / "warmup.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=_env(GUI_AGENT_API_KEY="test-key"),
    )

    assert result.returncode == 1, result.stdout
    assert "Traceback" not in result.stderr
    record = json.loads((tmp_path / "warmup.json").read_text(encoding="utf-8"))
    assert record["ready"] is False
    assert all(not entry["ok"] for entry in record["probes"])


def test_the_warmup_reaches_the_same_backend_the_tasks_will(
    server: str, tmp_path: Path, shot: Path
) -> None:
    """13.1.1's precedence, through the warmup's own flag parsing.

    A warmup that resolved its settings differently from the run would warm a
    different service and report success for it. Both call the same function now;
    this pins the order for this entry point too - the flag has to beat a
    stale ``GUI_AGENT_BASE_URL`` left in the environment.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provider",
            "openai_compatible",
            "--base-url",
            server,
            "--image",
            str(shot),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env=_env(
            GUI_AGENT_API_KEY="test-key",
            # Deliberately stale: the flag has to beat it (13.1.1).
            GUI_AGENT_BASE_URL="http://127.0.0.1:9/v1",
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"base_url   : {server}" in result.stdout
    assert len(_Handler.requests) == 2
