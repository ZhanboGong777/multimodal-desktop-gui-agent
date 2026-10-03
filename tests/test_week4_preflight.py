"""The Week 4 preflight check.

The point of this script is that four silent failures on the review machine became
loud ones. So the tests here do not check that it prints something - they check that it
*refuses* in the situations that used to look like a normal run:

* a planning call that cannot finish inside ``model.timeout_seconds`` (measured 124 s
  against a 120 s limit, reported as an unhelpful "APITimeoutError");
* a frame whose text is an application window rather than a desktop, which is what
  filled the element list with `'not set.'` and `'(825.4 ms'` and made the model invent
  targets;
* a server context window below what the prompt needs;
* no model loaded, so a load cost would be recorded as planning time.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "week4_preflight.py"


def _run(*extra: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env=env,
    )


def test_the_script_exists_and_is_runnable() -> None:
    assert SCRIPT.is_file(), "scripts/week4_preflight.py is the check the manual points at"
    result = _run("--help")
    assert result.returncode == 0
    for flag in ("--config", "--model", "--base-url", "--required-context", "--skip-model"):
        assert flag in result.stdout


def test_a_context_window_below_the_prompt_size_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4096 against a ~7 500 token prompt is what stopped an earlier run with HTTP 400."""
    payload = {
        "models": [
            {
                "name": "qwen2.5vl:7b",
                "context_length": 4096,
                "size_vram": 5_500_452_207,
            }
        ]
    }
    server = _StubServer(json.dumps(payload).encode())
    try:
        result = _run(
            "--skip-model",
            "--base-url",
            server.url,
            "--required-context",
            "8192",
        )
    finally:
        server.close()

    assert "context_length=4096" in result.stdout
    assert "below the 8192 tokens" in result.stdout
    assert "OLLAMA_CONTEXT_LENGTH" in result.stdout, "the message must carry the fix"
    assert result.returncode == 1


def test_a_context_window_that_is_big_enough_passes_that_check() -> None:
    payload = {"models": [{"name": "m", "context_length": 16384, "size_vram": 1}]}
    server = _StubServer(json.dumps(payload).encode())
    try:
        result = _run("--skip-model", "--base-url", server.url, "--required-context", "8192")
    finally:
        server.close()
    assert "covers the 8192 needed" in result.stdout


def test_no_model_loaded_is_reported_as_a_warning_not_a_pass() -> None:
    """Otherwise the first task run records a model load as its own planning time."""
    server = _StubServer(json.dumps({"models": []}).encode())
    try:
        result = _run("--skip-model", "--base-url", server.url)
    finally:
        server.close()
    assert "no model is loaded" in result.stdout
    assert "warm it up first" in result.stdout


def test_the_terminal_marker_list_covers_the_wording_that_was_actually_seen() -> None:
    """The markers are evidence-driven: these are strings from the real bad frames."""
    text = SCRIPT.read_text(encoding="utf-8")
    for marker in ("not set.", "(825.4 ms", "Cache hit 99%", "blocked", "pwsh"):
        assert marker in text, f"{marker!r} was in the polluted frame and should be a marker"


def test_the_screen_check_explains_the_element_cap_rather_than_just_failing() -> None:
    """A bare 'screen looks wrong' would send the operator to the wrong place."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "max_elements" in text, "the cap is the mechanism; say so"
    assert "ranked above contour boxes" in text


class _StubServer:
    """Answers one request with a fixed JSON body."""

    def __init__(self, body: bytes) -> None:
        import socket
        import threading

        self._body = body
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        port = self._sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                try:
                    conn.recv(65536)
                    head = (
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                        + str(len(self._body)).encode()
                        + b"\r\nConnection: close\r\n\r\n"
                    )
                    conn.sendall(head + self._body)
                    conn.shutdown(1)
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()
