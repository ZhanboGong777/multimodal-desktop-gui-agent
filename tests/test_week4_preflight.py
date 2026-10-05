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
import runpy
import subprocess
import sys
from collections.abc import Callable
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


@pytest.fixture(scope="module")
def check_context() -> Callable[[str, int], list[str]]:
    # In the Windows full suite, --skip-model still captured the live desktop:
    # OCR's copyright symbol then crashed GBK output before the context verdict
    # was printed. Exercise the production context check against our stub server
    # directly, so these network tests never depend on observation or the model.
    return runpy.run_path(str(SCRIPT))["check_context"]


def test_the_t04_verifier_refuses_a_locked_desktop_without_running_anything() -> None:
    """The one case that sends a real message must not start on a session it cannot drive.

    Measured on this node, repeatedly: when the screen locks mid-run the capture fails with
    `BitBlt` from Windows and pyautogui aborts with its corner fail-safe, so a run started in that
    state dispatches nothing and records a verdict about the wrong thing. The verifier therefore
    checks the session first and exits `2` - "not a state where a run is meaningful" - rather than
    letting the CLI plan a task it cannot carry out.

    This test runs on whichever session the suite happens to be in, so it asserts the *contract*
    rather than one outcome: exit code 0 with the checks printed, or exit code 2 with a refusal,
    and never a traceback. `--run` is deliberately not exercised - the case sends a real message.
    """
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "week4_t04_verify.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )

    assert completed.returncode in (0, 2), completed.stdout + completed.stderr
    assert "T04 preflight" in completed.stdout, completed.stdout

    if completed.returncode == 2:
        assert "REFUSING" in completed.stdout, completed.stdout


def test_the_t04_verifier_does_not_disable_the_pointer_fail_safe() -> None:
    """The fail-safe is the last guard before a mis-computed coordinate becomes a real click.

    It aborts when the pointer reaches a screen corner, and it fired during this branch's work -
    correctly, when a coordinate belonged to a window that had already moved. A script that turned
    it off to "make the run go through" would be removing the check that caught that.
    """
    source = (REPO_ROOT / "scripts" / "week4_t04_verify.py").read_text(encoding="utf-8")

    assert "FAILSAFE" not in source, "the verifier must not touch pyautogui's fail-safe"
    assert "pyautogui.position()" in source, "it should read the pointer, to notice a dead session"


def test_the_script_exists_and_is_runnable() -> None:

    assert SCRIPT.is_file(), "scripts/week4_preflight.py is the check the manual points at"
    result = _run("--help")
    assert result.returncode == 0
    for flag in ("--config", "--model", "--base-url", "--required-context", "--skip-model"):
        assert flag in result.stdout


def test_a_context_window_below_the_prompt_size_is_reported(
    check_context: Callable[[str, int], list[str]], capsys: pytest.CaptureFixture[str]
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
        problems = check_context(server.url, 8192)
    finally:
        server.close()

    output = capsys.readouterr().out
    assert "context_length=4096" in output
    assert "below the 8192 tokens" in output
    assert "OLLAMA_CONTEXT_LENGTH" in output, "the message must carry the fix"
    assert "[FAIL]" in output
    assert problems == ["context window 4096 < 8192"], "the CLI must receive a blocking problem"


def test_a_context_window_that_is_big_enough_passes_that_check(
    check_context: Callable[[str, int], list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"models": [{"name": "m", "context_length": 16384, "size_vram": 1}]}
    server = _StubServer(json.dumps(payload).encode())
    try:
        problems = check_context(server.url, 8192)
    finally:
        server.close()
    output = capsys.readouterr().out
    assert "covers the 8192 needed" in output
    assert "[ok]" in output
    assert problems == [], "this check must not block a sufficient context window"


def test_no_model_loaded_is_reported_as_a_warning_not_a_pass(
    check_context: Callable[[str, int], list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise the first task run records a model load as its own planning time."""
    server = _StubServer(json.dumps({"models": []}).encode())
    try:
        problems = check_context(server.url, 8192)
    finally:
        server.close()
    output = capsys.readouterr().out
    assert "no model is loaded" in output
    assert "[warn]" in output
    assert "[ok]" not in output
    assert len(problems) == 1, "a warning must still prevent the CLI from reporting a pass"
    assert "warm it up first" in problems[0]
    assert "scripts/week4_warmup.py" in problems[0], "the blocking problem must carry the fix"


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
