"""The loop against a real OpenAI-compatible server over a real socket.

The unit tests inject a fake planner, which cannot show that the prompt actually
carries what the model needs to aim. Here a local server answers the real request:
it reads the element ids out of the prompt it was sent and returns a plan that
targets one of them. If the runtime ever stopped putting the observation into the
prompt, this test would fail with "no element matches" rather than passing on a
canned reply.

Only the vision model itself is simulated. The HTTP call, the JSON parsing, the
plan validation, the action resolution and the dry-run dispatch all run for real.
"""

from __future__ import annotations

import base64
import json
import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

from gui_agent.config import ModelConfig
from gui_agent.models import create_model_client
from gui_agent.planning import TaskPlanner
from gui_agent.recording import RunSession
from gui_agent.runtime import (
    ActionAdapter,
    ExecutionOptions,
    ObservationSnapshot,
    TaskRecorder,
    TaskRunner,
    TaskSpec,
)
from gui_agent.runtime.recorder import TaskRecorder as _Recorder  # noqa: F401 - re-export clarity
from gui_agent.runtime.schemas import ElementRef
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult, BoundingBox, Point, ScreenInfo

ELEMENT_ID = re.compile(r"obs-\d+-e\d+")
#: A prompt line reads  obs-0002-e001  'Browser'  conf=0.90  center=(250,78)  ...
ELEMENT_LINE = re.compile(r"(obs-\d+-e\d+)\s+'([^']*)'")


class _Handler(BaseHTTPRequestHandler):
    """Answers with a plan aimed at an element the caller actually listed."""

    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(body)

        prompt = _prompt_text(body)
        pairs = ELEMENT_LINE.findall(prompt)
        target, target_text = pairs[0] if pairs else (None, None)

        steps: list[dict[str, Any]] = []
        if target:
            # Both forms, as a real model would produce them: the id points at the
            # frame the plan was written from, and the text is what lets the
            # adapter re-locate the same control after the runner re-observes.
            steps.append(
                {
                    "step_id": "step-1",
                    "description": "Click the first visible labelled element",
                    "action_type": "click",
                    "target_text": target_text,
                    "arguments": {"element_id": target},
                    "expected_result": "the screen reacts",
                }
            )
        steps.append(
            {
                "step_id": f"step-{len(steps) + 1}",
                "description": "Stop",
                "action_type": "finish",
            }
        )

        plan = {
            "task_id": "t1",
            "instruction": "interact with the screen",
            "summary": f"targeting {target}",
            "steps": steps,
            "assumptions": [],
            "requires_confirmation": True,
            "errors": [],
        }
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(plan)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr logging."""


def _prompt_text(body: dict[str, Any]) -> str:
    """Flatten whatever the client sent into one searchable string."""
    chunks: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(str(block.get("text", "")))
    return "\n".join(chunks)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_model_client reads the key from the environment.

    The stub never checks it, but the client refuses to start without one - which
    is itself the behaviour Week 3 added a test for.
    """
    monkeypatch.setenv("GUI_AGENT_API_KEY", "test-key")


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


class _Observer:
    def __init__(self, frames: list[ObservationSnapshot]) -> None:
        self.frames = frames
        self.calls = 0

    def observe(self, *, observation_id: str | None = None) -> ObservationSnapshot:
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return frame


class _Executor:
    def __init__(self) -> None:
        self.actions: list[tuple[Any, bool]] = []

    def execute(self, action, *, dry_run=None, screen=None) -> ActionResult:
        self.actions.append((action, bool(dry_run)))
        return ActionResult(action=action, success=True, dry_run=bool(dry_run))


def _frame(
    observation_id: str, texts: tuple[str, ...], image_path: str | None = None
) -> ObservationSnapshot:
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=datetime.now(UTC),
        image_path=image_path,
        screen_info=ScreenInfo(
            screenshot_width=1920, screenshot_height=1080, control_width=1920, control_height=1080
        ),
        elements=[
            ElementRef(
                element_id=f"{observation_id}-e{index:03d}",
                text=text,
                bounding_box=BoundingBox(
                    left=100, top=60 + index * 40, right=400, bottom=96 + index * 40
                ),
                center=Point(x=250, y=78 + index * 40),
                confidence=0.9,
            )
            for index, text in enumerate(texts)
        ],
    )


def _run(server_url: str, tmp_path: Path, *, execute: bool = False, frames=None):
    client = create_model_client(
        ModelConfig(provider="openai_compatible", model_name="stub-vision", base_url=server_url)
    )
    observer = _Observer(
        frames
        if frames is not None
        else [
            _frame("obs-0001", ("Desktop", "Browser", "Files")),
            _frame("obs-0002", ("Desktop", "Browser", "Files")),
            _frame("obs-0003", ("Desktop", "Browser", "Files")),
        ]
    )
    executor = _Executor()
    recorder = TaskRecorder(RunSession.create(tmp_path, session_id="it-1"))
    runner = TaskRunner(
        observer=observer,
        planner=TaskPlanner(client, max_steps=5),
        adapter=ActionAdapter(platform="darwin"),
        executor=executor,
        verifier=Verifier(),
        recorder=recorder,
        sleep=lambda _s: None,
    )
    task = TaskSpec(
        case_id="IT",
        instruction="interact with the screen",
        expect_text=["Browser"],
        success_rules=["the browser is present"],
    )
    result = runner.run(task, ExecutionOptions(execute=execute, confirm=False))
    return result, executor


def test_the_prompt_carries_the_element_ids_the_model_needs(server: str, tmp_path: Path) -> None:
    """The server can only aim at an element because the runtime listed them."""
    result, executor = _run(server, tmp_path)

    sent = _Handler.requests[-1]
    pairs = ELEMENT_LINE.findall(_prompt_text(sent))
    assert pairs, "the observation never reached the model"
    # The plan is written from the first frame, so those are the ids it carries.
    planning_ids = [element_id for element_id, _ in pairs]
    assert all(element_id.startswith("obs-0001-") for element_id in planning_ids)

    # ... and the plan the server derived from them resolved, in a *later* frame.
    assert result.status == "dry_run_completed", result.notes
    assert len(executor.actions) == 1
    action, dispatched_dry = executor.actions[0]
    assert dispatched_dry is True, "a dry run must not dispatch"
    assert (action.x, action.y) == (250, 78), "the coordinate must come from the listed element"

    # The step that acted was resolved against the re-observed frame, not the one
    # the plan was written from - which is why the text target had to be re-bound.
    assert result.steps[0].observation_id == "obs-0002"
    assert result.steps[0].resolved is not None
    assert "re-bound from obs-0001" in result.steps[0].resolved.note


def test_the_prompt_asks_for_one_json_object(server: str, tmp_path: Path) -> None:
    _run(server, tmp_path)
    sent = _Handler.requests[-1]
    system = sent["messages"][0]
    assert system["role"] == "system"
    assert "ONE JSON object" in system["content"]
    assert "element_id" in system["content"]


def test_the_finish_step_is_never_dispatched(server: str, tmp_path: Path) -> None:
    result, executor = _run(server, tmp_path)
    kinds = [action.action_type for action, _ in executor.actions]
    assert kinds == ["click"], "finish closes the plan; it is not an action"
    assert any("finish reached" in note for note in result.notes)
    assert all(step.action_type != "finish" or step.resolved is None for step in result.steps)


def test_a_real_execution_marks_dry_run_false(server: str, tmp_path: Path) -> None:
    """The same path with --execute: the action must be dispatched for real."""
    result, executor = _run(server, tmp_path, execute=True)
    _, dispatched_dry = executor.actions[0]
    assert dispatched_dry is False
    assert result.execute is True
    assert result.status in {"succeeded", "failed"}, result.status


def test_the_run_is_recorded_with_the_observation_ids(server: str, tmp_path: Path) -> None:
    _run(server, tmp_path)
    steps = [
        json.loads(line)
        for line in (tmp_path / "it-1" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert steps, "the run must leave step records"
    assert steps[0]["observation_id"].startswith("obs-")
    assert (tmp_path / "it-1" / "task_summary.json").exists()


def _png(path: Path, colour: tuple[int, int, int]) -> Path:
    """A real PNG file. Distinct per colour so the bytes identify the frame."""
    Image.new("RGB", (8, 8), colour).save(path)
    return path


def _image_blocks(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Every image block in the request, wherever the client put it."""
    found: list[dict[str, Any]] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            found.extend(
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "image_url"
            )
    return found


def _decoded_image(url: str) -> bytes:
    prefix, _, encoded = url.partition(",")
    assert prefix.startswith("data:image/"), f"not a data URL: {prefix!r}"
    return base64.b64decode(encoded)


def test_the_model_receives_the_pixels_and_not_just_a_path(server: str, tmp_path: Path) -> None:
    """Section 8.1's acceptance line, at the only layer where it means anything.

    The hand-back note is explicit that a screenshot path in the prompt text is
    not the same as the model seeing the image, and that acceptance checks the
    request itself. Nothing did: the tests around the encoder call it directly,
    so deleting the image from every outgoing request - the one property the
    whole week exists to provide - left the entire suite green. This asserts on the
    body a real server received over a real socket, and compares the decoded
    bytes to the file the observation captured rather than to a prefix.
    """
    shot = _png(tmp_path / "frame-1.png", (255, 0, 0))
    _run(
        server,
        tmp_path,
        frames=[
            _frame("obs-0001", ("Desktop", "Browser", "Files"), image_path=str(shot)),
            _frame("obs-0002", ("Desktop", "Browser", "Files"), image_path=str(shot)),
            _frame("obs-0003", ("Desktop", "Browser", "Files"), image_path=str(shot)),
        ],
    )

    blocks = _image_blocks(_Handler.requests[-1])
    assert len(blocks) == 1, f"the request carried {len(blocks)} images, not one"
    url = blocks[0]["image_url"]["url"]
    assert _decoded_image(url) == shot.read_bytes(), (
        "the model was sent something other than the frame the observation captured"
    )
    assert "base64" in url.split(",", 1)[0]


def test_the_image_is_the_frame_the_plan_was_written_from(server: str, tmp_path: Path) -> None:
    """Planning happens once, against the current screen - so frame one, not two.

    A stale or later frame would be invisible to a prefix check and to every
    test that only asks whether an image was attached at all.
    """
    first = _png(tmp_path / "first.png", (255, 0, 0))
    later = _png(tmp_path / "later.png", (0, 0, 255))
    _run(
        server,
        tmp_path,
        frames=[
            _frame("obs-0001", ("Desktop", "Browser", "Files"), image_path=str(first)),
            _frame("obs-0002", ("Desktop", "Browser", "Files"), image_path=str(later)),
            _frame("obs-0003", ("Desktop", "Browser", "Files"), image_path=str(later)),
        ],
    )

    sent = _decoded_image(_image_blocks(_Handler.requests[-1])[0]["image_url"]["url"])
    assert sent == first.read_bytes()
    assert sent != later.read_bytes()
    # The same request listed the first frame's elements, so image and element
    # list describe one screen rather than two.
    ids = {
        element_id
        for element_id, _ in ELEMENT_LINE.findall(_prompt_text(_Handler.requests[-1]))
    }
    assert ids and all(element_id.startswith("obs-0001-") for element_id in ids)


def test_a_frame_without_a_screenshot_still_sends_a_text_request(
    server: str, tmp_path: Path
) -> None:
    """No image is not an error and not an empty image block.

    The observer can return a frame whose capture produced no file. The text turn
    still has to arrive, and the request must not carry a malformed block that a
    server would reject.
    """
    result, _ = _run(server, tmp_path)

    sent = _Handler.requests[-1]
    assert _image_blocks(sent) == []
    assert "Browser" in _prompt_text(sent), "the text turn went missing with the image"
    assert result.status == "dry_run_completed", result.notes


def test_an_unreadable_screenshot_is_refused_rather_than_dropped(
    server: str, tmp_path: Path
) -> None:
    """A path that no longer exists stops the run instead of silently blinding it.

    Between the observation and the request the file can be gone - a capture
    directory cleaned up, a tmp reaper, the next frame overwriting it. The run is
    blocked before the request is sent, and the note names the file: a model
    asked to plan from nothing would otherwise return a confident plan about a
    screen it never saw, which is the failure this whole layer exists to avoid.
    """
    missing = tmp_path / "gone.png"
    _png(missing, (0, 255, 0))
    frames = [
        _frame("obs-0001", ("Desktop", "Browser", "Files"), image_path=str(missing)),
        _frame("obs-0002", ("Desktop", "Browser", "Files")),
        _frame("obs-0003", ("Desktop", "Browser", "Files")),
    ]
    missing.unlink()

    result, executor = _run(server, tmp_path, frames=frames)

    assert result.status == "blocked", result.notes
    assert executor.actions == [], "nothing may be dispatched when the model never saw the screen"
    assert _Handler.requests == [], "no request should have been sent at all"
    assert "image not found" in " ".join(result.notes)
    assert "gone.png" in " ".join(result.notes)


def test_a_screenshot_that_is_not_an_image_is_refused(server: str, tmp_path: Path) -> None:
    """A wrong type is refused rather than guessed at, over the real path.

    Guessing the MIME type from the bytes would be a reasonable-looking
    convenience and the wrong one: the endpoint's own decoder is the only thing
    that knows what it accepts, and a mislabelled image fails there with a
    message that says nothing about which file was at fault.
    """
    not_an_image = tmp_path / "screen.txt"
    not_an_image.write_text("this is not a screenshot", encoding="utf-8")

    result, executor = _run(
        server,
        tmp_path,
        frames=[
            _frame("obs-0001", ("Desktop", "Browser", "Files"), image_path=str(not_an_image)),
            _frame("obs-0002", ("Desktop", "Browser", "Files")),
            _frame("obs-0003", ("Desktop", "Browser", "Files")),
        ],
    )

    assert result.status == "blocked", result.notes
    assert executor.actions == []
    assert _Handler.requests == []
    note = " ".join(result.notes)
    assert "unsupported image type" in note
    assert ".txt" in note
    assert ".png" in note, "the refusal should say what it does accept"
