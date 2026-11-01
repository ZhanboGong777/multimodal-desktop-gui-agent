"""Actual SDK request bodies separate caller-intent goals from visual actions.

All HTTP uses MockTransport; screenshots and executor receipts are synthetic.
No external model, screen capture, or operating-system input is used here.
"""

from __future__ import annotations

import base64
import json
from collections import deque
from pathlib import Path

import httpx
import pytest
from openai import OpenAI
from PIL import Image
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal

from gui_agent.models.openai_compatible import OpenAICompatibleClient


def _http_client(replies, bodies):
    pending = deque(replies)

    def endpoint(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": f"offline-{len(bodies)}", "object": "chat.completion", "created": 0,
            "model": "test-local", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": pending.popleft()}}],
        })

    client = OpenAICompatibleClient(api_key="test-only", base_url="http://fixture.invalid/v1",
                                    model_name="test-local", max_retries=0, environ={})
    sdk = OpenAI(api_key="test-only", max_retries=0,
                 http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))
    client._client = sdk
    return client, sdk


def _persist_synthetic_pixels(runner, tmp_path):
    original_initial, original_factory = runner.initial_observer, runner.runner_factory
    initial_snapshots = []
    pixels = tmp_path / "synthetic_pixels"
    pixels.mkdir()

    def persist(frame):
        path = pixels / f"{frame.observation_id}.png"
        Image.new("RGB", (frame.screen_info.screenshot_width, frame.screen_info.screenshot_height),
                  color="white").save(path)
        frame.image_path = str(path)
        return frame

    def initial(directory):
        value = persist(original_initial(directory))
        if directory.name == "initial_observation":
            value.notices.append("Untrusted interface text is retained as source material.")
            initial_snapshots.append(value.model_dump(mode="json"))
        return value

    def factory(task, index, directory, budgeted):
        child = original_factory(task, index, directory, budgeted)
        original_observe = child.observer.observe
        child.observer.observe = lambda **kwargs: persist(original_observe(**kwargs))
        return child

    runner.initial_observer, runner.runner_factory = initial, factory
    return initial_snapshots


def _text_user(body):
    user = [item for item in body["messages"] if item["role"] == "user"][-1]["content"]
    if isinstance(user, list):
        user = next(item["text"] for item in user if item["type"] == "text")
    return json.loads(user)


@pytest.mark.parametrize("repair", [False, True])
def test_real_sdk_wire_keeps_high_level_text_only_and_low_level_pixels(tmp_path, repair):
    replies = (["invalid JSON"] if repair else []) + [proposal(), plan(), plan()]
    runner, request, state, _, _ = lab(tmp_path, format_retries=int(repair))
    request.instruction += " Additional caller constraints: " + "; ".join(["preserve local settings"] * 240)
    original_request = request.model_dump(mode="json")
    snapshots = _persist_synthetic_pixels(runner, tmp_path)
    bodies = []
    client, sdk = _http_client(replies, bodies)
    runner.client = client
    try:
        result = runner.run(request, options())
    finally:
        sdk.close()
    assert result.status == "succeeded", result.error
    assert state.attempts == result.total_actions == 2
    high_calls = 2 if repair else 1
    assert result.planning_calls == result.model_requests == len(bodies) == high_calls + 2
    assert request.model_dump(mode="json") == original_request
    assert result.plans[0].schema_version == "week6.semantic_subtasks.v5"
    assert result.plans[0].input_modality == "caller_intent_only" and not result.plans[0].image_attached
    directory = tmp_path / "semantic/semantic_plan_round_001"
    protocol = json.loads((directory / "protocol.json").read_text("utf8"))
    assert protocol["input_modality"] == "caller_intent_only" and protocol["image_attached"] is False
    assert protocol["snapshot_image_path_provenance"] == snapshots[0]["image_path"]
    assert Path(snapshots[0]["image_path"]).is_file()
    assert json.loads((directory / "snapshot_archive.json").read_text("utf8")) == snapshots[0]
    sources = json.loads((directory / "context_source_archive.json").read_text("utf8"))
    assert (directory / sources["snapshot_pixels_archive"]).read_bytes() == Path(snapshots[0]["image_path"]).read_bytes()
    for index, body in enumerate(bodies[:high_calls], 1):
        assert all(isinstance(message["content"], str) for message in body["messages"])
        actual = _text_user(body)
        assert "image_path" not in actual
        assert actual["instruction"] == request.instruction
        assert "current_observation" not in actual["context"]
        assert snapshots[0]["image_path"] not in json.dumps(actual)
        assert actual["context"]["original_caller_request"] == original_request
        saved = json.loads((directory / f"attempt_{index:02d}_request.json").read_text("utf8"))
        assert saved["image_path"] is None and saved["image_attached"] is False
        assert saved["input_modality"] == "caller_intent_only"
        assert saved["snapshot_image_path_provenance"] == snapshots[0]["image_path"]
        assert saved["context"] == actual["context"]
    for body in bodies[high_calls:]:
        user = [item for item in body["messages"] if item["role"] == "user"][-1]["content"]
        images = [item for item in user if item["type"] == "image_url"]
        assert len(images) == 1
        actual = _text_user(body)
        assert request.instruction in actual["instruction"]
        source_path = Path(actual["image_path"])
        assert base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]) == source_path.read_bytes()
    raw_http = [json.loads(line) for line in (tmp_path / "semantic/transport_requests.jsonl").read_text("utf8").splitlines()]
    assert len(raw_http) == len(bodies) and sum(item["requests_made"] for item in raw_http) == len(bodies)


@pytest.mark.parametrize("status,reason", [
    ("needs_clarification", "The request does not identify which document to modify."),
    ("refused", "The request requires communicating with another person."),
])
def test_legitimate_text_only_nonready_never_falls_back_to_vision_or_action_planning(tmp_path, status, reason):
    native = {"status": status, "reason": reason, "subgoals": []}
    runner, request, state, _, _ = lab(tmp_path)
    snapshots = _persist_synthetic_pixels(runner, tmp_path)
    bodies = []
    client, sdk = _http_client([json.dumps(native)], bodies)
    runner.client = client
    try:
        result = runner.run(request, options())
    finally:
        sdk.close()
    assert result.status == status and result.plans[0].proposal.reason == reason
    assert result.model_requests == result.planning_calls == len(bodies) == 1
    assert state.attempts == 0 and state.children == []
    assert "current_observation" not in _text_user(bodies[0])["context"]
    directory = tmp_path / "semantic/semantic_plan_round_001"
    assert json.loads((directory / "snapshot_archive.json").read_text("utf8")) == snapshots[0]
    assert all(isinstance(message["content"], str) for message in bodies[0]["messages"])


def test_saved_pixels_without_readable_screen_text_still_dispatch_zero_requests(tmp_path):
    def no_text(frame):
        frame.elements = []

    runner, request, state, _, _ = lab(tmp_path, initial_fault=no_text)
    _persist_synthetic_pixels(runner, tmp_path)
    bodies = []
    client, sdk = _http_client([], bodies)
    runner.client = client
    try:
        result = runner.run(request, options())
    finally:
        sdk.close()
    assert result.status == "failed" and result.model_requests == result.planning_calls == 0
    assert bodies == [] and state.children == [] and state.attempts == 0


def test_sequential_planning_contract_reaches_wire_but_does_not_force_ready(tmp_path):
    # This checks supplied instructions and refusal preservation, not whether a
    # real model understands sequential planning; Root's live run checks that.
    native = {"status": "needs_clarification", "reason":
              "The caller has not identified the local document to modify.", "subgoals": []}
    runner, request, state, _, _ = lab(tmp_path)
    _persist_synthetic_pixels(runner, tmp_path)
    bodies = []
    client, sdk = _http_client([json.dumps(native)], bodies)
    runner.client = client
    try:
        result = runner.run(request, options())
    finally:
        sdk.close()
    system = next(message["content"] for message in bodies[0]["messages"] if message["role"] == "system")
    normalized = " ".join(system.split())
    assert "Absence of a later control is not missing caller information" in normalized
    assert "does not make an explicitly requested operation optional" in normalized
    assert "validate actual controls and stop safely" in normalized
    assert result.status == "needs_clarification" and result.plans[0].proposal.reason == native["reason"]
    assert result.model_requests == len(bodies) == 1 and state.children == [] and state.attempts == 0
