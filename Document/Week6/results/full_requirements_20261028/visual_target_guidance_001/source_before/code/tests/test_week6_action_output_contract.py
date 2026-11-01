"""Real SDK serialization with offline HTTP, without a model or executor.

Provider schemas guide generation; the unchanged local parser remains the
authority when a server returns a flat action, invalid enum, or too many steps.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections import deque
from pathlib import Path

import httpx
import pytest
from openai import OpenAI
from test_week6_automatic_decomposition import Clock

from gui_agent.models.base import ModelClient, ModelResponse
from gui_agent.models.openai_compatible import OpenAICompatibleClient
from gui_agent.planning.planner import ACTION_OUTPUT_CONTRACT_VERSION, TaskPlanner
from gui_agent.planning.prompts import SYSTEM_PROMPT
from gui_agent.planning.schemas import PLAN_ACTION_TYPES
from gui_agent.planning.workflow_decomposer import BudgetedModelClient

_PIXELS = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
)
_FLAT = json.dumps({"action_type": "double_click", "target_text": "Continue",
                    "arguments": {}, "status": "success"})


def native(instruction="Continue the requested local operation", *, count=1):
    return json.dumps({"task_id": "local-task", "instruction": instruction, "summary": "Continue locally",
                      "steps": [{"step_id": f"s{i}", "description": "Choose the visible continuation",
                                 "action_type": "click", "target_text": "Continue", "arguments": {},
                                 "status": "pending"} for i in range(count)], "errors": []})


def sdk(replies, bodies, *, on_http=None):
    pending = deque(replies)

    def endpoint(request):
        bodies.append(json.loads(request.content))
        if on_http:
            on_http()
        return httpx.Response(200, json={
            "id": f"offline-{len(bodies)}", "object": "chat.completion", "created": 0,
            "model": "offline-vision", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": pending.popleft()}}],
        })

    client = OpenAICompatibleClient(api_key="offline-only", base_url="http://fixture.invalid/v1",
                                    model_name="offline-vision", max_retries=0, environ={})
    transport = OpenAI(api_key="offline-only", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))
    client._client = transport
    return client, transport


def user(body):
    content = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
    if isinstance(content, list):
        content = next(item["text"] for item in content if item["type"] == "text")
    return json.loads(content)


def test_endpoint_receives_full_plan_schema_and_counted_repair_keeps_caller_context_pixels_and_native_records(tmp_path):
    bodies = []
    instruction = "Continue the requested local operation without changing the final rules"
    valid = json.loads(native(instruction))
    valid["steps"][0]["arguments"] = {"vendor_extension": {"flags": [True, 7, None]}}
    client, transport = sdk([_FLAT, json.dumps(valid)], bodies)
    clock = Clock()
    budgeted = BudgetedModelClient(client, max_requests=4, deadline=clock() + 100,
                                   clock=clock, audit_path=tmp_path / "transport_requests.jsonl")
    image = tmp_path / "frame.png"
    image.write_bytes(_PIXELS)
    context = {"visible_text": "Continue", "observation_id": "fresh-frame", "limits": "one next input"}
    original_context = dict(context)
    try:
        result = TaskPlanner(budgeted, max_steps=1, max_format_retries=1).plan(
            instruction, context=context, image_path=str(image))
    finally:
        transport.close()
    assert result.ok and result.attempts == len(bodies) == budgeted.request_count == 2
    assert result.plan.steps[0].arguments == valid["steps"][0]["arguments"]
    assert context == original_context
    response_format = bodies[0]["response_format"]
    assert response_format == bodies[1]["response_format"]
    assert response_format["json_schema"]["strict"] is False
    schema = response_format["json_schema"]["schema"]
    assert "steps" in schema["required"] and schema["properties"]["steps"]["maxItems"] == 1
    step = schema["$defs"]["PlanStep"]["properties"]
    assert step["action_type"]["enum"] == list(PLAN_ACTION_TYPES)
    assert step["arguments"]["additionalProperties"] is True
    for body in bodies:
        packet = user(body)
        assert packet["instruction"] == instruction and packet["image_path"] == str(image)
        assert all(packet["context"][key] == value for key, value in original_context.items())
        assert packet["context"]["max_steps"] == 1
        assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        images = [item for item in body["messages"][-1]["content"] if item["type"] == "image_url"]
        assert len(images) == 1 and base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]) == _PIXELS
    correction = user(bodies[1])["context"]["plan_format_correction"]
    assert correction["format_error"] == "the plan contains no steps"
    assert correction["previous_response"] == _FLAT
    assert correction["previous_response_sha256"] == hashlib.sha256(_FLAT.encode()).hexdigest()
    assert "has not been executed" in correction["repair_instruction"]
    evidence = result.metadata["action_output_contract"]
    assert evidence["schema_version"] == ACTION_OUTPUT_CONTRACT_VERSION
    records = [json.loads(line) for line in Path(evidence["audit_path"]).read_text("utf8").splitlines()]
    assert len(records) == 2 and records[0]["response"]["content"] == _FLAT
    assert records[0]["parse_error"] == "the plan contains no steps" and "parsed_plan" not in records[0]
    assert records[1]["response"]["content"] == json.dumps(valid)
    assert records[1]["parsed_plan"] == result.plan.as_dict()
    assert result.as_dict()["metadata"] == result.metadata


def test_explicit_unstructured_mode_omits_endpoint_schema_and_keeps_local_parser_authority():
    bodies = []
    client, transport = sdk([_FLAT], bodies)
    try:
        result = TaskPlanner(client, require_structured_output=False, max_format_retries=0).plan("Continue locally")
    finally:
        transport.close()
    assert not result.ok and result.error == "the plan contains no steps"
    assert "response_format" not in bodies[0]
    assert result.metadata["action_output_contract"]["response_format"] is None


@pytest.mark.parametrize("violation,error", [
    ("action", "unknown action_type"), ("status", "status"),
    ("step_bound", "above the limit of 1"), ("ambiguity", "the plan contains no steps"),
])
def test_server_violations_are_rejected_without_local_wrapping_status_repair_or_forced_finish(violation, error):
    candidate = json.loads(native(count=2 if violation == "step_bound" else 1))
    if violation == "action":
        candidate["steps"][0]["action_type"] = "unlisted_verb"
    elif violation == "status":
        candidate["steps"][0]["status"] = "success"
    elif violation == "ambiguity":
        candidate["steps"] = []
        candidate["errors"] = ["The intended control is ambiguous."]
    raw = json.dumps(candidate)
    bodies = []
    client, transport = sdk([raw], bodies)
    try:
        result = TaskPlanner(client, max_steps=1, max_format_retries=0).plan("Continue locally")
    finally:
        transport.close()
    assert not result.ok and result.plan is None and result.attempts == len(bodies) == 1
    assert error in result.error and result.response.content == raw
    assert result.metadata["action_output_contract"]["attempts"][0]["response"]["content"] == raw


@pytest.mark.parametrize("size,inline", [(2000, True), (2001, False)])
def test_large_invalid_reply_is_retained_in_full_but_not_truncated_into_format_feedback(size, inline):
    raw = "X" * size
    bodies = []
    client, transport = sdk([raw, native()], bodies)
    try:
        result = TaskPlanner(client, max_steps=1).plan("Continue the requested local operation")
    finally:
        transport.close()
    assert result.ok and result.attempts == len(bodies) == 2
    correction = user(bodies[1])["context"]["plan_format_correction"]
    assert correction["previous_response_characters"] == size
    assert correction["previous_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert ("previous_response" in correction) is inline
    if inline:
        assert correction["previous_response"] == raw
    assert result.metadata["action_output_contract"]["attempts"][0]["response"]["content"] == raw


def test_oversized_parser_error_is_archived_in_full_and_feedback_contains_only_its_length_and_hash():
    candidate = json.loads(native())
    candidate["steps"][0]["action_type"] = "UNTRUSTED_LONG_ACTION_" * 160
    bodies = []
    client, transport = sdk([json.dumps(candidate), native()], bodies)
    try:
        result = TaskPlanner(client, max_steps=1).plan("Continue locally")
    finally:
        transport.close()
    assert result.ok and result.attempts == len(bodies) == 2
    evidence = result.metadata["action_output_contract"]
    original_error = evidence["attempts"][0]["parse_error"]
    assert len(original_error) > evidence["inline_error_max_characters"] == 2000
    correction = user(bodies[1])["context"]["plan_format_correction"]
    assert "format_error" not in correction and "previous_response" not in correction
    assert correction["format_error_characters"] == len(original_error)
    assert correction["format_error_sha256"] == hashlib.sha256(original_error.encode()).hexdigest()
    assert "UNTRUSTED_LONG_ACTION_" not in json.dumps(correction)


@pytest.mark.parametrize("retries", [0, 1, 2])
def test_format_feedback_does_not_add_attempts_beyond_the_original_retry_cap(retries):
    bodies = []
    client, transport = sdk([_FLAT] * (retries + 1), bodies)
    try:
        result = TaskPlanner(client, max_format_retries=retries).plan("Continue locally")
    finally:
        transport.close()
    assert not result.ok and result.attempts == len(bodies) == retries + 1
    assert all(item["parse_error"] == "the plan contains no steps"
               for item in result.metadata["action_output_contract"]["attempts"])


def test_actual_http_budget_still_stops_format_retry_without_a_second_request(tmp_path):
    bodies = []
    client, transport = sdk([_FLAT, native()], bodies)
    clock = Clock()
    budgeted = BudgetedModelClient(client, max_requests=1, deadline=clock() + 100,
                                   clock=clock, audit_path=tmp_path / "transport_requests.jsonl")
    try:
        result = TaskPlanner(budgeted, max_format_retries=2).plan("Continue locally")
    finally:
        transport.close()
    assert not result.ok and result.attempts == 2 and len(bodies) == budgeted.request_count == 1
    assert "budget exhausted" in result.error
    assert len(result.metadata["action_output_contract"]["attempts"]) == 2


def test_late_response_is_not_repaired_or_used_after_the_shared_deadline():
    bodies = []
    clock = Clock()
    client, transport = sdk([native()], bodies, on_http=lambda: clock.advance(6))
    budgeted = BudgetedModelClient(client, max_requests=4, deadline=clock() + 5, clock=clock)
    try:
        result = TaskPlanner(budgeted, max_format_retries=2).plan("Continue locally")
    finally:
        transport.close()
    assert not result.ok and result.plan is None and result.attempts == len(bodies) == budgeted.request_count == 1
    assert "deadline" in result.error


def test_without_audit_path_only_planner_metadata_is_kept_and_provider_credentials_are_not_copied(monkeypatch):
    class Client(ModelClient):
        def complete(self, messages, **kwargs):
            return ModelResponse(content=native(), provider="offline", model_name="offline",
                                 raw_response={"api_key": "PROVIDER_CREDENTIAL_CANARY"})

    def unexpected_io(*args, **kwargs):
        raise AssertionError("ordinary client must not acquire an audit file")

    monkeypatch.setattr(Path, "open", unexpected_io)
    result = TaskPlanner(Client(model_name="offline", max_retries=0)).plan("Continue locally")
    assert result.ok and result.metadata["action_output_contract"]["audit_path"] is None
    serialized = json.dumps(result.as_dict())
    assert "PROVIDER_CREDENTIAL_CANARY" not in serialized and "raw_response" not in serialized


def test_successful_native_plan_is_blocked_if_its_append_audit_cannot_be_saved(tmp_path, monkeypatch):
    bodies = []
    client, transport = sdk([native()], bodies)
    clock = Clock()
    budgeted = BudgetedModelClient(client, max_requests=4, deadline=clock() + 100,
                                   clock=clock, audit_path=tmp_path / "transport_requests.jsonl")
    original_open = Path.open

    def fail_action_audit(path, *args, **kwargs):
        if path.name == "action_planning_attempts.jsonl":
            raise PermissionError("offline audit fault")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_action_audit)
    try:
        result = TaskPlanner(budgeted, max_format_retries=2).plan("Continue locally")
    finally:
        transport.close()
    assert not result.ok and result.plan is None and result.attempts == len(bodies) == budgeted.request_count == 1
    assert "audit persistence failed" in result.error
    assert result.metadata["action_output_contract"]["attempts"][0]["parsed_plan"]["steps"]


def test_client_mutation_cannot_rewrite_caller_context_schema_or_prior_attempt_evidence():
    class MutatingClient(ModelClient):
        def __init__(self):
            super().__init__(model_name="offline")
            self.usage = {"completion_tokens": 0}

        def complete(self, messages, **kwargs):
            raise AssertionError("this offline double records the generate boundary directly")

        def generate_multimodal(self, instruction, **kwargs):
            self.request_count += 1
            kwargs["context"]["nested"]["labels"].append("UNTRUSTED_MUTATION")
            kwargs["response_format"]["json_schema"]["schema"]["required"].clear()
            self.usage["completion_tokens"] = self.request_count
            return ModelResponse(content=_FLAT if self.request_count == 1 else native(),
                                 provider="offline", model_name="offline", usage=self.usage)

    context = {"nested": {"labels": ["Continue"]}}
    result = TaskPlanner(MutatingClient(), max_steps=1).plan("Continue locally", context=context)
    assert result.ok and result.attempts == 2
    evidence = result.metadata["action_output_contract"]
    assert context == {"nested": {"labels": ["Continue"]}}
    assert "steps" in evidence["response_format"]["json_schema"]["schema"]["required"]
    for attempt in evidence["attempts"]:
        assert attempt["request"]["context"]["nested"]["labels"] == ["Continue"]
        assert "steps" in attempt["request"]["response_format"]["json_schema"]["schema"]["required"]
    assert [attempt["response"]["usage"]["completion_tokens"] for attempt in evidence["attempts"]] == [1, 2]
