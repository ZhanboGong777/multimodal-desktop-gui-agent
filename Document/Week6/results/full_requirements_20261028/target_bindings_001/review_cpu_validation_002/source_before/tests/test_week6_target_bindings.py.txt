"""Observed target pairs guide native generation; runtime still owns dispatch."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_week6_action_output_contract import _FLAT, _PIXELS, native, sdk, user
from test_week6_automatic_decomposition import Clock, ScriptedClient, _frame
from test_week6_rolling_actions import options
from test_week6_semantic_subtasks import lab

from gui_agent.planning.planner import TaskPlanner, _task_plan_response_format
from gui_agent.planning.target_bindings import (
    TARGET_BINDING_SCHEMA_VERSION,
    TARGET_BINDINGS_VERSION,
    derive_target_bindings,
    validate_target_bindings,
)
from gui_agent.planning.workflow_decomposer import BudgetedModelClient
from gui_agent.runtime.action_adapter import TARGET_CONSISTENCY_ERROR_PREFIX
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import ElementRef, TaskSpec
from gui_agent.schemas import BoundingBox


def element(identifier, text="", *, source="ocr", confidence=.95, box=(10, 10, 110, 40)):
    bounds = BoundingBox(left=box[0], top=box[1], right=box[2], bottom=box[3])
    return ElementRef(element_id=identifier, text=text, source=source, confidence=confidence,
                      bounding_box=bounds, center=bounds.center)


def frame(*elements):
    return _frame("current", []).model_copy(update={"elements": list(elements)})


def sample():
    return frame(element("continue", "Continue"), element("cancel", "Cancel", box=(200, 10, 300, 40)),
                 element("second-continue", "Continue", box=(400, 10, 500, 40)),
                 element("visual", source="contour", box=(10, 150, 100, 200)))


def validator(bindings):
    schema = _task_plan_response_format(1, target_bindings=bindings)["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def candidate(query="Continue", args=None, *, action="click"):
    value = json.loads(native())
    value["steps"][0].update(action_type=action, target_text=query, arguments=args or {})
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf8")).hexdigest()


def test_raw_labels_confidence_and_positive_area_overlap_determine_candidates_without_task_knowledge():
    observed = frame(
        element("first", " Keep Original "), element("case", "keep original", box=(200, 10, 300, 40)),
        element("low", "Ignored", confidence=.34, box=(10, 60, 100, 90)),
        element("one", source="contour", confidence=.1, box=(0, 0, 120, 50)),
        element("edge", source="contour", box=(110, 10, 150, 40)),
        element("far", source="contour", box=(500, 400, 600, 450)))
    original = observed.model_dump(mode="json")
    packet = derive_target_bindings(observed)
    assert observed.model_dump(mode="json") == original
    assert packet["schema_version"] == TARGET_BINDINGS_VERSION and packet["min_confidence"] == .35
    assert packet["labels"] == [{"text": " Keep Original ", "element_ids": ["first", "one"]},
                                 {"text": "keep original", "element_ids": ["case"]}]
    records = {record["element_id"]: record for record in packet["candidates"]}
    assert records["one"]["classification"] == "anonymous_with_single_ocr"
    assert records["one"]["overlapping_ocr_ids"] == ["first"]
    assert records["low"]["classification"] == "untrusted_or_other"
    assert packet["anonymous_without_ocr_ids"] == ["edge", "far"]
    packet["labels"][0]["element_ids"].clear()
    assert observed.model_dump(mode="json") == original


@pytest.mark.parametrize("id_alias", ["element_id", "target_element"])
@pytest.mark.parametrize("query_alias", ["target_text", "target"])
def test_standard_jsonschema_rejects_wrong_observed_pair_and_accepts_matching_duplicate_label(id_alias, query_alias):
    check = validator(derive_target_bindings(sample()))
    args = {id_alias: "continue", "vendor": {"arbitrary": [True, None, 7]}}
    value = candidate(args=args)
    if query_alias == "target":
        value["steps"][0]["target_text"] = None
        args["target"] = "Continue"
    assert check.is_valid(value)
    args[id_alias] = "second-continue"
    assert check.is_valid(value), "Consistent ID can disambiguate duplicate labels."
    args[id_alias] = "cancel"
    assert not check.is_valid(value)
    args[id_alias] = "old-frame-id"
    assert not check.is_valid(value)


@pytest.mark.parametrize("primary,secondary", [("element_id", "target_element"), ("target_element", "element_id")])
def test_conflicting_secondary_id_and_query_alias_cannot_bypass_positive_union(primary, secondary):
    check = validator(derive_target_bindings(sample()))
    value = candidate(args={primary: "continue", secondary: "cancel"})
    assert not check.is_valid(value)
    value["steps"][0]["arguments"][secondary] = "second-continue"
    assert check.is_valid(value)
    value["steps"][0]["arguments"]["target"] = "Cancel"
    assert not check.is_valid(value)


@pytest.mark.parametrize("query,args,action", [
    ("arbitrary words spanning OCR runs", {}, "click"),
    (None, {"target": "arbitrary word run"}, "click"),
    (None, {"element_id": "cancel"}, "click"),
    (None, {"target_element": "cancel"}, "click"),
    ("unlabelled icon appearance", {"element_id": "visual"}, "double_click"),
    ("caller text", {"text": "arbitrary payload", "vendor": 12}, "type_text"),
    (None, {"seconds": .1}, "wait"),
    (None, {"start_x": 10, "end_x": 20}, "drag"),
])
def test_text_only_id_only_visual_and_nonpoint_paths_remain_native_choices(query, args, action):
    assert validator(derive_target_bindings(sample())).is_valid(candidate(query, args, action=action))


def test_mixed_container_and_external_subset_overlap_never_gain_visual_no_ocr_whitelist():
    mixed = element("container", source="contour", box=(0, 0, 400, 100))
    observed = frame(mixed, element("left", "Left"), element("right", "Right", box=(200, 10, 300, 40)))
    packet = derive_target_bindings(observed)
    assert packet["candidates"][0]["classification"] == "anonymous_mixed_ocr"
    assert not packet["anonymous_without_ocr_ids"]
    check = validator(packet)
    assert not check.is_valid(candidate("Left", {"element_id": "container"}))
    assert check.is_valid(candidate(None, {"element_id": "container"}))
    subset = observed.model_copy(update={"elements": [mixed]})
    filtered = derive_target_bindings(subset, source_snapshot=observed)
    assert filtered["scope"] == "authorised_subset"
    assert filtered["candidates"][0]["external_ocr_overlap"] is True
    assert filtered["candidates"][0]["classification"] == "untrusted_or_other"
    assert filtered["labels"] == filtered["anonymous_without_ocr_ids"] == []
    serialized = json.dumps(filtered)
    assert all(secret not in serialized for secret in ('"element_id": "left"', '"element_id": "right"', '"Left"', '"Right"'))
    assert not validator(filtered).is_valid(candidate("Left", {"element_id": "container"}))


@pytest.mark.parametrize("mutation", ["version", "stale", "classification", "whitelist", "duplicate", "zero_box", "extra"])
def test_bad_or_stale_binding_packet_fails_before_any_model_request_without_legacy_fallback(mutation):
    packet = derive_target_bindings(sample())
    if mutation == "version":
        packet["schema_version"] = "unsupported"
    elif mutation == "stale":
        packet["observation_id"] = "old"
    elif mutation == "classification":
        packet["candidates"][0]["classification"] = "anonymous_without_ocr"
    elif mutation == "whitelist":
        packet["anonymous_without_ocr_ids"].append("cancel")
    elif mutation == "duplicate":
        packet["candidates"].append(deepcopy(packet["candidates"][0]))
    elif mutation == "zero_box":
        packet["candidates"][0]["bounding_box"]["right"] = packet["candidates"][0]["bounding_box"]["left"]
    else:
        packet["unexpected"] = True
    client = ScriptedClient([native()])
    result = TaskPlanner(client).plan("Continue locally", context={"observation_id": "current", "target_bindings": packet})
    assert not result.ok and result.attempts == len(client.calls) == 0
    assert "invalid target_bindings" in result.error


def test_union_is_positive_self_contained_and_does_not_close_other_arguments():
    schema = _task_plan_response_format(1, target_bindings=derive_target_bindings(sample()))["json_schema"]["schema"]
    for branch in schema["$defs"]["PlanStep"]["anyOf"]:
        assert "properties" in branch and "required" in branch and "anyOf" not in branch
        assert branch["properties"]["arguments"]["additionalProperties"] is True

    def inspect(value):
        if isinstance(value, dict):
            assert not {"not", "if", "then", "allOf", "dependentSchemas"}.intersection(value)
            assert value.get("enum") != []
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    inspect(schema)
    assert validator(derive_target_bindings(frame())).is_valid(candidate("unknown but native text"))


def test_actual_sdk_serialization_and_each_format_attempt_bind_same_frame_schema_raw_response_and_hashes(tmp_path):
    packet = derive_target_bindings(sample())
    value = candidate(args={"element_id": "continue", "vendor": {"flags": [True, None]}})
    bodies = []
    client, transport = sdk([_FLAT, json.dumps(value)], bodies)
    clock = Clock()
    budgeted = BudgetedModelClient(client, max_requests=2, deadline=clock() + 50, clock=clock,
                                   audit_path=tmp_path / "transport_requests.jsonl")
    image = tmp_path / "actual.png"
    image.write_bytes(_PIXELS)
    instruction = "Continue the caller's chosen local operation"
    context = {"observation_id": "current", "visible_text": "Continue / Cancel", "target_bindings": packet}
    original = deepcopy(context)
    try:
        result = TaskPlanner(budgeted, max_steps=1).plan(instruction, context=context, image_path=str(image))
    finally:
        transport.close()
    assert result.ok and result.attempts == budgeted.request_count == len(bodies) == 2
    assert context == original and result.plan.steps[0].arguments == value["steps"][0]["arguments"]
    response_format = _task_plan_response_format(1, target_bindings=packet)
    expected = {"schema_version": TARGET_BINDING_SCHEMA_VERSION, "bindings": packet,
                "bindings_sha256": digest(packet), "response_format_sha256": digest(response_format),
                "generation_only": True}
    evidence = result.metadata["action_output_contract"]
    assert evidence["target_binding_contract"] == expected
    records = [json.loads(line) for line in Path(evidence["audit_path"]).read_text("utf8").splitlines()]
    assert [{key: value for key, value in record.items() if key != "schema_version"} for record in records] == evidence["attempts"]
    for body, record in zip(bodies, records, strict=True):
        assert body["response_format"] == response_format == record["request"]["response_format"]
        assert user(body)["instruction"] == instruction and user(body)["image_path"] == str(image)
        assert user(body)["context"]["target_bindings"] == packet
        assert record["target_binding_contract"] == expected
        assert len([item for item in body["messages"][-1]["content"] if item["type"] == "image_url"]) == 1
    assert records[0]["response"]["content"] == _FLAT
    assert user(bodies[1])["context"]["plan_format_correction"]["previous_response"] == _FLAT
    assert records[1]["response"]["content"] == json.dumps(value)


def test_no_binding_legacy_and_explicit_unstructured_mode_have_no_invented_contract():
    client = ScriptedClient([native(), native()])
    legacy = TaskPlanner(client).plan("Continue locally")
    assert legacy.ok and "target_binding_contract" not in legacy.metadata["action_output_contract"]
    assert "anyOf" not in client.calls[0]["kwargs"]["response_format"]["json_schema"]["schema"]["$defs"]["PlanStep"]
    packet = derive_target_bindings(sample())
    result = TaskPlanner(client, require_structured_output=False).plan("Continue locally", context={
        "observation_id": "current", "target_bindings": packet})
    assert result.ok and "response_format" not in client.calls[1]["kwargs"]
    assert result.metadata["action_output_contract"]["target_binding_contract"]["response_format_sha256"] is None


def test_message_context_exposes_only_authorised_subset_and_suppresses_external_overlap_without_text_leakage():
    editor = element("editor", source="contour", box=(10, 300, 300, 400))
    send = element("send", "Send", box=(310, 350, 370, 390))
    secret = element("private", "PRIVATE OUTSIDE TEXT", box=(0, 290, 30, 320))
    observed = frame(editor, send, secret)
    task = TaskSpec(case_id="message", instruction="Use the authorised local editor", expect_text=["MARKER"],
                    message_conversation="local-conversation")
    fake = SimpleNamespace(adapter=SimpleNamespace(platform="win32", min_confidence=.35),
                           _message_text_typed=False, _message_composer=None)
    unavailable = TaskRunner._context(fake, observed, task, options())
    assert unavailable["visible_text"] == "" and unavailable["target_bindings"]["candidates"] == []
    assert "PRIVATE" not in json.dumps(unavailable) and '"private"' not in json.dumps(unavailable)
    fake._message_composer = (observed,)
    fake._message_controls = lambda _frame: (editor.bounding_box, [editor, send])
    allowed = TaskRunner._context(fake, observed, task, options())
    assert allowed["message_task"]["candidate_ids"] == ["editor", "send"]
    assert [x["element_id"] for x in allowed["target_bindings"]["candidates"]] == ["editor", "send"]
    assert allowed["target_bindings"]["candidates"][0]["external_ocr_overlap"] is True
    assert not allowed["target_bindings"]["anonymous_without_ocr_ids"]
    assert "PRIVATE" not in json.dumps(allowed) and '"private"' not in json.dumps(allowed)


def test_provider_ignoring_candidate_schema_still_hits_real_runtime_guard_and_shared_replanning_without_native_rewrite(tmp_path):
    runtime, request, state, client, _ = lab(tmp_path)
    original_factory, original_complete = runtime.runner_factory, client.complete
    low_calls = []

    def complete(messages, **kwargs):
        context = json.loads(messages[-1]["content"]).get("context", {})
        if "observation_id" in context:
            low_calls.append(deepcopy(context))
            value = candidate("Advance control", {"element_id": context["observation_id"] + (
                "-e999" if len(low_calls) == 1 else "-e000")})
            client.responses[0] = json.dumps(value)
            check = Draft202012Validator(kwargs["response_format"]["json_schema"]["schema"])
            assert check.is_valid(value) is (len(low_calls) != 1)
        else:
            assert "target_bindings" not in json.dumps(context) and "Advance control" not in json.dumps(context)
        return original_complete(messages, **kwargs)

    client.complete = complete

    def factory(task, index, directory, budgeted):
        child = original_factory(task, index, directory, budgeted)
        observe = child.observer.observe

        def observed(**kwargs):
            value = observe(**kwargs)
            value.elements.append(element(value.observation_id + "-e999", "OTHER CONTROL", box=(300, 10, 400, 40)))
            return value

        child.observer.observe = observed
        return child

    runtime.runner_factory = factory
    result = runtime.run(request, options())
    assert result.status == "succeeded" and result.total_actions == state.attempts == 2
    assert result.model_requests == result.planning_calls == len(client.calls) == 4
    audits = [json.loads(path.read_text("utf8")) for path in (tmp_path / "semantic").glob("goal_*/001_*/semantic_goal_audit.json")]
    assert len(audits) == 2 and audits[0]["steps"][0]["action_result"] is None
    assert audits[0]["steps"][0]["error"].startswith(TARGET_CONSISTENCY_ERROR_PREFIX)
    assert all(value["eligible_for_semantic_credit"] for value in audits)
    first = next(path for path in (tmp_path / "semantic").rglob("action_planning_attempts.jsonl"))
    records = [json.loads(line) for line in first.read_text("utf8").splitlines()]
    raw = json.loads(records[0]["response"]["content"])
    assert raw["steps"][0]["arguments"]["element_id"].endswith("-e999")
    assert records[0]["parsed_plan"]["steps"][0]["arguments"] == raw["steps"][0]["arguments"]


def test_subset_cannot_invent_or_relabel_source_candidate():
    observed = sample()
    forged = observed.model_copy(deep=True)
    forged.elements[0].text = "A caller-selected replacement"
    with pytest.raises(ValueError, match="differs from its source"):
        derive_target_bindings(forged, source_snapshot=observed)
    packet = derive_target_bindings(observed)
    accepted = validate_target_bindings(packet, observation_id="current")
    accepted["candidates"].clear()
    assert packet["candidates"]
