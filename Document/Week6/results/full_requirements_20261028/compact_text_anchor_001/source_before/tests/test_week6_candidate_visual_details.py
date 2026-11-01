"""Pixel provenance, current scope, native output and no-input counterexamples."""
from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from PIL import Image
from test_runtime_runner import FakeExecutor, FakeObserver, _runner
from test_week6_action_output_contract_review import _sdk
from test_week6_rolling_actions import options
from test_week6_target_bindings import candidate, element, frame
from test_week6_target_bindings_review import auditor

from gui_agent.models.openai_compatible import OpenAICompatibleClient
from gui_agent.planning.planner import TaskPlanner, _task_plan_response_format
from gui_agent.planning.target_bindings import candidate_visual_details, derive_target_bindings
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.observation import describe_elements
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import TaskSpec

detail_auditor = auditor


def scene(tmp_path):
    observed = frame(element("label", "Keep", box=(10, 10, 110, 40)),
                     element("other", "Other", box=(10, 70, 110, 100)),
                     element("visual", source="contour", box=(200, 50, 360, 100)))
    observed.image_path = str(tmp_path / "original.png")
    image = Image.new("RGB", (480, 320), "#00aaff")
    image.paste("#f029af", (200, 50, 360, 100))
    image.save(observed.image_path)
    packet = derive_target_bindings(observed)
    details, crops = candidate_visual_details(packet, observed.image_path)
    context = {"observation_id": observed.observation_id, "target_bindings": packet,
               "visible_text": describe_elements(observed),
               "candidate_visual_details": details}
    return observed, context, crops


def test_details_are_original_pixels_with_current_ids_and_not_selected_by_task_or_labels(tmp_path):
    observed, context, crops = scene(tmp_path)
    manifest = context["candidate_visual_details"]
    assert len(crops) == len(manifest["regions"]) == 1
    assert manifest["regions"][0]["element_id"] == "visual"
    with Image.open(observed.image_path) as image:
        expected = image.crop((200, 50, 360, 100))
        actual = Image.open(BytesIO(crops[0]))
        assert actual.size == expected.size and actual.tobytes() == expected.tobytes()
    renamed = observed.model_copy(deep=True)
    renamed.elements[0].text = "A wholly unrelated caller phrase"
    other, other_crops = candidate_visual_details(derive_target_bindings(renamed), observed.image_path)
    assert manifest == other and crops == other_crops


def test_region_limits_and_external_ocr_overlap_cannot_expand_visible_scope(tmp_path):
    observed, _, _ = scene(tmp_path)
    observed.elements = [element(f"r{i}", source="contour", box=(200, 50, 360, 100)) for i in range(12)]
    packet = derive_target_bindings(observed)
    manifest, crops = candidate_visual_details(packet, observed.image_path)
    assert len(crops) == 8 and manifest["omitted_region_count"] == 4
    source = observed.model_copy(deep=True)
    source.elements.append(element("outside-ocr", "Excluded private label", box=(210, 60, 300, 80)))
    subset = derive_target_bindings(observed, source_snapshot=source)
    manifest, crops = candidate_visual_details(subset, observed.image_path)
    assert not crops and not manifest["regions"]
    assert "Excluded private label" not in json.dumps(manifest)


def test_pixel_cap_and_invalid_bounds_are_not_silently_clipped(tmp_path):
    observed, _, _ = scene(tmp_path)
    Image.new("RGB", (1600, 1000), "white").save(observed.image_path)
    observed.elements = [element("huge", source="contour", box=(0, 0, 1600, 1000))]
    manifest, crops = candidate_visual_details(derive_target_bindings(observed), observed.image_path)
    assert not crops and manifest["omitted_region_count"] == 1
    observed.elements[0] = element("outside", source="contour", box=(1550, 50, 1750, 100))
    with pytest.raises(ValueError, match="outside"):
        candidate_visual_details(derive_target_bindings(observed), observed.image_path)


@pytest.mark.parametrize("tamper", ["source_hash", "crop_hash", "id", "box", "scope", "disabled"])
def test_forged_or_disabled_details_are_rejected_before_any_http(tmp_path, tamper):
    observed, context, _ = scene(tmp_path)
    details = context["candidate_visual_details"]
    if tamper == "source_hash":
        details["source_image_sha256"] = "0" * 64
    elif tamper == "crop_hash":
        details["regions"][0]["png_sha256"] = "0" * 64
    elif tamper == "id":
        details["regions"][0]["element_id"] = "foreign"
    elif tamper == "box":
        details["regions"][0]["original_box"]["left"] += 1
    elif tamper == "scope":
        context["target_bindings"]["candidates"].pop()
        context["target_bindings"]["anonymous_without_ocr_ids"].clear()
    bodies = []
    client, transport = _sdk([json.dumps(candidate(None, {"element_id": "visual"}))], bodies)
    try:
        result = TaskPlanner(client, max_steps=1, visual_candidate_details=tamper != "disabled").plan(
            "Choose the requested control", image_path=observed.image_path, context=context)
    finally:
        transport.close()
    assert not result.ok and result.attempts == len(bodies) == 0
    assert "invalid candidate_visual_details" in result.error


@pytest.mark.parametrize("alias", ["target_element", "target"])
def test_compact_schema_requires_current_canonical_id_and_cannot_hide_a_secondary_target(tmp_path, alias):
    _, context, _ = scene(tmp_path)
    schema = _task_plan_response_format(1, target_bindings=context["target_bindings"], current_id_only=True)
    check = Draft202012Validator(schema["json_schema"]["schema"])
    assert check.is_valid(candidate(None, {"element_id": "visual"}))
    assert not check.is_valid(candidate(None, {"element_id": "stale"}))
    assert not check.is_valid(candidate("Keep"))
    assert not check.is_valid(candidate(None, {"element_id": "visual", alias: "other"}))
    assert check.is_valid(candidate(None, {"text": "Local content"}, action="type_text"))


def test_sdk_keeps_full_image_first_and_sends_exact_crops_in_one_original_request(tmp_path, detail_auditor):
    auditor = detail_auditor
    observed, context, crops = scene(tmp_path)
    bodies = []
    native = candidate(None, {"element_id": "visual"})
    client, transport = _sdk([json.dumps(native)], bodies)
    try:
        result = TaskPlanner(client, max_steps=1, visual_candidate_details=True).plan(
            "Choose by appearance", image_path=observed.image_path, context=context)
    finally:
        transport.close()
    assert result.ok and result.attempts == len(bodies) == 1
    assert result.plan.steps[0].arguments == {"element_id": "visual"}
    parts = bodies[0]["messages"][-1]["content"]
    images = [base64.b64decode(p["image_url"]["url"].split(",", 1)[1])
              for p in parts if p["type"] == "image_url"]
    assert hashlib.sha256(images[0]).hexdigest() == context["candidate_visual_details"]["source_image_sha256"]
    assert images[1:] == crops
    assert "original bounding_box" in parts[2]["text"]
    assert auditor.audit_visual_details(context, context["target_bindings"], observed)
    forged = deepcopy(context)
    forged["candidate_visual_details"]["regions"][0]["png_sha256"] = "0" * 64
    with pytest.raises(auditor.AuditError, match="detail_manifest"):
        auditor.audit_visual_details(forged, forged["target_bindings"], observed)
    contract = result.metadata["action_output_contract"]
    assert contract["target_binding_contract"]["schema_version"].endswith("v3.current_id_details")
    call = {"instruction": "Choose by appearance", "image_path": observed.image_path, "context": context,
            "attempts": 1, "model_requests": 1, "task_id": "task-1", "ok": True,
            "parsed_plan": result.plan.as_dict(),
            "response": {**result.response.as_dict(), "content": result.response.content}}
    records = [{"schema_version": contract["schema_version"], **r} for r in contract["attempts"]]
    assert len(auditor.audit_action_attempts(call, observed, records, bodies[0]["response_format"])) == 1


def test_runner_enabled_context_and_runtime_still_refuse_wrong_readable_id_without_input(tmp_path):
    observed, _, _ = scene(tmp_path)
    bodies = []
    raw = candidate("Keep", {"element_id": "other"})
    client, transport = _sdk([json.dumps(raw)], bodies)
    executor = FakeExecutor()
    planner = TaskPlanner(client, max_steps=1, max_format_retries=0, visual_candidate_details=True)
    observer = FakeObserver([observed, observed])
    observer.foreground_matches = lambda _: True
    runner, _ = _runner(tmp_path, observer, planner, executor)
    try:
        result = runner.run(TaskSpec(case_id="details", instruction="Keep the current item",
                                    expect_text=["Unreached final"]), options(max_planning_attempts=1))
    finally:
        transport.close()
    assert len(bodies) == 1 and not executor.actions
    assert result.status != "succeeded" and result.action_count == 0
    assert "target consistency conflict:" in result.steps[0].error


def test_disabled_runner_does_not_add_details_and_old_transport_remains_one_image(tmp_path):
    observed, _, _ = scene(tmp_path)
    fake = SimpleNamespace(adapter=ActionAdapter(), planner=SimpleNamespace(visual_candidate_details=False))
    context = TaskRunner._context(fake, observed, TaskSpec(case_id="old", instruction="Choose locally"), options())
    assert "candidate_visual_details" not in context
    messages = [{"role": "user", "content": json.dumps({"context": context})}]
    payload = OpenAICompatibleClient.to_vision_messages(messages, observed.image_path)
    assert len([p for p in payload[0]["content"] if p["type"] == "image_url"]) == 1
