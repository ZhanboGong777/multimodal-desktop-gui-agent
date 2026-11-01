"""Independent finite-generation tests with saved synthetic frames and offline SDK.

Runtime remains wider than this generation grammar. No model, OCR, GUI capture
or OS input is called, and canned replies never establish live task success.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image
from test_runtime_runner import FakeExecutor, FakeObserver, _runner
from test_week6_action_output_contract_review import _sdk
from test_week6_automatic_decomposition import _frame
from test_week6_target_bindings_review import _element, _native, _run_offline, _snapshot

from gui_agent.planning.planner import TaskPlanner, _task_plan_response_format
from gui_agent.planning.target_bindings import derive_target_bindings, observed_query_contract
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.observation import describe_elements
from gui_agent.runtime.schemas import ExecutionOptions, TaskSpec


@pytest.fixture
def auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/week6_audit_semantic_acceptance.py"
    spec = importlib.util.spec_from_file_location("observed_queries_independent_auditor", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _scene(labels):
    frame = _frame("observed-query", [])
    frame.elements = [_element(f"q-{index}", text, box) for index, (text, box) in enumerate(labels)]
    return frame


def _schema(frame):
    return _task_plan_response_format(1, target_bindings=derive_target_bindings(frame))["json_schema"]["schema"]


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


def test_review_contract_is_rebuilt_independently_from_scoped_source_order_and_raw_literal_labels(auditor):
    frame = _snapshot()
    frame.elements[0].text = "  Choice A  "
    packet = derive_target_bindings(frame)
    produced = observed_query_contract(packet)
    assert produced == auditor.independent_observed_query_contract(packet)
    assert auditor.audit_observed_query_contract(packet, produced) == produced
    assert produced["full_label_queries"][0] == "  Choice A  "
    assert produced["limits"] == {"tokens_per_label": 32, "tokens_per_span": 4, "elements_per_run": 4,
        "extra_query_characters": 160, "extra_queries": 128, "search_steps": 4096}
    assert all(set(item) == {"query", "source_kind", "source_element_ids"} for item in produced["extra_queries"])
    assert "Choice" in {item["query"] for item in produced["extra_queries"]}
    assert all(item["query"] not in produced["full_label_queries"] for item in produced["extra_queries"])


@pytest.mark.parametrize("text,arguments", [
    ("Future missing control", {}),
    ("Future missing control", {"element_id": ""}),
    (None, {"target": "Future missing control"}),
    ("Choice A", {"target": "Future missing control"}),
    ("Future missing control", {"target": "Choice A"}),
    ("Future missing control", {"element_id": "", "target_element": "label-a"}),
    ("Future missing control", {"element_id": "label-a", "target_element": ""}),
])
def test_review_standard_validator_rejects_unobserved_no_id_queries_and_secondary_alias_bypasses(text, arguments):
    assert not Draft202012Validator(_schema(_snapshot())).is_valid(_native(text=text, arguments=arguments))


@pytest.mark.parametrize("text,arguments,action", [
    ("Choice A", {}, "click"),
    ("Choice", {}, "double_click"),
    (None, {"target": "Choice"}, "right_click"),
    ("Choice A", {"element_id": "duplicate-a", "target": "Choice A"}, "click"),
    (None, {"element_id": "heading"}, "click"),
    ("Visual glyph absent from OCR", {"element_id": "unknown-box"}, "click"),
    (None, {"target": "Visual glyph absent from OCR", "target_element": "unknown-box"}, "click"),
    (None, {"text": "The requested text", "vendor_extension": 7}, "type_text"),
    (None, {"duration": .1}, "wait"),
])
def test_review_observed_literals_bounded_tokens_aliases_id_only_visual_and_nonpoint_paths_remain(text, arguments, action):
    assert Draft202012Validator(_schema(_snapshot())).is_valid(_native(text=text, arguments=arguments, action=action))


def test_review_generation_restriction_is_not_falsely_presented_as_full_runtime_substring_equivalence():
    frame = _scene([("Choice A", (10, 10, 100, 30))])
    raw = _native(text="hoice", arguments={})
    assert not Draft202012Validator(_schema(frame)).is_valid(raw)
    from gui_agent.planning.schemas import PlanStep

    resolved = ActionAdapter(platform="darwin").resolve(PlanStep.model_validate(raw["steps"][0]), frame)
    assert resolved.element_id == "q-0"  # Pure resolution, no execution.


@pytest.mark.parametrize("kind", ["reading", "spatial", "far"])
def test_review_only_observed_geometric_runs_are_added_and_their_original_witness_ids_are_bound(auditor, kind):
    if kind == "reading":
        frame = _scene([("North", (10, 10, 50, 30)), ("gate", (60, 10, 100, 30)), ("button", (110, 10, 160, 30))])
        wanted, origin, ids = "North gate button", "reading_order_run", ["q-0", "q-1", "q-2"]
    elif kind == "spatial":
        frame = _scene([("North", (200, 10, 240, 30)), ("Foreign", (10, 30, 80, 50)), ("gate", (200, 40, 240, 60))])
        wanted, origin, ids = "North gate", "spatial_run", ["q-0", "q-2"]
    else:
        frame = _scene([("North", (10, 10, 50, 30)), ("gate", (500, 10, 550, 30))])
        wanted, origin, ids = "North gate", None, []
    packet = derive_target_bindings(frame)
    actual = observed_query_contract(packet)
    assert actual == auditor.independent_observed_query_contract(packet)
    matching = [item for item in actual["extra_queries"] if item["query"] == wanted]
    if origin:
        assert matching == [{"query": wanted, "source_kind": origin, "source_element_ids": ids}]
        assert Draft202012Validator(_schema(frame)).is_valid(_native(text=wanted, arguments={}))
    else:
        assert matching == []
        assert not Draft202012Validator(_schema(frame)).is_valid(_native(text=wanted, arguments={}))


@pytest.mark.parametrize("boundary", ["tokens", "characters", "count", "search"])
def test_review_limits_and_unknown_omission_lower_bounds_are_explicit_and_independently_recomputed(auditor, boundary):
    if boundary == "tokens":
        labels = [(" ".join(f"Token{index:02d}" for index in range(33)), (10, 10, 100, 30))]
    elif boundary == "characters":
        labels = [(" ".join(letter * 60 for letter in "ABCD"), (10, 10, 100, 30))]
    elif boundary == "count":
        labels = [(f"Term{index} Tail{index}", (index * 100, 10, index * 100 + 40, 30)) for index in range(100)]
    else:
        labels = [("Same", (index * 100, 10, index * 100 + 40, 30)) for index in range(150)]
    packet = derive_target_bindings(_scene(labels))
    actual = observed_query_contract(packet)
    assert actual == auditor.independent_observed_query_contract(packet)
    assert actual["search"]["truncated"] is True
    assert len(actual["extra_queries"]) <= 128 and actual["search"]["steps"] <= 4096
    assert all(len(item["query"]) <= 160 for item in actual["extra_queries"])
    if boundary == "tokens":
        assert actual["search"]["labels_with_token_truncation"] == 1
        assert "Token32" not in {item["query"] for item in actual["extra_queries"]}
        assert len(actual["full_label_queries"][0]) > 160
        assert actual["search"]["omitted_queries_at_least"] == 0
    elif boundary == "characters":
        assert actual["search"]["overlength_queries"] > 0 and len(actual["full_label_queries"][0]) > 160
    elif boundary == "count":
        assert len(actual["extra_queries"]) == 128 and actual["search"]["count_limit_reached"] is True
        assert actual["search"]["omitted_queries_at_least"] == 1
    else:
        assert actual["search"]["search_limit_reached"] is True and actual["search"]["steps"] == 4096
        assert actual["search"]["omitted_queries_at_least"] == 0


def test_review_message_final_subset_cannot_borrow_excluded_text_or_use_its_ocr_to_build_new_queries(auditor):
    source = _snapshot()
    subset = source.model_copy(update={"elements": [source.elements[3], source.elements[5]]})
    packet = derive_target_bindings(subset, source_snapshot=source)
    context = {"observation_id": source.observation_id, "visible_text": describe_elements(subset), "target_bindings": packet}
    rebuilt = auditor.audit_target_bindings(context, source, authorised_candidate_ids=["single-box", "unknown-box"])
    queries = observed_query_contract(rebuilt)
    assert queries == auditor.independent_observed_query_contract(rebuilt)
    assert queries["full_label_queries"] == queries["extra_queries"] == []
    assert packet["candidates"][0]["external_ocr_overlap"] is True
    assert packet["anonymous_without_ocr_ids"] == ["unknown-box"]
    assert "Choice A" not in json.dumps(queries) and "label-a" not in json.dumps(queries)


@pytest.mark.parametrize("tamper", ["query", "source", "order", "limits", "truncated", "boolean_count"])
def test_review_saved_query_metadata_cannot_be_edited_or_promoted_to_trusted_caller_data(auditor, tamper):
    packet = derive_target_bindings(_snapshot())
    saved = observed_query_contract(packet)
    if tamper == "query":
        saved["extra_queries"][0]["query"] = "Caller invented future control"
    elif tamper == "source":
        saved["extra_queries"][0]["source_element_ids"] = ["outside-scope-id"]
    elif tamper == "order":
        saved["extra_queries"].reverse()
    elif tamper == "limits":
        saved["limits"]["extra_queries"] = 129
    elif tamper == "truncated":
        saved["search"]["truncated"] = not saved["search"]["truncated"]
    else:
        saved["search"]["labels_with_token_truncation"] = False
    with pytest.raises(auditor.AuditError, match="observed_queries"):
        auditor.audit_observed_query_contract(packet, saved)


def test_review_actual_sdk_wire_schema_native_and_each_v2_attempt_contract_are_exactly_bound(auditor, tmp_path):
    replies = ["Malformed native canary", json.dumps(_native())]
    frame, call, records, bodies, result, budgeted = _run_offline(tmp_path, replies)
    assert result.ok and budgeted.request_count == len(bodies) == 2
    bindings = auditor.audit_target_bindings(call["context"], frame)
    expected = _task_plan_response_format(1, target_bindings=bindings)
    rebuilt = auditor.independent_observed_query_contract(bindings)
    audited = auditor.audit_action_attempts(call, frame, records, expected)
    assert len(audited) == 2 and all(body["response_format"] == expected for body in bodies)
    for row in records:
        contract = row["target_binding_contract"]
        assert contract["schema_version"] == "gui_agent.target_binding_schema.v2"
        assert contract["observed_query_contract"] == rebuilt
        assert contract["observed_query_contract_sha256"] == _digest(rebuilt)
        assert contract["response_format_sha256"] == _digest(expected)
    assert [row["response"]["content"] for row in records] == replies
    # Rewriting the schema and rehashing the saved contract cannot change the
    # independently reconstructed schema sent for these original source pixels.
    records[1]["request"]["response_format"]["json_schema"]["schema"]["properties"]["steps"]["maxItems"] = 2
    records[1]["target_binding_contract"]["response_format_sha256"] = _digest(records[1]["request"]["response_format"])
    with pytest.raises(auditor.AuditError, match="action_target_binding_contract|action_attempt_request"):
        auditor.audit_action_attempts(call, frame, records, expected)


@pytest.mark.parametrize("tamper", ["drop", "version", "hash", "metadata"])
def test_review_v2_live_attempts_cannot_silently_drop_or_replace_the_query_contract(auditor, tmp_path, tamper):
    frame, call, records, _, _, _ = _run_offline(tmp_path, [json.dumps(_native())])
    bindings = auditor.audit_target_bindings(call["context"], frame)
    expected = _task_plan_response_format(1, target_bindings=bindings)
    contract = records[0]["target_binding_contract"]
    if tamper == "drop":
        contract.pop("observed_query_contract")
        contract.pop("observed_query_contract_sha256")
    elif tamper == "version":
        contract["schema_version"] = "gui_agent.target_binding_schema.v1"
    elif tamper == "hash":
        contract["observed_query_contract_sha256"] = "0" * 64
    else:
        contract["observed_query_contract"]["full_label_queries"].append("Caller future label")
        contract["observed_query_contract_sha256"] = _digest(contract["observed_query_contract"])
    with pytest.raises(auditor.AuditError, match="action_target_binding_contract"):
        auditor.audit_action_attempts(call, frame, records, expected)


def test_review_provider_ignoring_schema_retains_unknown_native_but_runtime_refuses_without_dispatch_or_local_id_repair(auditor, tmp_path):
    source = _snapshot()
    fresh = copy.deepcopy(source)
    fresh.observation_id = "fresh-query-frame"
    for index, frame in enumerate((source, fresh)):
        frame.image_path = str(tmp_path / f"frame-{index}.png")
        Image.new("RGB", (1470, 956), "white").save(frame.image_path)
    raw = json.dumps(_native(text="Unobserved future control", arguments={"element_id": ""}))
    bodies = []
    client, sdk = _sdk([raw], bodies)
    executor, observer = FakeExecutor(), FakeObserver([source, fresh])
    observer.foreground_matches = lambda frame: frame.foreground_stable
    runner, recorder = _runner(tmp_path, observer, TaskPlanner(client, max_steps=1, max_format_retries=0), executor)
    try:
        result = runner.run(TaskSpec(case_id="query-review", instruction="Choose the requested local control",
            expect_text=["Caller complete"]), ExecutionOptions(execute=True, confirm=False, max_planning_attempts=1,
                verification_timeout_seconds=0))
    finally:
        sdk.close()
    assert result.status != "succeeded" and result.action_count == result.unconfirmed_action_attempts == 0
    assert not executor.actions and len(bodies) == result.planning_calls == 1
    assert not Draft202012Validator(bodies[0]["response_format"]["json_schema"]["schema"]).is_valid(json.loads(raw))
    assert result.steps[0].action_result is None and result.steps[0].error.startswith("no element matches ")
    assert recorder.read_steps()[0]["resolved"] is None
    # The unchanged parser still accepts the native plan structure. The auditor
    # rejects its generation qualification; neither step proves an OS success.
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    frame, call, records, _, parsed, _ = _run_offline(qualification, [raw])
    assert parsed.ok and call["response"]["content"] == raw and call["parsed_plan"]["steps"][0]["arguments"] == {"element_id": ""}
    expected = _task_plan_response_format(1, target_bindings=auditor.audit_target_bindings(call["context"], frame))
    with pytest.raises(auditor.AuditError, match="action_native_violates_predeclared_generation_schema"):
        auditor.audit_action_attempts(call, frame, records, expected)
