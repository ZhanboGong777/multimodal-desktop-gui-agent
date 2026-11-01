"""Finite generation queries are a strict subset of unchanged runtime grounding."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from itertools import pairwise

import pytest
from jsonschema import Draft202012Validator
from test_week6_action_output_contract import _PIXELS, native, sdk, user
from test_week6_automatic_decomposition import ScriptedClient
from test_week6_target_bindings import candidate, element, frame, validator
from test_week6_visual_target_guidance import native as visual_native
from test_week6_visual_target_guidance import runtime

from gui_agent.perception.grounding import find_text
from gui_agent.planning.planner import TaskPlanner, _task_plan_response_format
from gui_agent.planning.target_bindings import (
    OBSERVED_QUERY_CONTRACT_VERSION,
    TARGET_BINDING_SCHEMA_VERSION,
    derive_target_bindings,
    observed_query_contract,
)
from gui_agent.runtime.action_adapter import _is_next_in_label, _match_adjacent_runs
from gui_agent.runtime.schemas import ExecutionOptions


def queries(contract):
    return [row["query"] for row in contract["extra_queries"]]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf8")).hexdigest()


def test_contract_is_deterministic_copy_isolated_and_preserves_raw_case_full_labels_and_sources():
    observed = frame(element("one", " Keep-These Words! "),
                     element("two", "Words!", box=(200, 10, 300, 40)))
    packet = derive_target_bindings(observed)
    before = deepcopy(packet)
    contract = observed_query_contract(packet)
    assert contract["schema_version"] == OBSERVED_QUERY_CONTRACT_VERSION
    assert contract["full_label_queries"] == [" Keep-These Words! ", "Words!"]
    assert contract["extra_queries"][:3] == [
        {"query": "Keep", "source_kind": "label_token_span", "source_element_ids": ["one"]},
        {"query": "Keep-These", "source_kind": "label_token_span", "source_element_ids": ["one"]},
        {"query": "Keep-These Words", "source_kind": "label_token_span", "source_element_ids": ["one"]},
    ]
    assert contract == observed_query_contract(packet) and packet == before
    assert len(queries(contract)) == len(set(queries(contract)))
    assert not set(queries(contract)).intersection(contract["full_label_queries"])
    contract["extra_queries"][0]["source_element_ids"].clear()
    contract["limits"]["extra_queries"] = -1
    assert observed_query_contract(packet)["limits"]["extra_queries"] == 128
    assert observed_query_contract(packet)["extra_queries"][0]["source_element_ids"] == ["one"]


@pytest.mark.parametrize("alias", ["target_text", "target"])
def test_observed_full_label_token_and_canonical_run_queries_accept_only_native_no_id_alias(alias):
    observed = frame(element("one", "Open account", box=(10, 10, 130, 40)),
                     element("two", "settings", box=(135, 10, 220, 40)))
    packet = derive_target_bindings(observed)
    check = validator(packet)
    for query in ("Open account", "account", "Open account settings"):
        value = candidate(query if alias == "target_text" else None,
                          {} if alias == "target_text" else {"target": query})
        assert check.is_valid(value)
    bad = candidate("Future status" if alias == "target_text" else None,
                    {} if alias == "target_text" else {"target": "Future status"})
    assert not check.is_valid(bad)


@pytest.mark.parametrize("args", [
    {}, {"element_id": ""}, {"target_element": ""}, {"element_id": None, "target_element": ""},
    {"target": "Unobserved wording"}, {"element_id": "", "target_element": "", "target": "Unobserved wording"},
])
def test_unknown_text_cannot_escape_through_empty_ids_or_secondary_query_alias(args):
    packet = derive_target_bindings(frame(element("label", "Known label")))
    assert not validator(packet).is_valid(candidate("Unobserved wording", args))
    assert not validator(packet).is_valid(candidate(None, {**args, "target": "Unobserved wording"}))


def test_unchanged_runtime_substring_can_be_valid_while_finite_generation_rejects_it():
    observed = frame(element("label", "Configuration"))
    assert find_text(observed.elements, "figur", min_confidence=.35).found
    packet = derive_target_bindings(observed)
    assert "figur" not in queries(observed_query_contract(packet))
    assert not validator(packet).is_valid(candidate("figur"))
    assert validator(packet).is_valid(candidate("Configuration"))


def test_no_ocr_queries_keep_native_visual_id_id_only_nonpoint_and_empty_plan_paths():
    packet = derive_target_bindings(frame(element("icon", source="contour")))
    check = validator(packet)
    contract = observed_query_contract(packet)
    assert not contract["full_label_queries"] and not contract["extra_queries"]
    assert check.is_valid(candidate("Visible unlabelled shape", {"element_id": "icon"}))
    assert check.is_valid(candidate(None, {"target_element": "icon"}))
    assert check.is_valid(candidate("arbitrary content", {"text": "unchanged"}, action="type_text"))
    assert check.is_valid(candidate(None, {"seconds": .1}, action="wait"))
    assert not check.is_valid(candidate("Visible unlabelled shape"))
    assert not check.is_valid(candidate(None))
    empty = json.loads(native())
    empty["steps"] = []
    empty["errors"] = ["No certain current target"]
    assert check.is_valid(empty)


def test_full_labels_are_not_truncated_by_extra_query_character_limit():
    label = "X" * 180 + " tail"
    packet = derive_target_bindings(frame(element("label", label)))
    contract = observed_query_contract(packet)
    assert contract["full_label_queries"] == [label]
    assert validator(packet).is_valid(candidate(label))
    assert all(len(query) <= 160 for query in queries(contract))
    assert contract["search"]["overlength_queries"] > 0
    assert contract["search"]["truncated"] and contract["search"]["omitted_queries_at_least"] == 0


def test_token_truncation_does_not_claim_an_omitted_unique_query_or_drop_full_label():
    label = " ".join(["Repeated"] * 40)
    contract = observed_query_contract(derive_target_bindings(frame(element("many", label))))
    assert contract["full_label_queries"] == [label]
    assert queries(contract) == [" ".join(["Repeated"] * size) for size in range(1, 5)]
    assert contract["search"]["labels_with_token_truncation"] == 1
    assert contract["search"]["truncated"]
    assert not contract["search"]["count_limit_reached"]
    assert contract["search"]["omitted_queries_at_least"] == 0


def test_extra_count_limit_stops_at_first_new_qualified_query_beyond_128():
    observed = frame(element("first", " ".join(f"Word{i}" for i in range(32))),
                     element("second", " ".join(f"Next{i}" for i in range(12)), box=(500, 500, 900, 530)))
    contract = observed_query_contract(derive_target_bindings(observed))
    assert len(contract["extra_queries"]) == 128
    assert contract["search"]["count_limit_reached"] and contract["search"]["omitted_queries_at_least"] == 1
    assert not contract["search"]["search_limit_reached"]


def test_search_limit_is_exact_and_does_not_invent_unseen_unique_omissions():
    observed = frame(*(element(f"label{i}", "Repeated", box=(i * 100, i * 100, i * 100 + 60, i * 100 + 25))
                       for i in range(100)))
    contract = observed_query_contract(derive_target_bindings(observed))
    assert contract["search"]["steps"] == 4096
    assert contract["search"]["search_limit_reached"] and contract["search"]["truncated"]
    assert not contract["extra_queries"] and contract["search"]["omitted_queries_at_least"] == 0
    assert contract["search"]["candidate_queries_considered"] < contract["search"]["steps"]


def test_canonical_runs_are_at_most_four_elements_and_each_pair_has_geometry():
    observed = frame(*(element(f"part{i}", f"Part{i}", box=(i * 55, 10, i * 55 + 50, 40)) for i in range(5)))
    contract = observed_query_contract(derive_target_bindings(observed))
    records = {item.element_id: item for item in observed.elements}
    assert "Part0 Part1 Part2 Part3" in queries(contract)
    assert "Part0 Part1 Part2 Part3 Part4" not in queries(contract)
    for row in contract["extra_queries"]:
        witnesses = [records[identifier] for identifier in row["source_element_ids"]]
        assert len(witnesses) <= 4
        if len(witnesses) > 1:
            assert all(_is_next_in_label(a, b) for a, b in pairwise(witnesses))
        assert (find_text(witnesses, row["query"], min_confidence=.35).found
                or _match_adjacent_runs(witnesses, row["query"], .35))


def test_reading_order_alone_cannot_join_unrelated_labels_but_spatial_path_can_skip_interleaving():
    far = frame(element("first", "Open", box=(10, 10, 60, 35)),
                element("last", "settings", box=(600, 400, 690, 430)))
    # Existing runtime is deliberately broader than the generation subset.
    assert _match_adjacent_runs(far.elements, "Open settings", .35)
    assert "Open settings" not in queries(observed_query_contract(derive_target_bindings(far)))
    near = frame(element("first", "Open", box=(10, 10, 80, 35)),
                 element("middle", "Unrelated", box=(600, 25, 690, 50)),
                 element("last", "settings", box=(10, 40, 100, 65)))
    rows = observed_query_contract(derive_target_bindings(near))["extra_queries"]
    assert {"query": "Open settings", "source_kind": "spatial_run",
            "source_element_ids": ["first", "last"]} in rows


def test_unresolvable_tokenless_join_is_recorded_without_generation_permission():
    observed = frame(element("first", "保存", box=(10, 10, 60, 40)),
                     element("second", "设置", box=(65, 10, 115, 40)))
    packet = derive_target_bindings(observed)
    contract = observed_query_contract(packet)
    assert contract["full_label_queries"] == ["保存", "设置"]
    assert "保存 设置" not in queries(contract) and contract["search"]["unresolvable_queries"] > 0
    assert not validator(packet).is_valid(candidate("保存 设置"))


def test_scope_low_confidence_and_non_ocr_text_do_not_supply_extra_queries():
    outside = element("outside", "OUTSIDE PRIVATE", box=(300, 10, 440, 40))
    source = frame(element("inside", "Allowed phrase"), outside,
                   element("weak", "LOW CONFIDENCE", confidence=.34),
                   element("other", "Non OCR phrase", source="detector"))
    subset = source.model_copy(update={"elements": source.elements[:1]})
    contract = observed_query_contract(derive_target_bindings(subset, source_snapshot=source))
    assert "PRIVATE" not in json.dumps(contract) and all(row["source_element_ids"] == ["inside"]
                                                        for row in contract["extra_queries"])
    full = observed_query_contract(derive_target_bindings(source))
    assert "LOW CONFIDENCE" not in full["full_label_queries"]
    assert "Non OCR phrase" in full["full_label_queries"]  # Existing full-label policy is retained.
    assert not any("weak" in row["source_element_ids"] or "other" in row["source_element_ids"]
                   for row in full["extra_queries"])


def test_actual_sdk_wire_and_every_attempt_bind_canonical_contract_without_native_rewriting(tmp_path):
    packet = derive_target_bindings(frame(element("label", "Open account settings")))
    value = candidate("account settings")
    bodies = []
    client, transport = sdk([json.dumps(value)], bodies)
    image = tmp_path / "actual.png"
    image.write_bytes(_PIXELS)
    context = {"observation_id": "current", "target_bindings": packet, "custom": {"keep": True}}
    original = deepcopy(context)
    try:
        result = TaskPlanner(client, max_steps=1).plan("Use the selected local setting", context=context,
                                                      image_path=str(image))
    finally:
        transport.close()
    assert result.ok and result.attempts == len(bodies) == 1 and context == original
    expected = observed_query_contract(packet)
    evidence = result.metadata["action_output_contract"]
    binding = evidence["target_binding_contract"]
    assert binding["schema_version"] == TARGET_BINDING_SCHEMA_VERSION
    assert binding["observed_query_contract"] == expected
    assert binding["observed_query_contract_sha256"] == digest(expected)
    assert evidence["attempts"][0]["target_binding_contract"] == binding
    assert evidence["attempts"][0]["response"]["content"] == json.dumps(value)
    assert result.plan.steps[0].target_text == "account settings"
    response_format = _task_plan_response_format(1, target_bindings=packet)
    assert bodies[0]["response_format"] == response_format
    assert binding["response_format_sha256"] == digest(response_format)
    assert Draft202012Validator(response_format["json_schema"]["schema"]).is_valid(value)
    assert user(bodies[0])["context"]["target_bindings"] == packet
    assert user(bodies[0])["image_path"] == str(image)


def test_legacy_and_explicit_unstructured_mode_do_not_gain_hidden_schema_or_extra_requests():
    client = ScriptedClient([native(), native()])
    legacy = TaskPlanner(client).plan("Continue locally")
    assert legacy.ok and "target_binding_contract" not in legacy.metadata["action_output_contract"]
    packet = derive_target_bindings(frame(element("label", "Known phrase")))
    result = TaskPlanner(client, require_structured_output=False).plan("Continue locally", context={
        "observation_id": "current", "target_bindings": packet})
    assert result.ok and len(client.calls) == 2 and "response_format" not in client.calls[1]["kwargs"]
    evidence = result.metadata["action_output_contract"]["target_binding_contract"]
    assert evidence["observed_query_contract"] == observed_query_contract(packet)
    assert evidence["response_format_sha256"] is None


def test_provider_ignoring_new_schema_is_not_locally_repaired_and_runtime_still_stops_without_input(tmp_path):
    rejected = visual_native()
    runner, task, executor, bodies, budgeted, transport, _ = runtime(tmp_path, [rejected], max_requests=1)
    try:
        result = runner.run(task, ExecutionOptions(execute=True, confirm=False, max_actions=1,
            max_planning_attempts=1, verification_timeout_seconds=0))
    finally:
        transport.close()
    assert len(bodies) == budgeted.request_count == 1
    check = Draft202012Validator(bodies[0]["response_format"]["json_schema"]["schema"])
    assert not check.is_valid(json.loads(rejected))
    assert result.status != "succeeded" and result.action_count == len(executor.actions) == 0
    assert result.steps[0].action_result is None and result.steps[0].error.startswith("no element matches ")
    records = [json.loads(line) for line in (tmp_path / "action_planning_attempts.jsonl").read_text("utf8").splitlines()]
    assert records[0]["response"]["content"] == rejected
    assert records[0]["parsed_plan"]["steps"][0]["arguments"] == {"element_id": ""}
