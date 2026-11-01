"""Read-only audit of an ended, explicitly selected semantic live run.

No model, OCR, capture, executor or process-control operation is performed. Saved
literal predicates are recomputed with the pinned ordinary Verifier. A runtime
or acceptance success boolean by itself is never sufficient evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gui_agent.models.base import ModelRequest
from gui_agent.planning.parser import PlanParseError, parse_plan
from gui_agent.planning.planner import (
    _FORMAT_REPAIR_INSTRUCTION,
    _INLINE_FORMAT_ERROR_CHARACTERS,
    _INLINE_FORMAT_RESPONSE_CHARACTERS,
    ACTION_OUTPUT_CONTRACT_VERSION,
    _task_plan_response_format,
)
from gui_agent.planning.prompts import SYSTEM_PROMPT
from gui_agent.planning.schemas import PlanStep
from gui_agent.planning.semantic_workflow import (
    _INLINE_CORRECTION_MAX_CHARACTERS,
    SEMANTIC_CHECK_MAX_CHARACTERS,
    SEMANTIC_CHECKS_PER_LIST,
    SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR,
    SEMANTIC_GENERIC_REPAIR,
    SEMANTIC_INPUT_MODALITY,
    SEMANTIC_INSTRUCTION_MAX_CHARACTERS,
    SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR,
    SEMANTIC_PROTOCOL_VERSION,
    SEMANTIC_REASON_MAX_CHARACTERS,
    SEMANTIC_REPAIR_POLICY_VERSION,
    SEMANTIC_STEP_POLICY,
    SEMANTIC_SYSTEM_PROMPT,
    SemanticDecomposer,
    SemanticPlan,
    SemanticWorkflowResult,
    _DuplicateSemanticPostconditionsError,
    _final_task,
    _goal_task,
    _initial_guard,
    _PreexistingSemanticPostconditionsError,
    _semantic_response_format,
    _unique_keys,
    _validate_native_predicates,
)
from gui_agent.planning.workflow_decomposer import WorkflowRequest, _rolling_frame_guard
from gui_agent.runtime.action_adapter import (
    ActionAdapter,
    ActionResolutionError,
    validate_target_consistency,
)
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.observation import describe_elements
from gui_agent.runtime.recorder import _mask_step
from gui_agent.runtime.schemas import ObservationSnapshot
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult


class AuditError(ValueError):
    pass


_GENERATION_TARGETING_POLICY = {
    "schema_version": "gui_agent.generation_targeting_policy.v1",
    "instruction": (
        "Generation guidance only; it grants neither input nor success authority. "
        "Use only candidates in the current target_bindings scope, including the final authorised subset. "
        "Use text-only targeting only when current OCR labels or adjacent OCR words can locate the target. "
        "If a target is visible in the screenshot but absent from OCR, choose its matching current ID yourself "
        "from target_bindings.anonymous_without_ocr_ids; never invent or replace an ID. "
        "If no matching allowed candidate is certain, return no steps and explain in errors."
    ),
    "generation_only": True,
}
_BASE_RECOVERY_INSTRUCTION = (
    "Plan from the current observation; previous attempts do not prove success. "
    "Do not replay completed inputs to recreate an earlier state. "
    "Existing task checks and remaining budgets still apply."
)
_CONSISTENCY_RECOVERY_INSTRUCTION = (
    " The previous plan's target_text conflicted with its explicit element ID "
    "or identified a container with ambiguous labels. No input was dispatched "
    "for that rejected step. Use the current observation and select an ID "
    "consistent with the same requested target; do not change the task goal."
)
_NO_MATCH_RECOVERY_INSTRUCTION = (
    " The rejected text-only target did not resolve to current OCR or adjacent words; "
    "no input was dispatched for that step. Use only current allowed candidates. "
    "If the target is visible in the screenshot without OCR, choose its matching "
    "anonymous_without_ocr ID yourself; do not repeat unresolved text-only targeting. "
    "If uncertain, return no steps with errors. Do not change the task goal."
)


def require(condition: Any, code: str) -> None:
    if not condition:
        raise AuditError(code)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate_json_key")
        value[key] = item
    return value


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "timestamp_without_timezone")
    return result


class Evidence:
    def __init__(self, container: Path):
        self.container = container.resolve()
        self.run = (self.container / "run").resolve()
        self.files: dict[str, str] = {}

    def bind(self, path: Path, *, scope: Path | None = None, label: str | None = None) -> Path:
        path = path.resolve()
        require(path.is_file(), f"missing_evidence:{path.name}")
        require(path.is_relative_to((scope or self.container).resolve()), "evidence_path_escaped_selected_scope")
        key = label or path.relative_to(self.container).as_posix()
        digest = sha(path)
        require(key not in self.files or self.files[key] == digest, "evidence_changed_while_auditing")
        self.files[key] = digest
        return path

    def external(self, path: Path, label: str) -> Path:
        return self.bind(path, scope=path.parent, label=label)

    def load(self, path: Path, *, run_only: bool = True):
        target = self.bind(path, scope=self.run if run_only else self.container)
        return json.loads(target.read_bytes(), object_pairs_hook=unique,
                          parse_constant=lambda _v: (_ for _ in ()).throw(AuditError("nonfinite_json")))

    def lines(self, path: Path):
        target = self.bind(path, scope=self.run)
        rows = target.read_bytes().splitlines()
        return [json.loads(row, object_pairs_hook=unique) for row in rows if row.strip()]

    def frame(self, path: Path) -> ObservationSnapshot:
        value = self.load(path)
        payload = {key: value[key] for key in ObservationSnapshot.model_fields if key in value}
        payload["notices"] = value.get("notices", value.get("ocr_notices", []))
        frame = ObservationSnapshot.model_validate(payload)
        self.bind(Path(frame.image_path), scope=self.run)
        require(not _initial_guard(frame), "unreadable_or_unscoped_saved_frame")
        return frame


def same_check(actual: dict | None, expected) -> bool:
    if actual is None:
        return False
    return actual == expected.model_dump(mode="json")


def command_value(command: list[str], flag: str) -> str:
    require(command.count(flag) == 1, f"owner_command_missing_or_duplicate:{flag}")
    index = command.index(flag)
    require(index + 1 < len(command), "owner_command_truncated")
    return command[index + 1]


def audit_repair_context(evidence: Evidence, directory: Path, index: int, context: dict,
                         *, inline_previous_response: bool = True) -> None:
    keys = {"constraint_error", "previous_response_record", "previous_response_characters", "previous_response"}
    if index == 1:
        require(not keys.intersection(context), "first_attempt_contains_unattributed_repair_context")
        return
    previous_path = directory / f"attempt_{index - 1:02d}_response.json"
    previous = evidence.load(previous_path)
    error = evidence.load(directory / f"attempt_{index - 1:02d}_error.json")
    content = previous["content"]
    require(Path(context.get("previous_response_record", "")).resolve() == previous_path.resolve()
            and type(context.get("previous_response_characters")) is int
            and context["previous_response_characters"] == len(content)
            and context.get("constraint_error") == error["error"], "repair_reference_count_or_error_not_bound_to_original_reply")
    if inline_previous_response and len(content) <= _INLINE_CORRECTION_MAX_CHARACTERS:
        require(context.get("previous_response") == content, "small_original_reply_not_preserved_in_repair_context")
    else:
        require("previous_response" not in context,
                "oversized_original_reply_was_echoed_into_repair_context" if inline_previous_response
                else "conversation_raw_reply_was_echoed_into_correction_context")


def expected_native_mapping(native: dict, parsed: SemanticPlan, tasks: list) -> list[dict]:
    return [{"goal_id": goal.goal_id, "native_instruction": native["subgoals"][index]["instruction"],
             "validated_instruction": goal.instruction, "compiled_instruction": task.instruction,
             "expect_text": goal.expect_text, "forbid_text": goal.forbid_text}
            for index, (goal, task) in enumerate(zip(parsed.subgoals, tasks, strict=True))]


_INTENT_CONTEXT_KEYS = {"original_caller_request", "minimum_objectives", "maximum_objectives",
                        "revision", "completed_subgoals", "failure", "input_modality", "image_attached"}
_REPAIR_CONTEXT_KEYS = {"constraint_error", "previous_response_record", "previous_response_characters",
                        "previous_response"}
_LOGICAL_GOAL_ORDER = ("goal_id", "instruction", "expect_text", "forbid_text")
_LOGICAL_GOAL_KEYS = set(_LOGICAL_GOAL_ORDER)


def audited_completed_goals(completed: list[dict]) -> list[dict]:
    """Use the producer's stable wire order, not process-random set order."""
    return [{key: item[key] for key in _LOGICAL_GOAL_ORDER} for item in completed]


def audited_logical_failure(failure: dict | None) -> dict | None:
    """Reconstruct bounded logical feedback, never original screen diagnostics."""
    if failure is None:
        return None
    status = failure.get("status")
    if status == "rejected_before_input":
        return {
            "category": "semantic_condition_preexisting_before_input",
            "rejected_goal": {key: failure["goal"][key] for key in _LOGICAL_GOAL_ORDER},
            "required_revision": (
                "The rejected predicted condition was already true before its operation. "
                "No input or completion is credited. Regenerate only unfinished work with "
                "a distinguishable AFTER-state, or group remaining operations with a justified "
                "combined outcome. Do not repeat verified objectives or the rejected contract."
            ),
        }
    if status == "failed" and _LOGICAL_GOAL_KEYS <= set(failure):
        goal = {key: failure[key] for key in _LOGICAL_GOAL_ORDER}
        verification = failure.get("task_verification")
        evidence = (verification.get("evidence", {}) if type(verification) is dict
                    and verification.get("outcome") == "failed" else {})
        if (type(evidence) is not dict
                or evidence.get("verification_source") != "frozen_model_semantic_subtask_postconditions"):
            evidence = {}
        feedback = {}
        for evidence_key, goal_key, output_key in (
            ("missing", "expect_text", "missing_expected_text"),
            ("unexpected", "forbid_text", "present_forbidden_text"),
        ):
            values = evidence.get(evidence_key)
            if evidence_key in evidence:
                require(type(values) is list and all(type(value) is str for value in values)
                        and len(values) == len(set(values))
                        and all(value in goal[goal_key] for value in values),
                        "failed_feedback_literal_not_in_original_native_contract")
            feedback[output_key] = values
        return {"category": "semantic_postconditions_unsatisfied", "failed_goal": goal,
                "predicate_feedback": feedback,
                "required_revision": (
                    "The previous model's predicted conditions were not achieved; the caller "
                    "request has not changed. Missing predicted text is not missing caller "
                    "information or a contradiction in caller parameters. The feedback lists "
                    "only literals from that original model contract; null means unavailable, "
                    "not an empty or passed check. Past inputs alone do not prove completion. "
                    "Plan only unfinished operations, do not replay completed selections or "
                    "verified objectives, and do not repeat the failed contract unchanged. "
                    "Use a justified distinguishable outcome within the same remaining budgets."
                )}
    return {"category": "semantic_postconditions_unsatisfied" if status == "failed" else
            "caller_final_unsatisfied" if "reason" in failure else "previous_objective_not_completed"}


def audit_logical_failure_shape(logical: dict | None) -> None:
    if logical is None:
        return
    require(type(logical) is dict, "highlevel_logical_failure_not_object")
    if set(logical) == {"category"}:
        require(type(logical["category"]) is str, "highlevel_logical_failure_category_not_text")
        return
    category = logical.get("category")
    if category == "semantic_condition_preexisting_before_input":
        goal_key = "rejected_goal"
        require(set(logical) == {"category", goal_key, "required_revision"},
                "highlevel_rejection_has_unattributed_fields")
    else:
        goal_key = "failed_goal"
        require(category == "semantic_postconditions_unsatisfied"
                and set(logical) == {"category", goal_key, "predicate_feedback", "required_revision"},
                "highlevel_failed_contract_has_unattributed_fields")
    goal = logical[goal_key]
    require(type(goal) is dict and set(goal) == _LOGICAL_GOAL_KEYS
            and type(goal["goal_id"]) is str and type(goal["instruction"]) is str
            and all(type(goal[key]) is list and len(goal[key]) <= SEMANTIC_CHECKS_PER_LIST
                    and all(type(value) is str and len(value) <= SEMANTIC_CHECK_MAX_CHARACTERS
                            for value in goal[key]) for key in ("expect_text", "forbid_text")),
            "highlevel_logical_goal_contains_non_native_fields")
    if goal_key == "rejected_goal":
        expected = audited_logical_failure({"status": "rejected_before_input", "goal": goal})
    else:
        feedback = logical["predicate_feedback"]
        require(type(feedback) is dict
                and set(feedback) == {"missing_expected_text", "present_forbidden_text"},
                "highlevel_predicate_feedback_has_unattributed_fields")
        evidence = {"verification_source": "frozen_model_semantic_subtask_postconditions"}
        for key, source_key in (("missing_expected_text", "missing"), ("present_forbidden_text", "unexpected")):
            if feedback[key] is not None:
                evidence[source_key] = feedback[key]
        expected = audited_logical_failure({"status": "failed", **goal,
            "task_verification": {"outcome": "failed", "evidence": evidence}})
    require(logical == expected, "highlevel_logical_failure_not_exact_fixed_contract")


def audit_native_predicates(native: dict, _request: WorkflowRequest) -> None:
    """Independently check raw v6 predictions before normalizing model classes."""
    require(type(native) is dict and set(native) == {"status", "reason", "subgoals"},
            "native_semantic_response_fields_changed")
    require(type(native.get("status")) is str and type(native.get("reason")) is str
            and len(native["reason"]) <= SEMANTIC_REASON_MAX_CHARACTERS,
            "native_semantic_status_or_reason_type_changed")
    require(type(native.get("subgoals")) is list, "native_semantic_goals_are_not_a_list")
    for goal in native["subgoals"]:
        require(type(goal) is dict and set(goal) == _LOGICAL_GOAL_KEYS,
                "native_semantic_goal_fields_missing_or_changed")
        instruction = goal["instruction"]
        require(type(goal["goal_id"]) is str and type(instruction) is str
                and 0 < len(instruction) <= SEMANTIC_INSTRUCTION_MAX_CHARACTERS
                and instruction == instruction.strip(), "native_instruction_not_preserved_without_normalization")
        for field in ("expect_text", "forbid_text"):
            values = goal[field]
            require(type(values) is list and len(values) <= SEMANTIC_CHECKS_PER_LIST
                    and all(type(value) is str and 0 < len(value) <= SEMANTIC_CHECK_MAX_CHARACTERS
                            and value == value.strip() for value in values),
                    "native_predicate_is_not_bounded_strict_text_preserved_without_normalization")


def _audit_semantic_repair_category(evidence: Evidence, directory: Path, index: int, saved: dict,
                                    request: WorkflowRequest, before: ObservationSnapshot,
                                    completed: list[dict], maximum: int, revision: bool,
                                    blocked_signatures: tuple[tuple[Any, ...], ...] = ()) -> tuple[str | None, str | None]:
    """Recompute the previous local rejection; never trust an error's claimed type."""
    audit_repair_context(evidence, directory, index, saved["context"], inline_previous_response=False)
    category = None
    content = None
    if index > 1:
        previous = evidence.load(directory / f"attempt_{index - 1:02d}_response.json")
        error = evidence.load(directory / f"attempt_{index - 1:02d}_error.json")
        require(previous.get("error") is None and type(previous.get("content")) is str
                and bool(previous["content"].strip()), "semantic_repair_after_unusable_transport_reply")
        content = previous["content"]
        try:
            native = json.loads(content, object_pairs_hook=_unique_keys)
            _validate_native_predicates(native, request)
            parsed = SemanticPlan.model_validate(native, strict=True)
            SemanticDecomposer(max_subgoals=maximum)._validate(
                parsed, request, before, completed, revision, blocked_signatures=blocked_signatures)
        except ValueError as exc:
            require(error.get("error") == str(exc), "semantic_repair_error_not_reproduced_from_original_reply")
            if isinstance(exc, _DuplicateSemanticPostconditionsError):
                category = "duplicate_postconditions"
            elif isinstance(exc, _PreexistingSemanticPostconditionsError):
                category = "preexisting_postconditions"
        else:
            raise AuditError("semantic_repair_after_locally_valid_reply")
    return category, content


def _audited_semantic_system(protocol: dict, category: str | None) -> str:
    require(protocol.get("repair_policy") == {
        "schema_version": SEMANTIC_REPAIR_POLICY_VERSION,
        "trigger": "local_validation_error_with_typed_duplicate_or_preexisting_category",
        "system_feedback_sha256": hashlib.sha256(SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR.encode()).hexdigest(),
        "generic_feedback_sha256": hashlib.sha256(SEMANTIC_GENERIC_REPAIR.encode()).hexdigest(),
        "preexisting_feedback_sha256": hashlib.sha256(SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR.encode()).hexdigest(),
        "placement": "fixed_system_hint_before_exact_response_schema_and_final_user_repair",
        "messages": "system_base_user_optional_previous_assistant_final_user",
        "history": "immediately_previous_reply_only_rebuilt_from_base",
        "previous_response_max_characters": _INLINE_CORRECTION_MAX_CHARACTERS,
        "oversized_previous_response": "archived_in_full_assistant_omitted_no_truncation",
        "raw_response_in_correction_context": False,
    }, "semantic_repair_policy_missing_or_changed")
    instructions = SEMANTIC_SYSTEM_PROMPT
    if category == "duplicate_postconditions":
        instructions += "\n" + SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR + "\n"
    elif category == "preexisting_postconditions":
        instructions += "\n" + SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR + "\n"
    system = (instructions + "\nExact response JSON schema (also supplied as response_format):\n"
              + json.dumps(protocol["response_format"]["json_schema"]["schema"],
                           ensure_ascii=False, sort_keys=True))
    return system


def audit_semantic_repair_system(evidence: Evidence, directory: Path, index: int, saved: dict,
                                protocol: dict, request: WorkflowRequest, before: ObservationSnapshot,
                                completed: list[dict], maximum: int, revision: bool,
                                blocked_signatures: tuple[tuple[Any, ...], ...] = ()) -> str:
    category, _ = _audit_semantic_repair_category(evidence, directory, index, saved, request,
                                                before, completed, maximum, revision, blocked_signatures)
    system = _audited_semantic_system(protocol, category)
    require(saved.get("system") == system, "semantic_repair_system_not_exactly_bound_to_typed_prior_rejection")
    return system


def audit_semantic_repair_messages(evidence: Evidence, directory: Path, index: int, saved: dict,
                                  protocol: dict, request: WorkflowRequest, before: ObservationSnapshot,
                                  completed: list[dict], maximum: int, revision: bool,
                                  base_context: dict,
                                  blocked_signatures: tuple[tuple[Any, ...], ...] = ()) -> list[dict]:
    category, content = _audit_semantic_repair_category(evidence, directory, index, saved, request,
                                                      before, completed, maximum, revision, blocked_signatures)
    system = _audited_semantic_system(protocol, category)
    require(saved.get("system") == system, "semantic_repair_system_not_exactly_bound_to_typed_prior_rejection")
    messages = [{"role": "system", "content": system},
                *ModelRequest(instruction=request.instruction, context=base_context).to_messages()]
    context = base_context
    if index > 1:
        error = evidence.load(directory / f"attempt_{index - 1:02d}_error.json")
        context = {**base_context, "constraint_error": error["error"],
                   "previous_response_record": str((directory / f"attempt_{index - 1:02d}_response.json").resolve()),
                   "previous_response_characters": len(content)}
        if len(content) <= _INLINE_CORRECTION_MAX_CHARACTERS:
            messages.append({"role": "assistant", "content": content})
        repair_instruction = SEMANTIC_GENERIC_REPAIR
        if category == "duplicate_postconditions":
            repair_instruction = SEMANTIC_DUPLICATE_POSTCONDITIONS_REPAIR
        elif category == "preexisting_postconditions":
            repair_instruction = SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR
        messages.append({"role": "user", "content": json.dumps({
            "instruction": request.instruction, "context": context, "repair_instruction": repair_instruction,
        }, ensure_ascii=False)})
    require(saved.get("context") == context, "semantic_conversation_base_or_correction_context_changed")
    require(saved.get("messages") == messages, "semantic_conversation_messages_not_exact_original_roles_and_content")
    return messages


def audit_highlevel_intent_binding(saved: dict, protocol: dict, plan_record: dict,
                                  observed: ObservationSnapshot, archived: dict) -> None:
    require(isinstance(observed.image_path, str) and bool(observed.image_path),
            "highlevel_original_pixel_provenance_missing")
    require(plan_record.get("input_modality") == protocol.get("input_modality")
            == saved.get("input_modality") == saved.get("context", {}).get("input_modality") == SEMANTIC_INPUT_MODALITY
            and plan_record.get("image_attached") is False and protocol.get("image_attached") is False
            and saved.get("image_attached") is False and saved.get("context", {}).get("image_attached") is False
            and "image_path" in saved and saved["image_path"] is None,
            "highlevel_call_is_not_explicitly_text_only")
    require(saved.get("snapshot_image_path_provenance") == protocol.get("snapshot_image_path_provenance")
            == observed.image_path and archived == observed.model_dump(mode="json"),
            "highlevel_original_snapshot_or_pixel_provenance_changed")
    context = saved["context"]
    require(protocol.get("snapshot_sent_to_model") is False
            and _INTENT_CONTEXT_KEYS <= set(context) <= _INTENT_CONTEXT_KEYS | _REPAIR_CONTEXT_KEYS,
            "highlevel_intent_context_contains_screen_or_unattributed_fields")
    audit_logical_failure_shape(context["failure"])
    require(isinstance(context["completed_subgoals"], list)
            and all(isinstance(goal, dict) and set(goal) == _LOGICAL_GOAL_KEYS
                    for goal in context["completed_subgoals"]),
            "highlevel_intent_context_contains_runtime_completion_or_failure_evidence")


def audit_snapshot_archive(evidence: Evidence, directory: Path, observed: ObservationSnapshot,
                           completed: list[dict], failure: dict | None) -> dict:
    archived = evidence.load(directory / "snapshot_archive.json")
    require(archived == observed.model_dump(mode="json"), "archived_snapshot_not_the_latest_actual_frame")
    sources = evidence.load(directory / "context_source_archive.json")
    expected_name = "snapshot_pixels" + Path(observed.image_path).suffix
    require(sources == {
        "completed_subgoals": completed, "failure": failure,
        "snapshot_archive": "snapshot_archive.json", "snapshot_pixels_archive": expected_name,
        "snapshot_pixels_sha256": sha(evidence.bind(Path(observed.image_path), scope=evidence.run)),
        "snapshot_pixels_source_status": "copied_byte_exact", "sent_to_high_level_model": False,
    }, "archived_runtime_context_or_pixel_source_not_bound")
    pixels = evidence.bind(directory / expected_name, scope=directory)
    require(sha(pixels) == sources["snapshot_pixels_sha256"], "archived_snapshot_pixels_changed")
    return archived


def audit_pre_input_rejection(evidence: Evidence, root: Path, round_index: int,
                              goal_index: int, goal, task, before: ObservationSnapshot) -> dict:
    """Qualify a refused contract without granting action or semantic credit."""
    checked = Verifier().check_task(task, before)
    require(checked.passed, "rejected_contract_did_not_preexist")
    record = evidence.load(root / f"goal_rejection_{round_index:03d}_{goal_index:03d}.json")
    expected = {
        "schema_version": "week6.semantic.rejected_pre_input_contract.v1",
        "round": round_index, "goal_index": goal_index,
        "goal": goal.model_dump(mode="json"), "status": "rejected_before_input",
        "reason": "semantic goal checks already hold before input; no completion credit",
        "before_reference": str((root / f"contract_before_{round_index:03d}_{goal_index:03d}" / "observation.json").resolve()),
        "before_verification": checked.model_dump(mode="json"),
        "confirmed_inputs": 0, "eligible_for_semantic_credit": False,
    }
    require(record == expected and type(record["round"]) is int
            and type(record["goal_index"]) is int and type(record["confirmed_inputs"]) is int
            and record["eligible_for_semantic_credit"] is False,
            "rejected_pre_input_contract_changed_or_credited")
    require(not (root / f"goal_contract_{round_index:03d}_{goal_index:03d}.json").exists()
            and not (root / f"goal_{round_index:03d}_{goal_index:03d}_{goal.goal_id}").exists(),
            "rejected_pre_input_contract_has_execution_artifacts")
    return record


def audit_lowlevel_image_binding(call: dict, observed: ObservationSnapshot) -> None:
    require(isinstance(observed.image_path, str) and bool(observed.image_path)
            and isinstance(call.get("image_path"), str) and bool(call["image_path"])
            and Path(call["image_path"]).resolve() == Path(observed.image_path).resolve(),
            "lowlevel_action_plan_lacks_its_actual_saved_image")


def audit_target_bindings(context: dict, observed: ObservationSnapshot, *,
                           authorised_candidate_ids: list[str] | None = None,
                           min_confidence: float = 0.35) -> dict:
    """Recompute bindings from saved pixels' actual candidate records.

    A message caller must independently establish its final authorised candidate
    subset before passing IDs here; context.candidate_ids is not authorisation.
    The semantic live harness is non-message and always audits the full frame.
    """
    source_ids = [element.element_id for element in observed.elements]
    require(len(source_ids) == len(set(source_ids)), "target_binding_source_has_duplicate_ids")
    if authorised_candidate_ids is None:
        candidates = list(observed.elements)
        scope = "all_snapshot_candidates"
    else:
        require(type(authorised_candidate_ids) is list
                and len(authorised_candidate_ids) == len(set(authorised_candidate_ids))
                and all(identifier in source_ids for identifier in authorised_candidate_ids),
                "target_binding_authorised_subset_not_independently_in_source")
        candidates = [element for element in observed.elements if element.element_id in authorised_candidate_ids]
        require([element.element_id for element in candidates] == authorised_candidate_ids,
                "target_binding_authorised_subset_order_changed")
        scope = "authorised_subset"
    require(type(min_confidence) in (float, int) and math.isfinite(min_confidence)
            and 0 <= min_confidence <= 1, "target_binding_audited_threshold_invalid")

    def overlaps(left, right):
        return (max(left.left, right.left) < min(left.right, right.right)
                and max(left.top, right.top) < min(left.bottom, right.bottom))

    selected = {element.element_id for element in candidates}
    local_ocr = [element for element in candidates if element.source == "ocr"
                 and element.text.strip() and element.confidence >= min_confidence]
    excluded_ocr = [element for element in observed.elements if element.element_id not in selected
                    and element.source == "ocr" and element.text.strip() and element.confidence >= min_confidence]
    values, labels, anonymous = [], {}, []
    for element in candidates:
        external = (scope == "authorised_subset" and not element.text.strip() and element.source == "contour"
                    and any(overlaps(element.bounding_box, other.bounding_box) for other in excluded_ocr))
        raw = {"element_id": element.element_id, "text": element.text, "source": element.source,
               "confidence": element.confidence, "bounding_box": element.bounding_box.model_dump(mode="json"),
               "external_ocr_overlap": external}
        category, compatible, local = "untrusted_or_other", [], []
        if element.text.strip() and element.confidence >= min_confidence:
            category, compatible = "readable_label", [element.text]
        elif not element.text.strip() and element.source == "contour":
            local = [other for other in local_ocr if overlaps(element.bounding_box, other.bounding_box)]
            if external:
                category = "untrusted_or_other"
            elif not local:
                category = "anonymous_without_ocr"
                anonymous.append(element.element_id)
            elif len(local) == 1:
                category, compatible = "anonymous_with_single_ocr", [local[0].text]
            else:
                category = "anonymous_mixed_ocr"
        values.append({**raw, "classification": category, "compatible_texts": compatible,
                       "overlapping_ocr_ids": [other.element_id for other in local]})
        for label in compatible:
            labels.setdefault(label, []).append(element.element_id)
    expected = {"schema_version": "gui_agent.target_bindings.v1", "observation_id": observed.observation_id,
                "scope": scope, "min_confidence": min_confidence, "candidates": values,
                "labels": [{"text": label, "element_ids": identifiers} for label, identifiers in labels.items()],
                "anonymous_without_ocr_ids": anonymous}
    subset = observed.model_copy(update={"elements": candidates})
    require(context.get("observation_id") == observed.observation_id
            and context.get("visible_text") == describe_elements(subset), "target_binding_visible_candidate_scope_changed")
    require(context.get("target_bindings") == expected, "target_bindings_not_rebuilt_from_actual_scoped_snapshot")
    return expected


def audit_message_modality(call: dict, messages: list[dict]) -> None:
    require(all(isinstance(message.get("content"), str) for message in messages),
            "saved_preencoding_messages_unexpectedly_contain_attachments")
    users = [message for message in messages if message["role"] == "user"]
    if call["kind"] == "semantic_decomposition":
        index = call.get("semantic_attempt_index", 1)
        require(type(index) is int and 1 <= index <= 3 and len(users) == (1 if index == 1 else 2),
                "native_request_user_turn_inventory_changed")
        for user in users:
            payload = json.loads(user["content"], object_pairs_hook=unique)
            require(call["image_path"] is None and "image_path" not in payload,
                    "highlevel_text_request_attached_a_top_level_image")
    else:
        require(len(users) == 1, "native_request_user_turn_inventory_changed")
        payload = json.loads(users[0]["content"], object_pairs_hook=unique)
        require(isinstance(call["image_path"], str) and payload.get("image_path") == call["image_path"],
                "lowlevel_visual_request_dropped_the_original_image")


def audit_action_attempts(call: dict, observed: ObservationSnapshot, records: list[dict],
                          expected_response_format: dict, *, maximum_attempts: int = 2,
                          authorised_candidate_ids: list[str] | None = None) -> list[dict]:
    """Rebuild each low-level request and correction from the original replies.

    This function does not confer input credit. A malformed reply remains a
    malformed reply, and the unchanged parser decides whether a plan exists.
    The live harness uses SDK retries=0; every accepted attempt has exactly one
    actual dispatch. Missing/budget-denied attempts cannot establish acceptance.
    """
    audit_lowlevel_image_binding(call, observed)
    require(type(call.get("attempts")) is int and 0 < call["attempts"] <= maximum_attempts
            and len(records) == call["attempts"]
            and type(call.get("model_requests")) is int and call["model_requests"] == len(records),
            "action_attempt_or_actual_request_inventory_changed")
    base_context = {**deepcopy(call["context"]), "max_steps": 1}
    require("plan_format_correction" not in base_context, "action_base_context_contains_unattributed_correction")
    binding_contract = None
    bindings = None
    if "target_bindings" in base_context:
        bindings = audit_target_bindings(call["context"], observed,
                                         authorised_candidate_ids=authorised_candidate_ids)
        expected_format = _task_plan_response_format(1, target_bindings=bindings)

        def digest(value):
            return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()

        queries = independent_observed_query_contract(bindings)
        binding_contract = {"schema_version": "gui_agent.target_binding_schema.v2", "bindings": bindings,
                            "bindings_sha256": digest(bindings),
                            "response_format_sha256": digest(expected_format), "generation_only": True,
                            "observed_query_contract": queries, "observed_query_contract_sha256": digest(queries)}
    else:
        expected_format = _task_plan_response_format(1)
    require(expected_response_format == expected_format, "action_expected_schema_not_rebuilt_from_scoped_snapshot")
    correction = None
    requests = []
    last_plan = None
    for index, record in enumerate(records, 1):
        require(record.get("schema_version") == ACTION_OUTPUT_CONTRACT_VERSION
                and type(record.get("attempt")) is int and record["attempt"] == index,
                "action_attempt_contract_or_order_changed")
        if binding_contract is None:
            require("target_binding_contract" not in record, "legacy_action_attempt_injected_a_binding_contract")
        else:
            require(record.get("target_binding_contract") == binding_contract,
                    "action_target_binding_contract_or_digest_changed")
            audit_observed_query_contract(bindings, record["target_binding_contract"]["observed_query_contract"])
        context = deepcopy(base_context)
        if correction is not None:
            context["plan_format_correction"] = correction
        expected_request = {"instruction": call["instruction"], "context": context,
                            "image_path": call["image_path"], "system": SYSTEM_PROMPT,
                            "response_format": expected_response_format}
        require(record.get("request") == expected_request, "action_attempt_request_or_schema_not_reconstructed")
        response = record.get("response")
        require(type(response) is dict and set(response) == {
            "provider", "model_name", "latency_ms", "usage", "error", "content_length", "content"}
            and type(response["content"]) is str and type(response["content_length"]) is int
            and response["content_length"] == len(response["content"]), "action_attempt_native_response_changed")
        content = response["content"]
        require(response["error"] is None and bool(content.strip()), "action_attempt_has_unusable_transport_response")
        require(response["provider"] == call["response"]["provider"]
                and response["model_name"] == call["response"]["model_name"], "action_attempt_model_identity_changed")
        try:
            parsed = parse_plan(content, instruction=call["instruction"], max_steps=1, task_id=call["task_id"])
        except PlanParseError as exc:
            error = str(exc)
            require(record.get("parse_error") == error and "parsed_plan" not in record
                    and "model_error" not in record, "action_format_error_not_reproduced_from_original_reply")
            correction = {"repair_instruction": _FORMAT_REPAIR_INSTRUCTION,
                          "format_error_characters": len(error),
                          "format_error_sha256": hashlib.sha256(error.encode("utf-8")).hexdigest(),
                          "previous_response_characters": len(content),
                          "previous_response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
            if len(error) <= _INLINE_FORMAT_ERROR_CHARACTERS:
                correction["format_error"] = error
            if len(content) <= _INLINE_FORMAT_RESPONSE_CHARACTERS:
                correction["previous_response"] = content
            last_plan = None
        else:
            try:
                native = json.loads(content, object_pairs_hook=unique)
            except ValueError as exc:
                raise AuditError("action_native_is_not_exact_json_for_schema_qualification") from exc
            require(not list(Draft202012Validator(expected_response_format["json_schema"]["schema"]).iter_errors(native)),
                    "action_native_violates_predeclared_generation_schema")
            require(index == len(records) and record.get("parse_error") is None
                    and "model_error" not in record and record.get("parsed_plan") == parsed.as_dict(),
                    "action_valid_plan_followed_by_another_attempt_or_changed")
            last_plan = parsed.as_dict()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *ModelRequest(
            instruction=call["instruction"], context=context, image_path=call["image_path"]).to_messages()]
        requests.append({"kind": "primitive_action_plan", "system": SYSTEM_PROMPT,
                         "instruction": call["instruction"], "context": context, "image_path": call["image_path"],
                         "messages": messages, "response_format": expected_response_format,
                         "response": response, "requests": 1})
    require(call.get("ok") is (last_plan is not None) and call.get("parsed_plan") == last_plan,
            "action_call_verdict_or_native_plan_differs_from_attempt_ledger")
    for key in ("content", "provider", "model_name", "error", "usage", "content_length"):
        require(call["response"].get(key) == records[-1]["response"][key], "action_final_response_borrowed_or_changed")
    return requests


def independent_observed_query_contract(bindings: dict) -> dict:
    """Rebuild the finite query event stream from independently scoped records.

    The producer helper is deliberately not called. Ordinary frozen runtime text
    matchers only qualify each <=4-element witness; they perform no OCR or input.
    This finite generation subset is not the complete runtime query language.
    """
    from gui_agent.perception.grounding import find_text
    from gui_agent.runtime.action_adapter import _match_adjacent_runs
    from gui_agent.runtime.schemas import ElementRef
    from gui_agent.schemas import BoundingBox

    limits = {"tokens_per_label": 32, "tokens_per_span": 4, "elements_per_run": 4,
              "extra_query_characters": 160, "extra_queries": 128, "search_steps": 4096}
    full = [label["text"] for label in bindings["labels"]]
    source = []
    for row in bindings["candidates"]:
        if row["source"] == "ocr" and row["text"].strip() and row["confidence"] >= bindings["min_confidence"]:
            bounds = BoundingBox(**row["bounding_box"])
            source.append(ElementRef(element_id=row["element_id"], text=row["text"], source="ocr",
                                     confidence=row["confidence"], bounding_box=bounds, center=bounds.center))
    tokens = {}
    truncation_count = 0
    for item in source:
        values = []
        for match in re.finditer(r"[0-9a-z_]+", item.text, re.IGNORECASE | re.ASCII):
            values.append((match.start(), match.end()))
            if len(values) == 33:
                truncation_count += 1
                break
        tokens[item.element_id] = values[:32]

    def adjacent(first, second):
        a, b = first.bounding_box, second.bounding_box
        if abs(b.top - a.top) <= 40 and 0 <= b.left - a.right <= 40:
            return True
        overlap = max(0, min(a.right, b.right) - max(a.left, b.left))
        return overlap / min(a.right - a.left, b.right - b.left) >= .30 and -40 <= b.top - a.bottom <= 40

    ordered = sorted(source, key=lambda item: (item.bounding_box.top, item.bounding_box.left))

    def events():
        for item in source:
            spans = tokens[item.element_id]
            for start in range(len(spans)):
                for length in range(1, min(4, len(spans) - start) + 1):
                    yield ("candidate", item.text[spans[start][0]:spans[start + length - 1][1]],
                           "label_token_span", [item], True)
        for start in range(len(ordered)):
            for length in range(2, min(4, len(ordered) - start) + 1):
                witness = ordered[start:start + length]
                yield ("candidate", " ".join(item.text.strip() for item in witness), "reading_order_run", witness,
                       all(adjacent(a, b) for a, b in pairwise(witness)))

        def spatial(path):
            if len(path) == 4:
                return
            for item in ordered:
                yield ("comparison", "", "", [], True)
                if item.element_id in {part.element_id for part in path} or not adjacent(path[-1], item):
                    continue
                witness = [*path, item]
                yield ("candidate", " ".join(part.text.strip() for part in witness), "spatial_run", witness, True)
                yield from spatial(witness)

        for item in ordered:
            yield from spatial([item])

    search = {"steps": 0, "candidate_queries_considered": 0, "duplicate_queries": 0,
              "overlength_queries": 0, "unresolvable_queries": 0,
              "labels_with_token_truncation": truncation_count, "count_limit_reached": False,
              "search_limit_reached": False, "truncated": False, "omitted_queries_at_least": 0}
    extra, seen = [], set(full)
    for kind, query, origin, witnesses, geometry in events():
        if search["steps"] == limits["search_steps"]:
            search["search_limit_reached"] = True
            break
        search["steps"] += 1
        if kind == "comparison":
            continue
        search["candidate_queries_considered"] += 1
        if not geometry:
            continue
        if query in seen:
            search["duplicate_queries"] += 1
            continue
        if len(query) > limits["extra_query_characters"]:
            search["overlength_queries"] += 1
            continue
        if not query.strip() or not (find_text(witnesses, query, min_confidence=bindings["min_confidence"]).found
                                     or _match_adjacent_runs(witnesses, query, bindings["min_confidence"])):
            search["unresolvable_queries"] += 1
            continue
        if len(extra) == limits["extra_queries"]:
            search["count_limit_reached"] = True
            search["omitted_queries_at_least"] = 1
            break
        extra.append({"query": query, "source_kind": origin,
                      "source_element_ids": [item.element_id for item in witnesses]})
        seen.add(query)
    search["truncated"] = bool(truncation_count or search["overlength_queries"]
                                or search["search_limit_reached"] or search["count_limit_reached"])
    return {"schema_version": "gui_agent.observed_text_queries.v1", "full_label_queries": full,
            "extra_queries": extra, "limits": limits, "search": search}


def audit_observed_query_contract(bindings: dict, saved: dict) -> dict:
    expected = independent_observed_query_contract(bindings)
    require(json.dumps(saved, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            == json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "observed_queries_not_rebuilt_from_final_scoped_snapshot")
    return expected


def audit_generation_targeting_policy(context: dict, *, previous_pass_steps: list[dict] | None = None) -> dict:
    """Require the fixed guidance, never infer a target or grant input credit.

    Direct legacy attempt-reconstruction helpers remain usable without this
    new field. The selected live audit calls this function for *every* action
    planning call, so removing the policy cannot downgrade a new run to legacy.
    Raw step errors are read solely to reproduce the fixed branch condition;
    none of their text, target labels or geometry enters the expected hint.
    """
    policy = context.get("targeting_policy")
    require(type(policy) is dict and policy.get("generation_only") is True
            and policy == _GENERATION_TARGETING_POLICY,
            "generation_targeting_policy_missing_changed_or_authorising")
    if previous_pass_steps is None:
        require("recovery" not in context, "initial_action_call_injected_recovery_guidance")
        return deepcopy(_GENERATION_TARGETING_POLICY)
    feedback = context.get("recovery")
    require(type(feedback) is dict and set(feedback) == {
        "previous_pass", "reason", "failed_step_index", "actions_used", "actions_remaining",
        "remaining_ms", "recent_steps", "omitted_steps", "instruction"},
        "recovery_guidance_has_missing_or_unattributed_fields")
    expected = _BASE_RECOVERY_INSTRUCTION
    if feedback["reason"] == "target_resolution":
        if any((item.get("error") or "").startswith("target consistency conflict:")
               for item in previous_pass_steps):
            expected += _CONSISTENCY_RECOVERY_INSTRUCTION
        if any(item.get("action_result") is None and (item.get("error") or "").startswith("no element matches ")
               for item in previous_pass_steps):
            expected += _NO_MATCH_RECOVERY_INSTRUCTION
    require(feedback["instruction"] == expected, "recovery_guidance_not_the_exact_fixed_instruction")
    return deepcopy(_GENERATION_TARGETING_POLICY)


def audit_no_match_recovery_origin(native_step: dict, fresh: ObservationSnapshot, recorded: dict, *, platform: str) -> None:
    """Reproduce the rejected resolution without capture, OCR or dispatch."""
    require(recorded.get("action_result") is None and recorded.get("resolved") is None
            and recorded.get("after_observation_id") is None,
            "no_match_guidance_attached_to_an_executed_or_resolved_step")
    try:
        ActionAdapter(platform=platform).resolve(PlanStep.model_validate(native_step), fresh)
    except ActionResolutionError as exc:
        require(str(exc) == recorded.get("error") and str(exc).startswith("no element matches "),
                "no_match_guidance_error_not_reproduced_from_original_native_and_fresh_frame")
    else:
        raise AuditError("no_match_guidance_did_not_originate_in_failed_resolution")


def owner_audit(evidence: Evidence, owner_harness: Path, fixture: Path) -> dict:
    claim = evidence.load(evidence.container / "owner_claim.json", run_only=False)
    start = evidence.load(evidence.container / "owner_start.json", run_only=False)
    receipt = evidence.load(evidence.container / "owner_receipt.json", run_only=False)
    evidence.bind(evidence.container / "stdout.txt")
    require(claim.get("schema") == "week6.semantic.live.claim.v1"
            and receipt.get("schema") == "week6.semantic.live.actual_exit.v1", "wrong_owner_schema")
    require(type(receipt.get("child_exit_code")) is int and receipt["child_exit_code"] == 0,
            "owner_did_not_observe_real_exit_zero")
    require(claim["owner"] == start["owner"] == receipt["owner"]
            and start["launcher"] == receipt["launcher"]
            and claim["command"] == start["command"] == receipt["command"], "owner_identity_or_command_changed")
    command = receipt["command"]
    require(Path(command[1]).resolve() == ROOT / "scripts/week6_semantic_acceptance.py"
            and Path(command_value(command, "--directory")).resolve() == evidence.run
            and Path(claim["batch"]).resolve() == evidence.run
            and Path(command_value(command, "--fixture-directory")).resolve() == fixture.resolve(),
            "owner_command_not_bound_to_explicit_selected_run")
    digest = sha(evidence.external(owner_harness.resolve(), "explicit_owner_harness"))
    require(start["owner_harness_sha256"] == receipt["owner_harness_sha256"] == digest,
            "owner_harness_sha_mismatch")
    require(timestamp(claim["started_utc"]) == timestamp(start["started_utc"]) == timestamp(receipt["started_utc"])
            and timestamp(receipt["finished_utc"]) >= timestamp(receipt["started_utc"])
            and type(receipt["elapsed_seconds"]) in (float, int) and receipt["elapsed_seconds"] >= 0,
            "owner_exit_timestamps_invalid")
    for identity in [receipt["owner"], receipt["launcher"], *receipt["observed_descendants"]]:
        require(type(identity.get("pid")) is int and identity["pid"] > 0
                and type(identity.get("create_time")) in (float, int)
                and math.isfinite(identity["create_time"]) and identity["create_time"] > 0,
                "owner_process_creation_identity_missing")
    return receipt


def audit_role_model_configuration(contract: dict, configured, command: list[str]) -> None:
    """Bind role selection to the actual command and pinned configuration."""
    require(contract["configuration"] == configured.model_dump(mode="json"),
            "role_model_configuration_differs_from_pinned_yaml")
    text_model = (command_value(command, "--text-model")
                  if "--text-model" in command else contract["model"])
    require(contract["text_model"] == text_model, "text_model_differs_from_actual_owner_command")
    if "--text-model" in command:
        require(configured.model.provider == "openai_compatible"
                and configured.model.model_name == contract["model"],
                "visual_model_differs_from_pinned_role_configuration")
        require(contract["model_transport"] == {
            "base_url": configured.model.base_url,
            "temperature": configured.model.temperature,
            "max_tokens": configured.model.max_tokens,
            "max_retries": configured.model.max_retries,
            "timeout_seconds": configured.model.timeout_seconds,
        }, "role_model_transport_differs_from_pinned_yaml")


def source_audit(evidence: Evidence, contract: dict, command: list[str]) -> dict:
    pins = contract["source_sha256"]
    config = Path(command_value(command, "--config")).resolve()
    require(config.is_relative_to(ROOT), "selected_config_escaped_repository")
    expected = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    expected |= {config.relative_to(ROOT).as_posix(), "scripts/week6_semantic_acceptance.py",
                 "scripts/week6_lab_acceptance.py", "scripts/week6_local_lab.py",
                 "scripts/week6_agent_cli.py", "pyproject.toml"}
    require(set(pins) == expected, "source_pin_inventory_differs_from_executed_harness_contract")
    for name, digest in pins.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "source_pin_path_escaped_repository")
        evidence.external(path, "current_source/" + name)
        snapshot = evidence.run / "source_snapshot" / name
        snapshot = snapshot.with_suffix(snapshot.suffix + ".txt")
        require(sha(evidence.bind(snapshot, scope=evidence.run)) == digest == sha(path),
                f"source_snapshot_current_or_before_pin_changed:{name}")
    if "text_model" in contract:
        from gui_agent.config import load_config

        configured = load_config(config)
        audit_role_model_configuration(contract, configured, command)
    return {"pin_count": len(pins), "all_saved_before_and_current_equal": True}


def audit(container: Path, owner_harness: Path, fixture: Path, evidence: Evidence) -> dict:
    owner = owner_audit(evidence, owner_harness, fixture)
    contract = evidence.load(evidence.run / "contract.json")
    require(contract.get("schema") == "week6.semantic_live.contract.v1"
            and contract.get("automatic_protocol") == "semantic_subtasks", "not_semantic_live_contract")
    sources = source_audit(evidence, contract, owner["command"])
    request = WorkflowRequest.model_validate(contract["request"])
    project = command_value(owner["command"], "--project")
    colour = command_value(owner["command"], "--colour")
    mode = command_value(owner["command"], "--mode")
    expected_request = WorkflowRequest(workflow_id="semantic_lab",
        instruction=f"In Week6 Local Lab, prepare a {colour} report for project {project}, "
                    "review it and commit it locally. Use the visible controls. This is an offline test "
                    "with no external communication. If review is temporarily unavailable, retry within budget.",
        expect_text=[f"Project {project}", f"Theme {colour}", "LOCAL COMPLETE"])
    require(request == expected_request and contract["fixture_mode"] == mode
            and contract["independent_expected"] == {"project": project, "colour": colour,
                "stage": "completed", "committed": True,
                "event_order": ["project_selected", "theme_selected", "review_accepted", "local_commit"],
                "review_rejections": int(mode == "transient")}, "owner_parameters_or_original_caller_final_changed")
    root = evidence.run / "automatic"
    raw = evidence.load(evidence.run / "runtime_result.json")
    typed = SemanticWorkflowResult.model_validate(raw)
    require(typed.workflow_id == request.workflow_id, "runtime_caller_identity_changed")
    require(raw == evidence.load(root / "automatic_workflow_summary.json"), "runtime_root_summary_mismatch")
    acceptance = evidence.load(evidence.run / "acceptance_observed.json")
    require(acceptance["automatic_result"] == raw, "acceptance_embedded_another_runtime")
    require(typed.status == "succeeded" and typed.accounting_complete and typed.unconfirmed_action_attempts == 0,
            "runtime_not_completed_with_complete_known_accounting")
    require(Path(typed.evidence_directory).resolve() == root.resolve(), "runtime_directory_changed")
    setup = evidence.load(root / "semantic_request.json")
    require(setup["schema_version"] == SEMANTIC_PROTOCOL_VERSION
            and setup["caller_request"] == request.model_dump(mode="json")
            and setup["options"] == contract["options"]
            and setup["max_model_requests"] == contract["max_model_requests"], "caller_or_shared_limits_changed")
    require(evidence.load(root / "semantic_preflight.json") == {
        "schema_version": SEMANTIC_PROTOCOL_VERSION, "supported": True, "issue": "",
        "scope": "declared unresolved literal forms and contradictory caller final predicates only",
    }, "semantic_caller_preflight_not_supported_or_changed")
    options = contract["options"]
    require(0 < typed.total_actions <= options["max_actions"]
            and 0 < typed.planning_calls <= options["max_planning_attempts"]
            and 0 < typed.model_requests <= contract["max_model_requests"]
            and typed.elapsed_ms < options["task_timeout_seconds"] * 1000, "global_budget_exceeded")
    verifier, final_task = Verifier(), _final_task(request)
    latest = evidence.frame(root / "initial_observation/observation.json")
    require(verifier.check_task(final_task, latest).outcome == "failed", "caller_final_preexisting_or_inconclusive")
    completed, outcomes, plan_calls, actions, stage_reviews, model_calls, all_steps = [], [], 0, 0, [], [], []
    action_attempt_rows = evidence.lines(root / "action_planning_attempts.jsonl")
    action_attempt_cursor = 0
    action_call_count = 0
    workflow_cursor, failure = 0, None
    rejected_contracts, rejected_signatures = [], set()
    for round_index, plan_record in enumerate(raw["plans"], 1):
        blocked_signatures = tuple(rejected_signatures | {
            (row["instruction"].casefold(), tuple(sorted(x.casefold() for x in row["expect_text"])),
             tuple(sorted(x.casefold() for x in row["forbid_text"])))
            for row in outcomes if row["status"] != "succeeded"
        })
        directory = root / f"semantic_plan_round_{round_index:03d}"
        require(Path(plan_record["evidence_directory"]).resolve() == directory.resolve()
                and evidence.load(directory / "result.json") == plan_record
                and plan_record["status"] == "ready", "accepted_plan_record_missing_or_changed")
        native = evidence.load(directory / "native_proposal.json")
        audit_native_predicates(native, request)
        parsed = SemanticPlan.model_validate(native, strict=True)
        require(parsed.model_dump(mode="json") == native == plan_record["proposal"],
                "native_semantic_proposal_not_original_or_normalized")
        signatures = {(goal.instruction.casefold(), tuple(sorted(x.casefold() for x in goal.expect_text)),
                       tuple(sorted(x.casefold() for x in goal.forbid_text))) for goal in parsed.subgoals}
        require(not signatures.intersection(rejected_signatures), "previously_rejected_contract_replayed")
        protocol = evidence.load(directory / "protocol.json")
        maximum = protocol["response_format"]["json_schema"]["schema"]["properties"]["subgoals"]["maxItems"]
        require(type(maximum) is int and 2 <= maximum <= 32, "semantic_schema_objective_bound_changed")
        expected_format = _semantic_response_format(request, maximum)
        require(protocol["schema_version"] == SEMANTIC_PROTOCOL_VERSION
                and protocol["caller_request"] == request.model_dump(mode="json")
                and protocol["revision"] == (round_index > 1)
                and protocol["response_format"] == expected_format
                and "predicate_pools" not in protocol
                and protocol.get("postcondition_source") == "untrusted_model_prediction"
                and protocol.get("final_source") == "unchanged_caller", "semantic_protocol_or_caller_changed")
        archived = audit_snapshot_archive(evidence, directory, latest, completed, failure)
        logical_failure = audited_logical_failure(failure)
        expected_context = {"original_caller_request": request.model_dump(mode="json"),
            "minimum_objectives": 1 if round_index > 1 else 2, "maximum_objectives": maximum,
            "revision": round_index > 1,
            "completed_subgoals": audited_completed_goals(completed),
            "failure": logical_failure, "input_modality": SEMANTIC_INPUT_MODALITY, "image_attached": False}
        require(protocol.get("output_constraints") == {
            "schema_version": "week6.semantic_output_constraints.v1",
            "reason_max_characters": SEMANTIC_REASON_MAX_CHARACTERS,
            "instruction_max_characters": SEMANTIC_INSTRUCTION_MAX_CHARACTERS,
            "check_max_characters": SEMANTIC_CHECK_MAX_CHARACTERS,
            "checks_per_list_max": SEMANTIC_CHECKS_PER_LIST,
            "inline_correction_max_characters": _INLINE_CORRECTION_MAX_CHARACTERS,
            "oversized_previous_response": "archived_in_full_not_repeated_in_correction_context"},
            "semantic_output_constraints_changed_or_unversioned")
        attempts = sorted(directory.glob("attempt_*_request.json"))
        require(len(attempts) == plan_record["planning_calls"] >= 1, "decomposition_attempts_incomplete")
        for index, path in enumerate(attempts, 1):
            saved = evidence.load(path)
            response = evidence.load(directory / f"attempt_{index:02d}_response.json")
            context = saved["context"]
            messages = audit_semantic_repair_messages(evidence, directory, index, saved, protocol,
                                                      request, latest, completed, maximum, round_index > 1,
                                                      expected_context, blocked_signatures)
            system = messages[0]["content"]
            require(saved["instruction"] == request.instruction
                    and saved["response_format"] == protocol["response_format"]
                    and {key: value for key, value in context.items() if key not in _REPAIR_CONTEXT_KEYS}
                    == expected_context
                    and response["provider"] == "openai_compatible"
                    and response["model_name"] == contract.get("text_model", contract["model"]),
                    "decomposition_prompt_or_completed_context_changed")
            audit_highlevel_intent_binding(saved, protocol, plan_record, latest, archived)
            model_calls.append({"system": system, "instruction": request.instruction,
                                "context": context, "image_path": saved["image_path"], "response": response,
                                "requests": 1, "kind": "semantic_decomposition", "messages": messages,
                                "response_format": protocol["response_format"],
                                "semantic_attempt_index": index})
        require(json.loads(response["content"], object_pairs_hook=unique) == native,
                "native_proposal_differs_from_last_accepted_raw_reply")
        SemanticDecomposer(max_subgoals=maximum)._validate(
            parsed, request, latest, completed, round_index > 1, blocked_signatures=blocked_signatures)
        compilation = evidence.load(directory / "compiled_workflow.json")
        expected_tasks = [_goal_task(goal, request) for goal in parsed.subgoals]
        require(compilation["native_proposal"] == native and compilation["schema_version"] == SEMANTIC_PROTOCOL_VERSION
                and compilation["workflow"] == plan_record["workflow"]
                and compilation["workflow"]["workflow_id"] == request.workflow_id
                and compilation["workflow"]["instruction"] == request.instruction
                and compilation["workflow"]["subgoals"] == [task.model_dump(mode="json") for task in expected_tasks]
                and compilation["workflow"]["final_check"] == final_task.model_dump(mode="json"), "native_to_compiled_mapping_changed")
        expected_mapping = expected_native_mapping(native, parsed, expected_tasks)
        require(compilation["mapping"] == expected_mapping, "native_goal_mapping_not_exact")
        plan_calls += plan_record["planning_calls"]
        for goal_index, (goal, task) in enumerate(zip(parsed.subgoals, expected_tasks, strict=True), 1):
            before = evidence.frame(root / f"contract_before_{round_index:03d}_{goal_index:03d}/observation.json")
            require(not _rolling_frame_guard(latest, before), "goal_before_frame_not_fresh")
            if verifier.check_task(task, before).passed:
                rejection = audit_pre_input_rejection(evidence, root, round_index, goal_index, goal, task, before)
                require(round_index < len(raw["plans"]), "unrecovered_preexisting_contract_in_accepted_run")
                rejected_contracts.append(rejection)
                rejected_signatures.add((goal.instruction.casefold(), tuple(sorted(x.casefold() for x in goal.expect_text)),
                                         tuple(sorted(x.casefold() for x in goal.forbid_text))))
                latest, failure = before, rejection
                break
            # A failed goal stops the remainder of this round; later goals remain unexecuted.
            require(workflow_cursor < len(raw["goal_outcomes"])
                    and raw["goal_outcomes"][workflow_cursor]["round"] == round_index,
                    "native_semantic_sequence_contains_an_unexecuted_required_goal")
            recorded = raw["goal_outcomes"][workflow_cursor]
            require(recorded["goal_id"] == goal.goal_id, "semantic_goal_execution_order_changed")
            require(verifier.check_task(task, before).outcome == "failed",
                    "goal_preconditions_preexisting_or_frame_not_fresh")
            frozen = evidence.load(root / f"goal_contract_{round_index:03d}_{goal_index:03d}.json")
            require(frozen["schema_version"] == SEMANTIC_PROTOCOL_VERSION
                    and frozen["round"] == round_index and frozen["goal"] == goal.model_dump(mode="json")
                    and frozen["task"] == task.model_dump(mode="json")
                    and frozen["before"] == before.model_dump(mode="json")
                    and frozen["caller_final"] == final_task.model_dump(mode="json")
                    and frozen["postconditions_fixed_before_input"] is True, "pre_input_semantic_contract_changed")
            goal_dir = root / f"goal_{round_index:03d}_{goal_index:03d}_{goal.goal_id}"
            workflow = raw["workflow_results"][workflow_cursor]
            require(Path(recorded["evidence_directory"]).resolve() == goal_dir.resolve()
                    and evidence.load(goal_dir / "workflow_summary.json") == workflow
                    and recorded["status"] == workflow["status"]
                    and recorded["verification"] == workflow["final_verification"]
                    and recorded["task_verification"] == workflow["subgoals"][0]["verification"]
                    and workflow["accounting_complete"] and workflow["unconfirmed_action_attempts"] == 0,
                    "goal_workflow_record_or_accounting_changed")
            child_dir = goal_dir / f"001_{goal.goal_id}"
            child = evidence.load(child_dir / "task_summary.json")
            require(child["case_id"] == goal.goal_id and child["instruction"] == task.instruction
                    and child["execute"] is True and child["provider"] == "openai_compatible"
                    and child["model_name"] == contract["model"] and child["unconfirmed_action_attempts"] == 0,
                    "semantic_child_identity_or_unknown_dispatch")
            audit_row = evidence.load(child_dir / "semantic_goal_audit.json")
            require(Path(workflow["subgoals"][0]["summary_reference"]).resolve() == (child_dir / "task_summary.json").resolve()
                    and workflow["subgoals"][0]["verification"] == child["verification"]
                    and audit_row["child_verification"] == child["verification"]
                    and audit_row["child_status"] == child["status"]
                    and audit_row["post_observation_id"] == child["verification"]["evidence"]["observation_id"],
                    "semantic_child_verdict_or_post_frame_binding_changed")
            steps = evidence.lines(child_dir / "steps.jsonl")
            require([_mask_step(step) for step in audit_row["steps"]] == child["steps"] == steps,
                    "original_semantic_steps_journal_mismatch")
            limit = evidence.load(child_dir / "single_step_planning_limit.json")
            require(Path(audit_row["planning_limit_reference"]).resolve() == (child_dir / "single_step_planning_limit.json").resolve()
                    and limit["protocol_version"] == SEMANTIC_PROTOCOL_VERSION
                    and limit["raw_step_verification_policy"] == SEMANTIC_STEP_POLICY
                    and limit["applied_max_steps"] == 1 and limit["restored"] is True
                    and limit["restored_max_steps"] == limit["original_max_steps"], "single_step_limit_or_restoration_missing")
            frames = {}

            def child_frame(identifier, frames=frames, child_dir=child_dir, before=before):
                require(isinstance(identifier, str) and identifier and Path(identifier).name == identifier,
                        "unsafe_or_empty_observation_id")
                if identifier not in frames:
                    value = evidence.frame(child_dir / f"{identifier}.json")
                    require(value.observation_id == identifier and not _rolling_frame_guard(before, value),
                            "child_frame_not_fresh_or_original_scope")
                    require(Path(value.image_path).resolve().is_relative_to(child_dir.resolve()),
                            "child_borrowed_image_from_another_goal")
                    frames[identifier] = value
                return frames[identifier]

            calls = limit["calls"]
            require(len(calls) == child["planning_calls"] == workflow["planning_calls"], "child_planning_call_inventory_incomplete")
            require(len(steps) == len(calls), "single_step_call_and_step_journal_inventory_changed")
            recovery_rows = child.get("recovery_records", [])
            require(len(recovery_rows) == len(calls) - 1, "recovery_guidance_inventory_incomplete")
            if recovery_rows:
                require(evidence.lines(child_dir / "recovery.jsonl") == recovery_rows,
                        "recovery_guidance_journal_differs_from_original_summary")
            native_steps, native_source_frames = [], []
            for call_index, call in enumerate(calls):
                require(call["max_steps"] == 1 and call["returned_steps"] in (None, 0, 1)
                        and call["instruction"] == task.instruction and call["task_id"] == goal.goal_id,
                        "action_planner_goal_or_one_step_bound_changed")
                frame = child_frame(call["context"]["observation_id"])
                audit_lowlevel_image_binding(call, frame)
                require("message_task" not in call["context"], "semantic_child_unexpectedly_uses_message_candidate_scope")
                previous_steps = None if call_index == 0 else [steps[call_index - 1]]
                audit_generation_targeting_policy(call["context"], previous_pass_steps=previous_steps)
                if call_index:
                    recovery = recovery_rows[call_index - 1]
                    feedback = call["context"]["recovery"]
                    previous_step = steps[call_index - 1]
                    previous_plan = calls[call_index - 1].get("parsed_plan")
                    require(previous_plan is not None and len(previous_plan["steps"]) == 1,
                            "recovery_guidance_without_the_previous_native_one_step_plan")
                    previous_native = previous_plan["steps"][0]
                    require(all(previous_native[key] == previous_step[key]
                                for key in ("step_id", "description", "action_type"))
                            and recovery["next_pass"] == call_index + 1
                            and recovery["observation_id"] == frame.observation_id
                            and recovery["outcome"] == "plan_ready" and recovery["feedback"] == feedback
                            and feedback["previous_pass"] == call_index,
                            "recovery_guidance_not_bound_to_previous_native_pass_or_current_frame")
                    used_before = sum(step["action_result"] is not None for step in steps[:call_index])
                    require(type(feedback["actions_used"]) is int and feedback["actions_used"] == used_before
                            and type(feedback["actions_remaining"]) is int
                            and feedback["actions_remaining"] == options["max_actions"] - actions - used_before,
                            "recovery_guidance_changed_the_remaining_action_budget")
                    if previous_step["action_result"] is None and (previous_step["error"] or "").startswith("no element matches "):
                        require(feedback["reason"] == "target_resolution", "no_match_guidance_reclassified_the_failed_pass")
                        audit_no_match_recovery_origin(previous_native, child_frame(previous_step["observation_id"]),
                                                       previous_step, platform=call["context"]["platform"])
                bindings = audit_target_bindings(call["context"], frame)
                response_format = _task_plan_response_format(1, target_bindings=bindings)
                count = call["attempts"]
                require(type(count) is int and count > 0, "action_call_attempt_count_invalid")
                records = action_attempt_rows[action_attempt_cursor:action_attempt_cursor + count]
                model_calls.extend(audit_action_attempts(call, frame, records, response_format))
                action_attempt_cursor += count
                action_call_count += 1
                if call["ok"]:
                    require(call["response"]["provider"] == "openai_compatible"
                            and call["response"]["model_name"] == contract["model"]
                            and not call["response"].get("error"), "native_action_response_provider_changed")
                    rebuilt = parse_plan(call["response"]["content"], instruction=task.instruction, max_steps=1, task_id=goal.goal_id)
                    require(rebuilt.model_dump(mode="json") == call["parsed_plan"], "native_action_plan_changed")
                    native_steps.extend(rebuilt.model_dump(mode="json")["steps"])
                    native_source_frames.extend([frame] * len(rebuilt.steps))
            inputs, executor_failures, used_plans = 0, 0, 0
            for step in steps:
                if step["action_result"] is None:
                    continue
                receipt = ActionResult.model_validate(step["action_result"])
                require(receipt.action is not None and not receipt.dry_run
                        and receipt.action.action_type == step["action_type"]
                        and step["resolved"]["action"] == step["action_result"]["action"], "invalid_real_action_receipt")
                matching = [i for i in range(used_plans, len(native_steps))
                            if native_steps[i]["step_id"] == step["step_id"]
                            and native_steps[i]["action_type"] == step["action_type"]
                            and native_steps[i]["description"] == step["description"]]
                require(matching, "step_not_attributable_to_ordered_native_action_plan")
                native_step = native_steps[matching[0]]
                try:
                    validate_target_consistency(PlanStep.model_validate(native_step), native_source_frames[matching[0]])
                except ActionResolutionError as exc:
                    raise AuditError("executed_native_target_conflicted_with_its_original_snapshot") from exc
                used_plans = matching[0] + 1
                raw_check = verifier.check_step(expected_result=native_step.get("expected_result"),
                    action_type=step["action_type"], before=child_frame(step["observation_id"]),
                    after=child_frame(step["after_observation_id"]) if step["after_observation_id"] else None,
                    action_result=receipt)
                require(same_check(step["verification"], raw_check), "raw_model_step_verdict_rewritten")
                inputs += int(receipt.success and step["action_type"] in semantic_input_types())
                executor_failures += int(not receipt.success)
                all_steps.append({"goal_id": goal.goal_id, "stage_key": f"{round_index}:{goal_index}:{goal.goal_id}",
                    "step": step, "action": receipt.action.model_dump(mode="json"),
                    "before_captured_at": child_frame(step["observation_id"]).captured_at.isoformat(),
                    "after_captured_at": child_frame(step["after_observation_id"]).captured_at.isoformat()
                    if step["after_observation_id"] else None})
            post = child_frame(audit_row["post_observation_id"])
            checked = verifier.check_task(task, post)
            require(audit_row["schema_version"] == SEMANTIC_PROTOCOL_VERSION
                    and audit_row["frozen_task"] == task.model_dump(mode="json")
                    and same_check(audit_row["semantic_postcondition_verification"], checked)
                    and audit_row["confirmed_input_count"] == inputs
                    and audit_row["planning_accounting_complete"] is True,
                    "semantic_postcondition_audit_or_input_count_changed")
            if workflow["final_observation_reference"]:
                require(Path(workflow["final_observation_reference"]).resolve()
                        == (child_dir / f"{workflow['final_observation_id']}.json").resolve(),
                        "goal_final_reference_borrowed_from_another_directory")
                latest = child_frame(workflow["final_observation_id"])
            else:
                latest = post
            if recorded["status"] == "succeeded":
                require(checked.passed and inputs > 0 and audit_row["eligible_for_semantic_credit"] is True
                        and child["status"] == "succeeded" and workflow["status"] == "succeeded"
                        and not _rolling_frame_guard(post, latest)
                        and verifier.check_task(task, latest).passed, "unqualified_completed_semantic_goal")
                completed.append(recorded)
            else:
                require(checked.outcome == "failed" and not audit_row["eligible_for_semantic_credit"],
                        "failed_goal_rescued_or_nonliteral_failure_replanned")
            outcomes.append(recorded)
            actions += len([step for step in steps if step["action_result"] is not None])
            plan_calls += len(calls)
            require(workflow["total_actions"] == child.get("action_count", len(steps))
                    == len([step for step in steps if step["action_result"] is not None]), "known_action_counts_disagree")
            stage_reviews.append({"round": round_index, "goal_id": goal.goal_id,
                "stage_key": f"{round_index}:{goal_index}:{goal.goal_id}", "instruction": goal.instruction,
                "expect_text": goal.expect_text, "forbid_text": goal.forbid_text, "status": recorded["status"],
                "confirmed_inputs": inputs, "executor_failures": executor_failures,
                "semantic_postcondition": checked.outcome,
                "raw_step_outcomes": [step["verification"]["outcome"] for step in steps if step["verification"]]})
            workflow_cursor += 1
            if recorded["status"] != "succeeded":
                revision = root / f"revision_observation_{round_index:03d}/observation.json"
                revised_before = evidence.frame(revision)
                require(not _rolling_frame_guard(latest, revised_before)
                        and not verifier.check_task(final_task, revised_before).passed, "caller_final_rescued_failed_semantic_goal")
                latest = revised_before
                failure = recorded
                break
        else:
            final_candidate = root / f"caller_final_round_{round_index:03d}/observation.json"
            final_frame = evidence.frame(final_candidate)
            require(not _rolling_frame_guard(latest, final_frame), "independent_caller_final_not_fresh")
            latest = final_frame
            if not verifier.check_task(final_task, final_frame).passed:
                failure = {"reason": "completed objectives did not satisfy unchanged caller final predicates"}
            else:
                require(round_index == len(raw["plans"]), "semantic_replanned_after_caller_final_already_passed")
    require(rejected_contracts == raw.get("rejected_goal_contracts", []) == typed.rejected_goal_contracts,
            "rejected_contract_ledger_omitted_or_changed")
    if rejected_contracts:
        require(rejected_contracts == evidence.load(root / "semantic_rejected_contracts.json"),
                "rejected_contract_persisted_ledger_changed")
    else:
        require(not (root / "semantic_rejected_contracts.json").exists(), "unassigned_rejected_contract_ledger")
    require(workflow_cursor == len(raw["workflow_results"]) == len(raw["goal_outcomes"])
            and raw["workflow_result"] == raw["workflow_results"][-1]
            and completed == raw["completed_subgoals"] and outcomes == evidence.load(root / "semantic_goal_ledger.json")
            and actions == typed.total_actions and plan_calls == typed.planning_calls, "root_goal_ledger_or_shared_counts_disagree")
    require(action_attempt_cursor == len(action_attempt_rows), "unassigned_or_duplicated_action_attempt_records")
    final_path = Path(typed.final_observation_reference).resolve()
    require(final_path == (root / f"caller_final_round_{len(raw['plans']):03d}/observation.json").resolve(), "caller_final_reference_changed")
    final_check = verifier.check_task(final_task, evidence.frame(final_path))
    require(final_check.passed and same_check(raw["final_verification"], final_check), "immutable_caller_final_not_independently_passed")
    http = transport_audit(evidence, root, model_calls, typed.model_requests, contract)
    monitor = monitor_audit(evidence, root, owner, typed, completed)
    transitions = fixture_audit(evidence, fixture, acceptance, contract, owner, all_steps, stage_reviews)
    return {"passed": True, "source": sources, "actual_owner_child_exit_code": owner["child_exit_code"],
            "actual_model_requests": http, "planning_calls": plan_calls, "known_actions": actions,
            "unconfirmed_action_attempts": 0, "native_semantic_stages": stage_reviews,
            "rejected_pre_input_contracts": rejected_contracts,
            "monitor": monitor, "transition_attribution": transitions,
            "action_output_audit": {"planning_calls": action_call_count,
                                    "attempt_records": action_attempt_cursor,
                                    "format_retry_attempts": action_attempt_cursor - action_call_count,
                                    "bindings_rebuilt_from_saved_snapshots": True,
                                    "native_generation_schema_qualified": True,
                                    "ledger_reference": str(root / "action_planning_attempts.jsonl"),
                                    "scope": "Generation qualification is not execution authorisation. An unused conflicting secondary alias may be schema-ineligible without implying that the legacy adapter rejected its unused field."},
            "postcondition_source": "untrusted_model_prediction",
            "semantic_meaning_review_required": True,
            "literal_matching_limit": "The unchanged Verifier uses casefold substring matching, including possible negation collisions. Bound predicates and attributed events do not by themselves prove arbitrary instruction semantics; manually review each native instruction against its independent transitions.",
            "transport_evidence_scope": "Actual HTTP dispatch counts, saved pre-encoding request messages, archived full snapshot/pixel provenance and original provider replies are bound to pinned client code. High-level caller-intent-only input excludes archived observations; low-level input retains its actual screenshot. Wire-body behavior is separately covered by offline SDK MockTransport tests; this real-run logger did not retain raw wire request bodies.",
            "scope": "Ended controlled local-window run. Native ordered semantic goals, immutable pre-input literal postconditions, actual receipts, independent fixture events and a fresh caller final are separately bound. No general-app success rate is inferred."}


def semantic_input_types():
    return {"click", "double_click", "right_click", "drag", "scroll", "type_text", "key_press", "hotkey"}


def transport_audit(evidence, root, calls, expected_total, contract):
    dispatches = evidence.lines(root / "transport_requests.jsonl")
    cumulative = 0
    for index, dispatch in enumerate(dispatches, 1):
        require(dispatch["dispatch"] == index and dispatch["count_source"] == "httpx_request_event"
                and type(dispatch["requests_made"]) is int and dispatch["requests_made"] >= 0,
                "actual_http_dispatch_inventory_invalid")
        cumulative += dispatch["requests_made"]
        require(dispatch["requests_total"] == cumulative, "actual_http_total_nonmonotonic")
    require(cumulative == expected_total, "actual_http_total_differs_from_root")
    cursor = 0
    for call in calls:
        if call["kind"] == "semantic_decomposition":
            require(type(call.get("messages")) is list, "semantic_conversation_call_lacks_validated_messages")
            messages = call["messages"]
        else:
            messages = [{"role": "system", "content": call["system"]}, *ModelRequest(
                instruction=call["instruction"], context=call["context"], image_path=call["image_path"]).to_messages()]
        audit_message_modality(call, messages)
        expected_response_format = call.get("response_format")
        if expected_response_format is None and call["kind"] == "semantic_decomposition":
            # Older pure helper callers pass the already independently audited
            # system/messages without repeating the protocol object. Its exact
            # public schema text still fixes the same strict transport schema.
            marker = "\nExact response JSON schema (also supplied as response_format):\n"
            require(call["system"].count(marker) == 1, "semantic_transport_schema_not_uniquely_declared")
            schema = json.loads(call["system"].split(marker, 1)[1], object_pairs_hook=unique)
            expected_response_format = {"type": "json_schema", "json_schema": {
                "name": "semantic_subtask_sequence", "strict": True, "schema": schema}}
        require(type(expected_response_format) is dict, "native_call_lacks_independently_validated_transport_schema")
        made, replies = 0, []
        while made < call["requests"]:
            require(cursor < len(dispatches), "native_call_has_no_remaining_http_record")
            dispatch = dispatches[cursor]
            expected_model = (contract.get("text_model", contract["model"])
                              if call["kind"] == "semantic_decomposition" else contract["model"])
            require(dispatch["messages"] == messages and dispatch["provider"] == "openai_compatible"
                    and dispatch["model_name"] == expected_model, "actual_http_prompt_or_call_binding_changed")
            require("response_format" in dispatch and dispatch["response_format"] == expected_response_format,
                    "actual_http_schema_argument_changed")
            require(dispatch.get("error") == "", "actual_http_dispatch_error_cannot_establish_acceptance")
            made += dispatch["requests_made"]
            segment_replies = []
            for response in dispatch["http_responses"]:
                body = response.get("body")
                if "text_model" in contract:
                    require(isinstance(body, dict) and body.get("model") == expected_model,
                            "actual_http_response_role_model_changed")
                for choice in body.get("choices", []) if isinstance(body, dict) else []:
                    content = choice.get("message", {}).get("content")
                    if isinstance(content, str):
                        replies.append(content)
                        segment_replies.append(content)
            if dispatch.get("response") and dispatch["response"].get("content"):
                require(segment_replies and dispatch["response"]["content"] == segment_replies[-1],
                        "actual_http_normalized_reply_differs_from_original_response_body")
            cursor += 1
        require(made == call["requests"], "native_call_http_segment_count_changed")
        if call["response"] and call["response"].get("content"):
            require(replies and call["response"]["content"] == replies[-1], "native_response_borrowed_from_another_call")
    require(cursor == len(dispatches), "unassigned_actual_http_dispatches")
    return cumulative


def monitor_audit(evidence, root, owner, result, completed):
    evidence.bind(root / "run_status.json", scope=evidence.run)
    events = evidence.lines(root / "status_events.jsonl")
    read = read_run_status(root)
    require(read.get("snapshot") is not None and not read.get("diagnostics"), "root_monitor_journal_snapshot_disagree")
    final = read["snapshot"]
    require(final["event"] == "finished" and final["task_status"] == "succeeded"
            and final["actions_used"] == result.total_actions
            and final["planning_pass"] == result.planning_calls
            and final["subgoals_completed"] == len(completed)
            and final["subgoals_total"] >= len(completed), "root_monitor_not_bound_to_completed_ledger")
    identities = [owner["launcher"], *owner["observed_descendants"]]
    matched = [row for row in identities if row["pid"] == final["pid"]]
    require(len(matched) == 1 and final["process_identity"].startswith("windows:"), "actual_trainer_pid_identity_missing")
    saved_time = int(final["process_identity"].split(":", 1)[1]) / 10_000_000 - 11644473600
    require(abs(saved_time - matched[0]["create_time"]) < .001, "actual_cpython_pid_was_reused_or_misattributed")
    require(all(row["subgoals_total"] >= row["subgoals_completed"] for row in events), "monitor_completed_exceeds_declared_total")
    return {"actual_cpython": matched[0], "events": len(events), "terminal_status": final["task_status"],
            "completed_semantic_goals": final["subgoals_completed"], "granularity": "event-driven; detailed inputs remain in child journals"}


def fixture_audit(evidence, fixture, acceptance, contract, owner, steps, stages):
    originals = {}
    for filename in ("independent_state.json", "independent_events.jsonl"):
        path = evidence.external(fixture / filename, "explicit_fixture/" + filename)
        originals[filename] = path.read_bytes()
        archived = evidence.container / "fixture_evidence" / filename
        if archived.exists():
            require(evidence.bind(archived).read_bytes() == originals[filename], "archived_fixture_differs_from_original")
    state = json.loads(originals["independent_state.json"], object_pairs_hook=unique)
    events = [json.loads(row, object_pairs_hook=unique) for row in originals["independent_events.jsonl"].splitlines() if row]
    require(state == acceptance["state"] and events == acceptance["independent_events"], "fixture_originals_differ_from_embedded_observation")
    expected = contract["independent_expected"]
    order = ["project_selected", "theme_selected", "review_accepted", "local_commit"]
    require(state["project"] == expected["project"] and state["colour"] == expected["colour"]
            and state["stage"] == "completed" and state["committed"] is True
            and [event["event"] for event in events if event["event"] in order] == order
            and sum(event["event"] == "review_rejected" for event in events) == expected["review_rejections"],
            "independent_fixture_parameters_sequence_or_recovery_failed")
    requests = sorted((evidence.run / "sky_bridge").glob("*_request.json"))
    receipts, owners = [], []
    require(len(requests) == len(steps), "sky_requests_not_bijective_with_known_runtime_inputs")
    for request_path, step in zip(requests, steps, strict=True):
        request = evidence.load(request_path)
        receipt = evidence.load(request_path.with_name(request_path.name.replace("_request.json", "_receipt.json")))
        require(receipt["id"] == request["id"] and receipt["action"] == request["action"] == step["action"]
                and receipt["transport"] == "Sky click" and type(receipt["success"]) is bool
                and receipt["success"] == step["step"]["action_result"]["success"]
                and request["required_window_title"] == "Week6 Local Lab"
                and timestamp(receipt["finished_utc"]) >= timestamp(receipt["started_utc"]), "sky_input_receipt_or_runtime_binding_changed")
        require(timestamp(step["before_captured_at"]) <= timestamp(receipt["started_utc"]),
                "input_receipt_precedes_its_actual_planning_observation")
        if receipt["success"]:
            require(step["after_captured_at"] is not None
                    and timestamp(receipt["finished_utc"]) <= timestamp(step["after_captured_at"]),
                    "successful_input_lacks_a_strictly_later_actual_post_frame")
        receipts.append(receipt)
        owners.append({"goal_id": step["goal_id"], "stage_key": step["stage_key"]})
    require(receipts == acceptance["input_receipts"], "acceptance_embedded_different_sky_receipts")
    transitions = []
    for event in events:
        if event["event"] not in {*order, "review_rejected"}:
            continue
        when = timestamp(event["utc"])
        matching = [index for index, row in enumerate(receipts) if row["success"]
                    and timestamp(row["started_utc"]) <= when <= timestamp(row["finished_utc"])]
        require(len(matching) == 1 and timestamp(owner["started_utc"]) <= when <= timestamp(owner["finished_utc"]),
                "semantic_transition_not_uniquely_attributed_to_real_owned_input")
        index = matching[0]
        transitions.append({"event": event["event"], "utc": event["utc"],
                            "input_receipt": receipts[index]["id"], **owners[index]})
    require(len({row["input_receipt"] for row in transitions}) == len(transitions), "one_input_crossed_multiple_semantic_transitions")
    completed_keys = {stage["stage_key"] for stage in stages if stage["status"] == "succeeded"}
    require(completed_keys <= {row["stage_key"] for row in transitions}, "completed_goal_has_no_independent_semantic_transition")
    for stage in stages:
        stage["independent_transitions"] = [row["event"] for row in transitions if row["stage_key"] == stage["stage_key"]]
    return transitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument("--owner-harness", type=Path, required=True)
    parser.add_argument("--fixture-directory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "audit_output_exists_no_overwrite")
    evidence = Evidence(args.container)
    try:
        result = audit(args.container, args.owner_harness, args.fixture_directory.resolve(), evidence)
    except (AuditError, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    result.update(schema="week6.semantic_live.independent_audit.v1", container=str(args.container.resolve()),
                  auditor_sha256=sha(Path(__file__)), evidence_sha256=evidence.files,
                  operations={"model_calls": 0, "gui_operations": 0, "experiment_reruns": 0},
                  raw_evidence_modified=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "out": str(args.out), "sha256": sha(args.out),
                      "error": result.get("error", "")}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
