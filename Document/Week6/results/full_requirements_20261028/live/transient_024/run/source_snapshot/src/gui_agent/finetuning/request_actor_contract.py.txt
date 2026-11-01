"""Versioned train-only actor context from concordant original request spans.

The editable published request export is provenance, not a signed pre-action
timestamp or independent planner gold. This module does not admit candidates,
literal targets or token lengths, and never consumes dev/test or predictions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from . import actor_contract as legacy
from .data import _actions
from .diagnostics import load_development_rows
from .evaluation import file_sha256, strict_json_loads, value_sha256

VERSION = "week5.screenagent.request-actor-train.v1"
PLAN_SCHEMA = "week5.request-actor-train-plan.v1"
POLICY = "two concordant fixed-position original send_prompt declarations; absent metadata explicit"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
V1_TRAIN_PATH = REPOSITORY_ROOT / "data/processed/week5_screenagent_20261009/train.jsonl"
RAW_TRAIN_ROOT = REPOSITORY_ROOT / "data/raw/screenagent/source/data/ScreenAgent/train"
TRAIN_ROWS = 696
FORBIDDEN_PARTS = legacy.FORBIDDEN_PARTS | {
    "dev", "development", "validation", "val", "dev.jsonl", "validation.jsonl", "val.jsonl",
}
ZERO_BUDGETS = ("model_calls", "processor_calls", "detector_calls", "GUI_calls")
DERIVED_FIELDS = {"instruction_source", "actor_step", "actor_step_source"}


def _refuse(reason: str) -> dict[str, Any]:
    return {"eligible": False, "reason": reason}


def verify_request_actor_request(
    raw: Mapping[str, Any], width: int, height: int,
) -> dict[str, Any]:
    """Derive a step only from request declarations, never targets or a plan item.

    Absent independent current_task is supported by this new version. A present
    empty, invalid or contradictory field is never overwritten. The legacy
    verifier receives a copy only after span selection, for fixed goal/dimension
    and exclusive actor-family checks; its original policy is unchanged.
    """
    prompt = raw.get("send_prompt")
    if not isinstance(prompt, str) or not prompt:
        return _refuse("missing_original_send_prompt")
    lines = prompt.splitlines()
    if len(lines) < 6 or lines[0] not in legacy.TEMPLATES:
        return _refuse("unsupported_actor_template")
    active = [index for index, line in enumerate(lines) if "现在的子任务是" in line]
    operations = [index for index, line in enumerate(lines) if line == legacy.OPERATIONS]
    if len(active) != 2 or len(operations) != 1:
        return _refuse("missing_or_ambiguous_activity_declarations")
    first, last = active
    final = max(index for index, line in enumerate(lines) if line)
    if not (1 < first < operations[0] < last == final):
        return _refuse("activity_declarations_outside_actor_positions")
    if legacy.PLAN_LINE not in lines[2:first]:
        return _refuse("unsupported_actor_template")
    front, tail = lines[first], lines[last]
    if not (front.startswith(legacy.ACTIVE_PREFIX) and front.endswith(legacy.ACTIVE_END)
            and tail.startswith(legacy.ACTIVE_PREFIX) and tail.endswith(legacy.FINAL_END)):
        return _refuse("unsupported_activity_declaration_syntax")
    step = front[len(legacy.ACTIVE_PREFIX):-len(legacy.ACTIVE_END)]
    tail_step = tail[len(legacy.ACTIVE_PREFIX):-len(legacy.FINAL_END)]
    if step != tail_step:
        return _refuse("conflicting_request_activity_declarations")
    if not step.strip() or any(character in step for character in "\r\n\x00"):
        return _refuse("empty_or_invalid_request_activity")
    independent_present = "current_task" in raw
    if independent_present:
        current = raw["current_task"]
        if not isinstance(current, str) or not current.strip():
            return _refuse("empty_or_invalid_independent_current_task")
        if any(character in current for character in "\r\n\x00"):
            return _refuse("unsupported_multiline_current_task")
        if current != step:
            return _refuse("request_activity_differs_from_current_task")
    shadow = dict(raw)
    shadow["current_task"] = step
    decision = legacy.verify_actor_request(shadow, width, height)
    if not decision["eligible"]:
        return decision
    return {**decision, "reason": "verified_original_request_actor_train",
            "version": VERSION, "independent_current_task_present": independent_present,
            "independent_current_task_status": "provided_concordant" if independent_present else "not_provided",
            "step_selected_from": "two_fixed_original_send_prompt_declarations",
            "legacy_contract_result_not_claimed": True}


def _forbid_nontrain_path(path: Path) -> None:
    if any(part.casefold() in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError("test/dev/validation path forbidden before reading data")


def _repository_path(path: str | Path, *, strict: bool = False) -> Path:
    value = Path(path)
    _forbid_nontrain_path(value)
    resolved = value.resolve(strict=strict)
    _forbid_nontrain_path(resolved)
    if not resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError("path escapes repository")
    return resolved


def _sha_text(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("registered SHA256 must be 64 hex characters")
    return value.lower()


def _registered_plan(plan_path: str | Path, digest: str, logical_split: str) -> tuple[Path, dict, str]:
    if logical_split != "train":
        raise ValueError("logical split must be train before reading any plan or data")
    expected = _sha_text(digest)
    path = _repository_path(plan_path, strict=True)
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError("registered plan SHA256 mismatch")
    plan = strict_json_loads(content.decode("utf-8-sig"))
    if not isinstance(plan, dict) or (plan.get("schema"), plan.get("status"),
                                    plan.get("contract_version"), plan.get("logical_split")) != (
            PLAN_SCHEMA, "registered", VERSION, "train"):
        raise ValueError("registered train-only request actor plan required")
    info, budgets = plan.get("train_input"), plan.get("budgets")
    if not isinstance(info, dict) or not isinstance(budgets, dict):
        raise TypeError("registered source input and budgets required")
    if type(info.get("rows")) is not int or info["rows"] != TRAIN_ROWS or (
            type(budgets.get("source_rows")) is not int or budgets["source_rows"] != TRAIN_ROWS):
        raise ValueError("registered source row budget must preserve the whole V1 train denominator")
    if any(type(budgets.get(key)) is not int or budgets[key] != 0 for key in ZERO_BUDGETS):
        raise ValueError("source preparation requires zero model/processor/detector/GUI budgets")
    _sha_text(info.get("sha256"))
    for field in (info.get("path"), plan.get("raw_train_root"), plan.get("output_dir")):
        if not isinstance(field, str) or not Path(field).is_absolute():
            raise ValueError("registered absolute input/root/output paths required")
    train = _repository_path(info["path"])
    root = _repository_path(plan["raw_train_root"])
    output = _repository_path(plan["output_dir"])
    if train != V1_TRAIN_PATH.resolve() or root != RAW_TRAIN_ROOT.resolve():
        raise ValueError("only the original registered V1 train and official raw train root are accepted")
    if output.is_relative_to(root) or output.is_relative_to(train.parent):
        raise ValueError("output must not be inside the frozen source datasets")
    if output.exists():
        raise ValueError("output already exists; collision refused before reading source data")
    return path, plan, actual


def _conversion_sources() -> dict[str, str]:
    sources = legacy._conversion_sources()
    sources[str(Path(__file__).resolve())] = file_sha256(__file__)
    cli = Path(__file__).resolve().parents[3] / "scripts/week5_prepare_request_actor_train.py"
    if cli.is_file():
        sources[str(cli)] = file_sha256(cli)
    return sources


def _raw_path(root: Path, row: Mapping[str, Any]) -> Path:
    name = row.get("record_path")
    if not isinstance(name, str) or not name:
        raise ValueError("original raw record_path required")
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in name or relative.suffix != ".json":
        raise ValueError("unsafe source record path")
    _forbid_nontrain_path(Path(str(relative)))
    path = _repository_path(root / str(relative), strict=True)
    if not path.is_relative_to(root):
        raise ValueError("raw source path escapes training root")
    return path


def _write_new(path: Path, text: str, *, owned_files: set[Path] | None = None) -> None:
    # Refuse a replaced output parent before any partial dataset is written.
    if _repository_path(path) != path:
        raise ValueError("output path changed before writing")
    with path.open("x", encoding="utf-8") as handle:
        if owned_files is not None:
            owned_files.add(path)
        handle.write(text)


def prepare_request_actor_train(
    plan_path: str | Path, plan_sha256: str, *, logical_split: str = "train",
) -> dict[str, Any]:
    """Execute a registered, source-only whole-train derivation, retaining refusals.

    This accepts one fixed V1 train source only. It is not the candidate/literal/
    length admission stage and does not register or start any training campaign.
    """
    registered, plan, plan_hash = _registered_plan(plan_path, plan_sha256, logical_split)
    train = _repository_path(plan["train_input"]["path"], strict=True)
    root = _repository_path(plan["raw_train_root"], strict=True)
    output = _repository_path(plan["output_dir"])
    dependencies = _conversion_sources()
    content = train.read_bytes()
    train_hash = hashlib.sha256(content).hexdigest()
    if train_hash != _sha_text(plan["train_input"]["sha256"]):
        raise ValueError("registered V1 train SHA256 mismatch")
    original = [strict_json_loads(line) for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    if len(original) != TRAIN_ROWS:
        raise ValueError("whole V1 train row denominator differs from registered source budget")
    # Validate every declared path and split before the shared verifier opens images.
    image_paths = {}
    raw_paths = {}
    for row in original:
        if not isinstance(row, dict) or row.get("source_split") != "train":
            raise ValueError("only original training-source rows are accepted")
        if DERIVED_FIELDS & row.keys():
            raise ValueError("use original V1 data without derived actor/goal context")
        name = row.get("image_path")
        if not isinstance(name, str) or not name:
            raise ValueError("original prepared image_path required")
        image = Path(name)
        image = image if image.is_absolute() else train.parent / image
        image_paths[row["sample_id"]] = _repository_path(image, strict=True)
        raw_paths[row["sample_id"]] = _raw_path(root, row)
    checked = {row["sample_id"]: row for row in load_development_rows(train, split="train")}
    records, derived, reasons = [], [], Counter()
    for row in original:
        current = checked[row["sample_id"]]
        if value_sha256(row) != current["source_sample_sha256"]:
            raise ValueError("prepared row changed after initial validation")
        if "image_file_sha256" in row and row["image_file_sha256"] != current["image_file_sha256"]:
            raise ValueError("prepared image container differs from original identity")
        path, raw, digest = legacy.read_actor_source(root, row)
        if path != raw_paths[row["sample_id"]]:
            raise ValueError("raw source path changed after path validation")
        # Refuse non-train or escaped source-image paths before the image helper opens them.
        image_name = raw.get("saved_image_name")
        if not isinstance(image_name, str) or not image_name or Path(image_name).name != image_name or ":" in image_name:
            raise ValueError("unsafe source image name")
        declared_image = _repository_path(path.parent / "images" / image_name, strict=True)
        if not declared_image.is_relative_to(root):
            raise ValueError("source screenshot escapes training root")
        # Native image identity is source-bound independently of the request decision.
        image = legacy._source_image(root, path, raw, current)
        if image != declared_image:
            raise ValueError("source screenshot path changed after path validation")
        record = {"split": "train", "sample_id": row["sample_id"], "group_id": row["group_id"],
                  "source_sample_sha256": current["source_sample_sha256"],
                  "raw_record_path": str(path), "raw_record_sha256": digest,
                  "prepared_image_path": str(image_paths[row["sample_id"]]),
                  "prepared_image_file_sha256": current["image_file_sha256"],
                  "image_rgb_sha256": current["image_sha256"],
                  "source_image_path": str(image), "source_image_sha256": current["image_file_sha256"],
                  "whole_actions_sha256": value_sha256(row["actions"]),
                  "annotations_sha256": value_sha256(row.get("action_annotations", [])),
                  "independent_current_task_present": "current_task" in raw}
        # Check complete native supervision for every original row, including
        # request refusals. Labels never choose or repair the input spans.
        actions, annotations = _actions(raw, "screenagent", row["image_width"], row["image_height"])
        if actions != row["actions"] or annotations != row.get("action_annotations", []):
            raise ValueError("prepared whole labels/annotations differ from raw actor source")
        record["whole_labels_and_annotations_raw_verified"] = True
        decision = verify_request_actor_request(raw, row["image_width"], row["image_height"])
        record.update(eligible=decision["eligible"], reason=decision["reason"])
        if decision["eligible"]:
            new = copy.deepcopy(row)
            new["instruction"] = decision["instruction"]
            new["instruction_source"] = {
                "version": VERSION, "selected_field": decision["goal_field"],
                "policy": "first nonempty task_prompt, task_prompt_en, instruction; verbatim, never target-dependent",
                "raw_record_path": row["record_path"], "raw_record_sha256": digest,
            }
            new["actor_step"] = decision["actor_step"]
            new["actor_step_source"] = {
                "version": VERSION, "policy": POLICY, "raw_record_path": row["record_path"],
                "raw_record_sha256": digest,
                **{key: decision[key] for key in ("request_prompt_sha256", "declaration_line_indices",
                    "request_template", "source_revision", "independent_current_task_present",
                    "independent_current_task_status", "step_selected_from", "legacy_contract_result_not_claimed")},
                "chronology_claim": "Editable original request export; no signed pre-response timestamp",
            }
            preserved = {key: value for key, value in new.items()
                         if key not in DERIVED_FIELDS and key != "instruction"}
            if preserved != {key: value for key, value in row.items() if key != "instruction"}:
                raise ValueError("whole original row fields were not preserved")
            record.update(actor_step_sha256=value_sha256(decision["actor_step"]),
                          independent_current_task_status=decision["independent_current_task_status"],
                          declaration_line_indices=decision["declaration_line_indices"],
                          request_prompt_sha256=decision["request_prompt_sha256"])
            derived.append(new)
        records.append(record)
        reasons[decision["reason"]] += 1

    def verify_sources() -> None:
        for path in (registered, train, root, output):
            if _repository_path(path) != path:
                raise ValueError("registered source/output path changed during conversion")
        if file_sha256(registered) != plan_hash or file_sha256(train) != train_hash:
            raise ValueError("registered plan or prepared train changed during conversion")
        for record in records:
            for path_field, digest_field in (("raw_record_path", "raw_record_sha256"),
                    ("prepared_image_path", "prepared_image_file_sha256"),
                    ("source_image_path", "source_image_sha256")):
                path = Path(record[path_field])
                if _repository_path(path, strict=True) != path:
                    raise ValueError("source path changed during conversion")
                if file_sha256(path) != record[digest_field]:
                    raise ValueError("raw source or screenshot changed during conversion")
        if _conversion_sources() != dependencies:
            raise ValueError("conversion source code changed during conversion")

    verify_sources()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    owned_files: set[Path] = set()
    try:
        train_output = output / "train.jsonl"
        _write_new(train_output, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                                   allow_nan=False) + "\n" for row in derived),
                   owned_files=owned_files)
        outputs = {"train": {"path": str(train_output), "sha256": file_sha256(train_output),
                             "samples": len(derived), "groups": len({row["group_id"] for row in derived}),
                             "excluded_samples": len(original) - len(derived)}}
        manifest = {
            "version": VERSION, "policy": POLICY, "logical_split": "train",
            "status": "completed_source_contract_preparation", "original_rows": len(original),
            "original_groups": len({row["group_id"] for row in original}),
            "registered_plan": {"path": str(registered), "sha256": plan_hash},
            "input_data": {"train": {"path": str(train), "sha256": train_hash, "rows": len(original)}},
            "raw_training_root": str(root), "records": records, "reason_counts": dict(reasons),
            "outputs": outputs, "conversion_dependencies_sha256": dependencies,
            "input_contract": "original goal + two verified original request declarations + native screenshot",
            "scope": "original training-source context/image contract only; not candidate/literal/length admission",
            "excluded_model_input_fields": list(legacy.EXCLUDED_INPUT_FIELDS),
            "labels_and_annotations_preserved_and_raw_verified": True,
            "identities_groups_images_dimensions_preserved": True, "inputs_unchanged_during_conversion": True,
            "independent_current_task_absence_count": sum(not r["independent_current_task_present"] for r in records),
            "legacy_contract_result_not_claimed": True, "new_sources_are_independent_holdout": False,
            "candidate_semantic_admission_performed": False, "literal_admission_performed": False,
            "token_length_preflight_performed": False, "training_admission_performed": False,
            "development_data_consumed": False, "test_data_consumed": False, "predictions_consumed": False,
            "model_or_processor_loaded": False, "detector_called": False, "GUI_called": False,
            "chronology_limit": "Original request exports are editable and lack signed capture timestamps. "
                                "Concordant spans do not prove independent planner gold or immutable pre-action timing.",
            "deployment_requirement": "A planner or operator supplies current-step context; never infer it "
                                      "from the current response, gold chunk, action description or plan item.",
            "W5_complete": False, "T04_closed": False,
        }
        verify_sources()
        _write_new(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   owned_files=owned_files)
        verify_sources()
        if file_sha256(train_output) != outputs["train"]["sha256"]:
            raise ValueError("output data changed during conversion")
    except (Exception, KeyboardInterrupt) as error:
        if manifest_path in owned_files and _repository_path(manifest_path) == manifest_path:
            manifest_path.unlink(missing_ok=True)
        failure = {"status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error": f"{type(error).__name__}: {error}", "success_manifest_valid": False}
        try:
            if _repository_path(output) == output:
                _write_new(output / "failure.json", json.dumps(failure, ensure_ascii=False) + "\n")
        except (OSError, ValueError):
            pass
        raise
    return manifest
