#!/usr/bin/env python
"""Pair completed full-128 development validation studies; never test or desktop E2E."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gui_agent.finetuning.evaluation import (
    file_sha256,
    safe_prompt_input,
    score_generation,
    value_sha256,
)
from gui_agent.finetuning.prompts import build_user_prompt
from gui_agent.finetuning.training import load_prepared_records

SCOPE = "development_validation_not_independent_test_or_desktop"
SCORING_SOURCES = (
    "src/gui_agent/finetuning/evaluation.py", "src/gui_agent/finetuning/prompts.py",
    "src/gui_agent/finetuning/data.py", "src/gui_agent/finetuning/training.py",
)
IDENTITY_FIELDS = ("sample_id", "group_id", "image_sha256", "image_file_sha256")
TARGET_FIELDS = ("actions", "action_annotations", "image_width", "image_height")
INFERENCE_FIELDS = ("base_model_id", "base_model_revision", "device", "compute_dtype", "load_in_4bit",
                    "attention_implementation", "processor_use_fast", "min_pixels")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def verify_writer_only_training_change(
    archive: str | Path, current: str | Path, historical_sha256: str,
) -> dict[str, Any]:
    archive, current = Path(archive).resolve(), Path(current).resolve()
    archived_sha = file_sha256(archive)
    require(archived_sha == historical_sha256, "legacy training source archive SHA does not match frozen history")
    trees = []
    for path in (archive, current):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        writers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_write_json"]
        require(len(writers) == 1, "exactly one top-level _write_json function required for writer-only verification")
        tree.body = [node for node in tree.body if node is not writers[0]]
        trees.append(ast.dump(tree, include_attributes=False))
    require(trees[0] == trees[1], "training AST changed outside the sole allowed _write_json function")
    return {"mode": "verified_writer_only_change", "allowed_function": "_write_json",
            "historical_sha256": historical_sha256, "archived_sha256": archived_sha,
            "archive_path": str(archive), "current_sha256": file_sha256(current),
            "ast_outside_write_json_identical": True}


def load_study(
    label: str, directory: str | Path, pixels: int = 384, *, legacy_training_source: str | Path | None = None,
) -> dict[str, Any]:
    """Check recorded before/after provenance plus current dataset/images on CPU.

    Historical metadata need not contain later hardening fields. It must still
    prove source/data/artifact stability with its actual recorded evidence.
    Multi-gigabyte weights are not reread or loaded by this comparison.
    """
    directory = Path(directory).resolve()
    evidence_bytes = {name: (directory / name).read_bytes() for name in (
        "metadata.json", "summary.json", "selection.json", "raw.jsonl")}
    evidence_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in evidence_bytes.items()}
    metadata = json.loads(evidence_bytes["metadata.json"].decode("utf-8"))
    summary = json.loads(evidence_bytes["summary.json"].decode("utf-8"))
    frozen = metadata["frozen_before_generation"]
    require(metadata.get("status") == "completed" and summary.get("complete") is True,
            f"{label}: incomplete validation study")
    require(frozen.get("scope") == summary.get("scope") == SCOPE,
            f"{label}: development-validation scope required")
    require(summary.get("planned_samples_per_cell") == 128 and summary.get("interrupted") is False
            and summary.get("provenance_unchanged") is True, f"{label}: full128 unchanged provenance required")
    require(metadata.get("study_id") == value_sha256(frozen), f"{label}: frozen study hash mismatch")
    require(metadata.get("raw_sha256") == evidence_hashes["raw.jsonl"], f"{label}: raw hash mismatch")
    after = metadata.get("provenance_after", {})
    require(not metadata.get("provenance_errors"), f"{label}: provenance verification reported errors")
    for key in ("dataset_sha256", "source_files", "base", "adapter", "processor"):
        require(key in frozen and after.get(key) == frozen[key], f"{label}: before/after provenance mismatch: {key}")
    for key in ("training_config_sha256", "image_file_sha256"):
        if key in frozen:
            require(after.get(key) == frozen[key], f"{label}: hardened before/after provenance mismatch: {key}")
    for role in ("base", "adapter", "processor"):
        manifest = frozen[role]
        require(bool(manifest.get("files")) and manifest.get("sha256") == value_sha256(manifest["files"]),
                f"{label}: invalid recorded {role} artifact hash")
    training_path = "src/gui_agent/finetuning/training.py"
    training_provenance = {"mode": "exact_current_source",
                           "historical_sha256": frozen["source_files"].get(training_path),
                           "current_sha256": file_sha256(REPOSITORY / training_path)}
    for source in SCORING_SOURCES:
        historical = frozen["source_files"].get(source)
        actual = file_sha256(REPOSITORY / source)
        if historical == actual:
            continue
        if source == "src/gui_agent/finetuning/training.py" and legacy_training_source is not None:
            training_provenance = verify_writer_only_training_change(legacy_training_source, REPOSITORY / source, historical)
        else:
            require(False, f"{label}: scoring/prompt/data/training source does not match: {source}")
    config = frozen["training_config"]
    require(all(key in config for key in INFERENCE_FIELDS), f"{label}: inference configuration evidence missing")
    dataset = Path(config["validation_jsonl"]).resolve()
    require(dataset != Path(config["train_jsonl"]).resolve(), f"{label}: validation cannot be train")
    require(file_sha256(dataset) == frozen["dataset_sha256"], f"{label}: current validation input hash changed")
    samples = sorted(load_prepared_records(dataset), key=lambda row: row["sample_id"])
    require(len(samples) == 128, f"{label}: full128 validation input required")
    train_rows = [json.loads(line) for line in Path(config["train_jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(train_rows), f"{label}: current training partition identity unavailable")
    for key in ("sample_id", "group_id", "image_sha256"):
        require(all(row.get(key) for row in train_rows), f"{label}: training identity missing: {key}")
        require(not ({sample[key] for sample in samples} & {row[key] for row in train_rows}),
                f"{label}: train/validation overlap: {key}")
    selection = [{**{key: sample[key] for key in IDENTITY_FIELDS},
                  "source_sample_sha256": value_sha256(sample)} for sample in samples]
    require(selection == frozen["selection"] == json.loads(evidence_bytes["selection.json"].decode("utf-8")),
            f"{label}: selected IDs/groups/images/input hashes differ from full validation")
    # Prepared-data loading checks native dimensions and decoded RGB identity.
    # Check the original container bytes as well, without modifying the image.
    for sample in samples:
        require(file_sha256(sample["image_path"]) == sample["image_file_sha256"],
                f"{label}: current image hash changed: {sample['sample_id']}")
    if "image_file_sha256" in frozen:
        require(frozen["image_file_sha256"] == {sample["sample_id"]: sample["image_file_sha256"] for sample in samples},
                f"{label}: hardened full image map differs from current validation")
    raw = [json.loads(line) for line in evidence_bytes["raw.jsonl"].decode("utf-8").splitlines() if line.strip()]
    verdicts = [row for row in raw if row.get("record_type") == "provenance_verdict"]
    require(len(verdicts) == 1 and verdicts[0].get("unchanged") is True,
            f"{label}: raw provenance verdict missing or changed")
    budgets = frozen["pixel_budgets"]
    require(pixels in budgets and len(set(budgets)) == len(budgets), f"{label}: requested pixel cell unavailable")
    expected_ids = {sample["sample_id"] for sample in samples}
    requests = [row for row in raw if row.get("record_type") == "request"]
    allowed_variants = {f"adapter_original_pixels_{budget}" for budget in budgets}
    require({row["variant"] for row in requests} == allowed_variants,
            f"{label}: raw cells disagree with declared study")
    cells = summary["cells"]
    require(len(cells) == len(budgets) == summary["expected_cells"], f"{label}: cell completeness mismatch")
    by_sample = {sample["sample_id"]: sample for sample in samples}
    for variant in allowed_variants:
        rows = [row for row in requests if row["variant"] == variant]
        declared = [cell for cell in cells if cell["variant"] == variant]
        require(len(rows) == 128 and {row["sample_id"] for row in rows} == expected_ids
                and len(declared) == 1 and declared[0]["requests"] == declared[0]["inference_attempted"] == 128,
                f"{label}: missing or duplicate full128 requests in {variant}")
        for row in rows:
            sample = by_sample[row["sample_id"]]
            target = {key: sample.get(key, []) for key in TARGET_FIELDS}
            require(row.get("study_id") == metadata["study_id"] and row.get("inference_attempted") is True,
                    f"{label}: request provenance/inference missing")
            require(row["group_id"] == sample["group_id"] and row["image_file_sha256"] == sample["image_file_sha256"]
                    and Path(row["image_path"]).resolve() == Path(sample["image_path"]).resolve()
                    and row["instruction"] == sample["instruction"] and row["target"] == target,
                    f"{label}: raw input/target differs from validation: {sample['sample_id']}")
            require(row.get("image_unchanged", True) is True, f"{label}: request image changed")
            if "prompt" in row:
                require(row["prompt"] == build_user_prompt(safe_prompt_input(sample), "original"),
                        f"{label}: request prompt differs from recorded instruction/schema")
    variant = f"adapter_original_pixels_{pixels}"
    selected = {row["sample_id"]: row for row in requests if row["variant"] == variant}
    scores = {sid: score_generation(row["target"]["actions"], row.get("generation", ""),
                                   width=row["target"]["image_width"], height=row["target"]["image_height"],
                                   action_annotations=row["target"].get("action_annotations", []),
                                   status=row["status"]) for sid, row in selected.items()}
    identity = [{**{key: sample[key] for key in IDENTITY_FIELDS},
                 **{key: sample.get(key, []) for key in TARGET_FIELDS}} for sample in samples]
    require(all(file_sha256(directory / name) == digest for name, digest in evidence_hashes.items()),
            f"{label}: original evidence changed during CPU verification")
    require(file_sha256(dataset) == frozen["dataset_sha256"], f"{label}: dataset changed during CPU verification")
    return {"label": label, "directory": str(directory), "variant": variant, "samples": samples,
            "identity": identity, "rows": selected, "scores": scores,
            "frozen": frozen, "metadata": metadata,
            "input_sha256": frozen["dataset_sha256"],
            "training_source_provenance": training_provenance,
            "evidence_sha256": evidence_hashes,
            "legacy_fields_absent": [key for key in ("training_config_sha256", "image_file_sha256") if key not in frozen]}


def transitions(left: list[bool], right: list[bool]) -> dict[str, int]:
    require(len(left) == len(right), "paired denominator mismatch")
    counts = Counter((bool(a), bool(b)) for a, b in zip(left, right, strict=True))
    return {"eligible": len(left), "improved": counts[False, True], "worsened": counts[True, False],
            "both_correct": counts[True, True], "both_incorrect": counts[False, False],
            "net": counts[False, True] - counts[True, False]}


def text_slots(study: dict[str, Any]) -> list[bool]:
    result = []
    for sample in study["samples"]:
        row = study["rows"][sample["sample_id"]]
        score = study["scores"][sample["sample_id"]]
        predictions = score["predicted_actions"] if score["strict_json_valid"] else []
        for index, action in enumerate(sample["actions"]):
            if action["action_type"] != "type_text":
                continue
            predicted = predictions[index:index + 1]
            single = score_generation([action], json.dumps(predicted, ensure_ascii=False),
                                      width=sample["image_width"], height=sample["image_height"], status=row["status"])
            result.append(bool(single["text_exact_correct"]))
    return result


def compare(studies: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(studies) >= 2 and len({study["label"] for study in studies}) == len(studies),
            "at least two distinct study labels required")
    reference = studies[0]
    for study in studies[1:]:
        require(study["identity"] == reference["identity"], "full128 IDs/group/images/actions/annotations differ")
        for key in ("max_new_tokens", "seed", "base", "processor"):
            require(study["frozen"][key] == reference["frozen"][key], f"inference provenance differs: {key}")
        require(study["variant"] == reference["variant"], "comparison pixel budgets differ")
        require(all(study["frozen"]["training_config"][key] == reference["frozen"]["training_config"][key]
                    for key in INFERENCE_FIELDS), "inference loading/processor configuration differs")
    cells = []
    for study in studies:
        scores = study["scores"]
        counts = Counter(str(len(score["predicted_actions"])) if score["strict_json_valid"] else "invalid_or_failed"
                         for score in scores.values())
        cells.append({key: study[key] for key in (
            "label", "directory", "variant", "input_sha256", "evidence_sha256", "legacy_fields_absent", "training_source_provenance")}
                     | {"requests": 128, "study_id": study["metadata"]["study_id"],
                        "adapter_sha256": study["frozen"]["adapter"]["sha256"],
                        "recorded_training_config_sha256": study["frozen"].get("training_config_sha256"),
                        "inference_configuration": {key: study["frozen"]["training_config"][key] for key in INFERENCE_FIELDS},
                        "whole_action_exact": sum(score["strict_action_exact"] for score in scores.values()),
                        "type_sequence_exact": sum(score["action_type_sequence_exact"] for score in scores.values()),
                        "text_exact_correct": sum(score["text_exact_correct"] for score in scores.values()),
                        "text_exact_total": sum(score["text_exact_total"] for score in scores.values()),
                        "prediction_action_count_distribution": dict(counts),
                        "target_action_count_distribution": dict(Counter(str(len(s["actions"])) for s in study["samples"]))})
    pairs = []
    for left, right in combinations(studies, 2):
        ids = [sample["sample_id"] for sample in left["samples"]]
        pairs.append({"left": left["label"], "right": right["label"],
                      **{name: transitions([bool(left["scores"][sid][key]) for sid in ids],
                                           [bool(right["scores"][sid][key]) for sid in ids]) for name, key in (
                          ("whole_action_exact", "strict_action_exact"), ("type_sequence_exact", "action_type_sequence_exact"))},
                      "text_exact_slots": transitions(text_slots(left), text_slots(right)),
                      "instruction_changed_samples": sum(a["instruction"] != b["instruction"] for a, b in zip(left["samples"], right["samples"], strict=True))})
    return {"scope": SCOPE, "sample_count": 128, "identity_sha256": value_sha256(reference["identity"]),
            "provenance_scope": "recorded pre/post model artifacts; current dataset/images and original scorer verified on CPU",
            "instruction_differences_allowed": True, "cells": cells, "paired": pairs,
            "limits": ["Development validation is used for tuning; no independent-test or desktop E2E claim.",
                       "Pairwise changes are descriptive; they do not isolate adapter effects when instructions also differ.",
                       "Historical fields absent in evidence are reported; original evidence is never rewritten."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", required=True, help="independent label=completed_directory; repeat")
    parser.add_argument("--variant", type=int, default=384, choices=(384, 1280))
    parser.add_argument("--variant384", action="store_const", const=384, dest="variant")
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-training-source", help="explicit archived training.py; only a verified _write_json difference is allowed")
    args = parser.parse_args()
    studies = []
    for specification in args.study:
        label, separator, path = specification.partition("=")
        require(bool(separator and label.strip() and path), "--study requires label=directory")
        studies.append(load_study(label.strip(), path, args.variant, legacy_training_source=args.legacy_training_source))
    result = compare(studies)
    output = Path(args.output)
    require(not output.exists(), "comparison output already exists; preserve evidence")
    require(all(not output.resolve().is_relative_to(Path(study["directory"])) for study in studies),
            "output must be outside original evidence directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "scope": SCOPE, "cells": len(studies)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
