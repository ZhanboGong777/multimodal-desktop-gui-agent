"""Prepare isolated train/development candidate data on CPU, without model calls.

detect freezes image-only snapshots and an unselected semantic review template.
finish consumes a closed external review, preserves complete native targets, and
measures the real local processor prefix/full lengths without truncation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gui_agent.finetuning.candidate_grounding import (
    DetectorConfig,
    center,
    detect_candidates,
    snapshot_hash,
    validate_snapshot,
)
from gui_agent.finetuning.evaluation import canonical_json, file_sha256, strict_json_loads
from gui_agent.finetuning.pointer_protocol import (
    MOUSE_TYPES,
    _candidate_evidence,
    build_pointer_messages,
    map_pointer_actions,
    select_candidates,
    selection_binding,
    validate_pointer_sample,
)
from gui_agent.finetuning.training import (
    assistant_labels,
    ensure_separate_validation,
    load_prepared_records,
)

POLICY = "control_balanced_v2"
COMPACT_POLICIES = ("spatial_v1", "control_balanced_v2", "query_relations_v3")
DEFAULT_PROCESSOR = ROOT / "outputs/week5_improvement_20261009/training_bilingual_003/processor"
DETECTOR_OPTIONS = {
    "max_candidates": 512, "contour_budget": 200, "contour_max_aspect": 64,
    "ocr_scale": 2, "ocr_merge_aligned": True,
}
MAX_SEQUENCE = 1536
MIN_PIXELS, MAX_PIXELS = 100352, 301056
FORBIDDEN = {"test", "official_test", "official-test", "test.jsonl",
             "official_test.jsonl", "official-test.jsonl", "test.zip"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_sha256(path) for path in sorted(set(paths))}


def _sources() -> dict[str, str]:
    return _hashes([Path(__file__), *(ROOT / "src/gui_agent/finetuning" / name for name in (
        "candidate_grounding.py", "pointer_protocol.py", "training.py", "evaluation.py",
        "data.py", "prompts.py", "pointer_candidates_v2.py", "pointer_candidates_v3.py",
    )), *(ROOT / "src/gui_agent/perception" / name for name in (
        "ocr.py", "ui_detection.py", "preprocessing.py",
    )), ROOT / "src/gui_agent/schemas.py"])


def _unchanged(expected: dict[str, str]) -> None:
    for name, sha in expected.items():
        if file_sha256(name) != sha:
            raise ValueError(f"bound input changed: {name}")


def _safe(path: str | Path) -> Path:
    value = Path(path).resolve(strict=True)
    if any(part.casefold() in FORBIDDEN for part in value.parts):
        raise ValueError("official test paths are forbidden before reading records")
    return value


def _cpu() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES to be empty")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized during CPU preparation")


def _failure(output: Path, phase: str, exc: BaseException) -> None:
    _json(output / f"failure_{phase}_{uuid4().hex}.json", {
        "status": "failed", "phase": phase, "at": _now(),
        "error": f"{type(exc).__name__}: {exc}", "accepted_as_success": False,
        "success_receipt_exists": (output / "result.json").exists(),
    })


def _ocr_assets(language: str, directory: str | Path | None) -> dict[str, Any]:
    if not isinstance(language, str) or not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language):
        raise ValueError("OCR language must be a nonempty '+' separated traineddata name list")
    languages = language.split("+")
    if len(set(languages)) != len(languages):
        raise ValueError("OCR language list must not repeat a language")
    if directory is None:
        return {"ocr_language": language, "tessdata_directory": None,
                "traineddata_files_sha256": {},
                "inherited_tessdata_prefix": os.environ.get("TESSDATA_PREFIX")}
    path = _safe(directory)
    if not path.is_dir():
        raise ValueError("tessdata directory must be an existing local directory")
    files = [path / f"{name}.traineddata" for name in languages]
    for file in files:
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"missing or empty local traineddata: {file}")
    return {"ocr_language": language, "tessdata_directory": str(path),
            "traineddata_files_sha256": _hashes(files), "inherited_tessdata_prefix": None}


@contextmanager
def _tessdata_context(assets: dict[str, Any]):
    present, previous = "TESSDATA_PREFIX" in os.environ, os.environ.get("TESSDATA_PREFIX")
    try:
        if assets["tessdata_directory"] is not None:
            os.environ["TESSDATA_PREFIX"] = assets["tessdata_directory"]
        yield
    finally:
        if present:
            os.environ["TESSDATA_PREFIX"] = previous
        else:
            os.environ.pop("TESSDATA_PREFIX", None)


def _input_rows(train: Path, validation: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = {"train": _safe(train), "validation": _safe(validation)}
    before = _hashes(list(paths.values()))
    loaded = {name: load_prepared_records(path) for name, path in paths.items()}
    ensure_separate_validation(loaded["train"], loaded["validation"])
    entries = []
    for split, path in paths.items():
        originals = [strict_json_loads(line) for line in path.read_text("utf-8").splitlines()
                     if line.strip()]
        if len(originals) != len(loaded[split]):
            raise ValueError("loader and original row count differ")
        for original, actual in zip(originals, loaded[split], strict=True):
            if original["actions"] != actual["actions"]:
                raise ValueError("native target must already have its exact prepared representation")
            if not isinstance(original.get("actor_step"), str) or not original["actor_step"].strip():
                raise ValueError("pointer preparation requires a verified current actor_step")
            entries.append({"split": split, "sample": original,
                            "image_path": actual["image_path"],
                            "image_file_sha256": actual["image_file_sha256"]})
    images = _hashes([Path(entry["image_path"]) for entry in entries])
    # Decoded RGB identity and file-container identity are different hashes.
    train_bytes = {r["image_file_sha256"] for r in entries if r["split"] == "train"}
    dev_bytes = {r["image_file_sha256"] for r in entries if r["split"] == "validation"}
    if train_bytes & dev_bytes:
        raise ValueError("training/validation leakage in image file bytes")
    _unchanged(before | images)
    return entries, before | images


def _slots(entry: dict[str, Any]) -> list[dict[str, Any]]:
    row = entry["sample"]
    slots = []
    for index, action in enumerate(row["actions"]):
        kind = action["action_type"]
        if kind not in MOUSE_TYPES:
            continue
        endpoints = [("start_", action["start"]), ("end_", action["end"])] if kind == "drag" else [("", action)]
        for endpoint, point in endpoints:
            slot = {
                "sample_id": row["sample_id"], "action_index": index, "endpoint": endpoint,
                "action_type": kind, "current_subtask": row["actor_step"],
                "target_point": {"x": point["x"], "y": point["y"]},
                "image_file_sha256": entry["image_file_sha256"],
                "snapshot_path": entry["snapshot_path"],
                "snapshot_file_sha256": entry["snapshot_file_sha256"],
                "decision": "pending", "preferred_candidate_id": None,
            }
            if entry.get("compact_policy", "control_balanced_v2") == "query_relations_v3":
                slot["instruction"] = row["instruction"]
            slots.append(slot)
    return slots


def detect_phase(
    train: str | Path, validation: str | Path, output: str | Path, *,
    resume: bool = False, config: DetectorConfig | None = None, detector: Any = None,
    compact_policy: str = POLICY, ocr_language: str = "eng",
    tessdata_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Detect once per distinct file hash; resumable snapshots are immutable."""
    _cpu()
    if compact_policy not in COMPACT_POLICIES:
        raise ValueError("unknown compact candidate policy")
    assets = _ocr_assets(ocr_language, tessdata_dir)
    entries, inputs = _input_rows(Path(train), Path(validation))
    if config is None:
        options = {**DETECTOR_OPTIONS, "ocr_language": ocr_language}
        if compact_policy == "query_relations_v3":
            options["contour_text_lines"] = True
        config = DetectorConfig(**options)
    actual_config = asdict(config)
    if actual_config["ocr_language"] != ocr_language:
        raise ValueError("detector configuration differs from the requested OCR language")
    if compact_policy == "query_relations_v3" and actual_config.get("contour_text_lines") is not True:
        raise ValueError("query_relations_v3 requires contour_text_lines=True")
    if compact_policy == "query_relations_v3" and (
        any(actual_config.get(name) != value for name, value in DETECTOR_OPTIONS.items())
        or actual_config.get("ocr") != "tesseract"
    ):
        raise ValueError("query_relations_v3 requires the fixed raw detector configuration")
    inputs |= assets["traineddata_files_sha256"]
    detector = detector or detect_candidates
    source = _sources()
    output = Path(output).resolve()
    expected = {"schema": "week5.pointer_full.detect.v1", "inputs_sha256": inputs,
                "source_sha256": source, "detector_config": actual_config,
                "compact_policy": compact_policy, **assets,
                "train_rows": sum(e["split"] == "train" for e in entries),
                "validation_rows": sum(e["split"] == "validation" for e in entries)}
    if output.exists():
        if not resume or (output / "result.json").exists():
            raise FileExistsError("refuse to overwrite existing detection output")
        registered = _read(output / "preregistration.json")
        if any(registered.get(key) != value for key, value in expected.items()):
            raise ValueError("resume inputs, sources or detector configuration changed")
    else:
        if resume:
            raise ValueError("resume requires an existing registered detection directory")
        output.mkdir(parents=True)
        _json(output / "preregistration.json", {**expected, "registered_at": _now()})
    try:
        calls, reused = 0, 0
        for row_index, entry in enumerate(entries, 1):
            digest = entry["image_file_sha256"]
            path = output / "snapshots" / f"{digest}.json"
            cache = output / "cache" / f"{digest}.json"
            if cache.exists():
                receipt = _read(cache)
                if (receipt["source_sha256"] != source or receipt["detector_config"] != actual_config
                        or receipt.get("compact_policy") != compact_policy
                        or receipt.get("ocr_assets") != assets):
                    raise ValueError("cached detector source/configuration changed")
                if receipt["snapshot_file_sha256"] != file_sha256(path):
                    raise ValueError("immutable detection cache changed")
                snapshot = _read(path)
                reused += 1
            else:
                if path.exists():
                    raise ValueError("orphan snapshot lacks an immutable cache receipt")
                with _tessdata_context(assets):
                    snapshot = detector(entry["image_path"], config)
                if snapshot["detector_config"].get("ocr_language") != ocr_language:
                    raise ValueError("raw snapshot OCR language differs from the requested producer")
                snapshot.update(compact_policy=compact_policy, **assets)
                validate_snapshot(snapshot, entry["image_path"])
                _json(path, snapshot)
                _json(cache, {"source_sha256": source, "detector_config": actual_config,
                              "compact_policy": compact_policy, "ocr_assets": assets,
                              "image_file_sha256": digest, "snapshot_file_sha256": file_sha256(path)})
                calls += 1
            if (snapshot["compact_policy"] != compact_policy or snapshot["detector_config"] != actual_config
                    or any(snapshot.get(key) != value for key, value in assets.items())):
                raise ValueError("snapshot configuration differs from the registered producer")
            validate_snapshot(snapshot, entry["image_path"])
            _candidate_evidence(snapshot)  # Raw evidence only: no v3 retention before review.
            entry.update(snapshot_path=str(path), snapshot_file_sha256=file_sha256(path),
                         snapshot_sha256=snapshot_hash(snapshot), compact_policy=compact_policy)
            print(canonical_json({"phase": "detect", "row": row_index, "rows": len(entries),
                                  "detector_calls": calls, "cache_reuses": reused,
                                  "candidate_count": len(snapshot["candidates"])}), flush=True)
        _unchanged(inputs | source)
        rows_path = output / "rows_and_snapshots.jsonl"
        _jsonl(rows_path, entries)
        reviews = [slot for entry in entries for slot in _slots(entry)]
        _json(output / "semantic_review_template.json", {
            "schema": "week5.pointer_semantic_review.v1", "status": "draft",
            "dataset_rows_sha256": file_sha256(rows_path),
            "provenance": "Unselected template; external pixel/semantic review required. Not human gold or hit-test evidence.",
            "reviews": reviews,
        })
        artifacts = _hashes([rows_path, output / "semantic_review_template.json",
                             *(output / "snapshots").glob("*.json"),
                             *(output / "cache").glob("*.json")])
        _unchanged(inputs | source | artifacts)
        _cpu()
        result = {**expected, "status": "closed_detection_only", "completed_at": _now(),
                  "rows": len(entries), "unique_images": len({e["image_file_sha256"] for e in entries}),
                  "detector_calls_this_invocation": calls, "cache_reuses_this_invocation": reused,
                  "mouse_review_slots": len(reviews), "artifacts_sha256": artifacts,
                  "automatic_semantic_acceptances": 0, "model_loads": 0,
                  "model_generations": 0, "training_runs": 0, "gui_actions": 0}
        _json(output / "result.json", result)
        return result
    except BaseException as exc:
        _failure(output, "detect", exc)
        raise


def _review_index(review: dict[str, Any], entries: list[dict[str, Any]], rows_sha: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    if review.get("schema") != "week5.pointer_semantic_review.v1" or review.get("status") != "closed":
        raise ValueError("finish requires a closed external semantic review")
    if review.get("dataset_rows_sha256") != rows_sha:
        raise ValueError("semantic review belongs to different frozen dataset rows")
    expected = {(s["sample_id"], s["action_index"], s["endpoint"]): s
                for entry in entries for s in _slots(entry)}
    actual = {}
    for record in review["reviews"]:
        key = (record["sample_id"], record["action_index"], record["endpoint"])
        if type(record["action_index"]) is not int or key not in expected or key in actual:
            raise ValueError("semantic review has duplicate, unknown or invalid slot identity")
        original = expected[key]
        for field in ("action_type", "current_subtask", "target_point", "image_file_sha256",
                      "snapshot_path", "snapshot_file_sha256"):
            if field not in record or record[field] != original[field]:
                raise ValueError(f"semantic review {field} differs from its frozen native slot")
        if "instruction" in original and record.get("instruction") != original["instruction"]:
            raise ValueError("v3 semantic review instruction differs from its frozen native request")
        point = record["target_point"]
        if set(point) != {"x", "y"} or any(type(v) is not int for v in point.values()):
            raise ValueError("review point requires exactly integer x/y")
        if record["decision"] not in {"accepted", "abstain"}:
            raise ValueError("closed review must explicitly accept or abstain for every slot")
        preferred = record["preferred_candidate_id"]
        if record["decision"] == "accepted" and (not isinstance(preferred, str) or not preferred):
            raise ValueError("accepted review requires a preferred original candidate ID")
        if record["decision"] == "abstain" and preferred is not None:
            raise ValueError("abstained review cannot carry a preferred candidate")
        actual[key] = record
    if set(actual) != set(expected):
        raise ValueError("closed semantic review must cover every mouse/drag endpoint slot")
    return actual


def _pointer_row(entry: dict[str, Any], reviews: dict[tuple[Any, ...], dict[str, Any]], evidence: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    row = {**copy.deepcopy(entry["sample"]), "image_path": entry["image_path"],
           "image_file_sha256": entry["image_file_sha256"],
           "candidate_snapshot": _read(Path(entry["snapshot_path"])),
           "candidate_label_evidence": evidence}
    snapshot = row["candidate_snapshot"]
    if file_sha256(entry["snapshot_path"]) != entry["snapshot_file_sha256"] or snapshot_hash(snapshot) != entry["snapshot_sha256"]:
        raise ValueError("frozen candidate snapshot changed")
    validate_snapshot(snapshot, row["image_path"])
    policy = snapshot.get("compact_policy", "spatial_v1")
    if policy not in COMPACT_POLICIES:
        raise ValueError("unknown compact candidate policy")
    if policy == "query_relations_v3":
        context = {"instruction": row["instruction"], "actor_step": row["actor_step"]}
        selected = select_candidates(snapshot, **context)
        row["candidate_selection_evidence"] = selection_binding(snapshot, **context)
    else:
        selected = select_candidates(snapshot)
    indices = {candidate["candidate_id"]: i for i, candidate in enumerate(selected, 1)}
    all_ids = {candidate["candidate_id"] for candidate in snapshot["candidates"]}
    targets, blockers = [], []
    for index, action in enumerate(row["actions"]):
        kind = action["action_type"]
        if kind not in MOUSE_TYPES:
            targets.append(copy.deepcopy(action))
            continue
        target = {"action_type": kind}
        endpoints = [("start_", action["start"]), ("end_", action["end"])] if kind == "drag" else [("", action)]
        for endpoint, point in endpoints:
            record = reviews[(row["sample_id"], index, endpoint)]
            preferred = record["preferred_candidate_id"]
            if record["decision"] == "abstain":
                blockers.append(f"action{index}:{endpoint}semantic_abstention")
                continue
            if preferred not in all_ids:
                raise ValueError("review preferred candidate is absent from the original snapshot")
            if preferred not in indices:
                blockers.append(f"action{index}:{endpoint}preferred_not_in_selected_table")
                continue
            identifier = indices[preferred]
            x, y = center(selected[identifier - 1]["bbox"])
            target.update({endpoint + "candidate": identifier,
                           endpoint + "dx": point["x"] - x, endpoint + "dy": point["y"] - y})
        if kind == "scroll":
            target["scroll_amount"] = action["scroll_amount"]
        targets.append(target)
    if not blockers:
        row["candidate_target"] = targets
        validate_pointer_sample(row)
        mapping = {**context, "expected_selection": row["candidate_selection_evidence"]} if policy == "query_relations_v3" else {}
        if map_pointer_actions(canonical_json(targets), snapshot, row["image_path"], **mapping) != row["actions"]:
            raise ValueError("candidate residual target failed exact native roundtrip")
    return row, blockers


def _list(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _encode(processor: Any, row: dict[str, Any], *, target: bool) -> tuple[str, dict[str, Any]]:
    from PIL import Image

    rendered = processor.apply_chat_template(build_pointer_messages(row, include_target=target),
                                            tokenize=False, add_generation_prompt=not target)
    with Image.open(row["image_path"]) as image:
        encoded = processor(text=[rendered], images=[image.convert("RGB")],
                            padding=False, return_tensors="pt")
    return rendered, {"input_ids": _list(encoded["input_ids"])[0],
                      "image_grid_thw": _list(encoded["image_grid_thw"])}


def _processor(path: Path) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(path), local_files_only=True,
                                        trust_remote_code=False, use_fast=True,
                                        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)


def _finish_phase(
    detection: str | Path, review_path: str | Path, output: str | Path, *,
    processor_path: str | Path = DEFAULT_PROCESSOR, processor_factory: Any = None,
    max_sequence: int = MAX_SEQUENCE,
) -> dict[str, Any]:
    """Measure every prefix and every complete reviewed target on CPU only."""
    _cpu()
    if type(max_sequence) is not int or max_sequence <= 0:
        raise ValueError("max_sequence must be a positive integer")
    detection, review_path = _safe(detection), _safe(review_path)
    closed = _read(detection / "result.json")
    if closed["status"] != "closed_detection_only":
        raise ValueError("detection phase is not complete")
    _unchanged(closed["inputs_sha256"] | closed["source_sha256"] | closed["artifacts_sha256"])
    rows_path = detection / "rows_and_snapshots.jsonl"
    entries = [strict_json_loads(line) for line in rows_path.read_text("utf-8").splitlines()]
    reviews = _review_index(_read(review_path), entries, file_sha256(rows_path))
    processor_path = _safe(processor_path)
    inputs = closed["inputs_sha256"] | closed["artifacts_sha256"] | _hashes([
        detection / "result.json", detection / "preregistration.json", review_path,
        *(path for path in processor_path.rglob("*") if path.is_file()),
    ])
    source = _sources()
    evidence = {"review_path": str(review_path), "review_sha256": file_sha256(review_path)}
    rows = [_pointer_row(entry, reviews, evidence) for entry in entries]
    policies = sorted({row["candidate_snapshot"].get("compact_policy", "spatial_v1") for row, _ in rows})
    _unchanged(inputs | source)
    output = Path(output).resolve()
    try:
        _json(output / "preregistration.json", {
            "schema": "week5.pointer_full.finish.v1", "registered_at": _now(),
            "inputs_sha256": inputs, "source_sha256": source, "rows": len(rows),
            "processor_path": str(processor_path), "processor_use_fast": True,
            "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
            "max_seq_length": max_sequence, "target_protocol": "candidate_compact",
            "prompt_variant": "conditional_actor", "compact_policies": policies,
            "compact_policy": policies[0] if len(policies) == 1 else "mixed",
            "ocr_language": closed["ocr_language"],
            "tessdata_directory": closed["tessdata_directory"],
            "traineddata_files_sha256": closed["traineddata_files_sha256"],
            "detector_config": closed["detector_config"],
        })
        processor = (processor_factory or _processor)(processor_path)
        _cpu()
        visual = {processor.tokenizer.convert_tokens_to_ids(token) for token in (
            "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")}
        visual.discard(None)
        eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos is None:
            raise ValueError("processor lacks the assistant end token")
        decisions, tokens, prepared = [], [], []
        admitted = {"train": [], "validation": []}
        calls = 0
        for entry, (row, blockers) in zip(entries, rows, strict=True):
            text, prefix = _encode(processor, row, target=False)
            calls += 1
            token_row = {"sample_id": row["sample_id"], "split": entry["split"],
                         "prefix_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "prefix": prefix, "full": None, "labels": None}
            decision = {"sample_id": row["sample_id"], "split": entry["split"],
                        "semantic_blockers": blockers, "prefix_length": len(prefix["input_ids"]),
                        "full_length": None, "complete_semantic_mapping": not blockers,
                        "admitted": False, "literal_visibility": [
                            {"action_index": i, "exact_substring_in_goal": a["text"] in row["instruction"],
                             "exact_substring_in_step": a["text"] in row["actor_step"],
                             "unique_literal_contract_proven": False}
                            for i, a in enumerate(row["actions"]) if a["action_type"] == "type_text"],
                        "partial_action_chunk_admitted": False}
            if not blockers:
                _, full = _encode(processor, row, target=True)
                calls += 1
                labels = assistant_labels(full["input_ids"], prefix["input_ids"],
                                          visual_ids=visual, pad_id=processor.tokenizer.pad_token_id)
                if full["image_grid_thw"] != prefix["image_grid_thw"]:
                    raise ValueError("prefix/full visual grids differ")
                if not any(token == eos and label == eos for token, label in
                           zip(full["input_ids"][len(prefix["input_ids"]):],
                               labels[len(prefix["input_ids"]):], strict=True)):
                    raise ValueError("assistant end token is not supervised")
                token_row.update(full=full, labels=labels)
                decision["full_length"] = len(full["input_ids"])
                decision["admitted"] = max(decision["prefix_length"], decision["full_length"]) <= max_sequence
                if decision["admitted"]:
                    admitted[entry["split"]].append(row)
            prepared.append({"split": entry["split"], "sample": row})
            decisions.append(decision)
            tokens.append(token_row)
        _jsonl(output / "all_rows.jsonl", prepared)
        _json(output / "decisions.json", decisions)
        _json(output / "actual_token_ids.json", tokens)
        for split, selected in admitted.items():
            path = output / f"admitted_{split}.jsonl"
            _jsonl(path, selected)
            if selected and len(load_prepared_records(path, target_protocol="candidate_compact")) != len(selected):
                raise ValueError("admitted dataset failed actual trainer loading")
        ensure_separate_validation(admitted["train"], admitted["validation"])
        artifacts = _hashes([output / name for name in (
            "all_rows.jsonl", "decisions.json", "actual_token_ids.json",
            "admitted_train.jsonl", "admitted_validation.jsonl", "preregistration.json",
        )])
        _unchanged(inputs | source | artifacts)
        _cpu()
        split_counts = {}
        for split, selected in admitted.items():
            subset = [d for d in decisions if d["split"] == split]
            split_counts[split] = {"all_rows": len(subset), "semantic_complete": sum(d["complete_semantic_mapping"] for d in subset),
                                   "admitted": len(selected),
                                   "prefix_overflow": sum(d["prefix_length"] > max_sequence for d in subset),
                                   "known_full_overflow": sum(d["full_length"] is not None and d["full_length"] > max_sequence for d in subset),
                                   "full_unavailable": sum(d["full_length"] is None for d in subset)}
        result = {"schema": "week5.pointer_full.finish.v1", "status": "closed_CPU_preparation_only",
                  "completed_at": _now(), "inputs_sha256": inputs, "source_sha256": source,
                  "artifacts_sha256": artifacts, "rows": len(rows), "splits": split_counts,
                  "processor_loads": 1, "actual_encode_calls": calls, "max_seq_length": max_sequence,
                  "compact_policies": policies, "compact_policy": policies[0] if len(policies) == 1 else "mixed",
                  "ocr_language": closed["ocr_language"], "tessdata_directory": closed["tessdata_directory"],
                  "traineddata_files_sha256": closed["traineddata_files_sha256"], "detector_config": closed["detector_config"],
                  "native_targets_preserved": True, "no_truncation": True,
                  "supervised_assistant_end_verified": True, "cuda_initialized": False,
                  "detector_calls": 0, "model_loads": 0, "model_generations": 0,
                  "training_runs": 0, "gui_actions": 0,
                  "literal_contract_limit": "Substring visibility is diagnostic only; no unique literal is inferred or injected from labels.",
                  "training_admission": False, "week5_complete": False, "t04_complete": False}
        _json(output / "result.json", result)
        return result
    except BaseException as exc:
        _failure(output, "finish", exc)
        raise


def finish_phase(
    detection: str | Path, review_path: str | Path, output: str | Path, *,
    processor_path: str | Path = DEFAULT_PROCESSOR, processor_factory: Any = None,
    max_sequence: int = MAX_SEQUENCE,
) -> dict[str, Any]:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _finish_phase(detection, review_path, output, processor_path=processor_path,
                             processor_factory=processor_factory, max_sequence=max_sequence)
    except BaseException as exc:
        if not any(output.glob("failure_finish_*.json")):
            _failure(output, "finish", exc)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    detect = phases.add_parser("detect", help="image-only detection and unselected review template")
    detect.add_argument("--train", required=True)
    detect.add_argument("--validation", required=True)
    detect.add_argument("--output-dir", required=True)
    detect.add_argument("--resume", action="store_true")
    detect.add_argument("--compact-policy", choices=COMPACT_POLICIES, default=POLICY)
    detect.add_argument("--ocr-language", default="eng")
    detect.add_argument("--tessdata-dir")
    finish = phases.add_parser("finish", help="closed external review and CPU processor encoding")
    finish.add_argument("--detection-dir", required=True)
    finish.add_argument("--review", required=True)
    finish.add_argument("--output-dir", required=True)
    finish.add_argument("--processor-path", default=str(DEFAULT_PROCESSOR))
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        if args.phase == "detect":
            result = detect_phase(args.train, args.validation, args.output_dir, resume=args.resume,
                                  compact_policy=args.compact_policy, ocr_language=args.ocr_language,
                                  tessdata_dir=args.tessdata_dir)
        else:
            result = finish_phase(args.detection_dir, args.review, args.output_dir,
                                  processor_path=args.processor_path)
        _unchanged(result["inputs_sha256"] | result["source_sha256"] | result["artifacts_sha256"])
        _cpu()
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"CPU preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(canonical_json({"success": True, "phase": args.phase, "status": result["status"],
                          "post_write_hash_verification_passed": True,
                          "result_sha256": file_sha256(Path(args.output_dir).resolve() / "result.json"),
                          "result_path": str(Path(args.output_dir).resolve() / "result.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
