#!/usr/bin/env python
"""Development-only full-validation image-resolution ablation; never desktop execution."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gui_agent.finetuning.evaluation import (
    HFConfig,
    HFWorker,
    artifact_manifest,
    distribution,
    file_sha256,
    runtime_versions,
    safe_prompt_input,
    score_generation,
    value_sha256,
)
from gui_agent.finetuning.prompts import build_user_prompt
from gui_agent.finetuning.training import TrainingConfig, load_prepared_records

WATCHED = (
    "scripts/week5_validation_ablation.py",
    "src/gui_agent/finetuning/evaluation.py",
    "src/gui_agent/finetuning/training.py",
    "src/gui_agent/finetuning/prompts.py",
    "src/gui_agent/finetuning/data.py",
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")


def append(path: Path, row: dict[str, Any], records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    records.append(row)


def completed_token_limit(result: dict[str, Any], cap: int) -> bool:
    return (
        result.get("status") == "error"
        and result.get("error_kind") == "token_limit"
        and type(result.get("generated_tokens")) is int
        and result["generated_tokens"] >= cap
    )


def sources() -> dict[str, str]:
    return {name: file_sha256(REPOSITORY / name) for name in WATCHED}


def validate_dataset(dataset: Path, config: TrainingConfig) -> list[dict[str, Any]]:
    expected = Path(config.validation_jsonl or "").resolve()
    if dataset.resolve() != expected:
        raise ValueError("this experiment accepts only the saved validation dataset, never test or train")
    samples = sorted(load_prepared_records(dataset), key=lambda row: row["sample_id"])
    for sample in samples:
        sample["source_sample_sha256"] = value_sha256(sample)
        score_generation(sample["actions"], "", width=sample["image_width"],
                         height=sample["image_height"],
                         action_annotations=sample.get("action_annotations", []))
    if not config.train_jsonl:
        raise ValueError("training identity required")
    train = load_prepared_records(config.train_jsonl)
    for key in ("sample_id", "group_id", "image_sha256"):
        if {row[key] for row in train} & {row[key] for row in samples}:
            raise ValueError(f"train/validation overlap: {key}")
    return samples


def summarize(records: list[dict[str, Any]], *, unchanged: bool) -> dict[str, Any]:
    """Use the original scoring function, retaining failed/not-started denominators."""
    cells = []
    for variant in sorted({r["variant"] for r in records if r["record_type"] == "request"}):
        rows = [r for r in records if r["record_type"] == "request" and r["variant"] == variant]
        totals: Counter = Counter()
        distances, timings = [], []
        sequence_lengths: Counter = Counter()
        for row in rows:
            target = row["target"]
            score = score_generation(
                target["actions"], row.get("generation", ""),
                width=target["image_width"], height=target["image_height"],
                action_annotations=target.get("action_annotations", []),
                status=row["status"] if unchanged and row.get("image_unchanged", True) else "provenance_changed",
            )
            totals.update({key: value for key, value in score.items() if type(value) in (int, bool)})
            if score["strict_json_valid"]:
                sequence_lengths[len(score["predicted_actions"])] += 1
            distances.extend(score["click_normalized_distances_all_targets"])
            if row.get("duration_ms") is not None:
                timings.append(row["duration_ms"])
        rate_pairs = {
            "strict_json": (totals["strict_json_valid"], len(rows)),
            "whole_action_exact": (totals["strict_action_exact"], len(rows)),
            "type_accuracy": (totals["action_type_correct"], totals["action_type_slots"]),
            "type_sequence_exact": (totals["action_type_sequence_exact"], len(rows)),
            "text_exact": (totals["text_exact_correct"], totals["text_exact_total"]),
            "click_bbox_hit": (totals["click_bbox_hits"], totals["click_bbox_eligible"]),
        }
        cells.append({
            "variant": variant, "requests": len(rows),
            "inference_attempted": sum(r.get("inference_attempted", False) for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "metrics": {name: {"correct": a, "total": b, "rate": a / b if b else None}
                        for name, (a, b) in rate_pairs.items()},
            "prediction_action_counts": dict(sequence_lengths),
            "percall_dispatch_to_reply_ms": distribution(timings),
            "distance_all_targets_missing_sqrt2": distribution(distances),
        })
    return {"scope": "development_validation_not_independent_test_or_desktop", "cells": cells,
            "provenance_unchanged": unchanged,
            "warmup_calls": sum(r["record_type"] == "warmup" for r in records)}


def request_record(sample: dict[str, Any], study: str, variant: str) -> dict[str, Any]:
    """A planned request has no measured duration until a real response supplies one."""
    return {
        "record_type": "request", "study_id": study, "variant": variant,
        "sample_id": sample["sample_id"], "group_id": sample["group_id"],
        "image_path": sample["image_path"], "image_file_sha256": sample["image_file_sha256"],
        "instruction": sample["instruction"], "inference_attempted": False,
        "generation": "", "duration_ms": None,
        "target": {key: sample.get(key, []) for key in (
            "actions", "action_annotations", "image_width", "image_height",
        )},
    }


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_new_tokens <= 0:
        raise ValueError("positive timeout and generation cap required")
    budgets = args.pixel_budget or [384, 1280]
    if any(type(n) is not int or n < 128 for n in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("distinct image token budgets >=128 required")
    payload = json.loads(Path(args.training_config).read_text(encoding="utf8"))
    config = TrainingConfig.from_dict(payload)
    samples = validate_dataset(Path(args.dataset), config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    raw_path = output / "raw.jsonl"
    selection = [{key: row[key] for key in (
        "sample_id", "group_id", "image_sha256", "image_file_sha256", "source_sample_sha256",
    )} for row in samples]
    frozen = {
        "scope": "development_validation_not_independent_test_or_desktop",
        "selection_method": "all saved validation samples, sorted by sample_id; no test read",
        "dataset_sha256": file_sha256(args.dataset), "selection": selection,
        "image_file_sha256": {sample["sample_id"]: sample["image_file_sha256"] for sample in samples},
        "training_config_sha256": file_sha256(args.training_config),
        "base": artifact_manifest(config.base_model_path), "adapter": artifact_manifest(args.adapter),
        "processor": artifact_manifest(config.processor_path or config.base_model_path),
        "training_config": config.to_dict(), "pixel_budgets": budgets,
        "max_new_tokens": args.max_new_tokens, "seed": 42,
        "runtime_versions": runtime_versions(), "source_files": sources(),
        "timeout_seconds": args.timeout,
        "recovery_policy": "typed completed token_limit resident; other failures reload; 3 consecutive infra failures abort cell",
    }
    study = value_sha256(frozen)
    metadata = {"study_id": study, "run_id": uuid.uuid4().hex, "frozen_before_generation": frozen,
                "started_at": datetime.now(UTC).isoformat(), "status": "running"}
    write_json(output / "metadata.json", metadata)
    write_json(output / "selection.json", selection)
    interrupted = False
    cleanup_errors = []
    worker = None

    def close_worker() -> None:
        nonlocal worker
        closing, worker = worker, None
        if closing is None:
            return
        try:
            closing.close()
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            cleanup_errors.append(error)
            append(raw_path, {"record_type": "worker_cleanup_error", "error": error}, records)
            if isinstance(exc, KeyboardInterrupt):
                raise

    try:
        for budget in budgets:
            variant = f"adapter_original_pixels_{budget}"
            actual_config = {**config.to_dict(), "max_pixels": budget * 28 * 28}
            worker = None
            consecutive_infrastructure_errors = 0
            try:
                for index, sample in enumerate(samples):
                    row = request_record(sample, study, variant)
                    if consecutive_infrastructure_errors >= 3:
                        append(raw_path, {**row, "status": "not_started", "error": "infrastructure stop"}, records)
                        continue
                    try:
                        if file_sha256(sample["image_path"]) != sample["image_file_sha256"]:
                            raise ValueError("image changed before inference")
                        prompt = build_user_prompt(safe_prompt_input(sample), "original")
                        if worker is None:
                            worker = HFWorker(HFConfig(actual_config, max_new_tokens=args.max_new_tokens),
                                              adapter_path=args.adapter)
                            append(raw_path, {"record_type": "load", "variant": variant, **worker.ready}, records)
                            warm = worker.request(samples[0]["image_path"],
                                                  build_user_prompt(safe_prompt_input(samples[0])),
                                                  timeout_seconds=args.timeout)
                            append(raw_path, {"record_type": "warmup", "variant": variant, **warm}, records)
                            if warm["status"] != "ok" and not completed_token_limit(warm, args.max_new_tokens):
                                raise RuntimeError(f"warmup failed: {warm.get('error')}")
                        row["inference_attempted"] = True
                        append(raw_path, {"record_type": "request_start", "variant": variant,
                                          "sample_id": sample["sample_id"]}, records)
                        result = worker.request(sample["image_path"], prompt, timeout_seconds=args.timeout)
                        row.update(result, prompt=prompt)
                        row["image_unchanged"] = file_sha256(sample["image_path"]) == sample["image_file_sha256"]
                        if result["status"] == "ok" or completed_token_limit(result, args.max_new_tokens):
                            consecutive_infrastructure_errors = 0
                        else:
                            consecutive_infrastructure_errors += 1
                            close_worker()
                    except KeyboardInterrupt:
                        if "status" not in row:
                            row.update(status="cancelled", error="KeyboardInterrupt")
                        raise
                    except Exception as exc:  # noqa: BLE001 - retain failure evidence
                        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                        consecutive_infrastructure_errors += 1
                        close_worker()
                    finally:
                        append(raw_path, row, records)
                    print(f"{variant} {index + 1}/{len(samples)}", flush=True)
            finally:
                close_worker()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        completed = {(row["variant"], row["sample_id"]) for row in records if row["record_type"] == "request"}
        for budget in budgets:
            variant = f"adapter_original_pixels_{budget}"
            for sample in samples:
                if (variant, sample["sample_id"]) not in completed:
                    append(raw_path, {**request_record(sample, study, variant), "status": "not_started",
                                      "error": "interrupted before request" if interrupted else "run ended before request"}, records)
        # A post-run read failure must invalidate scores, not discard the ledger.
        after, provenance_errors = {}, []
        checks = {
            "dataset_sha256": lambda: file_sha256(args.dataset),
            "source_files": sources,
            "image_file_sha256": lambda: {sample["sample_id"]: file_sha256(sample["image_path"]) for sample in samples},
            "training_config_sha256": lambda: file_sha256(args.training_config),
            "base": lambda: artifact_manifest(config.base_model_path),
            "adapter": lambda: artifact_manifest(args.adapter),
            "processor": lambda: artifact_manifest(config.processor_path or config.base_model_path),
        }
        for key, check in checks.items():
            try:
                after[key] = check()
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - finalize evidence despite verification failures
                interrupted |= isinstance(exc, KeyboardInterrupt)
                after[key] = None
                provenance_errors.append({"check": key, "error": f"{type(exc).__name__}: {exc}"})
        unchanged = all(after[key] == frozen[key] for key in after)
        append(raw_path, {"record_type": "provenance_verdict", "unchanged": unchanged,
                          "errors": provenance_errors}, records)
        summary = summarize(records, unchanged=unchanged)
        summary["interrupted"] = interrupted
        summary["planned_samples_per_cell"] = len(samples)
        summary["expected_cells"] = len(budgets)
        summary["provenance_errors"] = provenance_errors
        summary["worker_cleanup_errors"] = cleanup_errors
        summary["unmeasured_samples"] = sum(not row["inference_attempted"] for row in records if row["record_type"] == "request")
        summary["complete"] = not interrupted and not cleanup_errors and len(summary["cells"]) == len(budgets) and all(
            c["requests"] == len(samples) and c["inference_attempted"] == len(samples)
            for c in summary["cells"]
        )
        write_json(output / "summary.json", summary)
        metadata.update({"status": "completed" if summary["complete"] and unchanged else "incomplete",
                         "finished_at": datetime.now(UTC).isoformat(), "provenance_after": after,
                         "provenance_errors": provenance_errors,
                         "raw_sha256": file_sha256(raw_path)})
        write_json(output / "metadata.json", metadata)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["complete"] and unchanged else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixel-budget", type=int, action="append")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
