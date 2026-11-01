#!/usr/bin/env python
"""Real HF generation on frozen train16 capacity and validation16 development rows.

No desktop actions, test-set selection, repaired JSON or fabricated coordinates.
The CLI always uses HFWorker; CPU regression fixtures replace it only in tests.
"""

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

from gui_agent.finetuning.action_constraints import OUTPUT_CONSTRAINTS, constraint_descriptor
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
from gui_agent.finetuning.prompts import TRAINING_PROMPT_VARIANTS, build_user_prompt, prompt_sha256
from gui_agent.finetuning.training import TrainingConfig, _write_json, load_prepared_records

WATCHED = (
    "scripts/week5_targeted_probe.py", "src/gui_agent/finetuning/evaluation.py",
    "src/gui_agent/finetuning/training.py", "src/gui_agent/finetuning/prompts.py",
    "src/gui_agent/finetuning/data.py", "src/gui_agent/schemas.py",
    "src/gui_agent/datasets/schemas.py",
    "src/gui_agent/finetuning/action_constraints.py", "requirements-week5.txt",
)


def sources() -> dict[str, str]:
    return {name: file_sha256(REPOSITORY / name) for name in WATCHED}


def append(path: Path, row: dict[str, Any], records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    records.append(row)


def completed_token_limit(result: dict, cap: int) -> bool:
    return (result.get("status") == "error" and result.get("error_kind") == "token_limit"
            and type(result.get("generated_tokens")) is int and result["generated_tokens"] >= cap)


def load_selection(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Verify the pre-generation manifest and all actual original screenshots."""
    manifest = json.loads(Path(args.selection_manifest).read_text(encoding="utf8"))
    if manifest.get("based_on_model_outputs") is not False or manifest.get("test_read") is not False:
        raise ValueError("a frozen train/validation-only selection is required")
    all_samples = []
    for split, supplied in [("train16", args.train_jsonl), ("validation16", args.validation_jsonl)]:
        spec = manifest["selections"][split]
        path = Path(supplied).resolve()
        if path != Path(spec["jsonl"]["path"]).resolve() or file_sha256(path) != spec["jsonl"]["sha256"]:
            raise ValueError(f"{split} differs from the frozen dataset")
        source = spec["source"]
        if file_sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"{split} source JSONL changed")
        samples = load_prepared_records(path)
        expected = {row["sample_id"]: row for row in spec["samples"]}
        if len(samples) != 16 or len(expected) != 16 or {r["sample_id"] for r in samples} != set(expected):
            raise ValueError(f"{split} must contain exactly the frozen 16 sample IDs")
        for sample in samples:
            identity = expected[sample["sample_id"]]
            for actual, recorded in [("group_id", "group_id"), ("image_sha256", "image_rgb_sha256"),
                                     ("image_file_sha256", "image_file_sha256"), ("source", "source"),
                                     ("source_split", "source_split")]:
                if sample[actual] != identity[recorded]:
                    raise ValueError(f"{sample['sample_id']}: frozen identity changed: {actual}")
            if Path(sample["image_path"]).resolve() != Path(identity["image_path"]).resolve():
                raise ValueError("actual screenshot path changed")
            score_generation(sample["actions"], "", width=sample["image_width"],
                             height=sample["image_height"], action_annotations=sample.get("action_annotations", []))
            sample.update(split=split, stratum=identity["stratum"])
        all_samples.extend(sorted(samples, key=lambda row: row["sample_id"]))
    train = [r for r in all_samples if r["split"] == "train16"]
    val = [r for r in all_samples if r["split"] == "validation16"]
    for key in ("sample_id", "group_id", "image_sha256"):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f"train/validation overlap: {key}")
    return manifest, all_samples


def request_record(sample: dict, study: str, prompt: str, output_constraint: dict | None = None) -> dict:
    return {
        "record_type": "request", "study_id": study, "split": sample["split"],
        "scope": "train_only_capacity_not_heldout" if sample["split"] == "train16" else "development_validation_not_test",
        "stratum": sample["stratum"], "sample_id": sample["sample_id"], "group_id": sample["group_id"],
        "image_path": sample["image_path"], "image_sha256": sample["image_sha256"],
        "image_file_sha256": sample["image_file_sha256"], "instruction": sample["instruction"],
        "prompt": prompt, "inference_attempted": False, "generation": "", "duration_ms": None,
        "target": {key: sample.get(key, []) for key in ("actions", "action_annotations", "image_width", "image_height")},
        "output_constraint": output_constraint or constraint_descriptor("none"),
    }


def score_row(row: dict, *, unchanged: bool) -> dict:
    target = row["target"]
    score = score_generation(target["actions"], row.get("generation", ""),
                             width=target["image_width"], height=target["image_height"],
                             action_annotations=target.get("action_annotations", []),
                             status=row["status"] if unchanged and row.get("image_unchanged", True) else "provenance_changed")
    # Preserve the original aligned text metric. The capacity gate additionally
    # requires a complete type sequence, so extra/omitted actions cannot pass.
    score["row_all_text_exact"] = bool(score["text_exact_total"] and score["action_type_sequence_exact"]
                                       and score["text_exact_correct"] == score["text_exact_total"])
    return score


def summarize(records: list[dict], *, unchanged: bool) -> dict:
    result = {"scope": "train_capacity_and_development_validation_not_test_or_desktop",
              "provenance_unchanged": unchanged, "splits": []}
    for split in ("train16", "validation16"):
        rows = [r for r in records if r["record_type"] == "request" and r["split"] == split]
        totals = Counter()
        scores = []
        for row in rows:
            score = score_row(row, unchanged=unchanged)
            totals.update({k: v for k, v in score.items() if type(v) in (bool, int)})
            scores.append({"sample_id": row["sample_id"], "stratum": row["stratum"], "status": row["status"],
                           "target_action_count": len(row["target"]["actions"]), **score})
        pairs = {"strict_json": (totals["strict_json_valid"], len(rows)),
                 "whole_action_exact": (totals["strict_action_exact"], len(rows)),
                 "type_sequence_exact": (totals["action_type_sequence_exact"], len(rows)),
                 "text_exact": (totals["text_exact_correct"], totals["text_exact_total"]),
                 "row_all_text_exact": (totals["row_all_text_exact"], sum(s["text_exact_total"] > 0 for s in scores))}
        result["splits"].append({"split": split, "requests": len(rows),
            "inference_attempted": sum(r["inference_attempted"] for r in rows),
            "statuses": dict(Counter(r["status"] for r in rows)),
            "metrics": {k: {"correct": a, "total": b, "rate": a / b if b else None} for k, (a, b) in pairs.items()},
            "prediction_action_counts": dict(Counter(len(s["predicted_actions"]) for s in scores if s["strict_json_valid"])),
            "percall_dispatch_reply_ms": distribution([r["duration_ms"] for r in rows if r.get("duration_ms") is not None]),
            "rows": scores})
    return result


def run(args: argparse.Namespace) -> int:
    if any(not math.isfinite(v) or v <= 0 for v in [args.timeout, args.load_timeout]) or args.max_new_tokens <= 0:
        raise ValueError("positive finite generation/load deadlines required")
    config = TrainingConfig.from_json(args.training_config)
    constraint_mode = getattr(args, "output_constraint", "none")
    constraint = constraint_descriptor(constraint_mode)
    manifest, samples = load_selection(args)
    partitions = {"train": ["train16"], "validation": ["validation16"],
                  "both": ["train16", "validation16"]}[args.partition]
    samples = [s for s in samples if s["split"] in partitions]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "raw.jsonl"
    records: list[dict] = []
    prompts = {s["sample_id"]: build_user_prompt(safe_prompt_input(s, args.prompt_variant), args.prompt_variant)
               for s in samples}

    def snapshot() -> dict:
        return {"sources": sources(), "selection_manifest_sha256": file_sha256(args.selection_manifest),
                "train_jsonl_sha256": file_sha256(args.train_jsonl), "validation_jsonl_sha256": file_sha256(args.validation_jsonl),
                "source_jsonl_sha256": {k: file_sha256(v["source"]["path"]) for k, v in manifest["selections"].items()},
                "training_config_sha256": file_sha256(args.training_config),
                "images": {s["sample_id"]: file_sha256(s["image_path"]) for s in samples},
                "base": artifact_manifest(config.base_model_path), "adapter": artifact_manifest(args.adapter),
                "processor": artifact_manifest(config.processor_path or config.base_model_path),
                "output_constraint": constraint_descriptor(constraint_mode)}

    try:
        before = snapshot()
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - no missing planned denominator on pre-load failure
        failure = f"{type(exc).__name__}: {exc}"
        append(raw_path, {"record_type": "provenance_error", "phase": "before_model_load", "error": failure}, records)
        for sample in samples:
            append(raw_path, {**request_record(sample, "unverified_preload", prompts[sample["sample_id"]], constraint),
                              "status": "not_started", "error": failure}, records)
        summary = summarize(records, unchanged=False)
        summary.update(complete=False, planned_requests=len(samples), partition=args.partition,
                       interrupted=isinstance(exc, KeyboardInterrupt), provenance_errors=[failure])
        _write_json(output / "summary.json", summary)
        _write_json(output / "metadata.json", {"status": "incomplete", "phase": "before_model_load",
                    "provenance_before": None, "provenance_after": None, "error": failure,
                    "raw_sha256": file_sha256(raw_path), "no_model_worker_started": True})
        return 1
    frozen = {**before, "training_config": config.to_dict(), "prompt_variant": args.prompt_variant,
              "prompt_sha256": prompt_sha256(args.prompt_variant), "runtime_versions": runtime_versions(),
              "max_new_tokens": args.max_new_tokens, "timeout": args.timeout, "load_timeout": args.load_timeout,
              "seed": 42, "do_sample": False, "temperature": 0,
              "planned": {name: 16 for name in partitions}, "partition": args.partition, "prompt_by_sample": prompts,
              "recovery": "completed token-limit response keeps worker; other failure stops, no automatic reattempt"}
    study = value_sha256(frozen)
    meta = {"study_id": study, "run_id": uuid.uuid4().hex, "status": "running",
            "started_at": datetime.now(UTC).isoformat(), "frozen_before_generation": frozen}
    _write_json(output / "metadata.json", meta)
    _write_json(output / "selection.json", manifest)
    worker = None
    interrupted = False
    stop_reason = None
    cleanup_errors = []
    try:
        worker = HFWorker(HFConfig(config.to_dict(), max_new_tokens=args.max_new_tokens,
                                  output_constraint=constraint_mode),
                          adapter_path=args.adapter, load_timeout_seconds=args.load_timeout)
        append(raw_path, {"record_type": "load", **worker.ready}, records)
        warm = worker.request(samples[0]["image_path"], prompts[samples[0]["sample_id"]], timeout_seconds=args.timeout)
        append(raw_path, {"record_type": "warmup", **warm}, records)
        if warm["status"] != "ok" and not completed_token_limit(warm, args.max_new_tokens):
            raise RuntimeError(f"warmup failed: {warm.get('error')}")
        for sample in samples:
            row = request_record(sample, study, prompts[sample["sample_id"]], constraint)
            try:
                if file_sha256(sample["image_path"]) != sample["image_file_sha256"]:
                    raise ValueError("image changed before request")
                row["inference_attempted"] = True
                append(raw_path, {"record_type": "request_start", "split": row["split"], "sample_id": row["sample_id"]}, records)
                row.update(worker.request(sample["image_path"], row["prompt"], timeout_seconds=args.timeout))
                row["image_unchanged"] = file_sha256(sample["image_path"]) == sample["image_file_sha256"]
            except KeyboardInterrupt:
                row.update(status="cancelled", error="KeyboardInterrupt")
                raise
            except Exception as exc:  # noqa: BLE001 - preserve each actual failure
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                append(raw_path, row, records)
            if row["status"] != "ok" and not completed_token_limit(row, args.max_new_tokens):
                stop_reason = "request infrastructure failure"
                break
            print(f"{row['split']} {row['sample_id']} {row['status']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "KeyboardInterrupt"
    except Exception as exc:  # noqa: BLE001 - loading/warmup failure retains all planned rows
        stop_reason = f"{type(exc).__name__}: {exc}"
        append(raw_path, {"record_type": "run_error", "error": stop_reason}, records)
    finally:
        if worker is not None:
            try:
                worker.close()
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - retain cleanup failure
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
                interrupted |= isinstance(exc, KeyboardInterrupt)
        done = {r["sample_id"] for r in records if r["record_type"] == "request"}
        for sample in samples:
            if sample["sample_id"] not in done:
                append(raw_path, {**request_record(sample, study, prompts[sample["sample_id"]], constraint),
                                  "status": "not_started", "error": stop_reason or "run ended"}, records)
        provenance_errors = []
        try:
            after = snapshot()
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - provenance failure invalidates scores
            after = None
            provenance_errors.append(f"{type(exc).__name__}: {exc}")
            interrupted |= isinstance(exc, KeyboardInterrupt)
        unchanged = before == after
        append(raw_path, {"record_type": "provenance_verdict", "unchanged": unchanged,
                          "errors": provenance_errors}, records)
        summary = summarize(records, unchanged=unchanged)
        summary.update(interrupted=interrupted, stop_reason=stop_reason, worker_cleanup_errors=cleanup_errors,
                       provenance_errors=provenance_errors, planned_requests=len(samples), partition=args.partition,
                       complete=not interrupted and not cleanup_errors and unchanged
                       and all(s["requests"] == s["inference_attempted"] == 16
                               for s in summary["splits"] if s["split"] in partitions))
        _write_json(output / "summary.json", summary)
        meta.update(status="completed" if summary["complete"] else "incomplete", finished_at=datetime.now(UTC).isoformat(),
                    provenance_after=after, raw_sha256=file_sha256(raw_path))
        _write_json(output / "metadata.json", meta)
    return 0 if summary["complete"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["train-jsonl", "validation-jsonl", "selection-manifest", "training-config", "adapter", "output"]:
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--prompt-variant", choices=TRAINING_PROMPT_VARIANTS, default="targeted_sequence")
    parser.add_argument("--partition", choices=("train", "validation", "both"), default="both",
                        help="old baseline both; new train first, validation only after capacity gate passes")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-constraint", choices=OUTPUT_CONSTRAINTS, default="none")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--load-timeout", type=float, default=600)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
