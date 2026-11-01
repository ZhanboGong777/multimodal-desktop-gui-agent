#!/usr/bin/env python
"""Independent validation-only candidate experiment. No desktop actions execute.

coverage is CPU-only. run uses a real local HF worker and must be scheduled by
the GPU owner. It does not claim improvement on the frozen official test study.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gui_agent.finetuning.candidate_grounding import (
    EXPERIMENT_VERSION,
    DetectorConfig,
    candidate_prompt,
    coverage_score,
    detect_candidates,
    map_selection,
    snapshot_hash,
    validate_snapshot,
)
from gui_agent.finetuning.evaluation import (
    HFConfig,
    HFWorker,
    artifact_manifest,
    code_manifest,
    distribution,
    file_sha256,
    runtime_versions,
    safe_prompt_input,
    score_generation,
    strict_json_loads,
    value_sha256,
)
from gui_agent.finetuning.prompts import build_user_prompt
from gui_agent.finetuning.training import (
    TrainingConfig,
    ensure_separate_validation,
    load_prepared_records,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def samples(path: str) -> list[dict[str, Any]]:
    # Existing training-side loader verifies dimensions, actions, RGB/image hash
    # and refuses official test. This route is an explicitly new development study.
    records = load_prepared_records(path)
    original = {
        row["sample_id"]: row
        for row in (
            strict_json_loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return [
        {**row, "source_sample_sha256": value_sha256(original[row["sample_id"]])}
        for row in sorted(records, key=lambda row: row["sample_id"])
    ]


def identities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in (
                "sample_id",
                "group_id",
                "source_sample_sha256",
                "image_file_sha256",
                "image_sha256",
            )
        }
        for row in records
    ]


def development_binding(
    config: TrainingConfig, dataset: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind this development study to saved validation and prove train isolation.

    Labels are validated by the data loader; they never enter the detector or prompt.
    Source_split alone cannot distinguish official-train development holdouts.
    """
    if not config.validation_jsonl or not config.train_jsonl:
        raise ValueError("saved train and validation paths are required for development scope")
    validation = Path(config.validation_jsonl).resolve(strict=True)
    training = Path(config.train_jsonl).resolve(strict=True)
    if Path(dataset).resolve(strict=True) != validation or validation == training:
        raise ValueError("dataset must be the saved isolated validation file")
    train = load_prepared_records(training)
    ensure_separate_validation(train, records)
    return {
        "training_dataset": {"path": str(training), "sha256": file_sha256(training)},
        "validation_dataset": {"path": str(validation), "sha256": file_sha256(validation)},
        "training_image_files": {row["image_path"]: row["image_file_sha256"] for row in train},
        "training_samples": len(train),
        "validation_samples": len(records),
        "disjoint_sample_group_decoded_image": True,
    }


def binding_unchanged(binding: dict[str, Any]) -> bool:
    return all(
        file_sha256(binding[key]["path"]) == binding[key]["sha256"]
        for key in ("training_dataset", "validation_dataset")
    ) and all(
        file_sha256(path) == digest for path, digest in binding["training_image_files"].items()
    )


def producer_sources() -> dict[str, str]:
    paths = (
        "src/gui_agent/finetuning/candidate_grounding.py",
        "scripts/week5_candidate_grounding.py",
        "src/gui_agent/perception/ui_detection.py",
        "src/gui_agent/perception/ocr.py",
        "src/gui_agent/perception/preprocessing.py",
        "src/gui_agent/schemas.py",
        "src/gui_agent/finetuning/evaluation.py",
        "src/gui_agent/finetuning/training.py",
        "src/gui_agent/finetuning/data.py",
    )
    return {path: file_sha256(ROOT / path) for path in paths}


def detector_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("opencv-python", "opencv-python-headless", "pytesseract", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def coverage(args: argparse.Namespace) -> int:
    records = samples(args.dataset)
    config_path = Path(args.training_config).resolve(strict=True)
    binding = development_binding(TrainingConfig.from_json(config_path), args.dataset, records)
    config = DetectorConfig(
        ocr=args.ocr, max_candidates=args.max_candidates, contour_max_aspect=args.contour_max_aspect
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = producer_sources()
    frozen = {
        "schema_version": EXPERIMENT_VERSION,
        "scope": "validation_candidate_coverage_only",
        "detector_config": asdict(config),
        "dataset_sha256": file_sha256(args.dataset),
        "samples": identities(records),
        "source_files": sources,
        "development_binding": binding,
        "binding_training_config_sha256": file_sha256(config_path),
        "versions": runtime_versions(),
        "detector_versions": detector_versions(),
        "target_annotations_usage": "scoring only, after image-only candidate generation",
        "gpu_model_calls": 0,
        "desktop_task_e2e": False,
    }
    write_json(output / "metadata.json", frozen)
    totals: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    counts, durations, errors = [], [], []
    closest_distances = []
    for sample in records:
        started = time.perf_counter()
        row = {
            "sample_id": sample["sample_id"],
            "source_sample_sha256": sample["source_sample_sha256"],
        }
        try:
            snapshot = detect_candidates(sample["image_path"], config)
            if snapshot["image_file_sha256"] != sample["image_file_sha256"]:
                raise ValueError("candidate screenshot differs from frozen prepared image")
            validate_snapshot(snapshot, sample["image_path"])
            # Labels first enter this independent scoring function; never detector.
            scores = coverage_score(
                sample["actions"], sample.get("action_annotations", []), snapshot
            )
            row.update(
                snapshot=snapshot,
                snapshot_sha256=snapshot_hash(snapshot),
                coverage=scores,
                status="ok",
            )
            closest_distances.extend(scores["closest_candidate_center_normalized_distances"])
            totals.update({k: v for k, v in scores.items() if type(v) is int})
            counts.append(len(snapshot["candidates"]))
            modes[snapshot["mode"]] += 1
        except Exception as exc:  # noqa: BLE001 - failures remain in full validation denominator
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"sample_id": sample["sample_id"], "error": error})
            row.update(status="error", error=error)
            empty_scores = coverage_score(
                sample["actions"],
                sample.get("action_annotations", []),
                {
                    "candidates": [],
                    "image_width": sample["image_width"],
                    "image_height": sample["image_height"],
                },
            )
            closest_distances.extend(empty_scores["closest_candidate_center_normalized_distances"])
            totals.update({k: v for k, v in empty_scores.items() if type(v) is int})
        row["duration_ms"] = (time.perf_counter() - started) * 1000
        durations.append(row["duration_ms"])
        append(output / "candidates.jsonl", row)
        print(f"coverage {sample['sample_id']} {row['status']}", flush=True)
    provenance_error = None
    try:
        unchanged = (
            file_sha256(args.dataset) == frozen["dataset_sha256"]
            and file_sha256(config_path) == frozen["binding_training_config_sha256"]
            and binding_unchanged(binding)
            and all(file_sha256(ROOT / name) == digest for name, digest in sources.items())
            and all(
                file_sha256(sample["image_path"]) == sample["image_file_sha256"]
                for sample in records
            )
        )
    except Exception as exc:  # noqa: BLE001 - completed CPU requests must remain auditable
        unchanged = False
        provenance_error = f"{type(exc).__name__}: {exc}"
    write_json(
        output / "summary.json",
        {
            "schema_version": EXPERIMENT_VERSION,
            "samples": len(records),
            "totals": dict(totals),
            "candidate_center_annotation_coverage_rate": totals["candidate_center_hits_annotation"]
            / totals["bbox_eligible"]
            if totals["bbox_eligible"]
            else None,
            "coverage_is_oracle_upper_bound_not_selector_accuracy": True,
            "candidate_count": distribution(counts),
            "label_point_inside_candidate_rate": totals["target_point_inside_candidate"]
            / totals["click_targets"]
            if totals["click_targets"]
            else None,
            "closest_center_normalized_distance_oracle": distribution(closest_distances),
            "duration_ms": distribution(durations),
            "modes": dict(modes),
            "errors": errors,
            "provenance_unchanged": unchanged,
            "provenance_error": provenance_error,
        },
    )
    return 1 if errors or not unchanged else 0


def metrics(requests: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    distances = []
    for row in requests:
        s = score_generation(
            row["target"]["actions"],
            row["mapped_generation"],
            width=row["target"]["image_width"],
            height=row["target"]["image_height"],
            action_annotations=row["target"].get("action_annotations", []),
            status=row["status"],
        )
        for key in (
            "strict_json_valid",
            "strict_action_exact",
            "action_type_correct",
            "action_type_slots",
            "text_exact_correct",
            "text_exact_total",
            "click_bbox_hits",
            "click_bbox_eligible",
        ):
            totals[key] += s[key]
        distances.extend(s["click_normalized_distances_all_targets"])
    n = len(requests)
    return {
        "requests": n,
        "statuses": dict(Counter(r["status"] for r in requests)),
        "totals": dict(totals),
        "observed_request_wall_ms": distribution(
            [r["duration_ms"] for r in requests if r["duration_ms"] is not None]
        ),
        "estimated_recompute_pipeline_ms": distribution(
            [
                r["estimated_recompute_pipeline_ms"]
                for r in requests
                if r.get("estimated_recompute_pipeline_ms") is not None
            ]
        ),
        "duration_missing": sum(r["duration_ms"] is None for r in requests),
        "inference_attempted": sum(r["inference_attempted"] for r in requests),
        "selection_schema_valid": sum(
            r["status"] == "ok" and bool(r.get("selection_schema_valid")) for r in requests
        ),
        "model_request_ms": distribution(
            [r["model_duration_ms"] for r in requests if r["model_duration_ms"] is not None]
        ),
        "click_distance_with_missing_penalty": distribution(distances),
        "score_contract": "existing score_generation: whole native action array exact; half-open real bbox hits",
    }


def run(args: argparse.Namespace) -> int:
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.load_timeout)):
        raise ValueError("timeouts must be finite and positive")
    if args.max_new_tokens <= 0:
        raise ValueError("token budget must be positive")
    if args.limit is not None and args.limit <= 0 or args.seed < 0:
        raise ValueError("limit must be positive and seed nonnegative")
    pool = samples(args.dataset)
    config_path = Path(args.training_config).resolve()
    config = TrainingConfig.from_json(config_path)
    binding = development_binding(config, args.dataset, pool)
    records = (
        sorted(pool, key=lambda s: value_sha256({"seed": args.seed, "sample_id": s["sample_id"]}))[
            : args.limit
        ]
        if args.limit is not None
        else pool
    )
    cache = Path(args.coverage).resolve()
    coverage_metadata = strict_json_loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if (
        coverage_metadata.get("schema_version") != EXPERIMENT_VERSION
        or coverage_metadata.get("scope") != "validation_candidate_coverage_only"
    ):
        raise ValueError("unsupported candidate cache scope/schema")
    if coverage_metadata["samples"] != identities(pool) or coverage_metadata[
        "dataset_sha256"
    ] != file_sha256(args.dataset):
        raise ValueError("CPU candidate cache and validation data identities differ")
    if coverage_metadata.get("development_binding") != binding:
        raise ValueError("candidate cache lacks the current isolated validation binding")
    if coverage_metadata.get("source_files") != producer_sources():
        raise ValueError("candidate producer sources differ from current frozen code")
    coverage_summary = strict_json_loads((cache / "summary.json").read_text(encoding="utf-8"))
    if coverage_summary.get("provenance_unchanged") is not True or coverage_summary.get(
        "samples"
    ) != len(pool):
        raise ValueError("candidate producer did not pass complete provenance audit")
    detector_config = asdict(DetectorConfig(**coverage_metadata["detector_config"]))
    candidate_rows = [
        strict_json_loads(line)
        for line in (cache / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = {r["sample_id"]: r for r in candidate_rows}
    if len(candidates) != len(candidate_rows):
        raise ValueError("duplicate sample in candidate cache")
    if set(candidates) != {sample["sample_id"] for sample in pool}:
        raise ValueError("candidate cache sample denominator differs")
    for sample in pool:
        cached = candidates[sample["sample_id"]]
        if cached.get("source_sample_sha256") != sample["source_sample_sha256"]:
            raise ValueError("candidate cache source sample differs")
        if cached["status"] == "ok":
            snapshot = cached["snapshot"]
            if snapshot.get("detector_config") != detector_config or snapshot_hash(
                snapshot
            ) != cached.get("snapshot_sha256"):
                raise ValueError("candidate snapshot config/hash differs from producer")
            validate_snapshot(snapshot, sample["image_path"])
    sources = code_manifest(ROOT)
    artifact_paths = {
        "base": config.base_model_path,
        "adapter": args.adapter,
        "processor": config.processor_path or config.base_model_path,
    }
    artifacts = {name: artifact_manifest(path) for name, path in artifact_paths.items()}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = {
        "schema_version": EXPERIMENT_VERSION,
        "scope": "new_validation_development_experiment",
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "configuration": config.to_dict(),
        "source_files": sources,
        "dataset_sha256": file_sha256(args.dataset),
        "selection": identities(records),
        "development_binding": binding,
        "selection_policy": {
            "limit": args.limit,
            "seed": args.seed,
            "algorithm": "SHA256({seed,sample_id}) order when limit supplied; labels never used",
        },
        "coverage_file_sha256": file_sha256(cache / "candidates.jsonl"),
        "coverage_metadata_sha256": file_sha256(cache / "metadata.json"),
        "coverage_summary_sha256": file_sha256(cache / "summary.json"),
        "training_config_sha256": file_sha256(config_path),
        "model_artifacts": artifacts,
        "versions": runtime_versions(),
        "max_new_tokens": args.max_new_tokens,
        "timeout_seconds": args.timeout,
        "load_timeout_seconds": args.load_timeout,
        "prompt_schema": "native_action_array_original"
        if args.mode == "baseline"
        else EXPERIMENT_VERSION,
        "new_candidate_schema_not_trained_in_existing_adapter": True,
        "latency_scope": "observed request wall time excluding model load/warmup; cached detection separately reported",
        "estimated_recompute_latency_scope": "sum of separately measured historical CPU detection and observed request wall time; NOT observed live E2E",
        "worker_recovery": "only completed typed token_limit stays resident; other errors close",
        "desktop_task_e2e": False,
    }
    study_id = value_sha256(frozen)
    write_json(
        output / "metadata.json", {**frozen, "study_id": study_id, "run_id": uuid.uuid4().hex}
    )
    worker = None
    requests = []
    interrupted = False
    infrastructure_error = False
    unavailable_error = None

    def close_worker() -> None:
        nonlocal worker, infrastructure_error, interrupted
        closing, worker = worker, None
        if closing is None:
            return
        try:
            closing.close()
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - retain ledger even on cleanup failures
            infrastructure_error = True
            interrupted |= isinstance(exc, KeyboardInterrupt)
            append(
                output / "raw.jsonl",
                {"record_type": "worker_cleanup_error", "error": f"{type(exc).__name__}: {exc}"},
            )

    try:
        for sample in records:
            row = {
                "sample_id": sample["sample_id"],
                "source_sample_sha256": sample["source_sample_sha256"],
                "target": {
                    k: sample.get(k, [])
                    for k in ("actions", "action_annotations", "image_width", "image_height")
                },
                "status": "error",
                "generation": "",
                "mapped_generation": "",
                "model_duration_ms": None,
                "inference_attempted": False,
                "model_error_kind": None,
            }
            started = time.perf_counter()
            detector_ms = 0.0
            startup_ms = 0.0
            stage = "preparation"
            try:
                if unavailable_error is not None:
                    row["status"] = "load_or_warmup_error"
                    raise ValueError(unavailable_error)
                cached = candidates[sample["sample_id"]]
                if cached["source_sample_sha256"] != sample["source_sample_sha256"]:
                    raise ValueError("candidate sample content hash differs")
                if args.mode == "candidate":
                    if cached["status"] != "ok":
                        raise ValueError("CPU candidate detection failed for this sample")
                    snapshot = cached["snapshot"]
                    if snapshot_hash(snapshot) != cached["snapshot_sha256"]:
                        raise ValueError("candidate cache hash differs")
                    validate_snapshot(snapshot, sample["image_path"])
                    row["snapshot"] = snapshot
                    detector_ms = snapshot["detection_ms"]
                    if not snapshot["candidates"]:
                        raise ValueError("no visual candidates; no fabricated fallback")
                    prompt = candidate_prompt(sample["instruction"], snapshot)
                else:
                    prompt = build_user_prompt(safe_prompt_input(sample), "original")
                row["prompt"] = prompt
                row["prompt_sha256"] = value_sha256(prompt)
                if file_sha256(sample["image_path"]) != sample["image_file_sha256"]:
                    raise ValueError("image changed before model dispatch")
                if worker is None:
                    stage = "startup"
                    startup_started = time.perf_counter()
                    worker = HFWorker(
                        HFConfig(config.to_dict(), max_new_tokens=args.max_new_tokens),
                        adapter_path=args.adapter if args.checkpoint == "adapter" else None,
                        load_timeout_seconds=args.load_timeout,
                    )
                    append(
                        output / "raw.jsonl",
                        {"record_type": "worker_loaded", "worker": worker.ready},
                    )
                    warmup = worker.request(
                        sample["image_path"], prompt, timeout_seconds=args.timeout
                    )
                    append(
                        output / "raw.jsonl",
                        {
                            "record_type": "warmup",
                            "sample_id": sample["sample_id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "image_file_sha256": sample["image_file_sha256"],
                            **warmup,
                        },
                    )
                    if warmup["status"] != "ok" and not (
                        warmup["status"] == "error"
                        and warmup.get("error_kind") == "token_limit"
                        and type(warmup.get("generated_tokens")) is int
                        and warmup["generated_tokens"] >= args.max_new_tokens
                    ):
                        raise RuntimeError("model warmup failed")
                    # Model loading and warmup are separate; only this sample's
                    # detector/dispatch/mapping cost enters the request duration.
                    startup_ms = (time.perf_counter() - startup_started) * 1000
                stage = "dispatch"
                row["worker"] = worker.ready
                row["inference_attempted"] = True
                result = worker.request(sample["image_path"], prompt, timeout_seconds=args.timeout)
                row.update(
                    generation=result.get("generation", ""),
                    model_duration_ms=result.get("duration_ms"),
                    model_error_kind=result.get("error_kind"),
                    generated_tokens=result.get("generated_tokens"),
                    model_response=result,
                )
                if file_sha256(sample["image_path"]) != sample["image_file_sha256"]:
                    row["status"] = "provenance_changed"
                    raise ValueError("image changed during model call")
                if result["status"] != "ok":
                    row["status"] = result["status"]
                    row["error"] = result.get("error")
                    if not (
                        result["status"] == "error"
                        and result.get("error_kind") == "token_limit"
                        and type(result.get("generated_tokens")) is int
                        and result["generated_tokens"] >= args.max_new_tokens
                    ):
                        infrastructure_error = True
                        close_worker()
                else:
                    stage = "mapping"
                    if args.mode == "candidate":
                        row["mapped_generation"] = json.dumps(
                            map_selection(result["generation"], snapshot, sample["image_path"])
                        )
                        row["selection_schema_valid"] = True
                    else:
                        row["mapped_generation"] = result["generation"]
                    row["status"] = "ok"
            except KeyboardInterrupt:
                row["status"], row["error"], interrupted = "cancelled", "KeyboardInterrupt", True
                if stage == "startup":
                    startup_ms = (time.perf_counter() - startup_started) * 1000
            except Exception as exc:  # noqa: BLE001 - all requests retain failures and original generation
                row["error"] = f"{type(exc).__name__}: {exc}"
                if stage in {"startup", "dispatch"}:
                    infrastructure_error = True
                    if stage == "startup":
                        unavailable_error = row["error"]
                        row["status"] = "load_or_warmup_error"
                        startup_ms = (time.perf_counter() - startup_started) * 1000
                    close_worker()
            row["duration_ms"] = max(0.0, (time.perf_counter() - started) * 1000 - startup_ms)
            row["cached_detection_ms"] = detector_ms
            row["estimated_recompute_pipeline_ms"] = row["duration_ms"] + detector_ms
            row["excluded_model_load_warmup_ms"] = startup_ms
            append(output / "raw.jsonl", {"record_type": "request", **row})
            requests.append(row)
            print(f"{args.mode} {sample['sample_id']} {row['status']}", flush=True)
            if interrupted:
                break
    finally:
        close_worker()
        completed = {r["sample_id"] for r in requests}
        for sample in records:
            if sample["sample_id"] in completed:
                continue
            row = {
                "sample_id": sample["sample_id"],
                "source_sample_sha256": sample["source_sample_sha256"],
                "target": {
                    k: sample.get(k, [])
                    for k in ("actions", "action_annotations", "image_width", "image_height")
                },
                "status": "not_started",
                "generation": "",
                "mapped_generation": "",
                "duration_ms": None,
                "model_duration_ms": None,
                "inference_attempted": False,
                "error": "batch interrupted before request; no timing invented",
            }
            append(output / "raw.jsonl", {"record_type": "request", **row})
            requests.append(row)
        # Requests are persisted before post-run evidence checks. Never let a
        # provenance read failure discard the ledger or imply successful scores.
        provenance_error = None
        try:
            unchanged = (
                code_manifest(ROOT) == sources
                and binding_unchanged(binding)
                and file_sha256(args.dataset) == frozen["dataset_sha256"]
                and all(
                    file_sha256(sample["image_path"]) == sample["image_file_sha256"]
                    for sample in records
                )
                and all(
                    artifact_manifest(artifact_paths[name])["sha256"] == manifest["sha256"]
                    for name, manifest in artifacts.items()
                )
                and file_sha256(config_path) == frozen["training_config_sha256"]
                and file_sha256(cache / "metadata.json") == frozen["coverage_metadata_sha256"]
                and file_sha256(cache / "candidates.jsonl") == frozen["coverage_file_sha256"]
                and file_sha256(cache / "summary.json") == frozen["coverage_summary_sha256"]
            )
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - preserve ledger/summary on interrupted provenance scans
            interrupted |= isinstance(exc, KeyboardInterrupt)
            unchanged = False
            provenance_error = f"{type(exc).__name__}: {exc}"
        append(
            output / "raw.jsonl",
            {
                "record_type": "provenance_verdict",
                "unchanged": unchanged,
                "error": provenance_error,
            },
        )
        scoring_rows = (
            requests if unchanged else [{**r, "status": "provenance_changed"} for r in requests]
        )
        write_json(
            output / "summary.json",
            {
                **metrics(scoring_rows),
                "planned_samples": len(records),
                "interrupted": interrupted,
                "unmeasured_samples": sum(not r["inference_attempted"] for r in requests),
                "provenance_unchanged": unchanged,
                "provenance_error": provenance_error,
                "study_id": study_id,
            },
        )
    return 130 if interrupted else 1 if infrastructure_error or not unchanged else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    cpu = subs.add_parser(
        "coverage", help="CPU only; all official-train-heldout validation records"
    )
    cpu.add_argument("--dataset", required=True)
    cpu.add_argument(
        "--training-config",
        required=True,
        help="saved train/validation binding; no model is loaded",
    )
    cpu.add_argument("--output", required=True)
    cpu.add_argument("--ocr", choices=("tesseract", "none"), default="tesseract")
    cpu.add_argument("--max-candidates", type=int, default=64)
    cpu.add_argument("--contour-max-aspect", type=float, default=12)
    gpu = subs.add_parser(
        "run", help="real HF; root GPU owner only; validation development experiment"
    )
    for key in ("dataset", "coverage", "training-config", "adapter", "output"):
        gpu.add_argument("--" + key, required=True)
    gpu.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    gpu.add_argument("--checkpoint", choices=("base", "adapter"), default="adapter")
    gpu.add_argument("--max-new-tokens", type=int, default=256)
    gpu.add_argument("--timeout", type=float, default=120)
    gpu.add_argument("--load-timeout", type=float, default=600)
    gpu.add_argument(
        "--limit", type=int, help="fixed label-free subset of the full frozen validation pool"
    )
    gpu.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        return coverage(args) if args.command == "coverage" else run(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"candidate experiment error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
