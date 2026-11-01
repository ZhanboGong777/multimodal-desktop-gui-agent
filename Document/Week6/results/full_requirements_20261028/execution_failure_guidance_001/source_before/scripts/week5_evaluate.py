#!/usr/bin/env python
"""Run the frozen Week 5 2x2 HF comparison, or recompute its offline metrics.

Example: python scripts/week5_evaluate.py run --dataset heldout.jsonl
  --training-config training_config.json --adapter output/adapter --output outputs/eval
There is deliberately no mock backend and no desktop execution in this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gui_agent.finetuning.evaluation import (
    CELLS,
    SCHEMA_VERSION,
    HFConfig,
    HFWorker,
    artifact_manifest,
    code_manifest,
    file_sha256,
    load_prepared,
    runtime_versions,
    safe_prompt_input,
    select_samples,
    strict_json_loads,
    summarize_records,
    value_sha256,
)
from gui_agent.finetuning.prompts import (
    PROMPT_VERSION,
    build_user_prompt,
    prompt_rules,
    prompt_sha256,
)

WORKER_RECOVERY_POLICY = {
    "version": "completed-token-limit-resident-v1",
    "nonfatal_error_kind": "token_limit",
    "nonfatal_requirements": "status=error and integer generated_tokens>=max_new_tokens",
    "nonfatal_disposition": "retain error and failed denominator; reuse worker; no infrastructure failure",
    "other_error_timeout_cancelled": "close worker; reload and warmup before next request",
}


def _completed_token_limit(result: dict[str, Any], max_new_tokens: int) -> bool:
    return (
        result["status"] == "error"
        and result.get("error_kind") == "token_limit"
        and type(result.get("generated_tokens")) is int
        and result["generated_tokens"] >= max_new_tokens
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _append(path: Path, record: dict[str, Any], records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    records.append(record)


def _selection(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    selected = select_samples(samples, seed=42, max_samples=args.max_samples)
    result = {
        "dataset_sha256": file_sha256(args.dataset),
        "seed": 42,
        "cap": args.max_samples,
        "selection_method": "sort sample_id, Random(42).sample up to cap, sort selected sample_id",
        "samples": [
            {
                key: sample[key]
                for key in (
                    "sample_id",
                    "group_id",
                    "source_sample_sha256",
                    "image_sha256",
                    "image_file_sha256",
                )
            }
            for sample in selected
        ],
    }
    if args.selection:
        saved = strict_json_loads(Path(args.selection).read_text(encoding="utf-8"))
        if saved != result:
            raise ValueError("heldout selection does not match the frozen seed/data/image manifest")
    return result


def _record(
    sample: dict[str, Any],
    cell: str,
    study_id: str,
    run_id: str,
    result: dict[str, Any],
    prompt: str,
    worker_info: dict[str, Any] | None,
    warm_state: str,
) -> dict[str, Any]:
    return {
        "record_type": "request",
        "schema_version": SCHEMA_VERSION,
        "scope": "offline_gui_action_generation",
        "run_id": run_id,
        "study_id": study_id,
        "cell": cell,
        "sample_id": sample["sample_id"],
        "group_id": sample["group_id"],
        "source": sample["source"],
        "source_split": sample["source_split"],
        "source_sample_sha256": sample["source_sample_sha256"],
        "image_path": sample["image_path"],
        "image_sha256": sample["image_sha256"],
        "image_file_sha256": sample["image_file_sha256"],
        "target": {
            key: sample.get(key, [])
            for key in ("actions", "action_annotations", "image_width", "image_height")
        },
        "instruction": sample["instruction"],
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_variant": cell.split("_", 1)[1],
        "generation": result.get("generation", ""),
        "status": result["status"],
        "error": result.get("error"),
        "error_kind": result.get("error_kind"),
        "duration_ms": result.get("duration_ms"),
        "worker_inference_ms": result.get("worker_inference_ms"),
        "rendered_prompt": result.get("rendered_prompt"),
        "rendered_prompt_sha256": hashlib.sha256(
            result["rendered_prompt"].encode("utf-8")
        ).hexdigest()
        if result.get("rendered_prompt") is not None
        else None,
        "generated_tokens": result.get("generated_tokens"),
        "inference_attempted": result.get("inference_attempted", True),
        "worker": worker_info,
        "warm_state": warm_state,
    }


def _input_unchanged(sample: dict[str, Any]) -> bool:
    return file_sha256(sample["image_path"]) == sample["image_file_sha256"]


def _training_inputs(config: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Check training/validation identities without exposing labels to the worker."""
    evidence = {}
    for name in ("train_jsonl", "validation_jsonl"):
        if not config.get(name):
            continue
        path = Path(config[name]).resolve()
        rows = [
            strict_json_loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for key in ("sample_id", "group_id", "image_sha256"):
            overlap = {row[key] for row in rows} & {row[key] for row in selected}
            if overlap:
                raise ValueError(f"heldout/training leakage in {name}.{key}")
        evidence[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "samples": len(rows),
            "groups": len({row["group_id"] for row in rows}),
        }
    return evidence


def run(args: argparse.Namespace) -> int:
    if (
        not math.isfinite(args.timeout)
        or not math.isfinite(args.load_timeout)
        or args.timeout <= 0
        or args.load_timeout <= 0
        or args.warmup_calls < 0
    ):
        raise ValueError("timeouts must be positive and warmup count nonnegative")
    cells = args.cell or list(CELLS)
    if len(cells) != len(set(cells)):
        raise ValueError("a cell may run only once")
    samples = load_prepared(args.dataset)
    selection = _selection(samples, args)
    chosen_ids = {sample["sample_id"] for sample in selection["samples"]}
    selected = sorted(
        (sample for sample in samples if sample["sample_id"] in chosen_ids),
        key=lambda sample: sample["sample_id"],
    )
    config_path = Path(args.training_config).resolve()
    config_payload = strict_json_loads(config_path.read_text(encoding="utf-8"))
    # The training side owns validation and model loading; both cells receive the
    # same saved, effective configuration. This import does not load a model.
    from gui_agent.finetuning.training import TrainingConfig

    config_payload = TrainingConfig.from_dict(config_payload).to_dict()
    base_path = config_payload["base_model_path"]
    if config_payload["base_model_revision"] != "895c3a49bc3fa70a340399125c650a463535e71c":
        raise ValueError("base model revision differs from the preregistered checkpoint")
    if config_payload["base_model_id"] != "Qwen/Qwen2-VL-2B-Instruct":
        raise ValueError("base model ID differs from the preregistered checkpoint")
    inference = HFConfig(config_payload, max_new_tokens=args.max_new_tokens, seed=42)
    base_manifest = artifact_manifest(base_path)
    adapter_manifest = artifact_manifest(args.adapter)
    processor_path = config_payload.get("processor_path") or base_path
    processor_is_base = Path(processor_path).resolve() == Path(base_path).resolve()
    processor_manifest = base_manifest if processor_is_base else artifact_manifest(processor_path)
    training_inputs = _training_inputs(config_payload, selected)
    source_before = code_manifest(REPOSITORY)
    config_file_hash = file_sha256(config_path)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "inference_config": asdict(inference),
        "runtime_versions": runtime_versions(),
        "base_artifact_sha256": base_manifest["sha256"],
        "adapter_artifact_sha256": adapter_manifest["sha256"],
        "processor_artifact_sha256": processor_manifest["sha256"],
        "training_inputs": training_inputs,
        "source_files": source_before,
        "selection": selection,
        "prompt_version": PROMPT_VERSION,
        "prompts": {
            variant: {"text": prompt_rules(variant), "sha256": prompt_sha256(variant)}
            for variant in ("original", "improved")
        },
        "warmup_calls_per_worker": args.warmup_calls,
        "percall_timeout_seconds": args.timeout,
        "load_timeout_seconds": args.load_timeout,
        "worker_recovery_policy": WORKER_RECOVERY_POLICY,
    }
    study_id = value_sha256(frozen)
    run_id = uuid.uuid4().hex
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "selection.json", selection)
    _write_json(
        output / "metadata.json",
        {
            **frozen,
            "study_id": study_id,
            "run_id": run_id,
            "cells": cells,
            "started_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "config_file_sha256": config_file_hash,
            "base_artifact_manifest": base_manifest,
            "adapter_artifact_manifest": adapter_manifest,
            "processor_artifact_manifest": processor_manifest,
            "desktop_task_e2e": False,
            "selection_frozen_before_generation": True,
        },
    )
    records: list[dict[str, Any]] = []
    raw_path = output / "raw.jsonl"
    _append(
        raw_path,
        {"record_type": "study", "run_id": run_id, "study_id": study_id, "metadata": frozen},
        records,
    )
    interrupted = False
    infrastructure_error = False
    worker: HFWorker | None = None
    completed: set[tuple[str, str]] = set()
    try:
        for cell in cells:
            variant = cell.split("_", 1)[1]
            unavailable_error = None
            for sample in selected:
                prompt = build_user_prompt(safe_prompt_input(sample), prompt_variant=variant)
                worker_info = None
                warm_state = "cold"
                if unavailable_error is not None:
                    result = {
                        "status": "load_or_warmup_error",
                        "generation": "",
                        "error": unavailable_error,
                        "duration_ms": None,
                        "inference_attempted": False,
                    }
                    _append(
                        raw_path,
                        _record(
                            sample, cell, study_id, run_id, result, prompt, None, "unavailable"
                        ),
                        records,
                    )
                    completed.add((cell, sample["sample_id"]))
                    continue
                if worker is None:
                    startup_started = time.perf_counter()
                    try:
                        worker = HFWorker(
                            inference,
                            adapter_path=args.adapter if cell.startswith("adapter") else None,
                            load_timeout_seconds=args.load_timeout,
                        )
                        worker_info = worker.ready
                        if worker.ready["versions"] != frozen["runtime_versions"]:
                            raise RuntimeError(
                                "worker package versions differ from the frozen runtime"
                            )
                        _append(
                            raw_path,
                            {
                                "record_type": "worker_loaded",
                                "run_id": run_id,
                                "cell": cell,
                                "worker": worker_info,
                            },
                            records,
                        )
                        for warmup_index in range(args.warmup_calls):
                            warm = worker.request(
                                selected[0]["image_path"],
                                build_user_prompt(
                                    safe_prompt_input(selected[0]), prompt_variant=variant
                                ),
                                timeout_seconds=args.timeout,
                            )
                            _append(
                                raw_path,
                                {
                                    "record_type": "warmup",
                                    "run_id": run_id,
                                    "cell": cell,
                                    "index": warmup_index,
                                    **warm,
                                },
                                records,
                            )
                            # A completed forward pass warms the model even when
                            # output was truncated; warmup never gates on answer quality.
                            if warm["status"] != "ok" and not _completed_token_limit(
                                warm, args.max_new_tokens
                            ):
                                raise RuntimeError(f"warmup failed: {warm.get('error')}")
                        worker_info = {**worker.ready, "warmup_calls": args.warmup_calls}
                    except Exception as exc:  # noqa: BLE001 - retain loader/warmup failures in all-sample denominator
                        if worker is not None:
                            worker.close()
                            worker = None
                        infrastructure_error = True
                        unavailable_error = f"{type(exc).__name__}: {exc}"
                        _append(
                            raw_path,
                            {
                                "record_type": "startup_error",
                                "run_id": run_id,
                                "cell": cell,
                                "error": unavailable_error,
                                "startup_elapsed_ms": (time.perf_counter() - startup_started)
                                * 1000,
                                "worker": worker_info,
                            },
                            records,
                        )
                        result = {
                            "status": "load_or_warmup_error",
                            "generation": "",
                            "error": unavailable_error,
                            "duration_ms": None,
                            "inference_attempted": False,
                        }
                        _append(
                            raw_path,
                            _record(
                                sample,
                                cell,
                                study_id,
                                run_id,
                                result,
                                prompt,
                                worker_info,
                                "unavailable",
                            ),
                            records,
                        )
                        completed.add((cell, sample["sample_id"]))
                        continue
                worker_info = {**worker.ready, "warmup_calls": args.warmup_calls}
                warm_state = (
                    "warmup_completed"
                    if args.warmup_calls
                    else "cold_first_call"
                    if worker.request_count == 0
                    else "resident_no_warmup"
                )
                call_started = time.perf_counter()
                try:
                    if not _input_unchanged(sample):
                        result = {
                            "status": "provenance_changed",
                            "generation": "",
                            "error": "image changed before dispatch",
                            "duration_ms": None,
                            "inference_attempted": False,
                        }
                    else:
                        result = worker.request(
                            sample["image_path"], prompt, timeout_seconds=args.timeout
                        )
                    try:
                        if not _input_unchanged(sample):
                            result = {
                                **result,
                                "observed_status": result["status"],
                                "status": "provenance_changed",
                                "error": "image changed during call",
                            }
                    except Exception as exc:  # noqa: BLE001 - retain raw generation when postcall evidence cannot be read
                        result = {
                            **result,
                            "observed_status": result["status"],
                            "status": "provenance_changed",
                            "error": f"postcall image check failed: {type(exc).__name__}: {exc}",
                        }
                except KeyboardInterrupt:
                    interrupted = True
                    result = {
                        "status": "cancelled",
                        "generation": "",
                        "error": "KeyboardInterrupt",
                        "duration_ms": (time.perf_counter() - call_started) * 1000,
                    }
                except Exception as exc:  # noqa: BLE001 - every dispatch failure is a recorded sample
                    result = {
                        "status": "error",
                        "generation": "",
                        "error": f"{type(exc).__name__}: {exc}",
                        "duration_ms": (time.perf_counter() - call_started) * 1000,
                    }
                record = _record(
                    sample, cell, study_id, run_id, result, prompt, worker_info, warm_state
                )
                record["observed_status"] = result.get("observed_status", result["status"])
                _append(raw_path, record, records)
                completed.add((cell, sample["sample_id"]))
                print(f"{cell} {sample['sample_id']} {result['status']}", flush=True)
                completed_token_limit = _completed_token_limit(result, args.max_new_tokens)
                if result["status"] != "ok" and not completed_token_limit:
                    infrastructure_error = True
                if (
                    result["status"] in {"timeout", "error", "cancelled"}
                    and not completed_token_limit
                ):
                    worker.close()
                    worker = None
                if interrupted:
                    break
            if worker is not None:
                worker.close()
                worker = None
            if interrupted:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if worker is not None:
            worker.close()
        # Cancellation preserves the exact planned denominator, with unmeasured
        # jobs explicitly marked (never invented durations or generations).
        for cell in cells:
            for sample in selected:
                if (cell, sample["sample_id"]) not in completed:
                    result = {
                        "status": "not_started",
                        "generation": "",
                        "duration_ms": None,
                        "inference_attempted": False,
                        "error": "batch interrupted",
                    }
                    prompt = build_user_prompt(
                        safe_prompt_input(sample), prompt_variant=cell.split("_", 1)[1]
                    )
                    _append(
                        raw_path,
                        _record(
                            sample, cell, study_id, run_id, result, prompt, None, "unavailable"
                        ),
                        records,
                    )
        try:
            source_after = code_manifest(REPOSITORY)
            unchanged = (
                source_after == source_before
                and file_sha256(args.dataset) == selection["dataset_sha256"]
                and file_sha256(config_path) == config_file_hash
                and all(_input_unchanged(sample) for sample in selected)
                and all(
                    file_sha256(item["path"]) == item["sha256"] for item in training_inputs.values()
                )
                and artifact_manifest(base_path)["sha256"] == base_manifest["sha256"]
                and artifact_manifest(args.adapter)["sha256"] == adapter_manifest["sha256"]
                and (
                    processor_is_base
                    or artifact_manifest(processor_path)["sha256"] == processor_manifest["sha256"]
                )
            )
            verdict = {
                "record_type": "provenance_verdict",
                "run_id": run_id,
                "unchanged": unchanged,
                "source_after": source_after,
            }
        except Exception as exc:  # noqa: BLE001 - post-measurement provenance failures cannot discard requests
            verdict = {
                "record_type": "provenance_verdict",
                "run_id": run_id,
                "unchanged": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _append(raw_path, verdict, records)
        summary = summarize_records(records)
        summary.update(
            {
                "planned_samples_per_cell": len(selected),
                "selected_cells": cells,
                "run_id": run_id,
                "study_id": study_id,
                "interrupted": interrupted,
                "provenance_unchanged": verdict["unchanged"],
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_json(output / "summary.json", summary)
    return 130 if interrupted else 1 if infrastructure_error or not verdict["unchanged"] else 0


def aggregate(args: argparse.Namespace) -> int:
    records = []
    for filename in args.raw:
        records.extend(
            strict_json_loads(line)
            for line in Path(filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("aggregate output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, summarize_records(records))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    measurement = commands.add_parser("run", help="real local HF inference only")
    measurement.add_argument("--dataset", required=True)
    measurement.add_argument("--training-config", required=True)
    measurement.add_argument("--adapter", required=True)
    measurement.add_argument(
        "--output", required=True, help="new directory; existing output is refused"
    )
    measurement.add_argument(
        "--cell",
        action="append",
        choices=CELLS,
        help="default all four; repeated flags select a subset",
    )
    measurement.add_argument("--selection", help="reuse and verify a previous selection.json")
    measurement.add_argument("--max-samples", type=int, choices=(48, 64), default=48)
    measurement.add_argument("--max-new-tokens", type=int, default=256)
    measurement.add_argument("--timeout", type=float, default=120)
    measurement.add_argument("--load-timeout", type=float, default=600)
    measurement.add_argument("--warmup-calls", type=int, default=1)
    reduction = commands.add_parser(
        "aggregate", help="recompute scores from raw generations and targets"
    )
    reduction.add_argument("--raw", action="append", required=True)
    reduction.add_argument("--output", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args) if args.command == "run" else aggregate(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"evaluation error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
