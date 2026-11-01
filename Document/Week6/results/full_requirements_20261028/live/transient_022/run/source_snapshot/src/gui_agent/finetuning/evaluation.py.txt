"""Auditable offline GUI action evaluation, separate from desktop task E2E.

Model libraries are imported only inside the isolated HF worker. Scoring and
aggregation need no GPU, model download, endpoint, or desktop controller.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import platform
import queue
import random
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "week5.offline-evaluation.v1"
CELLS = ("base_original", "base_improved", "adapter_original", "adapter_improved")
CLICK_TYPES = frozenset({"click", "double_click", "right_click"})
POINT_TYPES = CLICK_TYPES | {"move", "scroll"}
_REQUIRED = {
    **{name: {"x", "y"} for name in POINT_TYPES},
    "scroll": {"x", "y", "scroll_amount"},
    "type_text": {"text"},
    "key_press": {"key"},
    "hotkey": {"keys"},
    "drag": {"start", "end"},
    "wait": {"duration"},
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def strict_json_loads(text: str) -> Any:
    """Read the complete reply; never extract a fence, repair, or coerce it."""
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def validate_actions(value: Any) -> list[dict[str, Any]]:
    """The preregistered native-pixel DesktopAction subset, with strict types."""
    if not isinstance(value, list) or not value:
        raise ValueError("expected a non-empty JSON action array")
    for index, action in enumerate(value):
        if not isinstance(action, dict):
            raise ValueError(f"action {index} must be an object")  # noqa: TRY004 - uniform validation API
        kind = action.get("action_type")
        if not isinstance(kind, str) or kind not in _REQUIRED:
            raise ValueError(f"action {index} has an unknown action_type")
        required = _REQUIRED[kind] | {"action_type"}
        if set(action) != required:
            raise ValueError(f"action {index} has missing or unexpected fields")
        for name in ("x", "y", "scroll_amount"):
            if name in action and type(action[name]) is not int:
                raise ValueError(f"action {index}.{name} must be an integer")
        for name in ("text", "key"):
            if name in action and (not isinstance(action[name], str) or not action[name]):
                raise ValueError(f"action {index}.{name} must be a non-empty string")
        if "keys" in action:
            keys = action["keys"]
            if (
                not isinstance(keys, list)
                or not keys
                or any(not isinstance(key, str) or not key for key in keys)
            ):
                raise ValueError(f"action {index}.keys must be non-empty strings")
        for name in ("start", "end"):
            if name in action:
                point = action[name]
                if (
                    not isinstance(point, dict)
                    or set(point) != {"x", "y"}
                    or any(type(point[axis]) is not int for axis in ("x", "y"))
                ):
                    raise ValueError(f"action {index}.{name} must be an integer point")
        if "duration" in action:
            duration = action["duration"]
            if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
                raise ValueError(f"action {index}.duration must be finite and nonnegative")
    return value


def parse_generation(text: str) -> tuple[list[dict[str, Any]] | None, bool, str | None]:
    try:
        payload = strict_json_loads(text)
    except (ValueError, TypeError) as exc:
        return None, False, str(exc)
    try:
        return validate_actions(payload), True, None
    except ValueError as exc:
        return None, True, str(exc)


def _bbox(annotation: Any, width: int, height: int) -> dict[str, int] | None:
    if not isinstance(annotation, Mapping) or annotation.get("target_bbox") is None:
        return None
    box = annotation["target_bbox"]
    if (
        not isinstance(box, Mapping)
        or set(box) != {"left", "top", "right", "bottom"}
        or any(type(value) is not int for value in box.values())
    ):
        raise ValueError("target_bbox must contain four integer edges")
    if not (0 <= box["left"] < box["right"] <= width and 0 <= box["top"] < box["bottom"] <= height):
        raise ValueError("target_bbox is outside the native image")
    return dict(box)


def score_generation(
    target: Sequence[dict[str, Any]],
    generation: str,
    *,
    width: int,
    height: int,
    action_annotations: Sequence[dict[str, Any]] = (),
    status: str = "ok",
) -> dict[str, Any]:
    """Invalid/failed requests have no credited action, even with a partial JSON reply.

    Action types are aligned by index with denominator max(target,predicted)
    (extra actions are errors). Text retains whitespace and Unicode exactly.
    Click hit uses half-open pixel boxes. Missing click coordinates incur sqrt(2)
    in the all-target distance as an explicit penalty; observed distances and
    their coverage are also reported so the penalty cannot look like a reading.
    """
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("positive native image dimensions are required")
    expected = validate_actions(list(target))
    if action_annotations and len(action_annotations) != len(expected):
        raise ValueError("action_annotations must align with target actions")
    parsed, syntax_valid, parse_error = parse_generation(generation)
    valid = parsed is not None and status == "ok"
    predicted = parsed if valid else []
    slots = max(len(expected), len(predicted))
    type_correct = sum(
        index < len(predicted) and action["action_type"] == predicted[index]["action_type"]
        for index, action in enumerate(expected)
    )
    text_total = text_correct = clicks = hit_total = hits = distance_missing = 0
    distances: list[float] = []
    penalized_distances: list[float] = []
    bounds_errors = 0
    for index, action in enumerate(expected):
        candidate = predicted[index] if index < len(predicted) else None
        same_kind = candidate is not None and candidate["action_type"] == action["action_type"]
        if action["action_type"] == "type_text":
            text_total += 1
            text_correct += bool(same_kind and candidate["text"] == action["text"])
        if action["action_type"] not in CLICK_TYPES:
            continue
        clicks += 1
        annotation = action_annotations[index] if action_annotations else {}
        box = _bbox(annotation, width, height)
        hit_total += box is not None
        if same_kind:
            x, y = candidate["x"], candidate["y"]
            in_bounds = 0 <= x < width and 0 <= y < height
            bounds_errors += not in_bounds
            if box is not None:
                hits += bool(
                    in_bounds
                    and box["left"] <= x < box["right"]
                    and box["top"] <= y < box["bottom"]
                )
            distance = math.hypot((x - action["x"]) / width, (y - action["y"]) / height)
            distances.append(distance)
            penalized_distances.append(distance)
        else:
            distance_missing += 1
            penalized_distances.append(math.sqrt(2.0))
    return {
        "syntax_json_valid": syntax_valid and status == "ok",
        "strict_json_valid": valid,
        "parse_error": parse_error,
        "predicted_actions": parsed,
        "strict_action_exact": valid and predicted == expected,
        "action_type_correct": type_correct,
        "action_type_slots": slots,
        "action_type_sequence_exact": valid
        and len(predicted) == len(expected)
        and type_correct == len(expected),
        "text_exact_correct": text_correct,
        "text_exact_total": text_total,
        "click_total": clicks,
        "click_bbox_eligible": hit_total,
        "click_bbox_hits": hits,
        "click_bbox_absent": clicks - hit_total,
        "click_distance_missing": distance_missing,
        "click_normalized_distances_observed": distances,
        "click_normalized_distances_all_targets": penalized_distances,
        "click_prediction_out_of_bounds": bounds_errors,
        "extra_predicted_actions": max(0, len(predicted) - len(expected)),
    }


def _paired_comparisons(groups: dict, study_samples: dict) -> list[dict[str, Any]]:
    paired_comparisons = []
    for study_id in sorted(study_samples):
        for label, before_cell, after_cell in (
            ("adapter_effect_original", "base_original", "adapter_original"),
            ("adapter_effect_improved", "base_improved", "adapter_improved"),
            ("prompt_effect_base", "base_original", "base_improved"),
            ("prompt_effect_adapter", "adapter_original", "adapter_improved"),
        ):
            before = {
                row["sample_id"]: (row, scores, status)
                for row, scores, status in groups.get((study_id, before_cell), [])
            }
            after = {
                row["sample_id"]: (row, scores, status)
                for row, scores, status in groups.get((study_id, after_cell), [])
            }
            if not before or not after:
                continue
            common = sorted(before.keys() & after.keys())
            outcomes = Counter()
            json_delta = 0
            latency_differences = []
            eligible = set(before) == set(after)
            for sample_id in common:
                first, first_score, first_status = before[sample_id]
                second, second_score, second_status = after[sample_id]
                eligible &= (
                    first.get("inference_attempted", True)
                    and second.get("inference_attempted", True)
                    and first_status in {"ok", "error", "timeout"}
                    and second_status in {"ok", "error", "timeout"}
                )
                a, b = first_score["strict_action_exact"], second_score["strict_action_exact"]
                outcomes[
                    "both_success"
                    if a and b
                    else "improved"
                    if b
                    else "worsened"
                    if a
                    else "both_failed"
                ] += 1
                json_delta += int(second_score["strict_json_valid"]) - int(
                    first_score["strict_json_valid"]
                )
                if first.get("duration_ms") is not None and second.get("duration_ms") is not None:
                    latency_differences.append(second["duration_ms"] - first["duration_ms"])
            n = len(common)
            paired_comparisons.append(
                {
                    "study_id": study_id,
                    "comparison": label,
                    "before_cell": before_cell,
                    "after_cell": after_cell,
                    "paired_samples": n,
                    "comparison_eligible": eligible,
                    "outcomes": dict(outcomes),
                    "strict_success_rate_delta_after_minus_before": (
                        outcomes["improved"] - outcomes["worsened"]
                    )
                    / n
                    if n
                    else None,
                    "strict_json_rate_delta_after_minus_before": json_delta / n if n else None,
                    "paired_duration_n": len(latency_differences),
                    "mean_duration_delta_ms_after_minus_before": sum(latency_differences)
                    / len(latency_differences)
                    if latency_differences
                    else None,
                }
            )
    return paired_comparisons


def nearest_rank(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0,1]")
    if not ordered:
        return None
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in ordered
    ):
        raise ValueError("durations/distances must be finite and nonnegative")
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def distribution(values: Iterable[float]) -> dict[str, Any]:
    data = list(values)
    return {
        "n": len(data),
        "p50": nearest_rank(data, 0.50),
        "p90": nearest_rank(data, 0.90),
        "p95": nearest_rank(data, 0.95),
        "mean": sum(data) / len(data) if data else None,
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "correct": numerator,
        "total": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    require_final_verdict: bool = True,
) -> dict[str, Any]:
    """Recompute scores from raw target/generation; distrust cached score fields.

    Final provenance events protect an interrupted run and a changed checkpoint
    from receiving success credit. Every recorded request stays in its denominator.
    Different frozen studies are never silently merged into a model comparison.
    """
    verdicts = {}
    for record in records:
        if record.get("record_type") == "provenance_verdict":
            if record["run_id"] in verdicts:
                raise ValueError("duplicate final provenance verdict")
            verdicts[record["run_id"]] = record
    groups: dict[tuple[str, str], list[tuple[Mapping[str, Any], dict[str, Any], str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    sample_fingerprints: dict[tuple[str, str], str] = {}
    for record in records:
        if record.get("record_type") != "request":
            continue
        if record["cell"] not in CELLS:
            raise ValueError("unknown evaluation cell")
        sample_key = (record["study_id"], record["sample_id"])
        fingerprint = value_sha256(
            {
                key: record.get(key)
                for key in (
                    "target",
                    "image_sha256",
                    "image_file_sha256",
                    "source_sample_sha256",
                    "instruction",
                )
            }
        )
        if sample_key in sample_fingerprints and sample_fingerprints[sample_key] != fingerprint:
            raise ValueError("the same heldout sample differs across comparison cells")
        sample_fingerprints[sample_key] = fingerprint
        identity = (record["study_id"], record["cell"], record["sample_id"])
        if identity in seen:
            raise ValueError(f"duplicate evaluation request: {identity}")
        seen.add(identity)
        status = record["status"]
        verdict = verdicts.get(record["run_id"])
        if verdict is not None and not verdict.get("unchanged", False):
            status = "provenance_changed"
        elif require_final_verdict and verdict is None:
            status = "provenance_pending"
        target = record["target"]
        scores = score_generation(
            target["actions"],
            record.get("generation", ""),
            width=target["image_width"],
            height=target["image_height"],
            action_annotations=target.get("action_annotations", []),
            status=status,
        )
        groups.setdefault((record["study_id"], record["cell"]), []).append((record, scores, status))
    cells: list[dict[str, Any]] = []
    study_samples: dict[str, dict[str, set[str]]] = {}
    for (study_id, cell), rows in sorted(groups.items()):
        n = len(rows)
        sums = Counter()
        all_duration: list[float] = []
        successful_duration: list[float] = []
        worker_duration: list[float] = []
        observed: list[float] = []
        penalized: list[float] = []
        for record, scores, _ in rows:
            sums.update({key: value for key, value in scores.items() if type(value) in (bool, int)})
            duration = record.get("duration_ms")
            if duration is not None:
                all_duration.append(duration)
                if scores["strict_action_exact"]:
                    successful_duration.append(duration)
            if record.get("worker_inference_ms") is not None:
                worker_duration.append(record["worker_inference_ms"])
            observed.extend(scores["click_normalized_distances_observed"])
            penalized.extend(scores["click_normalized_distances_all_targets"])
        sample_ids = {record["sample_id"] for record, _, _ in rows}
        study_samples.setdefault(study_id, {})[cell] = sample_ids
        cells.append(
            {
                "study_id": study_id,
                "cell": cell,
                "requests": n,
                "statuses": dict(Counter(status for _, _, status in rows)),
                "syntax_json_valid": _rate(sums["syntax_json_valid"], n),
                "strict_json_valid": _rate(sums["strict_json_valid"], n),
                "strict_action_exact": _rate(sums["strict_action_exact"], n),
                "action_type_sequence_exact": _rate(sums["action_type_sequence_exact"], n),
                "action_type_accuracy": _rate(
                    sums["action_type_correct"], sums["action_type_slots"]
                ),
                "text_exact_match": _rate(sums["text_exact_correct"], sums["text_exact_total"]),
                "click_bbox_hit": _rate(sums["click_bbox_hits"], sums["click_bbox_eligible"]),
                "click_bbox_absent": sums["click_bbox_absent"],
                "click_distance_missing": sums["click_distance_missing"],
                "click_prediction_out_of_bounds": sums["click_prediction_out_of_bounds"],
                "click_distance_observed": distribution(observed),
                "click_distance_all_targets_penalty_sqrt2": distribution(penalized),
                "percall_e2e_ms_all_recorded_requests": distribution(all_duration),
                "percall_e2e_ms_strict_action_success": distribution(successful_duration),
                "worker_preprocess_generate_decode_ms": distribution(worker_duration),
                "duration_missing": n - len(all_duration),
                "inference_attempted": sum(
                    record.get("inference_attempted", True) for record, _, _ in rows
                ),
                "extra_predicted_actions": sums["extra_predicted_actions"],
            }
        )
    comparisons = []
    for study_id, by_cell in sorted(study_samples.items()):
        present = set(by_cell)
        shared = set.intersection(*by_cell.values()) if by_cell else set()
        union = set.union(*by_cell.values()) if by_cell else set()
        all_attempted = all(
            record.get("inference_attempted", True)
            for (row_study, _), rows in groups.items()
            if row_study == study_id
            for record, _, _ in rows
        )
        terminal_provenance = all(
            status in {"ok", "error", "timeout"}
            for (row_study, _), rows in groups.items()
            if row_study == study_id
            for _, _, status in rows
        )
        comparisons.append(
            {
                "study_id": study_id,
                "cells_present": sorted(present),
                "same_recorded_sample_sets": shared == union,
                "complete_2x2_same_samples": present == set(CELLS)
                and shared == union
                and all_attempted
                and terminal_provenance,
                "shared_sample_count": len(shared),
                "union_sample_count": len(union),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "offline_gui_action_generation",
        "desktop_task_e2e": False,
        "quantile_method": "nearest_rank_ceil(p*n)",
        "strict_success_definition": "complete native-pixel action array equals the target exactly",
        "latency_scope": "parent request dispatch to response/error; model load and warmup excluded",
        "missing_distance_penalty": math.sqrt(2.0),
        "cells": cells,
        "comparisons": comparisons,
        "paired_comparisons": _paired_comparisons(groups, study_samples),
        "warmup_calls": sum(record.get("record_type") == "warmup" for record in records),
        "startup_errors": [
            dict(record) for record in records if record.get("record_type") == "startup_error"
        ],
    }


def load_prepared(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path).resolve()
    samples = []
    ids: set[str] = set()
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        sample = strict_json_loads(line)
        if not isinstance(sample, dict):
            raise ValueError(f"line {line_number}: prepared sample must be an object")  # noqa: TRY004
        for key in ("sample_id", "group_id", "instruction", "source_split"):
            if not isinstance(sample.get(key), str) or not sample[key]:
                raise ValueError(f"line {line_number}: {key} is required")
        if sample["sample_id"] in ids:
            raise ValueError(f"duplicate sample_id: {sample['sample_id']}")
        ids.add(sample["sample_id"])
        if sample["source_split"].casefold() == "train":
            raise ValueError("evaluation refuses an official training-split sample")
        if "source" not in sample:
            raise ValueError("prepared sample needs source provenance")
        width, height = sample.get("image_width"), sample.get("image_height")
        score_generation(
            sample["actions"],
            "",
            width=width,
            height=height,
            action_annotations=sample.get("action_annotations", []),
        )
        for action in sample["actions"]:
            points = (
                [action]
                if action["action_type"] in POINT_TYPES
                else [action["start"], action["end"]]
                if action["action_type"] == "drag"
                else []
            )
            if any(not (0 <= point["x"] < width and 0 <= point["y"] < height) for point in points):
                raise ValueError("target coordinate is outside the native image")
        image = Path(sample["image_path"])
        if not image.is_absolute():
            image = dataset_path.parent / image
        image = image.resolve()
        from PIL import Image

        with Image.open(image) as loaded:
            if loaded.size != (width, height):
                raise ValueError(f"native dimensions do not match {sample['sample_id']}")
            rgb = loaded.convert("RGB")
            decoded_hash = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
        if sample.get("image_sha256") and sample["image_sha256"] != decoded_hash:
            raise ValueError("image hash differs from the prepared decoded RGB metadata")
        samples.append(
            {
                **sample,
                "image_path": str(image),
                "image_sha256": decoded_hash,
                "image_file_sha256": file_sha256(image),
                "source_sample_sha256": value_sha256(sample),
            }
        )
    if not samples:
        raise ValueError("prepared heldout dataset is empty")
    return samples


def select_samples(
    samples: Sequence[dict[str, Any]], *, seed: int = 42, max_samples: int = 48
) -> list[dict[str, Any]]:
    if max_samples not in (48, 64):
        raise ValueError("the preregistered cap must be 48 or 64")
    ordered = sorted(samples, key=lambda sample: sample["sample_id"])
    if len(ordered) > max_samples:
        ordered = random.Random(seed).sample(ordered, max_samples)
    return sorted(ordered, key=lambda sample: sample["sample_id"])


def safe_prompt_input(sample: Mapping[str, Any], prompt_variant: str = "original") -> dict[str, Any]:
    """The only sample-derived text passed to the prompt builder."""
    from .prompts import prompt_input

    return prompt_input(sample, prompt_variant)


def artifact_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"artifact directory missing: {root}")
    files = {
        str(file.relative_to(root)).replace("\\", "/"): file_sha256(file)
        for file in sorted(root.rglob("*"))
        if file.is_file() and ".cache" not in file.parts
    }
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    return {"path": str(root), "files": files, "sha256": value_sha256(files)}


def code_manifest(repository: str | Path) -> dict[str, str]:
    root = Path(repository)
    paths = list((root / "src/gui_agent/finetuning").glob("*.py"))
    paths += list((root / "scripts").glob("week5_*.py"))
    return {
        str(path.relative_to(root)).replace("\\", "/"): file_sha256(path) for path in sorted(paths)
    }


def runtime_versions() -> dict[str, str | None]:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "Pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _spatial_inference_module(protocol: str) -> Any:
    if protocol == "native_spatial_grid_head":
        from . import native_spatial_grid_inference as module
    elif protocol == "native_spatial_head":
        from . import native_spatial_inference as module
    else:
        raise ValueError("explicit spatial target protocol required")
    return module


@dataclass(frozen=True)
class HFConfig:
    training_config: dict[str, Any]
    max_new_tokens: int = 256
    seed: int = 42
    output_constraint: str = "none"
    input_image_protocol: str = "native"
    visual_asset_manifest_path: str | None = None
    visual_asset_manifest_sha256: str | None = None
    visual_original_root: str | None = None
    visual_asset_root: str | None = None
    visual_prompt_version: str | None = None
    target_version: str | None = None
    spatial_head_artifact_dir: str | None = None
    spatial_head_role: str | None = None
    spatial_head_manifest_sha256: str | None = None
    spatial_expected_bindings: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.training_config, dict) or self.max_new_tokens <= 0:
            raise ValueError("invalid training configuration/token budget")
        from .action_constraints import OUTPUT_CONSTRAINTS

        if self.output_constraint not in OUTPUT_CONSTRAINTS:
            raise ValueError("unknown output_constraint")
        if self.training_config.get("target_protocol", "native") == "candidate_compact" and self.output_constraint != "none":
            raise ValueError("candidate_compact cannot use the native json_actions output_constraint")
        from .native_spatial_inference import HEAD_CONFIG_FIELDS

        if self.training_config.get("target_protocol") in {"native_spatial_head", "native_spatial_grid_head"}:
            _spatial_inference_module(self.training_config["target_protocol"]).validate_spatial_config(self)
        elif self.target_version is not None or any(getattr(self, field) is not None for field in HEAD_CONFIG_FIELDS):
            raise ValueError("spatial head authority requires its explicit target protocol")
        # The raster validator imports value_sha256 from this module. Keep this
        # dependency lazy to permit either module to be imported first.
        from .visual_pointer_inference import validate_worker_config

        validate_worker_config(self)


def _hf_worker(connection: Any, config: HFConfig, adapter_path: str | None) -> None:
    """Labels are never sent to this process. stdout is not a control protocol."""
    started = time.perf_counter()
    try:
        from .visual_pointer_inference import validate_worker_config

        validate_worker_config(config)
        spatial = config.training_config.get("target_protocol") in {"native_spatial_head", "native_spatial_grid_head"}
        head_identity, spatial_head, spatial_decoder = None, None, None
        if spatial:
            spatial_api = _spatial_inference_module(config.training_config["target_protocol"])
            authority_fields = spatial_api.authority_fields
            build_spatial_decoder = spatial_api.build_spatial_decoder
            inspect_head = spatial_api.inspect_head
            load_resident_head = spatial_api.load_resident_head
            validate_head_role = spatial_api.validate_head_role

            validate_head_role(config, adapter_path)
            head_identity = inspect_head(config)
        import torch
        from PIL import Image

        from .training import TrainingConfig, load_model_and_processor

        visual_registry = None
        if config.input_image_protocol in {"visual_pointer_v2", "visual_pointer_gallery_v3"}:
            from .visual_pointer_inference import registry_from_config, registry_identity

            visual_registry = registry_from_config(config)

        model, processor = load_model_and_processor(
            TrainingConfig.from_dict(config.training_config),
            adapter_path=adapter_path,
            for_training=False,
        )
        model.eval()
        if spatial:
            spatial_head, head_identity = load_resident_head(model, config, expected_identity=head_identity)
            spatial_decoder = build_spatial_decoder(config, processor=processor, model=model)
        constraint = None
        if config.output_constraint == "json_actions":
            from .action_constraints import NativeActionConstraint

            constraint = NativeActionConstraint(processor.tokenizer, model.generation_config.eos_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        connection.send(
            {
                "kind": "ready",
                "input_image_protocol": config.input_image_protocol,
                **(authority_fields(head_identity) if spatial else {}),
                **({"visual_asset_registry": registry_identity(visual_registry)}
                   if visual_registry is not None else {}),
                **({"visual_prompt_version": config.visual_prompt_version} if config.visual_prompt_version is not None else {}),
                "load_ms": (time.perf_counter() - started) * 1000,
                "versions": runtime_versions(),
                "cuda_version": torch.version.cuda,
                "device_map": getattr(model, "hf_device_map", None),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                **({"output_constraint": constraint.descriptor, "output_constraint_eos": constraint.eos_descriptor,
                    "model_generation_config": model.generation_config.to_dict(),
                    "model_pad_token_id": model.generation_config.pad_token_id,
                    "generate_explicit_kwargs": {"max_new_tokens": config.max_new_tokens,
                                                 "do_sample": False, "use_cache": True},
                    "constraint_hook": "prefix_allowed_tokens_fn"} if constraint is not None else {}),
            }
        )
    except Exception as exc:  # noqa: BLE001 - isolated loader failures must reach the parent ledger
        connection.send(
            {
                "kind": "load_error",
                "error": f"{type(exc).__name__}: {exc}",
                "load_ms": (time.perf_counter() - started) * 1000,
                "versions": runtime_versions(),
            }
        )
        connection.close()
        return
    while True:
        try:
            request = connection.recv()
        except EOFError:
            break
        if request is None:
            break
        content = ""
        rendered = None
        joint_output = None
        call_started = time.perf_counter()
        try:
            descriptor = None
            if visual_registry is not None:
                from .visual_pointer_inference import (
                    assert_visual_raster,
                    input_ack,
                    validate_visual_request,
                )

                if spatial:
                    checked_spatial_request = spatial_api.checked_spatial_request

                    request = checked_spatial_request(request, head_identity=head_identity)
                descriptor = validate_visual_request(request, registry=visual_registry)
                if descriptor.input_image_protocol != config.input_image_protocol:
                    raise ValueError("request and worker configured visual input protocol differ")
                if descriptor.visual_prompt_version != config.visual_prompt_version:
                    raise ValueError("request and worker configured visual_prompt_version differ")
                image, prompt = descriptor.to_pil(), descriptor.prompt
            else:
                if not isinstance(request, dict) or set(request) != {"image_path", "prompt"}:
                    raise ValueError("native worker requires an exclusively native request")
                if any(not isinstance(request[field], str) or not request[field] for field in ("image_path", "prompt")):
                    raise ValueError("native request requires image path and prompt strings")
                with Image.open(request["image_path"]) as source:
                    image = source.convert("RGB")
                prompt = request["prompt"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
            if config.input_image_protocol == "visual_pointer_gallery_v3" and inputs["input_ids"].shape[1] > 1536:
                raise ValueError("gallery image-expanded input exceeds1536; no truncation performed")
            inputs = inputs.to(model.device)
            if descriptor is not None:
                descriptor.assert_current()
                assert_visual_raster(descriptor, image)
            if spatial:
                grid_arguments = {}
                if config.training_config["target_protocol"] == "native_spatial_grid_head":
                    grid_arguments["grid_input"] = spatial_api.build_grid_input(
                        request["visible_sample"], descriptor, inputs,
                        image_token_id=model.config.image_token_id,
                        expected_bindings=config.spatial_expected_bindings,
                    )
                joint_output = spatial_api.generate_joint_output(model, spatial_head, inputs, spatial_decoder,
                    native_width=descriptor.width, native_height=descriptor.height, **grid_arguments)
                content = joint_output["generation"]
                generated_tokens = len(joint_output["generated_token_ids"])
            else:
                with torch.inference_mode():
                    if constraint is None:
                        generated = model.generate(
                            **inputs, max_new_tokens=config.max_new_tokens, do_sample=False, use_cache=True
                        )
                    else:
                        generated = model.generate(
                            **inputs, max_new_tokens=config.max_new_tokens, do_sample=False, use_cache=True,
                            prefix_allowed_tokens_fn=constraint.new_prefix_function(),
                        )
                suffix = generated[:, inputs["input_ids"].shape[1] :]
                content = processor.batch_decode(
                    suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                generated_tokens = suffix.shape[1]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            truncated = (generated_tokens >= config.max_new_tokens
                         and (joint_output["generated_token_ids"][-1] != spatial_decoder.eos_id if spatial
                              else int(suffix[0, -1]) not in eos_ids))
            if descriptor is not None:
                assert_visual_raster(descriptor, image)
            acknowledgement = input_ack(descriptor, visual_registry) if descriptor is not None else None
            if spatial:
                spatial_input_ack = spatial_api.spatial_input_ack

                acknowledgement = spatial_input_ack(acknowledgement, head_identity=head_identity)
            connection.send(
                {
                    "kind": "response",
                    **(joint_output if spatial else {}),
                    "generation": content,
                    "status": joint_output["status"] if spatial else "error" if truncated else "ok",
                    "error": joint_output["error"] if spatial else "max_new_tokens reached without EOS" if truncated else None,
                    # The complete synchronized forward pass returned normally.
                    # Token-budget failure is a failed answer, not a dead worker.
                    "error_kind": "token_limit" if truncated else None,
                    "generated_tokens": generated_tokens,
                    "rendered_prompt": rendered,
                    "worker_inference_ms": (time.perf_counter() - call_started) * 1000,
                    **({"input_ack": acknowledgement} if acknowledgement is not None else {}),
                    **({"output_constraint": constraint.descriptor, "output_constraint_eos": constraint.eos_descriptor,
                        "last_token_id": int(suffix[0, -1]),
                        "resolved_eos_token_ids": list(eos if isinstance(eos, list) else [eos]),
                        "output_constraint_eos_token_ids": constraint.eos_descriptor["constrained_eos_token_ids"]}
                       if constraint is not None else {}),
                }
            )
        except Exception as exc:  # noqa: BLE001 - every inference failure becomes an auditable reply
            connection.send(
                {
                    "kind": "response",
                    **(joint_output if joint_output is not None else {}),
                    "generation": content,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "rendered_prompt": rendered,
                    "worker_inference_ms": (time.perf_counter() - call_started) * 1000,
                }
            )
    connection.close()


class HFWorker:
    """One resident model per cell; hard wall deadlines terminate its process."""

    def __init__(
        self,
        config: HFConfig,
        *,
        adapter_path: str | None = None,
        load_timeout_seconds: float = 600,
    ) -> None:
        if not math.isfinite(load_timeout_seconds) or load_timeout_seconds <= 0:
            raise ValueError("model loading timeout must be finite and positive")
        from .visual_pointer_inference import validate_worker_config

        validate_worker_config(config)
        self.target_protocol = config.training_config.get("target_protocol", "native")
        self.spatial_head_artifact = None
        if self.target_protocol in {"native_spatial_head", "native_spatial_grid_head"}:
            spatial_api = _spatial_inference_module(self.target_protocol)
            inspect_head, validate_head_role = spatial_api.inspect_head, spatial_api.validate_head_role

            validate_head_role(config, adapter_path)
            self.spatial_head_artifact = inspect_head(config)
        started = time.perf_counter()
        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=_hf_worker, args=(child_connection, config, adapter_path), daemon=True
        )
        self.request_count = 0
        self.input_image_protocol = config.input_image_protocol
        self.visual_asset_registry = None
        if self.input_image_protocol in {"visual_pointer_v2", "visual_pointer_gallery_v3"}:
            from .visual_pointer_inference import registry_from_config, registry_identity

            self.visual_asset_registry = registry_identity(registry_from_config(config))
        try:
            self.process.start()
            child_connection.close()
            remaining = load_timeout_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("HF worker model loading timed out")
            # A readable partial pipe header is not a completed handshake. Keep
            # the entire blocking recv off the caller's deadline/control thread.
            replies: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

            def receive_ready() -> None:
                try:
                    replies.put(("ready", self.connection.recv()))
                except BaseException as exc:  # noqa: BLE001 - propagate transport cancellation to owner
                    replies.put(("error", exc))

            threading.Thread(target=receive_ready, daemon=True).start()
            try:
                kind, response = replies.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("HF worker model loading timed out") from exc
            if kind == "error":
                if isinstance(response, EOFError):
                    raise RuntimeError("HF worker exited while loading") from response
                raise response
            self.ready = response
            if self.ready["kind"] != "ready":
                raise RuntimeError(self.ready.get("error", "HF worker failed to load"))
            if self.ready.get("input_image_protocol", "native") != self.input_image_protocol:
                raise RuntimeError("HF worker ready input_image_protocol differs")
            if self.visual_asset_registry is not None and self.ready.get("visual_asset_registry") != self.visual_asset_registry:
                raise RuntimeError("HF worker ready trusted visual registry differs")
            from .visual_pointer_prompts import declared_prompt_version

            if declared_prompt_version(self.ready) != config.visual_prompt_version:
                raise RuntimeError("HF worker ready visual_prompt_version differs")
            if self.input_image_protocol == "native" and self.ready.get("visual_asset_registry") is not None:
                raise RuntimeError("native HF worker ready cannot carry a visual registry")
            if self.target_protocol in {"native_spatial_head", "native_spatial_grid_head"}:
                assert_spatial_ready = _spatial_inference_module(self.target_protocol).assert_spatial_ready

                assert_spatial_ready(self.ready, head_identity=self.spatial_head_artifact,
                                     visual_registry_identity=self.visual_asset_registry)
            self.ready["process_id"] = self.process.pid
        except BaseException:
            # __init__ failures never assign a worker to the caller. This owner
            # must release a partially started process, including on Ctrl+C.
            self.close()
            raise
        finally:
            child_connection.close()

    def request(self, image_path: str, prompt: str, *, timeout_seconds: float) -> dict[str, Any]:
        if getattr(self, "input_image_protocol", "native") != "native":
            raise ValueError("visual worker cannot accept a native request")
        return self._request_payload({"image_path": image_path, "prompt": prompt}, timeout_seconds=timeout_seconds)

    def request_visual(self, envelope: Mapping[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
        if getattr(self, "target_protocol", "native") in {"native_spatial_head", "native_spatial_grid_head"}:
            raise ValueError("spatial worker requires an explicit spatial request")
        if getattr(self, "input_image_protocol", "native") not in {"visual_pointer_v2", "visual_pointer_gallery_v3"}:
            raise ValueError("native worker cannot accept a visual request")
        from .visual_pointer_inference import checked_envelope

        payload = checked_envelope(envelope)
        if payload["input_image_protocol"] != self.input_image_protocol:
            raise ValueError("worker and request visual input protocol differ")
        return self._request_payload(payload, timeout_seconds=timeout_seconds)

    def request_spatial(self, request: Mapping[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
        if getattr(self, "target_protocol", "native") not in {"native_spatial_head", "native_spatial_grid_head"}:
            raise ValueError("legacy worker cannot accept a spatial request")
        checked_spatial_request = _spatial_inference_module(self.target_protocol).checked_spatial_request
        from .visual_pointer_inference import checked_envelope

        envelope = checked_spatial_request(request, head_identity=self.spatial_head_artifact)
        checked_envelope(envelope)
        return self._request_payload(dict(request), timeout_seconds=timeout_seconds)

    def _request_payload(self, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("request timeout must be finite and positive")
        started = time.perf_counter()
        self.request_count += 1
        # poll() alone is not a hard timeout: a partial pipe header makes it
        # readable while recv() can still block. The deadline covers send+recv.
        replies: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def exchange() -> None:
            try:
                self.connection.send(payload)
                replies.put(("response", self.connection.recv()))
            except (EOFError, BrokenPipeError, OSError, ValueError) as exc:
                replies.put(("error", f"{type(exc).__name__}: {exc}"))

        thread = threading.Thread(target=exchange, daemon=True)
        thread.start()
        try:
            kind, result = replies.get(timeout=timeout_seconds)
        except queue.Empty:
            self.close()
            return {
                "status": "timeout",
                "generation": "",
                "error": "hard percall timeout",
                "duration_ms": (time.perf_counter() - started) * 1000,
                "worker_inference_ms": None,
            }
        if kind == "error":
            self.close()
            return {
                "status": "error",
                "generation": "",
                "error": result,
                "duration_ms": (time.perf_counter() - started) * 1000,
                "worker_inference_ms": None,
            }
        result["duration_ms"] = (time.perf_counter() - started) * 1000
        return result

    def close(self) -> None:
        try:
            if self.process.is_alive():
                self.process.terminate()
            # join before start raises AssertionError. A start() failure still
            # needs pipe cleanup, and a partially started child has a pid.
            if self.process.pid is not None:
                self.process.join(timeout=10)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=10)
        finally:
            self.connection.close()
