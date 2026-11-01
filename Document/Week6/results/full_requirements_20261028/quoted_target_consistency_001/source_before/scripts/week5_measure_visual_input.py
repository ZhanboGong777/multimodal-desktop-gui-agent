"""CPU-only admission gate exercising the production visual training encoder."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from week5_prepare_visual_input import bound_read, merge_binding, safe_path, sha, write_new

from gui_agent.finetuning.evaluation import canonical_json, strict_json_loads
from gui_agent.finetuning.pointer_protocol import pointer_user_prompt
from gui_agent.finetuning.training import (
    TrainingConfig,
    encode_training_sample,
    load_prepared_records,
)
from gui_agent.finetuning.visual_pointer import VISUAL_PREFIX


def fingerprint(tensor) -> dict:
    import torch

    if tensor.device.type != "cpu":
        raise ValueError("processor output left the registered CPU gate")
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "value_sha256": hashlib.sha256(raw).hexdigest()}


class ObservedProcessor:
    """Delegate to a real processor while recording the exact input and output."""
    def __init__(self, processor):
        self.processor = processor
        self.calls = []
        self.templates = []

    def __getattr__(self, key):
        return getattr(self.processor, key)

    def apply_chat_template(self, messages, **kwargs):
        rendered = self.processor.apply_chat_template(messages, **kwargs)
        self.templates.append({"messages": copy.deepcopy(messages), "kwargs": dict(kwargs),
                               "rendered_text": rendered})
        return rendered

    def __call__(self, **kwargs):
        if set(kwargs) != {"text", "images", "padding", "truncation", "return_tensors"}:
            raise ValueError("encoder processor signature drift or truncation option")
        if (kwargs["padding"] is not False or kwargs["truncation"] is not False
                or kwargs["return_tensors"] != "pt"):
            raise ValueError("registered nonpadding tensor call differs")
        if len(kwargs["images"]) != 1 or len(kwargs["text"]) != 1:
            raise ValueError("one complete original view per processor call required")
        image = kwargs["images"][0]
        raster_sha = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
        batch = self.processor(**kwargs)
        self.calls.append({"input_rgb_sha256": raster_sha, "input_mode": image.mode,
                           "input_width": image.width, "input_height": image.height,
                           "rendered_text": kwargs["text"][0],
                           "input_ids": batch["input_ids"][0].tolist(),
                           "pixel_values": fingerprint(batch["pixel_values"]),
                           "image_grid_thw": fingerprint(batch["image_grid_thw"]),
                           "grid_values": batch["image_grid_thw"].tolist()})
        return batch


def stats(values: list[int]) -> dict:
    return {"count": len(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None}


def run(plan_path: Path, expected_sha: str) -> int:
    started = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    bound = {str(plan_path): expected_sha}
    error = None
    cancelled = False
    result = None
    output = None
    proxy = None
    owned_publication = None
    owned_files: set[Path] = set()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU-only CUDA_VISIBLE_DEVICES='' required")
        import torch
        from transformers import AutoProcessor

        if torch.cuda.is_initialized():
            raise ValueError("CUDA was initialized before CPU admission")
        plan = strict_json_loads(bound_read(plan_path, expected_sha).decode("utf-8"))
        if (plan["schema"], plan["status"]) != ("week5.visual_input_processor_plan.v1", "registered_before_execution"):
            raise ValueError("caller-pinned processor plan required")
        for name, digest in plan["bound_sha256"].items():
            path = safe_path(name, ROOT)
            bound_read(path, digest)
            merge_binding(bound, path, digest)
        if bound.get(str(Path(__file__).resolve())) != sha(Path(__file__)):
            raise ValueError("actual CPU helper is unregistered")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow", "numpy")}
        if versions != plan["runtime_versions"]:
            raise ValueError("registered CPU processor runtime changed")
        if plan["budgets"] != {"source_views": 404, "available_visual_rows": 338,
                              "refused_layout": 66, "processor_loads": 1,
                              "processor_encode_calls": 676, "model_loads": 0,
                              "training_runs": 0, "gui_calls": 0}:
            raise ValueError("fixed actual processor budget differs")
        output = safe_path(plan["output_dir"], ROOT / "outputs", existing=False)
        if output.exists() or output.parent != plan_path.parent:
            raise ValueError("fresh measurement output directory required")
        config = TrainingConfig.from_dict(plan["training_config"])
        if (config.input_image_protocol, config.target_protocol, config.prompt_variant,
                config.max_seq_length, config.min_pixels, config.max_pixels,
                config.processor_use_fast) != ("visual_pointer_v2", "candidate_compact",
                                              "conditional_actor", 1536, 100352, 301056, True):
            raise ValueError("registered image/supervision/pixel/length policy differs")
        preparation_path = safe_path(plan["preparation_manifest"]["path"], ROOT / "outputs")
        if bound.get(str(preparation_path)) != plan["preparation_manifest"]["sha256"]:
            raise ValueError("preparation result not pinned in plan")
        preparation = strict_json_loads(bound_read(preparation_path, plan["preparation_manifest"]["sha256"]).decode("utf-8"))
        if preparation["status"] != "completed_input_asset_registration_only" or preparation["registered_views"] != 338:
            raise ValueError("complete visual input registration required")
        archives = plan["producer_source_archives"]
        archive_allowlist = {str(ROOT / "src/gui_agent/finetuning/evaluation.py")}
        if set(archives) != archive_allowlist or any(name not in preparation["bound_sha256"] for name in archives):
            raise ValueError("producer archive is not an original preparation binding")
        for name, digest in preparation["bound_sha256"].items():
            path = safe_path(name, ROOT)
            if name in archives:
                info = archives[name]
                if (path.suffix != ".py" or not path.is_relative_to(ROOT / "src")
                        or info["sha256"] != digest
                        or name not in bound or not isinstance(info["current_consumer_sha256"], str)
                        or info["current_consumer_sha256"] != bound.get(name)
                        or info["current_consumer_sha256"] == digest):
                    raise ValueError("only explicitly registered historical producer source may use an archive")
                path = safe_path(info["archive_path"], plan_path.parent / "producer_source_archive_001")
                if bound.get(str(path)) != digest:
                    raise ValueError("historical producer archive is not caller-pinned")
            bound_read(path, digest)
            merge_binding(bound, path, digest)
        for name, digest in preparation["output_sha256"].items():
            path = safe_path(name, ROOT / "outputs")
            bound_read(path, digest)
            merge_binding(bound, path, digest)
        registry_config_path = preparation_path.parent / "registry_config.json"
        registry_config = strict_json_loads(bound_read(registry_config_path, bound[str(registry_config_path)]).decode("utf-8"))
        if any(config.to_dict().get(key) != value for key, value in registry_config.items()):
            raise ValueError("trusted processor config differs from approved asset registry")
        row_path = safe_path(plan["visual_rows_path"], ROOT / "outputs")
        rows = [strict_json_loads(line) for line in bound_read(row_path, bound[str(row_path)]).decode("utf-8").splitlines() if line.strip()]
        decision_path = safe_path(plan["input_decisions_path"], ROOT / "outputs")
        decisions = [strict_json_loads(line) for line in bound_read(decision_path, bound[str(decision_path)]).decode("utf-8").splitlines() if line.strip()]
        if len(rows) != 338 or len(decisions) != 404 or len({row["sample_id"] for row in rows}) != 338:
            raise ValueError("actual raster and full denominator mismatch")
        indexed = {row["sample_id"]: row for row in decisions}
        if len(indexed) != 404 or sum(d["status"] == "refused_layout" for d in decisions) != 66:
            raise ValueError("lost original refusal or duplicated row")
        processor_path = safe_path(config.processor_path, ROOT / "outputs")
        actual_files = {str(path): sha(path) for path in processor_path.rglob("*") if path.is_file()}
        if actual_files != plan["processor_files"] or not actual_files:
            raise ValueError("processor/tokenizer/template file set differs")
        output.mkdir(exist_ok=False)
        processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True,
                                                  trust_remote_code=False, use_fast=True,
                                                  min_pixels=100352, max_pixels=301056)
        proxy = ObservedProcessor(processor)
        admitted, token_records, full_lengths, prefix_lengths = [], [], [], []
        eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (type(eos) is not int or eos == processor.tokenizer.pad_token_id
                or eos == getattr(processor.tokenizer, "unk_token_id", None)):
            raise ValueError("resolved scalar assistant EOS required")
        for index, row in enumerate(rows):
            proxy.calls.clear()
            proxy.templates.clear()
            overflow = False
            try:
                encoded = encode_training_sample(proxy, row, config)
            except ValueError as exc:
                # Only the production encoder's explicit nontruncating length refusal is normal.
                if "tokens exceed max_seq_length=1536; no truncation performed" not in str(exc):
                    raise
                overflow = True
                encoded = None
            if len(proxy.calls) != 2:
                raise ValueError("whole full/prefix pair was not actually encoded")
            if len(proxy.templates) != 2:
                raise ValueError("complete prefix/full conversation pair was not rendered")
            prefix_template, full_template = proxy.templates
            prefix_messages = prefix_template["messages"]
            full_messages = full_template["messages"]
            expected_prompt = VISUAL_PREFIX + pointer_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"])
            if (len(prefix_messages) != 1 or len(full_messages) != 2
                    or full_messages[0] != prefix_messages[0]
                    or prefix_messages[0] != {"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": expected_prompt}]}
                    or full_messages[1] != {"role": "assistant", "content": canonical_json(row["candidate_target"])}):
                raise ValueError("input-only prompt or complete original assistant target differs")
            full, prefix = proxy.calls
            if (full["rendered_text"] != full_template["rendered_text"]
                    or prefix["rendered_text"] != prefix_template["rendered_text"]):
                raise ValueError("encoded text differs from captured complete conversations")
            for call in (full, prefix):
                if (call["input_rgb_sha256"] != row["model_native_rgb_sha256"]
                        or call["input_rgb_sha256"] == row["image_sha256"]
                        or call["input_mode"] != "RGB"
                        or (call["input_width"], call["input_height"]) != (row["image_width"], row["image_height"])):
                    raise ValueError("production encoder received original or mismatched raster")
            if full["pixel_values"] != prefix["pixel_values"] or full["image_grid_thw"] != prefix["image_grid_thw"]:
                raise ValueError("real prefix/full visual tensor or grid differs")
            full_ids, prefix_ids = full["input_ids"], prefix["input_ids"]
            if len(full_ids) <= len(prefix_ids) or full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError("expanded actual generation prefix differs")
            full_lengths.append(len(full_ids))
            prefix_lengths.append(len(prefix_ids))
            if overflow != (len(full_ids) > 1536 or len(prefix_ids) > 1536):
                raise ValueError("encoder length refusal disagrees with actual complete lengths")
            labels = None if encoded is None else encoded["labels"][0].tolist()
            visual_ids = {processor.tokenizer.convert_tokens_to_ids(token) for token in
                          ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")}
            visual_ids.discard(None)
            if not overflow:
                if eos in visual_ids:
                    raise ValueError("resolved assistant end token aliases a visual token")
                if len(full_ids) > 1536 or len(prefix_ids) > 1536 or labels is None:
                    raise ValueError("production encoder admitted overflowing or unsupervised pair")
                if labels[:len(prefix_ids)] != [-100] * len(prefix_ids):
                    raise ValueError("expanded input prefix leaked into loss")
                expected_labels = [-100] * len(prefix_ids) + list(full_ids[len(prefix_ids):])
                for position, token in enumerate(full_ids):
                    if token in visual_ids or token == processor.tokenizer.pad_token_id:
                        expected_labels[position] = -100
                if labels != expected_labels:
                    raise ValueError("complete real assistant labels differ from independent mask")
                if any(label != -100 for token, label in zip(full_ids, labels, strict=True)
                       if token in visual_ids or token == processor.tokenizer.pad_token_id):
                    raise ValueError("visual or pad token supervised")
                if not any(token == label == eos for token, label in zip(full_ids, labels, strict=True)):
                    raise ValueError("complete assistant EOS is not supervised")
                admitted.append({**row, "training_admission": True, "tokens_not_measured": False})
            decision = indexed[row["sample_id"]]
            decision.update(status="refused_length" if overflow else "processor_admitted",
                            prefix_tokens=len(prefix_ids), full_tokens=len(full_ids),
                            same_annotated_raster_verified=True, same_visual_tensors_verified=True,
                            supervised_eos_verified=not overflow, training_admission=not overflow)
            token_records.append({"sample_id": row["sample_id"], "full": full, "prefix": prefix,
                                  "labels": labels, "eos_token_id": eos,
                                  "tokenizer_generic_eos_token_id": processor.tokenizer.eos_token_id,
                                  "template_records": list(proxy.templates),
                                  "resolved_visual_token_ids": sorted(visual_ids),
                                  "pad_token_id": processor.tokenizer.pad_token_id,
                                  "input_prompt_sha256": hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
                                  "complete_target_json_sha256": hashlib.sha256(canonical_json(row["candidate_target"]).encode("utf-8")).hexdigest(),
                                  "overflow_without_truncation": overflow})
            if (index + 1) % 32 == 0 or index + 1 == 338:
                print(f"ACTUAL PROCESSOR {index + 1}/338 admitted={len(admitted)}; denominator=404", flush=True)
        admitted_path = output / "admitted_rows.jsonl"
        write_new(admitted_path, "".join(canonical_json(row) + "\n" for row in admitted))
        # Exercise the production loader on only the separately measured subset.
        if admitted and len(load_prepared_records(admitted_path, target_protocol=config.target_protocol,
                                                  input_image_protocol=config.input_image_protocol,
                                                  visual_registry=config.visual_registry())) != len(admitted):
            raise ValueError("production loader rejected measured subset")
        write_new(output / "decisions.jsonl", "".join(canonical_json(row) + "\n" for row in decisions))
        write_new(output / "actual_token_ids.jsonl", "".join(canonical_json(row) + "\n" for row in token_records))
        if any(sha(Path(path)) != digest for path, digest in bound.items()) or torch.cuda.is_initialized():
            raise ValueError("source drift or CUDA initialization during gate")
        if {str(path): sha(path) for path in processor_path.rglob("*") if path.is_file()} != actual_files:
            raise ValueError("processor files changed during actual encoding")
        result = {"schema": "week5.visual_input_processor_result.v1", "status": "completed_CPU_processor_admission_only",
                  "source_views": 404, "available_visual_rows": 338, "layout_refused": 66,
                  "processor_admitted": len(admitted), "length_refused": 338 - len(admitted),
                  "processor_loads": 1, "processor_encode_calls": 676, "model_loads": 0,
                  "training_runs": 0, "gui_calls": 0, "cuda_initialized": False,
                  "prefix_token_stats": stats(prefix_lengths), "full_token_stats": stats(full_lengths),
                  "same_raster_tensor_grid_pairs_verified": 338, "runtime_versions": versions,
                  "historical_producer_source_archives": archives,
                  "bound_sha256": bound,
                  "output_sha256": {str(path): sha(path) for path in output.iterdir() if path.is_file()},
                  "processor_readability_or_model_quality_proven": False,
                  "W5_complete": False, "T04_closed": False}
        owned_publication = output / "manifest.json"
        write_new(owned_publication, result, owned_files=owned_files)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - preserve actual gate failure
        error = {"type": type(exc).__name__, "message": str(exc)}
        cancelled = isinstance(exc, KeyboardInterrupt)
    try:
        stable = all(Path(path).exists() and sha(Path(path)) == digest for path, digest in bound.items())
        if result is not None:
            stable = stable and all(sha(Path(path)) == digest for path, digest in result["output_sha256"].items())
    except OSError:
        stable = False
    code = 0 if error is None and stable and result is not None else 1
    if code != 0 and owned_publication in owned_files:
        safe_path(str(owned_publication), ROOT / "outputs").unlink()
    receipt = {"schema": "week5.visual_input_processor_execution_receipt.v1",
               "command": sys.orig_argv, "cwd": str(Path.cwd()), "started_at_utc": started,
               "finished_at_utc": datetime.now(UTC).isoformat(), "elapsed_seconds": time.perf_counter() - start,
               "exit_code": code, "error": error, "cancelled": cancelled, "bound_inputs_stable": stable,
               "result_valid": code == 0, "bound_sha256": bound,
               "result_path": str(owned_publication) if owned_publication else None,
               "observed": None if result is None else {key: result[key] for key in
                   ("source_views", "available_visual_rows", "processor_admitted", "length_refused", "processor_encode_calls")},
               "model_loads": 0, "training_runs": 0, "gui_calls": 0}
    write_new(plan_path.parent / "processor_execution_receipt_001.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("exit_code", "elapsed_seconds", "error", "observed")}), flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    return run(safe_path(str(args.plan), ROOT / "outputs"), args.plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
