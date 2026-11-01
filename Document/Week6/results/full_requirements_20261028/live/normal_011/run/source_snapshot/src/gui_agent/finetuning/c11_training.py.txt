"""Portable C11 recipe, separated from the historical machine-bound owners.

This entry point prepares new data and creates a NEW run. It does not recreate
the historical acceptance evidence. No desktop controller is imported.
"""
from __future__ import annotations

import random
from pathlib import Path

from . import resized_contract as contract
from . import structured_ce as objective
from .training import (
    TrainingConfig,
    assistant_labels,
    ensure_separate_validation,
    load_prepared_records,
    sha256_file,
)

BASE_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ROWS = 2482
EPOCHS = 8
ACCUMULATION = 8


def recipe(base_model: str) -> TrainingConfig:
    return TrainingConfig(
        base_model_path=base_model, base_model_id=BASE_ID, base_model_revision=REVISION,
        epochs=EPOCHS, gradient_accumulation_steps=ACCUMULATION,
        prompt_variant="conditional_actor",
        min_pixels=100352, max_pixels=802816, max_seq_length=1536,
        learning_rate=5e-5, weight_decay=0.0, seed=42,
    )


def epoch_groups(count: int, epoch: int) -> list[list[int]]:
    contract.need(type(count) is int and count > 0 and type(epoch) is int and 1 <= epoch <= 8,
                  "positive count and epoch 1..8 required")
    order = list(range(count))
    random.Random(42 + epoch - 1).shuffle(order)
    return [order[i:i + ACCUMULATION] for i in range(0, count, ACCUMULATION)]


def _write(path: Path, value) -> None:
    path.write_text(contract.canonical(value) + "\n", encoding="utf-8")


def _image(row):
    from PIL import Image

    contract.need(sha256_file(row["image_path"]) == row["image_file_sha256"],
                  "registered source image changed")
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    contract.need(image.size == (row["image_width"], row["image_height"]), "image size changed")
    return image


def _encode(processor, row, *, include_target: bool):
    geometry = contract.expected_geometry(row["image_width"], row["image_height"])
    messages = contract.build_messages(row, geometry, include_target=include_target)
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=not include_target)
    encoded = processor(text=[text], images=[_image(row)], padding=False,
                        truncation=False, return_tensors="pt")
    contract.need(encoded["input_ids"].shape[0] == 1
                  and encoded["input_ids"].shape[1] <= 1536, "untruncated single row <=1536 required")
    observed = contract.geometry_from_grid(row["image_width"], row["image_height"],
                                           encoded["image_grid_thw"][0].tolist())
    contract.need(observed == geometry, "actual processor geometry differs")
    return encoded


def _signature(encoded):
    """IDs, grid and pixels are checked again on every training replay."""
    import hashlib

    result = {}
    for key, tensor in encoded.items():
        cpu = tensor.detach().cpu().contiguous()
        result[key] = {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
                       "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest()}
    return result


def prepare(train_jsonl: str, validation_jsonl: str, base_model: str, output: str):
    """CPU processor qualification; no model weights are loaded."""
    import torch
    from transformers import AutoProcessor

    rows = load_prepared_records(train_jsonl)
    validation = load_prepared_records(validation_jsonl)
    ensure_separate_validation(rows, validation)
    contract.need(len(rows) == ROWS and all(row["source_split"] == "train" for row in rows),
                  "C11 budget requires exactly2482 train rows; no test/validation rows")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    base = Path(base_model).resolve(strict=True)
    config = recipe(str(base))
    decoder_path = base / "tokenizer.json"
    decoder = objective.load_bound_decoder(str(decoder_path), sha256_file(decoder_path))
    processor = AutoProcessor.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False, use_fast=True,
        min_pixels=100352, max_pixels=802816,
    )
    visual = {processor.tokenizer.convert_tokens_to_ids(value) for value in
              ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")}
    visual.discard(None)
    with (destination / "qualified.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            prefix, full = _encode(processor, row, include_target=False), _encode(
                processor, row, include_target=True)
            contract.need(all(torch.equal(prefix[key], full[key]) for key in
                              ("pixel_values", "image_grid_thw")), "label-side pixels differ")
            ids, prefix_ids = full["input_ids"][0].tolist(), prefix["input_ids"][0].tolist()
            labels = assistant_labels(ids, prefix_ids, visual_ids=visual,
                                      pad_id=processor.tokenizer.pad_token_id)
            target = contract.canonical(contract.to_resized(row["actions"],
                       contract.expected_geometry(row["image_width"], row["image_height"])))
            weights = objective.value_weights(full_ids=ids, prefix_ids=prefix_ids,
                                               labels=labels, target_json=target, decoder=decoder)
            stream.write(contract.canonical({"row": row, "full_ids": ids,
                "prefix_ids": prefix_ids, "labels": labels, "weights": list(weights.weights),
                "signature": _signature(full)}) + "\n")
    plan = {"schema": "week5.portable_C11_preparation.v1", "historical_run_reproduction": False,
            "config": config.to_dict(), "rows": len(rows), "epochs": EPOCHS,
            "train_sha256": sha256_file(train_jsonl),
            "validation_sha256": sha256_file(validation_jsonl),
            "qualified_sha256": sha256_file(destination / "qualified.jsonl"),
            "tokenizer_sha256": sha256_file(decoder_path),
            "base_model_tree": {p.relative_to(base).as_posix(): sha256_file(p)
                                for p in sorted(base.rglob("*"))
                                if p.is_file() and ".cache" not in p.parts},
            "manual_objective": objective.STRATEGY, "processor_calls": 2 * len(rows)}
    _write(destination / "plan.json", plan)
    return plan


def train(prepared: str, output: str):
    """Eight full epochs; the last accumulation group uses its actual size."""
    import torch

    from .embedding_memory import AutocastForward, restore_frozen_embeddings, validate_audit
    from .training import load_model_and_processor

    source = Path(prepared).resolve(strict=True)
    plan = contract.strict_json((source / "plan.json").read_text("utf-8"))
    contract.need(plan["schema"] == "week5.portable_C11_preparation.v1"
                  and plan["rows"] == ROWS and plan["epochs"] == EPOCHS
                  and plan["manual_objective"] == objective.STRATEGY
                  and sha256_file(source / "qualified.jsonl") == plan["qualified_sha256"],
                  "prepared plan/data identity changed")
    config = TrainingConfig.from_dict(plan["config"])
    contract.need(config.to_dict() == recipe(config.base_model_path).to_dict(), "fixed recipe changed")
    base = Path(config.base_model_path)
    contract.need({p.relative_to(base).as_posix(): sha256_file(p) for p in sorted(base.rglob("*"))
                   if p.is_file() and ".cache" not in p.parts} == plan["base_model_tree"],
                  "registered base/processor bytes changed")
    qualified = [contract.strict_json(line) for line in
                 (source / "qualified.jsonl").read_text("utf-8").splitlines()]
    contract.need(len(qualified) == ROWS, "complete2482 qualified rows required")
    decoder = objective.load_bound_decoder(str(base / "tokenizer.json"), plan["tokenizer_sha256"])
    for entry in qualified:
        row = entry["row"]
        target = contract.canonical(contract.to_resized(row["actions"],
                   contract.expected_geometry(row["image_width"], row["image_height"])))
        weights = objective.value_weights(full_ids=entry["full_ids"], prefix_ids=entry["prefix_ids"],
                                           labels=entry["labels"], target_json=target, decoder=decoder)
        contract.need(list(weights.weights) == entry["weights"], "saved token weights changed")
        _image(row).close()
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    counts = {"forwards": 0, "backwards": 0, "optimizer_steps": 0, "processor_calls": 0,
              "epochs": 0, "skips": 0, "generate_calls": 0}
    try:
        torch.manual_seed(42)
        model, processor = load_model_and_processor(config, for_training=True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        contract.need(sum(p.numel() for p in parameters) == 3686400
                      and len(parameters) == 288
                      and all(p.dtype == torch.float32 for p in parameters), "FP32 language LoRA differs")
        audit = restore_frozen_embeddings(model)
        validate_audit(audit)
        _write(destination / "memory_audit.json", audit)
        forward = AutocastForward(model)
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.0)
        with (destination / "records.jsonl").open("x", encoding="utf-8") as stream:
            for epoch in range(1, EPOCHS + 1):
                for group in epoch_groups(ROWS, epoch):
                    optimizer.zero_grad(set_to_none=True)
                    for index in group:
                        entry = qualified[index]
                        batch = _encode(processor, entry["row"], include_target=True)
                        counts["processor_calls"] += 1
                        contract.need(_signature(batch) == entry["signature"], "replayed tensors changed")
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        labels = torch.tensor([entry["labels"]], dtype=torch.long, device=model.device)
                        weights = torch.tensor([entry["weights"]], device=model.device)
                        returned = forward(**batch, use_cache=False, return_dict=True)
                        counts["forwards"] += 1
                        losses = objective.causal_ce(returned.logits, labels, weights)
                        del returned
                        loss, raw = losses["weighted_ce"], losses["raw_ce"]
                        contract.need(bool(torch.isfinite(loss)), "nonfinite loss; stop, never skip")
                        (loss / len(group)).backward()
                        counts["backwards"] += 1
                        stream.write(contract.canonical({"epoch": epoch, "row": index,
                            "raw_ce": float(raw.detach()), "weighted_ce": float(loss.detach()),
                            "accumulation_divisor": len(group)}) + "\n")
                        stream.flush()
                        del losses, loss, raw, batch, labels, weights
                    contract.need(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                      for p in parameters), "nonfinite/missing LoRA gradient")
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    counts["optimizer_steps"] += 1
                counts["epochs"] += 1
        contract.need(counts == {"forwards": 19856, "backwards": 19856,
            "optimizer_steps": 2488, "processor_calls": 19856, "epochs": 8,
            "skips": 0, "generate_calls": 0}, "incomplete fixed training budget")
        model.save_pretrained(destination / "adapter", safe_serialization=True)
        result = {"status": "completed_new_run", "historical_acceptance_replayed": False,
                  "counts": counts, "prepared_plan_sha256": sha256_file(source / "plan.json"),
                  "adapter_sha256": sha256_file(destination / "adapter/adapter_model.safetensors")}
        _write(destination / "result.json", result)
        return result
    except BaseException as error:
        _write(destination / "result.json", {"status": "execution_failed", "counts": counts,
                                            "error": f"{type(error).__name__}: {error}"})
        raise
