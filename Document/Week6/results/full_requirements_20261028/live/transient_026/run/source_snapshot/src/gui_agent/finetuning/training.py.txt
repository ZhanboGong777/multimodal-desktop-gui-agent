"""Local Qwen-VL LoRA training, with loss only on the assistant action array.

Heavy dependencies are imported inside the functions that use them. Importing
this module must not load CUDA, a model, or change the Week 4 environment.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

SAMPLING_POLICIES = ("uniform", "balanced_action_chunks", "coverage_first_balanced_action_chunks")
LOSS_STRATEGIES = ("uniform", "registered_stratum_balanced")
TOKEN_LOSS_STRATEGIES = ("uniform", "candidate_offset_numeric_value_fixed4",
                         "candidate4_offset8_teacher_forced_digit_place_geometry8")
SPATIAL_TARGET_PROTOCOLS = ("native_spatial_head", "native_spatial_grid_head")
TARGET_PROTOCOLS = ("native", "candidate_compact", *SPATIAL_TARGET_PROTOCOLS)
VISUAL_INPUT_PROTOCOLS = ("visual_pointer_v2", "visual_pointer_gallery_v3")
INPUT_IMAGE_PROTOCOLS = ("native", *VISUAL_INPUT_PROTOCOLS)
VISUAL_REGISTRY_FIELDS = ("visual_asset_manifest_path", "visual_asset_manifest_sha256",
                          "visual_original_root", "visual_asset_root")
VISUAL_ROW_FIELDS = ("model_image_path", "model_png_sha256", "model_native_rgb_sha256",
                     "visual_prompt_version", "visual_metadata_path", "visual_metadata_file_sha256", "raw_snapshot_path",
                     "raw_snapshot_file_sha256", *VISUAL_REGISTRY_FIELDS)


@dataclass(frozen=True)
class TrainingConfig:
    base_model_path: str = "models/Qwen2-VL-2B-Instruct"
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    base_model_revision: str = "895c3a49bc3fa70a340399125c650a463535e71c"
    train_jsonl: str | None = None
    validation_jsonl: str | None = None
    output_dir: str | None = None
    processor_path: str | None = None
    initial_adapter_path: str | None = None
    processor_use_fast: bool = True
    device: str = "cuda:0"
    compute_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    attention_implementation: str = "sdpa"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    gradient_checkpointing: bool = True
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 384 * 28 * 28
    max_seq_length: int = 1536
    epochs: int = 2
    max_steps: int | None = None
    max_micro_steps: int | None = None
    max_optimizer_attempts: int | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 42
    prompt_variant: str = "original"
    target_protocol: str = "native"
    input_image_protocol: str = "native"
    visual_asset_manifest_path: str | None = None
    visual_asset_manifest_sha256: str | None = None
    visual_original_root: str | None = None
    visual_asset_root: str | None = None
    visual_prompt_version: str | None = None
    sampling_policy: str = "uniform"
    loss_strategy: str = "uniform"
    token_loss_strategy: str = "uniform"
    numeric_tokenizer_json_path: str | None = None
    numeric_tokenizer_json_sha256: str | None = None
    spatial_tokenizer_json_path: str | None = None
    spatial_tokenizer_json_sha256: str | None = None
    spatial_head_learning_rate: float | None = None
    smoke: bool = False

    def __post_init__(self) -> None:
        for name in ("lora_rank", "lora_alpha", "gradient_accumulation_steps", "min_pixels",
                     "max_pixels", "max_seq_length", "epochs"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.micro_batch_size != 1:
            raise ValueError("this memory-bounded trainer supports micro_batch_size=1 only")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps <= 0):
            raise ValueError("max_steps must be a positive optimizer-step count or null")
        for name in ("max_micro_steps", "max_optimizer_attempts"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels")
        if self.compute_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("compute_dtype must be bfloat16, float16 or float32")
        if self.prompt_variant not in _prompts().TRAINING_PROMPT_VARIANTS:
            raise ValueError("unknown training prompt; improved is evaluation-only")
        if not isinstance(self.target_protocol, str) or self.target_protocol not in TARGET_PROTOCOLS:
            raise ValueError("unknown training target_protocol")
        if self.target_protocol == "candidate_compact" and self.prompt_variant != "conditional_actor":
            raise ValueError("candidate_compact requires prompt_variant=conditional_actor")
        if not isinstance(self.input_image_protocol, str) or self.input_image_protocol not in INPUT_IMAGE_PROTOCOLS:
            raise ValueError("unknown training input_image_protocol")
        if self.input_image_protocol in VISUAL_INPUT_PROTOCOLS:
            if self.visual_prompt_version is not None:
                from .visual_pointer_prompts import checked_prompt_version

                checked_prompt_version(self.visual_prompt_version)
            from .visual_pointer_prompts import (
                GALLERY_PROMPT_VERSIONS,
                GALLERY_SPATIAL_PROMPT_VERSION,
            )

            if self.prompt_variant != "conditional_actor" or self.target_protocol not in {"candidate_compact", *SPATIAL_TARGET_PROTOCOLS}:
                raise ValueError("visual input requires conditional_actor and an explicit visual target protocol")
            if (self.input_image_protocol == "visual_pointer_gallery_v3") != (self.visual_prompt_version in GALLERY_PROMPT_VERSIONS):
                raise ValueError("gallery input protocol requires its explicit gallery prompt")
            if (self.target_protocol in SPATIAL_TARGET_PROTOCOLS) != (self.visual_prompt_version == GALLERY_SPATIAL_PROMPT_VERSION):
                raise ValueError("semantic gallery prompt requires native_spatial_head target protocol")
            if self.input_image_protocol == "visual_pointer_gallery_v3" and (
                self.min_pixels, self.max_pixels, self.max_seq_length, self.processor_use_fast
            ) != (100352, 301056, 1536, True):
                raise ValueError("gallery requires fixed1536/100352/301056/fast processor profile")
            if self.max_seq_length > 1536:
                raise ValueError("visual_pointer_v2 max_seq_length cannot exceed1536")
            for name in VISUAL_REGISTRY_FIELDS:
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"visual_pointer_v2 requires caller config {name}")
            from .visual_pointer_input import _path, _sha

            for name in ("visual_asset_manifest_path", "visual_original_root", "visual_asset_root"):
                _path(getattr(self, name))
            _sha(self.visual_asset_manifest_sha256, "caller manifest")
        elif self.visual_prompt_version is not None or any(getattr(self, name) is not None for name in VISUAL_REGISTRY_FIELDS):
            raise ValueError("native input protocol cannot use a visual asset registry")
        if not isinstance(self.sampling_policy, str) or self.sampling_policy not in SAMPLING_POLICIES:
            raise ValueError("unknown training sampling_policy")
        if not isinstance(self.loss_strategy, str) or self.loss_strategy not in LOSS_STRATEGIES:
            raise ValueError("unknown training loss_strategy")
        if self.loss_strategy == "registered_stratum_balanced" and (
            self.max_steps is None or self.max_optimizer_attempts != self.max_steps or self.max_micro_steps is None
        ):
            raise ValueError("registered_stratum_balanced requires positive max_steps=max_optimizer_attempts and max_micro_steps")
        if not isinstance(self.token_loss_strategy, str) or self.token_loss_strategy not in TOKEN_LOSS_STRATEGIES:
            raise ValueError("unknown token_loss_strategy")
        if self.token_loss_strategy != "uniform":
            if self.target_protocol != "candidate_compact" or self.prompt_variant != "conditional_actor":
                raise ValueError("numeric VALUE objective requires candidate_compact and conditional_actor")
            if (not isinstance(self.numeric_tokenizer_json_path, str)
                    or not Path(self.numeric_tokenizer_json_path).is_absolute()):
                raise ValueError("numeric objective requires caller absolute tokenizer metadata path")
            digest = self.numeric_tokenizer_json_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("numeric objective requires caller tokenizer metadata SHA256")
        elif self.numeric_tokenizer_json_path is not None or self.numeric_tokenizer_json_sha256 is not None:
            raise ValueError("uniform token objective refuses numeric decoder metadata")
        if self.target_protocol in SPATIAL_TARGET_PROTOCOLS:
            from .visual_pointer_prompts import GALLERY_SPATIAL_PROMPT_VERSION

            if (self.prompt_variant, self.input_image_protocol, self.visual_prompt_version) != (
                    "conditional_actor", "visual_pointer_gallery_v3", GALLERY_SPATIAL_PROMPT_VERSION):
                raise ValueError("native_spatial_head requires the explicit semantic Gallery prompt")
            if self.initial_adapter_path is not None or self.validation_jsonl is not None:
                raise ValueError("native_spatial_head forbids warm start and validation")
            if self.loss_strategy != "registered_stratum_balanced" or self.token_loss_strategy != "uniform":
                raise ValueError("native_spatial_head requires CE1/point1 with class weights1")
            if (self.learning_rate, self.spatial_head_learning_rate) != (1e-4, 1e-3):
                raise ValueError("native_spatial_head requires fixed LoRA/head learning rates")
            if (self.epochs, self.max_steps, self.max_optimizer_attempts, self.max_micro_steps,
                    self.gradient_accumulation_steps, self.sampling_policy) != (
                    2, 144, 144, 1144, 8, "coverage_first_balanced_action_chunks"):
                raise ValueError("native_spatial_head requires the fixed bounded balanced schedule")
            if not isinstance(self.spatial_tokenizer_json_path, str) or not Path(self.spatial_tokenizer_json_path).is_absolute():
                raise ValueError("native_spatial_head requires caller absolute tokenizer metadata path")
            digest = self.spatial_tokenizer_json_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("native_spatial_head requires caller tokenizer metadata SHA256")
        elif any(value is not None for value in (
                self.spatial_tokenizer_json_path, self.spatial_tokenizer_json_sha256, self.spatial_head_learning_rate)):
            raise ValueError("spatial head config fields require native_spatial_head")
        if self.initial_adapter_path is not None and (
            not isinstance(self.initial_adapter_path, str) or not self.initial_adapter_path.strip()
        ):
            raise ValueError("initial_adapter_path must be a nonempty path or null")
        for name in ("load_in_4bit", "gradient_checkpointing", "smoke", "processor_use_fast"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.target_modules or any(not isinstance(name, str) or not name for name in self.target_modules):
            raise ValueError("target_modules must contain module names")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_modules"] = list(self.target_modules)
        return result

    def visual_registry(self) -> Any:
        return visual_registry(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingConfig:
        names = {field.name for field in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError("unknown training config fields: " + ", ".join(sorted(unknown)))
        values = dict(data)
        if "target_modules" in values:
            if not isinstance(values["target_modules"], (list, tuple)):
                raise ValueError("target_modules must be an array of module names")
            values["target_modules"] = tuple(values["target_modules"])
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainingConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("training config must be a JSON object")
        return cls.from_dict(data)


def _spatial_modules(protocol: str) -> tuple[Any, Any]:
    if protocol == "native_spatial_grid_head":
        from . import native_spatial_grid_artifacts as artifacts
        from . import native_spatial_grid_head as core
    elif protocol == "native_spatial_head":
        from . import native_spatial_artifacts as artifacts
        from . import native_spatial_head as core
    else:
        raise ValueError("explicit spatial target protocol required")
    return core, artifacts


def _prompts() -> Any:
    from . import prompts
    return prompts


def visual_registry(config: TrainingConfig) -> Any:
    """Construct only from trusted caller configuration, never sample claims."""
    if config.input_image_protocol == "native":
        return None
    from .visual_pointer_input import VisualAssetRegistry

    return VisualAssetRegistry(config.visual_asset_manifest_path, config.visual_asset_manifest_sha256,
                               config.visual_original_root, config.visual_asset_root)


def _input_protocol(sample: Mapping[str, Any], protocol: str, registry: Any, visual_prompt_version: str | None = None) -> None:
    if protocol not in INPUT_IMAGE_PROTOCOLS:
        raise ValueError("unknown training input_image_protocol")
    if protocol == "native":
        if visual_prompt_version is not None or registry is not None or sample.get("input_image_protocol", "native") != "native" or any(
                key in sample for key in VISUAL_ROW_FIELDS):
            raise ValueError("native input protocol refuses visual asset fields; choose visual_pointer_v2 explicitly")
    else:
        from .visual_pointer_input import VisualAssetRegistry

        if not isinstance(registry, VisualAssetRegistry):
            raise TypeError("visual input requires a trusted caller VisualAssetRegistry")
        if sample.get("input_image_protocol") != protocol:
            raise ValueError("sample and configured input_image_protocol differ")
        from .visual_pointer_prompts import declared_prompt_version

        if declared_prompt_version(sample) != visual_prompt_version:
            raise ValueError("sample and caller visual_prompt_version differ")


def _visual_messages(sample: Mapping[str, Any], prompt_variant: str, target_protocol: str,
                     include_target: bool, raster: Any, visual_prompt_version: str | None = None) -> list[dict[str, Any]]:
    if target_protocol not in {"candidate_compact", *SPATIAL_TARGET_PROTOCOLS} or prompt_variant != "conditional_actor":
        raise ValueError("visual input requires explicit target protocol and conditional_actor")
    if target_protocol in SPATIAL_TARGET_PROTOCOLS:
        spatial_core, _ = _spatial_modules(target_protocol)
        canonical, semantic_target = spatial_core.canonical, spatial_core.semantic_target
        from .visual_pointer_prompts import GALLERY_SPATIAL_PROMPT_VERSION

        if visual_prompt_version != GALLERY_SPATIAL_PROMPT_VERSION or raster.visual_prompt_version != visual_prompt_version:
            raise ValueError("native spatial messages require approved semantic Gallery prompt")
        raster.assert_current()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": raster.prompt}]}]
        if include_target:
            target = semantic_target(sample["actions"], native_width=sample["image_width"], native_height=sample["image_height"])
            if canonical(sample.get("native_spatial_target")) != canonical(target):
                raise ValueError("native spatial label sidecar differs from native complete actions")
            messages.append({"role": "assistant", "content": target["semantic_json"]})
        raster.assert_current()
        return messages
    from .visual_pointer import build_visual_pointer_messages

    if raster.visual_prompt_version != visual_prompt_version:
        raise ValueError("approved visual prompt version differs from caller configuration")
    raster.assert_current()
    messages = build_visual_pointer_messages(sample, include_target=include_target,
                                             visual_prompt_version=raster.visual_prompt_version)
    if messages[0]["content"][1]["text"] != raster.prompt:
        raise ValueError("visual training messages differ from the registered prompt")
    raster.assert_current()
    return messages


def build_messages(
    sample: Mapping[str, Any], prompt_variant: str = "original", *, include_target: bool = False,
    target_protocol: str = "native",
    input_image_protocol: str = "native", visual_registry: Any = None,
    visual_prompt_version: str | None = None,
) -> list[dict[str, Any]]:
    """The inference conversation is identical to the training user prefix.

    Task-only profiles pass instruction and original screenshot dimensions.
    Conditional acting also requires an explicit pre-request actor_step. Action
    annotations, source metadata and expected actions are never user inputs.
    """
    _input_protocol(sample, input_image_protocol, visual_registry, visual_prompt_version)
    if input_image_protocol in VISUAL_INPUT_PROTOCOLS:
        from .visual_pointer_input import validate_visual_input

        raster = validate_visual_input(sample, registry=visual_registry)
        return _visual_messages(sample, prompt_variant, target_protocol, include_target, raster, visual_prompt_version)
    if target_protocol not in TARGET_PROTOCOLS:
        raise ValueError("unknown training target_protocol")
    if target_protocol in SPATIAL_TARGET_PROTOCOLS:
        raise ValueError("native_spatial_head messages require registered semantic Gallery input")
    if target_protocol == "candidate_compact":
        if prompt_variant != "conditional_actor":
            raise ValueError("candidate_compact requires prompt_variant=conditional_actor")
        from .pointer_protocol import build_pointer_messages

        return build_pointer_messages(sample, include_target=include_target)
    visible = _prompts().prompt_input(sample, prompt_variant)
    user = _prompts().build_user_prompt(visible, prompt_variant=prompt_variant)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": user},
    ]}]
    if include_target:
        messages.append({"role": "assistant", "content": _prompts().action_target_json(sample["actions"])})
    return messages


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prepared_records(
    path: str | Path, *, target_protocol: str = "native",
    input_image_protocol: str = "native", visual_registry: Any = None,
    visual_prompt_version: str | None = None,
) -> list[dict[str, Any]]:
    """Validate actual images and training provenance without loading a model."""
    if target_protocol not in TARGET_PROTOCOLS:
        raise ValueError("unknown training target_protocol")
    if input_image_protocol not in INPUT_IMAGE_PROTOCOLS:
        raise ValueError("unknown training input_image_protocol")
    if target_protocol in SPATIAL_TARGET_PROTOCOLS:
        from .visual_pointer_prompts import GALLERY_SPATIAL_PROMPT_VERSION

        if (input_image_protocol, visual_prompt_version) != ("visual_pointer_gallery_v3", GALLERY_SPATIAL_PROMPT_VERSION):
            raise ValueError("native_spatial_head records require registered semantic Gallery input")
    if input_image_protocol in VISUAL_INPUT_PROTOCOLS and target_protocol not in {"candidate_compact", *SPATIAL_TARGET_PROTOCOLS}:
        raise ValueError("visual input requires candidate_compact or native_spatial_head supervision")
    source = Path(path).resolve()
    records = []
    seen = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{source}:{line_number}: sample must be an object")
        _input_protocol(row, input_image_protocol, visual_registry, visual_prompt_version)
        raster = None
        if input_image_protocol in VISUAL_INPUT_PROTOCOLS:
            if row.get("training_admission") is not True:
                raise ValueError("visual training requires a separately admitted export with training_admission=True")
            from .visual_pointer_input import validate_visual_input

            raster = validate_visual_input(row, registry=visual_registry)
            if raster.visual_prompt_version != visual_prompt_version:
                raise ValueError("approved visual prompt version differs from caller configuration")
        for name in ("sample_id", "group_id", "image_path", "instruction", "source", "source_split"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"{source}:{line_number}: {name} must be nonempty text")
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        if row["source_split"].casefold() in {"test", "official_test", "official-test"}:
            raise ValueError(f"official test sample cannot enter training/validation: {row['sample_id']}")
        if not isinstance(row.get("actions"), list) or not row["actions"]:
            raise ValueError(f"{row['sample_id']}: actions must be a nonempty array")
        for name in ("image_width", "image_height"):
            if type(row.get(name)) is not int or row[name] <= 0:
                raise ValueError(f"{row['sample_id']}: {name} must be a positive integer")
        from .data import validate_actions
        actions = validate_actions(row["actions"], row["image_width"], row["image_height"])
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = source.parent / image_path
        image_path = image_path.resolve(strict=True)
        with Image.open(image_path) as image:
            image.load()
            if image.size != (row["image_width"], row["image_height"]):
                raise ValueError(f"{row['sample_id']}: image dimensions do not match prepared metadata")
            # Prepared-data image identity is decoded RGB plus dimensions, not
            # the PNG/JPEG container bytes. Keep both hashes under distinct names.
            rgb = image.convert("RGB")
            actual_sha = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
            ).hexdigest()
        if row.get("image_sha256") and row["image_sha256"] != actual_sha:
            raise ValueError(f"{row['sample_id']}: image hash differs from prepared metadata")
        record = {**row, "actions": actions, "image_path": str(image_path),
                  "image_sha256": actual_sha, "image_file_sha256": sha256_file(image_path)}
        if target_protocol == "candidate_compact":
            from .pointer_protocol import validate_pointer_sample

            validate_pointer_sample(record)
        if target_protocol in SPATIAL_TARGET_PROTOCOLS:
            spatial_core, _ = _spatial_modules(target_protocol)
            canonical, semantic_target = spatial_core.canonical, spatial_core.semantic_target

            target = semantic_target(actions, native_width=row["image_width"], native_height=row["image_height"])
            if canonical(row.get("native_spatial_target")) != canonical(target) or row.get("tokens_not_measured") is not False:
                raise ValueError("native_spatial_head requires freshly measured complete semantic target admission")
        if raster is not None:
            raster.assert_current()
        records.append(record)
    if not records:
        raise ValueError(f"prepared dataset is empty: {source}")
    return records


def ensure_separate_validation(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> None:
    for key in ("sample_id", "group_id", "image_sha256"):
        shared = {row[key] for row in train} & {row[key] for row in validation}
        if shared:
            raise ValueError(f"training/validation leakage in {key}: {sorted(shared)[:3]}")


def assistant_labels(
    full_ids: Sequence[int], prompt_ids: Sequence[int], *, visual_ids: set[int], pad_id: int | None,
) -> list[int]:
    """Mask the expanded multimodal prefix, not an unexpanded text-token count."""
    boundary = len(prompt_ids)
    if boundary >= len(full_ids) or list(full_ids[:boundary]) != list(prompt_ids):
        raise ValueError("complete conversation and image-expanded user prefix do not match")
    labels = [-100] * boundary + list(full_ids[boundary:])
    for index, token in enumerate(full_ids):
        if token in visual_ids or (pad_id is not None and token == pad_id):
            labels[index] = -100
    if not any(value != -100 for value in labels):
        raise ValueError("sample has no assistant tokens to supervise")
    return labels


def encode_training_sample(processor: Any, sample: Mapping[str, Any], config: TrainingConfig) -> dict[str, Any]:
    import torch

    registry = config.visual_registry()
    _input_protocol(sample, config.input_image_protocol, registry, config.visual_prompt_version)
    raster = None
    if config.input_image_protocol in VISUAL_INPUT_PROTOCOLS:
        from .visual_pointer_input import validate_visual_input

        raster = validate_visual_input(sample, registry=registry)
        prefix_messages = _visual_messages(sample, config.prompt_variant, config.target_protocol, False, raster, config.visual_prompt_version)
        full_messages = _visual_messages(sample, config.prompt_variant, config.target_protocol, True, raster, config.visual_prompt_version)
    else:
        prefix_messages = build_messages(sample, config.prompt_variant, target_protocol=config.target_protocol)
        full_messages = build_messages(sample, config.prompt_variant, include_target=True, target_protocol=config.target_protocol)
    prefix = processor.apply_chat_template(
        prefix_messages,
        tokenize=False, add_generation_prompt=True,
    )
    full = processor.apply_chat_template(
        full_messages,
        tokenize=False, add_generation_prompt=False,
    )
    if raster is None:
        with Image.open(sample["image_path"]) as image:
            rgb = image.convert("RGB")
            encoded = processor(text=[full], images=[rgb], padding=False, return_tensors="pt")
            prompt = processor(text=[prefix], images=[rgb], padding=False, return_tensors="pt")
    else:
        full_rgb, prefix_rgb = raster.to_pil(), raster.to_pil()
        encoded = processor(text=[full], images=[full_rgb], padding=False, truncation=False, return_tensors="pt")
        raster.assert_current()
        if full_rgb.mode != "RGB" or full_rgb.size != raster.model_size or full_rgb.tobytes() != raster.rgb_bytes:
            raise ValueError("processor mutated the registered full visual raster")
        prompt = processor(text=[prefix], images=[prefix_rgb], padding=False, truncation=False, return_tensors="pt")
        raster.assert_current()
        if prefix_rgb.mode != "RGB" or prefix_rgb.size != raster.model_size or prefix_rgb.tobytes() != raster.rgb_bytes:
            raise ValueError("processor mutated the registered prefix visual raster")
        for key in ("pixel_values", "image_grid_thw"):
            left, right = encoded.get(key), prompt.get(key)
            if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
                    or left.numel() == 0 or left.dtype != right.dtype or left.device != right.device
                    or left.shape != right.shape or not torch.isfinite(left).all().item()
                    or not torch.equal(left, right)):
                raise ValueError(f"full/prefix visual {key} tensors differ or are invalid")
        grid = encoded["image_grid_thw"]
        if grid.shape != (1, 3) or grid.dtype not in (torch.int32, torch.int64) or not (grid > 0).all().item():
            raise ValueError("visual image_grid_thw must describe one positive integer image grid")
        if _visual_messages(sample, config.prompt_variant, config.target_protocol, True, raster, config.visual_prompt_version) != full_messages:
            raise ValueError("complete visual supervision changed during encoding")
    full_ids = encoded["input_ids"][0].tolist()
    prompt_ids = prompt["input_ids"][0].tolist()
    if len(full_ids) > config.max_seq_length:
        raise ValueError(f"{sample['sample_id']}: {len(full_ids)} tokens exceed max_seq_length={config.max_seq_length}; no truncation performed")
    if raster is not None and len(prompt_ids) > config.max_seq_length:
        raise ValueError(f"{sample['sample_id']}: image-expanded prefix {len(prompt_ids)} tokens exceed max_seq_length={config.max_seq_length}; no truncation performed")
    visual_tokens = ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
    visual_ids = {processor.tokenizer.convert_tokens_to_ids(token) for token in visual_tokens}
    visual_ids.discard(None)
    labels = assistant_labels(full_ids, prompt_ids, visual_ids=visual_ids, pad_id=processor.tokenizer.pad_token_id)
    if raster is not None:
        eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (type(eos) is not int or eos in visual_ids or eos == processor.tokenizer.pad_token_id
                or eos == getattr(processor.tokenizer, "unk_token_id", None)
                or not any(token == eos and label == eos for token, label in zip(full_ids, labels, strict=True))):
            raise ValueError("visual complete target requires a resolved supervised assistant end token")
        raster.assert_current()
    encoded["labels"] = torch.tensor([labels], dtype=encoded["input_ids"].dtype)
    if config.target_protocol in SPATIAL_TARGET_PROTOCOLS:
        spatial_core, _ = _spatial_modules(config.target_protocol)
        training_plan = spatial_core.training_plan
        from .numeric_value_loss import load_bound_decoder

        decoder = load_bound_decoder(config.spatial_tokenizer_json_path, config.spatial_tokenizer_json_sha256)
        if processor.tokenizer.convert_tokens_to_ids("<|im_end|>") != decoder.eos_id:
            raise ValueError("actual processor assistant end ID differs from pinned spatial tokenizer")
        plan_arguments = {}
        if config.target_protocol == "native_spatial_grid_head":
            if raster is None:
                raise ValueError("grid spatial plan requires a registered Gallery raster")
            raster.assert_current()
            metadata_payload = Path(sample["visual_metadata_path"]).read_bytes()
            if hashlib.sha256(metadata_payload).hexdigest() != sample["visual_metadata_file_sha256"]:
                raise ValueError("grid metadata changed after caller registration")
            metadata = spatial_core.strict_json(metadata_payload)
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            model_configuration = json.loads((Path(config.base_model_path) / "config.json").read_text(encoding="utf8"))
            if image_token_id != model_configuration.get("image_token_id"):
                raise ValueError("grid tokenizer and model image placeholder IDs differ")
            vision = model_configuration.get("vision_config", {})
            if (vision.get("patch_size"), vision.get("spatial_merge_size")) != (14, 2):
                raise ValueError("grid requires the pinned patch14/merge2 profile")
            plan_arguments = {"metadata": metadata, "grid_thw": encoded["image_grid_thw"][0].tolist(),
                              "image_token_id": image_token_id}
            raster.assert_current()
        encoded["_native_spatial_plan"] = training_plan(
            full_ids=full_ids, prefix_ids=prompt_ids, labels=labels,
            target=sample["native_spatial_target"], decoder=decoder, **plan_arguments,
        )
    return dict(encoded)


def _visual_name(name: str) -> bool:
    return bool(set(name.split(".")) & {"visual", "vision_tower", "vision_model", "vision_encoder", "multi_modal_projector"})


def language_lora_targets(model: Any, suffixes: Sequence[str]) -> list[str]:
    targets = [name for name, _ in model.named_modules()
               if name.rsplit(".", 1)[-1] in suffixes and not _visual_name(name)]
    if not targets:
        raise ValueError("no language projection modules matched the LoRA configuration")
    return targets


def freeze_visual_encoder(model: Any) -> int:
    count = 0
    for name, parameter in model.named_parameters():
        if _visual_name(name):
            parameter.requires_grad_(False)
            count += parameter.numel()
    if count == 0:
        raise ValueError("no visual encoder parameters found; cannot assert visual freezing")
    return count


def initial_adapter_evidence(config: TrainingConfig) -> dict[str, Any] | None:
    """Check warm-start weights before CUDA work; optimizer state is never resumed."""
    if config.initial_adapter_path is None:
        return None
    path = Path(config.initial_adapter_path).resolve(strict=True)
    evidence = _saved_file_evidence(path, adapter=True)
    adapter = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    expected = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
                "r": config.lora_rank, "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout}
    for name, value in expected.items():
        if adapter.get(name) != value:
            raise ValueError(f"initial adapter {name} does not match training config")
    base = adapter.get("base_model_name_or_path")
    if not isinstance(base, str) or Path(base).resolve() != Path(config.base_model_path).resolve():
        raise ValueError("initial adapter base model path does not match training config")
    targets = adapter.get("target_modules")
    if not isinstance(targets, list) or not targets or any(
        not isinstance(name, str) or _visual_name(name) for name in targets
    ) or {name.rsplit(".", 1)[-1] for name in targets} != set(config.target_modules):
        raise ValueError("initial adapter targets must match language-only training modules")
    for name in ("modules_to_save", "rank_pattern", "alpha_pattern", "use_dora", "use_rslora"):
        if adapter.get(name):
            raise ValueError(f"unsupported initial adapter option: {name}")
    return {**evidence, "optimizer_state_resumed": False,
            "scope": "adapter weight warm start with a newly created optimizer"}


def load_model_and_processor(
    config: TrainingConfig | Mapping[str, Any], adapter_path: str | Path | None = None,
    *, for_training: bool = False,
) -> tuple[Any, Any]:
    """Load local weights consistently for training and base/adapter evaluation."""
    import torch
    from transformers import (
        AutoConfig,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )

    config = config if isinstance(config, TrainingConfig) else TrainingConfig.from_dict(config)
    if for_training and adapter_path is not None:
        raise ValueError("training warm start uses initial_adapter_path; optimizer-state resume is not implemented")
    if for_training:
        initial_adapter_evidence(config)
    base_path = Path(config.base_model_path).resolve(strict=True)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable; CPU training is not substituted")
    if config.load_in_4bit and not config.device.startswith("cuda"):
        raise ValueError("the selected 4bit training/evaluation configuration requires CUDA")
    dtype = getattr(torch, config.compute_dtype)
    processor = AutoProcessor.from_pretrained(
        config.processor_path or str(base_path), local_files_only=True, trust_remote_code=False,
        min_pixels=config.min_pixels, max_pixels=config.max_pixels, use_fast=config.processor_use_fast,
    )
    architecture = AutoConfig.from_pretrained(str(base_path), local_files_only=True, trust_remote_code=False)
    classes = {"qwen2_vl": Qwen2VLForConditionalGeneration, "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration}
    if architecture.model_type not in classes:
        raise ValueError(f"unsupported vision-language architecture: {architecture.model_type}")
    options: dict[str, Any] = {
        "local_files_only": True, "trust_remote_code": False, "torch_dtype": dtype,
        "attn_implementation": config.attention_implementation, "device_map": {"": config.device},
    }
    if config.load_in_4bit:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = classes[architecture.model_type].from_pretrained(str(base_path), **options)
    if for_training:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        model.config.use_cache = False
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        targets = language_lora_targets(model, config.target_modules)
        freeze_visual_encoder(model)
        if config.initial_adapter_path is None:
            model = get_peft_model(model, LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                target_modules=targets, bias="none", task_type="CAUSAL_LM",
            ))
        else:
            model = PeftModel.from_pretrained(
                model, str(Path(config.initial_adapter_path).resolve()),
                is_trainable=True, local_files_only=True,
            )
        freeze_visual_encoder(model)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable or any(_visual_name(name) or "lora_" not in name for name in trainable):
            raise ValueError("trainable parameters must be language LoRA adapters only")
        model.train()
    elif adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False, local_files_only=True)
        model.eval()
    else:
        model.eval()
    return model, processor


def accumulation_groups(indices: Sequence[int], size: int) -> Iterator[list[int]]:
    if size <= 0:
        raise ValueError("accumulation size must be positive")
    for offset in range(0, len(indices), size):
        yield list(indices[offset:offset + size])


def _sampling_strata(samples: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    if not samples:
        raise ValueError("training sampling requires a nonempty dataset")
    single, multiple = [], []
    for index, sample in enumerate(samples):
        actions = sample.get("actions") if isinstance(sample, Mapping) else None
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ValueError("training sampling requires nonempty action arrays")
        (single if len(actions) == 1 else multiple).append(index)
    return single, multiple


def training_sampling_plan(
    samples: Sequence[Mapping[str, Any]], sampling_policy: str = "uniform",
) -> dict[str, Any]:
    """Label length determines training frequency only; labels are not request inputs."""
    if not isinstance(sampling_policy, str) or sampling_policy not in SAMPLING_POLICIES:
        raise ValueError("unknown training sampling_policy")
    single, multiple = _sampling_strata(samples)
    balanced = sampling_policy in {"balanced_action_chunks", "coverage_first_balanced_action_chunks"}
    if balanced and (not single or not multiple):
        raise ValueError(f"{sampling_policy} requires both single-action and multi-action samples")
    target = max(len(single), len(multiple))
    single_visits = target if balanced else len(single)
    multiple_visits = target if balanced else len(multiple)
    result = {"policy": sampling_policy, "unique_samples": len(samples),
            "single_action_unique_samples": len(single), "multi_action_unique_samples": len(multiple),
            "planned_single_action_visits_per_epoch": single_visits,
            "planned_multi_action_visits_per_epoch": multiple_visits,
            "planned_samples_per_epoch": single_visits + multiple_visits}
    if sampling_policy == "coverage_first_balanced_action_chunks":
        result.update(coverage_unique_first=len(samples), supplement_visits=2 * target - len(samples),
                      bounded_prefix_strata_balance_guaranteed=False)
    return result


def epoch_sample_indices(
    samples: Sequence[Mapping[str, Any]], sampling_policy: str = "uniform", *, seed: int, epoch: int,
) -> list[int]:
    """Keep the historical uniform shuffle; balance with full repeats and random remainder."""
    plan = training_sampling_plan(samples, sampling_policy)
    rng = random.Random(seed + epoch)
    if sampling_policy == "uniform":
        indices = list(range(len(samples)))
    elif sampling_policy == "coverage_first_balanced_action_chunks":
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        supplements = []
        for stratum in _sampling_strata(samples):
            needed = plan["planned_single_action_visits_per_epoch"] - len(stratum)
            repeats, remainder = divmod(needed, len(stratum))
            supplements.extend(stratum * repeats)
            if remainder:
                supplements.extend(rng.sample(stratum, remainder))
        rng.shuffle(supplements)
        return indices + supplements
    else:
        indices = []
        for stratum in _sampling_strata(samples):
            repeats, remainder = divmod(plan["planned_single_action_visits_per_epoch"], len(stratum))
            indices.extend(stratum * repeats)
            if remainder:
                indices.extend(rng.sample(stratum, remainder))
    rng.shuffle(indices)
    return indices


def epoch_sampling_evidence(
    samples: Sequence[Mapping[str, Any]], indices: Sequence[int], *,
    sampling_policy: str, seed: int, epoch: int,
) -> dict[str, Any]:
    counts = Counter(samples[index]["sample_id"] for index in indices)
    single_visits = sum(len(samples[index]["actions"]) == 1 for index in indices)
    return {"epoch": epoch + 1, "policy": sampling_policy, "sampling_seed": seed + epoch,
            "indices": list(indices),
            "indices_sha256": hashlib.sha256(json.dumps(list(indices), separators=(",", ":")).encode()).hexdigest(),
            "planned_samples": len(indices), "unique_samples": len(counts),
            "planned_visit_counts": dict(sorted(counts.items())),
            "single_action_visits": single_visits, "multi_action_visits": len(indices) - single_visits}


def training_objective_plan(samples: Sequence[Mapping[str, Any]], config: TrainingConfig) -> dict[str, Any]:
    """Freeze nominal stratum weights from the complete bounded visit prefix.

    This is a label-side training objective, never a model request. Equal nominal
    weight mass does not assert equal observed losses or gradient contributions.
    """
    weights = {"single_action": 1.0, "multi_action": 1.0}
    if config.loss_strategy == "uniform":
        return {"strategy": "uniform", "weights": weights, "registered_prefix": None,
                "scope": "historical mean per-example assistant-token loss; no stratum reweighting"}
    prefix, single, multiple, micro = [], 0, 0, 0
    limit = min(config.max_steps, 1) if config.smoke else config.max_steps
    stopped = False
    for epoch in range(config.epochs):
        indices = epoch_sample_indices(samples, config.sampling_policy, seed=config.seed, epoch=epoch)
        for group_index, group in enumerate(accumulation_groups(indices, config.gradient_accumulation_steps)):
            if len(prefix) >= min(limit, config.max_optimizer_attempts) or micro + len(group) > config.max_micro_steps:
                stopped = True
                break
            count = sum(len(samples[index]["actions"]) == 1 for index in group)
            single += count
            multiple += len(group) - count
            micro += len(group)
            prefix.append({"epoch": epoch + 1, "group_index": group_index,
                           "sample_ids": [samples[index]["sample_id"] for index in group]})
        if stopped:
            break
    if not single or not multiple:
        raise ValueError("registered_stratum_balanced requires both strata in the bounded training prefix before loading")
    weights = {"single_action": micro / (2 * single), "multi_action": micro / (2 * multiple)}
    mass = {"single_action": single * weights["single_action"], "multi_action": multiple * weights["multi_action"]}
    return {"strategy": config.loss_strategy, "weights": weights, "registered_prefix": prefix,
            "prefix_sha256": hashlib.sha256(json.dumps(prefix, sort_keys=True, separators=(",", ":"),
                                                       ensure_ascii=False).encode("utf-8")).hexdigest(),
            "prefix_counts": {"single_action": single, "multi_action": multiple, "micro_steps": micro,
                              "optimizer_attempts": len(prefix)},
            "nominal_weight_mass": mass, "mean_nominal_weight": sum(mass.values()) / micro,
            "caps": {"applied_updates": config.max_steps, "optimizer_attempts": config.max_optimizer_attempts,
                     "micro_steps": config.max_micro_steps, "epochs": config.epochs},
            "denominator": "actual accumulation group size, never sum of weights",
            "scope": "nominal mass only; no 50-percent actual loss/gradient claim; skipped updates never extend this prefix"}


def _example_loss_weight(sample: Mapping[str, Any], weights: Mapping[str, float] | None) -> float:
    if weights is None:
        return 1.0
    actions = sample.get("actions")
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("stratum loss weights require a nonempty complete action array")
    return weights["single_action" if len(actions) == 1 else "multi_action"]


def backward_group(
    samples: Sequence[Mapping[str, Any]], forward: Callable[[Mapping[str, Any]], tuple[Any, int]],
    backward: Callable[[Any], None], *, loss_weights: Mapping[str, float] | None = None,
) -> Iterator[tuple[Mapping[str, Any], float, int]]:
    """Scale by actual group size; optional fixed weights do not renormalize it."""
    if not samples:
        raise ValueError("cannot accumulate an empty group")
    if loss_weights is not None and (
        set(loss_weights) != {"single_action", "multi_action"} or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in loss_weights.values()
        )
    ):
        raise ValueError("positive finite fixed single/multi loss weights required")
    for sample in samples:
        loss, supervised_tokens = forward(sample)
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise ValueError(f"{sample['sample_id']}: non-finite training loss")
        if loss_weights is None:
            backward(loss / len(samples))
        else:
            backward(loss * _example_loss_weight(sample, loss_weights) / len(samples))
        yield sample, value, supervised_tokens


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    # Windows readers/antivirus may briefly deny an otherwise atomic replace.
    # Keep the valid destination and completed temporary JSON until it succeeds;
    # never fall back to truncating the destination or hiding a persistent lock.
    deadline = time.monotonic() + 2.0
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 7:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(0.025 * 2 ** attempt, 0.25, remaining))


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _source_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).resolve(), Path(__file__).with_name("prompts.py"),
             Path(__file__).with_name("data.py"),
             Path(__file__).with_name("numeric_value_loss.py"),
             Path(__file__).with_name("offset_geometry_loss.py"),
             Path(__file__).with_name("native_spatial_head.py"),
             Path(__file__).with_name("native_spatial_artifacts.py"),
             Path(__file__).with_name("native_spatial_grid_head.py"),
             Path(__file__).with_name("native_spatial_grid_artifacts.py"),
             Path(__file__).with_name("native_spatial_grid_geometry.py"),
             Path(__file__).with_name("pointer_protocol.py"),
             Path(__file__).with_name("pointer_candidates_v2.py"),
             Path(__file__).with_name("pointer_candidates_v3.py"),
             Path(__file__).with_name("candidate_grounding.py"),
             Path(__file__).with_name("visual_pointer_input.py"),
             Path(__file__).with_name("visual_pointer.py"),
             Path(__file__).with_name("visual_pointer_v2.py"),
             Path(__file__).with_name("candidate_gallery.py"),
             Path(__file__).with_name("visual_pointer_prompts.py"),
             repository / "src" / "gui_agent" / "schemas.py",
             repository / "src" / "gui_agent" / "datasets" / "schemas.py",
             repository / "scripts" / "week5_train_lora.py"]
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def _parameter_hash(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            # A byte view works for bfloat16, which numpy cannot represent.
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _data_evidence(path: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"jsonl_path": str(Path(path).resolve()), "jsonl_sha256": sha256_file(path),
            "samples": len(rows), "groups": len({row["group_id"] for row in rows}),
            "records": [{**{name: row[name] for name in (
                "sample_id", "group_id", "source", "source_split", "image_path",
                "image_sha256", "image_file_sha256",
            )}, **{name: row[name] for name in ("input_image_protocol", *VISUAL_ROW_FIELDS, "training_admission")
                   if name in row}} for row in rows]}


def _saved_file_evidence(path: Path, *, adapter: bool) -> dict[str, Any]:
    """Assert reload inputs exist; actual reload is verified by held-out evaluation."""
    required = ({"adapter_model.safetensors", "adapter_config.json"} if adapter else
                {"preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json"})
    missing = [name for name in sorted(required)
               if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if not adapter:
        for alternatives in (("tokenizer.json",), ("chat_template.jinja", "chat_template.json")):
            if not any((path / name).is_file() and (path / name).stat().st_size > 0
                       for name in alternatives):
                missing.append(" or ".join(alternatives))
    if missing:
        raise RuntimeError(f"saved {'adapter' if adapter else 'processor'} is incomplete: "
                           + ", ".join(missing))
    return {"path": str(path), "required_files_present": True,
            "reload_verified": False, "files": {
                file.relative_to(path).as_posix(): {"bytes": file.stat().st_size,
                                                   "sha256": sha256_file(file)}
                for file in sorted(path.rglob("*")) if file.is_file()
            }}


def _base_model_evidence(base: Path, config: TrainingConfig) -> dict[str, Any]:
    base_files = [file for file in sorted(base.iterdir())
                  if file.is_file() and file.suffix in {".json", ".safetensors", ".bin", ".txt"}]
    files = {file.name: {"bytes": file.stat().st_size, "sha256": sha256_file(file)}
             for file in base_files}
    evidence: dict[str, Any] = {
        "path": str(base), "model_id": config.base_model_id,
        "expected_revision": config.base_model_revision,
        "files": {name: item["sha256"] for name, item in files.items()}, "file_details": files,
        "provenance_status": "unverified", "upstream_revision_verified_by_trainer": False,
        "provenance_scope": "local weight bytes/SHA and model/revision claims compared to the download manifest; no independent upstream trust proof",
    }
    manifest_path = base / "week5_download_manifest.json"
    if not manifest_path.is_file():
        evidence["download_manifest"] = {"status": "missing", "path": str(manifest_path),
                                         "weight_hashes_match": None}
        return evidence
    manifest_file = {"path": str(manifest_path), "bytes": manifest_path.stat().st_size,
                     "sha256": files[manifest_path.name]["sha256"], "weight_hashes_match": None}
    evidence["download_manifest"] = manifest_file
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        evidence["provenance_status"] = "manifest_invalid"
        manifest_file["error"] = str(exc)
        return evidence
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        evidence["provenance_status"] = "manifest_invalid"
        manifest_file["error"] = "base-model download manifest must contain a files array"
        return evidence
    declared = {item["path"]: item for item in manifest["files"]
                if isinstance(item, dict) and isinstance(item.get("path"), str)}
    weight_names = sorted(name for name in files if name.endswith((".safetensors", ".bin")))
    checks = [{"path": name, "actual_bytes": files[name]["bytes"],
               "actual_sha256": files[name]["sha256"],
               "manifest_bytes": declared.get(name, {}).get("bytes"),
               "manifest_sha256": declared.get(name, {}).get("sha256"),
               "matched": declared.get(name, {}).get("bytes") == files[name]["bytes"]
               and declared.get(name, {}).get("sha256") == files[name]["sha256"]}
              for name in weight_names]
    declared_weights = {name for name in declared if name.endswith((".safetensors", ".bin"))}
    weights_match = bool(checks) and all(item["matched"] for item in checks) and declared_weights == set(weight_names)
    metadata_matches = (manifest.get("model_id") == config.base_model_id
                        and manifest.get("revision") == config.base_model_revision)
    matched = weights_match and metadata_matches
    evidence["provenance_status"] = "download_manifest_matched" if matched else "manifest_mismatch"
    evidence["download_manifest"] = {
        "path": str(manifest_path), "bytes": manifest_path.stat().st_size,
        "sha256": files[manifest_path.name]["sha256"],
        "declared_model_id": manifest.get("model_id"), "declared_revision": manifest.get("revision"),
        "metadata_matches_config": metadata_matches, "weight_hashes_match": weights_match,
        "weight_checks": checks, "weight_serving_source": manifest.get("weight_serving_source"),
    }
    return evidence


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Run actual AdamW updates; all terminal states have persisted evidence.

    One microbatch is one screenshot. Each accumulation group optimizes the
    mean of its per-example assistant-token losses. The last partial group is
    divided by its own size. Validation is read separately and never backpropagated.
    A smoke run still trains, but at most one optimizer update; it is labelled smoke.
    """
    if not config.output_dir:
        raise ValueError("output_dir is required")
    output = Path(config.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite existing training evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "running", "run_kind": "smoke" if config.smoke else "training",
        "started_at": datetime.now(UTC).isoformat(), "config": config.to_dict(),
        "versions": _versions(), "source_hashes": _source_hashes(),
        "optimizer_steps": 0, "optimizer_attempts": 0, "skipped_optimizer_steps": 0,
        "micro_steps": 0, "samples_seen": 0, "epochs_completed": 0,
        "step_metrics": [], "validation_metrics": [], "artifacts": {},
        "loss_scope": "assistant action JSON and assistant end token only; prompt and vision masked",
        "accumulation_objective": "mean per-example assistant-token loss; actual final group size",
        "seed": config.seed, "bitwise_determinism_claimed": False,
        "optimizer_state_resumed": False,
        "sampling_policy": config.sampling_policy, "epoch_sampling": [],
        "loss_strategy": config.loss_strategy, "loss_objective_prefix_match": None,
        "token_loss_strategy": config.token_loss_strategy,
        "cuda_peak_stats_reset_before_loop": False,
        "actual_token_objective": {"completed_microbatches": 0, "supervised_tokens": 0,
                                   "numeric_active_tokens": 0, "boundary_mixed_tokens": 0,
                                   "masked_numeric_tokens": 0, "nominal_token_weight_mass": 0.0,
                                   "raw_ce_sum": 0.0, "token_weighted_ce_sum": 0.0,
                                   "class_times_token_weighted_ce_sum": 0.0,
                                   "scope": "completed training microbatches; nominal weights are not loss/gradient share"},
        "actual_loss_objective": {"single_action_visits": 0, "multi_action_visits": 0,
                                  "single_action_weight_mass": 0.0, "multi_action_weight_mass": 0.0,
                                  "raw_example_loss_sum": 0.0, "weighted_example_loss_sum": 0.0,
                                  "scope": "completed microbatches, including skipped attempts/in-progress groups; nominal mass is not gradient share"},
        "actual_sampling": {"single_action_visits": 0, "multi_action_visits": 0,
                            "unique_samples": 0, "visit_counts": {},
                            "scope": "completed training microbatches, including skipped-update attempts and in-progress groups"},
        "input_image_protocol": config.input_image_protocol,
        "visual_prompt_version": config.visual_prompt_version,
        "budget_caps": {"applied_updates": config.max_steps, "micro_steps": config.max_micro_steps,
                        "optimizer_attempts": config.max_optimizer_attempts},
    }
    model = processor = spatial_head = None
    spatial_bindings = None
    initial_hash = None
    _write_json(output / "training_config.json", config.to_dict())
    _write_json(output / "training_summary.json", summary)

    def persist() -> None:
        summary["elapsed_seconds"] = time.perf_counter() - started
        _write_json(output / "training_summary.json", summary)

    try:
        if not config.train_jsonl:
            raise ValueError("train_jsonl is required")
        numeric_decoder = None
        geometry_vocabulary = None
        if config.token_loss_strategy != "uniform":
            from .numeric_value_loss import load_bound_decoder

            numeric_decoder = load_bound_decoder(config.numeric_tokenizer_json_path,
                                                 config.numeric_tokenizer_json_sha256)
            summary["token_loss_objective"] = {
                "strategy": config.token_loss_strategy, "numeric_VALUE_weight": 4.0,
                "other_active_weight": 1.0, "masked_weight": 0.0,
                "normalizer": "original shifted active-token count",
                "metadata_path": config.numeric_tokenizer_json_path,
                "metadata_sha256": numeric_decoder.source_sha256,
                "boundary_mixed": "whole indivisible token4, explicitly counted",
                "default_model_CE_requested": False,
            }
            if config.token_loss_strategy == "candidate4_offset8_teacher_forced_digit_place_geometry8":
                from .offset_geometry_loss import NumberVocabulary

                geometry_vocabulary = NumberVocabulary.from_decoder(numeric_decoder)
                summary["token_loss_objective"].pop("numeric_VALUE_weight")
                summary["token_loss_objective"].update(
                    candidate_VALUE_weight=4.0, signed_offset_VALUE_weight=8.0,
                    geometry_coefficient=8.0,
                    bare_digit_token_ids=[list(item) for item in geometry_vocabulary.shapes[(b"", b"")]],
                    boundary_mixed="whole indivisible offset-overlap token8, candidate-only token4; explicitly counted",
                    geometry_probability_space="conditional softmax over exactly ten pinned bare ASCII digit tokens",
                    geometry_error="expected absolute digit error times decimal place / max(1, original-native axis dimension-1)",
                    geometry_scope="teacher-forced digit places; insertion, length and sign errors are not measured by auxiliary",
                )
                summary["actual_geometry_objective"] = {
                    "completed_microbatches": 0, "candidate_active_tokens": 0, "offset_active_tokens": 0,
                    "offset_sign_active_tokens": 0, "candidate_value_fields": 0, "offset_value_fields": 0,
                    "offset_signed_fields": 0, "offset_digit_positions": 0, "geometry_entries": 0,
                    "geometry_auxiliary_skipped": 0, "geometry_skip_reasons": {},
                    "candidate_categorical_fields": 0, "cross_field_tokens": 0,
                    "raw_ce_sum": 0.0, "main_weighted_ce_sum": 0.0, "geometry_auxiliary_sum": 0.0,
                    "total_loss_sum": 0.0, "geometry_expected_absolute_error_sum": 0.0,
                    "class_times_total_loss_sum": 0.0,
                    "scope": "completed training microbatches; categorical markers excluded from auxiliary skips",
                }
        if config.target_protocol in SPATIAL_TARGET_PROTOCOLS:
            from .numeric_value_loss import load_bound_decoder

            spatial_decoder = load_bound_decoder(config.spatial_tokenizer_json_path, config.spatial_tokenizer_json_sha256)
            summary["spatial_objective"] = {
                "semantic_CE_weight": 1.0, "point_loss_weight": 1.0, "class_weights": 1.0,
                "manual_CE_calls_per_microbatch": 1, "model_forward_calls_per_microbatch": 1,
                "default_model_CE_requested": False, "separate_diagnostic_backward_calls": 0,
                "point_loss": "mean endpoint/axis SmoothL1 normalized native xy",
                "beta_axis": "20/max(1,native_axis-1)", "completed_microbatches": 0,
                "semantic_ce_sum": 0.0, "point_loss_sum": 0.0, "total_loss_sum": 0.0,
                "point_count": 0, "supervised_tokens": 0,
            }
            summary["loss_scope"] = "complete semantic assistant JSON/EOS CE1 plus separate normalized native point SmoothL1; prefix and vision masked"
        if config.target_protocol == "native_spatial_grid_head":
            spatial_core, _ = _spatial_modules(config.target_protocol)
            summary["spatial_objective"].update(
                grid_auxiliary_CE_weight=0.5, grid_aux_ce_sum=0.0,
                grid_auxiliary_target="input-valid anchor Gaussian native distance sigma20px; labels only in loss",
                grid_refine_loss_weight=0.5, grid_refine_loss_beta=0.1, grid_refine_loss_sum=0.0,
                grid_refine_target="label-only eligible-cell Gaussian native distance sigma20px; local fractions inside input-derived bounds; nondegenerate axes only",
                head_configuration=spatial_core.head_configuration(),
            )
            summary["loss_scope"] += "; plus grid auxiliary CE0.5 and eligible-cell local-fraction SmoothL1(beta0.1)0.5 after all cell predictions"
        registry = config.visual_registry()
        train = load_prepared_records(config.train_jsonl, target_protocol=config.target_protocol,
                                      input_image_protocol=config.input_image_protocol, visual_registry=registry,
                                      visual_prompt_version=config.visual_prompt_version)
        validation = (
            load_prepared_records(config.validation_jsonl, target_protocol=config.target_protocol,
                                  input_image_protocol=config.input_image_protocol, visual_registry=registry,
                                  visual_prompt_version=config.visual_prompt_version)
            if config.validation_jsonl else []
        )
        ensure_separate_validation(train, validation)
        summary["train_data"] = _data_evidence(config.train_jsonl, train)
        summary["validation_data"] = (
            _data_evidence(config.validation_jsonl, validation) if config.validation_jsonl else None
        )
        # Reject a degenerate balanced dataset before any model or CUDA loading.
        summary["sampler"] = training_sampling_plan(train, config.sampling_policy)
        summary["loss_objective"] = training_objective_plan(train, config)
        objective_weights = (summary["loss_objective"]["weights"]
                             if config.loss_strategy == "registered_stratum_balanced" else None)
        if config.target_protocol in SPATIAL_TARGET_PROTOCOLS and (
                objective_weights != {"single_action": 1.0, "multi_action": 1.0}
                or summary["loss_objective"]["prefix_counts"] != {
                    "single_action": 572, "multi_action": 572, "micro_steps": 1144, "optimizer_attempts": 144}
                and not config.smoke):
            raise ValueError("native_spatial_head requires the registered exact572/572 visits and class weights1")
        summary["accumulation_objective"] = (
            "fixed stratum-weighted per-example assistant-token loss / actual group size"
            if objective_weights is not None else summary["accumulation_objective"]
        )
        persist()
        base = Path(config.base_model_path).resolve(strict=True)
        summary["base_model"] = _base_model_evidence(base, config)
        summary["initial_adapter"] = initial_adapter_evidence(config)
        persist()
        if summary["base_model"]["provenance_status"].startswith("manifest_"):
            raise ValueError("base-model weights or revision differ from the download manifest")
        import torch

        random.seed(config.seed)
        import numpy as np
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device.startswith("cuda"):
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        load_started = time.perf_counter()
        model, processor = load_model_and_processor(config, for_training=True)
        summary["model_load_seconds"] = time.perf_counter() - load_started
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("model has no trainable adapter parameters")
        summary["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
        summary["trainable_names"] = [name for name, parameter in model.named_parameters()
                                      if parameter.requires_grad]
        summary["frozen_visual_parameters"] = freeze_visual_encoder(model)
        initial_hash = _parameter_hash(model)
        summary["initial_adapter_parameter_sha256"] = initial_hash
        lora_parameters = list(parameters)
        head_parameters = []
        optimizer_groups = parameters
        if config.target_protocol in SPATIAL_TARGET_PROTOCOLS:
            spatial_core, spatial_artifacts = _spatial_modules(config.target_protocol)
            save_spatial_head_artifact = spatial_artifacts.save_spatial_head_artifact
            make_spatial_head, resolve_final_norm = spatial_core.make_spatial_head, spatial_core.resolve_final_norm
            from .visual_pointer_prompts import GALLERY_SPATIAL_HEADER

            resolve_final_norm(model)
            spatial_head = make_spatial_head(seed=config.seed).to(device=config.device, dtype=torch.float32)
            spatial_head.train()
            spatial_bindings = {
                "base_model_id": config.base_model_id, "base_model_revision": config.base_model_revision,
                "base_model_config_sha256": sha256_file(Path(config.base_model_path) / "config.json"),
                "eos_token_id": spatial_decoder.eos_id,
                "prompt_version": config.visual_prompt_version,
                "prompt_sha256": hashlib.sha256(GALLERY_SPATIAL_HEADER.encode("utf8")).hexdigest(),
                "tokenizer_json_sha256": spatial_decoder.source_sha256,
                "core_source_sha256": sha256_file(Path(spatial_core.__file__)),
                "training_source_sha256": sha256_file(Path(__file__)),
            }
            if config.target_protocol == "native_spatial_grid_head":
                spatial_bindings.update(
                    geometry_source_sha256=sha256_file(Path(__file__).with_name("native_spatial_grid_geometry.py")),
                    marker_source_sha256=sha256_file(Path(__file__).with_name("native_spatial_head.py")),
                )
            initial_head = save_spatial_head_artifact(spatial_head, output, role="initial", bindings=spatial_bindings)
            summary["spatial_head_initial"] = initial_head
            summary["artifacts"].update({"spatial_head_initial": str(output / initial_head["filename"]),
                                         "spatial_head_config": str(output / "spatial_head_config.json"),
                                         "spatial_head_manifest": str(output / "spatial_head_manifest.json")})
            head_parameters = list(spatial_head.parameters())
            parameters += head_parameters
            optimizer_groups = [{"params": lora_parameters, "lr": config.learning_rate},
                                {"params": head_parameters, "lr": config.spatial_head_learning_rate}]
            summary["spatial_head_trainable_parameters"] = sum(parameter.numel() for parameter in head_parameters)
            summary["optimizer_parameter_groups"] = [
                {"owner": "language_lora", "learning_rate": config.learning_rate,
                 "parameters": sum(parameter.numel() for parameter in lora_parameters)},
                {"owner": "external_spatial_head", "learning_rate": config.spatial_head_learning_rate,
                 "parameters": sum(parameter.numel() for parameter in head_parameters)},
            ]
        optimizer = torch.optim.AdamW(optimizer_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=(
            config.device.startswith("cuda") and config.compute_dtype == "float16"
        ))
        dtype = getattr(torch, config.compute_dtype)
        use_autocast = config.device.startswith("cuda") and dtype != torch.float32
        step_limit = min(config.max_steps or 1, 1) if config.smoke else config.max_steps
        budget_stop = None
        summary["effective_max_steps"] = step_limit
        summary["planned_epoch_optimizer_groups"] = math.ceil(
            summary["sampler"]["planned_samples_per_epoch"] / config.gradient_accumulation_steps
        )
        if config.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(config.device)
            summary["cuda_peak_stats_reset_before_loop"] = True
        persist()

        forward_diagnostics: dict[str, dict[str, Any]] = {}

        def forward(sample: Mapping[str, Any]) -> tuple[Any, int]:
            encoded = encode_training_sample(processor, sample, config)
            token_count = int((encoded["labels"][:, 1:] != -100).sum().item())
            if not token_count:
                raise ValueError(f"{sample['sample_id']}: no shifted assistant target tokens")
            batch = {key: value.to(config.device) for key, value in encoded.items() if not key.startswith("_")}
            with (torch.autocast("cuda", dtype=dtype) if use_autocast else nullcontext()):
                if spatial_head is not None:
                    spatial_core, _ = _spatial_modules(config.target_protocol)
                    forward_objective = spatial_core.forward_objective

                    result = forward_objective(model, spatial_head, batch, encoded["_native_spatial_plan"])
                    loss = result["total_loss"]
                    if result["supervised_tokens"] != token_count:
                        raise ValueError("spatial objective shifted token denominator differs")
                    forward_diagnostics[sample["sample_id"]] = {
                        **{name: float(result[name].detach().item()) for name in ("semantic_ce", "point_loss", "total_loss")},
                        **{name: result[name] for name in ("point_count", "supervised_tokens", "manual_CE_calls", "model_forward_calls",
                                                          "default_model_CE_requested")},
                        "marker_positions": list(encoded["_native_spatial_plan"].marker_positions),
                        "head_activation_diagnostics": result["head_activation_diagnostics"],
                        **({
                            "grid_aux_ce": float(result["grid_aux_ce"].detach().item()),
                            "grid_refine_loss": float(result["grid_refine_loss"].detach().item()),
                            "grid_refine_diagnostics": result["grid_refine_diagnostics"],
                            "teacher_prediction_unit": result["prediction_unit"].detach().cpu().tolist(),
                            "target_unit": [list(point) for point in encoded["_native_spatial_plan"].target_unit],
                            "native_size": [encoded["_native_spatial_plan"].native_width, encoded["_native_spatial_plan"].native_height],
                        }
                           if config.target_protocol == "native_spatial_grid_head" else {}),
                    }
                elif config.token_loss_strategy == "uniform":
                    loss = model(**batch).loss
                elif config.token_loss_strategy == "candidate4_offset8_teacher_forced_digit_place_geometry8":
                    from .offset_geometry_loss import (
                        forward_objective,
                        plan_from_encoded_sample,
                        plan_statistics,
                    )

                    plan = plan_from_encoded_sample(
                        encoded, sample["candidate_target"], numeric_decoder,
                        native_width=sample["image_width"], native_height=sample["image_height"],
                        vocabulary=geometry_vocabulary,
                    )
                    result = forward_objective(model, batch, plan)
                    loss = result["total_loss"]
                    if result["supervised_tokens"] != token_count:
                        raise ValueError("geometry objective shifted token denominator differs")
                    forward_diagnostics[sample["sample_id"]] = {
                        **plan_statistics(plan),
                        **{name: float(result[name].detach().item()) for name in (
                            "raw_ce", "main_weighted_ce", "geometry_auxiliary", "total_loss",
                            "geometry_expected_absolute_error_sum",
                        )},
                        "token_weighted_ce": float(loss.detach().item()),
                        "token_weighted_ce_scope": "total main weighted CE plus geometry auxiliary",
                        "supervised_tokens": token_count,
                        "nominal_token_weight_mass": result["nominal_token_weight_mass"],
                        "manual_CE_calls": result["manual_CE_calls"],
                        "default_model_CE_requested": False,
                    }
                else:
                    from .numeric_value_loss import forward_objective, weights_from_encoded_sample

                    plan = weights_from_encoded_sample(encoded, sample["candidate_target"], numeric_decoder)
                    token_weights = torch.tensor([plan.weights], dtype=torch.float32, device=config.device)
                    result = forward_objective(model, batch, strategy=config.token_loss_strategy,
                                               token_weights=token_weights)
                    loss = result["weighted_ce"]
                    if result["supervised_tokens"] != token_count:
                        raise ValueError("numeric objective shifted token denominator differs")
                    forward_diagnostics[sample["sample_id"]] = {
                        "raw_ce": float(result["raw_ce"].detach().item()),
                        "token_weighted_ce": float(loss.detach().item()),
                        "supervised_tokens": token_count,
                        "numeric_active_tokens": len(plan.numeric_tokens) - plan.masked_numeric_tokens,
                        "boundary_mixed_tokens": len(plan.boundary_mixed_tokens),
                        "masked_numeric_tokens": plan.masked_numeric_tokens,
                        "nominal_token_weight_mass": result["nominal_token_weight_mass"],
                        "manual_CE_calls": result["manual_CE_calls"],
                        "default_model_CE_requested": False,
                    }
            return loss, token_count

        for epoch in range(config.epochs):
            summary["epoch_started"] = epoch + 1
            indices = epoch_sample_indices(train, config.sampling_policy, seed=config.seed, epoch=epoch)
            summary["epoch_sampling"].append(epoch_sampling_evidence(
                train, indices, sampling_policy=config.sampling_policy, seed=config.seed, epoch=epoch,
            ))
            persist()
            groups = list(accumulation_groups(indices, config.gradient_accumulation_steps))
            for group_index, group in enumerate(groups):
                if config.max_optimizer_attempts is not None and summary["optimizer_attempts"] >= config.max_optimizer_attempts:
                    budget_stop = "max_optimizer_attempts"
                    break
                if config.max_micro_steps is not None and summary["micro_steps"] + len(group) > config.max_micro_steps:
                    budget_stop = "max_micro_steps"
                    break
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses, token_counts, sample_ids, example_weights = [], [], [], []
                token_losses, token_diagnostics, spatial_diagnostics = [], [], []
                samples = [train[index] for index in group]
                for sample, loss, tokens in backward_group(samples, forward,
                                                          lambda loss: scaler.scale(loss).backward(),
                                                          loss_weights=objective_weights):
                    summary["micro_steps"] += 1
                    summary["samples_seen"] += 1
                    sampling = summary["actual_sampling"]
                    sampling["single_action_visits" if len(sample["actions"]) == 1 else "multi_action_visits"] += 1
                    identifier = sample["sample_id"]
                    if identifier not in sampling["visit_counts"]:
                        sampling["unique_samples"] += 1
                    sampling["visit_counts"][identifier] = sampling["visit_counts"].get(identifier, 0) + 1
                    if numeric_decoder is not None:
                        diagnostic = dict(forward_diagnostics[sample["sample_id"]])
                        token_losses.append(loss)
                        loss = diagnostic["raw_ce"]
                        diagnostic.update(sample_id=sample["sample_id"],
                                          micro_step=summary["micro_steps"], epoch=epoch + 1,
                                          group_index=group_index,
                                          class_weight=_example_loss_weight(sample, objective_weights))
                        diagnostic["class_times_token_weighted_ce"] = (
                            diagnostic["token_weighted_ce"] * diagnostic["class_weight"]
                        )
                        if not math.isfinite(loss):
                            raise ValueError(f"{sample['sample_id']}: non-finite raw CE")
                        token_diagnostics.append(diagnostic)
                        with (output / "token_objective_samples.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                            handle.write(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                        token_objective = summary["actual_token_objective"]
                        token_objective["completed_microbatches"] += 1
                        for name in ("supervised_tokens", "numeric_active_tokens", "boundary_mixed_tokens",
                                     "masked_numeric_tokens", "nominal_token_weight_mass"):
                            token_objective[name] += diagnostic[name]
                        token_objective["raw_ce_sum"] += loss
                        token_objective["token_weighted_ce_sum"] += diagnostic["token_weighted_ce"]
                        token_objective["class_times_token_weighted_ce_sum"] += diagnostic["class_times_token_weighted_ce"]
                        if geometry_vocabulary is not None:
                            geometry = summary["actual_geometry_objective"]
                            geometry["completed_microbatches"] += 1
                            for name in ("candidate_active_tokens", "offset_active_tokens", "offset_sign_active_tokens",
                                         "candidate_value_fields", "offset_value_fields", "offset_signed_fields",
                                         "offset_digit_positions", "geometry_entries", "geometry_auxiliary_skipped",
                                         "candidate_categorical_fields", "cross_field_tokens"):
                                geometry[name] += diagnostic[name]
                            for reason, count in diagnostic["geometry_skip_reasons"].items():
                                geometry["geometry_skip_reasons"][reason] = geometry["geometry_skip_reasons"].get(reason, 0) + count
                            for name in ("raw_ce", "main_weighted_ce", "geometry_auxiliary", "total_loss"):
                                geometry[f"{name}_sum"] += diagnostic[name]
                            geometry["geometry_expected_absolute_error_sum"] += diagnostic["geometry_expected_absolute_error_sum"]
                            geometry["class_times_total_loss_sum"] += diagnostic["class_times_token_weighted_ce"]
                    if spatial_head is not None:
                        diagnostic = {**forward_diagnostics[sample["sample_id"]], "sample_id": sample["sample_id"],
                                      "micro_step": summary["micro_steps"], "epoch": epoch + 1, "group_index": group_index}
                        spatial_diagnostics.append(diagnostic)
                        with (output / "spatial_objective_samples.jsonl").open("a", encoding="utf8", newline="\n") as handle:
                            handle.write(json.dumps(diagnostic, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                        spatial = summary["spatial_objective"]
                        spatial["completed_microbatches"] += 1
                        for name in ("semantic_ce", "point_loss", "total_loss"):
                            spatial[name + "_sum"] += diagnostic[name]
                        for name in ("point_count", "supervised_tokens"):
                            spatial[name] += diagnostic[name]
                        if config.target_protocol == "native_spatial_grid_head":
                            spatial["grid_aux_ce_sum"] += diagnostic["grid_aux_ce"]
                            spatial["grid_refine_loss_sum"] += diagnostic["grid_refine_loss"]
                    losses.append(loss)
                    token_counts.append(tokens)
                    sample_ids.append(sample["sample_id"])
                    weight = _example_loss_weight(sample, objective_weights)
                    example_weights.append(weight)
                    objective = summary["actual_loss_objective"]
                    stratum = "single_action" if len(sample["actions"]) == 1 else "multi_action"
                    objective[f"{stratum}_visits"] += 1
                    objective[f"{stratum}_weight_mass"] += weight
                    objective["raw_example_loss_sum"] += loss
                    objective["weighted_example_loss_sum"] += loss * weight
                    # Preserve evidence even if the next microbatch is interrupted.
                    summary["in_progress_group"] = {
                        "epoch": epoch + 1, "group_index": group_index,
                        "planned_size": len(group), "sample_ids": sample_ids.copy(),
                        "losses": losses.copy(), "supervised_tokens": token_counts.copy(),
                        "loss_weights": example_weights.copy(),
                    }
                    persist()
                scaler.unscale_(optimizer)
                spatial_preclip_norms = None
                if spatial_head is not None:
                    def preclip_norm(group_parameters):
                        gradients = [parameter.grad.detach().float().norm(2) for parameter in group_parameters if parameter.grad is not None]
                        value = float(torch.stack(gradients).norm(2).item()) if gradients else 0.0
                        return value if math.isfinite(value) else None
                    spatial_preclip_norms = {"lora_preclip_norm": preclip_norm(lora_parameters),
                                             "head_preclip_norm": preclip_norm(head_parameters),
                                             "scope": "actual accumulated total-objective gradients after unscale, before one combined clip",
                                             "additional_backward_calls": 0}
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm,
                                                      error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
                summary["optimizer_attempts"] += 1
                summary["skipped_optimizer_steps"] += int(skipped)
                summary["optimizer_steps"] += int(not skipped)
                if config.device.startswith("cuda"):
                    torch.cuda.synchronize(config.device)
                metric = {
                    "epoch": epoch + 1, "group_index": group_index,
                    "optimizer_step": summary["optimizer_steps"], "update_applied": not skipped,
                    "optimizer_attempt": summary["optimizer_attempts"], "micro_steps": summary["micro_steps"],
                    "effective_batch_size": len(group), "sample_ids": sample_ids,
                    "mean_example_loss": sum(losses) / len(losses),
                    "mean_weighted_example_loss": sum(loss * weight for loss, weight in zip(losses, example_weights)) / len(losses),
                    "loss_strategy": config.loss_strategy, "loss_weights": example_weights,
                    "token_weighted_loss": sum(loss * tokens for loss, tokens in zip(losses, token_counts))
                    / sum(token_counts), "supervised_tokens": sum(token_counts),
                    "gradient_norm": float(norm) if math.isfinite(float(norm)) else None,
                    "scaler_before": previous_scale, "scaler_after": scaler.get_scale(),
                    "elapsed_seconds": time.perf_counter() - step_started,
                }
                if numeric_decoder is not None:
                    metric.update(
                        token_loss_strategy=config.token_loss_strategy,
                        token_weighted_example_losses=token_losses,
                        mean_token_objective_loss=sum(token_losses) / len(token_losses),
                        mean_class_times_token_objective_loss=sum(
                            loss * weight for loss, weight in zip(token_losses, example_weights)
                        ) / len(token_losses),
                        token_objective_samples=token_diagnostics,
                    )
                    if geometry_vocabulary is not None:
                        metric.update(**{
                            f"mean_{name}": sum(item[name] for item in token_diagnostics) / len(token_diagnostics)
                            for name in ("main_weighted_ce", "geometry_auxiliary", "total_loss")
                        })
                if spatial_head is not None:
                    metric.update(**spatial_preclip_norms, spatial_objective_samples=spatial_diagnostics,
                                  **{f"mean_{name}": sum(item[name] for item in spatial_diagnostics) / len(spatial_diagnostics)
                                     for name in (("semantic_ce", "point_loss", "total_loss")
                                                  + (("grid_aux_ce", "grid_refine_loss")
                                                     if config.target_protocol == "native_spatial_grid_head" else ()))})
                summary["step_metrics"].append(metric)
                summary.pop("in_progress_group", None)
                persist()
                if step_limit is not None and summary["optimizer_steps"] >= step_limit:
                    if group_index == len(groups) - 1:
                        summary["epochs_completed"] += 1
                    break
            else:
                summary["epochs_completed"] += 1
            if validation:
                model.eval()
                values, tokens = [], []
                validation_started = time.perf_counter()
                with torch.no_grad():
                    for sample in validation:
                        loss, count = forward(sample)
                        value = (forward_diagnostics[sample["sample_id"]]["raw_ce"]
                                 if numeric_decoder is not None else float(loss.item()))
                        if not math.isfinite(value):
                            raise ValueError(f"{sample['sample_id']}: non-finite validation loss")
                        values.append(value)
                        tokens.append(count)
                summary["validation_metrics"].append({
                    "epoch": epoch + 1, "optimizer_step": summary["optimizer_steps"],
                    "examples": len(values), "mean_example_loss": sum(values) / len(values),
                    "token_weighted_loss": sum(value * count for value, count in zip(values, tokens))
                    / sum(tokens), "supervised_tokens": sum(tokens),
                    "elapsed_seconds": time.perf_counter() - validation_started,
                })
                model.train()
                persist()
            if budget_stop is not None or (step_limit is not None and summary["optimizer_steps"] >= step_limit):
                break
        if summary["optimizer_steps"] == 0:
            raise RuntimeError("no optimizer update completed; training cannot be claimed successful")
        if objective_weights is not None:
            actual_prefix = [{key: step[key] for key in ("epoch", "group_index", "sample_ids")}
                             for step in summary["step_metrics"]]
            summary["loss_objective_prefix_match"] = actual_prefix == summary["loss_objective"]["registered_prefix"]
            if not summary["loss_objective_prefix_match"]:
                raise RuntimeError("actual training visits differ from registered loss-weight prefix")
        summary["final_adapter_parameter_sha256"] = _parameter_hash(model)
        summary["adapter_parameters_changed"] = summary["final_adapter_parameter_sha256"] != initial_hash
        if not summary["adapter_parameters_changed"]:
            raise RuntimeError("adapter parameters did not change despite optimizer attempts")
        if spatial_head is not None:
            _, spatial_artifacts = _spatial_modules(config.target_protocol)
            save_spatial_head_artifact = spatial_artifacts.save_spatial_head_artifact

            if sha256_file(config.spatial_tokenizer_json_path) != spatial_decoder.source_sha256:
                raise RuntimeError("spatial decoder metadata changed during training")
            final_head = save_spatial_head_artifact(spatial_head, output, role="final", bindings=spatial_bindings)
            summary["spatial_head_final"] = final_head
            summary["spatial_head_parameters_changed"] = final_head["parameter_sha256"] != summary["spatial_head_initial"]["parameter_sha256"]
            if not summary["spatial_head_parameters_changed"]:
                raise RuntimeError("spatial head parameters did not change despite applied updates")
            summary["artifacts"].update({"spatial_head_final": str(output / final_head["filename"]),
                                         "spatial_objective_samples": str(output / "spatial_objective_samples.jsonl")})
        if numeric_decoder is not None:
            if sha256_file(config.numeric_tokenizer_json_path) != numeric_decoder.source_sha256:
                raise RuntimeError("numeric decoder metadata changed during training")
            summary["artifacts"]["token_objective_samples"] = str(output / "token_objective_samples.jsonl")
        summary["source_hashes_after"] = _source_hashes()
        if summary["source_hashes_after"] != summary["source_hashes"]:
            raise RuntimeError("training source files changed during the run")
        summary["initial_adapter_after"] = initial_adapter_evidence(config)
        if summary["initial_adapter_after"] != summary["initial_adapter"]:
            raise RuntimeError("initial adapter weights changed during the run")
        model.save_pretrained(output / "adapter", safe_serialization=True, save_embedding_layers=False)
        processor.save_pretrained(output / "processor")
        summary["artifact_file_evidence"] = {
            "adapter": _saved_file_evidence(output / "adapter", adapter=True),
            "processor": _saved_file_evidence(output / "processor", adapter=False),
            "training_config": {"bytes": (output / "training_config.json").stat().st_size,
                                "sha256": sha256_file(output / "training_config.json")},
        }
        if spatial_head is not None:
            summary["artifact_file_evidence"]["spatial_head"] = {
                "bindings": spatial_bindings,
                "files": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)}
                          for name in ("spatial_head_initial.pt", "spatial_head_final.pt",
                                       "spatial_head_config.json", "spatial_head_manifest.json",
                                       "spatial_objective_samples.jsonl")},
                "initial_parameter_sha256": summary["spatial_head_initial"]["parameter_sha256"],
                "final_parameter_sha256": summary["spatial_head_final"]["parameter_sha256"],
            }
        if numeric_decoder is not None:
            diagnostics_path = output / "token_objective_samples.jsonl"
            summary["artifact_file_evidence"]["token_objective_samples"] = {
                "bytes": diagnostics_path.stat().st_size, "sha256": sha256_file(diagnostics_path),
                "completed_microbatches": summary["actual_token_objective"]["completed_microbatches"],
            }
        summary["artifacts"].update({"adapter": str(output / "adapter"),
                                "processor": str(output / "processor"),
                                "training_config": str(output / "training_config.json")})
        summary["status"] = "completed"
        summary["termination_reason"] = budget_stop or (
            "smoke_step_limit" if config.smoke else "max_steps"
            if step_limit is not None and summary["optimizer_steps"] >= step_limit else "epochs"
        )
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - persist every failure/cancellation
        summary["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if model is not None and summary["optimizer_steps"] > 0:
            try:
                if spatial_head is not None:
                    _, spatial_artifacts = _spatial_modules(config.target_protocol)
                    save_spatial_head_artifact = spatial_artifacts.save_spatial_head_artifact

                    partial_head = save_spatial_head_artifact(spatial_head, output, role="partial", bindings=spatial_bindings)
                    summary["spatial_head_partial"] = partial_head
                    summary["artifacts"]["spatial_head_partial"] = str(output / partial_head["filename"])
                model.save_pretrained(output / "adapter_partial", safe_serialization=True,
                                      save_embedding_layers=False)
                summary.setdefault("artifact_file_evidence", {})["partial_adapter"] = (
                    _saved_file_evidence(output / "adapter_partial", adapter=True)
                )
                summary["artifacts"]["partial_adapter"] = str(output / "adapter_partial")
                if processor is not None:
                    processor.save_pretrained(output / "processor_partial")
                    summary["artifact_file_evidence"]["partial_processor"] = (
                        _saved_file_evidence(output / "processor_partial", adapter=False)
                    )
                    summary["artifacts"]["partial_processor"] = str(output / "processor_partial")
                summary["final_adapter_parameter_sha256"] = _parameter_hash(model)
                summary["adapter_parameters_changed"] = (
                    summary["final_adapter_parameter_sha256"] != initial_hash
                )
            except (Exception, KeyboardInterrupt) as save_error:  # noqa: BLE001 - preserve primary error
                summary["partial_save_error"] = {"type": type(save_error).__name__,
                                                  "message": str(save_error)}
    finally:
        summary["finished_at"] = datetime.now(UTC).isoformat()
        if model is not None and config.device.startswith("cuda"):
            import torch
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(config.device)
            summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(config.device)
            summary["peak_cuda_scope"] = (
                "actual one-device training since the existing pre-loop reset"
                if summary["cuda_peak_stats_reset_before_loop"] else "process lifetime; pre-loop reset was not reached"
            )
        persist()
    return summary
