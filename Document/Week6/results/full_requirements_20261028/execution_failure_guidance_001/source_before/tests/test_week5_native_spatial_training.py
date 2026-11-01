"""Synthetic semantic Gallery and trainer wiring; no actual model or dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_week5_gallery_protocol as galleries
import test_week5_visual_pointer_input as asset_helpers

from gui_agent.finetuning import native_spatial_artifacts as artifacts
from gui_agent.finetuning import native_spatial_head as spatial
from gui_agent.finetuning import numeric_value_loss, training
from gui_agent.finetuning import visual_pointer_input as visual
from gui_agent.finetuning.visual_pointer_prompts import (
    GALLERY_SPATIAL_HEADER,
    GALLERY_SPATIAL_PROMPT_VERSION,
    build_visual_user_prompt,
)

visual_asset = galleries.visual_asset
font_asset = galleries.font_asset
gallery_asset = galleries.gallery_asset


def spatial_config(tmp_path, **changes):
    values = {"prompt_variant": "conditional_actor", "target_protocol": "native_spatial_head",
              "input_image_protocol": "visual_pointer_gallery_v3", "visual_prompt_version": GALLERY_SPATIAL_PROMPT_VERSION,
              "visual_asset_manifest_path": str(tmp_path / "manifest.json"), "visual_asset_manifest_sha256": "a" * 64,
              "visual_original_root": str(tmp_path), "visual_asset_root": str(tmp_path),
              "spatial_tokenizer_json_path": str(tmp_path / "tokenizer.json"), "spatial_tokenizer_json_sha256": "b" * 64,
              "spatial_head_learning_rate": 1e-3, "learning_rate": 1e-4, "epochs": 2,
              "max_steps": 144, "max_optimizer_attempts": 144, "max_micro_steps": 1144,
              "sampling_policy": "coverage_first_balanced_action_chunks", "loss_strategy": "registered_stratum_balanced"}
    values.update(changes)
    return training.TrainingConfig(**values)


@pytest.mark.parametrize("change", [
    {"learning_rate": 2e-4}, {"spatial_head_learning_rate": 1e-4}, {"initial_adapter_path": "old"},
    {"validation_jsonl": "validation.jsonl"}, {"loss_strategy": "uniform"}, {"epochs": 3},
    {"visual_prompt_version": "week5.visual-pointer-prompt.gallery.v4"},
])
def test_spatial_config_refuses_unregistered_changes(tmp_path, change):
    with pytest.raises(ValueError):
        spatial_config(tmp_path, **change)
    assert training.TrainingConfig.from_dict(spatial_config(tmp_path).to_dict()) == spatial_config(tmp_path)


def rebind_semantic_gallery(asset):
    asset.row["visual_prompt_version"] = GALLERY_SPATIAL_PROMPT_VERSION
    prompt = build_visual_user_prompt(asset.row["instruction"], asset.row["actor_step"], asset.row["candidate_snapshot"],
                                       visual_prompt_version=GALLERY_SPATIAL_PROMPT_VERSION)
    asset.metadata.update(visual_prompt_version=GALLERY_SPATIAL_PROMPT_VERSION,
                          prompt_sha256=hashlib.sha256(prompt.encode("utf8")).hexdigest())
    asset_helpers.write_json(asset.sidecar, asset.metadata)
    asset.row["visual_metadata_file_sha256"] = training.sha256_file(asset.sidecar)
    entry = {**visual.visual_input_binding(asset.row), "font_sha256": asset.metadata["font_sha256"],
             "config_sha256": asset.metadata["config_sha256"]}
    asset_helpers.write_json(asset.manifest, {"schema": visual.GALLERY_MANIFEST_SCHEMA, "status": "approved",
        "input_image_protocol": visual.GALLERY_PROTOCOL, "visual_prompt_version": GALLERY_SPATIAL_PROMPT_VERSION,
        "entries": [entry]})
    asset.registry = visual.VisualAssetRegistry(asset.manifest, training.sha256_file(asset.manifest),
                                               asset.original_root, asset.asset_root)
    asset.row["native_spatial_target"] = spatial.semantic_target(asset.row["actions"],
        native_width=asset.row["image_width"], native_height=asset.row["image_height"])
    return prompt


def test_semantic_gallery_requires_new_registry_and_preserves_exact_png(gallery_asset):
    asset = gallery_asset
    before_png = asset.model.read_bytes()
    old_registry = asset.registry
    prompt = rebind_semantic_gallery(asset)
    with pytest.raises(ValueError):
        visual.validate_visual_input(asset.row, registry=old_registry)
    descriptor = visual.validate_visual_input(asset.row, registry=asset.registry)
    assert descriptor.prompt == prompt and asset.model.read_bytes() == before_png
    messages = training.build_messages(asset.row, "conditional_actor", include_target=True,
        target_protocol="native_spatial_head", input_image_protocol=visual.GALLERY_PROTOCOL,
        visual_registry=asset.registry, visual_prompt_version=GALLERY_SPATIAL_PROMPT_VERSION)
    assert messages[0]["content"][1]["text"] == prompt
    assert messages[1]["content"] == asset.row["native_spatial_target"]["semantic_json"]
    assert "candidate dx dy" not in prompt and 'point="native_head"' in prompt
    asset.row["native_spatial_target"]["point_supervision"][0]["x"] += 1
    with pytest.raises(ValueError, match="sidecar"):
        training.build_messages(asset.row, "conditional_actor", include_target=True,
            target_protocol="native_spatial_head", input_image_protocol=visual.GALLERY_PROTOCOL,
            visual_registry=asset.registry, visual_prompt_version=GALLERY_SPATIAL_PROMPT_VERSION)


def test_synthetic_trainer_owns_two_groups_and_saves_both_exact_heads(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch

    # CPU AdamW also probes accelerator graph capture. Keep this synthetic test
    # on CPU so later preparation tests do not inherit a CUDA context.
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    class SyntheticQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = torch.nn.Parameter(torch.linspace(-1, 1, 2048))
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.norm = torch.nn.LayerNorm(2048)
            self.model.language_model.norm.requires_grad_(False)
            self.visual = torch.nn.Linear(1, 1)
            self.visual.requires_grad_(False)
            self.calls = 0

        def forward(self, input_ids, **kwargs):
            assert "labels" not in kwargs and kwargs["use_cache"] is False
            self.calls += 1
            states = self.lora[None, None].expand(1, input_ids.shape[1], 2048)
            normed = self.model.language_model.norm(states)
            return SimpleNamespace(logits=normed[..., :100])

        def save_pretrained(self, path, **kwargs):
            path = Path(path)
            path.mkdir()
            (path / "adapter_config.json").write_text("{}", encoding="utf8")
            (path / "adapter_model.safetensors").write_bytes(self.lora.detach().numpy().tobytes())

    class SyntheticProcessor:
        def save_pretrained(self, path):
            path = Path(path)
            path.mkdir()
            for name in ("preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "chat_template.jinja"):
                (path / name).write_text("{}", encoding="utf8")

    rows = [{"sample_id": str(index), "group_id": str(index), "source": "synthetic", "source_split": "train",
             "image_path": str(tmp_path / "unused.png"), "image_sha256": "c" * 64, "image_file_sha256": "d" * 64,
             "image_width": 40, "image_height": 30,
             "actions": [{"action_type": "click", "x": 4, "y": 5}] + ([{"action_type": "key_press", "key": "enter"}] if index >= 4 else [])}
            for index in range(8)]
    source = tmp_path / "synthetic_rows.jsonl"
    source.write_text(json.dumps(rows), encoding="utf8")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl", "hidden_size": 2048,
        "architectures": ["Qwen2_5_VLForConditionalGeneration"]}), encoding="utf8")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"synthetic decoder bytes")
    tokenizer_sha = training.sha256_file(tokenizer)
    config = spatial_config(tmp_path, base_model_path=str(base), train_jsonl=str(source), output_dir=str(tmp_path / "out"),
                            device="cpu", load_in_4bit=False, compute_dtype="float32", smoke=True,
                            spatial_tokenizer_json_sha256=tokenizer_sha)
    model, processor = SyntheticQwen(), SyntheticProcessor()
    monkeypatch.setattr(numeric_value_loss, "load_bound_decoder", lambda *_: SimpleNamespace(source_sha256=tokenizer_sha, eos_id=99))
    monkeypatch.setattr(training, "load_prepared_records", lambda *_, **__: rows)
    monkeypatch.setattr(training, "load_model_and_processor", lambda *_, **__: (model, processor))

    def encode(processor, row, config):
        target = spatial.semantic_target(row["actions"], native_width=40, native_height=30)
        decoder = SimpleNamespace(pieces={7: target["semantic_json"].encode("utf8"), 99: b"<|im_end|>", 8: b"\n"}, eos_id=99)
        ids, labels = [1, 7, 99, 8], [-100, 7, 99, 8]
        plan = spatial.training_plan(full_ids=ids, prefix_ids=[1], labels=labels, target=target, decoder=decoder)
        return {"input_ids": torch.tensor([ids]), "labels": torch.tensor([labels]), "_native_spatial_plan": plan}

    monkeypatch.setattr(training, "encode_training_sample", encode)
    summary = training.run_training(config)
    assert summary["status"] == "completed", summary.get("error")
    assert model.calls == summary["micro_steps"] == 8 and summary["optimizer_steps"] == 1
    assert [group["learning_rate"] for group in summary["optimizer_parameter_groups"]] == [1e-4, 1e-3]
    assert summary["spatial_head_parameters_changed"] and summary["adapter_parameters_changed"]
    step = summary["step_metrics"][0]
    assert step["head_preclip_norm"] > 0 and step["lora_preclip_norm"] > 0
    assert step["additional_backward_calls"] == 0 and step["mean_total_loss"] == pytest.approx(step["mean_semantic_ce"] + step["mean_point_loss"])
    assert summary["spatial_objective"]["completed_microbatches"] == 8
    ledger = summary["artifact_file_evidence"]["spatial_head"]
    assert ledger["bindings"]["prompt_sha256"] == hashlib.sha256(GALLERY_SPATIAL_HEADER.encode("utf8")).hexdigest()
    assert ledger["bindings"]["training_source_sha256"] == training.sha256_file(training.__file__)
    head, identity = artifacts.load_spatial_head_artifact(config.output_dir, role="initial",
        expected_bindings=ledger["bindings"], expected_manifest_sha256=ledger["files"]["spatial_head_manifest.json"]["sha256"])
    assert identity["parameter_sha256"] == summary["spatial_head_initial"]["parameter_sha256"]
    assert head[0].weight.dtype == torch.float32
    from gui_agent.finetuning.evaluation import HFConfig
    from gui_agent.finetuning.native_spatial_inference import inspect_head

    hf_config = HFConfig(config.to_dict(), target_version=spatial.TARGET_VERSION,
        input_image_protocol=config.input_image_protocol, visual_prompt_version=config.visual_prompt_version,
        visual_asset_manifest_path=config.visual_asset_manifest_path,
        visual_asset_manifest_sha256=config.visual_asset_manifest_sha256,
        visual_original_root=config.visual_original_root, visual_asset_root=config.visual_asset_root,
        spatial_head_artifact_dir=config.output_dir, spatial_head_role="initial",
        spatial_head_manifest_sha256=ledger["files"]["spatial_head_manifest.json"]["sha256"],
        spatial_expected_bindings=ledger["bindings"])
    assert inspect_head(hf_config)["parameter_sha256"] == summary["spatial_head_initial"]["parameter_sha256"]
    assert not torch.cuda.is_initialized()
