"""Meaningful CPU causal/gradient tests for the private combined grid trainer."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import test_week5_native_spatial_head as legacy_test

from gui_agent.finetuning import native_spatial_grid_head as core
from gui_agent.finetuning import native_spatial_head as legacy
from gui_agent.finetuning import numeric_value_loss, training
from gui_agent.finetuning import visual_pointer_input as visual


def plan(actions=None):
    metadata = {"native_size": [256, 256], "canvas_size": [256, 256],
                "overview_bbox": [0, 0, 256, 256], "overview_rgb_byteexact": True, "id_table": []}
    actions = actions if actions is not None else [{"action_type": "click", "x": 30, "y": 220}]
    target = core.semantic_target(actions, native_width=256, native_height=256)
    prefix = (1, 2, 499, 499, 499, 499, 3)
    full = prefix + legacy_test.byte_ids(target["semantic_json"]) + (500, 20)
    return core.training_plan(full_ids=full, prefix_ids=prefix,
        labels=(-100,) * len(prefix) + full[len(prefix):], target=target,
        decoder=legacy_test.fake_decoder(), metadata=metadata, grid_thw=[1, 4, 4], image_token_id=499)


def test_joint_gradient_reaches_every_spatial_branch_and_causal_model(monkeypatch):
    import torch
    model, head = legacy_test.toy_model(torch), core.make_spatial_head()
    selected_plan = plan()
    ce_calls = []
    original = torch.nn.functional.cross_entropy

    def count_ce(*args, **kwargs):
        ce_calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", count_ce)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = core.forward_objective(model, head, legacy_test.tensor_batch(torch, selected_plan), selected_plan)
    spatial_loss = (result["point_loss"] + 0.5 * result["grid_aux_ce"]
                    + 0.5 * result["grid_refine_loss"])
    model_gradient = torch.autograd.grad(spatial_loss, model.embedding.weight, retain_graph=True)[0]
    assert torch.isfinite(model_gradient).all() and model_gradient.abs().sum() > 0
    result["total_loss"].backward()
    assert result["prediction_unit"].dtype == torch.float32
    assert model.forward_calls == len(ce_calls) == 1
    assert result["manual_CE_calls"] == result["model_forward_calls"] == 1
    assert result["default_model_CE_requested"] is False
    assert result["grid_refine_diagnostics"]["additional_head_forward_calls"] == 0
    torch.testing.assert_close(result["total_loss"], result["semantic_ce"] + spatial_loss)
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert selected_plan.image_positions == (2, 3, 4, 5)
    assert max(selected_plan.image_positions) < selected_plan.prefix_length
    assert not torch.cuda.is_initialized()


def test_target_change_cannot_change_forward_prediction():
    import torch
    model, head = legacy_test.toy_model(torch), core.make_spatial_head()
    a = plan([{"action_type": "click", "x": 30, "y": 220}])
    b = plan([{"action_type": "click", "x": 220, "y": 30}])
    assert a.full_ids == b.full_ids and a.features == b.features
    first = core.forward_objective(model, head, legacy_test.tensor_batch(torch, a), a)
    second = core.forward_objective(model, head, legacy_test.tensor_batch(torch, b), b)
    assert torch.equal(first["prediction_unit"], second["prediction_unit"])
    assert not torch.equal(first["grid_aux_ce"], second["grid_aux_ce"])
    assert not torch.equal(first["grid_refine_loss"], second["grid_refine_loss"])


def test_nonmouse_microbatch_never_calls_spatial_head(monkeypatch):
    import torch
    model, head = legacy_test.toy_model(torch), core.make_spatial_head()
    selected_plan = plan([{"action_type": "type_text", "text": "native_head literal"}])

    def forbidden(*args, **kwargs):
        raise AssertionError("no point implies no head call")

    monkeypatch.setattr(core, "prepare_key_cache", forbidden)
    monkeypatch.setattr(core, "predict_with_cache", forbidden)
    result = core.forward_objective(model, head, legacy_test.tensor_batch(torch, selected_plan), selected_plan)
    assert result["point_count"] == 0
    assert result["point_loss"].item() == result["grid_aux_ce"].item() == result["grid_refine_loss"].item() == 0
    assert result["grid_refine_loss"].requires_grad
    assert result["grid_refine_diagnostics"]["zero_markers"] is True
    result["total_loss"].backward()
    assert all(parameter.grad is None for parameter in head.parameters())


def test_wrong_or_future_image_placeholder_fails_closed():
    selected_plan = plan()
    values = vars(selected_plan)
    semantic_fields = {key: values[key] for key in ("full_ids", "labels")}
    target = core.semantic_target([{"action_type": "click", "x": 30, "y": 220}], native_width=256, native_height=256)
    metadata = {"native_size": [256, 256], "canvas_size": [256, 256],
                "overview_bbox": [0, 0, 256, 256], "overview_rgb_byteexact": True, "id_table": []}
    with pytest.raises(ValueError, match="prefix raster"):
        core.training_plan(**semantic_fields, prefix_ids=selected_plan.full_ids[:selected_plan.prefix_length],
            target=target, decoder=legacy_test.fake_decoder(), metadata=metadata, grid_thw=[1, 4, 4], image_token_id=3)


def grid_config(tmp_path, **changes):
    values = {"target_protocol": core.TARGET_PROTOCOL, "prompt_variant": "conditional_actor",
              "input_image_protocol": "visual_pointer_gallery_v3", "visual_prompt_version": "week5.visual-pointer-prompt.gallery-native-spatial.v1",
              "visual_asset_manifest_path": str(tmp_path / "SYNTHETIC_manifest.json"), "visual_asset_manifest_sha256": "a"*64,
              "visual_original_root": str(tmp_path), "visual_asset_root": str(tmp_path),
              "spatial_tokenizer_json_path": str(tmp_path / "SYNTHETIC_tokenizer.json"), "spatial_tokenizer_json_sha256": "b"*64,
              "spatial_head_learning_rate": 1e-3, "learning_rate": 1e-4, "epochs": 2,
              "max_steps": 144, "max_optimizer_attempts": 144, "max_micro_steps": 1144,
              "sampling_policy": "coverage_first_balanced_action_chunks", "loss_strategy": "registered_stratum_balanced"}
    values.update(changes)
    return training.TrainingConfig(**values)


def test_grid_config_roundtrip_and_target_wrapper_preserve_semantic_json(tmp_path):
    cfg = grid_config(tmp_path)
    assert training.TrainingConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    actions = [{"action_type": "click", "x": 0, "y": 79}, {"action_type": "type_text", "text": "native_head 中文 019"}]
    old = legacy.semantic_target(actions, native_width=100, native_height=80)
    new = core.semantic_target(actions, native_width=100, native_height=80)
    assert new == {**old, "target_protocol": core.TARGET_PROTOCOL}
    assert new["semantic_json"] == old["semantic_json"] and new["point_supervision"] == old["point_supervision"]
    selected_core, selected_artifacts = training._spatial_modules(cfg.target_protocol)
    assert selected_core is core and selected_artifacts.SCHEMA == "week5.native-spatial-grid-artifacts.v1"


def test_grid_profile_refuses_drift_while_legacy_defaults_remain_valid(tmp_path):
    for change in ({"max_seq_length": 1535}, {"max_pixels": 301055}, {"processor_use_fast": False},
                   {"learning_rate": 2e-4}, {"spatial_head_learning_rate": 2e-3}, {"epochs": 3},
                   {"max_micro_steps": 1143}, {"initial_adapter_path": "old"}, {"validation_jsonl": "validation"},
                   {"visual_prompt_version": "week5.visual-pointer-prompt.gallery.v4"}):
        with pytest.raises(ValueError):
            grid_config(tmp_path, **change)
    native = training.TrainingConfig()
    compact = training.TrainingConfig(target_protocol="candidate_compact", prompt_variant="conditional_actor")
    old_spatial = grid_config(tmp_path, target_protocol=legacy.TARGET_PROTOCOL)
    assert (native.target_protocol, native.input_image_protocol, native.prompt_variant) == ("native", "native", "original")
    assert native.spatial_head_learning_rate is None and native.spatial_tokenizer_json_path is None
    assert training.TrainingConfig.from_dict(native.to_dict()) == native
    assert training.TrainingConfig.from_dict(compact.to_dict()) == compact
    assert training._spatial_modules(old_spatial.target_protocol)[0] is legacy


def synthetic_encoder(tmp_path, monkeypatch):
    import torch
    cfg = grid_config(tmp_path, base_model_path=str(tmp_path))
    selected_plan = plan()
    metadata = {"native_size": [256, 256], "canvas_size": [256, 256], "overview_bbox": [0, 0, 256, 256],
                "overview_rgb_byteexact": True, "id_table": []}
    sidecar = tmp_path / "SYNTHETIC_geometry.json"
    sidecar.write_text(json.dumps(metadata), encoding="utf8")
    (tmp_path / "config.json").write_text(json.dumps({"image_token_id": 499, "vision_config": {"patch_size": 14, "spatial_merge_size": 2}}), encoding="utf8")
    actions = [{"action_type": "click", "x": 30, "y": 220}]
    sample = {"sample_id": "SYNTHETIC_encoder", "actions": actions, "image_width": 256, "image_height": 256,
              "native_spatial_target": core.semantic_target(actions, native_width=256, native_height=256),
              "visual_metadata_path": str(sidecar), "visual_metadata_file_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    image = SimpleNamespace(mode="RGB", size=(256, 256), tobytes=lambda: b"synthetic CPU raster placeholder")
    checks = []
    raster = SimpleNamespace(prompt="SYNTHETIC input only", visual_prompt_version=cfg.visual_prompt_version, model_size=image.size,
                             rgb_bytes=image.tobytes(), to_pil=lambda: image, assert_current=lambda: checks.append(1))
    sentinel_registry = object()
    monkeypatch.setattr(training, "visual_registry", lambda config: sentinel_registry)
    def input_protocol(sample, protocol, registry, version):
        assert protocol == cfg.input_image_protocol and registry is sentinel_registry and version == cfg.visual_prompt_version
    monkeypatch.setattr(training, "_input_protocol", input_protocol)
    monkeypatch.setattr(visual, "validate_visual_input", lambda sample, registry: raster)
    monkeypatch.setattr(numeric_value_loss, "load_bound_decoder", lambda *args: legacy_test.fake_decoder())
    class Processor:
        def __init__(self):
            ids = {"<|image_pad|>": 499, "<|video_pad|>": 498, "<|vision_start|>": 497, "<|vision_end|>": 496, "<|im_end|>": 500}
            self.tokenizer = SimpleNamespace(convert_tokens_to_ids=ids.get, pad_token_id=0, unk_token_id=None)
            self.calls, self.templates = [], []
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and messages[0]["content"][1]["text"] == raster.prompt
            self.templates.append(messages)
            if len(messages) == 1:
                assert add_generation_prompt is True
                return "SYNTHETIC_PREFIX"
            assert add_generation_prompt is False and messages[1]["content"] == sample["native_spatial_target"]["semantic_json"]
            return "SYNTHETIC_FULL"
        def __call__(self, *, text, images, padding, truncation, return_tensors):
            assert padding is False and truncation is False and return_tensors == "pt" and images == [image]
            self.calls.append(text[0])
            ids = selected_plan.full_ids if text == ["SYNTHETIC_FULL"] else selected_plan.full_ids[:selected_plan.prefix_length]
            return {"input_ids": torch.tensor([ids], dtype=torch.long), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                    "pixel_values": torch.full((16, 1176), .5, dtype=torch.float32), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    return cfg, sample, Processor(), checks


def test_encoder_consumes_bound_metadata_and_exact_full_prefix_without_truncation(tmp_path, monkeypatch):
    import torch
    cfg, sample, processor, checks = synthetic_encoder(tmp_path, monkeypatch)
    encoded = training.encode_training_sample(processor, sample, cfg)
    encoded_plan = encoded["_native_spatial_plan"]
    assert processor.calls == ["SYNTHETIC_FULL", "SYNTHETIC_PREFIX"] and len(processor.templates) == 2 and len(checks) >= 5
    assert encoded_plan.image_positions == (2, 3, 4, 5) and encoded_plan.features.total_token_count == 4
    assert encoded_plan.full_ids == tuple(encoded["input_ids"][0].tolist())
    assert encoded["labels"][0, :encoded_plan.prefix_length].tolist() == [-100]*encoded_plan.prefix_length
    assert 500 in encoded["labels"][0].tolist() and encoded["labels"][0, -1] == 20
    assert not torch.cuda.is_initialized()


def test_encoder_refuses_changed_registered_geometry_after_two_exact_calls(tmp_path, monkeypatch):
    cfg, sample, processor, _ = synthetic_encoder(tmp_path, monkeypatch)
    sample["visual_metadata_file_sha256"] = "0"*64
    with pytest.raises(ValueError, match="metadata changed"):
        training.encode_training_sample(processor, sample, cfg)
    assert processor.calls == ["SYNTHETIC_FULL", "SYNTHETIC_PREFIX"]
