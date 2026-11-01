"""Synthetic annotated-image transport and complete assistant supervision guards."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_week5_visual_pointer as fixtures
import test_week5_visual_pointer_input as assets
from PIL import Image

from gui_agent.finetuning import training
from gui_agent.finetuning import visual_pointer_input as vi
from gui_agent.finetuning.evaluation import file_sha256
from gui_agent.finetuning.visual_pointer_prompts import (
    DEDUP_PROMPT_VERSION,
    build_visual_user_prompt,
)

font_asset = fixtures.font_asset


@pytest.fixture
def visual_asset(tmp_path, font_asset):
    asset = assets.Asset(tmp_path, font_asset)
    row = asset.row
    target = [{"action_type": "click", "candidate": 1, "dx": 2, "dy": -3},
              {"action_type": "type_text", "text": " Exact  A B\n中文  "},
              {"action_type": "hotkey", "keys": ["ctrl", "a"]},
              {"action_type": "wait", "duration": .25}]
    native = [{"action_type": "click", "x": 122, "y": 137}, *copy.deepcopy(target[1:])]
    snapshot_path = asset.asset_root / "label_side_snapshot.json"
    assets.write_json(snapshot_path, row["candidate_snapshot"])
    review_path = asset.asset_root / "label_side_review.json"
    assets.write_json(review_path, {"schema": "week5.pointer_semantic_review.v1", "status": "closed",
        "reviews": [{"sample_id": row["sample_id"], "action_index": 0, "decision": "accepted",
                     "action_type": "click", "current_subtask": row["actor_step"],
                     "instruction": row["instruction"], "target_point": {"x": 122, "y": 137},
                     "preferred_candidate_id": "native-A", "image_file_sha256": row["image_file_sha256"],
                     "snapshot_path": str(snapshot_path), "snapshot_file_sha256": file_sha256(snapshot_path)}]})
    row.update(group_id="original-group", source="synthetic-source", source_split="train",
               actions=native, candidate_target=target, training_admission=False,
               action_annotations=[{"retained_native_annotation": "SECRET_BBOX_LABEL"}],
               candidate_label_evidence={"review_path": str(review_path), "review_sha256": file_sha256(review_path)})
    return asset


def config_for(asset, **kwargs):
    return training.TrainingConfig(input_image_protocol="visual_pointer_v2", target_protocol="candidate_compact",
        prompt_variant="conditional_actor", visual_asset_manifest_path=str(asset.manifest),
        visual_asset_manifest_sha256=file_sha256(asset.manifest), visual_original_root=str(asset.original_root),
        visual_asset_root=str(asset.asset_root), **kwargs)


@pytest.fixture
def dedup_visual_asset(visual_asset):
    return assets.dedup_asset(visual_asset)


def test_dedup_caller_version_same_full_prefix_native_identity_and_complete_target(dedup_visual_asset):
    asset = dedup_visual_asset
    row = copy.deepcopy(asset.row)
    config = config_for(asset, visual_prompt_version=DEDUP_PROMPT_VERSION)
    assert training.TrainingConfig.from_dict(config.to_dict()) == config
    processor = SpyProcessor()
    encoded = training.encode_training_sample(processor, row, config)
    prefix, full = processor.messages
    assert full[0] == prefix[0]
    assert prefix[0]["content"][1]["text"] == build_visual_user_prompt(
        row["instruction"], row["actor_step"], row["candidate_snapshot"], visual_prompt_version=DEDUP_PROMPT_VERSION)
    assert json.loads(full[-1]["content"]) == row["candidate_target"]
    assert encoded["labels"].tolist() == [[-100] * 6 + [8, 9, 99]]
    assert [call["rgb"] for call in processor.calls] == [processor.calls[0]["rgb"]] * 2
    assert row == asset.row and "SECRET" not in json.dumps(prefix)
    assert "visual_pointer_prompts.py" in {Path(path).name for path in training._source_hashes()}


@pytest.mark.parametrize("new_row", [False, True])
def test_caller_version_cannot_select_other_registered_prompt(visual_asset, new_row):
    asset = assets.dedup_asset(visual_asset) if new_row else visual_asset
    wrong_config = config_for(asset, visual_prompt_version=None if new_row else DEDUP_PROMPT_VERSION)
    processor = SpyProcessor()
    with pytest.raises(ValueError, match="visual_prompt_version differ"):
        training.encode_training_sample(processor, asset.row, wrong_config)
    assert not processor.calls


def test_row_cannot_self_approve_dedup_without_new_manifest(visual_asset):
    row = dict(visual_asset.row, visual_prompt_version=DEDUP_PROMPT_VERSION)
    processor = SpyProcessor()
    with pytest.raises(ValueError):
        training.encode_training_sample(processor, row, config_for(visual_asset, visual_prompt_version=DEDUP_PROMPT_VERSION))
    assert not processor.calls


def test_approved_descriptor_version_must_match_caller_before_processor(dedup_visual_asset, monkeypatch):
    asset = dedup_visual_asset
    approved = vi.validate_visual_input(asset.row, registry=asset.registry)
    monkeypatch.setattr(vi, "validate_visual_input", lambda *a, **kw: replace(approved, visual_prompt_version=None))
    processor = SpyProcessor()
    with pytest.raises(ValueError, match="approved visual prompt version differs"):
        training.encode_training_sample(processor, asset.row, config_for(asset, visual_prompt_version=DEDUP_PROMPT_VERSION))
    assert not processor.calls


@pytest.mark.parametrize("version", [None, "", "unknown"])
def test_explicit_bad_row_version_fails_closed_legacy_is_missing(visual_asset, version):
    row = dict(visual_asset.row, visual_prompt_version=version)
    with pytest.raises(ValueError):
        training.build_messages(row, "conditional_actor", target_protocol="candidate_compact",
                                input_image_protocol="visual_pointer_v2", visual_registry=visual_asset.registry)


@pytest.mark.parametrize("version", ["", "unknown", []])
def test_config_rejects_unknown_visual_version_and_native_rejects_known(visual_asset, version):
    with pytest.raises(ValueError, match="visual_prompt_version"):
        config_for(visual_asset, visual_prompt_version=version)
    with pytest.raises(ValueError, match="native input protocol"):
        training.TrainingConfig(visual_prompt_version=DEDUP_PROMPT_VERSION)


def test_native_row_explicit_null_visual_version_is_refused_before_messages():
    with pytest.raises(ValueError, match="native input protocol refuses"):
        training.build_messages({"visual_prompt_version": None})


def test_dedup_inference_prefix_ignores_all_supervision(dedup_visual_asset):
    asset = dedup_visual_asset
    row = fixtures.VisibleOnly(copy.deepcopy(asset.row))
    row.update(sample_id=object(), actions=object(), candidate_target=object(), review=object(), candidate_label_evidence=object())
    result = training.build_messages(row, "conditional_actor", target_protocol="candidate_compact",
                                    input_image_protocol="visual_pointer_v2", visual_registry=asset.registry,
                                    visual_prompt_version=DEDUP_PROMPT_VERSION)
    assert result[0]["content"][1]["text"] == asset.prompt


def test_dedup_loader_uses_caller_version_preserves_original_supervision(dedup_visual_asset, tmp_path):
    asset = dedup_visual_asset
    asset.row["training_admission"] = True
    path = tmp_path / "synthetic_dedup_train.jsonl"
    path.write_text(json.dumps(asset.row), encoding="utf-8")
    kwargs = {"target_protocol": "candidate_compact", "input_image_protocol": "visual_pointer_v2", "visual_registry": asset.registry}
    with pytest.raises(ValueError, match="visual_prompt_version differ"):
        training.load_prepared_records(path, **kwargs)
    result = training.load_prepared_records(path, **kwargs, visual_prompt_version=DEDUP_PROMPT_VERSION)[0]
    for key in ("group_id", "image_path", "image_sha256", "image_file_sha256", "actions", "action_annotations", "candidate_target"):
        assert result[key] == asset.row[key]


def test_dedup_prompt_version_mutation_during_processor_is_refused(dedup_visual_asset):
    asset = dedup_visual_asset

    def change_version(call):
        if call == 1:
            asset.row.pop("visual_prompt_version")

    with pytest.raises(ValueError):
        training.encode_training_sample(SpyProcessor(on_call=change_version), asset.row,
                                        config_for(asset, visual_prompt_version=DEDUP_PROMPT_VERSION))


def test_cli_dedup_version_coverage_policy_and_hard_budgets(dedup_visual_asset, monkeypatch):
    path = Path(training.__file__).resolve().parents[3] / "scripts/week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("synthetic_dedup_budget_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = []
    monkeypatch.setattr(cli, "run_training", lambda cfg: seen.append(cfg) or {"status": "cancelled"})
    config = config_for(dedup_visual_asset, visual_prompt_version=DEDUP_PROMPT_VERSION,
                        sampling_policy="coverage_first_balanced_action_chunks", max_micro_steps=352,
                        max_optimizer_attempts=44)
    argv = ["--input-image-protocol", "visual_pointer_v2", "--target-protocol", "candidate_compact",
            "--prompt-variant", "conditional_actor", "--visual-prompt-version", DEDUP_PROMPT_VERSION,
            "--sampling-policy", config.sampling_policy, "--max-micro-steps", "352", "--max-optimizer-attempts", "44"]
    for key in training.VISUAL_REGISTRY_FIELDS:
        argv += ["--" + key.replace("_", "-"), getattr(config, key)]
    assert cli.main(argv) == 130 and seen == [config]


class SpyProcessor:
    def __init__(self, *, fault=None, on_call=None):
        self.messages, self.calls = [], []
        self.fault, self.on_call = fault, on_call
        ids = {"<|vision_start|>": 901, "<|image_pad|>": 902, "<|vision_end|>": 903,
               "<|video_pad|>": 904, "<|im_end|>": 99}
        self.tokenizer = SimpleNamespace(pad_token_id=0, unk_token_id=1000, convert_tokens_to_ids=lambda s: ids.get(s))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages.append(copy.deepcopy(messages))
        assert tokenize is False and add_generation_prompt == (len(messages) == 1)
        return "prefix" if add_generation_prompt else "full"

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        import torch

        image = images[0]
        self.calls.append({"text": text, "rgb": image.tobytes(), "size": image.size, "mode": image.mode,
                           "padding": padding, "return_tensors": return_tensors})
        assert padding is False and truncation is False and return_tensors == "pt"
        prefix = [1, 901, 902, 902, 903, 7]
        token_ids = prefix + ([8, 9, 99] if text == ["full"] else [])
        fingerprint = list(hashlib.sha256(image.tobytes()).digest())
        result = {"input_ids": torch.tensor([token_ids]), "attention_mask": torch.ones(1, len(token_ids)),
                  "pixel_values": torch.tensor([fingerprint], dtype=torch.float32),
                  "image_grid_thw": torch.tensor([[1, 2, 2]])}
        if self.fault == "zero_grid":
            result["image_grid_thw"] = torch.zeros(1, 3, dtype=torch.int64)
        elif self.fault == "float_grid":
            result["image_grid_thw"] = result["image_grid_thw"].float()
        if text == ["prefix"]:
            if self.fault in {"pixels", "grid"}:
                key = "pixel_values" if self.fault == "pixels" else "image_grid_thw"
                result[key] = result[key] + 1
            elif self.fault == "dtype":
                result["pixel_values"] = result["pixel_values"].double()
            elif self.fault == "not_tensor":
                result["image_grid_thw"] = [[1, 2, 2]]
            elif self.fault == "missing":
                result.pop("pixel_values")
            elif self.fault == "nan":
                result["pixel_values"][0, 0] = float("nan")
            elif self.fault == "prefix_ids":
                result["input_ids"][0, 0] = 555
            elif self.fault == "prefix_length":
                result["input_ids"] = torch.tensor([[1] * 1540])
        elif self.fault == "full_length":
            result["input_ids"] = torch.tensor([prefix + [8] * 1531 + [99]])
        elif self.fault == "eos":
            result["input_ids"][0, -1] = 98
        if self.fault == "image_mutation":
            image.putpixel((0, 0), (255, 0, 0))
        elif self.fault == "image_size":
            image._size = (image.height, image.width)
        elif self.fault == "image_mode":
            image._mode = "L"
        if self.on_call is not None:
            self.on_call(len(self.calls))
        return result


def test_explicit_config_roundtrip_defaults_and_trusted_registry(visual_asset):
    native = training.TrainingConfig()
    assert native.input_image_protocol == "native" and native.visual_registry() is None
    legacy = native.to_dict()
    for key in ("input_image_protocol", *training.VISUAL_REGISTRY_FIELDS):
        legacy.pop(key)
    assert training.TrainingConfig.from_dict(legacy) == native
    config = config_for(visual_asset)
    assert training.TrainingConfig.from_dict(config.to_dict()) == config
    registry = config.visual_registry()
    assert registry.manifest_sha256 == visual_asset.registry.manifest_sha256
    for key in ("manifest_path", "original_root", "asset_root"):
        assert Path(getattr(registry, key)) == Path(getattr(visual_asset.registry, key))


@pytest.mark.parametrize("changes", [
    {"input_image_protocol": None}, {"input_image_protocol": "visual"},
    {"target_protocol": "native"}, {"prompt_variant": "original"}, {"max_seq_length": 1537},
    {"visual_asset_manifest_path": None}, {"visual_asset_manifest_sha256": ""},
    {"visual_original_root": "relative"}, {"visual_asset_root": True},
])
def test_visual_config_cannot_silently_loosen_or_infer_registry(visual_asset, changes):
    data = config_for(visual_asset).to_dict() | changes
    with pytest.raises((TypeError, ValueError)):
        training.TrainingConfig.from_dict(data)
    with pytest.raises(ValueError, match="native input protocol"):
        training.TrainingConfig(visual_asset_manifest_path=str(visual_asset.manifest))


def test_encoder_validates_once_and_full_prefix_use_exact_annotated_rgb(visual_asset, monkeypatch):
    row, config = visual_asset.row, config_for(visual_asset)
    original = copy.deepcopy(row)
    source_bytes = Path(row["image_path"]).read_bytes()
    calls = []
    validate = vi.validate_visual_input
    monkeypatch.setattr(vi, "validate_visual_input", lambda *a, **kw: calls.append(True) or validate(*a, **kw))
    processor = SpyProcessor()
    result = training.encode_training_sample(processor, row, config)
    assert calls == [True]
    with Image.open(visual_asset.model) as model, Image.open(row["image_path"]) as native:
        annotated = model.convert("RGB").tobytes()
        assert annotated != native.convert("RGB").tobytes()
    assert [c["text"] for c in processor.calls] == [["full"], ["prefix"]]
    assert all(c["rgb"] == annotated and c["size"] == (1024, 768) and c["mode"] == "RGB" for c in processor.calls)
    prefix, full = processor.messages
    assert full[:1] == prefix
    assert prefix[0]["content"][1]["text"] == visual_asset.prompt
    assert json.loads(full[-1]["content"]) == row["candidate_target"]
    assert "SECRET" not in json.dumps(prefix) and "中文" not in json.dumps(prefix, ensure_ascii=False)
    assert result["labels"].tolist() == [[-100] * 6 + [8, 9, 99]]
    assert row == original and Path(row["image_path"]).read_bytes() == source_bytes
    # Admission False is intentionally allowed for the separately registered CPU gate.
    assert row["training_admission"] is False


def test_visual_inference_messages_never_access_supervision(visual_asset):
    row = fixtures.VisibleOnly(copy.deepcopy(visual_asset.row))
    row.update(sample_id=object(), actions=object(), candidate_target=object(), review=object(),
               candidate_label_evidence=object())
    result = training.build_messages(row, "conditional_actor", target_protocol="candidate_compact",
                                    input_image_protocol="visual_pointer_v2", visual_registry=visual_asset.registry)
    assert result[0]["content"][1]["text"] == visual_asset.prompt


@pytest.mark.parametrize("operation", ["messages", "encode", "load"])
def test_legacy_native_route_refuses_visual_rows_before_any_processor(visual_asset, tmp_path, operation):
    processor = SpyProcessor()
    with pytest.raises(ValueError, match="native input protocol refuses"):
        if operation == "messages":
            training.build_messages(visual_asset.row, "conditional_actor", target_protocol="candidate_compact")
        elif operation == "encode":
            training.encode_training_sample(processor, visual_asset.row, training.TrainingConfig())
        else:
            path = tmp_path / "synthetic_train.jsonl"
            path.write_text(json.dumps(visual_asset.row), encoding="utf-8")
            training.load_prepared_records(path)
    assert processor.calls == []


@pytest.mark.parametrize("fault", ["pixels", "grid", "dtype", "not_tensor", "missing", "nan", "prefix_ids", "eos", "image_mutation", "image_size", "image_mode", "zero_grid", "float_grid"])
def test_visual_tensor_mask_end_and_input_integrity_fail_closed(visual_asset, fault):
    processor = SpyProcessor(fault=fault)
    with pytest.raises(ValueError, match="visual|prefix do not match|end token"):
        training.encode_training_sample(processor, visual_asset.row, config_for(visual_asset))


@pytest.mark.parametrize("fault", ["full_length", "prefix_length"])
def test_full_and_prefix_1536_limits_have_no_truncation_or_action_trimming(visual_asset, fault):
    row = copy.deepcopy(visual_asset.row)
    processor = SpyProcessor(fault=fault)
    with pytest.raises(ValueError, match="tokens exceed max_seq_length=1536; no truncation performed"):
        training.encode_training_sample(processor, row, config_for(visual_asset))
    assert len(processor.calls) == 2
    assert row["candidate_target"] == visual_asset.row["candidate_target"]


@pytest.mark.parametrize("kind", ["metadata", "model", "native", "context", "target", "review"])
def test_mutation_during_processor_calls_rejected(visual_asset, kind):
    def mutate(call):
        if call != 2:
            return
        if kind in {"metadata", "model", "native"}:
            path = visual_asset.sidecar if kind == "metadata" else visual_asset.model if kind == "model" else Path(visual_asset.row["image_path"])
            path.write_bytes(path.read_bytes() + b"changed during synthetic processor")
        elif kind == "context":
            visual_asset.row["actor_step"] = "a changed request"
        elif kind == "target":
            visual_asset.row["candidate_target"][1]["text"] = "truncated replacement"
        else:
            path = Path(visual_asset.row["candidate_label_evidence"]["review_path"])
            path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        training.encode_training_sample(SpyProcessor(on_call=mutate), visual_asset.row, config_for(visual_asset))


def test_whole_native_target_mismatch_refused_before_processor(visual_asset):
    visual_asset.row["candidate_target"].pop()
    processor = SpyProcessor()
    with pytest.raises(ValueError, match="complete native action sequence"):
        training.encode_training_sample(processor, visual_asset.row, config_for(visual_asset))
    assert not processor.calls


def test_visual_loader_requires_separate_admission_preserves_native_group_and_whole_labels(visual_asset, tmp_path):
    path = tmp_path / "synthetic_admitted_train.jsonl"
    path.write_text(json.dumps(visual_asset.row), encoding="utf-8")
    kwargs = {"target_protocol": "candidate_compact", "input_image_protocol": "visual_pointer_v2", "visual_registry": visual_asset.registry}
    with pytest.raises(ValueError, match="separately admitted export"):
        training.load_prepared_records(path, **kwargs)
    visual_asset.row["training_admission"] = True
    path.write_text(json.dumps(visual_asset.row), encoding="utf-8")
    loaded = training.load_prepared_records(path, **kwargs)[0]
    for key in ("sample_id", "group_id", "image_path", "image_sha256", "image_file_sha256", "actions", "action_annotations", "candidate_target"):
        assert loaded[key] == visual_asset.row[key]
    assert loaded["image_sha256"] != loaded["model_native_rgb_sha256"]
    assert loaded["image_path"] != loaded["model_image_path"]
    other_view = dict(loaded, sample_id="different-display-view", group_id="different-claim", model_native_rgb_sha256="0" * 64)
    with pytest.raises(ValueError, match="image_sha256"):
        training.ensure_separate_validation([loaded], [other_view])


def test_run_training_passes_caller_registry_and_refuses_before_model_loading(visual_asset, tmp_path, monkeypatch):
    path = tmp_path / "synthetic_unadmitted.jsonl"
    path.write_text(json.dumps(visual_asset.row), encoding="utf-8")
    monkeypatch.setattr(training, "load_model_and_processor", lambda *a, **kw: pytest.fail("real model loader reached"))
    config = replace(config_for(visual_asset), train_jsonl=str(path), output_dir=str(tmp_path / "synthetic_refusal"))
    summary = training.run_training(config)
    assert summary["status"] == "failed" and summary["optimizer_steps"] == summary["micro_steps"] == 0
    assert "separately admitted export" in summary["error"]["message"]
    assert summary["config"]["input_image_protocol"] == "visual_pointer_v2"
    assert summary["config"]["visual_asset_manifest_sha256"] == config.visual_asset_manifest_sha256
    assert str(Path(training.__file__).resolve().with_name("visual_pointer_input.py")) in summary["source_hashes"]


def test_cli_explicit_visual_protocol_and_registry_flags(tmp_path, visual_asset, monkeypatch):
    path = Path(training.__file__).resolve().parents[3] / "scripts/week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("synthetic_visual_train_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = []
    monkeypatch.setattr(cli, "run_training", lambda cfg: seen.append(cfg) or {"status": "cancelled"})
    config = config_for(visual_asset)
    argv = ["--input-image-protocol", "visual_pointer_v2", "--target-protocol", "candidate_compact", "--prompt-variant", "conditional_actor"]
    for key in training.VISUAL_REGISTRY_FIELDS:
        argv += ["--" + key.replace("_", "-"), getattr(config, key)]
    assert cli.main(argv) == 130 and seen[0] == config
    assert cli.main(["--input-image-protocol", "visual_pointer_v2"]) == 2
