"""Offline regression only: tiny CPU tensors and explicit synthetic processors.

These tests never load a real model, initialize CUDA, download or train on the
desktop. The synthetic lifecycle tests check accounting and persisted failures.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gui_agent.finetuning import training
from gui_agent.finetuning.training import TrainingConfig


def sample(tmp_path: Path, name: str = "a", *, color: str = "red") -> dict:
    image = tmp_path / f"{name}.png"
    Image.new("RGB", (20, 10), color).save(image)
    return {"sample_id": name, "group_id": f"group-{name}", "image_path": str(image),
            "instruction": "Click the visible control", "image_width": 20, "image_height": 10,
            "source": "synthetic-test-fixture", "source_split": "train",
            "actions": [{"action_type": "click", "x": 4, "y": 5}],
            "action_annotations": [{"target_bbox": [1, 2, 3, 4]}]}


def jsonl(tmp_path: Path, rows: list[dict], name: str = "train") -> Path:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_config_round_trip_and_unknown_fields(tmp_path):
    config = TrainingConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    assert TrainingConfig.from_json(path) == config
    with pytest.raises(ValueError, match="unknown training config fields: surprise"):
        TrainingConfig.from_dict({"surprise": 1})


@pytest.mark.parametrize("changes", [
    {"gradient_accumulation_steps": 0}, {"max_steps": 0}, {"epochs": True},
    {"micro_batch_size": 2}, {"min_pixels": 200, "max_pixels": 100},
    {"compute_dtype": "half"}, {"prompt_variant": "improved"}, {"load_in_4bit": 1},
    {"learning_rate": float("nan")}, {"weight_decay": -1}, {"lora_dropout": 1},
    {"seed": -1}, {"target_modules": "q_proj"},
])
def test_invalid_config_is_explicit(changes):
    with pytest.raises(ValueError):
        TrainingConfig.from_dict(changes)


def test_user_messages_exclude_all_labels_and_provenance(tmp_path):
    row = sample(tmp_path)
    row["actions"] = [{"action_type": "type_text", "text": "SECRET_ACTION_LABEL"}]
    row["action_annotations"] = [{"target_bbox": "SECRET_BBOX"}]
    row["source"] = "SECRET_SOURCE"
    row["sample_id"] = "SECRET_ID"
    messages = training.build_messages(row)
    assert messages[0]["content"][0] == {"type": "image"}
    rendered = json.dumps(messages)
    assert "SECRET" not in rendered
    assert str(tmp_path) not in rendered
    full = training.build_messages(row, include_target=True)
    assert full[0] == messages[0]
    assert json.loads(full[1]["content"]) == row["actions"]
    assert isinstance(json.loads(full[1]["content"]), list)


def test_improved_inference_uses_same_bare_array_contract(tmp_path):
    row = sample(tmp_path)
    original = training.build_messages(row)
    improved = training.build_messages(row, "improved")
    assert original != improved
    for messages in (original, improved):
        assert '"action_type"' in messages[0]["content"][1]["text"]
        assert "Output a JSON array" in messages[0]["content"][1]["text"]


def test_labels_mask_expanded_image_prompt_and_pad_but_keep_assistant_end():
    prompt = [1, 901, 902, 902, 902, 903, 7]
    labels = training.assistant_labels(prompt + [8, 9, 99, 0], prompt,
                                      visual_ids={901, 902, 903}, pad_id=0)
    assert labels == [-100] * len(prompt) + [8, 9, 99, -100]


@pytest.mark.parametrize("full,prompt", [([1, 2], [1, 3]), ([1, 2], [1, 2]), ([1], [1, 2])])
def test_labels_refuse_nonmatching_or_empty_target(full, prompt):
    with pytest.raises(ValueError, match="prefix do not match"):
        training.assistant_labels(full, prompt, visual_ids=set(), pad_id=0)


def test_labels_refuse_fully_masked_assistant():
    with pytest.raises(ValueError, match="no assistant tokens"):
        training.assistant_labels([1, 902, 0], [1], visual_ids={902}, pad_id=0)


class SyntheticProcessor:
    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda token: {
            "<|vision_start|>": 901, "<|image_pad|>": 902,
            "<|vision_end|>": 903, "<|video_pad|>": 904,
        }[token])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        if len(messages) == 1:
            assert add_generation_prompt is True
            return "prefix"
        assert add_generation_prompt is False
        assert isinstance(json.loads(messages[1]["content"]), list)
        return "full"

    def __call__(self, *, text, images, padding, return_tensors):
        import torch

        self.calls.append((text, images[0].copy(), padding, return_tensors))
        ids = [1, 901, 902, 902, 902, 903, 7]
        if text == ["full"]:
            ids += [8, 9, 99]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids)),
                "pixel_values": torch.ones(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir()
        for name in ("preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
                     "tokenizer.json", "chat_template.jinja"):
            (path / name).write_text("{}", encoding="utf-8")


def test_encoding_uses_actual_image_for_full_and_prefix(tmp_path):
    processor = SyntheticProcessor()
    encoded = training.encode_training_sample(processor, sample(tmp_path), TrainingConfig())
    assert encoded["labels"].tolist() == [[-100] * 7 + [8, 9, 99]]
    assert encoded["pixel_values"].shape == (4, 3)
    assert len(processor.calls) == 2
    for text, image, padding, tensors in processor.calls:
        assert text in (["full"], ["prefix"])
        assert image.mode == "RGB" and image.size == (20, 10)
        assert padding is False and tensors == "pt"


def test_encoding_rejects_overlength_without_silent_truncation(tmp_path):
    with pytest.raises(ValueError, match="no truncation performed"):
        training.encode_training_sample(SyntheticProcessor(), sample(tmp_path),
                                        TrainingConfig(max_seq_length=9))


def test_prepared_rgb_identity_is_distinct_from_file_bytes(tmp_path):
    row = sample(tmp_path)
    with Image.open(row["image_path"]) as image:
        import hashlib
        row["image_sha256"] = hashlib.sha256(b"20x10:" + image.convert("RGB").tobytes()).hexdigest()
    loaded = training.load_prepared_records(jsonl(tmp_path, [row]))[0]
    assert loaded["image_sha256"] == row["image_sha256"]
    assert loaded["image_file_sha256"] == training.sha256_file(row["image_path"])
    assert loaded["image_sha256"] != loaded["image_file_sha256"]


@pytest.mark.parametrize("change,message", [
    ({"source_split": "test"}, "official test"),
    ({"image_width": 21}, "dimensions"),
    ({"image_sha256": "bad"}, "hash differs"),
    ({"actions": [{"action_type": "click", "x": 4, "y": 5, "target_bbox": []}]}, "extra_action_fields"),
    ({"actions": [{"action_type": "click", "x": 20, "y": 5}]}, "outside_image"),
])
def test_bad_prepared_supervision_is_refused(tmp_path, change, message):
    row = {**sample(tmp_path), **change}
    with pytest.raises(ValueError, match=message):
        training.load_prepared_records(jsonl(tmp_path, [row]))


def test_train_source_validation_is_allowed_but_duplicates_are_not(tmp_path):
    row = sample(tmp_path)
    assert training.load_prepared_records(jsonl(tmp_path, [row]))[0]["source_split"] == "train"
    with pytest.raises(ValueError, match="duplicate sample_id"):
        training.load_prepared_records(jsonl(tmp_path, [row, row]))


@pytest.mark.parametrize("field", ["sample_id", "group_id", "image_sha256"])
def test_validation_cannot_leak_training_identity(field):
    train = {"sample_id": "a", "group_id": "ga", "image_sha256": "sha-a"}
    validation = {"sample_id": "b", "group_id": "gb", "image_sha256": "sha-b", field: train[field]}
    with pytest.raises(ValueError, match=field):
        training.ensure_separate_validation([train], [validation])


def test_language_targets_never_include_visual_projections():
    model = SimpleNamespace(named_modules=lambda: [
        ("model.visual.blocks.0.q_proj", None), ("model.visual.merger.o_proj", None),
        ("model.language_model.layers.0.q_proj", None), ("model.layers.1.o_proj", None),
        ("model.layers.1.mlp", None),
    ])
    assert training.language_lora_targets(model, ("q_proj", "o_proj")) == [
        "model.language_model.layers.0.q_proj", "model.layers.1.o_proj",
    ]
    with pytest.raises(ValueError, match="no language projection"):
        training.language_lora_targets(model, ("not_present",))


def test_visual_freezing_is_explicit_and_requires_real_named_parameters():
    import torch

    visual = torch.nn.Parameter(torch.ones(2))
    language = torch.nn.Parameter(torch.ones(3))
    model = SimpleNamespace(named_parameters=lambda: [("model.visual.weight", visual),
                                                     ("model.layers.weight", language)])
    assert training.freeze_visual_encoder(model) == 2
    assert not visual.requires_grad and language.requires_grad
    with pytest.raises(ValueError, match="no visual encoder"):
        training.freeze_visual_encoder(SimpleNamespace(named_parameters=lambda: [("weight", language)]))


def test_partial_accumulation_has_full_mean_gradient():
    import torch

    scalar = torch.tensor(1.0, requires_grad=True)
    rows = [{"sample_id": str(value), "value": value} for value in (1, 2, 9)]
    gradients = []
    for group in training.accumulation_groups(list(range(3)), 2):
        scalar.grad = None
        logged = list(training.backward_group([rows[index] for index in group],
                      lambda row: (scalar * row["value"], 1), lambda loss: loss.backward()))
        gradients.append(scalar.grad.item())
        assert [item[1] for item in logged] == [rows[index]["value"] for index in group]
    assert gradients == [1.5, 9.0]  # Last one must not be divided by configured size 2.


def test_nonfinite_loss_never_backpropagates():
    import torch

    calls = []
    with pytest.raises(ValueError, match="non-finite"):
        list(training.backward_group([{"sample_id": "bad"}],
             lambda _: (torch.tensor(float("nan")), 1), calls.append))
    assert calls == []


@pytest.fixture
def synthetic_loader(tmp_path, monkeypatch):
    """Replace all HF/PEFT constructors; no real weights or GPU are touched."""
    import torch

    base = tmp_path / "synthetic-local-model"
    base.mkdir()
    calls = []
    visual = torch.nn.Parameter(torch.ones(2))
    language = torch.nn.Parameter(torch.ones(3))
    model = SimpleNamespace(
        config=SimpleNamespace(use_cache=True),
        named_modules=lambda: [("visual.q_proj", None), ("model.layers.0.q_proj", None)],
        named_parameters=lambda: [("model.visual.weight", visual),
                                  ("model.layers.0.q_proj.lora_A", language)],
        gradient_checkpointing_enable=lambda **kwargs: calls.append(("checkpoint", kwargs)),
        train=lambda: calls.append(("train", {})), eval=lambda: calls.append(("eval", {})),
    )
    def loader(name, result):
        return SimpleNamespace(from_pretrained=lambda path, **kwargs:
                               calls.append((name, {"path": path, **kwargs})) or result)
    hf = SimpleNamespace(
        AutoConfig=loader("architecture", SimpleNamespace(model_type="qwen2_vl")),
        AutoProcessor=loader("processor", "synthetic-processor"),
        Qwen2VLForConditionalGeneration=loader("model", model),
        Qwen2_5_VLForConditionalGeneration=loader("unused-model", model),
        BitsAndBytesConfig=lambda **kwargs: calls.append(("quantization", kwargs)) or kwargs,
    )
    def prepare(model, **kwargs):
        calls.append(("prepare", kwargs))
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)
        return model
    def add_lora(model, options):
        calls.append(("lora", options))
        language.requires_grad_(True)
        return model
    def load_adapter(model, path, **kwargs):
        calls.append(("adapter", {"path": path, **kwargs}))
        if kwargs.get("is_trainable"):
            language.requires_grad_(True)
        return model
    peft = SimpleNamespace(
        LoraConfig=lambda **kwargs: kwargs, prepare_model_for_kbit_training=prepare,
        get_peft_model=add_lora,
        PeftModel=SimpleNamespace(from_pretrained=load_adapter),
    )
    monkeypatch.setitem(sys.modules, "transformers", hf)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = TrainingConfig(base_model_path=str(base))
    return config, calls, model, visual, language


def test_training_loader_quantizes_locally_and_freezes_visual(synthetic_loader):
    config, calls, expected_model, visual, language = synthetic_loader
    model, processor = training.load_model_and_processor(config, for_training=True)
    assert model is expected_model and processor == "synthetic-processor"
    by_name = dict(calls)
    assert by_name["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert by_name["quantization"]["bnb_4bit_use_double_quant"] is True
    assert by_name["model"]["local_files_only"] is True
    assert by_name["model"]["trust_remote_code"] is False
    assert by_name["model"]["attn_implementation"] == "sdpa"
    assert by_name["processor"]["use_fast"] is True
    assert by_name["lora"]["target_modules"] == ["model.layers.0.q_proj"]
    assert not visual.requires_grad and language.requires_grad
    assert model.config.use_cache is False


@pytest.mark.parametrize("adapter", [None, "synthetic-adapter"])
def test_evaluation_loader_never_creates_trainable_lora(synthetic_loader, adapter):
    config, calls, _, _, _ = synthetic_loader
    training.load_model_and_processor(config, adapter_path=adapter, for_training=False)
    names = [name for name, _ in calls]
    assert "prepare" not in names and "lora" not in names and "train" not in names
    assert "eval" in names
    if adapter is not None:
        assert dict(calls)["adapter"]["is_trainable"] is False


def test_loader_no_silent_cpu_fallback(synthetic_loader, monkeypatch):
    import torch

    config, calls, _, _, _ = synthetic_loader
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device is unavailable"):
        training.load_model_and_processor(config, for_training=True)
    assert calls == []


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    import torch

    # This fixture owns a CPU optimizer. Its graph-capture health probe must
    # not initialize CUDA and contaminate later input-only preparation tests.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    class SyntheticAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.visual = torch.nn.Linear(1, 1)
            self.visual.requires_grad_(False)
            self.calls = []

        def forward(self, **batch):
            self.calls.append((self.training, torch.is_grad_enabled()))
            return SimpleNamespace(loss=((self.lora_weight * batch["value"] - 2) ** 2).mean())

        def save_pretrained(self, path, **kwargs):
            path = Path(path)
            path.mkdir(exist_ok=True)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            # Deliberately synthetic bytes: only real eval reload validates tensors.
            (path / "adapter_model.safetensors").write_text(
                json.dumps({"weight": self.lora_weight.detach().tolist()}), encoding="utf-8")

    model = SyntheticAdapter()
    processor = SyntheticProcessor()
    rows = [sample(tmp_path, str(index), color=color) for index, color in enumerate(("red", "blue", "green"))]
    train_path = jsonl(tmp_path, rows)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    config = TrainingConfig(base_model_path=str(base), train_jsonl=str(train_path),
                            output_dir=str(tmp_path / "out"), device="cpu", load_in_4bit=False,
                            compute_dtype="float32", epochs=1, gradient_accumulation_steps=2)
    monkeypatch.setattr(training, "load_model_and_processor", lambda *args, **kwargs: (model, processor))
    def encode(processor, row, config):
        return {"value": torch.tensor([float(int(row["sample_id"]) + 1)]),
                "labels": torch.tensor([[-100, 8, 9]])}
    monkeypatch.setattr(training, "encode_training_sample", encode)
    return config, model, processor, rows, encode


def test_synthetic_run_records_real_updates_and_partial_group(synthetic_run):
    config, model, _, _, _ = synthetic_run
    summary = training.run_training(config)
    assert summary["status"] == "completed"
    assert summary["optimizer_steps"] == 2 and summary["micro_steps"] == 3
    assert summary["epochs_completed"] == 1
    assert [metric["effective_batch_size"] for metric in summary["step_metrics"]] == [2, 1]
    assert summary["adapter_parameters_changed"] is True
    assert summary["source_hashes"] and summary["base_model"]["files"]
    persisted = json.loads((Path(config.output_dir) / "training_summary.json").read_text(encoding="utf-8"))
    assert persisted == summary
    assert Path(summary["artifacts"]["adapter"]).is_dir()
    evidence = summary["artifact_file_evidence"]["adapter"]
    assert evidence["required_files_present"] and not evidence["reload_verified"]
    adapter = Path(summary["artifacts"]["adapter"]) / "adapter_model.safetensors"
    assert evidence["files"][adapter.name] == {
        "bytes": adapter.stat().st_size, "sha256": training.sha256_file(adapter),
    }
    assert all(training_flag and grad for training_flag, grad in model.calls)


@pytest.mark.parametrize("smoke,max_steps,reason", [(True, 20, "smoke_step_limit"), (False, 1, "max_steps")])
def test_actual_step_limits_are_explicit(synthetic_run, smoke, max_steps, reason):
    config, _, _, _, _ = synthetic_run
    summary = training.run_training(replace(config, smoke=smoke, max_steps=max_steps))
    assert summary["status"] == "completed"
    assert summary["optimizer_steps"] == 1 and summary["samples_seen"] == 2
    assert summary["epochs_completed"] == 0
    assert summary["termination_reason"] == reason
    assert summary["run_kind"] == ("smoke" if smoke else "training")


def test_validation_uses_no_grad_and_no_training_updates(synthetic_run, tmp_path):
    config, model, _, _, _ = synthetic_run
    validation = sample(tmp_path, "8", color="white")
    config = replace(config, validation_jsonl=str(jsonl(tmp_path, [validation], "validation")))
    summary = training.run_training(config)
    assert summary["status"] == "completed"
    assert summary["optimizer_steps"] == 2 and summary["samples_seen"] == 3
    assert summary["validation_metrics"][0]["examples"] == 1
    assert model.calls[-1] == (False, False)
    assert model.training is True


@pytest.mark.parametrize("exception,status", [(KeyboardInterrupt(), "cancelled"), (RuntimeError("bad batch"), "failed")])
def test_interruption_keeps_actual_prior_updates_and_partial_adapter(synthetic_run, monkeypatch, exception, status):
    config, _, _, _, encode = synthetic_run
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise exception
        return encode(*args)
    monkeypatch.setattr(training, "encode_training_sample", interrupted)
    summary = training.run_training(config)
    assert summary["status"] == status and summary["optimizer_steps"] == 1
    assert summary["micro_steps"] == 2
    assert "adapter" not in summary["artifacts"]
    assert Path(summary["artifacts"]["partial_adapter"]).is_dir()
    assert summary["error"]["type"] == type(exception).__name__
    assert (Path(config.output_dir) / "training_config.json").is_file()


def test_zero_update_failure_is_not_success_or_adapter(synthetic_run, monkeypatch):
    config, _, _, _, _ = synthetic_run
    monkeypatch.setattr(training, "load_model_and_processor", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")))
    summary = training.run_training(config)
    assert summary["status"] == "failed" and summary["optimizer_steps"] == 0
    assert summary["artifacts"] == {}
    assert summary["error"]["message"] == "load failed"


def test_existing_evidence_is_preserved(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    previous = output / "training_summary.json"
    previous.write_text("prior evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        training.run_training(TrainingConfig(output_dir=str(output)))
    assert previous.read_text(encoding="utf-8") == "prior evidence"


def test_training_cli_overrides_config_and_propagates_cancelled(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("synthetic_week5_train_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(TrainingConfig(epochs=2).to_dict()), encoding="utf-8")
    configs = []
    monkeypatch.setattr(cli, "run_training", lambda config: configs.append(config) or {"status": "cancelled"})
    assert cli.main(["--config", str(config_path), "--epochs", "3", "--smoke", "--no-load-in-4bit"]) == 130
    assert configs[0].epochs == 3 and configs[0].smoke and not configs[0].load_in_4bit


def test_source_hashes_include_label_validation_and_shared_schemas():
    paths = {Path(name).as_posix() for name in training._source_hashes()}
    assert any(path.endswith("finetuning/data.py") for path in paths)
    assert any(path.endswith("gui_agent/schemas.py") for path in paths)
    assert any(path.endswith("datasets/schemas.py") for path in paths)


def test_expected_revision_without_manifest_is_unverified(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    evidence = training._base_model_evidence(tmp_path, TrainingConfig())
    assert evidence["expected_revision"] == TrainingConfig().base_model_revision
    assert evidence["provenance_status"] == "unverified"
    assert evidence["download_manifest"]["weight_hashes_match"] is None
    assert not evidence["upstream_revision_verified_by_trainer"]


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_invalid_download_manifest_retains_its_hash_without_trust(tmp_path, content):
    path = tmp_path / "week5_download_manifest.json"
    path.write_text(content, encoding="utf-8")
    evidence = training._base_model_evidence(tmp_path, TrainingConfig())
    assert evidence["provenance_status"] == "manifest_invalid"
    assert evidence["download_manifest"]["sha256"] == training.sha256_file(path)
    assert evidence["download_manifest"]["weight_hashes_match"] is None


def test_processor_fast_selection_is_explicit_in_config(synthetic_loader):
    config, calls, _, _, _ = synthetic_loader
    training.load_model_and_processor(replace(config, processor_use_fast=False))
    assert dict(calls)["processor"]["use_fast"] is False


@pytest.mark.parametrize("mutate", [None, "hash", "size", "revision", "missing_shard"])
def test_manifest_records_bytes_sha_and_actual_weight_checks(tmp_path, mutate):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"synthetic weights; no real model")
    config = TrainingConfig()
    manifest = {"model_id": config.base_model_id, "revision": config.base_model_revision,
                "files": [{"path": weight.name, "bytes": weight.stat().st_size,
                           "sha256": training.sha256_file(weight)}]}
    if mutate == "hash":
        manifest["files"][0]["sha256"] = "wrong"
    elif mutate == "size":
        manifest["files"][0]["bytes"] += 1
    elif mutate == "revision":
        manifest["revision"] = "different revision"
    elif mutate == "missing_shard":
        manifest["files"].append({"path": "absent.safetensors", "bytes": 4, "sha256": "absent"})
    path = tmp_path / "week5_download_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = training._base_model_evidence(tmp_path, config)
    assert evidence["provenance_status"] == ("download_manifest_matched" if mutate is None else "manifest_mismatch")
    assert evidence["download_manifest"]["sha256"] == training.sha256_file(path)
    assert evidence["download_manifest"]["bytes"] == path.stat().st_size
    assert not evidence["upstream_revision_verified_by_trainer"]


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "adapter_config.json"])
def test_missing_saved_adapter_files_cannot_be_claimed_complete(tmp_path, name):
    for candidate in ("adapter_model.safetensors", "adapter_config.json"):
        if candidate != name:
            (tmp_path / candidate).write_text("synthetic fixture", encoding="utf-8")
    with pytest.raises(RuntimeError, match=name):
        training._saved_file_evidence(tmp_path, adapter=True)


@pytest.mark.parametrize("name", ["preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json",
                                 "tokenizer.json", "chat_template.jinja"])
def test_missing_saved_processor_inputs_are_not_complete(tmp_path, name):
    processor = SyntheticProcessor()
    output = tmp_path / "processor"
    processor.save_pretrained(output)
    (output / name).unlink()
    with pytest.raises(RuntimeError, match="processor is incomplete"):
        training._saved_file_evidence(output, adapter=False)


def test_incomplete_save_changes_terminal_state_to_failed(synthetic_run, monkeypatch):
    config, model, _, _, _ = synthetic_run
    def broken_save(path, **kwargs):
        Path(path).mkdir()
        (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(model, "save_pretrained", broken_save)
    summary = training.run_training(config)
    assert summary["optimizer_steps"] == 2
    assert summary["status"] == "failed"
    assert "adapter_model.safetensors" in summary["error"]["message"]
    assert "adapter" not in summary["artifacts"]
    assert "partial_save_error" in summary


def test_manifest_mismatch_blocks_model_load_and_preserves_evidence(synthetic_run, monkeypatch):
    config, _, _, _, _ = synthetic_run
    base = Path(config.base_model_path)
    (base / "model.safetensors").write_bytes(b"synthetic weights")
    (base / "week5_download_manifest.json").write_text(json.dumps({
        "model_id": config.base_model_id, "revision": config.base_model_revision,
        "files": [{"path": "model.safetensors", "bytes": 17, "sha256": "wrong"}],
    }), encoding="utf-8")
    monkeypatch.setattr(training, "load_model_and_processor", lambda *args, **kwargs: pytest.fail("must not load"))
    summary = training.run_training(config)
    assert summary["status"] == "failed" and summary["optimizer_steps"] == 0
    assert summary["base_model"]["provenance_status"] == "manifest_mismatch"


def initial_adapter_fixture(tmp_path, config, **overrides):
    path = tmp_path / "initial-adapter"
    path.mkdir()
    adapter = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
               "r": config.lora_rank, "lora_alpha": config.lora_alpha,
               "lora_dropout": config.lora_dropout,
               "base_model_name_or_path": config.base_model_path,
               "target_modules": list(config.target_modules), **overrides}
    (path / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"explicit synthetic weight fixture")
    return replace(config, initial_adapter_path=str(path))


@pytest.mark.parametrize("changes", [
    {"r": 32}, {"lora_alpha": 32}, {"lora_dropout": 0.4},
    {"base_model_name_or_path": "different-base"},
    {"target_modules": ["model.visual.q_proj"]}, {"modules_to_save": ["lm_head"]},
])
def test_incompatible_warm_start_is_rejected_before_loading(tmp_path, synthetic_loader, changes):
    config, calls, _, _, _ = synthetic_loader
    config = initial_adapter_fixture(tmp_path, config, **changes)
    with pytest.raises(ValueError, match="initial adapter"):
        training.load_model_and_processor(config, for_training=True)
    assert calls == []


def test_warm_start_keeps_loaded_language_adapter_trainable(tmp_path, synthetic_loader):
    config, calls, _, visual, language = synthetic_loader
    config = initial_adapter_fixture(tmp_path, config)
    evidence_before = training.initial_adapter_evidence(config)
    training.load_model_and_processor(config, for_training=True)
    by_name = dict(calls)
    assert "lora" not in by_name
    assert by_name["adapter"]["is_trainable"] is True
    assert by_name["adapter"]["local_files_only"] is True
    assert language.requires_grad and not visual.requires_grad
    assert evidence_before == training.initial_adapter_evidence(config)
    assert evidence_before["optimizer_state_resumed"] is False


def test_warm_start_run_proves_original_adapter_unchanged(tmp_path, synthetic_run):
    config, _, _, _, _ = synthetic_run
    config = initial_adapter_fixture(tmp_path, config)
    summary = training.run_training(config)
    assert summary["status"] == "completed"
    assert summary["initial_adapter"] == summary["initial_adapter_after"]
    assert not summary["optimizer_state_resumed"]
    assert summary["adapter_parameters_changed"]


def test_changed_warm_start_weights_fail_even_after_optimizer_updates(tmp_path, synthetic_run, monkeypatch):
    config, _, _, _, encode = synthetic_run
    config = initial_adapter_fixture(tmp_path, config)
    weight = Path(config.initial_adapter_path) / "adapter_model.safetensors"
    def tamper(processor, row, config):
        weight.write_bytes(b"synthetic external change")
        return encode(processor, row, config)
    monkeypatch.setattr(training, "encode_training_sample", tamper)
    summary = training.run_training(config)
    assert summary["status"] == "failed" and summary["optimizer_steps"] == 2
    assert summary["error"]["message"] == "initial adapter weights changed during the run"
    assert "adapter" not in summary["artifacts"]


@pytest.mark.parametrize("variant", ["targeted_sequence", "current_step", "conditional_actor"])
def test_targeted_training_prefix_matches_inference_and_complete_labels(tmp_path, variant):
    class PromptRecorder(SyntheticProcessor):
        def apply_chat_template(self, messages, **kwargs):
            self.messages_seen.append(messages)
            return super().apply_chat_template(messages, **kwargs)
    processor = PromptRecorder()
    processor.messages_seen = []
    row = sample(tmp_path)
    row["actor_step"] = "Click the visible text field and enter the supplied value"
    row["actions"] += [{"action_type": "type_text", "text": "输入全文"},
                       {"action_type": "key_press", "key": "enter"}]
    training.encode_training_sample(processor, row, TrainingConfig(prompt_variant=variant))
    prefix, full = processor.messages_seen
    assert prefix == training.build_messages(row, variant)
    assert full[0] == prefix[0]
    assert json.loads(full[1]["content"]) == row["actions"]
    assert "输入全文" not in json.dumps(prefix, ensure_ascii=False)


def test_conditional_actor_training_and_evaluation_share_only_approved_input_fields(tmp_path):
    from gui_agent.finetuning.evaluation import safe_prompt_input
    from gui_agent.finetuning.prompts import build_user_prompt

    row = sample(tmp_path) | {"actor_step": "  输入 A  B\n ", "current_task": "SECRET_RAW",
                              "send_prompt": "SECRET_PROMPT", "actor_step_source": "SECRET_SOURCE"}
    inference = safe_prompt_input(row, "conditional_actor")
    messages = training.build_messages(row, "conditional_actor")
    assert messages[0]["content"][1]["text"] == build_user_prompt(inference, "conditional_actor")
    assert "SECRET" not in json.dumps(messages)
    assert "actor_step" not in safe_prompt_input(row)
    with pytest.raises(KeyError, match="actor_step"):
        training.build_messages(sample(tmp_path, "without-step"), "conditional_actor")


def test_legacy_training_config_keeps_native_target_protocol():
    legacy = TrainingConfig().to_dict()
    legacy.pop("target_protocol")
    config = TrainingConfig.from_dict(legacy)
    assert config.target_protocol == "native"
    assert config.to_dict()["target_protocol"] == "native"
    pointer = TrainingConfig(target_protocol="candidate_compact", prompt_variant="conditional_actor")
    assert TrainingConfig.from_dict(pointer.to_dict()) == pointer
    assert pointer.max_seq_length == 1536


@pytest.mark.parametrize("protocol,prompt", [
    ("candidate", "conditional_actor"), (None, "conditional_actor"),
    ("candidate_compact", "original"), ("candidate_compact", "current_step"),
])
def test_training_target_protocol_is_explicit_and_requires_actor(protocol, prompt):
    with pytest.raises(ValueError, match="target_protocol|requires prompt_variant"):
        TrainingConfig(target_protocol=protocol, prompt_variant=prompt)


@pytest.fixture
def pointer_protocol_stub(monkeypatch):
    """Test trainer routing without model or detector dependencies.

    The real protocol's geometry and label validation have separate tests.
    """
    def messages(row, *, include_target=False):
        user = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["actor_step"]},
        ]}]
        if include_target:
            user.append({"role": "assistant", "content": json.dumps(row["candidate_target"])})
        return user

    def validate(row):
        if not row.get("candidate_snapshot") or not row.get("candidate_label_evidence"):
            raise ValueError("candidate supervision is incomplete")
        if not row.get("candidate_target"):
            raise ValueError("candidate target is incomplete")

    module = SimpleNamespace(build_pointer_messages=messages, validate_pointer_sample=validate)
    monkeypatch.setitem(sys.modules, "gui_agent.finetuning.pointer_protocol", module)
    return module


def test_native_messages_do_not_read_pointer_targets(tmp_path, pointer_protocol_stub, monkeypatch):
    row = sample(tmp_path) | {"candidate_target": "SECRET_CANDIDATE_LABEL"}
    def forbidden(*args, **kwargs):
        raise AssertionError("native target unexpectedly used pointer protocol")
    monkeypatch.setattr(pointer_protocol_stub, "build_pointer_messages", forbidden)
    messages = training.build_messages(row, include_target=True)
    assert json.loads(messages[1]["content"]) == row["actions"]
    assert "SECRET" not in json.dumps(messages)


def test_pointer_encoding_keeps_complete_target_and_same_prefix(tmp_path, pointer_protocol_stub):
    class Recorder(SyntheticProcessor):
        def apply_chat_template(self, messages, **kwargs):
            self.messages_seen.append(messages)
            return super().apply_chat_template(messages, **kwargs)

    row = sample(tmp_path) | {
        "actor_step": "Focus the visible editor and enter the requested text",
        "candidate_target": [{"action_type": "click", "candidate": 2},
                             {"action_type": "type_text", "text": "完整文本 A  B\n "},
                             {"action_type": "key_press", "key": "enter"}],
    }
    processor = Recorder()
    processor.messages_seen = []
    config = TrainingConfig(target_protocol="candidate_compact", prompt_variant="conditional_actor")
    encoded = training.encode_training_sample(processor, row, config)
    prefix, full = processor.messages_seen
    assert prefix == training.build_messages(row, "conditional_actor", target_protocol="candidate_compact")
    assert full[0] == prefix[0]
    assert json.loads(full[1]["content"]) == row["candidate_target"]
    assert "完整文本" not in json.dumps(prefix, ensure_ascii=False)
    assert encoded["labels"].tolist() == [[-100] * 7 + [8, 9, 99]]
    with pytest.raises(ValueError, match="no truncation performed"):
        training.encode_training_sample(processor, row, replace(config, max_seq_length=9))
    assert all(padding is False for _, _, padding, _ in processor.calls)


def test_pointer_loader_retains_native_labels_and_passes_current_image_identity(
    tmp_path, pointer_protocol_stub, monkeypatch,
):
    row = sample(tmp_path) | {"candidate_target": [{"action_type": "click", "candidate": 1}],
                              "candidate_snapshot": {"fixture": True},
                              "candidate_label_evidence": {"fixture": True}}
    seen = []
    monkeypatch.setattr(pointer_protocol_stub, "validate_pointer_sample", lambda record: seen.append(record))
    loaded = training.load_prepared_records(jsonl(tmp_path, [row]), target_protocol="candidate_compact")
    assert len(seen) == 1 and seen[0] == loaded[0]
    assert loaded[0]["actions"] == row["actions"]
    assert loaded[0]["candidate_target"] == row["candidate_target"]
    assert loaded[0]["image_file_sha256"] == training.sha256_file(row["image_path"])


def test_pointer_supervision_is_refused_before_model_loading(
    synthetic_run, pointer_protocol_stub, monkeypatch,
):
    config, _, _, _, _ = synthetic_run
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("model must not load for invalid candidate supervision")
    monkeypatch.setattr(training, "load_model_and_processor", forbidden)
    summary = training.run_training(replace(
        config, target_protocol="candidate_compact", prompt_variant="conditional_actor",
    ))
    assert summary["status"] == "failed" and summary["optimizer_steps"] == 0
    assert summary["micro_steps"] == 0 and summary["artifacts"] == {}
    assert summary["error"]["message"] == "candidate supervision is incomplete"
    assert calls == []


def test_pointer_validation_dataset_is_checked_before_model_loading(
    synthetic_run, pointer_protocol_stub, monkeypatch, tmp_path,
):
    config, _, _, rows, _ = synthetic_run
    for row in rows:
        row.update(candidate_snapshot={"fixture": True}, candidate_label_evidence={"fixture": True},
                   candidate_target=[{"action_type": "click", "candidate": 1}])
    train = jsonl(tmp_path, rows, "pointer_train")
    invalid = sample(tmp_path, "invalid-validation", color="white")
    validation = jsonl(tmp_path, [invalid], "pointer_validation")
    monkeypatch.setattr(training, "load_model_and_processor", lambda *a, **k: pytest.fail("model loaded"))
    summary = training.run_training(replace(
        config, train_jsonl=str(train), validation_jsonl=str(validation),
        target_protocol="candidate_compact", prompt_variant="conditional_actor",
    ))
    assert summary["status"] == "failed" and summary["optimizer_steps"] == 0
    assert summary["error"]["message"] == "candidate supervision is incomplete"


def test_training_cli_selects_pointer_protocol_explicitly(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("synthetic_pointer_train_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    configs = []
    monkeypatch.setattr(cli, "run_training", lambda config: configs.append(config) or {"status": "cancelled"})
    assert cli.main(["--target-protocol", "candidate_compact",
                     "--prompt-variant", "conditional_actor"]) == 130
    assert configs[0].target_protocol == "candidate_compact"
    assert cli.main(["--target-protocol", "candidate_compact"]) == 2
    assert len(configs) == 1


@pytest.fixture
def reviewed_pointer_sample(tmp_path):
    """A complete frozen synthetic label review, without model or detection."""
    row = sample(tmp_path) | {"actor_step": "Focus the visible editor and enter supplied text",
                              "source": "SECRET_LABEL_PROVENANCE"}
    row["actions"] += [{"action_type": "type_text", "text": " A  B\n中文  "},
                       {"action_type": "key_press", "key": "enter"}]
    snapshot = {
        "image_file_sha256": training.sha256_file(row["image_path"]),
        "image_width": 20, "image_height": 10,
        "candidates": [{"candidate_id": "c001", "source": "contour", "role": "contour_region",
                        "bbox": {"left": 2, "top": 1, "right": 12, "bottom": 9},
                        "text": "", "confidence": 0.8}],
    }
    snapshot_path = tmp_path / "candidate_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    review_path = tmp_path / "semantic_review.json"
    review = {
        "schema": "week5.pointer_semantic_review.v1", "status": "closed",
        "reviews": [{"sample_id": row["sample_id"], "action_index": 0, "decision": "accepted",
                     "action_type": "click", "current_subtask": row["actor_step"],
                     "target_point": {"x": 4, "y": 5},
                     "preferred_candidate_id": "c001", "image_file_sha256": snapshot["image_file_sha256"],
                     "snapshot_path": str(snapshot_path),
                     "snapshot_file_sha256": training.sha256_file(snapshot_path)}],
    }
    review_path.write_text(json.dumps(review), encoding="utf-8")
    row.update(candidate_snapshot=snapshot,
               candidate_target=[{"action_type": "click", "candidate": 1, "dx": -3, "dy": 0},
                                 *row["actions"][1:]],
               candidate_label_evidence={"review_path": str(review_path),
                                         "review_sha256": training.sha256_file(review_path)})
    return row


def test_real_pointer_protocol_training_and_inference_share_prefix_and_complete_labels(reviewed_pointer_sample):
    from gui_agent.finetuning.pointer_protocol import (
        POINTER_PROTOCOLS,
        build_pointer_messages,
        map_pointer_actions,
    )

    row = reviewed_pointer_sample
    class Recorder(SyntheticProcessor):
        def apply_chat_template(self, messages, **kwargs):
            self.messages_seen.append(messages)
            return super().apply_chat_template(messages, **kwargs)
    config = TrainingConfig(target_protocol="candidate_compact", prompt_variant="conditional_actor")
    processor = Recorder()
    processor.messages_seen = []
    encoded = training.encode_training_sample(processor, row, config)
    prefix, full = processor.messages_seen
    # Spatial targets extend training while the legacy pointer protocol remains
    # unchanged; both legacy entries must retain their original order.
    assert tuple(training.TARGET_PROTOCOLS) == (*POINTER_PROTOCOLS, "native_spatial_head", "native_spatial_grid_head")
    assert prefix == build_pointer_messages(row)
    assert full[0] == prefix[0]
    target = json.loads(full[1]["content"])
    assert target == row["candidate_target"] and len(target) == 3
    assert target[1]["text"] == " A  B\n中文  "
    assert "SECRET" not in json.dumps(prefix) and "中文" not in json.dumps(prefix, ensure_ascii=False)
    mapped = map_pointer_actions(full[1]["content"], row["candidate_snapshot"], row["image_path"])
    assert mapped == row["actions"]
    assert mapped[0] == {"action_type": "click", "x": 4, "y": 5}
    assert encoded["labels"].tolist() == [[-100] * 7 + [8, 9, 99]]
    with pytest.raises(ValueError, match="no truncation performed"):
        training.encode_training_sample(processor, row, replace(config, max_seq_length=9))


def test_real_pointer_prefix_does_not_require_or_read_supervision(reviewed_pointer_sample):
    row = reviewed_pointer_sample
    complete = training.build_messages(row, "conditional_actor", target_protocol="candidate_compact")
    labels_removed = {key: value for key, value in row.items() if key not in {
        "actions", "candidate_target", "candidate_label_evidence", "action_annotations",
        "source", "source_split", "sample_id", "group_id",
    }}
    assert training.build_messages(labels_removed, "conditional_actor", target_protocol="candidate_compact") == complete


def test_real_pointer_loader_rejects_review_tamper_without_changing_native_mode(
    reviewed_pointer_sample, tmp_path,
):
    row = reviewed_pointer_sample
    path = jsonl(tmp_path, [row], "reviewed_pointer")
    loaded = training.load_prepared_records(path, target_protocol="candidate_compact")
    assert loaded[0]["candidate_target"] == row["candidate_target"]
    assert loaded[0]["actions"] == row["actions"]
    review_path = Path(row["candidate_label_evidence"]["review_path"])
    review_path.write_text(review_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantic review changed"):
        training.load_prepared_records(path, target_protocol="candidate_compact")
    assert training.load_prepared_records(path)[0]["actions"] == row["actions"]


def test_real_pointer_review_tamper_is_rejected_before_model_load(
    reviewed_pointer_sample, tmp_path, monkeypatch,
):
    row = reviewed_pointer_sample
    review_path = Path(row["candidate_label_evidence"]["review_path"])
    review_path.write_text("{}", encoding="utf-8")
    config = TrainingConfig(target_protocol="candidate_compact", prompt_variant="conditional_actor",
                            train_jsonl=str(jsonl(tmp_path, [row], "bad_pointer")),
                            output_dir=str(tmp_path / "refused_pointer"))
    monkeypatch.setattr(training, "load_model_and_processor", lambda *a, **k: pytest.fail("model loaded"))
    summary = training.run_training(config)
    assert summary["status"] == "failed" and summary["optimizer_steps"] == summary["micro_steps"] == 0
    assert summary["error"]["message"] == "candidate semantic review changed"
    assert "model_load_seconds" not in summary and summary["artifacts"] == {}


def test_training_source_hashes_cover_pointer_label_validation():
    paths = {Path(name).name for name in training._source_hashes()}
    assert {"pointer_protocol.py", "candidate_grounding.py"} <= paths


def chunk_sampling_rows(single: int, multiple: int) -> list[dict]:
    return [{"sample_id": str(index), "actions": [{"action_type": "key_press", "key": "enter"}]
             * (1 if index < single else 2)} for index in range(single + multiple)]


def test_legacy_config_keeps_uniform_sampling_default():
    legacy = TrainingConfig().to_dict()
    legacy.pop("sampling_policy")
    assert TrainingConfig.from_dict(legacy).sampling_policy == "uniform"
    assert TrainingConfig.from_dict(legacy).to_dict()["sampling_policy"] == "uniform"


@pytest.mark.parametrize("epoch,expected", [
    (0, [1, 3, 4, 2, 6, 0, 5]),
    (1, [5, 6, 4, 3, 1, 2, 0]),
    (2, [2, 1, 6, 0, 5, 4, 3]),
])
def test_uniform_sampler_pins_historical_seed_shuffle(epoch, expected):
    rows = chunk_sampling_rows(5, 2)
    assert training.epoch_sample_indices(rows, seed=42, epoch=epoch) == expected
    assert training.training_sampling_plan(rows)["planned_samples_per_epoch"] == 7


def test_balanced_sampling_full156_schedule_preserves_labels_and_coverage():
    rows = chunk_sampling_rows(130, 26)
    original = json.dumps(rows, sort_keys=True)
    plan = training.training_sampling_plan(rows, "balanced_action_chunks")
    assert plan["unique_samples"] == 156
    assert plan["single_action_unique_samples"] == 130 and plan["multi_action_unique_samples"] == 26
    assert plan["planned_samples_per_epoch"] == 260
    for epoch in range(4):
        indices = training.epoch_sample_indices(rows, "balanced_action_chunks", seed=42, epoch=epoch)
        counts = Counter(indices)
        assert counts == Counter({index: 1 if index < 130 else 5 for index in range(156)})
        assert indices == training.epoch_sample_indices(rows, "balanced_action_chunks", seed=42, epoch=epoch)
        assert [len(group) for group in training.accumulation_groups(indices, 8)] == [8] * 32 + [4]
    assert json.dumps(rows, sort_keys=True) == original


def test_balanced_remainder_is_distinct_uniform_and_reproducible_across_epochs():
    rows = chunk_sampling_rows(3, 7)
    orders, extras = [], set()
    for epoch in range(8):
        indices = training.epoch_sample_indices(rows, "balanced_action_chunks", seed=42, epoch=epoch)
        counts = Counter(indices)
        assert set(counts) == set(range(10)) and len(indices) == 14
        assert sorted(counts[index] for index in range(3)) == [2, 2, 3]
        assert all(counts[index] == 1 for index in range(3, 10))
        extras.update(index for index in range(3) if counts[index] == 3)
        assert indices == training.epoch_sample_indices(rows, "balanced_action_chunks", seed=42, epoch=epoch)
        orders.append(tuple(indices))
    assert len(set(orders)) == 8 and extras == {0, 1, 2}


@pytest.mark.parametrize("single,multiple", [(278, 40), (3, 7), (4, 4)])
def test_coverage_first_visits_every_unique_before_repeat_and_full_epoch_balances(single, multiple):
    rows = chunk_sampling_rows(single, multiple)
    original = json.dumps(rows, sort_keys=True)
    policy = "coverage_first_balanced_action_chunks"
    plan = training.training_sampling_plan(rows, policy)
    indices = training.epoch_sample_indices(rows, policy, seed=42, epoch=0)
    assert indices == training.epoch_sample_indices(rows, policy, seed=42, epoch=0)
    assert len(set(indices[:len(rows)])) == len(rows) and set(indices[:len(rows)]) == set(range(len(rows)))
    assert sum(index < single for index in indices) == sum(index >= single for index in indices) == max(single, multiple)
    assert plan["coverage_unique_first"] == len(rows)
    assert plan["supplement_visits"] == 2 * max(single, multiple) - len(rows)
    assert plan["bounded_prefix_strata_balance_guaranteed"] is False
    assert json.dumps(rows, sort_keys=True) == original
    if len(rows) == 318:
        bounded = indices[:352]
        assert set(bounded) == set(range(318)) and len(bounded) == 352
        assert sum(index < single for index in bounded) != sum(index >= single for index in bounded)


def test_coverage_first_epoch_tail_is_partial_and_no_diagnostic_id_priority():
    rows = chunk_sampling_rows(159, 159)
    policy = "coverage_first_balanced_action_chunks"
    first = training.epoch_sample_indices(rows, policy, seed=42, epoch=0)
    second = training.epoch_sample_indices(rows, policy, seed=42, epoch=1)
    assert first != second and set(first) == set(second) == set(range(318))
    rows[0]["sample_id"] = "fixed-diagnostic-id-must-not-change-order"
    assert first == training.epoch_sample_indices(rows, policy, seed=42, epoch=0)
    assert [len(group) for group in training.accumulation_groups(first, 8)] == [8] * 39 + [6]


@pytest.mark.parametrize("length", [1, 2])
def test_coverage_first_single_stratum_refuses_before_model_load(synthetic_run, monkeypatch, length):
    config, _, _, rows, _ = synthetic_run
    for row in rows:
        row["actions"] *= length
    Path(config.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(training, "load_model_and_processor", lambda *a, **k: pytest.fail("must not load"))
    summary = training.run_training(replace(config, sampling_policy="coverage_first_balanced_action_chunks"))
    assert summary["status"] == "failed" and summary["micro_steps"] == 0
    assert "requires both single-action and multi-action" in summary["error"]["message"]


@pytest.mark.parametrize("field", ["max_micro_steps", "max_optimizer_attempts"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_hard_training_budget_config_is_positive_integer_or_null(field, value):
    with pytest.raises(ValueError, match=field):
        TrainingConfig.from_dict({field: value})


def test_hard_micro_budget_preserves_completed_partial_group_and_never_shrinks_next(synthetic_run):
    config, _, _, _, _ = synthetic_run
    summary = training.run_training(replace(config, epochs=2, max_micro_steps=4))
    assert summary["status"] == "completed"
    assert summary["micro_steps"] == 3 and summary["optimizer_attempts"] == 2
    assert [step["effective_batch_size"] for step in summary["step_metrics"]] == [2, 1]
    assert summary["termination_reason"] == "max_micro_steps" and summary["epochs_completed"] == 1
    assert summary["budget_caps"]["micro_steps"] == 4
    assert summary["actual_sampling"]["single_action_visits"] == 3
    assert summary["actual_sampling"]["unique_samples"] == 3


def test_hard_micro_budget_refuses_first_group_without_partial_consumption(synthetic_run):
    config, model, _, _, _ = synthetic_run
    summary = training.run_training(replace(config, max_micro_steps=1))
    assert summary["status"] == "failed" and summary["micro_steps"] == 0
    assert summary["optimizer_attempts"] == 0 and not model.calls


def test_hard_optimizer_attempt_cap_prevents_compensation_after_skipped_update(synthetic_run, monkeypatch):
    import torch

    class SyntheticSkipScaler:
        def __init__(self, *args, **kwargs):
            self.steps, self.scale_value = 0, 8.0

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.scale_value

        def step(self, optimizer):
            self.steps += 1
            if self.steps > 1:
                optimizer.step()

        def update(self):
            if self.steps == 1:
                self.scale_value /= 2

    monkeypatch.setattr(torch.amp, "GradScaler", SyntheticSkipScaler)
    config, _, _, _, _ = synthetic_run
    summary = training.run_training(replace(config, epochs=3, max_steps=3,
                                            max_optimizer_attempts=2, max_micro_steps=4))
    assert summary["status"] == "completed" and summary["optimizer_steps"] == 1
    assert summary["optimizer_attempts"] == 2 and summary["skipped_optimizer_steps"] == 1
    assert summary["micro_steps"] == 3 and summary["termination_reason"] == "max_optimizer_attempts"
    assert [step["update_applied"] for step in summary["step_metrics"]] == [False, True]
    assert [step["optimizer_attempt"] for step in summary["step_metrics"]] == [1, 2]


@pytest.mark.parametrize("policy", ["", "unknown", None, []])
def test_invalid_sampling_policy_is_explicit_before_runtime(policy):
    with pytest.raises(ValueError, match="sampling_policy"):
        TrainingConfig.from_dict({"sampling_policy": policy})
    with pytest.raises(ValueError, match="sampling_policy"):
        training.training_sampling_plan(chunk_sampling_rows(1, 1), policy)


@pytest.mark.parametrize("policy", training.SAMPLING_POLICIES)
def test_empty_sampler_is_refused(policy):
    with pytest.raises(ValueError, match="nonempty dataset"):
        training.epoch_sample_indices([], policy, seed=42, epoch=0)


@pytest.mark.parametrize("length", [1, 2])
def test_balanced_single_stratum_fails_before_model_load(synthetic_run, monkeypatch, length):
    config, _, _, rows, _ = synthetic_run
    for row in rows:
        row["actions"] *= length
    Path(config.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(training, "load_model_and_processor", lambda *args, **kwargs: pytest.fail("must not load"))
    summary = training.run_training(replace(config, sampling_policy="balanced_action_chunks"))
    assert summary["status"] == "failed" and summary["optimizer_steps"] == summary["micro_steps"] == 0
    assert "requires both single-action and multi-action" in summary["error"]["message"]
    assert summary["sampling_policy"] == "balanced_action_chunks" and summary["epoch_sampling"] == []
    assert "model_load_seconds" not in summary and summary["artifacts"] == {}


def test_empty_balanced_dataset_fails_before_model_load(synthetic_run, monkeypatch):
    config, _, _, _, _ = synthetic_run
    Path(config.train_jsonl).write_text("", encoding="utf-8")
    monkeypatch.setattr(training, "load_model_and_processor", lambda *args, **kwargs: pytest.fail("must not load"))
    summary = training.run_training(replace(config, sampling_policy="balanced_action_chunks"))
    assert summary["status"] == "failed" and "dataset is empty" in summary["error"]["message"]
    assert summary["optimizer_steps"] == 0 and "model_load_seconds" not in summary


@pytest.mark.parametrize("configured,override", [
    ("uniform", "balanced_action_chunks"), ("balanced_action_chunks", "uniform"),
])
def test_training_cli_sampling_override_is_explicit(tmp_path, monkeypatch, configured, override):
    path = Path(__file__).resolve().parents[1] / "scripts/week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("synthetic_week5_sampling_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config_path = tmp_path / "sampling_config.json"
    config_path.write_text(json.dumps(TrainingConfig(sampling_policy=configured).to_dict()), encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "run_training", lambda config: seen.append(config) or {"status": "cancelled"})
    assert cli.main(["--config", str(config_path), "--sampling-policy", override]) == 130
    assert seen[0].sampling_policy == override


def test_synthetic_balanced_full156_lifecycle_records_actual132_updates(synthetic_run):
    config, model, _, originals, _ = synthetic_run
    rows = []
    for index in range(156):
        row = {**originals[index % 3], "sample_id": str(index), "group_id": f"synthetic-group-{index % 33}"}
        if index >= 130:
            row["actions"] = row["actions"] + [{"action_type": "key_press", "key": "enter"}]
        rows.append(row)
    Path(config.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config = replace(config, sampling_policy="balanced_action_chunks", epochs=4,
                     gradient_accumulation_steps=8, max_steps=132)
    summary = training.run_training(config)
    assert summary["status"] == "completed" and summary["epochs_completed"] == 4
    assert summary["optimizer_steps"] == summary["optimizer_attempts"] == 132
    assert summary["micro_steps"] == summary["samples_seen"] == len(model.calls) == 1040
    assert summary["skipped_optimizer_steps"] == 0 and summary["planned_epoch_optimizer_groups"] == 33
    counts = Counter(key for metric in summary["step_metrics"] for key in metric["sample_ids"])
    assert counts == Counter({str(index): 4 if index < 130 else 20 for index in range(156)})
    for epoch, schedule in enumerate(summary["epoch_sampling"], start=1):
        steps = [item for item in summary["step_metrics"] if item["epoch"] == epoch]
        actual = [key for item in steps for key in item["sample_ids"]]
        assert actual == [rows[index]["sample_id"] for index in schedule["indices"]]
        assert Counter(actual) == schedule["planned_visit_counts"]
        assert [item["effective_batch_size"] for item in steps] == [8] * 32 + [4]
        assert schedule["single_action_visits"] == schedule["multi_action_visits"] == 130
        assert schedule["indices_sha256"] == hashlib.sha256(
            json.dumps(schedule["indices"], separators=(",", ":")).encode()).hexdigest()
    persisted = json.loads((Path(config.output_dir) / "training_summary.json").read_text(encoding="utf-8"))
    assert persisted == summary and summary["validation_data"] is None


@pytest.mark.parametrize("smoke", [False, True])
def test_balanced_limits_keep_planned_schedule_separate_from_actual_visits(synthetic_run, smoke):
    config, _, _, rows, _ = synthetic_run
    rows[2]["actions"] += [{"action_type": "key_press", "key": "enter"}]
    Path(config.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary = training.run_training(replace(config, sampling_policy="balanced_action_chunks", epochs=4,
                                            max_steps=1, smoke=smoke))
    assert summary["status"] == "completed" and summary["optimizer_steps"] == 1
    assert summary["micro_steps"] == 2 and summary["epochs_completed"] == 0
    assert len(summary["epoch_sampling"]) == 1 and summary["epoch_sampling"][0]["planned_samples"] == 4
    actual = [key for item in summary["step_metrics"] for key in item["sample_ids"]]
    assert len(actual) == 2 and sum(summary["epoch_sampling"][0]["planned_visit_counts"].values()) == 4
    assert summary["termination_reason"] == ("smoke_step_limit" if smoke else "max_steps")


def test_balanced_interruption_never_counts_unexecuted_schedule(synthetic_run, monkeypatch):
    config, _, _, rows, encode = synthetic_run
    rows[2]["actions"] += [{"action_type": "key_press", "key": "enter"}]
    Path(config.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    calls = 0
    def cancelled(processor, row, config):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("synthetic cancellation")
        return encode(processor, row, config)
    monkeypatch.setattr(training, "encode_training_sample", cancelled)
    summary = training.run_training(replace(config, sampling_policy="balanced_action_chunks"))
    assert summary["status"] == "cancelled" and summary["optimizer_steps"] == 1
    assert summary["micro_steps"] == 2 and summary["epochs_completed"] == 0
    assert summary["epoch_sampling"][0]["planned_samples"] == 4
    assert len(summary["step_metrics"]) == 1 and "partial_adapter" in summary["artifacts"]


def stratum_config(**changes):
    return TrainingConfig(loss_strategy="registered_stratum_balanced", max_steps=44,
                          max_optimizer_attempts=44, max_micro_steps=352, **changes)


def test_loss_strategy_legacy_uniform_and_registered_hard_prefix_contract():
    legacy = TrainingConfig().to_dict()
    legacy.pop("loss_strategy")
    assert TrainingConfig.from_dict(legacy).loss_strategy == "uniform"
    for values in ({"loss_strategy": "unknown"}, {"loss_strategy": None},
                   {"loss_strategy": "registered_stratum_balanced", "max_steps": 2, "max_optimizer_attempts": 3, "max_micro_steps": 8},
                   {"loss_strategy": "registered_stratum_balanced", "max_steps": 2, "max_optimizer_attempts": 2}):
        with pytest.raises(ValueError, match="loss_strategy|registered_stratum_balanced"):
            TrainingConfig.from_dict(values)


def test_stratum_objective_prefix_338_unique_352_visits_has_equal_nominal_mass():
    rows = chunk_sampling_rows(286, 52)
    original = json.dumps(rows, sort_keys=True)
    cfg = stratum_config(sampling_policy="coverage_first_balanced_action_chunks", seed=42,
                         gradient_accumulation_steps=8, epochs=2)
    plan = training.training_objective_plan(rows, cfg)
    assert plan == training.training_objective_plan(rows, cfg)
    counts = plan["prefix_counts"]
    assert counts == {"single_action": 286, "multi_action": 66, "micro_steps": 352, "optimizer_attempts": 44}
    names = [name for group in plan["registered_prefix"] for name in group["sample_ids"]]
    assert len(set(names[:338])) == len(set(names)) == 338
    assert plan["weights"] == pytest.approx({"single_action": 352 / 572, "multi_action": 352 / 132})
    assert plan["nominal_weight_mass"] == pytest.approx({"single_action": 176, "multi_action": 176})
    assert plan["mean_nominal_weight"] == pytest.approx(1)
    assert json.dumps(rows, sort_keys=True) == original


def test_stratum_prefix_keeps_partial_epoch_tail_and_does_not_shrink_next_group():
    cfg = TrainingConfig(loss_strategy="registered_stratum_balanced", epochs=2, seed=42,
                         max_steps=4, max_optimizer_attempts=4, max_micro_steps=4, gradient_accumulation_steps=2)
    plan = training.training_objective_plan(chunk_sampling_rows(2, 1), cfg)
    assert [len(group["sample_ids"]) for group in plan["registered_prefix"]] == [2, 1]
    assert plan["prefix_counts"] == {"single_action": 2, "multi_action": 1, "micro_steps": 3, "optimizer_attempts": 2}
    assert {group["epoch"] for group in plan["registered_prefix"]} == {1}
    assert plan["weights"] == {"single_action": 0.75, "multi_action": 1.5}


def test_stratum_backward_real_linear_gradients_use_fixed_weight_and_actual_partial_size():
    import torch

    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    model.weight.data.fill_(0.25)
    rows = [{"sample_id": str(i), "actions": [{}] * count, "x": x, "y": y}
            for i, (count, x, y) in enumerate(((1, 1.0, 0.0), (2, 2.0, 1.0), (1, 4.0, 2.0)))]
    weights = {"single_action": 0.75, "multi_action": 1.5}
    for group in ([rows[0], rows[1]], [rows[2]]):
        model.zero_grad()
        def forward(row):
            prediction = model(torch.tensor([[row["x"]]], dtype=torch.float64))
            return (prediction - row["y"]).square().mean(), 2
        logged = list(training.backward_group(group, forward, lambda loss: loss.backward(), loss_weights=weights))
        expected = sum(2 * row["x"] * (0.25 * row["x"] - row["y"]) *
                       (0.75 if len(row["actions"]) == 1 else 1.5) for row in group) / len(group)
        assert model.weight.grad.item() == pytest.approx(expected)
        assert [loss for _, loss, _ in logged] == pytest.approx([(0.25 * row["x"] - row["y"]) ** 2 for row in group])


def test_uniform_backward_is_exactly_the_historical_cpu_gradient():
    import torch

    rows = [{"sample_id": str(i), "x": float(i + 1)} for i in range(3)]
    actual, expected = torch.tensor(0.7, requires_grad=True), torch.tensor(0.7, requires_grad=True)
    observed = list(training.backward_group(rows, lambda row: ((actual * row["x"] - 2).square(), 3),
                                             lambda loss: loss.backward()))
    for row in rows:
        ((expected * row["x"] - 2).square() / len(rows)).backward()
    assert torch.equal(actual.grad, expected.grad)
    assert len(observed) == 3


@pytest.mark.parametrize("dataset_has_multi", [False, True])
def test_stratum_degenerate_actual_prefix_fails_before_model_load(synthetic_run, monkeypatch, dataset_has_multi):
    cfg, model, _, rows, _ = synthetic_run
    if dataset_has_multi:
        # seed42 uniform puts row2 after the first group; it may not supply weights.
        rows[2]["actions"] += [{"action_type": "key_press", "key": "enter"}]
        Path(cfg.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(training, "load_model_and_processor", lambda *a, **k: pytest.fail("must not load"))
    summary = training.run_training(replace(cfg, loss_strategy="registered_stratum_balanced",
                                            max_steps=1, max_optimizer_attempts=1, max_micro_steps=2))
    assert summary["status"] == "failed" and "both strata in the bounded" in summary["error"]["message"]
    assert summary["micro_steps"] == 0 and not model.calls and "model_load_seconds" not in summary


def test_real_synthetic_trainer_wires_weighted_gradient_and_logs_raw_objective(synthetic_run, monkeypatch):
    cfg, model, _, rows, _ = synthetic_run
    rows[2]["actions"] += [{"action_type": "key_press", "key": "enter"}]
    Path(cfg.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    cfg = replace(cfg, loss_strategy="registered_stratum_balanced", max_steps=2,
                  max_optimizer_attempts=2, max_micro_steps=3, max_grad_norm=1000)
    planned = training.training_objective_plan(rows, cfg)
    order = [(name, len(group["sample_ids"])) for group in planned["registered_prefix"] for name in group["sample_ids"]]
    expected, observed = [], []
    forward = model.forward
    def record_forward(**batch):
        index = len(expected)
        name, size = order[index]
        x = batch["value"].item()
        weight = planned["weights"]["single_action" if len(rows[int(name)]["actions"]) == 1 else "multi_action"]
        expected.append(2 * x * (model.lora_weight.detach().item() * x - 2) * weight / size)
        return forward(**batch)
    monkeypatch.setattr(model, "forward", record_forward)
    model.lora_weight.register_hook(lambda gradient: observed.append(gradient.item()))
    summary = training.run_training(cfg)
    assert summary["status"] == "completed" and summary["loss_objective_prefix_match"] is True
    assert observed == pytest.approx(expected)
    assert summary["optimizer_attempts"] == summary["optimizer_steps"] == 2 and summary["micro_steps"] == 3
    assert summary["skipped_optimizer_steps"] == 0
    assert summary["actual_loss_objective"]["single_action_weight_mass"] == pytest.approx(1.5)
    assert summary["actual_loss_objective"]["multi_action_weight_mass"] == pytest.approx(1.5)
    for step in summary["step_metrics"]:
        assert step["mean_example_loss"] >= 0 and step["token_weighted_loss"] >= 0
        assert len(step["loss_weights"]) == step["effective_batch_size"]
        assert step["mean_weighted_example_loss"] >= 0


def test_stratum_objective_does_not_alter_full_labels_mask_or_assistant_end(tmp_path):
    import torch

    row = sample(tmp_path)
    row["actions"] += [{"action_type": "type_text", "text": "完整文字含空白 \n"}]
    cfg = stratum_config()
    uniform = training.encode_training_sample(SyntheticProcessor(), row, TrainingConfig())
    weighted = training.encode_training_sample(SyntheticProcessor(), row, cfg)
    assert all(torch.equal(uniform[key], weighted[key]) for key in uniform)
    assert weighted["labels"].tolist() == [[-100] * 7 + [8, 9, 99]]
    full = training.build_messages(row, include_target=True)
    assert json.loads(full[1]["content"]) == row["actions"]


def test_stratum_skipped_update_never_extends_the_frozen_objective_prefix(synthetic_run, monkeypatch):
    import torch

    class SkipFirst:
        def __init__(self, *args, **kwargs):
            self.attempts, self.value = 0, 8.0

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.value

        def step(self, optimizer):
            self.attempts += 1
            if self.attempts != 1:
                optimizer.step()

        def update(self):
            if self.attempts == 1:
                self.value /= 2

    cfg, model, _, rows, _ = synthetic_run
    rows[2]["actions"] += [{"action_type": "key_press", "key": "enter"}]
    Path(cfg.train_jsonl).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    cfg = replace(cfg, epochs=3, loss_strategy="registered_stratum_balanced", max_steps=2,
                  max_optimizer_attempts=2, max_micro_steps=3)
    monkeypatch.setattr(torch.amp, "GradScaler", SkipFirst)
    summary = training.run_training(cfg)
    assert summary["status"] == "completed" and summary["loss_objective_prefix_match"] is True
    assert summary["optimizer_attempts"] == 2 and summary["optimizer_steps"] == summary["skipped_optimizer_steps"] == 1
    assert summary["micro_steps"] == len(model.calls) == 3
    assert [step["update_applied"] for step in summary["step_metrics"]] == [False, True]
    assert summary["loss_objective"]["prefix_counts"]["optimizer_attempts"] == 2


def test_loss_strategy_cli_override_preserves_registered_budget(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("synthetic_stratum_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config = TrainingConfig(max_steps=2, max_optimizer_attempts=2, max_micro_steps=3)
    config_path = tmp_path / "fixture_config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    seen = []
    monkeypatch.setattr(cli, "run_training", lambda cfg: seen.append(cfg) or {"status": "cancelled"})
    assert cli.main(["--config", str(config_path), "--loss-strategy", "registered_stratum_balanced"]) == 130
    assert seen[0].loss_strategy == "registered_stratum_balanced" and seen[0].max_micro_steps == 3
