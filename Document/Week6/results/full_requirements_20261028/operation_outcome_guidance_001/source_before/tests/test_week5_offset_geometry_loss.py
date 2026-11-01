"""CPU tensors/synthetic vocabulary only; no HF/tokenizer/processor/images."""
from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import pytest

from gui_agent.finetuning import offset_geometry_loss as geometry


@pytest.fixture(autouse=True)
def torch(monkeypatch):
    # Real CPU AdamW probes accelerator availability; no hardware probing is
    # part of these synthetic gradient/loop tests.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch as module

    assert not module.cuda.is_initialized()
    monkeypatch.setattr(module.accelerator, "is_available", lambda: False)
    yield module
    assert not module.cuda.is_initialized()


def fixture(target=None, *, width=1024, height=768, overrides=()):
    import torch

    target = target or [{"action_type": "click", "candidate": 7, "dx": -351, "dy": 0}]
    pieces = {i: bytes([i]) for i in range(256)}
    pieces.update({256: b"user prefix123", 257: b"<|im_end|>"})
    for identifier, piece in overrides:
        pieces[identifier] = piece
    decoder = geometry.legacy.TokenBytes(MappingProxyType(pieces), 257, "0" * 64)
    text = geometry.legacy.canonical(target)
    content = text.encode("utf8")
    suffix, position = [], 0
    ordered = sorted(overrides, key=lambda item: len(item[1]), reverse=True)
    while position < len(content):
        match = next((item for item in ordered if content.startswith(item[1], position)), None)
        if match is None:
            suffix.append(content[position])
            position += 1
        else:
            suffix.append(match[0])
            position += len(match[1])
    prefix = [256, 256]
    full = prefix + suffix + [257, 10]
    labels = [-100, -100] + full[2:]
    plan = geometry.prepare_field_plan(full_ids=full, prefix_ids=prefix, labels=labels, target_json=text,
        decoder=decoder, native_width=width, native_height=height)
    logits = torch.zeros((1, len(full), max(pieces) + 1), dtype=torch.float64, device="cpu", requires_grad=True)
    return plan, logits, torch.tensor([labels], dtype=torch.long), decoder, full, prefix, text


def test_signed_offset_fields_digits_axes_eos_and_candidate_are_separate():
    plan, _, _, decoder, full, _, _ = fixture()
    assert [entry.gt_digit for entry in plan.eligible if entry.field == "dx"] == [3, 5, 1]
    assert [entry.place_multiplier for entry in plan.eligible if entry.field == "dx"] == [100, 10, 1]
    assert all(entry.native_axis_span == 1023 for entry in plan.eligible if entry.field == "dx")
    assert all(entry.native_axis_span == 767 for entry in plan.eligible if entry.field == "dy")
    assert all(plan.weights[index] == 4 for index in plan.candidate_tokens)
    assert all(plan.weights[index] == 8 for index in plan.offset_tokens)
    assert all(entry.field != "candidate" for entry in plan.eligible)
    sign = [i for i in plan.offset_tokens if decoder.pieces[full[i]] == b"-"]
    assert sign and not ({entry.token_index for entry in plan.eligible} & set(sign))
    assert plan.weights[-2:] == (1, 1) and plan.weights[:2] == (0, 0)


def test_far_wrong_digit_more_costly_than_near_wrong_at_same_true_probability(torch):
    plan, logits, labels, *_ = fixture([{"action_type": "click", "candidate": 7, "dx": 5, "dy": 5}])
    entry = next(item for item in plan.eligible if item.field == "dx")
    near, far = logits.detach().clone(), logits.detach().clone()
    near[0, entry.token_index - 1, ord("6")] = 5
    far[0, entry.token_index - 1, ord("9")] = 5
    near_result = geometry.causal_geometry_loss(near, labels, plan)
    far_result = geometry.causal_geometry_loss(far, labels, plan)
    assert torch.equal(near_result["main_weighted_ce"], far_result["main_weighted_ce"])
    assert far_result["geometry_auxiliary"] > near_result["geometry_auxiliary"]
    assert far_result["total_loss"] > near_result["total_loss"]
    outside = near.clone()
    outside[0, entry.token_index - 1, ord("A")] = 20
    outside_result = geometry.causal_geometry_loss(outside, labels, plan)
    assert torch.equal(outside_result["geometry_auxiliary"], near_result["geometry_auxiliary"])
    assert outside_result["main_weighted_ce"] > near_result["main_weighted_ce"]


def test_expected_absolute_digit_error_cannot_cancel_and_gradient_direction_is_correct(torch):
    plan, logits, labels, *_ = fixture([{"action_type": "click", "candidate": 7, "dx": 5, "dy": 5}])
    entry = next(item for item in plan.eligible if item.field == "dx")
    with torch.no_grad():
        logits[0, entry.token_index - 1, ord("1")] = 6
        logits[0, entry.token_index - 1, ord("9")] = 6
    result = geometry.causal_geometry_loss(logits, labels, plan)
    assert result["geometry_error_values"][0] > 0
    result["geometry_auxiliary"].backward()
    assert logits.grad[0, entry.token_index - 1, ord("5")] < 0
    assert logits.grad[0, entry.token_index - 1, ord("9")] > 0
    assert logits.grad[0, entry.token_index - 1, ord("1")] > 0
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, -1]) == 0


def test_unsupported_multi_character_or_mixed_digit_aux_skips_without_losing_main(torch):
    target = [{"action_type": "click", "candidate": 7, "dx": 35, "dy": 0}]
    plan, logits, labels, *_ = fixture(target, overrides=((258, b"35"), (259, b"0}")))
    assert not plan.eligible
    skips = [entry for entry in plan.skipped if entry.field != "candidate"]
    assert len(skips) == 3 and all(entry.reason == "not_bare_single_ASCII_digit_token" for entry in skips)
    assert all(plan.weights[index] == 8 for index in plan.offset_tokens)
    result = geometry.causal_geometry_loss(logits, labels, plan)
    assert result["geometry_auxiliary"] == 0 and result["total_loss"] == result["main_weighted_ce"]
    result["total_loss"].backward()
    assert all(torch.count_nonzero(logits.grad[0, index - 1]) > 0 for index in plan.offset_tokens)


def test_drag_both_endpoint_axes_and_text_digits_keep_exact_original_supervision():
    target = [{"action_type": "drag", "start_candidate": 1, "start_dx": -12, "start_dy": 5,
               "end_candidate": 8, "end_dx": 0, "end_dy": -35},
              {"action_type": "type_text", "text": "unicode输入2026"}, {"action_type": "wait", "duration": 3}]
    plan, _, _, _, _, _, _ = fixture(target)
    assert {entry.field for entry in plan.eligible} == geometry.OFFSET_FIELDS - {"dx", "dy"}
    assert sum(len(str(abs(value.value))) for value in plan.values if value.field in geometry.OFFSET_FIELDS) == len(plan.eligible)
    assert len(plan.candidate_tokens) == 2
    assert all(value.field not in {"text", "duration"} for value in plan.values)
    assert plan.weights[-2:] == (1, 1) and plan.masked_numeric_tokens == 0


@pytest.mark.parametrize("width,height,dx", [(0, 10, 0), (10, False, 0), ("10", 10, 0), (10, 10, 99)])
def test_invalid_native_axis_or_out_of_bounds_training_offset_refuses(width, height, dx):
    with pytest.raises(ValueError):
        fixture([{"action_type": "click", "candidate": 7, "dx": dx, "dy": 0}], width=width, height=height)


def test_unit_native_axis_uses_clamp1_and_causal_shift_masks_previous_prefix(torch):
    plan, logits, labels, *_ = fixture([{"action_type": "click", "candidate": 7, "dx": 0, "dy": 0}], width=1, height=1)
    assert all(entry.native_axis_span == 1 for entry in plan.eligible)
    result = geometry.causal_geometry_loss(logits, labels, plan)
    result["total_loss"].backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0  # predicts a masked prefix label
    assert torch.count_nonzero(logits.grad[:, 1]) > 0  # final prefix row predicts assistant '['
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    for entry in plan.eligible:
        assert torch.count_nonzero(logits.grad[0, entry.token_index - 1]) > 0


def test_one_forward_no_labels_one_manual_CE_and_old_fixed4_exact(monkeypatch, torch):
    plan, logits, labels, *_ = fixture()
    from torch.nn import functional as F

    calls, forward_calls = [], []
    original_ce = F.cross_entropy

    def ce(*args, **kwargs):
        calls.append(1)
        return original_ce(*args, **kwargs)

    class Spy:
        def __call__(self, **batch):
            assert "labels" not in batch
            forward_calls.append(1)
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(F, "cross_entropy", ce)
    result = geometry.forward_objective(Spy(), {"input_ids": torch.zeros_like(labels), "labels": labels}, plan)
    assert len(calls) == len(forward_calls) == result["manual_CE_calls"] == 1
    assert result["default_model_CE_requested"] is False
    assert result["supervised_tokens"] == int((labels[:, 1:] != -100).sum())
    with pytest.raises(ValueError, match="unknown"):
        geometry.forward_objective(Spy(), {"labels": labels}, plan, strategy="unknown")
    assert len(forward_calls) == 1
    baseline = geometry.forward_objective(Spy(), {"labels": labels}, plan, strategy=geometry.legacy.STRATEGY)
    expected = geometry.legacy.causal_ce(logits, labels, torch.tensor([plan.legacy_weights], dtype=torch.float64))
    assert torch.equal(baseline["total_loss"], expected["weighted_ce"])
    assert torch.equal(baseline["raw_ce"], expected["raw_ce"])


def test_public_plan_adapter_keeps_labels_and_excludes_categorical_skips(torch):
    plan, _, labels, decoder, full, prefix, text = fixture()
    encoded = {"input_ids": torch.tensor([full]), "labels": labels.clone()}
    before = {name: item.clone() for name, item in encoded.items()}
    adapted = geometry.plan_from_encoded_sample(encoded, geometry.legacy.strict_json(text), decoder,
                                                native_width=1024, native_height=768)
    assert adapted == plan
    assert all(torch.equal(encoded[name], item) for name, item in before.items())
    stats = geometry.plan_statistics(adapted)
    assert stats["geometry_auxiliary_skipped"] == 0 and stats["geometry_skip_reasons"] == {}
    assert stats["candidate_categorical_fields"] == stats["candidate_value_fields"] == 1
    assert stats["offset_signed_fields"] == stats["offset_sign_active_tokens"] == 1
    assert stats["offset_digit_positions"] == stats["geometry_entries"] == 4
    assert len(stats["geometry_entries_sha256"]) == len(stats["token_weights_sha256"]) == 64
    encoded["labels"][0, len(prefix)] = -100
    with pytest.raises(ValueError, match="suffix"):
        geometry.plan_from_encoded_sample(encoded, geometry.legacy.strict_json(text), decoder,
                                         native_width=1024, native_height=768)


@pytest.mark.parametrize("class_strategy", ["uniform", "registered_stratum_balanced"])
def test_public_training_one_forward_CE_and_component_logs(tmp_path, monkeypatch, class_strategy, torch):
    import json
    from dataclasses import replace
    from pathlib import Path

    import test_week5_numeric_value_loss as numeric_tests
    import test_week5_training as legacy_tests
    from torch.nn import functional as F

    from gui_agent.finetuning import training

    config, model, processor, _, _ = legacy_tests.synthetic_run.__wrapped__(tmp_path, monkeypatch)
    rows = training.load_prepared_records(config.train_jsonl)
    rows[2]["actions"].append({"action_type": "type_text", "text": "42中文"})
    target = {"action_type": "click", "candidate": 7, "dx": -2, "dy": 3}
    for index, row in enumerate(rows):
        row["candidate_target"] = [target] + ([{"action_type": "type_text", "text": "42中文"}] if index == 2 else [])
    metadata, digest = numeric_tests.synthetic_metadata(numeric_tests.fixture([target]), tmp_path)
    config = replace(config, target_protocol="candidate_compact", prompt_variant="conditional_actor",
                     token_loss_strategy=geometry.STRATEGY, numeric_tokenizer_json_path=str(metadata),
                     numeric_tokenizer_json_sha256=digest, loss_strategy=class_strategy,
                     max_steps=2, max_optimizer_attempts=2, max_micro_steps=3, max_grad_norm=1000)
    monkeypatch.setattr(training, "load_prepared_records", lambda *a, **k: rows)
    current, forward_calls, CE_calls = {}, [], []

    def encode(actual_processor, row, actual_config):
        assert actual_processor is processor and actual_config is config
        args = numeric_tests.fixture(row["candidate_target"])
        current.update(args=args, row=row)
        return {"input_ids": torch.tensor([args["full_ids"]]), "labels": torch.tensor([args["labels"]])}

    def model_forward(**inputs):
        assert set(inputs) == {"input_ids"}
        forward_calls.append(current["row"]["sample_id"])
        count = len(current["args"]["full_ids"])
        logits = (torch.arange(count * 258).reshape(1, count, 258) % 11 / 9).float() * model.lora_weight

        class Output:
            @property
            def loss(self):
                pytest.fail("duplicate default model CE")

        output = Output()
        output.logits = logits
        return output

    original = F.cross_entropy

    def CE(*args, **kwargs):
        CE_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", CE)
    monkeypatch.setattr(training, "encode_training_sample", encode)
    monkeypatch.setattr(model, "forward", model_forward)
    summary = training.run_training(config)
    assert summary["status"] == "completed", summary.get("error")
    assert len(forward_calls) == len(CE_calls) == summary["micro_steps"] == 3
    assert [step["effective_batch_size"] for step in summary["step_metrics"]] == [2, 1]
    diagnostics = [json.loads(line) for line in Path(summary["artifacts"]["token_objective_samples"]).read_text(encoding="utf8").splitlines()]
    assert [item["sample_id"] for item in diagnostics] == forward_calls
    total = summary["actual_geometry_objective"]
    assert total["completed_microbatches"] == total["candidate_value_fields"] == 3
    assert total["geometry_auxiliary_skipped"] == 0 and total["geometry_skip_reasons"] == {}
    assert total["offset_value_fields"] == total["geometry_entries"] == total["offset_digit_positions"] == 6
    assert total["offset_sign_active_tokens"] == total["offset_signed_fields"] == 3
    for item in diagnostics:
        assert item["geometry_auxiliary"] == pytest.approx(8 * item["geometry_expected_absolute_error_sum"] / item["supervised_tokens"])
        assert item["total_loss"] == item["token_weighted_ce"] == pytest.approx(item["main_weighted_ce"] + item["geometry_auxiliary"])
        assert item["manual_CE_calls"] == 1 and item["default_model_CE_requested"] is False
        # Synthetic native dimensions are20x10; no canvas dimensions substituted.
        assert (item["native_width"], item["native_height"]) == (20, 10)
    for name in ("raw_ce", "main_weighted_ce", "geometry_auxiliary", "total_loss"):
        assert total[f"{name}_sum"] == pytest.approx(sum(item[name] for item in diagnostics))
    assert total["class_times_total_loss_sum"] == pytest.approx(sum(item["total_loss"] * item["class_weight"] for item in diagnostics))
    assert summary["token_loss_objective"]["signed_offset_VALUE_weight"] == 8
    assert summary["token_loss_objective"]["candidate_VALUE_weight"] == 4
    assert summary["token_loss_objective"]["geometry_coefficient"] == 8


def test_public_config_and_existing_CLI_accept_only_explicit_registered_strategy(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    from gui_agent.finetuning import training

    values = {"token_loss_strategy": geometry.STRATEGY, "target_protocol": "candidate_compact",
              "prompt_variant": "conditional_actor", "numeric_tokenizer_json_path": str(tmp_path / "synthetic.json"),
              "numeric_tokenizer_json_sha256": "0" * 64}
    assert training.TrainingConfig.from_dict(values).token_loss_strategy == geometry.STRATEGY
    for change in ({"token_loss_strategy": "unknown"}, {"numeric_tokenizer_json_sha256": None},
                   {"target_protocol": "native"}, {"numeric_tokenizer_json_path": "relative.json"}):
        with pytest.raises(ValueError):
            training.TrainingConfig.from_dict({**values, **change})
    path = Path(__file__).resolve().parents[1] / "scripts/week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("offset_synthetic_CLI", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    configs = []
    monkeypatch.setattr(cli, "run_training", lambda config: configs.append(config) or {"status": "cancelled"})
    assert cli.main(["--token-loss-strategy", geometry.STRATEGY, "--target-protocol", "candidate_compact",
                     "--prompt-variant", "conditional_actor", "--numeric-tokenizer-json-path", values["numeric_tokenizer_json_path"],
                     "--numeric-tokenizer-json-sha256", "0" * 64]) == 130
    assert len(configs) == 1 and configs[0].token_loss_strategy == geometry.STRATEGY
