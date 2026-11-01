"""Small synthetic ByteLevel/causal-gradient contract; no actual corpus/model."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_week5_training as legacy_tests

from gui_agent.finetuning import numeric_value_loss as value
from gui_agent.finetuning import training


@pytest.fixture
def torch(monkeypatch):
    # CPU AdamW probes accelerator graph-capture health even for CPU parameters.
    # Stub only hardware availability for this CPU unit; retain real CE, AdamW,
    # backward and CUDA-state checks. Never reset an initialized CUDA context.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import torch as module

    assert not module.cuda.is_initialized(), "CPU fixture inherited real CUDA initialization"
    monkeypatch.setattr(module.accelerator, "is_available", lambda: False)
    yield module
    assert not module.cuda.is_initialized(), "CPU fixture unexpectedly initialized CUDA"


@pytest.fixture
def synthetic_numeric_run(tmp_path, monkeypatch, torch):
    return legacy_tests.synthetic_run.__wrapped__(tmp_path, monkeypatch)


def fixture(actions, *, merges=()):
    # Byte IDs are deliberately tiny synthetic IDs, not a pretrained tokenizer.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {i: chr(i) for i in visible}
    extra = 0
    for i in range(256):
        if i not in visible:
            alphabet[i] = chr(256 + extra)
            extra += 1
    vocab = {alphabet[i]: i for i in range(256)}
    merge_ids = {}
    for i, merged in enumerate(merges, 258):
        raw = merged.encode("utf8")
        vocab["".join(alphabet[b] for b in raw)] = i
        merge_ids[raw] = i
    doc = {"model": {"type": "BPE", "vocab": vocab}, "decoder": {"type": "ByteLevel"},
           "added_tokens": [{"id": 256, "content": "<|im_end|>"}, {"id": 257, "content": "<|prefix|>"}]}
    data = json.dumps(doc, ensure_ascii=True).encode("utf8")
    decoder = value.TokenBytes.from_frozen_json(data, expected_sha256=hashlib.sha256(data).hexdigest(), eos_id=256)
    text = value.canonical(actions)
    raw, encoded, position = text.encode("utf8"), [], 0
    while position < len(raw):
        merge = next((piece for piece in merge_ids if raw.startswith(piece, position)), None)
        encoded.append(merge_ids[merge] if merge is not None else raw[position])
        position += len(merge) if merge is not None else 1
    full = [257, 257, *encoded, 256, 10]
    labels = [-100, -100, *full[2:]]
    return {"full_ids": full, "prefix_ids": full[:2], "labels": labels,
            "target_json": text, "decoder": decoder}


def click():
    return {"action_type": "click", "candidate": 49, "dx": 10, "dy": -2}


def test_only_structured_numeric_values_not_keys_text_keys_or_other_numbers():
    args = fixture([click(), {"action_type": "type_text", "text": '10 {"candidate":49,"dx":10} 中文\n'},
                    {"action_type": "hotkey", "keys": ["f10", "9"]}, {"action_type": "wait", "duration": 10}])
    plan = value.value_weights(**args)
    assert plan.value_span_count == 3 and len(plan.numeric_tokens) == 6
    assert plan.weights[:2] == (0, 0)
    assert all(plan.weights[i] == 4 for i in plan.numeric_tokens)
    assert all(w == 1 for i, w in enumerate(plan.weights) if i >= 2 and i not in plan.numeric_tokens)
    assert plan.weights[-2:] == (1, 1)  # im_end and newline remain supervised.
    assert plan.boundary_mixed_tokens == ()


def test_drag_both_endpoints_and_scroll_numeric_contract():
    args = fixture([{"action_type": "drag", "start_candidate": 1, "start_dx": 2, "start_dy": -3,
                     "end_candidate": 4, "end_dx": 5, "end_dy": 6},
                    {"action_type": "scroll", "candidate": 7, "dx": 8, "dy": 9, "scroll_amount": 10}])
    plan = value.value_weights(**args)
    assert plan.value_span_count == 9  # scroll_amount is intentionally not a numeric target field.


def test_mixed_numeric_syntax_token_weighted_once_and_reported():
    args = fixture([click()], merges=('10,', ':-'))
    plan = value.value_weights(**args)
    assert len(plan.boundary_mixed_tokens) == 2
    assert all(plan.weights[i] == 4 for i in plan.boundary_mixed_tokens)
    # Even a toy merge covering multiple eligible VALUES has one fixed4 weight.
    multi = value.value_weights(**fixture([click()], merges=('10,"dy":-2',)))
    assert len(multi.boundary_mixed_tokens) == 1
    assert multi.weights[multi.boundary_mixed_tokens[0]] == 4


def test_masked_numeric_label_stays_zero_and_eos_cannot_be_masked():
    args = fixture([click()])
    initial = value.value_weights(**args)
    args["labels"][initial.numeric_tokens[0]] = -100
    changed = value.value_weights(**args)
    assert changed.weights[initial.numeric_tokens[0]] == 0 and changed.masked_numeric_tokens == 1
    args["labels"][-2] = -100
    with pytest.raises(ValueError, match="end must remain supervised"):
        value.value_weights(**args)


@pytest.mark.parametrize("change", ["suffix", "prefix", "label", "fractional", "noncanonical"])
def test_binding_shape_and_numeric_schema_fail_closed(change):
    args = fixture([click()])
    if change == "suffix":
        args["full_ids"][2] = 65
        args["labels"][2] = 65
    elif change == "prefix":
        args["prefix_ids"][0] = 0
    elif change == "label":
        args["labels"][2] = 0
    elif change == "fractional":
        args = fixture([{**click(), "candidate": 1.5}])
    else:
        args["target_json"] = args["target_json"].replace(":", ": ", 1)
    with pytest.raises(ValueError):
        value.value_weights(**args)


def test_real_tiny_gradient_shift_original_count_normalization_and_eos(torch):
    logits = (torch.arange(2 * 5 * 4, dtype=torch.float64, device="cpu").reshape(2, 5, 4) % 7 / 5).requires_grad_()
    labels = torch.tensor([[-100, 2, 1, -100, 0], [-100, -100, 3, 0, -100]], dtype=torch.long)
    weights = torch.tensor([[0, 4, 1, 0, 1], [0, 0, 4, 1, 0]], dtype=torch.float64)
    actual = value.causal_ce(logits, labels, weights)
    terms, raw = [], []
    for b in range(2):
        for target_position in range(1, 5):
            label = int(labels[b, target_position])
            if label == -100:
                continue
            ce = -torch.log_softmax(logits[b, target_position - 1], dim=-1)[label]
            raw.append(ce)
            terms.append(ce * weights[b, target_position])
    expected = torch.stack(terms).sum() / len(terms)
    assert actual["supervised_tokens"] == 5 and actual["nominal_token_weight_mass"] == 11
    assert actual["weighted_ce"].item() == pytest.approx(expected.item())
    assert actual["raw_ce"].item() == pytest.approx(torch.stack(raw).mean().item())
    assert actual["weighted_ce"].item() != pytest.approx(torch.stack(terms).sum().item() / 11)
    a = torch.autograd.grad(actual["weighted_ce"], logits, retain_graph=True)[0]
    e = torch.autograd.grad(expected, logits)[0]
    assert torch.allclose(a, e, atol=1e-12, rtol=1e-12)
    assert a[0, 2].abs().sum() == 0 and a[0, 3].abs().sum() > 0  # ignore position and active final target.


def test_uniform_exact_and_value_forward_uses_no_default_loss_or_duplicate_ce(torch):
    args = fixture([click(), {"action_type": "type_text", "text": "42"}])
    plan = value.value_weights(**args)
    labels = torch.tensor([args["labels"]], dtype=torch.long)
    weights = torch.tensor([plan.weights], dtype=torch.float64)
    logits = torch.zeros((1, len(plan.weights), 258), dtype=torch.float64, requires_grad=True)
    calls = []
    original_loss = logits.sum() + 123
    class NoDefaultLoss:
        def __init__(self):
            self.logits = logits
        @property
        def loss(self):
            pytest.fail("field objective must not request/access default CE")
    def model(**batch):
        calls.append(set(batch))
        return SimpleNamespace(loss=original_loss) if "labels" in batch else NoDefaultLoss()
    batch = {"input_ids": torch.tensor([args["full_ids"]]), "labels": labels}
    legacy = value.forward_objective(model, batch)
    assert legacy["raw_ce"] is legacy["weighted_ce"] is original_loss
    result = value.forward_objective(model, batch, strategy=value.STRATEGY, token_weights=weights)
    assert calls == [{"input_ids", "labels"}, {"input_ids"}]
    assert result["CE_calls"] == result["manual_CE_calls"] == 1 and "labels" in batch
    gradient = torch.autograd.grad(result["weighted_ce"], logits)[0]
    eos_position = len(plan.weights) - 2
    assert weights[0, eos_position] == 1 and gradient[0, eos_position - 1].abs().sum() > 0
    text_position = args["full_ids"].index(ord("4"), plan.numeric_tokens[-1] + 1)
    assert weights[0, text_position] == 1 and gradient[0, text_position - 1].abs().sum() > 0


def test_pin_and_unknown_decoder_rejected_before_any_runtime():
    with pytest.raises(ValueError, match="metadata changed"):
        value.TokenBytes.from_frozen_json(b"{}", expected_sha256="0" * 64, eos_id=256)
    doc = {"model": {"type": "BPE"}, "decoder": {"type": "Unknown"}}
    data = json.dumps(doc).encode()
    with pytest.raises(ValueError, match="decoder"):
        value.TokenBytes.from_frozen_json(data, expected_sha256=hashlib.sha256(data).hexdigest(), eos_id=256)


def test_fixed_weights_cannot_be_trainable(torch):
    logits = torch.zeros((1, 3, 4), dtype=torch.float64)
    labels = torch.tensor([[-100, 2, 1]], dtype=torch.long)
    weights = torch.tensor([[0, 4, 1]], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="cannot be trainable"):
        value.causal_ce(logits, labels, weights)


def synthetic_metadata(args, tmp_path):
    # Rebuild metadata from the synthetic fixture's immutable byte table.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {i: chr(i) for i in visible}
    extra = 0
    for i in range(256):
        if i not in visible:
            alphabet[i] = chr(256 + extra)
            extra += 1
    doc = {"model": {"type": "BPE", "vocab": {alphabet[i]: i for i in range(256)}},
           "decoder": {"type": "ByteLevel"},
           "added_tokens": [{"id": 256, "content": "<|im_end|>"}, {"id": 257, "content": "<|prefix|>"}]}
    path = tmp_path / "synthetic_tokenizer.json"
    data = json.dumps(doc).encode("utf8")
    path.write_bytes(data)
    assert value.load_bound_decoder(str(path), hashlib.sha256(data).hexdigest()).pieces == args["decoder"].pieces
    return path, hashlib.sha256(data).hexdigest()


def test_existing_encoded_output_only_contiguous_mask_and_exact_suffix(tmp_path, torch):
    args = fixture([click()])
    encoded = {"input_ids": torch.tensor([args["full_ids"]]), "labels": torch.tensor([args["labels"]])}
    plan = value.weights_from_encoded_sample(encoded, [click()], args["decoder"])
    assert plan == value.value_weights(**args)
    # First '[' accidentally masked cannot become an inferred shorter target.
    encoded["labels"][0, 2] = -100
    with pytest.raises(ValueError, match="reconstruct"):
        value.weights_from_encoded_sample(encoded, [click()], args["decoder"])
    encoded["labels"][0, 2] = args["full_ids"][2]
    encoded["labels"][0, 4] = -100
    with pytest.raises(ValueError, match="whole assistant suffix"):
        value.weights_from_encoded_sample(encoded, [click()], args["decoder"])


@pytest.mark.parametrize("class_strategy", ["uniform", "registered_stratum_balanced"])
def test_real_trainer_single_ce_gradient_raw_diagnostics_and_partial_group(
    synthetic_numeric_run, tmp_path, monkeypatch, class_strategy, torch,
):
    cfg, model, processor, rows, _ = synthetic_numeric_run
    rows[:] = training.load_prepared_records(cfg.train_jsonl)
    rows[2]["actions"].append({"action_type": "type_text", "text": "42中文"})
    for i, row in enumerate(rows):
        row["candidate_target"] = [click()] + ([{"action_type": "type_text", "text": "42中文"}] if i == 2 else [])
    metadata_path, digest = synthetic_metadata(fixture([click()]), tmp_path)
    cfg = replace(cfg, target_protocol="candidate_compact", prompt_variant="conditional_actor",
                  token_loss_strategy=value.STRATEGY, numeric_tokenizer_json_path=str(metadata_path),
                  numeric_tokenizer_json_sha256=digest, loss_strategy=class_strategy,
                  max_steps=2, max_optimizer_attempts=2, max_micro_steps=3, max_grad_norm=1000)
    monkeypatch.setattr(training, "load_prepared_records", lambda *a, **k: rows)
    indices = training.epoch_sample_indices(rows, cfg.sampling_policy, seed=cfg.seed, epoch=0)
    group_sizes = {rows[i]["sample_id"]: len(group) for group in (indices[:2], indices[2:]) for i in group}
    class_weights = {"single_action": 0.75, "multi_action": 1.5} if class_strategy != "uniform" else {
        "single_action": 1.0, "multi_action": 1.0}
    expected, gradients, encode_calls, raw_expected, weighted_expected = [], [], [], [], []
    current = {}
    def encode(actual_processor, row, config):
        assert actual_processor is processor and config is cfg
        args = fixture(row["candidate_target"])
        current.update(row=row, args=args)
        encode_calls.append(row["sample_id"])
        return {"input_ids": torch.tensor([args["full_ids"]]), "labels": torch.tensor([args["labels"]])}
    def model_forward(**inputs):
        assert set(inputs) == {"input_ids"}  # No default transformer CE.
        args, row = current["args"], current["row"]
        count = len(args["labels"])
        coefficients = (torch.arange(count * 258).reshape(1, count, 258) % 11 / 9).float()
        logits = coefficients * model.lora_weight
        labels = args["labels"]
        plan = value.value_weights(**args)
        valid = [p for p in range(1, count) if labels[p] != -100]
        terms = [-torch.log_softmax(logits[0, p - 1], dim=-1)[labels[p]] for p in valid]
        raw = torch.stack(terms).mean()
        weighted = sum(term * plan.weights[p] for term, p in zip(terms, valid)) / len(valid)
        stratum = "single_action" if len(row["actions"]) == 1 else "multi_action"
        weight, size = class_weights[stratum], group_sizes[row["sample_id"]]
        # Analytical scalar derivative; no production causal_ce oracle.
        derivative = sum(
            ((torch.softmax(logits[0, p - 1].detach(), -1) * coefficients[0, p - 1]).sum()
             - coefficients[0, p - 1, labels[p]]) * plan.weights[p] for p in valid
        ) / len(valid) * weight / size
        expected.append(float(derivative))
        raw_expected.append(float(raw.detach()))
        weighted_expected.append(float(weighted.detach()))
        class Output:
            @property
            def loss(self):
                pytest.fail("trainer requested duplicate default CE")
        result = Output()
        result.logits = logits
        return result
    monkeypatch.setattr(training, "encode_training_sample", encode)
    monkeypatch.setattr(model, "forward", model_forward)
    model.lora_weight.register_hook(lambda gradient: gradients.append(float(gradient)))
    summary = training.run_training(cfg)
    assert summary["status"] == "completed", summary.get("error")
    assert gradients == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert len(encode_calls) == summary["micro_steps"] == 3
    assert [s["effective_batch_size"] for s in summary["step_metrics"]] == [2, 1]
    diagnostics = [json.loads(line) for line in Path(summary["artifacts"]["token_objective_samples"]).read_text(encoding="utf8").splitlines()]
    assert [d["sample_id"] for d in diagnostics] == encode_calls
    assert [d["raw_ce"] for d in diagnostics] == pytest.approx(raw_expected)
    assert [d["token_weighted_ce"] for d in diagnostics] == pytest.approx(weighted_expected)
    assert all(d["manual_CE_calls"] == 1 and d["default_model_CE_requested"] is False for d in diagnostics)
    assert summary["actual_token_objective"]["raw_ce_sum"] == pytest.approx(sum(raw_expected))
    for step, start, stop in zip(summary["step_metrics"], [0, 2], [2, 3]):
        assert step["mean_example_loss"] == pytest.approx(sum(raw_expected[start:stop]) / (stop - start))
        assert step["mean_token_objective_loss"] == pytest.approx(sum(weighted_expected[start:stop]) / (stop - start))
        ds = diagnostics[start:stop]
        assert step["mean_class_times_token_objective_loss"] == pytest.approx(
            sum(d["token_weighted_ce"] * d["class_weight"] for d in ds) / len(ds))


def test_token_config_and_cli_are_explicit_and_legacy_default(tmp_path, monkeypatch):
    import importlib.util

    cfg = training.TrainingConfig()
    assert cfg.token_loss_strategy == "uniform"
    for change in ({"token_loss_strategy": "unknown"}, {"numeric_tokenizer_json_path": str(tmp_path / "tokenizer.json")},
                   {"token_loss_strategy": value.STRATEGY}):
        with pytest.raises(ValueError):
            training.TrainingConfig.from_dict({**cfg.to_dict(), **change})
    path = Path(__file__).resolve().parents[1] / "scripts/week5_train_lora.py"
    spec = importlib.util.spec_from_file_location("numeric_synthetic_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = []
    monkeypatch.setattr(cli, "run_training", lambda config: seen.append(config) or {"status": "cancelled"})
    assert cli.main(["--token-loss-strategy", value.STRATEGY, "--target-protocol", "candidate_compact",
                     "--prompt-variant", "conditional_actor", "--numeric-tokenizer-json-path", str(tmp_path / "x.json"),
                     "--numeric-tokenizer-json-sha256", "0" * 64]) == 130
    assert seen[0].token_loss_strategy == value.STRATEGY and seen[0].numeric_tokenizer_json_sha256 == "0" * 64
