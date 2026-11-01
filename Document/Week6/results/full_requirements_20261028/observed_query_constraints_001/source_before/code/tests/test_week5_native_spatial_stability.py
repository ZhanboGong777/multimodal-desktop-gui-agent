"""Small CPU behavior checks for the registered stable spatial head."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from gui_agent.finetuning import native_spatial_artifacts as artifacts
from gui_agent.finetuning import native_spatial_head as core


@pytest.fixture
def torch_cpu(monkeypatch):
    import torch

    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    yield torch
    assert not torch.cuda.is_initialized()


def plan_for(actions):
    decoder = SimpleNamespace(pieces={**{b + 10: bytes([b]) for b in range(256)}, 500: b"<|im_end|>"}, eos_id=500)
    target = core.semantic_target(actions, native_width=100, native_height=80)
    prefix = (1, 2, 3)
    full = prefix + tuple(b + 10 for b in target["semantic_json"].encode()) + (500, 20)
    labels = (-100,) * len(prefix) + full[len(prefix):]
    return core.training_plan(full_ids=full, prefix_ids=prefix, labels=labels, target=target, decoder=decoder)


def model_for(torch):
    class Synthetic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora = torch.nn.Parameter(torch.linspace(-6, 6, 2048))
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.norm = torch.nn.LayerNorm(2048)
            self.calls = 0

        def forward(self, input_ids, **kwargs):
            assert "labels" not in kwargs and "head_activation_diagnostics" not in kwargs
            assert kwargs == {"use_cache": False, "output_hidden_states": False, "return_dict": True}
            self.calls += 1
            states = self.lora[None, None].expand(1, input_ids.shape[1], 2048)
            normalized = self.model.language_model.norm(states)
            return SimpleNamespace(logits=normalized[..., :512])

    return Synthetic()


def test_registered_feature_initialization_contract_and_cpu_rng_isolation(torch_cpu, monkeypatch):
    torch = torch_cpu
    config = core.head_configuration()
    assert config["head_version"] == "week5.native-spatial-head.mlp2048x128x2.v2"
    assert config["feature_normalization"] == {
        "kind": "LayerNorm", "normalized_shape": [2048], "eps": 1e-5,
        "elementwise_affine": False, "scope": "per token last feature axis",
        "dtype": "float32", "gradient": "live autograd; no detach"}
    assert config["output_linear_initialization"] == {"weight": "normal", "mean": 0.0, "std": 1e-3, "bias": "zero"}
    assert config["initialization_device"] == "cpu"
    assert config["initialization_rng"] == "isolated CPU default_generator; CUDA RNG untouched"
    assert (config["lora_learning_rate"], config["head_learning_rate"], config["point_loss_weight"]) == (1e-4, 1e-3, 1.0)

    def forbidden_cuda_seed(*args, **kwargs):
        raise AssertionError("head initialization must not seed CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden_cuda_seed)
    torch.random.default_generator.manual_seed(921)
    before = torch.get_rng_state().clone()
    head = core.make_spatial_head(seed=42)
    assert torch.equal(before, torch.get_rng_state())
    repeat = core.make_spatial_head(seed=42)
    assert all(torch.equal(a, b) for a, b in zip(head.parameters(), repeat.parameters()))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        expected = torch.nn.Sequential(torch.nn.Linear(2048, 128, dtype=torch.float32, device="cpu"), torch.nn.GELU(),
                                       torch.nn.Linear(128, 2, dtype=torch.float32, device="cpu"), torch.nn.Sigmoid())
        torch.nn.init.normal_(expected[2].weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(expected[2].bias)
    assert all(torch.equal(a, b) for a, b in zip(head.parameters(), expected.parameters()))
    assert len(head) == 4 and sum(p.numel() for p in head.parameters()) == 262530
    assert set(head.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in head.parameters())


def test_hash_valid_v1_artifact_is_refused_by_v2_contract(torch_cpu, tmp_path, monkeypatch):
    v1_config = copy.deepcopy(core.head_configuration())
    v1_config["head_version"] = "week5.native-spatial-head.mlp2048x128x2.v1"
    for name in ("feature_normalization", "first_linear_initialization", "initialization_device",
                 "initialization_rng", "output_linear_initialization"):
        v1_config.pop(name)
    bindings = {"base_model_id": "SYNTHETIC", "base_model_revision": "a" * 40,
                "base_model_config_sha256": "b" * 64, "prompt_version": "SYNTHETIC_PROMPT",
                "prompt_sha256": "c" * 64, "eos_token_id": 500,
                "tokenizer_json_sha256": "d" * 64, "core_source_sha256": "e" * 64,
                "training_source_sha256": "f" * 64}
    with monkeypatch.context() as old_authority:
        old_authority.setattr(artifacts, "head_configuration", lambda: copy.deepcopy(v1_config))
        artifacts.save_spatial_head_artifact(core.make_spatial_head(seed=42), tmp_path, role="initial", bindings=bindings)
        manifest_sha = hashlib.sha256((tmp_path / artifacts.MANIFEST_NAME).read_bytes()).hexdigest()
        identity = artifacts.inspect_spatial_head_artifact(tmp_path, role="initial", expected_bindings=bindings,
                                                         expected_manifest_sha256=manifest_sha)
        assert identity["role"] == "initial"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="saved spatial head config or bindings differ"):
        artifacts.inspect_spatial_head_artifact(tmp_path, role="initial", expected_bindings=bindings,
                                              expected_manifest_sha256=manifest_sha)
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}


def test_scale_normalization_is_live_and_observes_one_head_call(torch_cpu):
    torch = torch_cpu
    head = core.make_spatial_head(seed=42)
    base = torch.randn(4, 2048, generator=torch.Generator().manual_seed(88))
    calls, linear_calls = [], []
    handle = head.register_forward_hook(lambda *args: calls.append(1))
    linear_handle = head[2].register_forward_hook(lambda *args: linear_calls.append(1))
    outputs = []
    try:
        for scale in (1.0, 3.5714, 100.0):
            state = (base * scale + 20).requires_grad_()
            diagnostic = {}
            output = core._head_prediction(head, state, diagnostics=diagnostic)
            output[:, 0].sum().backward()
            assert state.grad is not None and float(state.grad.norm()) > 0
            assert output.dtype == torch.float32 and ((output >= 0) & (output <= 1)).all()
            assert diagnostic["head_forward_calls"] == 1
            assert diagnostic["diagnostic_head_forward_calls"] == diagnostic["diagnostic_backward_calls"] == 0
            assert diagnostic["saturated_axes_count"] == 0 and diagnostic["total_axes_count"] == 8
            assert diagnostic["point_count"] == 4
            assert all(abs(rms - 1.0) < 2e-5 for rms in diagnostic["post_normalization_feature_rms"])
            json.dumps(diagnostic, allow_nan=False)
            outputs.append(output.detach())
    finally:
        handle.remove()
        linear_handle.remove()
    assert len(calls) == len(linear_calls) == 3
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=1e-6, atol=1e-6)


def test_optional_diagnostics_preserve_output_and_all_gradients(torch_cpu):
    torch = torch_cpu
    left, right = core.make_spatial_head(seed=7), core.make_spatial_head(seed=7)
    base = torch.randn(3, 2048, generator=torch.Generator().manual_seed(11), dtype=torch.bfloat16)
    state_a, state_b = base.clone().requires_grad_(), base.clone().requires_grad_()
    diagnostic = {}
    output_a = core._head_prediction(left, state_a)
    output_b = core._head_prediction(right, state_b, diagnostics=diagnostic)
    output_a.square().sum().backward()
    output_b.square().sum().backward()
    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
    torch.testing.assert_close(state_a.grad, state_b.grad, rtol=0, atol=0)
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


@pytest.mark.parametrize("has_point", [True, False])
def test_same_model_head_forward_and_per_row_diagnostics(torch_cpu, has_point):
    torch = torch_cpu
    actions = [{"action_type": "click", "x": 80, "y": 16}] if has_point else [{"action_type": "type_text", "text": "synthetic"}]
    plan = plan_for(actions)
    model, head = model_for(torch), core.make_spatial_head(seed=42)
    calls = []
    handle = head.register_forward_hook(lambda *args: calls.append(1))
    batch = {"input_ids": torch.tensor([plan.full_ids]), "labels": torch.tensor([plan.labels])}
    try:
        result = core.forward_objective(model, head, batch, plan)
    finally:
        handle.remove()
    assert model.calls == 1 and result["model_forward_calls"] == result["manual_CE_calls"] == 1
    assert result["default_model_CE_requested"] is False
    assert len(calls) == int(has_point)
    diagnostic = result["head_activation_diagnostics"]
    assert diagnostic["head_forward_calls"] == int(has_point)
    assert diagnostic["point_count"] == int(has_point) and diagnostic["total_axes_count"] == 2 * int(has_point)
    assert diagnostic["diagnostic_model_forward_calls"] == diagnostic["diagnostic_head_forward_calls"] == diagnostic["diagnostic_backward_calls"] == 0
    json.dumps(diagnostic, allow_nan=False)
    torch.testing.assert_close(result["total_loss"], result["semantic_ce"] + result["point_loss"])
    if has_point:
        result["point_loss"].backward()
        assert float(model.lora.grad.norm()) > 0
        assert float(head[0].weight.grad.norm()) > 0 and float(head[2].weight.grad.norm()) > 0
    else:
        assert result["point_loss"].item() == 0 and diagnostic["pre_sigmoid_logit_min"] is None
