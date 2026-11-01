"""CPU synthetic contract checks, never a model/processor or accuracy evaluation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gui_agent.finetuning import native_spatial_head as core

SOURCE = Path(core.__file__)


def fake_decoder():
    return SimpleNamespace(pieces={**{byte + 10: bytes([byte]) for byte in range(256)},
                                  500: b"<|im_end|>"}, eos_id=500)


def byte_ids(text):
    return tuple(byte + 10 for byte in text.encode("utf8"))


def sample_target(actions=None, width=1024, height=768):
    if actions is None:
        actions = [{"action_type": "click", "x": 800, "y": 200},
                   {"action_type": "type_text", "text": "native_head 文字 019"},
                   {"action_type": "key_press", "key": "ENTER"}]
    return core.semantic_target(actions, native_width=width, native_height=height)


def sample_plan(target):
    prefix = (1, 2, 3)
    full = prefix + byte_ids(target["semantic_json"]) + (500, 20)
    labels = (-100,) * len(prefix) + full[len(prefix):]
    return core.training_plan(full_ids=full, prefix_ids=prefix, labels=labels,
                              target=target, decoder=fake_decoder())


@pytest.fixture
def torch_cpu():
    import torch
    assert not torch.cuda.is_initialized()
    return torch


def toy_model(torch):
    """Causal prefix sum with a Qwen-shaped norm path; not a pretrained model."""
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(512, 2048)
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.norm = torch.nn.LayerNorm(2048)
            self.lm = torch.nn.Linear(2048, 512)
            self.forward_calls, self.generate_calls, self.forward_kwargs = 0, 0, []
            self.forced_suffix = ()
            self.generation_config = SimpleNamespace(to_dict=lambda: {
                "repetition_penalty": 1.05, "eos_token_id": [500, 501],
                "num_beams": 4, "bad_words_ids": [[80]], "forced_eos_token_id": 501})
            self.generate_kwargs = None

        def forward(self, input_ids, *, use_cache=False, output_hidden_states=False,
                    return_dict=True, cache_position=None, past_key_values=None, **kwargs):
            assert "labels" not in kwargs
            assert output_hidden_states is False
            self.forward_calls += 1
            self.forward_kwargs.append(set(kwargs))
            embedded = self.embedding(input_ids)
            count = 0 if past_key_values is None else past_key_values[1]
            sums = embedded.cumsum(dim=1)
            if past_key_values is not None:
                sums = sums + past_key_values[0]
            divisor = torch.arange(count + 1, count + embedded.shape[1] + 1,
                                   device=embedded.device).reshape(1, -1, 1)
            states = self.model.language_model.norm(sums / divisor)
            past = (sums[:, -1:, :], count + embedded.shape[1]) if use_cache else None
            return SimpleNamespace(logits=self.lm(states), past_key_values=past)

        def generate(self, *, input_ids, max_new_tokens, do_sample, use_cache,
                     output_hidden_states, return_dict_in_generate, **kwargs):
            assert do_sample is False and use_cache is True
            assert output_hidden_states is False and return_dict_in_generate is False
            self.generate_calls += 1
            self.generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                                    "use_cache": use_cache, "output_hidden_states": output_hidden_states,
                                    "return_dict_in_generate": return_dict_in_generate, **kwargs}
            sequence = input_ids[0].tolist()
            output = self(input_ids=input_ids, use_cache=True, output_hidden_states=False,
                          return_dict=True, cache_position=torch.arange(input_ids.shape[1]))
            suffix = self.forced_suffix[:max_new_tokens]
            for index, identifier in enumerate(suffix):
                sequence.append(identifier)
                if index + 1 < len(suffix):
                    output = self(input_ids=torch.tensor([[identifier]], dtype=torch.long),
                                  use_cache=True, output_hidden_states=False, return_dict=True,
                                  cache_position=torch.tensor([len(sequence) - 1]),
                                  past_key_values=output.past_key_values)
            return torch.tensor([sequence], dtype=torch.long)
    return Toy()


def tensor_batch(torch, plan):
    return {"input_ids": torch.tensor([plan.full_ids], dtype=torch.long),
            "labels": torch.tensor([plan.labels], dtype=torch.long)}


def test_import_is_lazy_and_has_no_torch_model_or_processor():
    script = ("import runpy,sys;runpy.run_path(sys.argv[1]);"
              "assert 'torch' not in sys.modules;assert 'transformers' not in sys.modules;"
              "assert 'PIL' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script, str(SOURCE)], check=False,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_target_preserves_whole_literal_and_all_roles_without_gt_model_text():
    actions = [{"action_type": "type_text", "text": 'native_head 中文 "point" 019'},
               {"action_type": "drag", "start": {"x": 0, "y": 767},
                "end": {"x": 1023, "y": 0}},
               {"action_type": "hotkey", "keys": ["CTRL", "A"]},
               {"action_type": "scroll", "x": 10, "y": 20, "scroll_amount": -2}]
    before = core.canonical(actions)
    target = sample_target(actions)
    assert target["target_protocol"] == "native_spatial_head"
    assert target["target_version"] == "week5.native-spatial-target.v1"
    assert target["semantic_actions"][0] == actions[0]
    assert "1023" not in target["semantic_json"] and "767" not in target["semantic_json"]
    assert [(r["action_index"], r["field"]) for r in target["point_supervision"]] == [
        (1, "end_point"), (1, "start_point"), (3, "point")]
    values = [[row["u"], row["v"]] for row in target["point_supervision"]]
    assert core.materialize_semantic_actions(target["semantic_json"], values,
                                            native_width=1024, native_height=768) == actions
    target["semantic_actions"][2]["keys"].append("ALT")
    assert core.canonical(actions) == before


@pytest.mark.parametrize("raw", [
    "[]", '{}', '```json\n[{"action_type":"click","point":"native_head"}]\n```',
    '[{"action_type":"click","point":"native_head","point":"native_head"}]',
    '[{"action_type":"click","candidate":1,"dx":0,"dy":0}]',
    '[{"action_type":"click","point":true}]',
    '[{"action_type":"click","point":"native_head","x":10}]',
    '[{"action_type":"drag","start_point":"native_head"}]',
    '[{"action_type":"type_text","text":123}]',
    '[{"action_type":"hotkey","keys":[]}]',
    '[{"action_type":"wait","duration":NaN}]',
    '[{"action_type":"scroll","point":"native_head","scroll_amount":0}]',
    '[{"action_type":"unknown"}]',
])
def test_semantic_parser_rejects_invalid_whole_arrays(raw):
    with pytest.raises(ValueError):
        core.semantic_marker_spans(raw)


@pytest.mark.parametrize("value", [[], [[0.5]], [[True, 0.2]], [[0, 0.2]],
                                   [[float("nan"), 0.2]], [[1.01, 0.2]], [[0.5, -0.01]],
                                   [[0.5, 0.2], [0.5, 0.2]]])
def test_materialization_rejects_incomplete_or_invalid_head_values(value):
    target = sample_target()
    with pytest.raises(ValueError):
        core.materialize_semantic_actions(target["semantic_json"], value,
                                         native_width=1024, native_height=768)


def test_native_gt_and_dimensions_are_strict_no_coercion():
    for actions, width, height in [([{"action_type": "click", "x": True, "y": 0}], 4, 4),
                                   ([{"action_type": "click", "x": 4, "y": 0}], 4, 4),
                                   ([{"action_type": "click", "x": 0, "y": -1}], 4, 4),
                                   ([{"action_type": "click", "x": 0, "y": 0}], True, 4)]:
        with pytest.raises(ValueError):
            sample_target(actions, width, height)


def test_marker_bytes_unicode_whitespace_literal_and_order_are_preserved():
    text = (' [ {"text":"中文 native_head", "action_type":"type_text"},\n'
            '{"point" : "native_head", "action_type":"click"} ] ')
    spans = core.semantic_marker_spans(text)
    assert len(spans) == 1 and spans[0].action_index == 1
    assert text.encode("utf8")[spans[0].start:spans[0].end] == b"native_head"
    positions = core.marker_token_positions(text, byte_ids(text), fake_decoder(), start_position=7)
    assert positions == (7 + spans[0].end - 1,)
    assert core.materialize_semantic_actions(text, [[0.5, 0.5]], native_width=4,
                                            native_height=2) == [
        {"action_type": "type_text", "text": "中文 native_head"},
        {"action_type": "click", "x": 2, "y": 1}]
    escaped = '[{"action_type":"click","point":"native_\\u0068ead"}]'
    with pytest.raises(ValueError, match="exact unescaped"):
        core.semantic_marker_spans(escaped)


def test_mixed_token_uses_marker_own_state_and_ambiguous_two_marker_token_rejected():
    text = '[{"action_type":"click","point":"native_head"}]'
    decoder = SimpleNamespace(pieces={1: text.encode("utf8")}, eos_id=500)
    assert core.marker_token_positions(text, [1], decoder, start_position=8) == (8,)
    multiple = core.canonical([{"action_type": "drag", "start_point": "native_head",
                               "end_point": "native_head"}])
    decoder = SimpleNamespace(pieces={1: multiple.encode("utf8")}, eos_id=500)
    with pytest.raises(ValueError, match="share one state"):
        core.marker_token_positions(multiple, [1], decoder)


def test_training_plan_masks_only_prefix_and_supervises_full_semantics_eos():
    target = sample_target()
    plan = sample_plan(target)
    assert plan.marker_positions[0] == plan.prefix_length + core.semantic_marker_spans(
        target["semantic_json"])[0].end - 1
    assert plan.supervised_tokens == len(plan.full_ids) - plan.prefix_length
    assert plan.labels[plan.full_ids.index(500)] == 500
    labels = list(plan.labels); labels[plan.marker_positions[0]] = -100
    with pytest.raises(ValueError, match="complete semantic"):
        core.training_plan(full_ids=plan.full_ids, prefix_ids=plan.full_ids[:plan.prefix_length],
                           labels=labels, target=target, decoder=fake_decoder())
    bad = json.loads(core.canonical(target)); bad["point_supervision"][0]["u"] = 0.0
    with pytest.raises(ValueError, match="normalization"):
        sample_plan(bad)


def test_one_forward_one_ce_live_point_gradient_and_fp32_autocast(torch_cpu, monkeypatch):
    torch = torch_cpu
    model, head = toy_model(torch), core.make_spatial_head()
    plan = sample_plan(sample_target())
    ce_calls = []
    original = torch.nn.functional.cross_entropy
    def count(*args, **kwargs):
        ce_calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", count)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = core.forward_objective(model, head, tensor_batch(torch, plan), plan)
    point_gradient = torch.autograd.grad(result["point_loss"], model.embedding.weight,
                                         retain_graph=True)[0]
    assert torch.isfinite(point_gradient).all() and point_gradient.abs().sum() > 0
    result["total_loss"].backward()
    assert model.forward_calls == len(ce_calls) == result["manual_CE_calls"] == 1
    assert result["default_model_CE_requested"] is False
    assert result["prediction_unit"].dtype == torch.float32
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    assert sum(parameter.numel() for parameter in head.parameters()) == 262530
    assert not torch.cuda.is_initialized()


def test_point_loss_exact_per_axis_beta_and_degenerate_axis(torch_cpu):
    torch = torch_cpu
    prediction = torch.tensor([[0.2, 0.4], [0.8, 0.9]], requires_grad=True)
    target = torch.tensor([[0.0, 0.1], [0.7, 0.9]])
    delta = (prediction - target).abs()
    beta = torch.tensor([20 / 1023, 20 / 767])
    expected = torch.where(delta < beta, delta.square() / (2 * beta), delta - beta / 2).mean()
    actual = core.point_loss(prediction, target, native_width=1024, native_height=768)
    assert torch.equal(actual, expected)
    degenerate = torch.tensor([[0.9, 0.5]], requires_grad=True)
    loss = core.point_loss(degenerate, torch.tensor([[0.0, 0.0]]), native_width=1, native_height=2)
    assert torch.allclose(loss, torch.tensor(0.003125))
    loss.backward()
    assert degenerate.grad[0, 0] == 0
    zero = core.point_loss(degenerate, torch.tensor([[0.0, 0.0]]), native_width=1, native_height=1)
    assert zero == 0


def test_gt_labels_and_future_suffix_do_not_change_marker_state(torch_cpu):
    torch = torch_cpu
    model, head = toy_model(torch), core.make_spatial_head()
    first = sample_target()
    changed = sample_target([{"action_type": "click", "x": 20, "y": 650},
                             {"action_type": "type_text", "text": "native_head 文字 019"},
                             {"action_type": "key_press", "key": "ENTER"}])
    assert first["semantic_json"] == changed["semantic_json"]
    plan_a, plan_b = sample_plan(first), sample_plan(changed)
    result_a = core.forward_objective(model, head, tensor_batch(torch, plan_a), plan_a)
    result_b = core.forward_objective(model, head, tensor_batch(torch, plan_b), plan_b)
    assert torch.equal(result_a["prediction_unit"], result_b["prediction_unit"])
    assert not torch.equal(result_a["point_loss"], result_b["point_loss"])
    before = []
    handle = model.model.language_model.norm.register_forward_hook(
        lambda module, inputs, output: before.append(output[0, plan_a.marker_positions[0]].detach().clone()))
    ids = torch.tensor([plan_a.full_ids])
    model(input_ids=ids)
    changed_ids = ids.clone(); changed_ids[:, plan_a.marker_positions[0] + 1:] = 4
    model(input_ids=changed_ids)
    handle.remove()
    assert torch.equal(before[0], before[1])


def test_no_mouse_ce_keeps_full_array_and_never_invents_point(torch_cpu):
    torch = torch_cpu
    target = sample_target([{"action_type": "type_text", "text": "native_head"},
                            {"action_type": "wait", "duration": 0.1}])
    plan = sample_plan(target)
    model, head = toy_model(torch), core.make_spatial_head()
    result = core.forward_objective(model, head, tensor_batch(torch, plan), plan)
    assert result["point_count"] == 0 and result["point_loss"] == 0
    result["total_loss"].backward()
    assert all(parameter.grad is None for parameter in head.parameters())
    assert core.materialize_semantic_actions(target["semantic_json"], [], native_width=1024,
                                            native_height=768) == target["semantic_actions"]


def test_training_refuses_stale_batch_and_gt_kwargs_before_forward(torch_cpu):
    torch = torch_cpu
    plan = sample_plan(sample_target())
    model, head = toy_model(torch), core.make_spatial_head()
    batch = tensor_batch(torch, plan)
    batch["point_supervision"] = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="input-only"):
        core.forward_objective(model, head, batch, plan)
    assert model.forward_calls == 0
    batch = tensor_batch(torch, plan); batch["input_ids"][0, 0] = 4
    with pytest.raises(ValueError, match="immutable"):
        core.forward_objective(model, head, batch, plan)
    assert model.forward_calls == 0


def test_generation_one_call_cache_equals_teacher_forced_marker_state(torch_cpu):
    torch = torch_cpu
    target = sample_target()
    plan = sample_plan(target)
    model, head = toy_model(torch), core.make_spatial_head()
    training = core.forward_objective(model, head, tensor_batch(torch, plan), plan)
    model.eval(); head.eval()
    model.forced_suffix = byte_ids(target["semantic_json"]) + (500,)
    before = model.forward_calls
    result = core.generate_native_spatial(model, head,
        {"input_ids": torch.tensor([plan.full_ids[:plan.prefix_length]], dtype=torch.long)},
        fake_decoder(), native_width=1024, native_height=768)
    assert model.generate_calls == 1
    assert model.forward_calls - before == len(model.forced_suffix)
    assert result["generation_forward_calls"] == len(model.forced_suffix)
    assert result["repair_forward_calls"] == 0 and result["strict_joint_output_valid"] is True
    assert torch.allclose(torch.tensor(result["raw_spatial_output"]), training["prediction_unit"],
                          atol=1e-6, rtol=1e-6)
    assert result["mapped_actions"][1:] == target["semantic_actions"][1:]
    assert json.loads(result["generation"]) == result["mapped_actions"]
    assert not torch.cuda.is_initialized()


def test_final_unconsumed_marker_rejected_with_zero_repair_forward(torch_cpu):
    torch = torch_cpu
    text = '[{"action_type":"click","point":"native_head"}]'
    decoder = SimpleNamespace(pieces={1: text.encode("utf8")}, eos_id=500)
    model, head = toy_model(torch).eval(), core.make_spatial_head().eval()
    model.forced_suffix = (1,)
    with pytest.raises(ValueError, match="not consumed"):
        core.generate_native_spatial(model, head, {"input_ids": torch.tensor([[2, 3]])}, decoder,
                                     native_width=1024, native_height=768, max_new_tokens=1)
    assert model.generate_calls == model.forward_calls == 1


def test_cache_wrong_positions_full_recompute_and_gt_input_rejected(torch_cpu):
    torch = torch_cpu
    model, head = toy_model(torch).eval(), core.make_spatial_head().eval()
    with (pytest.raises(ValueError, match="cache_position"),
          core.GenerationCapture(model, head, prefix_ids=(1, 2), max_new_tokens=4)):
        model(input_ids=torch.tensor([[1, 2]]), use_cache=True,
              cache_position=torch.tensor([1, 2]))
    assert model.forward_calls == 0
    with (pytest.raises(ValueError, match="cached single-token"),
          core.GenerationCapture(model, head, prefix_ids=(1, 2), max_new_tokens=4)):
        model(input_ids=torch.tensor([[1, 2]]), use_cache=True,
              cache_position=torch.tensor([0, 1]))
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True,
              cache_position=torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="label-free"):
        core.generate_native_spatial(model, head, {"input_ids": torch.tensor([[1, 2]]),
                                     "actions": []}, fake_decoder(),
                                     native_width=1024, native_height=768)
    assert model.generate_calls == 0


def test_native_bounds_axis_one_and_fp32_head_architecture_refusal(torch_cpu):
    torch = torch_cpu
    text = '[{"action_type":"click","point":"native_head"}]'
    assert core.materialize_semantic_actions(text, [[0.99, 1.0]], native_width=1,
                                            native_height=2) == [{"action_type": "click", "x": 0, "y": 1}]
    assert core.materialize_semantic_actions(text, [[0.0, 0.0]], native_width=2190,
                                            native_height=1706)[0]["x"] == 0
    assert core.materialize_semantic_actions(text, [[1.0, 1.0]], native_width=2190,
                                            native_height=1706)[0] == {
        "action_type": "click", "x": 2189, "y": 1705}
    with pytest.raises(ValueError, match="architecture"):
        core._head_prediction(torch.nn.Linear(2048, 2), torch.zeros((1, 2048)))
    with pytest.raises(ValueError, match="FP32"):
        core._head_prediction(core.make_spatial_head().half(), torch.zeros((1, 2048)))


def test_peft_proxy_and_multiple_drag_marker_cache_contract_only(torch_cpu):
    from contextlib import contextmanager

    torch = torch_cpu
    class InjectedAdapter(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
            self.adapter = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 2048))
            self.active_adapter, self.disable_adapters = "default", False
        def forward(self, ids):
            return self.embedding(ids) + (0 if self.disable_adapters else
                (1 if self.active_adapter == "default" else -1) * self.adapter)
    class PeftProxy(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.active_peft_config = SimpleNamespace(is_prompt_learning=False, peft_type="LORA")
            self.generation_config = base.generation_config
            self.wrapper_generation_calls, self.hook_entries, self.hook_exits = 0, 0, 0
        def get_base_model(self):
            return self.base
        def forward(self, **kwargs):
            return self.base(**kwargs)
        def generate(self, **kwargs):
            # Installed PEFT preparation removes this key for non-prefix LoRA.
            self.wrapper_generation_calls += 1
            prepared = {"input_ids": kwargs["input_ids"], "use_cache": True,
                        "cache_position": torch.arange(kwargs["input_ids"].shape[1])}
            prepared.pop("cache_position")
            return self.base(**prepared)
        @contextmanager
        def _enable_peft_forward_hooks(self, **kwargs):
            assert "adapter_names" not in kwargs and "alora_offsets" not in kwargs
            self.hook_entries += 1
            try:
                yield
            finally:
                self.hook_exits += 1
    target = sample_target([{"action_type": "click", "x": 300, "y": 500},
                            {"action_type": "drag", "start": {"x": 50, "y": 75},
                             "end": {"x": 975, "y": 650}}])
    plan = sample_plan(target)
    base, head = toy_model(torch), core.make_spatial_head()
    base.embedding = InjectedAdapter(base.embedding)
    model = PeftProxy(base)
    result_train = core.forward_objective(model, head, tensor_batch(torch, plan), plan)
    model.eval(); head.eval()
    base.forced_suffix = byte_ids(target["semantic_json"]) + (500,)
    prefix = torch.tensor([plan.full_ids[:plan.prefix_length]], dtype=torch.long)
    with (pytest.raises(ValueError, match="cache_position"),
          core.GenerationCapture(model, head, prefix_ids=plan.full_ids[:plan.prefix_length], max_new_tokens=256)):
        model.generate(input_ids=prefix)
    before = base.forward_calls
    result = core.generate_native_spatial(model, head,
        {"input_ids": prefix},
        fake_decoder(), native_width=1024, native_height=768)
    assert plan.slot_keys == ((0, "point"), (1, "end_point"), (1, "start_point"))
    assert len(result["raw_spatial_output"]) == 3
    assert torch.allclose(torch.tensor(result["raw_spatial_output"]),
                          result_train["prediction_unit"], atol=1e-6, rtol=1e-6)
    assert base.generate_calls == 1 and result["repair_forward_calls"] == 0
    assert base.forward_calls - before == len(base.forced_suffix)
    assert model.wrapper_generation_calls == model.hook_entries == model.hook_exits == 1
    assert base.embedding.active_adapter == "default" and base.embedding.disable_adapters is False
    original = result["raw_spatial_output"]
    for active, disabled in (("second", False), ("default", True)):
        base.embedding.active_adapter, base.embedding.disable_adapters = active, disabled
        changed = core.generate_native_spatial(model, head, {"input_ids": prefix},
            fake_decoder(), native_width=1024, native_height=768)
        assert not torch.equal(torch.tensor(changed["raw_spatial_output"]), torch.tensor(original))
        assert (base.embedding.active_adapter, base.embedding.disable_adapters) == (active, disabled)
    assert model.wrapper_generation_calls == 1 and model.hook_entries == model.hook_exits == 3
    model.active_peft_config.alora_invocation_tokens = [7]
    with pytest.raises(ValueError, match="activated LoRA"):
        core.generate_native_spatial(model, head, {"input_ids": prefix},
            fake_decoder(), native_width=1024, native_height=768)
    assert base.generate_calls == 3 and model.hook_entries == model.hook_exits == 3
    assert list(result["mapped_actions"][1]) == ["action_type", "start", "end"]


def test_bare_non_qwen_norm_path_refused_before_any_model_forward(torch_cpu):
    torch = torch_cpu
    head = core.make_spatial_head().eval()
    model = torch.nn.Linear(4, 4).eval()
    model.generation_config = SimpleNamespace(to_dict=dict)
    with pytest.raises(ValueError, match="language_model.norm"):
        core.generate_native_spatial(model, head, {"input_ids": torch.tensor([[1, 2]])},
                                     fake_decoder(), native_width=1024, native_height=768)


@pytest.mark.parametrize("prompt_learning,peft_type", [(True, "LORA"), (False, "IA3")])
def test_unsupported_peft_wrapper_refused_before_unwrap_or_forward(torch_cpu,
                                                                prompt_learning, peft_type):
    torch = torch_cpu
    base = toy_model(torch)
    class Unsupported(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.active_peft_config = SimpleNamespace(is_prompt_learning=prompt_learning,
                                                     peft_type=peft_type)
            self.unwrap_calls = 0
        def get_base_model(self):
            self.unwrap_calls += 1
            return base
    wrapper = Unsupported()
    with pytest.raises(ValueError, match="nonprompt LoRA"):
        core.resolve_final_norm(wrapper)
    assert wrapper.unwrap_calls == base.forward_calls == 0


def test_generation_policy_overrides_inherited_controls_and_binds_one_call(torch_cpu):
    torch = torch_cpu
    target = sample_target([{"action_type": "wait", "duration": 0.1}])
    model, head = toy_model(torch).eval(), core.make_spatial_head().eval()
    original = model.generation_config.to_dict()
    model.forced_suffix = byte_ids(target["semantic_json"]) + (500,)
    result = core.generate_native_spatial(model, head,
        {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}, fake_decoder(),
        native_width=1024, native_height=768)
    actual = model.generate_kwargs
    assert actual["repetition_penalty"] == 1.0
    assert actual["do_sample"] is False and actual["use_cache"] is True
    assert actual["num_beams"] == actual["num_beam_groups"] == actual["num_return_sequences"] == 1
    assert actual["no_repeat_ngram_size"] == actual["encoder_no_repeat_ngram_size"] == 0
    assert actual["eos_token_id"] == actual["pad_token_id"] == 500
    assert actual["bad_words_ids"] is actual["forced_eos_token_id"] is actual["constraints"] is None
    assert actual["disable_compile"] is True and actual["prefill_chunk_size"] is None
    assert result["generation_policy_binding"] == {
        "model_generation_config": original, "overrides": actual}
    assert model.generation_config.to_dict() == original
    assert model.generate_calls == 1 and result["repair_forward_calls"] == 0
    assert core.objective_contract()["lora_learning_rate"] == 1e-4
    assert core.objective_contract()["head_learning_rate"] == 1e-3
