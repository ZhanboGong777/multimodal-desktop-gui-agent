"""Small synthetic CPU transport/parent-gate QA; no HF, pixels or real corpus."""
from __future__ import annotations

import copy
import hashlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gui_agent.finetuning import evaluation
from gui_agent.finetuning import native_spatial_grid_head as grid
from gui_agent.finetuning import native_spatial_grid_inference as inference
from gui_agent.finetuning import native_spatial_inference as legacy_inference
from gui_agent.finetuning import visual_pointer_inference as visual_transport

torch.set_num_threads(1)


def synthetic_metadata():
    return {"native_size": [56, 56], "canvas_size": [56, 56], "overview_bbox": [0, 0, 56, 56],
            "overview_rgb_byteexact": True, "id_table": []}


def frozen_input():
    return inference._freeze_input(synthetic_metadata(), {"input_binding_sha256": "a" * 64, "visual_metadata_file_sha256": "b" * 64},
                                   (1, 5000, 5000, 5000, 5000, 2), (1, 4, 4), 5000)


def decoder():
    return SimpleNamespace(eos_id=1000, pieces={**{token: bytes([token]) for token in range(256)},
                                              1000: b"<|im_end|>", 5000: b"<|image_pad|>"})


class ToyQwenTransport(torch.nn.Module):
    """Only an eager norm and scripted generation; no Transformers or weights."""
    def __init__(self, suffix, *, bad_cache=False, before_cached=None):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.norm = torch.nn.LayerNorm(2048)
        self.config = SimpleNamespace(image_token_id=5000)
        self.generation_config = SimpleNamespace(to_dict=dict, eos_token_id=1000)
        self.suffix, self.bad_cache, self.before_cached = suffix, bad_cache, before_cached
        self.forward_count = 0

    def forward(self, *, input_ids, cache_position, **kwargs):
        if self.forward_count and self.before_cached is not None:
            self.before_cached()
            self.before_cached = None
        self.forward_count += 1
        base = torch.arange(2048, dtype=torch.float32)[None, None, :]
        state = torch.sin(base * .013 + input_ids.float()[..., None] * .007)
        return self.model.language_model.norm(state)

    def generate(self, **kwargs):
        prefix = kwargs["input_ids"]
        for step in range(len(self.suffix)):
            ids = prefix if step == 0 else torch.tensor([[self.suffix[step - 1]]], dtype=torch.long)
            start = 0 if step == 0 else prefix.shape[1] + step - 1
            position = torch.arange(start, start + ids.shape[1], dtype=torch.long)
            if self.bad_cache and step == 1:
                position = position + 1
            self(input_ids=ids, cache_position=position, use_cache=True, output_hidden_states=False)
        return torch.cat((prefix, torch.tensor([self.suffix], dtype=torch.long)), dim=1)


def run(monkeypatch, text, *, raw_ids=None, bad_cache=False, before_cached=None, head=None):
    dec = decoder()
    ids = list(text.encode()) + [dec.eos_id] if raw_ids is None else raw_ids
    model = ToyQwenTransport(ids, bad_cache=bad_cache, before_cached=before_cached).eval()
    head = grid.make_spatial_head(seed=42).eval() if head is None else head
    original = frozen_input()
    inputs = {"input_ids": torch.tensor([original.original_prefix_ids]), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    monkeypatch.setattr(inference, "validate_model_contract", lambda model: model.model.language_model.norm)
    result = inference.generate_joint_output(model, head, inputs, dec, native_width=56, native_height=56, grid_input=original)
    return result, original, dec, model, head


def test_one_prefill_key_projection_and_single_parsed_marker_batch(monkeypatch):
    count = {"keys": 0, "batch": []}
    prepare, predict = grid.prepare_key_cache, grid.predict_with_cache
    def keys(*args):
        count["keys"] += 1
        assert tuple(args[1].shape) == (4, 2048)
        return prepare(*args)
    def batch(*args):
        count["batch"].append(tuple(args[1].shape))
        return predict(*args)
    monkeypatch.setattr(grid, "prepare_key_cache", keys)
    monkeypatch.setattr(grid, "predict_with_cache", batch)
    text = '[{"action_type":"drag","end_point":"native_head","start_point":"native_head"}]'
    result, original, dec, model, _ = run(monkeypatch, text)
    assert result["status"] == "ok" and result["strict_joint_output_valid"]
    assert count == {"keys": 1, "batch": [(2, 2048)]}
    assert result["generation_forward_calls"] == model.forward_count == len(result["generated_token_ids"])
    assert result["projected_key_cache_builds"] == result["grid_predict_calls"] == 1
    assert inference.generation_completed(result, eos_id=dec.eos_id)
    assert inference.validate_joint_reply(result, dec, native_width=56, native_height=56, expected_grid_input=original) == result["mapped_actions"]
    assert all(index < len(original.original_prefix_ids) for index in original.image_positions)


def test_nonmouse_structural_native_head_text_does_not_call_grid_prediction(monkeypatch):
    monkeypatch.setattr(grid, "predict_with_cache", lambda *args: pytest.fail("literal is not a structural endpoint"))
    text = '[{"action_type":"type_text","text":"native_head"}]'
    result, original, dec, _, _ = run(monkeypatch, text)
    assert result["status"] == "ok" and result["grid_predict_calls"] == 0
    assert inference.validate_joint_reply(result, dec, native_width=56, native_height=56, expected_grid_input=original) == [{"action_type": "type_text", "text": "native_head"}]


def test_completed_bad_semantics_is_zero_answer_but_model_cache_fault_is_incomplete(monkeypatch):
    reply, _, dec, _, _ = run(monkeypatch, "SYNTHETIC invalid JSON")
    assert reply["status"] == "error" and reply["grid_error_class"] == "semantic"
    assert inference.generation_completed(reply, eos_id=dec.eos_id)
    bad, _, dec, _, _ = run(monkeypatch, '[{"action_type":"click","point":"native_head"}]', bad_cache=True)
    assert bad["status"] == "error" and bad["grid_runtime_failure"]
    assert not inference.generation_completed(bad, eos_id=dec.eos_id)


def test_prediction_cache_failure_after_complete_generation_cannot_make_base_zero(monkeypatch):
    def broken(*args):
        raise RuntimeError("synthetic projected cache mismatch")
    monkeypatch.setattr(grid, "predict_with_cache", broken)
    reply, _, dec, _, _ = run(monkeypatch, '[{"action_type":"click","point":"native_head"}]')
    assert reply["grid_generation_stream_complete"] and reply["grid_runtime_failure"]
    assert reply["grid_error_class"] == "grid_prediction"
    assert not inference.generation_completed(reply, eos_id=dec.eos_id)


def test_unconsumed_last_marker_never_triggers_repair_or_prediction(monkeypatch):
    text = '[{"action_type":"click","point":"native_head"}]'
    dec = decoder()
    dec.pieces[7000] = text.encode()
    monkeypatch.setattr(grid, "predict_with_cache", lambda *args: pytest.fail("unconsumed marker cannot be predicted"))
    monkeypatch.setattr(sys.modules[__name__], "decoder", lambda: dec)
    reply, _, _, model, _ = run(monkeypatch, "", raw_ids=[32] * 255 + [7000])
    assert reply["generation_forward_calls"] == model.forward_count == 256
    assert reply["grid_predict_calls"] == reply["repair_forward_calls"] == 0
    assert reply["status"] == "error" and not reply["grid_runtime_failure"]
    assert inference.generation_completed(reply, eos_id=dec.eos_id)


@pytest.mark.parametrize("field", ["prediction_unit", "all_cell_unit_xy", "attention_weights", "all_token_attention_weights"])
def test_raw_serializer_success_cannot_bypass_independent_grid_math(monkeypatch, field):
    reply, original, dec, _, _ = run(monkeypatch, '[{"action_type":"click","point":"native_head"}]')
    poison = copy.deepcopy(reply)
    value = poison["raw_grid_output"][field][0]
    while isinstance(value[0], list):
        value = value[0]
    value[0] += .01
    with pytest.raises(ValueError, match="FP32"):
        inference.validate_joint_reply(poison, dec, native_width=56, native_height=56, expected_grid_input=original)


def test_feature_poison_and_bool_counter_are_rejected(monkeypatch):
    original = frozen_input()
    feature = inference.marker.strict_json(original.features_json)
    feature["native_unit_centers"][0][0] += .01
    with pytest.raises(ValueError):
        inference._assert_frozen_input(replace(original, features_json=inference.marker.canonical(feature)))
    reply, _, dec, _, _ = run(monkeypatch, '[{"action_type":"click","point":"native_head"}]')
    reply["projected_key_cache_builds"] = True
    assert not inference.generation_completed(reply, eos_id=dec.eos_id)


def test_parent_rebuild_uses_own_approved_metadata_and_pinned_tokenizer(tmp_path):
    metadata = synthetic_metadata()
    path = tmp_path / "SYNTHETIC_input_metadata.json"
    path.write_text(inference.marker.canonical(metadata), encoding="utf8")
    evidence = {"visual_metadata_path": str(path), "visual_metadata_file_sha256": inference._file_sha(path), "input_binding_sha256": "a" * 64}
    descriptor = SimpleNamespace(width=56, height=56, evidence_json=inference.marker.canonical(evidence), assert_current=lambda: None,
                                 input_image_protocol=inference.INPUT_PROTOCOL, visual_prompt_version=inference.PROMPT_VERSION)
    inputs = {"input_ids": torch.tensor([[1, 5000, 5000, 5000, 5000, 2]]), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    frozen = inference.build_grid_input(evidence, descriptor, inputs, image_token_id=5000, expected_bindings=inference._source_bindings())
    assert inference.rebuild_parent_grid_input(descriptor, frozen.evidence, decoder=decoder()) == frozen
    poisoned = copy.deepcopy(frozen.evidence); poisoned["features_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent approved"):
        inference.rebuild_parent_grid_input(descriptor, poisoned, decoder=decoder())
    path.write_text(inference.marker.canonical({**metadata, "native_size": [55, 56]}), encoding="utf8")
    with pytest.raises(ValueError, match="metadata bytes"):
        inference.rebuild_parent_grid_input(descriptor, frozen.evidence, decoder=decoder())


@pytest.mark.parametrize("api", [inference, legacy_inference])
def test_uninitialized_worker_routes_explicit_spatial_request_and_forbids_visual(monkeypatch, api):
    # __new__ avoids construction, process launch, pipe, model and processor.
    worker = evaluation.HFWorker.__new__(evaluation.HFWorker)
    worker.target_protocol = api.TARGET_PROTOCOL if api is inference else api.core.TARGET_PROTOCOL
    worker.input_image_protocol = api.INPUT_PROTOCOL
    worker.spatial_head_artifact = {"eos_token_id": 1000, "role": "final", "synthetic": True}
    visible = {field: "SYNTHETIC input" for field in visual_transport.VISIBLE_FIELDS}
    visible.update(input_image_protocol=api.INPUT_PROTOCOL, visual_prompt_version=api.PROMPT_VERSION,
                   candidate_snapshot=[], candidate_selection_evidence={})
    prompt = "SYNTHETIC prefix only"
    envelope = {"schema": visual_transport.REQUEST_SCHEMAS[api.INPUT_PROTOCOL], "input_image_protocol": api.INPUT_PROTOCOL,
                "visual_prompt_version": api.PROMPT_VERSION, "visible_sample": visible, "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    request = api.bind_spatial_request(envelope, head_identity=worker.spatial_head_artifact)
    sent, routes = [], []
    def exchange(payload, *, timeout_seconds):
        sent.append((payload, timeout_seconds))
        return {"synthetic_response": True}
    worker._request_payload = exchange
    actual_dispatch = evaluation._spatial_inference_module
    def dispatch(protocol):
        routes.append(protocol)
        return actual_dispatch(protocol)
    monkeypatch.setattr(evaluation, "_spatial_inference_module", dispatch)
    assert worker.request_spatial(request, timeout_seconds=3) == {"synthetic_response": True}
    assert sent == [(request, 3)] and routes == [worker.target_protocol]
    with pytest.raises(ValueError, match="explicit spatial"):
        worker.request_visual(envelope, timeout_seconds=3)
    with pytest.raises(ValueError, match="visual worker"):
        worker.request("SYNTHETIC_no_image_file", prompt, timeout_seconds=3)
    assert len(sent) == 1
    worker.target_protocol = "native"
    with pytest.raises(ValueError, match="legacy worker"):
        worker.request_spatial(request, timeout_seconds=3)
    assert len(sent) == 1
