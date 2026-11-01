"""Synthetic worker/transport spies: no HF, actual data, model or processor use."""
from __future__ import annotations

import copy
import json
import sys
import threading
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import test_week5_visual_pointer as fixtures
import test_week5_visual_pointer_input as input_fixtures
from PIL import Image

from gui_agent.finetuning import evaluation as ev
from gui_agent.finetuning import visual_pointer_inference as inference
from gui_agent.finetuning import visual_pointer_input as vi
from gui_agent.finetuning.candidate_grounding import center
from gui_agent.finetuning.pointer_protocol import selection_binding
from gui_agent.finetuning.visual_pointer_prompts import DEDUP_PROMPT_VERSION

font_asset = fixtures.font_asset


@pytest.fixture
def asset(tmp_path, font_asset):
    return input_fixtures.Asset(tmp_path, font_asset)


def config_for(asset):
    fields = {
        "visual_asset_manifest_path": str(asset.registry.manifest_path),
        "visual_asset_manifest_sha256": asset.registry.manifest_sha256,
        "visual_original_root": str(asset.registry.original_root),
        "visual_asset_root": str(asset.registry.asset_root),
    }
    training = {"input_image_protocol": vi.PROTOCOL, "target_protocol": "candidate_compact",
                "prompt_variant": "conditional_actor", **fields}
    if "visual_prompt_version" in asset.row:
        training["visual_prompt_version"] = asset.row["visual_prompt_version"]
        fields["visual_prompt_version"] = asset.row["visual_prompt_version"]
    return ev.HFConfig(training, input_image_protocol=vi.PROTOCOL, **fields)


def ready_for(asset):
    ready = {"kind": "ready", "input_image_protocol": vi.PROTOCOL,
             "visual_asset_registry": inference.registry_identity(asset.registry)}
    if "visual_prompt_version" in asset.row:
        ready["visual_prompt_version"] = asset.row["visual_prompt_version"]
    return ready


class SpyWorker:
    def __init__(self, asset, *, generation=None, mutation=None, ack_mutation=None, status="ok"):
        self.asset, self.ready = asset, ready_for(asset)
        self.generation = generation or '[{"action_type":"click","candidate":1,"dx":3,"dy":-2}]'
        self.mutation, self.ack_mutation, self.status = mutation, ack_mutation, status
        self.requests = []

    def request(self, *_args, **_kwargs):
        pytest.fail("visual bridge used the native request")

    def request_visual(self, envelope, **_kwargs):
        self.requests.append(copy.deepcopy(envelope))
        descriptor = inference.validate_visual_request(envelope, registry=self.asset.registry)
        ack = inference.input_ack(descriptor, self.asset.registry)
        if self.ack_mutation:
            self.ack_mutation(ack)
        if self.mutation:
            self.mutation()
        return {"generation": self.generation, "status": self.status, "input_ack": ack,
                "duration_ms": 1.0, "rendered_prompt": "synthetic rendered prefix"}


def invoke(worker, asset, **kwargs):
    return inference.request_visual_pointer_actions(
        worker, asset.row, registry=asset.registry, timeout_seconds=1, **kwargs)


def test_versioned_envelope_ack_config_and_native_whole_mapping(asset):
    old_png = asset.model.read_bytes()
    input_fixtures.dedup_asset(asset)
    config = config_for(asset)
    assert config.visual_prompt_version == config.training_config["visual_prompt_version"] == DEDUP_PROMPT_VERSION
    descriptor, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    assert envelope["visual_prompt_version"] == envelope["visible_sample"]["visual_prompt_version"] == descriptor.visual_prompt_version
    result = invoke(SpyWorker(asset), asset)
    x, y = center(asset.row["candidate_snapshot"]["candidates"][0]["bbox"])
    assert result["mapped_actions"] == [{"action_type": "click", "x": x + 3, "y": y - 2}]
    assert result["input_ack"]["visual_prompt_version"] == DEDUP_PROMPT_VERSION
    assert result["input_ack_valid"] and asset.model.read_bytes() == old_png


@pytest.mark.parametrize("change", ["envelope_missing", "visible_missing", "unknown", "null", "prompt_injected"])
def test_request_cannot_downgrade_or_inject_approved_versioned_prompt(asset, change):
    input_fixtures.dedup_asset(asset)
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    if change == "envelope_missing":
        envelope.pop("visual_prompt_version")
    elif change == "visible_missing":
        envelope["visible_sample"].pop("visual_prompt_version")
    elif change == "unknown":
        envelope["visual_prompt_version"] = "arbitrary"
    elif change == "null":
        envelope["visual_prompt_version"] = None
    else:
        import hashlib

        envelope["prompt"] += " Injected instructions"
        envelope["prompt_sha256"] = hashlib.sha256(envelope["prompt"].encode()).hexdigest()
    with pytest.raises(ValueError):
        inference.validate_visual_request(envelope, registry=asset.registry)


@pytest.mark.parametrize("change", ["ready_missing", "ack_missing", "ack_unknown", "request_mutation"])
def test_bridge_rejects_version_change_without_native_mapping(asset, change):
    input_fixtures.dedup_asset(asset)
    worker = SpyWorker(asset)
    if change == "ready_missing":
        worker.ready.pop("visual_prompt_version")
        with pytest.raises(ValueError):
            invoke(worker, asset)
        assert not worker.requests
        return
    if change == "ack_missing":
        worker.ack_mutation = lambda ack: ack.pop("visual_prompt_version")
    elif change == "ack_unknown":
        worker.ack_mutation = lambda ack: ack.update(visual_prompt_version="unknown")
    else:
        worker.mutation = lambda: asset.row.pop("visual_prompt_version")
    result = invoke(worker, asset)
    assert result["mapped_actions"] is None and result["mapping_error"] and not result["input_ack_valid"]


def test_new_version_rejected_by_native_and_mixed_hf_training_config(asset):
    input_fixtures.dedup_asset(asset)
    with pytest.raises(ValueError):
        ev.HFConfig({}, visual_prompt_version=DEDUP_PROMPT_VERSION)
    config = config_for(asset)
    config_dict = asdict(config)
    config_dict["visual_prompt_version"] = None
    with pytest.raises(ValueError):
        ev.HFConfig(**config_dict)


def test_child_uses_only_approved_short_prompt_and_ack_version(asset, monkeypatch):
    input_fixtures.dedup_asset(asset)
    state = install_child_spies(monkeypatch)
    descriptor, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    responses = child_responses(config_for(asset), [envelope])
    assert responses[0]["visual_prompt_version"] == DEDUP_PROMPT_VERSION
    assert responses[1]["status"] == "ok" and responses[1]["input_ack"] == inference.input_ack(descriptor, asset.registry)
    assert state.processor_calls == state.generate_calls == 1 and state.texts == [asset.prompt]


def test_child_configured_legacy_refuses_new_manifest_prompt_before_processor(asset, monkeypatch):
    input_fixtures.dedup_asset(asset)
    state = install_child_spies(monkeypatch)
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    config_dict = asdict(config_for(asset))
    config_dict["visual_prompt_version"] = None
    config_dict["training_config"].pop("visual_prompt_version")
    responses = child_responses(ev.HFConfig(**config_dict), [envelope])
    assert responses[1]["status"] == "error" and "input_ack" not in responses[1]
    assert state.processor_calls == state.generate_calls == 0


def test_default_native_and_visual_config_spawn_roundtrip(asset):
    assert ev.HFConfig({}).input_image_protocol == "native"
    config = config_for(asset)
    assert ev.HFConfig(**asdict(config)) == config
    assert inference.registry_identity(inference.registry_from_config(config)) == inference.registry_identity(asset.registry)


@pytest.mark.parametrize("change", [
    {"input_image_protocol": "unknown"},
    {"input_image_protocol": "native"},
    {"visual_asset_manifest_path": None},
    {"visual_asset_manifest_path": "relative.json"},
    {"visual_asset_manifest_sha256": "A" * 64},
    {"output_constraint": "json_actions"},
])
def test_mixed_missing_unknown_configuration_rejected(asset, change):
    payload = asdict(config_for(asset))
    payload.update(change)
    with pytest.raises(ValueError):
        ev.HFConfig(**payload)


@pytest.mark.parametrize("field,value", [
    ("input_image_protocol", "native"), ("target_protocol", "native"),
    ("prompt_variant", "original"), ("visual_asset_manifest_sha256", "0" * 64),
])
def test_worker_and_training_contract_must_agree(asset, field, value):
    payload = asdict(config_for(asset))
    payload["training_config"][field] = value
    with pytest.raises(ValueError):
        ev.HFConfig(**payload)


def test_native_registry_is_not_silently_ignored(asset):
    with pytest.raises(ValueError, match="native inference"):
        ev.HFConfig({}, visual_asset_root=str(asset.asset_root))


def test_native_config_does_not_require_visual_validator_import(monkeypatch):
    import builtins
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.endswith("visual_pointer_input"), "native configuration imported the visual validator"
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert ev.HFConfig({}).input_image_protocol == "native"


@pytest.mark.parametrize("policy", ["spatial_v1", "control_balanced_v2", "query_relations_v3"])
def test_visual_native_roundtrip_all_policies_without_label_reads(tmp_path, font_asset, policy):
    asset = input_fixtures.Asset(tmp_path, font_asset, policy=policy)
    original = copy.deepcopy(asset.row)
    visible = fixtures.VisibleOnly(asset.row)
    visible.update(actions=object(), candidate_target=object(), candidate_label_evidence=object(),
                   sample_id=object(), review=object(), targets=object(), action_annotations=object())
    worker = SpyWorker(asset)
    result = inference.request_visual_pointer_actions(worker, visible, registry=asset.registry, timeout_seconds=1)
    assert result["input_ack_valid"] and result["mapping_error"] is None
    first = inference.visible_sample(original)["candidate_snapshot"]["candidates"][0]
    x, y = center(first["bbox"])
    assert result["mapped_actions"] == [{"action_type": "click", "x": x + 3, "y": y - 2}]
    envelope = worker.requests[0]
    assert set(envelope["visible_sample"]) == inference.VISIBLE_FIELDS
    assert envelope["prompt"] == asset.prompt and "actions" not in envelope["visible_sample"]
    assert asset.row["image_path"] == original["image_path"]


def test_complete_drag_scroll_unicode_and_nonmouse_fields_preserved(asset):
    actions = [
        {"action_type": "drag", "start_candidate": 1, "start_dx": 1, "start_dy": -2,
         "end_candidate": 2, "end_dx": -3, "end_dy": 4},
        {"action_type": "scroll", "candidate": 2, "dx": 0, "dy": 0, "scroll_amount": -7},
        {"action_type": "type_text", "text": "原文 Ω\n"},
        {"action_type": "hotkey", "keys": ["ctrl", "a"]},
        {"action_type": "key_press", "key": "enter"}, {"action_type": "wait", "duration": 1.25},
    ]
    result = invoke(SpyWorker(asset, generation=json.dumps(actions, ensure_ascii=False)), asset)
    assert result["mapping_error"] is None
    assert result["mapped_actions"][2:] == actions[2:]
    assert result["mapped_actions"][0] == {"action_type": "drag", "start": {"x": 121, "y": 138},
                                          "end": {"x": 397, "y": 304}}
    assert result["mapped_actions"][1] == {"action_type": "scroll", "x": 400, "y": 300, "scroll_amount": -7}


@pytest.mark.parametrize("kind", ["missing", "registry", "context", "png"])
def test_missing_or_mismatched_ack_keeps_raw_and_refuses_mapping(asset, kind):
    def change(ack):
        if kind == "missing":
            ack.clear()
        elif kind == "registry":
            ack["visual_asset_registry"]["manifest_sha256"] = "0" * 64
        elif kind == "context":
            ack["input_evidence"]["context_sha256"] = "0" * 64
        else:
            ack["input_evidence"]["model_png_sha256"] = "0" * 64
    worker = SpyWorker(asset, ack_mutation=change)
    result = invoke(worker, asset)
    assert result["raw_reply"]["generation"] == worker.generation
    assert not result["input_ack_valid"] and result["mapped_actions"] is None
    assert "acknowledgement" in result["mapping_error"]


@pytest.mark.parametrize("kind", ["goal", "step", "geometry", "policy", "replacement", "image", "png", "manifest"])
def test_mutation_during_request_fail_closed_all_policy_binding(asset, kind):
    def change():
        if kind == "goal":
            asset.row["instruction"] += " changed"
        elif kind == "step":
            asset.row["actor_step"] += " changed"
        elif kind == "geometry":
            asset.row["candidate_snapshot"]["candidates"][0]["bbox"]["left"] += 1
        elif kind == "policy":
            asset.row["candidate_snapshot"]["compact_policy"] = "spatial_v1"
        elif kind == "replacement":
            asset.row["candidate_snapshot"] = copy.deepcopy(asset.row["candidate_snapshot"])
            asset.row["candidate_snapshot"]["candidates"].reverse()
        elif kind == "image":
            asset.row["image_path"] = asset.row["model_image_path"]
        elif kind == "png":
            asset.model.write_bytes(b"stale model raster")
        else:
            asset.manifest.write_text("{}", encoding="utf-8")
    worker = SpyWorker(asset, mutation=change)
    result = invoke(worker, asset)
    assert result["mapped_actions"] is None and result["mapping_error"]
    assert result["raw_reply"]["generation"] == worker.generation


@pytest.mark.parametrize("change", ["top_label", "nested_label", "missing", "prompt", "native", "extra"])
def test_envelope_is_closed_visible_only_and_child_recomputes_prompt(asset, change):
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    if change == "top_label":
        envelope["actions"] = object()
    elif change == "nested_label":
        envelope["visible_sample"]["candidate_snapshot"]["candidates"][0]["actions"] = object()
    elif change == "missing":
        del envelope["visible_sample"]["model_png_sha256"]
    elif change == "prompt":
        import hashlib
        envelope["prompt"] += " forged"
        envelope["prompt_sha256"] = hashlib.sha256(envelope["prompt"].encode()).hexdigest()
    elif change == "native":
        envelope["visible_sample"]["input_image_protocol"] = "native"
    else:
        envelope["visible_sample"]["sample_id"] = object()
    with pytest.raises((ValueError, TypeError)):
        inference.validate_visual_request(envelope, registry=asset.registry)


def install_child_spies(monkeypatch, *, mutation=None, processor_mutation=None, generation_raster_mutation=False,
                        raw='[{"action_type":"wait","duration":1}]'):
    state = SimpleNamespace(processor_calls=0, generate_calls=0, images=[], raw_images=[], texts=[], seeds=[], loaded=0)
    torch = SimpleNamespace(manual_seed=state.seeds.append, cuda=SimpleNamespace(is_available=lambda: False),
                            version=SimpleNamespace(cuda=None), inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)

    class Inputs(dict):
        def to(self, _device):
            return self

    class Suffix:
        shape = (1, 1)
        def __getitem__(self, _index):
            return 99

    class Generated:
        def __getitem__(self, _index):
            return Suffix()

    class Model:
        device = "synthetic-only"
        generation_config = SimpleNamespace(eos_token_id=99)
        def eval(self):
            pass
        def generate(self, **_kwargs):
            state.generate_calls += 1
            if mutation:
                mutation()
            if generation_raster_mutation:
                state.raw_images[-1].putpixel((0, 0), (1, 2, 3))
            return Generated()

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            state.texts.append(messages[0]["content"][1]["text"])
            return "synthetic rendered: " + state.texts[-1]
        def __call__(self, **kwargs):
            state.processor_calls += 1
            assert kwargs["padding"] is True and kwargs["return_tensors"] == "pt"
            state.images.append(kwargs["images"][0].copy())
            state.raw_images.append(kwargs["images"][0])
            if processor_mutation:
                processor_mutation(kwargs["images"][0])
            return Inputs(input_ids=SimpleNamespace(shape=(1, 1)))
        def batch_decode(self, *_args, **_kwargs):
            return [raw]

    def load(*_args, **_kwargs):
        state.loaded += 1
        return Model(), Processor()
    monkeypatch.setitem(sys.modules, "gui_agent.finetuning.training", SimpleNamespace(
        TrainingConfig=SimpleNamespace(from_dict=lambda data: data), load_model_and_processor=load))
    monkeypatch.setattr(ev, "runtime_versions", lambda: {"synthetic": "no-HF"})
    return state


def child_responses(config, requests):
    responses = []
    pending = iter([*requests, None])
    class Connection:
        closed = False
        def recv(self):
            return next(pending)
        def send(self, value):
            responses.append(value)
        def close(self):
            self.closed = True
    connection = Connection()
    ev._hf_worker(connection, config, None)
    assert connection.closed
    return responses


def test_child_itself_opens_model_png_and_acknowledges_exact_input(asset, monkeypatch):
    state = install_child_spies(monkeypatch)
    descriptor, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    responses = child_responses(config_for(asset), [envelope])
    assert responses[0]["input_image_protocol"] == vi.PROTOCOL
    assert responses[1]["status"] == "ok"
    assert responses[1]["input_ack"] == inference.input_ack(descriptor, asset.registry)
    assert state.processor_calls == state.generate_calls == 1 and state.texts == [asset.prompt]
    with Image.open(asset.model) as model:
        assert state.images[0].mode == "RGB" and state.images[0].tobytes() == model.convert("RGB").tobytes()
    with Image.open(asset.row["image_path"]) as native:
        assert state.images[0].tobytes() != native.convert("RGB").tobytes()


@pytest.mark.parametrize("protocol", ["native", vi.PROTOCOL])
def test_child_native_and_visual_requests_are_mutually_rejected(asset, monkeypatch, protocol):
    state = install_child_spies(monkeypatch)
    _, visual = inference.prepare_visual_request(asset.row, registry=asset.registry)
    native = {"image_path": asset.row["image_path"], "prompt": "synthetic native"}
    config, request = (ev.HFConfig({}), visual) if protocol == "native" else (config_for(asset), native)
    responses = child_responses(config, [request])
    assert responses[1]["status"] == "error"
    assert state.processor_calls == state.generate_calls == 0


def test_child_catches_post_generation_staleness_preserving_raw(asset, monkeypatch):
    state = install_child_spies(monkeypatch, mutation=lambda: asset.model.write_bytes(b"mutated"))
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    responses = child_responses(config_for(asset), [envelope])
    assert state.generate_calls == 1
    assert responses[1]["status"] == "error" and "SHA256" in responses[1]["error"]
    assert responses[1]["generation"] == '[{"action_type":"wait","duration":1}]'
    assert "input_ack" not in responses[1]


def test_child_checks_after_processor_before_any_generate(asset, monkeypatch):
    state = install_child_spies(monkeypatch, processor_mutation=lambda _image: asset.sidecar.write_bytes(b"stale"))
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    responses = child_responses(config_for(asset), [envelope])
    assert state.processor_calls == 1 and state.generate_calls == 0
    assert responses[1]["status"] == "error" and responses[1]["generation"] == ""


def test_child_processor_inplace_raster_mutation_prevents_generate_and_ack(asset, monkeypatch):
    state = install_child_spies(monkeypatch, processor_mutation=lambda image: image.putpixel((0, 0), (1, 2, 3)))
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    responses = child_responses(config_for(asset), [envelope])
    assert state.processor_calls == 1 and state.generate_calls == 0
    assert responses[1]["status"] == "error" and "immutable RGB" in responses[1]["error"]
    assert responses[1]["generation"] == "" and "input_ack" not in responses[1]


def test_child_post_generation_inplace_raster_mutation_preserves_raw_without_ack(asset, monkeypatch):
    state = install_child_spies(monkeypatch, generation_raster_mutation=True)
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    responses = child_responses(config_for(asset), [envelope])
    assert state.generate_calls == 1
    assert responses[1]["status"] == "error" and "immutable RGB" in responses[1]["error"]
    assert responses[1]["generation"] == '[{"action_type":"wait","duration":1}]'
    assert "input_ack" not in responses[1]


def test_child_checks_asset_itself_even_if_parent_envelope_was_valid(asset, monkeypatch):
    state = install_child_spies(monkeypatch)
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    asset.sidecar.write_text("{}", encoding="utf-8")
    responses = child_responses(config_for(asset), [envelope])
    assert responses[1]["status"] == "error" and state.generate_calls == state.processor_calls == 0


def test_child_revalidates_mutable_config_before_loader(asset, monkeypatch):
    state = install_child_spies(monkeypatch)
    config = config_for(asset)
    config.training_config["input_image_protocol"] = "native"
    responses = child_responses(config, [])
    assert responses[0]["kind"] == "load_error" and state.loaded == 0


def startup_spies(monkeypatch, ready):
    state = SimpleNamespace(sent=[], terminated=False, parent_closed=False, child_closed=False)
    class Parent:
        def send(self, value):
            state.sent.append(value)
        def recv(self):
            return copy.deepcopy(ready)
        def close(self):
            state.parent_closed = True
    class Child:
        def close(self):
            state.child_closed = True
    class Process:
        pid = None
        alive = False
        def start(self):
            self.pid, self.alive = 100, True
        def is_alive(self):
            return self.alive
        def terminate(self):
            state.terminated, self.alive = True, False
        def join(self, **_kwargs):
            pass
    process = Process()
    monkeypatch.setattr(ev.multiprocessing, "get_context", lambda _: SimpleNamespace(
        Pipe=lambda: (Parent(), Child()), Process=lambda **_kwargs: process))
    return state


@pytest.mark.parametrize("change", ["missing", "native", "registry"])
def test_spawn_ready_requires_explicit_visual_protocol_and_trust(asset, monkeypatch, change):
    ready = ready_for(asset)
    if change == "missing":
        ready.pop("input_image_protocol")
    elif change == "native":
        ready["input_image_protocol"] = "native"
    else:
        ready["visual_asset_registry"]["manifest_sha256"] = "0" * 64
    state = startup_spies(monkeypatch, ready)
    with pytest.raises(RuntimeError, match="differs"):
        ev.HFWorker(config_for(asset), load_timeout_seconds=1)
    assert state.terminated and state.parent_closed and state.child_closed


def test_worker_mutual_rejection_occurs_before_send_and_request_count(asset, monkeypatch):
    state = startup_spies(monkeypatch, ready_for(asset))
    worker = ev.HFWorker(config_for(asset), load_timeout_seconds=1)
    with pytest.raises(ValueError, match="native request"):
        worker.request(asset.row["image_path"], "native", timeout_seconds=1)
    assert worker.request_count == 0 and state.sent == []
    worker.close()
    state = startup_spies(monkeypatch, {"kind": "ready", "input_image_protocol": "native"})
    native = ev.HFWorker(ev.HFConfig({}), load_timeout_seconds=1)
    with pytest.raises(ValueError, match="visual request"):
        native.request_visual({}, timeout_seconds=1)
    assert native.request_count == 0 and state.sent == []
    native.close()


def test_visual_transport_has_hard_timeout_and_preserves_counter(asset):
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    release = threading.Event()
    sent = []
    class Connection:
        def send(self, value):
            sent.append(value)
        def recv(self):
            release.wait(1)
            raise EOFError("closed")
        def close(self):
            release.set()
    class Process:
        pid = 1
        alive = True
        def is_alive(self):
            return self.alive
        def terminate(self):
            self.alive = False
        def join(self, **_kwargs):
            pass
    worker = ev.HFWorker.__new__(ev.HFWorker)
    worker.connection, worker.process = Connection(), Process()
    worker.request_count, worker.input_image_protocol = 0, vi.PROTOCOL
    result = worker.request_visual(envelope, timeout_seconds=.01)
    assert result["status"] == "timeout" and worker.request_count == 1 and not worker.process.alive
    assert sent == [envelope]


@pytest.mark.parametrize("raw", ["```json\n[]\n```", "[]", '[{"action_type":"click","candidate":3,"dx":0,"dy":0}]'])
def test_raw_invalid_reply_not_repaired_or_fallback(asset, raw):
    result = invoke(SpyWorker(asset, generation=raw), asset)
    assert result["raw_reply"]["generation"] == raw and result["mapped_actions"] is None
    assert result["mapping_error"]
    assert result["input_ack_valid"]  # Raster acknowledgement is independent of action quality.


def test_training_and_inference_prefix_factory_agree_without_targets(asset):
    from gui_agent.finetuning.visual_pointer import build_visual_pointer_messages
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    assert envelope["prompt"] == build_visual_pointer_messages(asset.row)[0]["content"][1]["text"]


def test_current_selection_evidence_mutation_refused_before_request(asset):
    asset.row["candidate_selection_evidence"] = selection_binding(
        asset.row["candidate_snapshot"], instruction=asset.row["instruction"], actor_step="other")
    worker = SpyWorker(asset)
    with pytest.raises(ValueError, match="binding differs"):
        invoke(worker, asset)
    assert worker.requests == []
