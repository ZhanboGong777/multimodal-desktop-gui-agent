"""CPU toy model/transport only. No pretrained model, processor, image or UI."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gui_agent.finetuning import native_spatial_inference as inf

PACKAGE = Path(inf.__file__).parent

core = inf.core


def decoder():
    return SimpleNamespace(pieces={**{byte + 10: bytes([byte]) for byte in range(256)},
                                  500: b"<|im_end|>"}, eos_id=500)


def byte_ids(text):
    return tuple(byte + 10 for byte in text.encode("utf8"))


def config(**updates):
    fields = dict(  # noqa: C408 - keyword configuration fixture is easier to scan
        training_config={"target_protocol": core.TARGET_PROTOCOL,
                         "input_image_protocol": inf.INPUT_PROTOCOL, "visual_prompt_version": inf.PROMPT_VERSION,
                         "prompt_variant": "conditional_actor", "min_pixels": 100352,
                         "max_pixels": 301056, "max_seq_length": 1536, "processor_use_fast": True},
        input_image_protocol=inf.INPUT_PROTOCOL, visual_prompt_version=inf.PROMPT_VERSION, target_version=core.TARGET_VERSION,
        max_new_tokens=256, output_constraint="none", spatial_head_artifact_dir=str(PACKAGE.resolve()),
        spatial_head_role="initial", spatial_head_manifest_sha256="1" * 64,
        spatial_expected_bindings={"prompt_version": inf.PROMPT_VERSION, "prompt_sha256": "2" * 64,
                                  "tokenizer_json_sha256": "3" * 64, "core_source_sha256": "4" * 64,
                                  "training_source_sha256": "5" * 64, "base_model_config_sha256": "6" * 64,
                                  "eos_token_id": 500})
    fields.update(updates)
    return SimpleNamespace(**fields)


@pytest.fixture
def torch_cpu():
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    return torch


def toy_qwen(torch):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.norm = torch.nn.LayerNorm(2048)
            self.config = SimpleNamespace(model_type="qwen2_5_vl", text_config=SimpleNamespace(hidden_size=2048))
            self.generation_config = SimpleNamespace(eos_token_id=500, to_dict=lambda: {"eos_token_id": 500})
            self.forward_calls, self.generate_calls, self.kwargs = 0, 0, None
            self.suffix, self.cache_mutation, self.consume_mutation = (), None, None

        def forward(self, input_ids, *, cache_position, use_cache, output_hidden_states, **kwargs):
            assert use_cache is True and output_hidden_states is False and "labels" not in kwargs
            self.forward_calls += 1
            states = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2048)
            return self.model.language_model.norm(states)

        def generate(self, *, input_ids, **kwargs):
            self.generate_calls += 1
            self.kwargs = kwargs
            assert kwargs == core.generation_policy(eos_id=500)
            self(input_ids=input_ids, cache_position=torch.arange(input_ids.shape[1]),
                 use_cache=True, output_hidden_states=False)
            seq = input_ids[0].tolist()
            for offset, token in enumerate(self.suffix):
                seq.append(token)
                if offset + 1 < len(self.suffix):
                    pos = len(seq) - 1
                    position = self.cache_mutation(pos) if self.cache_mutation else pos
                    consumed = self.consume_mutation(token) if self.consume_mutation else token
                    self(input_ids=torch.tensor([[consumed]], dtype=torch.long),
                         cache_position=torch.tensor([position]), use_cache=True,
                         output_hidden_states=False)
            return torch.tensor([seq], dtype=torch.long)
    Toy.__name__ = inf.MODEL_CLASS[1]
    Toy.__module__ = inf.MODEL_CLASS[0]
    return Toy().eval()


def generated(torch, raw=None, suffix=None):
    raw = raw or core.canonical([{"action_type": "click", "point": "native_head"},
                                {"action_type": "type_text", "text": 'native_head 中文 019\n "x"'}])
    model = toy_qwen(torch)
    model.suffix = suffix if suffix is not None else byte_ids(raw) + (500,)
    head = core.make_spatial_head(seed=11).eval()
    reply = inf.generate_joint_output(model, head, {"input_ids": torch.tensor([[1, 2, 3]])},
                                     decoder(), native_width=101, native_height=51)
    return model, reply


def test_cpu_collection_does_not_import_torch_transformers_pil():
    # Evaluate an isolated import via a guarded import spy, no Python subprocess.
    import builtins
    previous = builtins.__import__
    accessed = []
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "PIL"}
        accessed.append(name)
        return previous(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location("gui_agent.finetuning.lazy_probe", PACKAGE / "native_spatial_inference.py")
    module = importlib.util.module_from_spec(spec)
    builtins.__import__ = guarded
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = previous
    assert accessed and module.MAX_NEW_TOKENS == 256


@pytest.mark.parametrize("updates", [
    {"max_new_tokens": 255}, {"max_new_tokens": True}, {"output_constraint": "json_actions"},
    {"spatial_head_role": None}, {"spatial_head_role": "seed"},
    {"spatial_head_artifact_dir": "relative"}, {"spatial_head_manifest_sha256": "A" * 64},
    {"visual_prompt_version": "week5.visual-pointer-prompt.gallery.v4"},
    {"input_image_protocol": "native"}, {"spatial_expected_bindings": {}},
])
def test_explicit_config_refuses_stale_missing_or_constraint(updates):
    with pytest.raises(ValueError):
        inf.validate_spatial_config(config(**updates))


def test_training_and_worker_head_binding_must_agree():
    good = config()
    inf.validate_spatial_config(good)
    good.training_config["spatial_head_role"] = "final"
    with pytest.raises(ValueError, match="authority differ"):
        inf.validate_spatial_config(good)


def test_one_greedy_generation_and_cache_consumed_own_marker_state(torch_cpu):
    model, reply = generated(torch_cpu)
    assert model.generate_calls == reply["generation_calls"] == 1
    assert model.forward_calls == reply["generation_forward_calls"] == len(reply["generated_token_ids"])
    assert reply["repair_forward_calls"] == 0 and reply["status"] == "ok"
    actions = inf.validate_joint_reply(reply, decoder(), native_width=101, native_height=51)
    assert actions[0]["x"] in range(101) and actions[0]["y"] in range(51)
    assert actions[1]["text"] == 'native_head 中文 019\n "x"'
    assert len(reply["marker_token_positions"]) == len(reply["raw_spatial_output"]) == 1
    assert reply["marker_token_positions"][0] in [item["position"] for item in reply["consumed_token_ids"]]


def test_drag_both_endpoints_and_nonmouse_literal_preserved(torch_cpu):
    raw = '[{"action_type":"drag","start_point":"native_head","end_point":"native_head"},' \
          '{"action_type":"hotkey","keys":["CTRL","A"]},{"action_type":"wait","duration":0.25}]'
    model, reply = generated(torch_cpu, raw=raw)
    assert reply["language_generation"] == raw and reply["status"] == "ok"
    actions = inf.validate_joint_reply(reply, decoder(), native_width=101, native_height=51)
    assert set(actions[0]) == {"action_type", "start", "end"}
    assert actions[1:] == core.strict_json(raw)[1:]
    assert len(reply["raw_spatial_output"]) == 2 and model.generate_calls == 1


def test_unconsumed_final_marker_fails_without_extra_model_forward(torch_cpu):
    raw = '[{"action_type":"click","point":"native_head"}]'
    special = decoder()
    special.pieces[501] = raw.encode()
    model = toy_qwen(torch_cpu)
    model.suffix = (501,)
    reply = inf.generate_joint_output(model, core.make_spatial_head().eval(),
              {"input_ids": torch_cpu.tensor([[1, 2, 3]])}, special, native_width=101, native_height=51)
    assert reply["status"] == "error" and "not consumed" in reply["error"]
    assert reply["language_generation"] == raw and reply["marker_token_positions"] == [3]
    assert reply["raw_spatial_output"] == [None] and reply["mapped_actions"] is None
    assert model.forward_calls == model.generate_calls == 1 and reply["repair_forward_calls"] == 0


@pytest.mark.parametrize("bad", ["cache", "consumed_id"])
def test_cache_position_and_consumed_ids_fail_whole_output(torch_cpu, bad):
    model = toy_qwen(torch_cpu)
    model.suffix = byte_ids('[{"action_type":"click","point":"native_head"}]') + (500,)
    if bad == "cache":
        model.cache_mutation = lambda pos: pos + 1
    else:
        model.consume_mutation = lambda token: token + 1
    reply = inf.generate_joint_output(model, core.make_spatial_head().eval(),
              {"input_ids": torch_cpu.tensor([[1, 2, 3]])}, decoder(), native_width=101, native_height=51)
    assert reply["status"] == "error" and reply["mapped_actions"] is None
    assert model.generate_calls == 1 and reply["repair_forward_calls"] == 0
    assert "cache_position" in reply["error"] if bad == "cache" else "consumed cached token differs" in reply["error"]


@pytest.mark.parametrize("mutation", [
    lambda reply: reply.update(language_generation='[{"action_type":"click","x":10,"y":20}]'),
    lambda reply: reply.update(raw_spatial_output=[[float("nan"), 0.0]]),
    lambda reply: reply.update(raw_spatial_output=[[0.0, 2.0]]),
    lambda reply: reply.update(marker_token_positions=[3]),
    lambda reply: reply.update(marker_byte_spans=[]),
    lambda reply: reply.update(consumed_token_ids=[]),
    lambda reply: reply.update(repair_forward_calls=1),
    lambda reply: reply.update(generation_calls=2),
    lambda reply: reply.update(strict_joint_output_valid=False),
    lambda reply: reply.update(mapped_actions=[{"action_type":"click","x":1,"y":1}]),
    lambda reply: reply.update(target_protocol="candidate_compact"),
])
def test_raw_joint_gate_refuses_even_when_native_serializer_json_valid(torch_cpu, mutation):
    _, reply = generated(torch_cpu)
    assert json.loads(reply["generation"])
    mutation(reply)
    with pytest.raises((ValueError, TypeError)):
        inf.validate_joint_reply(reply, decoder(), native_width=101, native_height=51)


def test_parent_bridge_cannot_use_candidate_or_gt_and_never_corrects(torch_cpu):
    _, reply = generated(torch_cpu)
    ack = {"schema": inf.ACK_SCHEMA, "authority": "test-only"}
    reply["input_ack"] = ack
    accepted = inf.accept_spatial_reply(reply, decoder(), expected_ack=ack, native_width=101, native_height=51)
    assert accepted["strict_joint_output_valid"] and accepted["mapped_actions"] == reply["mapped_actions"]
    assert accepted["raw_reply"] is reply
    assert {"candidate_snapshot", "actions", "target_bbox", "candidate_target"}.isdisjoint(
        inf.accept_spatial_reply.__code__.co_varnames)
    reply["input_ack"] = {"schema": "old_pointer_ack"}
    refused = inf.accept_spatial_reply(reply, decoder(), expected_ack=ack, native_width=101, native_height=51)
    assert refused["mapped_actions"] is None and refused["raw_reply"] is reply


@pytest.mark.parametrize("fault", ["class", "hidden", "path", "compile", "assistant"])
def test_model_known_identity_shape_path_generation_guard(torch_cpu, fault):
    model = toy_qwen(torch_cpu)
    if fault == "class":
        model = SimpleNamespace()
    elif fault == "hidden":
        model.config.text_config.hidden_size = 4096
    elif fault == "path":
        del model.model.language_model.norm
    elif fault == "compile":
        model.generation_config.compile_config = object()
    else:
        model.generation_config.prompt_lookup_num_tokens = 2
    with pytest.raises(ValueError):
        inf.validate_model_contract(model)


def test_nonprompt_peft_resolves_exact_same_base_norm(torch_cpu):
    from contextlib import contextmanager

    model = toy_qwen(torch_cpu)
    class Wrapper:
        generation_config = model.generation_config
        training = False
        def __init__(self):
            self.peft_config = {"default": SimpleNamespace(is_prompt_learning=False, peft_type="LORA")}
            self.active_peft_config = self.peft_config["default"]
            self.hook_entries, self.hook_exits, self.wrapper_calls = 0, 0, 0
        def get_base_model(self):
            return model
        def generate(self, **kwargs):
            self.wrapper_calls += 1
            prepared = {"input_ids": kwargs["input_ids"], "use_cache": True,
                        "output_hidden_states": False,
                        "cache_position": torch_cpu.arange(kwargs["input_ids"].shape[1])}
            prepared.pop("cache_position")
            return model(**prepared)
        @contextmanager
        def _enable_peft_forward_hooks(self, **kwargs):
            self.hook_entries += 1
            try:
                yield
            finally:
                self.hook_exits += 1
    Wrapper.__name__ = inf.PEFT_CLASS[1]
    Wrapper.__module__ = inf.PEFT_CLASS[0]
    wrapped = Wrapper()
    assert inf.validate_model_contract(wrapped) is model.model.language_model.norm
    model.suffix = byte_ids('[{"action_type":"click","point":"native_head"}]') + (500,)
    head = core.make_spatial_head().eval()
    inputs = {"input_ids": torch_cpu.tensor([[1, 2, 3]])}
    reply = inf.generate_joint_output(wrapped, head, inputs, decoder(), native_width=101, native_height=51)
    assert reply["status"] == "ok" and reply["strict_joint_output_valid"] is True
    assert wrapped.wrapper_calls == 0 and wrapped.hook_entries == wrapped.hook_exits == model.generate_calls == 1
    model.cache_mutation = lambda pos: pos + 1
    failed = inf.generate_joint_output(wrapped, head, inputs, decoder(), native_width=101, native_height=51)
    assert failed["status"] == "error" and "cache_position" in failed["error"]
    assert failed["repair_forward_calls"] == 0 and wrapped.hook_entries == wrapped.hook_exits == 2
    wrapped.peft_config = {"default": SimpleNamespace(is_prompt_learning=True, peft_type="LORA")}
    wrapped.active_peft_config = wrapped.peft_config["default"]
    with pytest.raises(ValueError, match="nonprompt"):
        inf.validate_model_contract(wrapped)


def test_request_ready_ack_carry_new_head_target_prompt_authority():
    head, registry = {"role": "initial", "manifest_sha256": "1" * 64, "eos_token_id": 500}, {"manifest_sha256": "2" * 64}
    envelope = {"input_image_protocol": inf.INPUT_PROTOCOL, "visual_prompt_version": inf.PROMPT_VERSION}
    request = inf.bind_spatial_request(envelope, head_identity=head)
    assert inf.checked_spatial_request(request, head_identity=head) == envelope
    ready = {"kind": "ready", "input_image_protocol": inf.INPUT_PROTOCOL,
             "visual_asset_registry": registry, **inf.authority_fields(head)}
    inf.assert_spatial_ready(ready, head_identity=head, visual_registry_identity=registry)
    ack = inf.spatial_input_ack(envelope, head_identity=head)
    assert ack["target_protocol"] == core.TARGET_PROTOCOL and ack["spatial_head_artifact"] == head
    request["spatial_head_artifact"] = {"role": "final"}
    with pytest.raises(ValueError, match="authority"):
        inf.checked_spatial_request(request, head_identity=head)
    ready.pop("target_protocol")
    with pytest.raises(ValueError, match="authority"):
        inf.assert_spatial_ready(ready, head_identity=head, visual_registry_identity=registry)


def test_label_tensor_refused_before_generate(torch_cpu):
    model = toy_qwen(torch_cpu)
    with pytest.raises(ValueError, match="label-free"):
        inf.generate_joint_output(model, core.make_spatial_head().eval(),
           {"input_ids": torch_cpu.tensor([[1, 2]]), "labels": torch_cpu.tensor([[1, 2]])}, decoder(),
           native_width=101, native_height=51)
    assert model.generate_calls == model.forward_calls == 0


@pytest.fixture
def head_artifacts(tmp_path, torch_cpu):
    from gui_agent.finetuning import native_spatial_artifacts as artifacts
    from gui_agent.finetuning.visual_pointer_prompts import GALLERY_SPATIAL_HEADER

    base = tmp_path / "base"
    base.mkdir()
    base_config = {"model_type": "qwen2_5_vl", "hidden_size": 2048,
                   "architectures": [inf.MODEL_CLASS[1]]}
    (base / "config.json").write_text(json.dumps(base_config), encoding="utf8")
    tokenizer = {"model": {"type": "BPE", "vocab": {"a": 1}}, "decoder": {"type": "ByteLevel"},
                 "added_tokens": [{"id": 500, "content": "<|im_end|>"}]}
    (base / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf8")
    output = tmp_path / "heads"
    output.mkdir()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    bindings = {"base_model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "base_model_revision": "a" * 40,
                "base_model_config_sha256": digest(base / "config.json"),
                "prompt_version": inf.PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(GALLERY_SPATIAL_HEADER.encode("utf8")).hexdigest(),
                "eos_token_id": 500, "tokenizer_json_sha256": digest(base / "tokenizer.json"),
                "core_source_sha256": digest(core.__file__), "training_source_sha256": digest(PACKAGE / "training.py")}
    first, final = core.make_spatial_head(seed=4), core.make_spatial_head(seed=7)
    artifacts.save_spatial_head_artifact(first, output, role="initial", bindings=bindings)
    artifacts.save_spatial_head_artifact(final, output, role="final", bindings=bindings)
    settings = config(spatial_head_artifact_dir=str(output), spatial_expected_bindings=bindings,
                      spatial_head_manifest_sha256=digest(output / artifacts.MANIFEST_NAME))
    settings.training_config["base_model_path"] = str(base)
    return settings, first, final, artifacts


def test_strict_saved_initial_and_final_loader_never_regenerates_seed(head_artifacts, torch_cpu):
    settings, first, final, _ = head_artifacts
    initial_identity = inf.inspect_head(settings)
    loaded, identity = inf.load_resident_head(toy_qwen(torch_cpu), settings, expected_identity=initial_identity)
    assert identity == initial_identity and identity["role"] == "initial" and not loaded.training
    assert all(torch_cpu.equal(value, loaded.state_dict()[name]) for name, value in first.state_dict().items())
    assert {"spatial_head_initial.pt", "spatial_head_final.pt", "spatial_head_config.json",
            "spatial_head_manifest.json"} == set(identity["file_ledger"])
    settings.spatial_head_role = "final"
    final_identity = inf.inspect_head(settings)
    loaded_final, _ = inf.load_resident_head(toy_qwen(torch_cpu), settings, expected_identity=final_identity)
    assert all(torch_cpu.equal(value, loaded_final.state_dict()[name]) for name, value in final.state_dict().items())
    assert identity["parameter_sha256"] != final_identity["parameter_sha256"]


@pytest.mark.parametrize("fault", ["missing", "manifest", "config", "other_role", "source", "tokenizer"])
def test_missing_stale_artifact_and_actual_bindings_fail_closed(head_artifacts, fault):
    settings, _, _, artifacts = head_artifacts
    directory = Path(settings.spatial_head_artifact_dir)
    if fault == "missing":
        (directory / "spatial_head_initial.pt").unlink()
    elif fault == "manifest":
        settings.spatial_head_manifest_sha256 = "f" * 64
    elif fault == "config":
        (directory / artifacts.CONFIG_NAME).write_text("{}", encoding="utf8")
    elif fault == "other_role":
        with (directory / "spatial_head_final.pt").open("ab") as stream:
            stream.write(b"changed final head")
    elif fault == "source":
        settings.spatial_expected_bindings["core_source_sha256"] = "f" * 64
    else:
        (Path(settings.training_config["base_model_path"]) / "tokenizer.json").write_text("{}", encoding="utf8")
    with pytest.raises((ValueError, FileNotFoundError)):
        if fault == "tokenizer":
            inf.build_spatial_decoder(settings)
        else:
            inf.inspect_head(settings)


def test_exact_tokenizer_and_processor_eos_bound_before_generation(head_artifacts, torch_cpu):
    settings, _, _, _ = head_artifacts
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=500, convert_tokens_to_ids=lambda token: 500))
    model = toy_qwen(torch_cpu)
    model.generation_config.eos_token_id = [500, 400]
    pinned = inf.build_spatial_decoder(settings, processor=processor, model=model)
    assert pinned.pieces[1] == b"a" and pinned.eos_id == 500
    processor.tokenizer.eos_token_id = 400
    with pytest.raises(ValueError, match="processor"):
        inf.build_spatial_decoder(settings, processor=processor, model=model)


def test_role_requires_initial_base_and_final_adapter(head_artifacts):
    settings, _, _, _ = head_artifacts
    inf.validate_head_role(settings, None)
    with pytest.raises(ValueError, match="base requires"):
        inf.validate_head_role(settings, "adapter")
    settings.spatial_head_role = "final"
    inf.validate_head_role(settings, "adapter")
    with pytest.raises(ValueError, match="base requires"):
        inf.validate_head_role(settings, None)


def install_worker_spies(monkeypatch, torch, *, fail_head_inspection=False):
    from gui_agent.finetuning import evaluation as ev
    from gui_agent.finetuning import training
    from gui_agent.finetuning import visual_pointer_inference as vi

    model = toy_qwen(torch)
    model.suffix = byte_ids('[{"action_type":"click","point":"native_head"}]') + (500,)
    state = SimpleNamespace(loads=0, processor_calls=0, batch_decode_calls=0)
    identity = {"role": "initial", "manifest_sha256": "1" * 64, "eos_token_id": 500}
    registry = {"manifest_sha256": "2" * 64}
    ack = {"input_image_protocol": inf.INPUT_PROTOCOL, "visual_prompt_version": inf.PROMPT_VERSION,
           "native_width": 101, "native_height": 51, "canvas_width": 101, "canvas_height": 200}
    image = SimpleNamespace(mode="RGB", size=(101, 200), tobytes=lambda: b"synthetic fake raster")
    descriptor = SimpleNamespace(
        width=101, height=51, model_size=(101, 200), rgb_bytes=b"synthetic fake raster",
        input_image_protocol=inf.INPUT_PROTOCOL, visual_prompt_version=inf.PROMPT_VERSION,
        prompt="synthetic approved semantic prompt", to_pil=lambda: image,
        assert_current=lambda: None, evidence_json="{}")

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][1]["text"] == descriptor.prompt
            return "synthetic chat prefix"
        def __call__(self, **kwargs):
            state.processor_calls += 1
            assert kwargs["images"] == [image]
            return Batch(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
        def batch_decode(self, *args, **kwargs):
            state.batch_decode_calls += 1
            pytest.fail("native spatial must use exact ByteLevel output rather than batch_decode")

    model.device = torch.device("cpu")
    def load(*args, **kwargs):
        state.loads += 1
        return model, Processor()
    def inspection(cfg):
        if fail_head_inspection:
            raise ValueError("synthetic missing pinned initial head")
        return identity
    monkeypatch.setattr(inf, "inspect_head", inspection)
    monkeypatch.setattr(inf, "load_resident_head", lambda *args, **kwargs: (core.make_spatial_head().eval(), identity))
    monkeypatch.setattr(inf, "build_spatial_decoder", lambda *args, **kwargs: decoder())
    monkeypatch.setattr(training, "load_model_and_processor", load)
    monkeypatch.setattr(training.TrainingConfig, "from_dict", lambda value: SimpleNamespace(**value))
    monkeypatch.setattr(vi, "registry_from_config", lambda cfg: registry)
    monkeypatch.setattr(vi, "registry_identity", lambda value: dict(value))
    monkeypatch.setattr(vi, "validate_visual_request", lambda envelope, **kwargs: descriptor)
    monkeypatch.setattr(vi, "input_ack", lambda *args: dict(ack))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fields = {"visual_asset_manifest_path": str(PACKAGE / "synthetic_manifest.json"),
              "visual_asset_manifest_sha256": "2" * 64, "visual_original_root": str(PACKAGE),
              "visual_asset_root": str(PACKAGE)}
    cfg = config(**fields)
    cfg.training_config.update(fields)
    cfg = ev.HFConfig(**vars(cfg))
    envelope = {"input_image_protocol": inf.INPUT_PROTOCOL, "visual_prompt_version": inf.PROMPT_VERSION}
    return ev, cfg, state, model, identity, registry, ack, descriptor, envelope


class Connection:
    def __init__(self, requests):
        self.requests, self.responses, self.closed = iter(requests), [], False
    def recv(self):
        return next(self.requests)
    def send(self, response):
        self.responses.append(response)
    def close(self):
        self.closed = True


def test_fake_child_resident_head_ready_wrapped_request_ack_and_raw_joint_reply(monkeypatch, torch_cpu):
    ev, cfg, state, model, identity, registry, ack, _, envelope = install_worker_spies(monkeypatch, torch_cpu)
    request = inf.bind_spatial_request(envelope, head_identity=identity)
    connection = Connection([request, envelope, {"image_path": "native.png", "prompt": "old"}, None])
    ev._hf_worker(connection, cfg, None)
    ready, successful, stale, native = connection.responses
    inf.assert_spatial_ready(ready, head_identity=identity, visual_registry_identity=registry)
    assert successful["input_ack"] == inf.spatial_input_ack(ack, head_identity=identity)
    assert inf.validate_joint_reply(successful, decoder(), native_width=101, native_height=51)
    assert stale["status"] == native["status"] == "error"
    assert "input_ack" not in stale and "input_ack" not in native
    assert state.loads == state.processor_calls == model.generate_calls == 1 and state.batch_decode_calls == 0
    assert connection.closed and successful["repair_forward_calls"] == 0


def test_missing_head_fails_child_before_model_processor_or_generation(monkeypatch, torch_cpu):
    ev, cfg, state, model, _, _, _, _, _ = install_worker_spies(
        monkeypatch, torch_cpu, fail_head_inspection=True)
    connection = Connection([None])
    ev._hf_worker(connection, cfg, None)
    assert connection.responses[0]["kind"] == "load_error" and "missing pinned" in connection.responses[0]["error"]
    assert state.loads == state.processor_calls == model.generate_calls == model.forward_calls == 0
    assert connection.closed


def test_fake_child_preserves_invalid_language_raw_reply_and_zero_repair(monkeypatch, torch_cpu):
    ev, cfg, state, model, identity, _, _, _, envelope = install_worker_spies(monkeypatch, torch_cpu)
    model.suffix = byte_ids('[{"action_type":"click","candidate":1,"dx":0,"dy":0}]') + (500,)
    connection = Connection([inf.bind_spatial_request(envelope, head_identity=identity), None])
    ev._hf_worker(connection, cfg, None)
    failed = connection.responses[1]
    assert failed["status"] == "error" and failed["language_generation"].startswith('[{"action_type":"click","candidate"')
    assert failed["generation"] == "" and failed["mapped_actions"] is None
    assert failed["generation_calls"] == model.generate_calls == state.processor_calls == 1
    assert failed["generation_forward_calls"] == model.forward_calls and failed["repair_forward_calls"] == 0


def test_actor_bridge_uses_native_axes_and_never_reads_gt_or_candidate_centers(monkeypatch, torch_cpu):
    from gui_agent.finetuning import visual_pointer_inference as vi
    _, reply = generated(torch_cpu)
    identity = {"role": "initial", "eos_token_id": 500}
    registry = {"manifest_sha256": "2" * 64}
    class PoisonLabels(dict):
        def __getitem__(self, key):
            assert key not in {"actions", "action_annotations", "candidate_target", "target_bbox"}
            return super().__getitem__(key)
    row = PoisonLabels({key: "synthetic-input" for key in vi.VISIBLE_FIELDS})
    row.update(input_image_protocol=inf.INPUT_PROTOCOL, visual_prompt_version=inf.PROMPT_VERSION,
               candidate_snapshot={"candidates": [{"bbox": {"left": 900, "top": 900, "right": 1000, "bottom": 1000}}]},
               candidate_selection_evidence={}, image_width=101, image_height=51, actions=object())
    visible = vi.visible_sample(row)
    envelope = {"input_image_protocol": inf.INPUT_PROTOCOL, "visual_prompt_version": inf.PROMPT_VERSION,
                "visible_sample": visible}
    descriptor = SimpleNamespace(input_image_protocol=inf.INPUT_PROTOCOL, visual_prompt_version=inf.PROMPT_VERSION,
                                 width=101, height=51, model_size=(101, 200), prompt="approved fake prompt",
                                 evidence_json="{}", assert_current=lambda: None)
    visual_ack = {"input_image_protocol": inf.INPUT_PROTOCOL, "visual_prompt_version": inf.PROMPT_VERSION}
    monkeypatch.setattr(vi, "registry_identity", lambda value: dict(value))
    monkeypatch.setattr(vi, "prepare_visual_request", lambda *args, **kwargs: (descriptor, envelope))
    monkeypatch.setattr(vi, "input_ack", lambda *args: visual_ack)
    reply["input_ack"] = inf.spatial_input_ack(visual_ack, head_identity=identity)
    worker = SimpleNamespace(ready={"kind": "ready", "input_image_protocol": inf.INPUT_PROTOCOL,
                                   "visual_asset_registry": registry, **inf.authority_fields(identity)})
    worker.request_spatial = lambda request, **kwargs: reply
    result = inf.request_native_spatial_actions(worker, row, registry=registry, head_identity=identity,
              decoder=decoder(), timeout_seconds=1)
    assert result["mapped_actions"] == reply["mapped_actions"] and result["strict_joint_output_valid"]
    assert result["mapped_actions"][0]["x"] < 101 and result["mapped_actions"][0]["y"] < 51


def test_legacy_candidate_mapper_cannot_accept_spatial_prompt():
    from gui_agent.finetuning import visual_pointer_inference as vi
    with pytest.raises(ValueError, match="never pointer mapping"):
        vi.request_visual_pointer_actions(SimpleNamespace(), {"visual_prompt_version": inf.PROMPT_VERSION},
                                          registry=object(), timeout_seconds=1)


def test_real_training_config_roundtrip_excludes_inference_authority(tmp_path):
    from gui_agent.finetuning.evaluation import HFConfig
    from gui_agent.finetuning.training import TrainingConfig

    settings = config()
    visual_fields = {"visual_asset_manifest_path": str(tmp_path / "manifest.json"),
                     "visual_asset_manifest_sha256": "2" * 64,
                     "visual_original_root": str(tmp_path), "visual_asset_root": str(tmp_path)}
    settings.training_config.update(
        **visual_fields, base_model_path=str(tmp_path / "base"), train_jsonl=str(tmp_path / "train.jsonl"),
        output_dir=str(tmp_path / "output"), loss_strategy="registered_stratum_balanced",
        learning_rate=1e-4, spatial_head_learning_rate=1e-3, epochs=2, max_steps=144,
        max_optimizer_attempts=144, max_micro_steps=1144, gradient_accumulation_steps=8,
        sampling_policy="coverage_first_balanced_action_chunks",
        spatial_tokenizer_json_path=str(tmp_path / "tokenizer.json"), spatial_tokenizer_json_sha256="3" * 64)
    for key, value in visual_fields.items():
        setattr(settings, key, value)
    transport = HFConfig(**vars(settings))
    language = TrainingConfig.from_dict(transport.training_config)
    assert language.target_protocol == core.TARGET_PROTOCOL and transport.target_version == core.TARGET_VERSION
    assert "target_version" not in transport.training_config
    assert not set(inf.HEAD_CONFIG_FIELDS) & set(transport.training_config)
