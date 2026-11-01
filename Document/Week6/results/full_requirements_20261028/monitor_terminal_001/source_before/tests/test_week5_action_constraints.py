"""CPU-only format/backend contracts; synthetic replies are never model evidence."""
from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import ClassVar

import pytest
from PIL import Image

from gui_agent.finetuning import action_constraints as constraints
from gui_agent.finetuning import evaluation


def test_none_descriptor_needs_no_backend_or_model_import(monkeypatch):
    def forbidden(*args):
        raise AssertionError("none must not inspect optional backend")

    monkeypatch.setattr(constraints, "_version", forbidden)
    monkeypatch.setattr(constraints, "_backend_sources", forbidden)
    assert constraints.constraint_descriptor("none") == {
        "mode": "none", "schema_sha256": None, "backend": None}


@pytest.mark.parametrize("mode", ["JSON", "repair", None])
def test_unknown_constraint_rejected_before_worker_start(mode):
    with pytest.raises(ValueError, match="output_constraint"):
        evaluation.HFConfig({}, output_constraint=mode)


def test_schema_is_fixed_and_each_action_has_exact_native_fields():
    branches = constraints.native_action_schema()["items"]["anyOf"]
    fields = {b["properties"]["action_type"]["enum"][0]: set(b["properties"]) for b in branches}
    assert fields == {kind: required | {"action_type"} for kind, required in evaluation._REQUIRED.items()}
    assert all(b["additionalProperties"] is False and set(b["required"]) == set(b["properties"])
               for b in branches)
    first = constraints.native_action_schema()
    first["items"]["anyOf"][0]["properties"]["action_type"]["enum"].append("gold label")
    assert constraints.native_action_schema()["items"]["anyOf"][0]["properties"]["action_type"]["enum"] == ["click"]


@pytest.fixture
def backend(monkeypatch):
    state = {"tokenizer_builds": 0, "parsers": [], "prefixes": []}
    monkeypatch.setattr(constraints, "_version", lambda name: constraints.BACKEND_VERSIONS.get(name, "fixture"))
    monkeypatch.setattr(constraints, "_backend_sources", dict)

    class Parser:
        def __init__(self, schema, config):
            self.schema, self.config = schema, config
            state["parsers"].append(self)

    def tokenizer_data(tokenizer):
        state["tokenizer_builds"] += 1
        return SimpleNamespace(tokenizer_alphabet="actual tokenizer vocabulary")

    def build(data, parser):
        value = SimpleNamespace(data=data, parser=parser)
        state["prefixes"].append(value)
        return value

    monkeypatch.setitem(sys.modules, "lmformatenforcer", SimpleNamespace(
        CharacterLevelParserConfig=lambda **kwargs: SimpleNamespace(**kwargs), JsonSchemaParser=Parser))
    monkeypatch.setitem(sys.modules, "lmformatenforcer.integrations.transformers", SimpleNamespace(
        build_token_enforcer_tokenizer_data=tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn=build))
    return state


def test_tokenizer_cache_is_shared_but_parsers_and_prefix_states_are_request_local(backend):
    policy = constraints.NativeActionConstraint(SimpleNamespace(eos_token_id=99), [99, 98])
    first, second = policy.new_prefix_function(), policy.new_prefix_function()
    assert backend["tokenizer_builds"] == 1 and first.data is second.data
    assert first is not second and first.parser is not second.parser
    assert vars(first.parser.config) == {**constraints.PARSER_CONFIG, "alphabet": "actual tokenizer vocabulary"}
    assert policy.eos_descriptor["model_eos_token_ids"] == [99, 98]
    assert policy.eos_descriptor["constrained_eos_token_ids"] == [99]
    assert "model_eos_token_ids" not in policy.descriptor


def test_backend_config_reset_cannot_restore_environment_defaults(backend, monkeypatch):
    policy = constraints.NativeActionConstraint(SimpleNamespace(eos_token_id=99), [99])
    original = policy._build_prefix

    def resetting_builder(data, parser):
        prefix = original(data, parser)
        parser.config = SimpleNamespace(max_json_array_length=20, max_consecutive_whitespaces=12,
                                        force_json_field_order=True, alphabet="wrong")
        return prefix

    monkeypatch.setattr(policy, "_build_prefix", resetting_builder)
    first = policy.new_prefix_function()
    assert vars(first.parser.config) == {**constraints.PARSER_CONFIG, "alphabet": "actual tokenizer vocabulary"}


def test_missing_or_unpinned_backend_is_fatal_without_fallback(monkeypatch):
    monkeypatch.setattr(constraints, "_version", lambda name: None)
    monkeypatch.setattr(constraints, "_backend_sources", dict)
    with pytest.raises(RuntimeError, match="requires pinned"):
        constraints.NativeActionConstraint(SimpleNamespace(eos_token_id=99), [99])


def test_tokenizer_eos_must_be_a_real_model_eos_before_tokenizer_setup(backend):
    with pytest.raises(ValueError, match="EOS"):
        constraints.NativeActionConstraint(SimpleNamespace(eos_token_id=99), [98])
    assert backend["tokenizer_builds"] == 0


@pytest.mark.parametrize("mode", ["none", "json_actions"])
def test_worker_uses_original_greedy_kwargs_or_fresh_constraint_per_request(tmp_path, monkeypatch, mode):
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (10, 10)).save(image)
    state = {"setups": 0, "prefixes": [], "calls": [], "responses": []}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: None, cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None), inference_mode=nullcontext))

    class Policy:
        descriptor: ClassVar = {"mode": "json_actions", "schema_sha256": "CPU fixture"}
        eos_descriptor: ClassVar = {"tokenizer_eos_token_id": 99, "model_eos_token_ids": [99, 98],
                          "constrained_eos_token_ids": [99]}

        def __init__(self, tokenizer, eos):
            state["setups"] += 1

        def new_prefix_function(self):
            prefix = object()
            state["prefixes"].append(prefix)
            return prefix

    monkeypatch.setattr(constraints, "NativeActionConstraint", Policy)

    class Inputs(dict):
        def to(self, device):
            return self

    class Suffix:
        shape = (1, 2)

        def __getitem__(self, key):
            return 99

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[99, 98], pad_token_id=98,
                                             to_dict=lambda: {"eos_token_id": [99, 98], "pad_token_id": 98})

        def eval(self):
            pass

    class Generated:
        def __getitem__(self, key):
            return Suffix()

    model = Model()

    def generate(**kwargs):
        state["calls"].append(kwargs)
        return Generated()

    model.generate = generate

    class Processor:
        tokenizer = object()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][1] == {"type": "text", "text": "no labels in this input"}
            return "synthetic rendered input"

        def __call__(self, **kwargs):
            return Inputs(input_ids=SimpleNamespace(shape=(1, 1)))

        def batch_decode(self, *args, **kwargs):
            return ['[{"action_type":"key_press","key":"enter"}]']

    monkeypatch.setitem(sys.modules, "gui_agent.finetuning.training", SimpleNamespace(
        TrainingConfig=SimpleNamespace(from_dict=lambda value: value),
        load_model_and_processor=lambda *args, **kwargs: (model, Processor())))
    monkeypatch.setattr(evaluation, "runtime_versions", lambda: {"fixture": "CPU synthetic"})
    requests = iter([{"image_path": str(image), "prompt": "no labels in this input"}] * 2 + [None])
    connection = SimpleNamespace(recv=lambda: next(requests), send=state["responses"].append, close=lambda: None)
    evaluation._hf_worker(connection, evaluation.HFConfig({}, output_constraint=mode), None)
    assert len(state["calls"]) == 2 and all(r["status"] == "ok" for r in state["responses"][1:])
    for i, kwargs in enumerate(state["calls"]):
        assert kwargs["max_new_tokens"] == 256 and kwargs["do_sample"] is False and kwargs["use_cache"] is True
        assert set(kwargs) == {"input_ids", "max_new_tokens", "do_sample", "use_cache"} | (
            {"prefix_allowed_tokens_fn"} if mode == "json_actions" else set())
        if mode == "json_actions":
            assert kwargs["prefix_allowed_tokens_fn"] is state["prefixes"][i]
    assert state["setups"] == int(mode == "json_actions")
    if mode == "json_actions":
        ready = state["responses"][0]
        assert ready["output_constraint"] == Policy.descriptor
        assert ready["output_constraint_eos"] == Policy.eos_descriptor
        assert ready["constraint_hook"] == "prefix_allowed_tokens_fn"
        assert ready["model_generation_config"] == {"eos_token_id": [99, 98], "pad_token_id": 98}
        assert ready["generate_explicit_kwargs"] == {"max_new_tokens": 256, "do_sample": False, "use_cache": True}
        for reply in state["responses"][1:]:
            assert reply["output_constraint"] == Policy.descriptor
            assert reply["output_constraint_eos"] == Policy.eos_descriptor
            assert reply["last_token_id"] == 99 and reply["resolved_eos_token_ids"] == [99, 98]
    else:
        assert all("output_constraint" not in reply and "output_constraint_eos" not in reply
                   for reply in state["responses"])
