"""Explicit native spatial inference contract; no HF/torch at import.

Nothing here accepts target actions, candidate coordinates, bbox labels, or a
fallback. The existing native and candidate pointer paths remain independent.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from . import native_spatial_head as core

PROMPT_VERSION = "week5.visual-pointer-prompt.gallery-native-spatial.v1"
INPUT_PROTOCOL = "visual_pointer_gallery_v3"
REQUEST_SCHEMA = "week5.native_spatial_request.v1"
ACK_SCHEMA = "week5.native_spatial_input_ack.v1"
MODEL_CLASS = ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
               "Qwen2_5_VLForConditionalGeneration")
PEFT_CLASS = ("peft.peft_model", "PeftModelForCausalLM")
MAX_NEW_TOKENS = 256
HEAD_CONFIG_FIELDS = ("spatial_head_artifact_dir", "spatial_head_role",
                      "spatial_head_manifest_sha256", "spatial_expected_bindings")


def _sha(value: Any, label: str) -> str:
    core.require(isinstance(value, str) and len(value) == 64
                 and all(ch in "0123456789abcdef" for ch in value),
                 f"explicit lowercase SHA256 required: {label}")
    return value


def _class_identity(value: Any) -> tuple[str, str]:
    return type(value).__module__, type(value).__name__


def validate_spatial_config(config: Any) -> None:
    """Only called on the explicit new target branch; legacy defaults stay intact."""
    from pathlib import Path

    train = config.training_config
    core.require(isinstance(train, dict)
                 and train.get("target_protocol") == core.TARGET_PROTOCOL
                 and config.target_version == core.TARGET_VERSION,
                 "explicit native spatial target protocol/version required")
    core.require(config.input_image_protocol == train.get("input_image_protocol") == INPUT_PROTOCOL
                 and config.visual_prompt_version == train.get("visual_prompt_version") == PROMPT_VERSION
                 and train.get("prompt_variant") == "conditional_actor",
                 "native spatial gallery/prompt authority differs")
    core.require(type(config.max_new_tokens) is int and config.max_new_tokens == MAX_NEW_TOKENS
                 and config.output_constraint == "none",
                 "native spatial requires unconstrained greedy cap256")
    core.require(tuple(train.get(key) for key in (
        "min_pixels", "max_pixels", "max_seq_length", "processor_use_fast"))
        == (100352, 301056, 1536, True), "native spatial gallery processor profile differs")
    directory = config.spatial_head_artifact_dir
    core.require(isinstance(directory, str) and Path(directory).is_absolute()
                 and ".." not in Path(directory).parts, "explicit absolute head artifact directory required")
    core.require(config.spatial_head_role in {"initial", "final"}, "explicit initial/final head role required")
    _sha(config.spatial_head_manifest_sha256, "head manifest")
    bindings = config.spatial_expected_bindings
    core.require(isinstance(bindings, dict) and bool(bindings), "explicit head bindings required")
    core.require(bindings.get("prompt_version") == PROMPT_VERSION,
                 "head bindings prompt differs")
    for key in ("prompt_sha256", "tokenizer_json_sha256", "core_source_sha256",
                "training_source_sha256", "base_model_config_sha256"):
        _sha(bindings.get(key), key)
    core.require(type(bindings.get("eos_token_id")) is int and bindings["eos_token_id"] >= 0,
                 "explicit tokenizer assistant EOS binding required")
    for field in HEAD_CONFIG_FIELDS:
        core.require(field not in train or train[field] == getattr(config, field),
                     "HF and training spatial head authority differ")


def validate_model_contract(model: Any) -> Any:
    """One known bare Qwen or nonprompt LoRA wrapper, no generic path fallback."""
    if _class_identity(model) == PEFT_CLASS:
        options = getattr(model, "peft_config", None)
        core.require(isinstance(options, dict) and bool(options), "PEFT adapter config required")
        for option in options.values():
            core.require(getattr(option, "is_prompt_learning", None) is False
                         and str(getattr(option, "peft_type", "")) in {"LORA", "PeftType.LORA"},
                         "only nonprompt LoRA wrapper is approved")
        base = model.get_base_model()
    else:
        core.require(_class_identity(model) == MODEL_CLASS, "known Qwen2.5-VL model class required")
        base = model
    core.require(_class_identity(base) == MODEL_CLASS, "known Qwen2.5-VL base identity required")
    config = getattr(base, "config", None)
    text_config = getattr(config, "text_config", None)
    core.require(getattr(config, "model_type", None) == "qwen2_5_vl"
                 and getattr(text_config, "hidden_size", None) == core.HIDDEN_SIZE,
                 "known Qwen language hidden_size2048 required")
    resolved, norm = core.resolve_final_norm(model)
    core.require(resolved is base and norm is base.model.language_model.norm,
                 "required final norm identity differs")
    generation = getattr(model, "generation_config", None)
    core.require(generation is not None and getattr(model, "_compiled_call", None) is None,
                 "ordinary eager generation required")
    for field in ("compile_config", "prompt_lookup_num_tokens", "assistant_early_exit",
                  "constraints", "force_words_ids", "stop_strings"):
        core.require(getattr(generation, field, None) is None,
                     f"unsupported generation option: {field}")
    core.require(getattr(generation, "is_assistant", False) is False,
                 "assistant/speculative generation is unsupported")
    return norm


def inspect_head(config: Any) -> dict:
    """Before HF construction: strict hashes/config/all-file ledger, no torch."""
    validate_spatial_config(config)
    from pathlib import Path

    from .native_spatial_artifacts import inspect_spatial_head_artifact
    from .visual_pointer_prompts import GALLERY_SPATIAL_HEADER

    bindings = config.spatial_expected_bindings
    train_path = Path(__file__).with_name("training.py")
    for path, key in ((Path(core.__file__), "core_source_sha256"), (train_path, "training_source_sha256")):
        core.require(hashlib.sha256(path.read_bytes()).hexdigest() == bindings[key],
                     f"current {key} differs from artifact authority")
    core.require(hashlib.sha256(GALLERY_SPATIAL_HEADER.encode("utf8")).hexdigest() == bindings["prompt_sha256"],
                 "current semantic prompt header differs from artifact authority")
    base = Path(config.training_config["base_model_path"])
    config_bytes = (base / "config.json").read_bytes()
    core.require(hashlib.sha256(config_bytes).hexdigest() == bindings["base_model_config_sha256"],
                 "local model config differs from artifact authority")
    document = core.strict_json(config_bytes)
    core.require(document.get("model_type") == "qwen2_5_vl"
                 and document.get("architectures") == [MODEL_CLASS[1]]
                 and document.get("hidden_size", document.get("text_config", {}).get("hidden_size")) == core.HIDDEN_SIZE,
                 "known local Qwen2.5-VL hidden2048 config required")

    identity = inspect_spatial_head_artifact(
        config.spatial_head_artifact_dir, role=config.spatial_head_role,
        expected_bindings=config.spatial_expected_bindings,
        expected_manifest_sha256=config.spatial_head_manifest_sha256)
    return {**identity, "eos_token_id": config.spatial_expected_bindings["eos_token_id"]}


def load_resident_head(model: Any, config: Any, *, expected_identity: Mapping) -> tuple[Any, dict]:
    """Load saved weights once, after known model identity/shape is confirmed."""
    validate_spatial_config(config)
    norm = validate_model_contract(model)
    from .native_spatial_artifacts import load_spatial_head_artifact

    head, identity = load_spatial_head_artifact(
        config.spatial_head_artifact_dir, role=config.spatial_head_role,
        expected_bindings=config.spatial_expected_bindings,
        expected_manifest_sha256=config.spatial_head_manifest_sha256,
        device=norm.weight.device)
    identity = {**identity, "eos_token_id": config.spatial_expected_bindings["eos_token_id"]}
    core.require(identity == dict(expected_identity), "head artifacts changed while loading model")
    head.eval()
    core.require(not head.training, "resident head eval mode required")
    return head, identity


def validate_head_role(config: Any, adapter_path: str | None) -> None:
    validate_spatial_config(config)
    expected = "initial" if adapter_path is None else "final"
    core.require(config.spatial_head_role == expected, "base requires saved initial head; adapter requires saved final head")


def build_spatial_decoder(config: Any, *, processor: Any = None, model: Any = None) -> Any:
    """The tokenizer bytes and assistant EOS come from caller-pinned authority."""
    from pathlib import Path

    from .numeric_value_loss import TokenBytes

    validate_spatial_config(config)
    root = config.training_config.get("processor_path") or config.training_config.get("base_model_path")
    core.require(isinstance(root, str) and Path(root).is_absolute(), "explicit tokenizer root required")
    eos = config.spatial_expected_bindings["eos_token_id"]
    decoder = TokenBytes.from_frozen_json((Path(root) / "tokenizer.json").read_bytes(),
        expected_sha256=config.spatial_expected_bindings["tokenizer_json_sha256"], eos_id=eos)
    if processor is not None:
        core.require(processor.tokenizer.convert_tokens_to_ids("<|im_end|>") == eos
                     and processor.tokenizer.eos_token_id == eos,
                     "processor and pinned tokenizer assistant EOS differ")
    if model is not None:
        actual_eos = model.generation_config.eos_token_id
        model_eos = actual_eos if isinstance(actual_eos, list) else [actual_eos]
        core.require(all(type(token) is int for token in model_eos) and eos in model_eos,
                     "model lacks pinned assistant EOS")
    return decoder


def authority_fields(head_identity: Mapping) -> dict:
    core.require(isinstance(head_identity, Mapping) and bool(head_identity), "explicit head identity required")
    return {"target_protocol": core.TARGET_PROTOCOL, "target_version": core.TARGET_VERSION,
            "head_version": core.HEAD_VERSION, "spatial_head_artifact": dict(head_identity),
            "visual_prompt_version": PROMPT_VERSION,
            "coordinate_policy": "native_head/no caller mapping or correction",
            "generate_explicit_kwargs": core.generation_policy(eos_id=head_identity["eos_token_id"])}


def assert_spatial_ready(ready: Mapping, *, head_identity: Mapping, visual_registry_identity: Mapping) -> None:
    core.require(ready.get("kind") == "ready" and ready.get("input_image_protocol") == INPUT_PROTOCOL
                 and ready.get("visual_asset_registry") == dict(visual_registry_identity),
                 "spatial ready visual input authority differs")
    expected = authority_fields(head_identity)
    core.require(all(ready.get(key) == value for key, value in expected.items()),
                 "spatial ready target/head/prompt/generation authority differs")
    core.require(ready.get("output_constraint") is None and not ready.get("constraint_hook"),
                 "native spatial requires unconstrained worker")


def bind_spatial_request(envelope: Mapping, *, head_identity: Mapping) -> dict:
    """Wrap the separately validated, registered new-prompt gallery envelope."""
    core.require(envelope.get("input_image_protocol") == INPUT_PROTOCOL
                 and envelope.get("visual_prompt_version") == PROMPT_VERSION,
                 "new registered gallery semantic prompt required")
    return {"schema": REQUEST_SCHEMA, "visual_request": dict(envelope),
            **authority_fields(head_identity)}


def checked_spatial_request(request: Mapping, *, head_identity: Mapping) -> dict:
    expected = {"schema", "visual_request", *authority_fields(head_identity)}
    core.require(isinstance(request, Mapping) and set(request) == expected
                 and request["schema"] == REQUEST_SCHEMA,
                 "explicit exclusively spatial request required")
    core.require(all(request[key] == value for key, value in authority_fields(head_identity).items()),
                 "spatial request target/head/prompt authority differs")
    visual = request["visual_request"]
    core.require(isinstance(visual, Mapping) and visual.get("input_image_protocol") == INPUT_PROTOCOL
                 and visual.get("visual_prompt_version") == PROMPT_VERSION,
                 "spatial request new gallery prompt differs")
    return dict(visual)


def spatial_input_ack(visual_ack: Mapping, *, head_identity: Mapping) -> dict:
    core.require(visual_ack.get("input_image_protocol") == INPUT_PROTOCOL
                 and visual_ack.get("visual_prompt_version") == PROMPT_VERSION,
                 "spatial ack must contain new approved gallery prompt")
    return {"schema": ACK_SCHEMA, "visual_input_ack": dict(visual_ack), **authority_fields(head_identity)}


def _suffix_text(ids: list[int], decoder: Any) -> str:
    semantic = ids[:-1] if ids and ids[-1] == decoder.eos_id else ids
    try:
        return b"".join(decoder.pieces[token] for token in semantic).decode("utf8", errors="strict")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError("invalid generated ByteLevel bytes") from exc


def generate_joint_output(model: Any, head: Any, inputs: Mapping, decoder: Any, *,
                          native_width: int, native_height: int) -> dict:
    """Exactly one greedy generate; return raw failed outputs without repairing."""
    import torch

    core.dimensions(native_width, native_height)
    validate_model_contract(model)
    policy = core.generation_policy_binding(model, decoder, max_new_tokens=MAX_NEW_TOKENS)
    core.require({"input_ids"} <= set(inputs) <= core.INPUT_KEYS
                 and all(isinstance(value, torch.Tensor) for value in inputs.values()),
                 "only label-free tensor model inputs allowed")
    ids = inputs["input_ids"]
    core.require(ids.dtype == torch.long and ids.ndim == 2 and ids.shape[0] == 1
                 and 0 < ids.shape[1] <= 1536, "one exact gallery prefix up to1536 required")
    core.require(not model.training and not head.training, "model and head eval mode required")
    prefix = tuple(ids[0].tolist())
    capture = core.GenerationCapture(model, head, prefix_ids=prefix, max_new_tokens=MAX_NEW_TOKENS)
    result = {"target_protocol": core.TARGET_PROTOCOL, "target_version": core.TARGET_VERSION,
              "head_version": core.HEAD_VERSION, "language_generation": "",
              "raw_spatial_output": [], "marker_token_positions": [], "marker_byte_spans": [],
              "generated_token_ids": [], "prefix_length": len(prefix), "consumed_token_ids": [],
              "generation": "", "mapped_actions": None, "strict_joint_output_valid": False,
              "generation_calls": 0, "generation_forward_calls": 0, "repair_forward_calls": 0,
              "generation_policy_binding": policy,
              "status": "error", "error": None}
    try:
        with capture, torch.inference_mode():
            result["generation_calls"] = 1
            generated = core.generate_with_cache_contract(model, inputs, policy["overrides"])
        core.require(isinstance(generated, torch.Tensor) and generated.dtype == torch.long
                     and generated.ndim == 2 and generated.shape[0] == 1
                     and tuple(generated[0, :len(prefix)].tolist()) == prefix,
                     "generated sequence prefix differs")
        suffix = generated[0, len(prefix):].tolist()
        result["generated_token_ids"] = suffix
        result["language_generation"] = _suffix_text(suffix, decoder)
        spans = core.semantic_marker_spans(result["language_generation"])
        result["marker_byte_spans"] = [
            {"action_index": span.action_index, "field": span.field,
             "start": span.start, "end": span.end} for span in spans]
        positions = core.marker_token_positions(result["language_generation"],
                    suffix[:-1] if suffix and suffix[-1] == decoder.eos_id else suffix,
                    decoder, start_position=len(prefix))
        result["marker_token_positions"] = list(positions)
        # Preserve each available raw vector even if another marker is missing.
        result["raw_spatial_output"] = [capture.states[pos].tolist() if pos in capture.states else None
                                        for pos in positions]
        decoded = core.decode_generated(suffix, decoder, capture,
                  native_width=native_width, native_height=native_height)
        result.update(decoded)
        if not suffix or suffix[-1] != decoder.eos_id:
            raise ValueError("assistant EOS missing; no retry or repair allowed")
        result.update(status="ok", error=None)
    except (ValueError, TypeError, KeyError, RuntimeError, AttributeError) as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}",
                      strict_joint_output_valid=False, generation="", mapped_actions=None)
    finally:
        result["generation_forward_calls"] = capture.calls
        result["consumed_token_ids"] = [
            {"position": position, "token_id": token} for position, token in sorted(capture.consumed.items())]
    return result


def validate_joint_reply(reply: Mapping, decoder: Any, *, native_width: int, native_height: int) -> list[dict]:
    """Parent-side raw joint gate; serializer output alone earns no JSON credit."""
    core.dimensions(native_width, native_height)
    core.require(reply.get("status") == "ok" and reply.get("target_protocol") == core.TARGET_PROTOCOL
                 and reply.get("target_version") == core.TARGET_VERSION
                 and reply.get("head_version") == core.HEAD_VERSION,
                 "successful explicit spatial model output required")
    core.require(reply.get("generation_calls") == 1 and type(reply["generation_calls"]) is int
                 and reply.get("repair_forward_calls") == 0 and type(reply["repair_forward_calls"]) is int,
                 "exactly one generation and zero repair forwards required")
    policy = reply.get("generation_policy_binding")
    core.require(isinstance(policy, dict) and set(policy) == {"model_generation_config", "overrides"}
                 and isinstance(policy["model_generation_config"], dict)
                 and policy["overrides"] == core.generation_policy(eos_id=decoder.eos_id),
                 "explicit fixed greedy generation policy proof differs")
    ids, prefix = reply.get("generated_token_ids"), reply.get("prefix_length")
    core.require(isinstance(ids, list) and 0 < len(ids) <= MAX_NEW_TOKENS
                 and all(type(token) is int and token >= 0 for token in ids)
                 and type(prefix) is int and 0 < prefix <= 1536,
                 "complete bounded generated IDs and prefix length required")
    core.require(ids.count(decoder.eos_id) == 1 and ids[-1] == decoder.eos_id,
                 "one terminal assistant EOS required")
    text = _suffix_text(ids, decoder)
    core.require(text == reply.get("language_generation"), "raw decoded language differs from token IDs")
    spans = core.semantic_marker_spans(text)
    positions = core.marker_token_positions(text, ids[:-1], decoder, start_position=prefix)
    core.require(reply.get("marker_token_positions") == list(positions)
                 and all(type(pos) is int for pos in reply["marker_token_positions"]),
                 "raw structural marker positions differ")
    expected_spans = [{"action_index": span.action_index, "field": span.field,
                       "start": span.start, "end": span.end} for span in spans]
    core.require(reply.get("marker_byte_spans") == expected_spans, "raw marker UTF8 byte spans differ")
    expected_consumed = [{"position": prefix + offset, "token_id": token}
                         for offset, token in enumerate(ids[:-1])]
    actual_consumed = reply.get("consumed_token_ids")
    core.require(actual_consumed == expected_consumed
                 and isinstance(actual_consumed, list)
                 and all(type(item.get("position")) is int and type(item.get("token_id")) is int
                         for item in actual_consumed), "cached consumed token stream differs")
    core.require(type(reply.get("generation_forward_calls")) is int
                 and reply["generation_forward_calls"] == len(ids)
                 and all(prefix <= pos < prefix + len(ids) - 1 for pos in positions),
                 "forward count/consumed marker proof differs")
    native = core.materialize_semantic_actions(text, reply.get("raw_spatial_output"),
             native_width=native_width, native_height=native_height)
    core.require(reply.get("strict_joint_output_valid") is True
                 and reply.get("generation") == core.canonical(native)
                 and reply.get("mapped_actions") == native,
                 "materialized complete native array differs from raw model joint output")
    return native


def accept_spatial_reply(reply: Mapping, decoder: Any, *, expected_ack: Mapping,
                         native_width: int, native_height: int) -> dict:
    """HfActor bridge helper: never invokes candidate mapping or GT correction."""
    result = {"raw_reply": reply, "input_ack": reply.get("input_ack"), "input_ack_valid": False,
              "raw_semantic_json_valid": False, "raw_spatial_output_valid": False,
              "strict_joint_output_valid": False, "mapped_actions": None, "mapping_error": None}
    try:
        core.require(reply.get("input_ack") == dict(expected_ack), "spatial worker input acknowledgement differs")
        result["input_ack_valid"] = True
        native = validate_joint_reply(reply, decoder, native_width=native_width, native_height=native_height)
        result.update(raw_semantic_json_valid=True, raw_spatial_output_valid=True,
                      strict_joint_output_valid=True, mapped_actions=native)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        result["mapping_error"] = f"{type(exc).__name__}: {exc}"
    return result


def request_native_spatial_actions(worker: Any, sample: Mapping, *, registry: Any,
                                   head_identity: Mapping, decoder: Any, timeout_seconds: float) -> dict:
    """Bind original input and consume a raw joint reply without candidate mapping."""
    import math

    from .visual_pointer_inference import (
        input_ack,
        prepare_visual_request,
        registry_identity,
        visible_sample,
    )

    core.require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds)
                 and timeout_seconds > 0, "positive finite request timeout required")
    assert_spatial_ready(getattr(worker, "ready", {}), head_identity=head_identity,
                         visual_registry_identity=registry_identity(registry))
    descriptor, envelope = prepare_visual_request(sample, registry=registry)
    core.require(descriptor.input_image_protocol == INPUT_PROTOCOL
                 and descriptor.visual_prompt_version == PROMPT_VERSION,
                 "registered native spatial gallery descriptor required")
    expected_ack = spatial_input_ack(input_ack(descriptor, registry), head_identity=head_identity)
    request = bind_spatial_request(envelope, head_identity=head_identity)
    reply = worker.request_spatial(request, timeout_seconds=timeout_seconds)
    result = accept_spatial_reply(reply, decoder, expected_ack=expected_ack,
             native_width=descriptor.width, native_height=descriptor.height)
    result.update(prompt=descriptor.prompt, visual_input_evidence=core.strict_json(descriptor.evidence_json))
    try:
        assert_spatial_ready(getattr(worker, "ready", {}), head_identity=head_identity,
                             visual_registry_identity=registry_identity(registry))
        descriptor.assert_current()
        core.require(visible_sample(sample) == envelope["visible_sample"],
                     "registered spatial input changed during request")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result.update(input_ack_valid=False, strict_joint_output_valid=False, mapped_actions=None,
                      mapping_error=f"{type(exc).__name__}: {exc}")
    return result
