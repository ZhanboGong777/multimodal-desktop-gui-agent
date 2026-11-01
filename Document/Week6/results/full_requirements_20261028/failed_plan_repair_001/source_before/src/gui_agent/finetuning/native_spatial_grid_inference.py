"""PRIVATE native-grid inference draft; importing never loads HF or Torch.

The semantic language and strict causal transport use unchanged legacy marker
contracts. Native points come from the learned grid head's complete raw cell
outputs, independently reconstructed by the parent without GT or snapping.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gui_agent.finetuning import native_spatial_head as marker
from gui_agent.finetuning import native_spatial_inference as legacy

TARGET_PROTOCOL = "native_spatial_grid_head"
TARGET_VERSION = marker.TARGET_VERSION
HEAD_VERSION = "week5.native-spatial-grid-head.qk128-cellrefine.v2"
PROMPT_VERSION = legacy.PROMPT_VERSION
INPUT_PROTOCOL = legacy.INPUT_PROTOCOL
REQUEST_SCHEMA = "week5.native_spatial_grid_request.v1"
ACK_SCHEMA = "week5.native_spatial_grid_input_ack.v1"
GRID_INPUT_SCHEMA = "week5.native_spatial_grid_features_input.v1"
MODEL_CLASS, PEFT_CLASS = legacy.MODEL_CLASS, legacy.PEFT_CLASS
MAX_NEW_TOKENS = legacy.MAX_NEW_TOKENS
HEAD_CONFIG_FIELDS = legacy.HEAD_CONFIG_FIELDS
BINDING_KEYS = frozenset({"base_model_id", "base_model_revision", "base_model_config_sha256",
                          "prompt_version", "prompt_sha256", "eos_token_id", "tokenizer_json_sha256",
                          "core_source_sha256", "training_source_sha256", "geometry_source_sha256", "marker_source_sha256"})
FLOAT32_MAX = 3.4028234663852886e38
RAW_FP32_TOLERANCE = 1e-6


def _grid_module():
    from gui_agent.finetuning import native_spatial_grid_head
    return native_spatial_grid_head


def _geometry_module():
    from gui_agent.finetuning import native_spatial_grid_geometry
    return native_spatial_grid_geometry


def _sha(value, label):
    return legacy._sha(value, label)


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _value_sha(value):
    return hashlib.sha256(marker.canonical(value).encode("utf8")).hexdigest()


def _source_bindings():
    grid, geometry = _grid_module(), _geometry_module()
    marker.require(grid.HEAD_VERSION == HEAD_VERSION, "exact private grid head version required")
    return {"core_source_sha256": _file_sha(grid.__file__), "geometry_source_sha256": _file_sha(geometry.__file__),
            "marker_source_sha256": _file_sha(marker.__file__)}


def validate_spatial_config(config: Any) -> None:
    train = config.training_config
    marker.require(isinstance(train, dict) and train.get("target_protocol") == TARGET_PROTOCOL
                   and config.target_version == TARGET_VERSION, "explicit independent grid target branch required")
    marker.require(config.input_image_protocol == train.get("input_image_protocol") == INPUT_PROTOCOL
                   and config.visual_prompt_version == train.get("visual_prompt_version") == PROMPT_VERSION
                   and train.get("prompt_variant") == "conditional_actor", "unchanged Gallery semantic input required")
    marker.require(type(config.max_new_tokens) is int and config.max_new_tokens == MAX_NEW_TOKENS
                   and config.output_constraint == "none", "unconstrained greedy cap256 required")
    marker.require(tuple(train.get(key) for key in ("min_pixels", "max_pixels", "max_seq_length", "processor_use_fast"))
                   == (100352, 301056, 1536, True), "same approved Gallery processor profile required")
    directory = config.spatial_head_artifact_dir
    marker.require(isinstance(directory, str) and Path(directory).is_absolute() and ".." not in Path(directory).parts,
                   "absolute grid head artifact directory required")
    marker.require(config.spatial_head_role in {"initial", "final"}, "saved initial/final role required")
    _sha(config.spatial_head_manifest_sha256, "grid head manifest")
    bindings = config.spatial_expected_bindings
    marker.require(isinstance(bindings, dict) and set(bindings) == BINDING_KEYS
                   and bindings["prompt_version"] == PROMPT_VERSION, "complete old9 plus geometry/marker grid bindings required")
    for key in BINDING_KEYS - {"base_model_id", "base_model_revision", "prompt_version", "eos_token_id"}:
        _sha(bindings[key], key)
    marker.require(type(bindings["eos_token_id"]) is int and bindings["eos_token_id"] >= 0,
                   "explicit pinned tokenizer EOS required")
    marker.require(isinstance(bindings["base_model_id"], str) and bool(bindings["base_model_id"])
                   and isinstance(bindings["base_model_revision"], str) and len(bindings["base_model_revision"]) == 40
                   and all(char in "0123456789abcdef" for char in bindings["base_model_revision"]), "exact model/revision bindings required")
    for field in HEAD_CONFIG_FIELDS:
        marker.require(field not in train or train[field] == getattr(config, field), "HF/trainer grid artifact authority differs")


validate_model_contract = legacy.validate_model_contract


def inspect_head(config):
    validate_spatial_config(config)
    from gui_agent.finetuning.native_spatial_grid_artifacts import inspect_spatial_head_artifact
    from gui_agent.finetuning.visual_pointer_prompts import GALLERY_SPATIAL_HEADER

    bindings = config.spatial_expected_bindings
    for key, value in _source_bindings().items():
        marker.require(bindings[key] == value, f"current {key} differs from saved grid artifact")
    marker.require(_file_sha(Path(marker.__file__).with_name("training.py")) == bindings["training_source_sha256"],
                   "current training source differs from grid artifact")
    marker.require(hashlib.sha256(GALLERY_SPATIAL_HEADER.encode("utf8")).hexdigest() == bindings["prompt_sha256"],
                   "semantic prompt header changed")
    document_path = Path(config.training_config["base_model_path"]) / "config.json"
    payload = document_path.read_bytes()
    marker.require(hashlib.sha256(payload).hexdigest() == bindings["base_model_config_sha256"], "local model config changed")
    document = marker.strict_json(payload)
    marker.require(document.get("model_type") == "qwen2_5_vl" and document.get("architectures") == [MODEL_CLASS[1]]
                   and document.get("hidden_size", document.get("text_config", {}).get("hidden_size")) == marker.HIDDEN_SIZE,
                   "known Qwen2048 local config required")
    identity = inspect_spatial_head_artifact(config.spatial_head_artifact_dir, role=config.spatial_head_role,
                                           expected_bindings=bindings, expected_manifest_sha256=config.spatial_head_manifest_sha256)
    return {**identity, "eos_token_id": bindings["eos_token_id"]}


def _validate_head(head, device):
    import torch

    grid = _grid_module()
    state = head.state_dict()
    marker.require(set(state) == set(grid.STATE_SHAPES), "grid head parameter fields differ")
    marker.require(not head.training and not any(isinstance(module, torch.nn.Dropout) for module in head.modules()),
                   "grid inference requires eval mode and zero dropout")
    marker.require(all(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(grid.STATE_SHAPES[key])
                       and value.dtype == torch.float32 and value.device == device and bool(torch.isfinite(value).all())
                       for key, value in state.items()), "finite FP32 exact grid head state/device required")


def load_resident_head(model, config, *, expected_identity):
    validate_spatial_config(config)
    norm = validate_model_contract(model)
    from gui_agent.finetuning.native_spatial_grid_artifacts import load_spatial_head_artifact

    head, identity = load_spatial_head_artifact(config.spatial_head_artifact_dir, role=config.spatial_head_role,
                                              expected_bindings=config.spatial_expected_bindings,
                                              expected_manifest_sha256=config.spatial_head_manifest_sha256, device=norm.weight.device)
    identity = {**identity, "eos_token_id": config.spatial_expected_bindings["eos_token_id"]}
    marker.require(identity == dict(expected_identity), "grid head changed during model load")
    head.eval()
    _validate_head(head, norm.weight.device)
    return head, identity


def validate_head_role(config, adapter_path):
    validate_spatial_config(config)
    marker.require(config.spatial_head_role == ("initial" if adapter_path is None else "final"),
                   "initial/base and final/adapter roles must match actual training")


def build_spatial_decoder(config, *, processor=None, model=None):
    from gui_agent.finetuning.numeric_value_loss import TokenBytes

    validate_spatial_config(config)
    root = config.training_config.get("processor_path") or config.training_config.get("base_model_path")
    marker.require(isinstance(root, str) and Path(root).is_absolute(), "absolute tokenizer root required")
    eos = config.spatial_expected_bindings["eos_token_id"]
    decoder = TokenBytes.from_frozen_json((Path(root) / "tokenizer.json").read_bytes(),
                                        expected_sha256=config.spatial_expected_bindings["tokenizer_json_sha256"], eos_id=eos)
    if processor is not None:
        marker.require(processor.tokenizer.convert_tokens_to_ids("<|im_end|>") == processor.tokenizer.eos_token_id == eos,
                       "actual processor EOS differs")
    if model is not None:
        actual = model.generation_config.eos_token_id
        values = actual if isinstance(actual, list) else [actual]
        marker.require(all(type(value) is int for value in values) and eos in values, "model EOS differs")
    return decoder


def authority_fields(head_identity):
    marker.require(isinstance(head_identity, Mapping) and bool(head_identity), "explicit grid head identity required")
    return {"target_protocol": TARGET_PROTOCOL, "target_version": TARGET_VERSION, "head_version": HEAD_VERSION,
            "spatial_head_artifact": dict(head_identity), "visual_prompt_version": PROMPT_VERSION,
            "coordinate_policy": "learned_grid_mixture/no caller mapping or correction",
            "grid_source_bindings": _source_bindings(),
            "generate_explicit_kwargs": marker.generation_policy(eos_id=head_identity["eos_token_id"])}


def assert_spatial_ready(ready, *, head_identity, visual_registry_identity):
    marker.require(isinstance(ready, Mapping) and ready.get("kind") == "ready" and ready.get("input_image_protocol") == INPUT_PROTOCOL
                   and ready.get("visual_asset_registry") == dict(visual_registry_identity), "grid ready registered visual input differs")
    marker.require(all(ready.get(key) == value for key, value in authority_fields(head_identity).items()), "grid ready authority differs")
    marker.require(ready.get("output_constraint") is None and not ready.get("constraint_hook"), "unconstrained raw grid worker required")


def bind_spatial_request(envelope, *, head_identity):
    marker.require(envelope.get("input_image_protocol") == INPUT_PROTOCOL and envelope.get("visual_prompt_version") == PROMPT_VERSION,
                   "registered unchanged semantic Gallery request required")
    return {"schema": REQUEST_SCHEMA, "visual_request": dict(envelope), **authority_fields(head_identity)}


def checked_spatial_request(request, *, head_identity):
    expected = {"schema", "visual_request", *authority_fields(head_identity)}
    marker.require(isinstance(request, Mapping) and set(request) == expected and request["schema"] == REQUEST_SCHEMA,
                   "explicit exclusively grid request shape required")
    marker.require(all(request[key] == value for key, value in authority_fields(head_identity).items()), "grid request source/head mismatch")
    envelope = request["visual_request"]
    marker.require(isinstance(envelope, Mapping) and envelope.get("input_image_protocol") == INPUT_PROTOCOL
                   and envelope.get("visual_prompt_version") == PROMPT_VERSION, "grid visual envelope differs")
    return dict(envelope)


def spatial_input_ack(visual_ack, *, head_identity):
    marker.require(isinstance(visual_ack, Mapping) and visual_ack.get("input_image_protocol") == INPUT_PROTOCOL
                   and visual_ack.get("visual_prompt_version") == PROMPT_VERSION, "approved grid Gallery acknowledgement required")
    return {"schema": ACK_SCHEMA, "visual_input_ack": dict(visual_ack), **authority_fields(head_identity)}


@dataclass(frozen=True)
class FrozenGridInput:
    """Only immutable serialized whitelist geometry and actual prefix evidence."""
    recipe_json: str
    features_json: str
    evidence_json: str
    original_prefix_ids: tuple[int, ...]
    image_positions: tuple[int, ...]

    @property
    def recipe(self):
        return marker.strict_json(self.recipe_json)

    @property
    def features(self):
        return _grid_module().GridFeatures.from_dict(marker.strict_json(self.features_json))

    @property
    def evidence(self):
        return marker.strict_json(self.evidence_json)


def _descriptor_metadata(descriptor):
    descriptor.assert_current()
    marker.require(descriptor.visual_prompt_version == PROMPT_VERSION and descriptor.input_image_protocol == INPUT_PROTOCOL,
                   "validated Gallery semantic descriptor required")
    evidence = marker.strict_json(descriptor.evidence_json)
    path = Path(evidence["visual_metadata_path"])
    digest = _sha(evidence["visual_metadata_file_sha256"], "approved metadata")
    marker.require(path.is_absolute() and ".." not in path.parts, "absolute approved metadata path required")
    payload = path.read_bytes()
    marker.require(hashlib.sha256(payload).hexdigest() == digest, "approved metadata bytes changed")
    metadata = marker.strict_json(payload)
    marker.require(metadata["native_size"] == [descriptor.width, descriptor.height], "metadata/descriptor native size differs")
    return metadata, evidence


def _freeze_input(metadata, descriptor_evidence, prefix_ids, grid_thw, image_token_id):
    geometry, grid = _geometry_module(), _grid_module()
    marker.require(type(image_token_id) is int and image_token_id >= 0 and isinstance(prefix_ids, (tuple, list))
                   and 0 < len(prefix_ids) <= 1536 and all(type(token) is int and token >= 0 for token in prefix_ids),
                   "exact original label-free prefix and image token id required")
    prefix_ids = tuple(prefix_ids)
    recipe = geometry.build_recipe(metadata, grid_thw, patch_size=14, merge_size=2)
    positions = tuple(index for index, token in enumerate(prefix_ids) if token == image_token_id)
    marker.require(len(positions) == recipe["token_count"] and positions == tuple(range(positions[0], positions[0] + len(positions)))
                   and all(index < len(prefix_ids) for index in positions), "complete contiguous original-prefix image placeholders required")
    features = grid.GridFeatures.from_recipe(recipe)
    feature_dict = features.to_dict()
    evidence = {"schema": GRID_INPUT_SCHEMA, "input_binding_sha256": descriptor_evidence["input_binding_sha256"],
                "metadata_file_sha256": descriptor_evidence["visual_metadata_file_sha256"],
                "grid_thw": list(grid_thw), "image_token_id": image_token_id, "original_prefix_ids": list(prefix_ids),
                "image_positions": list(positions), "recipe_sha256": _value_sha(recipe),
                "geometry_input_sha256": recipe["geometry_input_sha256"], "features_sha256": _value_sha(feature_dict),
                "source_bindings": _source_bindings()}
    return FrozenGridInput(marker.canonical(recipe), marker.canonical(feature_dict), marker.canonical(evidence), prefix_ids, positions)


def build_grid_input(sample, descriptor, inputs, *, image_token_id, expected_bindings):
    """Worker only: combine approved metadata with actual CPU/GPU processor IDs/grid."""
    import torch

    marker.require(isinstance(inputs, Mapping) and set(inputs) <= marker.INPUT_KEYS and {"input_ids", "image_grid_thw"} <= set(inputs)
                   and all(isinstance(value, torch.Tensor) for value in inputs.values()), "label-free actual tensor inputs required")
    for key, value in _source_bindings().items():
        marker.require(expected_bindings.get(key) == value, "feature source differs from loaded head authority")
    ids, image_grid = inputs["input_ids"], inputs["image_grid_thw"]
    marker.require(ids.dtype == image_grid.dtype == torch.long and ids.ndim == 2 and ids.shape[0] == 1
                   and tuple(image_grid.shape) == (1, 3), "one actual original prefix and image grid required")
    metadata, evidence = _descriptor_metadata(descriptor)
    marker.require(sample["visual_metadata_path"] == evidence["visual_metadata_path"]
                   and sample["visual_metadata_file_sha256"] == evidence["visual_metadata_file_sha256"],
                   "visible request metadata must belong to validated descriptor")
    frozen = _freeze_input(metadata, evidence, ids[0].tolist(), image_grid[0].tolist(), image_token_id)
    descriptor.assert_current()
    return frozen


def rebuild_parent_grid_input(descriptor, reply_evidence, *, decoder=None):
    """Parent independently rebuilds every geometry feature from its approved file."""
    marker.require(isinstance(reply_evidence, dict) and reply_evidence.get("schema") == GRID_INPUT_SCHEMA,
                   "raw source-bound original-prefix grid evidence required")
    metadata, evidence = _descriptor_metadata(descriptor)
    frozen = _freeze_input(metadata, evidence, reply_evidence["original_prefix_ids"], reply_evidence["grid_thw"], reply_evidence["image_token_id"])
    marker.require(frozen.evidence == reply_evidence, "worker grid features/source/prefix evidence differs from parent approved geometry")
    if decoder is not None:
        marker.require(decoder.pieces.get(reply_evidence["image_token_id"]) == b"<|image_pad|>", "grid image id differs from pinned parent tokenizer")
    return frozen


def _assert_frozen_input(value):
    marker.require(isinstance(value, FrozenGridInput) and type(value.original_prefix_ids) is type(value.image_positions) is tuple,
                   "immutable complete typed original grid input required")
    evidence, recipe = value.evidence, value.recipe
    features = _grid_module().GridFeatures.from_recipe(recipe)
    marker.require(value.features.to_dict() == features.to_dict() and evidence["features_sha256"] == _value_sha(features.to_dict())
                   and evidence["recipe_sha256"] == _value_sha(recipe) and evidence["geometry_input_sha256"] == recipe["geometry_input_sha256"]
                   and evidence["source_bindings"] == _source_bindings() and tuple(evidence["original_prefix_ids"]) == value.original_prefix_ids
                   and tuple(evidence["image_positions"]) == value.image_positions
                   and value.image_positions == tuple(index for index, token in enumerate(value.original_prefix_ids) if token == evidence["image_token_id"])
                   and len(value.image_positions) == recipe["token_count"], "typed feature/prefix/source drift")


class GridGenerationCapture(marker.GenerationCapture):
    """Exact inherited cache transport; only the final-norm consumer is different."""
    def __init__(self, model, head, *, grid_input, max_new_tokens=MAX_NEW_TOKENS):
        _assert_frozen_input(grid_input)
        super().__init__(model, head, prefix_ids=grid_input.original_prefix_ids, max_new_tokens=max_new_tokens)
        self.grid_input = grid_input
        self.key_cache, self.key_projection_calls = None, 0
        self.query_states = {}

    def _after(self, module, inputs, output):
        import torch

        marker.require(self.pending is not None, "grid norm lacks paired actual cache position")
        start, actual = self.pending
        marker.require(isinstance(output, torch.Tensor) and output.shape == (1, len(actual), marker.HIDDEN_SIZE)
                       and output.is_floating_point() and bool(torch.isfinite(output).all()), "finite exact same-forward final norm states required")
        if start == 0:
            marker.require(self.key_cache is None and self.key_projection_calls == 0 and actual == self.prefix_ids,
                           "exactly one original-prefix prefill cache allowed")
            positions = torch.tensor(self.grid_input.image_positions, dtype=torch.long, device=output.device)
            marker.require(bool((positions < len(self.prefix_ids)).all()), "assistant/future tokens cannot become image keys")
            image_states = output[0].index_select(0, positions)
            self.key_projection_calls += 1
            self.key_cache = _grid_module().prepare_key_cache(self.head, image_states, self.grid_input.features)
        else:
            marker.require(start >= len(self.prefix_ids) and self.key_cache is not None, "cached generated token before original image prefill")
            for offset, identifier in enumerate(actual):
                position = start + offset
                marker.require(position not in self.query_states, "duplicate consumed grid query token")
                self.consumed[position] = identifier
                self.query_states[position] = output[0, offset].detach()
        self.pending = None


def _raw_values(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def _close(actual, wanted):
    if isinstance(wanted, list):
        return isinstance(actual, list) and len(actual) == len(wanted) and all(_close(a, b) for a, b in zip(actual, wanted, strict=True))
    return type(actual) is float and math.isfinite(actual) and abs(actual) <= FLOAT32_MAX and abs(actual - wanted) <= RAW_FP32_TOLERANCE


def _finite_raw_fp32(value):
    if isinstance(value, list):
        return all(_finite_raw_fp32(item) for item in value)
    return type(value) is float and math.isfinite(value) and abs(value) <= FLOAT32_MAX


def _raw_grid_prediction(prediction, features):
    keys = ("cell_logits", "all_cell_refinement_logits", "all_cell_unit_xy", "attention_weights", "all_token_attention_weights", "prediction_unit")
    raw = {key: _raw_values(prediction[key]) for key in keys}
    marker.require(_finite_raw_fp32(raw["cell_logits"]) and _finite_raw_fp32(raw["all_cell_refinement_logits"]),
                   "raw cell logits/refinements must be actual finite FP32 values")
    reconstructed = _grid_module().reconstruct_prediction(raw["cell_logits"], raw["all_cell_refinement_logits"], features)
    for key in keys[2:]:
        marker.require(_close(raw[key], reconstructed[key]), f"actual FP32 {key} differs from parent-reconstructed grid head math")
    return raw, reconstructed


def _stream_complete(reply, eos_id):
    ids, prefix = reply.get("generated_token_ids"), reply.get("prefix_length")
    if not (isinstance(ids, list) and 0 < len(ids) <= MAX_NEW_TOKENS and all(type(token) is int and token >= 0 for token in ids)
            and type(prefix) is int and 0 < prefix <= 1536 and type(reply.get("generation_calls")) is int and reply["generation_calls"] == 1
            and type(reply.get("repair_forward_calls")) is int and reply["repair_forward_calls"] == 0
            and type(reply.get("generation_forward_calls")) is int and reply["generation_forward_calls"] == len(ids)):
        return False
    expected = [{"position": prefix + offset, "token_id": token} for offset, token in enumerate(ids[:-1])]
    policy = reply.get("generation_policy_binding")
    actual = reply.get("consumed_token_ids")
    return bool(actual == expected and isinstance(actual, list)
                and all(isinstance(item, dict) and type(item.get("position")) is type(item.get("token_id")) is int for item in actual)
                and isinstance(policy, dict)
                and set(policy) == {"model_generation_config", "overrides"} and isinstance(policy["model_generation_config"], dict)
                and policy["overrides"] == marker.generation_policy(eos_id=eos_id))


def generation_completed(reply, *, eos_id):
    """Grid model/cache failures never become a completed zero-score answer."""
    return bool(isinstance(reply, Mapping) and reply.get("target_protocol") == TARGET_PROTOCOL and reply.get("head_version") == HEAD_VERSION
                and reply.get("grid_runtime_failure") is False and reply.get("grid_generation_stream_complete") is True
                and type(reply.get("projected_key_cache_builds")) is int and reply["projected_key_cache_builds"] == 1
                and isinstance(reply.get("grid_input_evidence"), dict) and reply["grid_input_evidence"].get("schema") == GRID_INPUT_SCHEMA
                and reply["grid_input_evidence"].get("source_bindings") == _source_bindings()
                and _stream_complete(reply, eos_id))


def _assert_key_cache(capture, head):
    cache = capture.key_cache
    marker.require(isinstance(cache, _grid_module().KeyCache) and cache.head_identity == id(head)
                   and cache.parameter_versions == tuple(parameter._version for parameter in head.parameters())
                   and cache.features == capture.grid_input.features, "same immutable-feature cache and unmodified saved head required")


def generate_joint_output(model, head, inputs, decoder, *, native_width, native_height, grid_input):
    """One greedy HF generate, one prefill K cache, one parsed marker batch at most."""
    import torch

    marker.dimensions(native_width, native_height)
    norm = validate_model_contract(model)
    _validate_head(head, norm.weight.device)
    _assert_frozen_input(grid_input)
    marker.require(grid_input.recipe["native_size"] == [native_width, native_height],
                   "native dimensions must come from same immutable geometry")
    marker.require({"input_ids", "image_grid_thw"} <= set(inputs) <= marker.INPUT_KEYS
                   and all(isinstance(value, torch.Tensor) for value in inputs.values()), "only actual label-free grid tensor inputs allowed")
    marker.require(inputs["input_ids"].dtype == torch.long and tuple(inputs["input_ids"].shape) == (1, len(grid_input.original_prefix_ids))
                   and tuple(inputs["input_ids"][0].tolist()) == grid_input.original_prefix_ids
                   and inputs["image_grid_thw"].tolist() == [grid_input.evidence["grid_thw"]]
                   and not model.training, "actual prefix/grid differs from frozen original input")
    base, _ = marker.resolve_final_norm(model)
    marker.require(base.config.image_token_id == grid_input.evidence["image_token_id"], "actual Qwen image token id differs")
    marker.require(grid_input.evidence["source_bindings"] == _source_bindings(), "grid source changed before generate")
    policy = marker.generation_policy_binding(model, decoder, max_new_tokens=MAX_NEW_TOKENS)
    capture = GridGenerationCapture(model, head, grid_input=grid_input)
    result = {"target_protocol": TARGET_PROTOCOL, "target_version": TARGET_VERSION, "head_version": HEAD_VERSION,
              "language_generation": "", "raw_spatial_output": [], "marker_token_positions": [], "marker_byte_spans": [],
              "generated_token_ids": [], "prefix_length": len(grid_input.original_prefix_ids), "consumed_token_ids": [],
              "generation": "", "mapped_actions": None, "strict_joint_output_valid": False,
              "generation_calls": 0, "generation_forward_calls": 0, "repair_forward_calls": 0,
              "generation_policy_binding": policy, "grid_input_evidence": grid_input.evidence,
              "raw_grid_output": None, "projected_key_cache_builds": 0, "grid_predict_calls": 0,
              "grid_runtime_failure": True, "grid_generation_stream_complete": False,
              "status": "error", "error": None, "grid_error_class": "model_transport"}
    try:
        with capture, torch.inference_mode():
            result["generation_calls"] = 1
            generated = marker.generate_with_cache_contract(model, inputs, policy["overrides"])
        marker.require(isinstance(generated, torch.Tensor) and generated.dtype == torch.long and generated.ndim == 2 and generated.shape[0] == 1
                       and tuple(generated[0, :len(capture.prefix_ids)].tolist()) == capture.prefix_ids, "actual generated prefix differs")
        suffix = generated[0, len(capture.prefix_ids):].tolist()
        result.update(generated_token_ids=suffix, generation_forward_calls=capture.calls,
                      consumed_token_ids=[{"position": pos, "token_id": token} for pos, token in sorted(capture.consumed.items())],
                      projected_key_cache_builds=capture.key_projection_calls)
        marker.require(_stream_complete(result, decoder.eos_id) and capture.key_cache is not None and capture.key_projection_calls == 1,
                       "complete exact actual generation and one finite key cache required")
        _assert_key_cache(capture, head)
        marker.require(grid_input.evidence["source_bindings"] == _source_bindings(), "grid source drifted during generation")
        result.update(grid_generation_stream_complete=True, grid_runtime_failure=False, grid_error_class="semantic")
        text = legacy._suffix_text(suffix, decoder)
        result["language_generation"] = text
        spans = marker.semantic_marker_spans(text)
        positions = marker.marker_token_positions(text, suffix[:-1] if suffix and suffix[-1] == decoder.eos_id else suffix,
                                                 decoder, start_position=len(capture.prefix_ids))
        result.update(marker_token_positions=list(positions), marker_byte_spans=[{"action_index": span.action_index, "field": span.field,
                                                                               "start": span.start, "end": span.end} for span in spans])
        marker.require(bool(suffix) and suffix[-1] == decoder.eos_id and all(position in capture.query_states for position in positions),
                       "terminal assistant EOS and actually consumed marker states required; no repair")
        if positions:
            result.update(grid_error_class="grid_prediction", grid_runtime_failure=True)
            states = torch.stack([capture.query_states[position] for position in positions])
            result["grid_predict_calls"] = 1
            with torch.inference_mode():
                predicted = _grid_module().predict_with_cache(head, states, capture.key_cache)
            raw, reconstructed = _raw_grid_prediction(predicted, grid_input.features)
            result.update(raw_grid_output=raw, raw_spatial_output=reconstructed["prediction_unit"], grid_runtime_failure=False)
        else:
            result["raw_grid_output"] = {key: [] for key in ("cell_logits", "all_cell_refinement_logits", "all_cell_unit_xy",
                                                            "attention_weights", "all_token_attention_weights", "prediction_unit")}
        result["grid_error_class"] = "semantic"
        native = marker.materialize_semantic_actions(text, result["raw_spatial_output"], native_width=native_width, native_height=native_height)
        result.update(grid_runtime_failure=True, grid_error_class="source_binding")
        _assert_key_cache(capture, head)
        marker.require(grid_input.evidence["source_bindings"] == _source_bindings(), "grid source drifted during head prediction")
        result.update(generation=marker.canonical(native), mapped_actions=native, strict_joint_output_valid=True,
                      status="ok", error=None, grid_error_class=None, grid_runtime_failure=False)
    except (ValueError, TypeError, KeyError, RuntimeError, AttributeError, IndexError) as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}", strict_joint_output_valid=False, generation="", mapped_actions=None)
    finally:
        result.update(generation_forward_calls=capture.calls, projected_key_cache_builds=capture.key_projection_calls,
                      consumed_token_ids=[{"position": pos, "token_id": token} for pos, token in sorted(capture.consumed.items())])
    return result


def validate_joint_reply(reply, decoder, *, native_width, native_height, expected_grid_input=None):
    marker.require(isinstance(reply, Mapping) and reply.get("status") == "ok" and reply.get("target_protocol") == TARGET_PROTOCOL
                   and reply.get("target_version") == TARGET_VERSION and reply.get("head_version") == HEAD_VERSION,
                   "successful explicit raw grid output required")
    marker.require(isinstance(expected_grid_input, FrozenGridInput) and expected_grid_input.recipe["native_size"] == [native_width, native_height]
                   and reply.get("grid_input_evidence") == expected_grid_input.evidence, "parent-approved independent grid geometry required")
    _assert_frozen_input(expected_grid_input)
    marker.require(decoder.pieces.get(expected_grid_input.evidence["image_token_id"]) == b"<|image_pad|>",
                   "parent tokenizer image placeholder differs")
    marker.require(generation_completed(reply, eos_id=decoder.eos_id) and reply["prefix_length"] == len(expected_grid_input.original_prefix_ids),
                   "complete source-bound grid transport required")
    semantic = legacy._suffix_text(reply["generated_token_ids"], decoder)
    count = len(marker.semantic_marker_spans(semantic))
    marker.require(type(reply.get("grid_predict_calls")) is int and reply["grid_predict_calls"] == int(count > 0),
                   "one same parsed-marker batch only; zero repair queries")
    raw = reply.get("raw_grid_output")
    keys = {"cell_logits", "all_cell_refinement_logits", "all_cell_unit_xy", "attention_weights", "all_token_attention_weights", "prediction_unit"}
    marker.require(isinstance(raw, dict) and set(raw) == keys and len(raw["cell_logits"]) == count, "complete all-cell raw grid fields required")
    if count:
        observed, reconstructed = _raw_grid_prediction(raw, expected_grid_input.features)
        marker.require(observed == raw and reply["raw_spatial_output"] == reconstructed["prediction_unit"],
                       "native points must equal independently reconstructed model cell mixture")
    else:
        marker.require(all(value == [] for value in raw.values()) and reply["raw_spatial_output"] == [], "nonmouse output cannot invent head points")
    # Keep every old semantic/EOS/cache/marker/rounding check byte-for-byte in
    # the unchanged implementation after checking this independent branch.
    old_view = {**reply, "target_protocol": marker.TARGET_PROTOCOL, "head_version": marker.HEAD_VERSION}
    return legacy.validate_joint_reply(old_view, decoder, native_width=native_width, native_height=native_height)


def accept_spatial_reply(reply, decoder, *, expected_ack, native_width, native_height, expected_grid_input=None):
    marker.require(isinstance(reply, Mapping), "grid reply mapping required")
    result = {"raw_reply": reply, "input_ack": reply.get("input_ack"), "input_ack_valid": False,
              "raw_semantic_json_valid": False, "raw_spatial_output_valid": False, "strict_joint_output_valid": False,
              "mapped_actions": None, "mapping_error": None, "grid_generation_completed": False}
    try:
        marker.require(reply.get("input_ack") == dict(expected_ack), "grid input acknowledgement differs")
        result["input_ack_valid"] = True
        result["grid_generation_completed"] = generation_completed(reply, eos_id=decoder.eos_id)
        native = validate_joint_reply(reply, decoder, native_width=native_width, native_height=native_height,
                                      expected_grid_input=expected_grid_input)
        result.update(raw_semantic_json_valid=True, raw_spatial_output_valid=True, strict_joint_output_valid=True, mapped_actions=native)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        result["mapping_error"] = f"{type(exc).__name__}: {exc}"
    return result


def request_native_spatial_actions(worker, sample, *, registry, head_identity, decoder, timeout_seconds):
    from gui_agent.finetuning.visual_pointer_inference import (
        input_ack,
        prepare_visual_request,
        registry_identity,
        visible_sample,
    )

    marker.require(type(timeout_seconds) in (int, float) and math.isfinite(timeout_seconds) and timeout_seconds > 0, "positive finite request timeout required")
    assert_spatial_ready(getattr(worker, "ready", {}), head_identity=head_identity, visual_registry_identity=registry_identity(registry))
    descriptor, envelope = prepare_visual_request(sample, registry=registry)
    expected_ack = spatial_input_ack(input_ack(descriptor, registry), head_identity=head_identity)
    request = bind_spatial_request(envelope, head_identity=head_identity)
    reply = worker.request_spatial(request, timeout_seconds=timeout_seconds)
    grid_input, geometry_error = None, None
    try:
        grid_input = rebuild_parent_grid_input(descriptor, reply.get("grid_input_evidence"), decoder=decoder)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        geometry_error = f"{type(exc).__name__}: {exc}"
    result = accept_spatial_reply(reply, decoder, expected_ack=expected_ack, native_width=descriptor.width, native_height=descriptor.height,
                                 expected_grid_input=grid_input)
    result.update(prompt=descriptor.prompt, visual_input_evidence=marker.strict_json(descriptor.evidence_json))
    if geometry_error:
        result.update(strict_joint_output_valid=False, mapped_actions=None, grid_generation_completed=False, mapping_error=geometry_error)
    try:
        assert_spatial_ready(getattr(worker, "ready", {}), head_identity=head_identity, visual_registry_identity=registry_identity(registry))
        descriptor.assert_current()
        marker.require(visible_sample(sample) == envelope["visible_sample"], "registered grid input changed during request")
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result.update(input_ack_valid=False, strict_joint_output_valid=False, mapped_actions=None, grid_generation_completed=False,
                      mapping_error=f"{type(exc).__name__}: {exc}")
    return result
