"""Native spatial head contract: no model, processor, image or UI loading.

The semantic LM output and learned coordinates jointly form a new model output.
This is not a repair function for the old candidate/offset protocol. Imports of
torch are lazy; all labels are explicit caller-provided training supervision.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TARGET_PROTOCOL = "native_spatial_head"
TARGET_VERSION = "week5.native-spatial-target.v1"
HEAD_VERSION = "week5.native-spatial-head.mlp2048x128x2.v2"
MARKER = "native_head"
HIDDEN_SIZE = 2048
HIDDEN_WIDTH = 128
FEATURE_NORM_EPS = 1e-5
OUTPUT_INIT_STD = 1e-3
DIAGNOSTIC_SATURATION_EPS = 1e-6
POINT_TYPES = frozenset({"click", "double_click", "right_click", "move", "scroll"})
NONMOUSE_FIELDS = {
    "type_text": {"text"}, "key_press": {"key"}, "hotkey": {"keys"}, "wait": {"duration"},
}
INPUT_KEYS = frozenset({
    "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "position_ids",
    "token_type_ids",
})


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def strict_json(text: str | bytes) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON member")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def dimensions(width: int, height: int) -> tuple[int, int]:
    require(type(width) is int and type(height) is int and width > 0 and height > 0,
            "positive integer native dimensions required")
    return width, height


def _array(value: Any) -> None:
    require(isinstance(value, list) and bool(value), "whole nonempty action array required")
    require(all(isinstance(action, dict) for action in value), "every action must be an object")


def _nonmouse(action: dict) -> None:
    kind = action["action_type"]
    require(kind in NONMOUSE_FIELDS and set(action) == NONMOUSE_FIELDS[kind] | {"action_type"},
            "unknown action or missing/unexpected nonmouse fields")
    if kind in {"type_text", "key_press"}:
        name = "text" if kind == "type_text" else "key"
        require(isinstance(action[name], str) and bool(action[name]), "nonempty string required")
    elif kind == "hotkey":
        require(isinstance(action["keys"], list) and bool(action["keys"])
                and all(isinstance(key, str) and bool(key) for key in action["keys"]),
                "hotkey keys must be nonempty strings")
    elif kind == "wait":
        value = action["duration"]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "finite nonnegative wait duration required")


def validate_semantic_actions(value: Any) -> list[dict]:
    """Strict whole-array validation; no field coercion, dropped action or text repair."""
    _array(value)
    for action in value:
        kind = action.get("action_type")
        require(isinstance(kind, str), "action_type must be a string")
        if kind in POINT_TYPES:
            fields = {"action_type", "point"} | ({"scroll_amount"} if kind == "scroll" else set())
            require(set(action) == fields and action["point"] == MARKER,
                    "point action requires exactly the native_head marker")
            if kind == "scroll":
                require(type(action["scroll_amount"]) is int and action["scroll_amount"] != 0,
                        "scroll_amount must be a nonzero integer")
        elif kind == "drag":
            require(set(action) == {"action_type", "start_point", "end_point"}
                    and action["start_point"] == action["end_point"] == MARKER,
                    "drag requires exactly two endpoint markers")
        else:
            _nonmouse(action)
    return value


def _native_point(point: Mapping[str, Any], width: int, height: int) -> tuple[int, int]:
    require(set(point) == {"x", "y"} and all(type(point[axis]) is int for axis in ("x", "y")),
            "native point must contain exactly integer x/y")
    x, y = point["x"], point["y"]
    require(0 <= x < width and 0 <= y < height, "native supervision is out of bounds")
    return x, y


@dataclass(frozen=True)
class MarkerSpan:
    action_index: int
    field: str
    start: int
    end: int


def semantic_marker_spans(text: str) -> tuple[MarkerSpan, ...]:
    """UTF8 content spans of structural marker values, never strings in literal text."""
    require(isinstance(text, str), "raw semantic generation must be a string")
    actions = validate_semantic_actions(strict_json(text))
    decoder = json.JSONDecoder()
    pos = 0
    def whitespace(position):
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position
    pos = whitespace(pos)
    require(text[pos] == "[", "bare whole array required")
    pos = whitespace(pos + 1)
    spans = []
    for index, action in enumerate(actions):
        require(text[pos] == "{", "action object boundary differs")
        pos = whitespace(pos + 1)
        while text[pos] != "}":
            key, pos = decoder.raw_decode(text, pos)
            pos = whitespace(pos)
            require(text[pos] == ":", "member separator differs")
            begin = whitespace(pos + 1)
            _, end = decoder.raw_decode(text, begin)
            if key in {"point", "start_point", "end_point"}:
                require(text[begin:end] == '"native_head"',
                        "marker must use the exact unescaped native_head byte spelling")
                start_byte = len(text[:begin + 1].encode("utf8"))
                end_byte = len(text[:end - 1].encode("utf8"))
                spans.append(MarkerSpan(index, key, start_byte, end_byte))
            pos = whitespace(end)
            if text[pos] == ",":
                pos = whitespace(pos + 1)
            else:
                require(text[pos] == "}", "action member boundary differs")
        pos = whitespace(pos + 1)
        if index + 1 < len(actions):
            require(text[pos] == ",", "array action boundary differs")
            pos = whitespace(pos + 1)
    require(text[pos] == "]" and whitespace(pos + 1) == len(text), "trailing array bytes")
    return tuple(spans)


def semantic_target(actions: Any, *, native_width: int, native_height: int) -> dict:
    """Training only: derive semantic targets/coordinate labels from native actions."""
    width, height = dimensions(native_width, native_height)
    _array(actions)
    semantic, native_points = [], {}
    for index, source in enumerate(actions):
        kind = source.get("action_type")
        require(isinstance(kind, str), "native action_type must be a string")
        if kind in POINT_TYPES:
            expected = {"action_type", "x", "y"} | ({"scroll_amount"} if kind == "scroll" else set())
            require(set(source) == expected, "native point action fields differ")
            native_points[index, "point"] = _native_point(
                {"x": source["x"], "y": source["y"]}, width, height)
            action = {"action_type": kind, "point": MARKER}
            if kind == "scroll":
                action["scroll_amount"] = source["scroll_amount"]
        elif kind == "drag":
            require(set(source) == {"action_type", "start", "end"}, "native drag fields differ")
            action = {"action_type": kind, "start_point": MARKER, "end_point": MARKER}
            for field, endpoint in (("start_point", "start"), ("end_point", "end")):
                require(isinstance(source[endpoint], Mapping), "native drag endpoint must be an object")
                native_points[index, field] = _native_point(source[endpoint], width, height)
        else:
            _nonmouse(source)
            action = strict_json(canonical(source))
        semantic.append(action)
    validate_semantic_actions(semantic)
    text = canonical(semantic)
    supervision = []
    for span in semantic_marker_spans(text):
        x, y = native_points[span.action_index, span.field]
        supervision.append({"action_index": span.action_index, "field": span.field,
                            "x": x, "y": y,
                            "u": x / (width - 1) if width > 1 else 0.0,
                            "v": y / (height - 1) if height > 1 else 0.0})
    return {"target_protocol": TARGET_PROTOCOL, "target_version": TARGET_VERSION,
            "native_width": width, "native_height": height, "semantic_actions": semantic,
            "semantic_json": text, "point_supervision": supervision}


def _unit_pair(value: Any) -> tuple[float, float]:
    require(isinstance(value, (list, tuple)) and len(value) == 2,
            "head output must contain exactly two values")
    require(all(type(axis) is float and math.isfinite(axis) and 0 <= axis <= 1 for axis in value),
            "head output must be finite float values in [0,1]")
    return value[0], value[1]


def materialize_semantic_actions(text: str, point_values: Sequence, *,
                                 native_width: int, native_height: int) -> list[dict]:
    """Decode this new joint output only; no GT, candidate, fallback or clipping inputs."""
    width, height = dimensions(native_width, native_height)
    actions = validate_semantic_actions(strict_json(text))
    spans = semantic_marker_spans(text)
    require(isinstance(point_values, (list, tuple)) and len(point_values) == len(spans),
            "head output count differs from whole-array marker count")
    points = {}
    for span, value in zip(spans, point_values, strict=True):
        u, v = _unit_pair(value)
        points[span.action_index, span.field] = {
            "x": math.floor(u * (width - 1) + 0.5),
            "y": math.floor(v * (height - 1) + 0.5),
        }
    native = []
    for index, action in enumerate(actions):
        kind = action["action_type"]
        if kind in POINT_TYPES:
            converted = {"action_type": kind, **points[index, "point"]}
            if kind == "scroll":
                converted["scroll_amount"] = action["scroll_amount"]
        elif kind == "drag":
            converted = {"action_type": kind, "start": points[index, "start_point"],
                         "end": points[index, "end_point"]}
        else:
            converted = strict_json(canonical(action))
        native.append(converted)
    return native


def marker_token_positions(text: str, token_ids: Sequence[int], decoder: Any, *,
                           start_position: int = 0) -> tuple[int, ...]:
    require(type(start_position) is int and start_position >= 0, "nonnegative token offset required")
    require(all(type(identifier) is int and identifier >= 0 for identifier in token_ids),
            "token IDs must be nonnegative integers")
    try:
        pieces = [decoder.pieces[identifier] for identifier in token_ids]
    except KeyError as exc:
        raise ValueError("unknown ByteLevel token ID") from exc
    require(b"".join(pieces) == text.encode("utf8"), "token bytes do not reconstruct raw semantic array")
    positions, byte_position = [], 0
    token_ranges = []
    for index, piece in enumerate(pieces):
        require(isinstance(piece, bytes) and bool(piece), "nonempty decoded token bytes required")
        token_ranges.append((byte_position, byte_position + len(piece), start_position + index))
        byte_position += len(piece)
    for span in semantic_marker_spans(text):
        hits = [index for begin, end, index in token_ranges if begin <= span.end - 1 < end]
        require(len(hits) == 1, "marker last byte must belong to exactly one token")
        positions.append(hits[0])
    require(len(set(positions)) == len(positions), "multiple endpoint markers share one state token")
    return tuple(positions)


@dataclass(frozen=True)
class SpatialPlan:
    full_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prefix_length: int
    marker_positions: tuple[int, ...]
    target_unit: tuple[tuple[float, float], ...]
    slot_keys: tuple[tuple[int, str], ...]
    native_width: int
    native_height: int
    supervised_tokens: int


def training_plan(*, full_ids, prefix_ids, labels, target: Mapping, decoder: Any) -> SpatialPlan:
    full, prefix, labels = tuple(full_ids), tuple(prefix_ids), tuple(labels)
    require(all(type(token) is int and token >= 0 for token in full + prefix), "integer input IDs required")
    require(all(type(label) is int for label in labels), "integer labels required without coercion")
    require(0 < len(prefix) < len(full) and full[:len(prefix)] == prefix, "expanded prefix differs")
    require(len(labels) == len(full) and labels[:len(prefix)] == (-100,) * len(prefix)
            and labels[len(prefix):] == full[len(prefix):], "complete semantic/EOS labels and masked prefix required")
    require(target["target_protocol"] == TARGET_PROTOCOL and target["target_version"] == TARGET_VERSION,
            "explicit spatial target protocol/version required")
    width, height = dimensions(target["native_width"], target["native_height"])
    text = target["semantic_json"]
    require(canonical(validate_semantic_actions(target["semantic_actions"])) == text,
            "semantic target JSON differs from complete target actions")
    spans = semantic_marker_spans(text)
    supervision = target["point_supervision"]
    require(isinstance(supervision, list) and len(supervision) == len(spans), "point supervision count differs")
    units = []
    for span, row in zip(spans, supervision, strict=True):
        require(isinstance(row, dict) and set(row) == {"action_index", "field", "x", "y", "u", "v"}
                and type(row["action_index"]) is int
                and (row["action_index"], row["field"]) == (span.action_index, span.field),
                "supervision slot differs from structural marker order")
        x, y = _native_point({"x": row["x"], "y": row["y"]}, width, height)
        expected = (x / (width - 1) if width > 1 else 0.0,
                    y / (height - 1) if height > 1 else 0.0)
        actual = _unit_pair([row["u"], row["v"]])
        require(actual == expected, "point supervision normalization differs")
        units.append(actual)
    suffix = full[len(prefix):]
    require(suffix.count(decoder.eos_id) == 1, "one supervised assistant end required")
    eos = suffix.index(decoder.eos_id)
    require(decoder.pieces[decoder.eos_id] == b"<|im_end|>"
            and b"".join(decoder.pieces[token] for token in suffix[eos + 1:]) == b"\n",
            "assistant tail must be exactly end token and newline")
    positions = marker_token_positions(text, suffix[:eos], decoder, start_position=len(prefix))
    count = sum(value != -100 for value in labels[1:])
    require(count > 0, "no shifted supervised tokens")
    return SpatialPlan(full, labels, len(prefix), positions, tuple(units),
                       tuple((span.action_index, span.field) for span in spans), width, height, count)


def head_configuration() -> dict:
    return {"target_protocol": TARGET_PROTOCOL, "target_version": TARGET_VERSION,
            "head_version": HEAD_VERSION, "hidden_size": HIDDEN_SIZE, "hidden_width": HIDDEN_WIDTH,
            "output_size": 2, "activation": "GELU", "output_activation": "sigmoid",
            "parameter_dtype": "float32", "dropout": 0, "point_loss_weight": 1.0,
            "feature_normalization": {"kind": "LayerNorm", "normalized_shape": [HIDDEN_SIZE],
                                      "eps": FEATURE_NORM_EPS, "elementwise_affine": False,
                                      "scope": "per token last feature axis", "dtype": "float32",
                                      "gradient": "live autograd; no detach"},
            "first_linear_initialization": "torch.nn.Linear default reset_parameters",
            "initialization_device": "cpu",
            "initialization_rng": "isolated CPU default_generator; CUDA RNG untouched",
            "output_linear_initialization": {"weight": "normal", "mean": 0.0,
                                             "std": OUTPUT_INIT_STD, "bias": "zero"},
            "point_loss": "mean endpoint/axis SmoothL1 normalized native xy",
            "beta_axis": "20/max(1,native_axis-1)", "degenerate_axis": "effective prediction and target zero",
            "rounding": "floor(unit*(native_axis-1)+0.5), no clamp",
            "lora_learning_rate": 1e-4, "head_learning_rate": 1e-3,
            "marker_state": "final language norm output for token containing marker last content byte"}


def objective_contract() -> dict:
    """Registered joint objective, including the existing actual LoRA rate."""
    return {**head_configuration(), "semantic_ce_weight": 1.0,
            "semantic_ce": "complete assistant semantic JSON, end token and newline",
            "label_mask": "expanded prefix only", "manual_CE_calls": 1,
            "model_forward_calls": 1, "default_model_CE_requested": False}


def generation_policy(*, eos_id: int, max_new_tokens: int = 256) -> dict:
    """Fixed raw-logit greedy policy; no inherited grammar or search controls."""
    require(type(eos_id) is int and eos_id >= 0, "explicit decoder EOS token required")
    require(type(max_new_tokens) is int and 0 < max_new_tokens <= 256,
            "bounded generation cap1..256 required")
    return {"max_new_tokens": max_new_tokens, "do_sample": False,
            "num_beams": 1, "num_beam_groups": 1, "num_return_sequences": 1,
            "use_cache": True, "cache_implementation": "dynamic", "cache_config": None,
            "repetition_penalty": 1.0, "encoder_repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0, "encoder_no_repeat_ngram_size": 0,
            "bad_words_ids": None, "force_words_ids": None, "constraints": None,
            "forced_bos_token_id": None, "forced_eos_token_id": None,
            "suppress_tokens": None, "begin_suppress_tokens": None, "sequence_bias": None,
            "guidance_scale": None, "watermarking_config": None,
            "exponential_decay_length_penalty": None, "renormalize_logits": False,
            "min_length": 0, "min_new_tokens": 0, "max_time": None, "stop_strings": None,
            "temperature": 1.0, "top_p": 1.0, "top_k": 50,
            "penalty_alpha": None, "dola_layers": None,
            "prompt_lookup_num_tokens": None, "assistant_early_exit": None,
            "prefill_chunk_size": None, "disable_compile": True,
            "eos_token_id": eos_id, "pad_token_id": eos_id,
            "output_hidden_states": False, "return_dict_in_generate": False}


def generation_policy_binding(model, decoder: Any, *, max_new_tokens: int = 256) -> dict:
    """Bind the actual source generation config and explicit same-call overrides."""
    policy = generation_policy(eos_id=decoder.eos_id, max_new_tokens=max_new_tokens)
    configuration = getattr(model, "generation_config", None)
    require(callable(getattr(configuration, "to_dict", None)),
            "actual model generation configuration required")
    source = configuration.to_dict()
    require(isinstance(source, dict), "model generation configuration must be an object")
    return {"model_generation_config": strict_json(canonical(source)), "overrides": policy}


def generate_with_cache_contract(model, inputs: Mapping, generation_kwargs: Mapping):
    """Keep Qwen cache positions while retaining the same injected LoRA modules.

    Installed PeftModelForCausalLM.generate replaces Qwen preparation with a
    wrapper that removes cache_position even for ordinary LoRA. Calling the
    unchanged Qwen generation method retains its native cache transport; the
    PEFT hook context and the already injected modules retain adapter behavior.
    """
    from contextlib import nullcontext

    base, _ = resolve_final_norm(model)
    if base is model:
        context = nullcontext()
    else:
        option = model.active_peft_config
        require(getattr(option, "alora_invocation_tokens", None) is None,
                "activated LoRA requires unsupported generation offsets")
        require(base.generation_config is model.generation_config,
                "LoRA and Qwen generation configuration must be the same object")
        hooks = getattr(model, "_enable_peft_forward_hooks", None)
        require(callable(hooks), "ordinary LoRA forward hook context required")
        context = hooks(**inputs, **generation_kwargs)
    with context:
        return base.generate(**inputs, **generation_kwargs)


def make_spatial_head(*, seed: int = 42):
    import torch
    from torch import nn
    require(type(seed) is int and seed >= 0, "nonnegative integer head seed required")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_WIDTH, dtype=torch.float32, device="cpu"), nn.GELU(),
                             nn.Linear(HIDDEN_WIDTH, 2, dtype=torch.float32, device="cpu"), nn.Sigmoid())
        nn.init.normal_(head[2].weight, mean=0.0, std=OUTPUT_INIT_STD)
        nn.init.zeros_(head[2].bias)
        return head


def resolve_final_norm(model):
    """One explicit locally installed Qwen final-language-norm path, no fallback."""
    import torch
    unwrap = getattr(model, "get_base_model", None)
    peft = getattr(model, "active_peft_config", None)
    if callable(unwrap):
        kind = getattr(peft, "peft_type", None)
        kind = getattr(kind, "value", kind)
        require(peft is not None and getattr(peft, "is_prompt_learning", None) is False
                and kind == "LORA", "only nonprompt LoRA PEFT wrappers are supported")
        base = unwrap()
    else:
        require(peft is None, "PEFT wrapper requires an explicit get_base_model path")
        base = model
    try:
        norm = base.model.language_model.norm
    except AttributeError as exc:
        raise ValueError("required Qwen model.language_model.norm path missing") from exc
    require(isinstance(norm, torch.nn.Module) and tuple(norm.weight.shape) == (HIDDEN_SIZE,),
            "Qwen final language norm shape differs")
    return base, norm


def _head_prediction(head, states, *, diagnostics: dict | None = None):
    import torch
    from torch.nn import functional as F

    require(diagnostics is None or isinstance(diagnostics, dict), "optional diagnostic dictionary required")
    require(states.ndim == 2 and states.shape[-1] == HIDDEN_SIZE and states.is_floating_point(),
            "selected final language states must be Nx2048")
    parameters = tuple(head.parameters())
    require(isinstance(head, torch.nn.Sequential) and len(head) == 4
            and isinstance(head[0], torch.nn.Linear) and isinstance(head[1], torch.nn.GELU)
            and isinstance(head[2], torch.nn.Linear) and isinstance(head[3], torch.nn.Sigmoid)
            and (head[0].in_features, head[0].out_features, head[2].in_features, head[2].out_features)
            == (HIDDEN_SIZE, HIDDEN_WIDTH, HIDDEN_WIDTH, 2), "registered head architecture differs")
    require(bool(parameters) and all(parameter.dtype == torch.float32 and parameter.device == states.device
                                     for parameter in parameters), "FP32 head and state devices must agree")
    with torch.autocast(device_type=states.device.type, enabled=False):
        raw_states = states.float()
        features = F.layer_norm(raw_states, (HIDDEN_SIZE,), weight=None, bias=None, eps=FEATURE_NORM_EPS)
        logits = []
        handle = None
        if diagnostics is not None:
            def capture_logits(module, inputs, output):
                require(not logits, "one same-call output Linear diagnostic required")
                logits.append(output.detach())
            handle = head[2].register_forward_hook(capture_logits)
        try:
            prediction = head(features)
        finally:
            if handle is not None:
                handle.remove()
    require(prediction.shape == (states.shape[0], 2) and prediction.dtype == torch.float32
            and bool(torch.isfinite(prediction).all()) and bool(((prediction >= 0) & (prediction <= 1)).all()),
            "finite FP32 Nx2 unit head prediction required")
    if diagnostics is not None:
        require(len(logits) == 1, "missing same-call pre-sigmoid logits")
        raw_rms = raw_states.detach().square().mean(dim=-1).sqrt()
        feature_rms = features.detach().square().mean(dim=-1).sqrt()
        output = prediction.detach()
        saturation = (output <= DIAGNOSTIC_SATURATION_EPS) | (output >= 1 - DIAGNOSTIC_SATURATION_EPS)
        diagnostics.update(
            schema="week5.native_spatial_head_activation_diagnostics.v1",
            head_version=HEAD_VERSION,
            point_count=int(states.shape[0]),
            raw_state_rms=raw_rms.tolist(),
            post_normalization_feature_rms=feature_rms.tolist(),
            pre_sigmoid_logit_min=float(logits[0].min().item()) if output.numel() else None,
            pre_sigmoid_logit_max=float(logits[0].max().item()) if output.numel() else None,
            output_unit_min=float(output.min().item()) if output.numel() else None,
            output_unit_max=float(output.max().item()) if output.numel() else None,
            saturated_axes_count=int(saturation.sum().item()),
            total_axes_count=int(output.numel()),
            saturation_epsilon=DIAGNOSTIC_SATURATION_EPS,
            head_forward_calls=1,
            diagnostic_model_forward_calls=0,
            diagnostic_head_forward_calls=0,
            diagnostic_backward_calls=0,
            scope="same live head call; observations detached after feature/output computation",
        )
    return prediction


def point_loss(prediction, target, *, native_width: int, native_height: int):
    import torch
    from torch.nn import functional as F
    width, height = dimensions(native_width, native_height)
    require(prediction.ndim == target.ndim == 2 and prediction.shape == target.shape
            and prediction.shape[1] == 2 and prediction.dtype == target.dtype == torch.float32
            and prediction.device == target.device and not target.requires_grad,
            "aligned FP32 Nx2 fixed point labels required")
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all())
            and bool(((prediction >= 0) & (prediction <= 1)).all())
            and bool(((target >= 0) & (target <= 1)).all()), "finite unit coordinates required")
    if prediction.shape[0] == 0:
        return prediction.sum() * 0
    per_axis = []
    for axis, size in enumerate((width, height)):
        if size == 1:
            require(bool((target[:, axis] == 0).all()), "degenerate native axis target must be zero")
            per_axis.append(prediction[:, axis] * 0)
        else:
            per_axis.append(F.smooth_l1_loss(prediction[:, axis], target[:, axis],
                                            beta=20 / max(1, size - 1), reduction="none"))
    return torch.stack(per_axis, dim=1).mean()


def forward_objective(model, head, batch: Mapping, plan: SpatialPlan) -> dict:
    """One real/synthetic model forward and one CE, plus the separate spatial loss."""
    import torch
    from torch.nn import functional as F
    require(isinstance(plan, SpatialPlan) and {"input_ids", "labels"} <= set(batch)
            and set(batch) <= INPUT_KEYS | {"labels"}, "explicit spatial plan and input-only tensor batch required")
    require(all(isinstance(value, torch.Tensor) for value in batch.values()), "model batch must contain tensors only")
    ids, labels = batch["input_ids"], batch["labels"]
    require(ids.dtype == labels.dtype == torch.long and ids.shape == labels.shape == (1, len(plan.full_ids))
            and tuple(ids[0].tolist()) == plan.full_ids and tuple(labels[0].tolist()) == plan.labels,
            "model batch differs from exact immutable spatial token plan")
    _, norm = resolve_final_norm(model)
    captures = []
    head_diagnostics = {"schema": "week5.native_spatial_head_activation_diagnostics.v1",
                        "head_version": HEAD_VERSION, "point_count": 0,
                        "raw_state_rms": [], "post_normalization_feature_rms": [],
                        "pre_sigmoid_logit_min": None, "pre_sigmoid_logit_max": None,
                        "output_unit_min": None, "output_unit_max": None,
                        "saturated_axes_count": 0, "total_axes_count": 0,
                        "saturation_epsilon": DIAGNOSTIC_SATURATION_EPS,
                        "head_forward_calls": 0, "diagnostic_model_forward_calls": 0,
                        "diagnostic_head_forward_calls": 0, "diagnostic_backward_calls": 0,
                        "scope": "no structural marker; head not called"}
    def capture(module, inputs, output):
        require(not captures and output.shape == (1, len(plan.full_ids), HIDDEN_SIZE),
                "one final language norm full forward required")
        if plan.marker_positions:
            index = torch.tensor(plan.marker_positions, dtype=torch.long, device=output.device)
            captures.append(_head_prediction(head, output[0].index_select(0, index), diagnostics=head_diagnostics))
        else:
            captures.append(torch.empty((0, 2), dtype=torch.float32, device=output.device))
    handle = norm.register_forward_hook(capture)
    try:
        output = model(**{key: value for key, value in batch.items() if key != "labels"},
                       use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    require(len(captures) == 1, "required final norm was not consumed exactly once")
    logits = output.logits
    require(logits.ndim == 3 and logits.shape[:2] == labels.shape and logits.is_floating_point()
            and logits.device == labels.device, "aligned causal logits required")
    active = labels[:, 1:] != -100
    selected = logits[:, :-1, :][active].float()
    selected_labels = labels[:, 1:][active]
    require(int(active.sum().item()) == plan.supervised_tokens and bool(torch.isfinite(selected).all()),
            "supervised token count or finite logits differs")
    ce = F.cross_entropy(selected, selected_labels)
    target = torch.tensor(plan.target_unit, dtype=torch.float32, device=logits.device).reshape(-1, 2)
    spatial = point_loss(captures[0], target, native_width=plan.native_width,
                        native_height=plan.native_height)
    return {"semantic_ce": ce, "point_loss": spatial, "total_loss": ce + spatial,
            "prediction_unit": captures[0], "supervised_tokens": plan.supervised_tokens,
            "point_count": len(plan.marker_positions), "manual_CE_calls": 1,
            "head_activation_diagnostics": head_diagnostics,
            "default_model_CE_requested": False, "model_forward_calls": 1}


class GenerationCapture:
    """Bounded post-token states: one prefix prefill, then cached single tokens only."""
    def __init__(self, model, head, *, prefix_ids: Sequence[int], max_new_tokens: int):
        require(bool(prefix_ids) and all(type(token) is int and token >= 0 for token in prefix_ids),
                "explicit immutable generation prefix IDs required")
        require(type(max_new_tokens) is int and 0 < max_new_tokens <= 256, "bounded generation cap1..256 required")
        self.base, self.norm = resolve_final_norm(model)
        self.head, self.prefix_ids = head, tuple(prefix_ids)
        self.max_new_tokens = max_new_tokens
        self.calls, self.next_position, self.pending = 0, 0, None
        self.consumed, self.states, self.handles = {}, {}, []

    def __enter__(self):
        require(not self.handles and self.calls == 0, "generation capture is single-use")
        self.handles = [self.base.register_forward_pre_hook(self._before, with_kwargs=True),
                        self.norm.register_forward_hook(self._after)]
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if exc_type is None:
            require(self.calls > 0 and self.pending is None, "generation final norm pending/missing")

    def _before(self, module, args, kwargs):
        import torch
        require(self.pending is None and not args, "unpaired or positional generation forward")
        ids = kwargs.get("input_ids")
        require(isinstance(ids, torch.Tensor) and ids.dtype == torch.long and ids.ndim == 2 and ids.shape[0] == 1,
                "one-view generation input IDs required")
        actual = tuple(ids[0].tolist())
        if self.calls == 0:
            require(actual == self.prefix_ids, "generation prefill differs from exact prefix")
        else:
            require(len(actual) == 1, "generation must use cached single-token forwards")
        require(self.calls < self.max_new_tokens and kwargs.get("use_cache") is True
                and kwargs.get("output_hidden_states") is not True,
                "generation budget/cache/hidden-state contract differs")
        cache_position = kwargs.get("cache_position")
        require(isinstance(cache_position, torch.Tensor) and cache_position.dtype == torch.long
                and cache_position.ndim == 1
                and tuple(cache_position.tolist()) == tuple(range(self.next_position,
                                                                 self.next_position + len(actual))),
                "cache_position must follow the complete consumed token stream")
        self.pending = (self.next_position, actual)
        self.next_position += len(actual)
        self.calls += 1

    def _after(self, module, inputs, output):
        require(self.pending is not None, "final norm lacks paired generation token input")
        start, actual = self.pending
        require(output.shape == (1, len(actual), HIDDEN_SIZE), "generation final norm geometry differs")
        if start >= len(self.prefix_ids):
            prediction = _head_prediction(self.head, output[0]).detach()
            for offset, identifier in enumerate(actual):
                position = start + offset
                require(position not in self.states, "duplicate consumed generation state")
                self.consumed[position] = identifier
                self.states[position] = prediction[offset]
        self.pending = None


def decode_generated(generated_ids: Sequence[int], decoder: Any, capture: GenerationCapture, *,
                     native_width: int, native_height: int) -> dict:
    """Never perform a repair forward for an unconsumed final marker token."""
    ids = tuple(generated_ids)
    require(0 < len(ids) <= capture.max_new_tokens and all(type(token) is int and token >= 0 for token in ids),
            "bounded generated IDs required")
    eos = ids.count(decoder.eos_id)
    require(eos <= 1 and (not eos or ids[-1] == decoder.eos_id), "ambiguous/nonterminal assistant end")
    semantic_ids = ids[:-1] if eos else ids
    require(bool(semantic_ids), "empty semantic model output")
    try:
        text = b"".join(decoder.pieces[token] for token in semantic_ids).decode("utf8", errors="strict")
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError("invalid generated ByteLevel bytes") from exc
    positions = marker_token_positions(text, semantic_ids, decoder, start_position=len(capture.prefix_ids))
    require(capture.calls == len(ids)
            and set(capture.consumed) == set(range(len(capture.prefix_ids),
                                                  len(capture.prefix_ids) + len(ids) - 1)),
            "cache must contain exactly returned generated stream except unconsumed final token")
    for position, token in capture.consumed.items():
        offset = position - len(capture.prefix_ids)
        require(0 <= offset < len(ids) and ids[offset] == token,
                "consumed cached token differs from actual generated token IDs")
    require(all(position in capture.states for position in positions),
            "marker token not consumed; no extra repair forward allowed")
    values = [capture.states[position].tolist() for position in positions]
    native = materialize_semantic_actions(text, values, native_width=native_width,
                                         native_height=native_height)
    return {"target_protocol": TARGET_PROTOCOL, "target_version": TARGET_VERSION,
            "head_version": HEAD_VERSION, "language_generation": text,
            "raw_spatial_output": values, "marker_token_positions": list(positions),
            "generation": canonical(native), "mapped_actions": native,
            "strict_joint_output_valid": True, "generation_forward_calls": capture.calls,
            "repair_forward_calls": 0}


def generate_native_spatial(model, head, inputs: Mapping, decoder: Any, *, native_width: int,
                            native_height: int, max_new_tokens: int = 256) -> dict:
    """Exactly one unconstrained greedy generate call; no GT or retry inputs."""
    import torch
    dimensions(native_width, native_height)
    require({"input_ids"} <= set(inputs) <= INPUT_KEYS, "generation inputs must be label-free")
    require(all(isinstance(value, torch.Tensor) for value in inputs.values()), "generation inputs must contain tensors only")
    require(not model.training and not head.training, "generation requires model/head eval mode")
    ids = inputs["input_ids"]
    require(ids.dtype == torch.long and ids.ndim == 2 and ids.shape[0] == 1,
            "one-view prefix input IDs required")
    prefix = tuple(ids[0].tolist())
    binding = generation_policy_binding(model, decoder, max_new_tokens=max_new_tokens)
    with (GenerationCapture(model, head, prefix_ids=prefix, max_new_tokens=max_new_tokens) as capture,
          torch.no_grad()):
        generated = generate_with_cache_contract(model, inputs, binding["overrides"])
    require(isinstance(generated, torch.Tensor) and generated.dtype == torch.long and generated.ndim == 2
            and generated.shape[0] == 1 and tuple(generated[0, :len(prefix)].tolist()) == prefix,
            "generated full sequence prefix differs")
    result = decode_generated(generated[0, len(prefix):].tolist(), decoder, capture,
                              native_width=native_width, native_height=native_height)
    result["generation_policy_binding"] = binding
    return result
