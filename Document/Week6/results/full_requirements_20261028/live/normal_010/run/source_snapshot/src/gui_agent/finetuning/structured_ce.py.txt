"""C10 structure6 / coordinate4 / other2 complete causal objective.

GT is label-side only. Decode saved token IDs using a caller-pinned ByteLevel
vocabulary, then match structured canonical JSON value spans. A token that
mixes numeric bytes with syntax is weighted whole and counted explicitly.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

VALUE_WEIGHT = 4.0
NON_COORDINATE_WEIGHT = 2.0
STRUCTURE_WEIGHT = 6.0
STRATEGY = "resized_native_structure6_coordinate4_other2"
NUMERIC_FIELDS = frozenset({'x', 'y'})
FIELDS = {
    **{kind: {'x', 'y'} for kind in ('click', 'double_click', 'right_click', 'move')},
    'scroll': {'x', 'y', 'scroll_amount'}, 'drag': {'start', 'end'},
    'type_text': {'text'}, 'key_press': {'key'}, 'hotkey': {'keys'}, 'wait': {'duration'},
}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_json(data: str | bytes) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON member")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


@dataclass(frozen=True)
class TokenBytes:
    pieces: Any
    eos_id: int
    source_sha256: str

    @classmethod
    def from_frozen_json(cls, data: bytes, *, expected_sha256: str, eos_id: int):
        require(hashlib.sha256(data).hexdigest() == expected_sha256, "tokenizer metadata changed")
        doc = strict_json(data)
        require(doc["model"]["type"] == "BPE" and doc["decoder"]["type"] == "ByteLevel",
                "only the pinned ByteLevel BPE decoder is supported")
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        inverse = {chr(value): value for value in visible}
        extra = 0
        for value in range(256):
            if value not in visible:
                inverse[chr(256 + extra)] = value
                extra += 1
        pieces = {}
        for word, identifier in doc["model"]["vocab"].items():
            require(isinstance(word, str) and bool(word) and type(identifier) is int and identifier >= 0
                    and identifier not in pieces, "invalid vocabulary IDs")
            try:
                pieces[identifier] = bytes(inverse[ch] for ch in word)
            except KeyError as exc:
                raise ValueError("unknown ByteLevel symbol") from exc
        seen_added = set()
        for entry in doc["added_tokens"]:
            identifier, content = entry["id"], entry["content"]
            require(type(identifier) is int and identifier >= 0 and identifier not in seen_added
                    and isinstance(content, str) and bool(content), "invalid added token")
            seen_added.add(identifier)
            value = content.encode("utf8")
            require(identifier not in pieces or pieces[identifier] == value, "conflicting added token")
            pieces[identifier] = value
        require(type(eos_id) is int and pieces.get(eos_id) == b"<|im_end|>", "explicit assistant end token required")
        return cls(MappingProxyType(pieces), eos_id, expected_sha256)


def numeric_value_spans(text: str) -> list[tuple[int, int]]:
    """Exact UTF8 coordinate VALUE spans; all text, syntax and other numbers excluded."""
    target = strict_json(text)
    require(isinstance(target, list) and bool(target) and canonical(target) == text,
            'canonical nonempty whole array required')
    for action in target:
        require(isinstance(action, dict) and action.get('action_type') in FIELDS, 'unknown action')
        require(set(action) == FIELDS[action['action_type']] | {'action_type'}, 'whole fields differ')
        points = [action[name] for name in ('start', 'end')] if action['action_type'] == 'drag' else [action]
        for point in points:
            if action['action_type'] == 'drag':
                require(isinstance(point, dict) and set(point) == {'x', 'y'}, 'drag point fields differ')
            for key in set(point) & NUMERIC_FIELDS:
                require(type(point[key]) is int and point[key] >= 0, 'coordinate VALUE must be nonnegative integer')
    offsets = [0]
    for ch in text:
        offsets.append(offsets[-1] + len(ch.encode('utf8')))
    parser = json.JSONDecoder()
    spans = []
    def visit(position):
        if text[position] == '[':
            position += 1
            while text[position] != ']':
                position = visit(position)
                if text[position] == ',': position += 1
            return position + 1
        if text[position] == '{':
            position += 1
            while text[position] != '}':
                key, position = parser.raw_decode(text, position)
                require(text[position] == ':', 'member boundary')
                position += 1
                begin = position
                position = visit(position)
                if key in NUMERIC_FIELDS:
                    spans.append((offsets[begin], offsets[position]))
                if text[position] == ',': position += 1
            return position + 1
        _, end = parser.raw_decode(text, position)
        return end
    require(visit(0) == len(text), 'whole JSON boundary')
    return sorted(spans)


def structural_byte_positions(text: str) -> tuple[int, ...]:
    """JSON punctuation outside strings; never punctuation inside copied text."""
    numeric_value_spans(text)  # Reuse canonical whole-array and action-schema validation.
    positions = []
    inside = escaped = False
    for position, value in enumerate(text.encode("utf8")):
        if inside:
            if escaped:
                escaped = False
            elif value == 92:
                escaped = True
            elif value == 34:
                inside = False
        elif value == 34:
            inside = True
        elif value in b"{}[],:":
            positions.append(position)
    require(not inside and not escaped and bool(positions), "complete JSON structure required")
    return tuple(positions)


@dataclass(frozen=True)
class WeightPlan:
    weights: tuple[float, ...]
    numeric_tokens: tuple[int, ...]
    boundary_mixed_tokens: tuple[int, ...]
    value_span_count: int
    masked_numeric_tokens: int
    structural_tokens: tuple[int, ...]
    structural_boundary_mixed_tokens: tuple[int, ...]
    structural_byte_count: int
    structure_numeric_overlap_tokens: tuple[int, ...]


def value_weights(*, full_ids, prefix_ids, labels, target_json: str, decoder: TokenBytes) -> WeightPlan:
    """Label-position weights; public inputs/IDs/labels are never rewritten."""
    full_ids, prefix_ids, labels = tuple(full_ids), tuple(prefix_ids), tuple(labels)
    require(all(type(i) is int and i >= 0 for i in full_ids + prefix_ids), "integer input IDs required")
    require(len(labels) == len(full_ids) and all(type(i) is int for i in labels), "label shape/type differs")
    boundary = len(prefix_ids)
    require(0 < boundary < len(full_ids) and full_ids[:boundary] == prefix_ids, "expanded prefix differs")
    require(labels[:boundary] == (-100,) * boundary, "prefix must stay masked")
    require(all(label in (-100, identifier) for label, identifier in zip(labels, full_ids, strict=True)),
            "active labels differ from immutable full IDs")
    suffix = full_ids[boundary:]
    try:
        pieces = [decoder.pieces[i] for i in suffix]
    except KeyError as exc:
        raise ValueError("saved suffix has unknown token") from exc
    expected = (target_json + "<|im_end|>\n").encode("utf8")
    require(b"".join(pieces) == expected, "saved whole suffix does not reconstruct exactly")
    require(suffix.count(decoder.eos_id) == 1, "ambiguous assistant end")
    eos_position = boundary + suffix.index(decoder.eos_id)
    require(labels[eos_position] == decoder.eos_id, "assistant end must remain supervised")
    spans = numeric_value_spans(target_json)
    weights = [0.0 if value == -100 else NON_COORDINATE_WEIGHT for value in labels]
    numeric, mixed, masked, position = [], [], 0, 0
    for offset, piece in enumerate(pieces, boundary):
        end = position + len(piece)
        hits = [(a, b) for a, b in spans if a < end and b > position]
        if hits:
            numeric.append(offset)
            covered = sum(min(b, end) - max(a, position) for a, b in hits)
            if covered < len(piece):
                mixed.append(offset)
            if labels[offset] == -100:
                masked += 1
            else:
                weights[offset] = VALUE_WEIGHT
        position = end
    require(weights[eos_position] == NON_COORDINATE_WEIGHT, "end-token weight changed")
    structural_positions = structural_byte_positions(target_json)
    structural, structure_mixed, position = [], [], 0
    for offset, piece in enumerate(pieces, boundary):
        end = position + len(piece)
        covered = sum(position <= point < end for point in structural_positions)
        if covered:
            structural.append(offset)
            if covered < len(piece):
                structure_mixed.append(offset)
            if labels[offset] != -100:
                weights[offset] = STRUCTURE_WEIGHT
        position = end
    require(weights[eos_position] == NON_COORDINATE_WEIGHT, "structural rule changed EOS")
    return WeightPlan(tuple(weights), tuple(numeric), tuple(mixed), len(spans), masked,
                      tuple(structural), tuple(structure_mixed), len(structural_positions),
                      tuple(sorted(set(structural) & set(numeric))))


def load_bound_decoder(path: str, expected_sha256: str) -> TokenBytes:
    """Read caller-pinned metadata only; never create a tokenizer or encode."""
    source = Path(path)
    require(source.is_absolute(), "caller absolute tokenizer metadata path required")
    data = source.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "tokenizer metadata changed")
    doc = strict_json(data)
    ends = [entry["id"] for entry in doc["added_tokens"] if entry["content"] == "<|im_end|>"]
    require(len(ends) == 1, "one explicit assistant end in pinned metadata required")
    return TokenBytes.from_frozen_json(data, expected_sha256=expected_sha256, eos_id=ends[0])


def weights_from_encoded_sample(encoded, target, decoder: TokenBytes) -> WeightPlan:
    """Use the existing one-view encoder output; contiguous masked prefix only."""
    import torch

    ids, labels = encoded["input_ids"], encoded["labels"]
    require(ids.dtype == labels.dtype == torch.long and ids.ndim == labels.ndim == 2
            and ids.shape == labels.shape and ids.shape[0] == 1,
            "one-view long IDs and labels required")
    full, actual = ids[0].tolist(), labels[0].tolist()
    boundary = next((i for i, label in enumerate(actual) if label != -100), len(actual))
    require(0 < boundary < len(full), "contiguous masked generation prefix required")
    require(actual[boundary:] == full[boundary:], "whole assistant suffix must remain supervised")
    # If '[' had been masked, this exact target reconstruction fails closed.
    return value_weights(full_ids=full, prefix_ids=full[:boundary], labels=actual,
                         target_json=canonical(target), decoder=decoder)


def causal_ce(logits, labels, weights):
    """One CE on active shifted positions; never cast the whole visual prefix."""
    import torch
    from torch.nn import functional as F

    require(logits.ndim == 3 and logits.is_floating_point(), "floating BxTxV logits required")
    require(labels.dtype == torch.long and labels.shape == logits.shape[:2], "long BxT labels required")
    require(weights.shape == labels.shape and weights.is_floating_point(), "floating aligned weights required")
    require(labels.device == weights.device == logits.device, "device mismatch")
    require(not weights.requires_grad, "fixed token weights cannot be trainable")
    require(bool(torch.isfinite(weights).all()), "nonfinite token weight")
    require(bool((weights[labels == -100] == 0).all()), "masked token must have zero weight")
    require(bool(((weights[labels != -100] == NON_COORDINATE_WEIGHT) | (weights[labels != -100] == VALUE_WEIGHT) | (weights[labels != -100] == STRUCTURE_WEIGHT)).all()),
            "active weights must be fixed2, fixed4 or fixed6")
    shifted = labels[:, 1:]
    active = shifted != -100
    count = int(active.sum().item())
    require(count > 0, "no shifted supervised tokens")
    actual_labels = shifted[active]
    require(bool(((actual_labels >= 0) & (actual_labels < logits.shape[-1])).all()), "label outside vocabulary")
    selected = logits[:, :-1, :][active]
    # Keep float64 in numerical CPU tests; cast half/bfloat16 only after selecting.
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    losses = F.cross_entropy(selected, actual_labels, reduction="none")
    active_weights = weights[:, 1:][active].to(losses.dtype)
    return {"raw_ce": losses.mean(), "weighted_ce": (losses * active_weights).sum() / count,
            "supervised_tokens": count, "nominal_token_weight_mass": float(active_weights.sum().item()),
            "CE_calls": 1, "normalizer": "original shifted active-token count"}


def forward_objective(model, batch, *, strategy: str = "uniform", token_weights=None):
    """Future integration sketch: legacy model.loss or one manual CE, exclusively."""
    if strategy == "uniform":
        loss = model(**batch).loss
        return {"raw_ce": loss, "weighted_ce": loss, "strategy": "uniform", "manual_CE_calls": 0}
    require(strategy == STRATEGY and token_weights is not None, "unknown/unprepared field strategy")
    require("labels" in batch, "immutable training labels required")
    # No labels means no default transformer CE alongside the manual field CE.
    output = model(**{key: value for key, value in batch.items() if key != "labels"})
    return {**causal_ce(output.logits, batch["labels"], token_weights), "strategy": STRATEGY,
            "manual_CE_calls": 1, "default_model_CE_requested": False}
