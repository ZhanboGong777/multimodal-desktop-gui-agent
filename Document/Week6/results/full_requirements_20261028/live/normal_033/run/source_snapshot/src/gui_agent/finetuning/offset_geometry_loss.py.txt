"""Additive candidate4/offset8 CE plus teacher-forced digit-place geometry.

No model/processor/tokenizer instance or image is created. Integer spans are
label-side only. Only bare single ASCII digit tokens receive auxiliary geometry;
other tokens keep full main CE. Candidate numbers never receive ordinal loss.
This teacher-forced surrogate cannot measure free-decoding insertion/sign errors.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from types import MappingProxyType

from . import numeric_value_loss as legacy

CANDIDATE_WEIGHT = 4.0
OFFSET_WEIGHT = 8.0
GEOMETRY_COEFFICIENT = 8.0
STRATEGY = "candidate4_offset8_teacher_forced_digit_place_geometry8"
CANDIDATE_FIELDS = frozenset({"candidate", "start_candidate", "end_candidate"})
X_FIELDS = frozenset({"dx", "start_dx", "end_dx"})
Y_FIELDS = frozenset({"dy", "start_dy", "end_dy"})
OFFSET_FIELDS = X_FIELDS | Y_FIELDS
require = legacy.require


@dataclass(frozen=True)
class NumberVocabulary:
    source_sha256: str
    shapes: object

    @classmethod
    def from_decoder(cls, decoder):
        """Frozen bare digits0..9 only; no invented multi-digit token classes."""
        items = []
        for identifier, piece in decoder.pieces.items():
            if len(piece) == 1 and 48 <= piece[0] <= 57:
                items.append((identifier, piece[0] - 48))
        require(len(items) == 10 and {value for _, value in items} == set(range(10)),
                "one pinned token per bare ASCII digit0..9 required")
        frozen = {(b"", b""): tuple(sorted(items))}
        return cls(decoder.source_sha256, MappingProxyType(frozen))


@dataclass(frozen=True)
class FieldValue:
    action_index: int
    field: str
    start: int
    end: int
    value: int


def field_value_spans(text):
    """Match the exact old canonical VALUE contract, adding field identities."""
    original_spans = legacy.numeric_value_spans(text)
    byte_offsets = [0]
    for char in text:
        byte_offsets.append(byte_offsets[-1] + len(char.encode("utf8")))
    decoder = legacy.json.JSONDecoder()
    position, action_index, values = 1, 0, []
    while text[position] != "]":
        _, object_end = decoder.raw_decode(text, position)
        member = position + 1
        while text[member] != "}":
            key, member = decoder.raw_decode(text, member)
            require(text[member] == ":", "invalid value boundary")
            begin = member + 1
            value, member = decoder.raw_decode(text, begin)
            if key in legacy.NUMERIC_FIELDS:
                values.append(FieldValue(action_index, key, byte_offsets[begin], byte_offsets[member], value))
            if text[member] == ",":
                member += 1
        require(member + 1 == object_end, "object boundary differs")
        position = object_end + 1 if text[object_end] == "," else object_end
        action_index += 1
    require([(value.start, value.end) for value in values] == original_spans, "legacy VALUE spans changed")
    return tuple(values)


@dataclass(frozen=True)
class GeometryEntry:
    action_index: int
    field: str
    gt: int
    absolute_gt: int
    gt_digit: int
    right_digits: int
    place_multiplier: int
    token_index: int
    native_axis_span: int
    prefix_hex: str
    suffix_hex: str
    token_ids: tuple[int, ...]
    magnitudes: tuple[int, ...]


@dataclass(frozen=True)
class SkippedEntry:
    action_index: int
    field: str
    gt: int
    reason: str
    magnitude_digit_index: int | None = None


@dataclass(frozen=True)
class FieldPlan:
    weights: tuple[float, ...]
    legacy_weights: tuple[float, ...]
    labels: tuple[int, ...]
    numeric_tokens: tuple[int, ...]
    candidate_tokens: tuple[int, ...]
    offset_tokens: tuple[int, ...]
    mixed_tokens: tuple[int, ...]
    cross_field_tokens: tuple[int, ...]
    values: tuple[FieldValue, ...]
    eligible: tuple[GeometryEntry, ...]
    skipped: tuple[SkippedEntry, ...]
    masked_numeric_tokens: int
    vocabulary_source_sha256: str
    native_width: int
    native_height: int
    sign_tokens: tuple[int, ...]


def prepare_field_plan(*, full_ids, prefix_ids, labels, target_json, decoder,
                       native_width, native_height, vocabulary=None):
    """Preserve all IDs/labels; strengthen signed offsets including mixed syntax."""
    require(type(native_width) is type(native_height) is int and native_width > 0 and native_height > 0,
            "positive original-native axis spans required")
    baseline = legacy.value_weights(full_ids=full_ids, prefix_ids=prefix_ids, labels=labels,
                                    target_json=target_json, decoder=decoder)
    vocabulary = vocabulary or NumberVocabulary.from_decoder(decoder)
    require(vocabulary.source_sha256 == decoder.source_sha256, "frozen numeric vocabulary source differs")
    values = field_value_spans(target_json)
    intervals, candidate_tokens, offset_tokens, cross_field = [], [], [], []
    weights = list(baseline.weights)
    byte_position = 0
    for token_index in range(len(prefix_ids), len(full_ids)):
        piece = decoder.pieces[full_ids[token_index]]
        end = byte_position + len(piece)
        hits = [value for value in values if value.start < end and value.end > byte_position]
        intervals.append((token_index, byte_position, end, piece, hits))
        if any(value.field in CANDIDATE_FIELDS for value in hits):
            candidate_tokens.append(token_index)
        if any(value.field in OFFSET_FIELDS for value in hits):
            offset_tokens.append(token_index)
            # Same old whole-token mixed rule: an offset overlap weights the
            # complete active token8; candidate-only overlap remains4. Record
            # any cross-VALUE case, which is never auxiliary-eligible.
            if labels[token_index] != -100:
                weights[token_index] = OFFSET_WEIGHT
        if len(hits) > 1:
            cross_field.append(token_index)
        byte_position = end
    signs = tuple(index for index, begin, end, _, hits in intervals
                  if any(value.field in OFFSET_FIELDS and value.value < 0
                         and begin <= value.start < end for value in hits))
    eligible, skipped = [], []
    target_bytes = target_json.encode("utf8")
    alternatives = vocabulary.shapes[(b"", b"")]
    require(all(decoder.pieces[identifier] == str(digit).encode("ascii") for identifier, digit in alternatives),
            "frozen bare-digit vocabulary changed")
    for value in values:
        if value.field in CANDIDATE_FIELDS:
            skipped.append(SkippedEntry(value.action_index, value.field, value.value, "candidate_categorical_no_ordinal"))
            continue
        begin = value.start + int(value.value < 0)
        end = value.end
        digits = target_bytes[begin:end]
        require(digits == str(abs(value.value)).encode("ascii"), "integer magnitude parsing differs")
        axis_size = native_width if value.field in X_FIELDS else native_height
        axis_span = max(1, axis_size - 1)
        require(abs(value.value) <= axis_span, "original-native offset exceeds its possible axis span")
        for digit_index, digit_byte in enumerate(digits):
            reason = None
            digit_begin = begin + digit_index
            overlaps = [item for item in intervals if item[1] <= digit_begin < item[2]]
            require(len(overlaps) == 1, "saved token bytes do not cover one target digit")
            token_index, token_begin, token_end, piece, hits = overlaps[0]
            if token_begin != digit_begin or token_end != digit_begin + 1 or piece != bytes([digit_byte]):
                reason = "not_bare_single_ASCII_digit_token"
            elif len(hits) != 1 or hits[0] != value:
                reason = "token_overlaps_other_numeric_value"
            elif labels[token_index] == -100:
                reason = "digit_token_masked"
            elif (full_ids[token_index], digit_byte - 48) not in alternatives:
                reason = "GT_digit_not_in_frozen_vocabulary"
            if reason is not None:
                skipped.append(SkippedEntry(value.action_index, value.field, value.value, reason, digit_index))
            else:
                right_digits = len(digits) - digit_index - 1
                eligible.append(GeometryEntry(value.action_index, value.field, value.value, abs(value.value),
                    digit_byte - 48, right_digits, 10 ** right_digits, token_index, axis_span,
                    "", "", tuple(item[0] for item in alternatives),
                    tuple(item[1] for item in alternatives)))
    require(all(weights[index] == 0 for index, label in enumerate(labels) if label == -100), "prefix/pad supervision changed")
    return FieldPlan(tuple(weights), baseline.weights, tuple(labels), baseline.numeric_tokens,
                     tuple(candidate_tokens), tuple(offset_tokens), baseline.boundary_mixed_tokens,
                     tuple(cross_field), values, tuple(eligible), tuple(skipped), baseline.masked_numeric_tokens,
                     vocabulary.source_sha256, native_width, native_height, signs)


def causal_geometry_loss(logits, labels, plan):
    """One CE plus teacher-forced digit E|d-GTd|*decimal_place/native_axis."""
    import torch
    from torch.nn import functional as F

    require(logits.ndim == 3 and logits.is_floating_point() and logits.shape[0] == 1,
            "one-view floating BxTxV logits required")
    require(labels.dtype == torch.long and labels.shape == logits.shape[:2]
            and labels[0].tolist() == list(plan.labels) and labels.device == logits.device, "immutable aligned labels differ")
    weights = torch.tensor(plan.weights, dtype=torch.float64 if logits.dtype == torch.float64 else torch.float32,
                           device=logits.device)[None, :]
    require(bool((weights[labels == -100] == 0).all()) and
            bool(((weights[labels != -100] == 1) | (weights[labels != -100] == 4) | (weights[labels != -100] == 8)).all()),
            "fixed1/4/8 weights or prefix changed")
    shifted = labels[:, 1:]
    active = shifted != -100
    count = int(active.sum().item())
    require(count > 0, "no shifted supervised tokens")
    actual = shifted[active]
    require(bool(((actual >= 0) & (actual < logits.shape[-1])).all()), "label outside logit vocabulary")
    selected = logits[:, :-1, :][active]
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    losses = F.cross_entropy(selected, actual, reduction="none")
    main = (losses * weights[:, 1:][active].to(losses.dtype)).sum() / count
    errors = []
    for entry in plan.eligible:
        require(0 < entry.token_index < labels.shape[1] and labels[0, entry.token_index] != -100,
                "auxiliary position is not active causal supervision")
        require(all(0 <= identifier < logits.shape[-1] for identifier in entry.token_ids), "auxiliary token outside logit vocabulary")
        indices = torch.tensor(entry.token_ids, dtype=torch.long, device=logits.device)
        shape_logits = logits[0, entry.token_index - 1].index_select(0, indices)
        if shape_logits.dtype in (torch.float16, torch.bfloat16):
            shape_logits = shape_logits.float()
        probabilities = torch.softmax(shape_logits, dim=0)
        distances = torch.tensor([abs(value - entry.gt_digit) * entry.place_multiplier / entry.native_axis_span for value in entry.magnitudes],
                                 dtype=shape_logits.dtype, device=logits.device)
        errors.append((probabilities * distances).sum())
    error_sum = torch.stack(errors).sum() if errors else main * 0
    auxiliary = GEOMETRY_COEFFICIENT * error_sum / count
    return {"raw_ce": losses.mean(), "main_weighted_ce": main, "geometry_expected_absolute_error_sum": error_sum,
            "geometry_auxiliary": auxiliary, "total_loss": main + auxiliary, "supervised_tokens": count,
            "nominal_token_weight_mass": float(weights[:, 1:][active].sum().item()),
            "geometry_entries": len(plan.eligible),
            "geometry_auxiliary_skipped": sum(entry.field in OFFSET_FIELDS for entry in plan.skipped),
            "geometry_error_values": errors, "CE_calls": 1, "normalizer": "original shifted active-token count"}


def forward_objective(model, batch, plan, *, strategy=STRATEGY):
    """Synthetic/future API: one forward with labels removed, one manual CE."""
    import torch

    require(strategy in {STRATEGY, legacy.STRATEGY}, "unknown offset geometry strategy")
    require("labels" in batch, "immutable labels required")
    output = model(**{key: value for key, value in batch.items() if key != "labels"})
    if strategy == legacy.STRATEGY:
        weights = torch.tensor(plan.legacy_weights, dtype=torch.float64 if output.logits.dtype == torch.float64 else torch.float32,
                               device=output.logits.device)[None, :]
        result = legacy.causal_ce(output.logits, batch["labels"], weights)
        return {**result, "total_loss": result["weighted_ce"], "strategy": strategy,
                "manual_CE_calls": 1, "default_model_CE_requested": False}
    return {**causal_geometry_loss(output.logits, batch["labels"], plan), "strategy": STRATEGY,
            "manual_CE_calls": 1, "default_model_CE_requested": False}


def plan_from_encoded_sample(encoded, target, decoder, *, native_width, native_height, vocabulary=None):
    """Use an existing encoder output; never alter IDs, mask, target or image."""
    # Old adapter performs exact one-view dtype, contiguous prefix, full suffix,
    # whole target and EOS validation. It never calls a processor or model.
    legacy.weights_from_encoded_sample(encoded, target, decoder)
    full = encoded["input_ids"][0].tolist()
    labels = encoded["labels"][0].tolist()
    boundary = next(index for index, label in enumerate(labels) if label != -100)
    return prepare_field_plan(full_ids=full, prefix_ids=full[:boundary], labels=labels,
                              target_json=legacy.canonical(target), decoder=decoder,
                              native_width=native_width, native_height=native_height, vocabulary=vocabulary)


def plan_statistics(plan):
    """Label-side coverage; categorical markers never count as skipped aux."""
    candidates = [value for value in plan.values if value.field in CANDIDATE_FIELDS]
    offsets = [value for value in plan.values if value.field in OFFSET_FIELDS]
    skipped = [entry for entry in plan.skipped if entry.field in OFFSET_FIELDS]
    def active_count(positions):
        return sum(plan.labels[index] != -100 for index in positions)
    return {
        "numeric_active_tokens": active_count(plan.numeric_tokens),
        "boundary_mixed_tokens": len(plan.mixed_tokens),
        "masked_numeric_tokens": plan.masked_numeric_tokens,
        "candidate_active_tokens": active_count(plan.candidate_tokens),
        "offset_active_tokens": active_count(plan.offset_tokens),
        "offset_sign_active_tokens": active_count(plan.sign_tokens),
        "candidate_value_fields": len(candidates), "offset_value_fields": len(offsets),
        "offset_signed_fields": sum(value.value < 0 for value in offsets),
        "offset_digit_positions": sum(len(str(abs(value.value))) for value in offsets),
        "geometry_entries": len(plan.eligible), "geometry_auxiliary_skipped": len(skipped),
        "geometry_skip_reasons": dict(sorted(Counter(entry.reason for entry in skipped).items())),
        "candidate_categorical_fields": len(candidates),
        "cross_field_tokens": len(plan.cross_field_tokens),
        "native_width": plan.native_width, "native_height": plan.native_height,
        "vocabulary_source_sha256": plan.vocabulary_source_sha256,
        "geometry_entries_sha256": hashlib.sha256(legacy.canonical([asdict(entry) for entry in plan.eligible]).encode("utf8")).hexdigest(),
        "token_weights_sha256": hashlib.sha256(legacy.canonical(plan.weights).encode("utf8")).hexdigest(),
    }
