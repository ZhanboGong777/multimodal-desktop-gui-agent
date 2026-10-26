"""Train-only, GT-independent display-number permutations of preselected regions.

This module never detects regions, loads a processor/model, or executes actions.
The frozen v3 selector runs before labels are considered. The derived spatial_v1
snapshot contains only that <=64-item set, so selection becomes serialization.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .candidate_grounding import snapshot_hash, validate_snapshot
from .evaluation import CLICK_TYPES, canonical_json, validate_actions
from .pointer_protocol import CONTEXT_POLICY, select_candidates, selection_binding

VERSION = "week5.pointer-numbering.v1"
VIEWS = 4
SEED = 42
MOUSE = CLICK_TYPES | {"move", "scroll", "drag"}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def numbered_view(sample: Mapping[str, Any], *, view_index: int, seed: int = SEED) -> dict[str, Any]:
    """Return visible evidence plus label-side bijection; never access GT fields."""
    _require(type(seed) is int and seed == SEED, "numbering seed must be the registered42")
    _require(type(view_index) is int and 0 <= view_index < VIEWS, "view_index must be0..3")
    identifier = sample["sample_id"]
    _require(isinstance(identifier, str) and bool(identifier.strip()), "nonempty parent sample_id required")
    snapshot = sample["candidate_snapshot"]
    _require(snapshot.get("compact_policy") == CONTEXT_POLICY, "parent preselection must be query_relations_v3")
    _require((snapshot["image_width"], snapshot["image_height"]) == (sample["image_width"], sample["image_height"]), "parent dimensions differ")
    validate_snapshot(dict(snapshot), sample["image_path"])
    context = {"instruction": sample["instruction"], "actor_step": sample["actor_step"]}
    parent_binding = selection_binding(snapshot, **context)
    _require(sample.get("candidate_selection_evidence") == parent_binding, "parent selection binding differs")
    selected = select_candidates(snapshot, **context)
    _require(0 < len(selected) <= 64, "preselected table must have1..64 regions")
    order = list(range(1, len(selected) + 1))
    if view_index:
        def shuffle_key(index: int) -> tuple[str, int]:
            material = {"version": VERSION, "seed": seed, "parent_sample_id": identifier,
                        "view_index": view_index, "parent_display_index": index}
            return hashlib.sha256(canonical_json(material).encode("utf8")).hexdigest(), index
        order.sort(key=shuffle_key)
    candidates = []
    for new_index, old_index in enumerate(order, 1):
        original = selected[old_index - 1]
        candidate = copy.deepcopy(original)
        candidate["candidate_id"] = f"v{new_index:03d}"
        candidate["original_id"] = original["candidate_id"]
        candidate["raw_source_snapshot_sha256"] = parent_binding["snapshot_sha256"]
        candidate["original_detection_evidence_inherited"] = True
        candidates.append(candidate)
    derived = copy.deepcopy(dict(snapshot))
    derived.update(candidates=candidates, compact_policy="spatial_v1", preselection_policy=CONTEXT_POLICY,
                   numbering_version=VERSION, numbering_view_index=view_index, numbering_seed=seed,
                   parent_sample_id=identifier, parent_raw_snapshot_sha256=snapshot_hash(dict(snapshot)),
                   parent_context_sha256=parent_binding["context_sha256"],
                   raw_detection_metadata_inherited=True, detector_rerun=False)
    actual = select_candidates(derived, **context)
    _require(actual == candidates, "derived spatial_v1 must serialize the entire numbered set")
    return {"candidate_snapshot": derived, "parent_selection_evidence": parent_binding,
            "old_to_new": {old_index: new_index for new_index, old_index in enumerate(order, 1)},
            "new_to_old": {new_index: old_index for new_index, old_index in enumerate(order, 1)},
            "view_index": view_index, "seed": seed}


def verify_numbered_view(sample: Mapping[str, Any], view: Mapping[str, Any]) -> None:
    """Fail closed if context, raw evidence, selected IDs or derived order changed."""
    expected = numbered_view(sample, view_index=view["view_index"], seed=view["seed"])
    _require(dict(view) == expected, "numbered view binding/evidence differs from fresh derivation")


def remap_target(target: Sequence[Mapping[str, Any]], old_to_new: Mapping[int, int]) -> list[dict[str, Any]]:
    """Relabel every pointer endpoint; preserve all residual/nonmouse fields."""
    mapping = dict(old_to_new)
    _require(bool(mapping) and all(type(k) is int and type(v) is int for k, v in mapping.items()), "integer bijection required")
    expected = set(range(1, len(mapping) + 1))
    _require(set(mapping) == set(mapping.values()) == expected, "mapping must be a complete table bijection")
    _require(isinstance(target, (list, tuple)) and bool(target), "complete nonempty target array required")
    result = []
    for item in target:
        _require(isinstance(item, Mapping), "target action must be an object")
        action = copy.deepcopy(dict(item))
        kind = action.get("action_type")
        fields = ("start_candidate", "end_candidate") if kind == "drag" else ("candidate",)
        if kind in MOUSE:
            required = {"action_type", "start_candidate", "start_dx", "start_dy", "end_candidate", "end_dx", "end_dy"} if kind == "drag" else {"action_type", "candidate", "dx", "dy"} | ({"scroll_amount"} if kind == "scroll" else set())
            _require(set(action) == required, "unexpected/missing pointer target fields")
            for field in fields:
                _require(type(action[field]) is int and action[field] in mapping, "pointer target ID is outside the selected table")
                action[field] = mapping[action[field]]
            offsets = ("start_dx", "start_dy", "end_dx", "end_dy") if kind == "drag" else ("dx", "dy")
            _require(all(type(action[name]) is int for name in offsets), "pointer offsets must be integers")
            if kind == "scroll":
                _require(type(action["scroll_amount"]) is int and action["scroll_amount"] != 0, "scroll_amount must be nonzero integer")
        else:
            validate_actions([action])
        result.append(action)
    return result
