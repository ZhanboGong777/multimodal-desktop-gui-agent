"""Opt-in compact candidate supervision; original native targets stay separate.

Legacy selectors read detection evidence only. The opt-in contextual version
also reads the supplied goal and current subtask. Semantic reviews remain
label-side AI annotations, never inference inputs or proof of a hit-test.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .candidate_grounding import ROLES, center, native_box, snapshot_hash, validate_snapshot
from .evaluation import (
    CLICK_TYPES,
    canonical_json,
    file_sha256,
    strict_json_loads,
    validate_actions,
    value_sha256,
)

PROTOCOL_VERSION = "week5.candidate-compact.v1"
POINTER_PROTOCOLS = ("native", "candidate_compact")
MAX_CANDIDATES = 64
SOURCE_QUOTA = 32
GRID_SIDE = 4
TEXT_CHARACTERS = 24
MOUSE_TYPES = CLICK_TYPES | {"move", "scroll", "drag"}
CONTEXT_POLICY = "query_relations_v3"


def _candidate_evidence(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    width, height = snapshot["image_width"], snapshot["image_height"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("candidate dimensions must be positive integers")
    candidates = snapshot["candidates"]
    if not isinstance(candidates, list):
        raise TypeError("candidate evidence must be an array")
    ids, boxes = set(), set()
    for item in candidates:
        if not isinstance(item, dict):
            raise TypeError("candidate must be an object")
        identifier = item["candidate_id"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("candidate IDs must be unique nonempty strings")
        box = native_box(item["bbox"], width, height)
        geometry = tuple(box[k] for k in ("left", "top", "right", "bottom"))
        if geometry in boxes:
            raise ValueError("duplicate candidate geometry")
        if item["source"] not in ROLES or item["role"] != ROLES[item["source"]]:
            raise ValueError("candidate role lacks detection evidence")
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("candidate confidence must be finite in [0,1]")
        if not isinstance(item["text"], str):
            raise TypeError("candidate OCR text must be a string")
        ids.add(identifier)
        boxes.add(geometry)
    return candidates


def select_candidates(
    snapshot: Mapping[str, Any], *, instruction: str | None = None,
    actor_step: str | None = None,
) -> list[dict[str, Any]]:
    """Select raw proposals; context is explicit and labels are never inputs."""
    candidates = _candidate_evidence(snapshot)
    width, height = snapshot["image_width"], snapshot["image_height"]
    policy = snapshot.get("compact_policy", "spatial_v1")
    if policy == CONTEXT_POLICY:
        from .pointer_candidates_v3 import select_query_relations

        if any(not isinstance(s, str) or not s.strip() for s in (instruction, actor_step)):
            raise ValueError("v3 selection requires explicit goal and current subtask")
        return select_query_relations(candidates, width, height,
                                      instruction=instruction, actor_step=actor_step)
    if policy == "control_balanced_v2":
        from .pointer_candidates_v2 import select_control_balanced

        return select_control_balanced(candidates, width, height)
    if policy != "spatial_v1":
        raise ValueError("unknown compact candidate selection policy")
    pools: dict[str, list[dict[str, Any]]] = {}
    for source in ROLES:
        buckets = defaultdict(list)
        for item in candidates:
            if item["source"] == source:
                x, y = center(item["bbox"])
                buckets[(min(GRID_SIDE - 1, y * GRID_SIDE // height),
                         min(GRID_SIDE - 1, x * GRID_SIDE // width))].append(item)

        def rank(item: dict[str, Any], source: str = source) -> tuple[Any, ...]:
            box = item["bbox"]
            area = (box["right"] - box["left"]) * (box["bottom"] - box["top"])
            return (-item["confidence"] if source == "ocr" else 0, -area, item["candidate_id"])

        for bucket in buckets.values():
            bucket.sort(key=rank)
        ordered = []
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:
                    ordered.append(buckets[key].pop(0))
        pools[source] = ordered
    selected = pools["ocr"][:SOURCE_QUOTA] + pools["contour"][:SOURCE_QUOTA]
    remaining = MAX_CANDIDATES - len(selected)
    extras = pools["ocr"][SOURCE_QUOTA:] + pools["contour"][SOURCE_QUOTA:]
    selected.extend(extras[:remaining])
    return sorted(selected, key=lambda item: item["candidate_id"])


def selection_binding(
    snapshot: Mapping[str, Any], *, instruction: str, actor_step: str,
) -> dict[str, Any]:
    """Bind candidate numbering to the exact request and immutable raw evidence."""
    if any(not isinstance(s, str) or not s.strip() for s in (instruction, actor_step)):
        raise ValueError("selection binding requires explicit goal and current subtask")
    selected = select_candidates(snapshot, instruction=instruction, actor_step=actor_step)
    return {"policy": snapshot.get("compact_policy", "spatial_v1"),
            "context_sha256": value_sha256({"instruction": instruction, "actor_step": actor_step}),
            "snapshot_sha256": snapshot_hash(dict(snapshot)),
            "selected_candidate_ids": [c["candidate_id"] for c in selected]}


def pointer_user_prompt(instruction: str, actor_step: str, snapshot: Mapping[str, Any]) -> str:
    if any(not isinstance(s, str) or not s.strip() for s in (instruction, actor_step)):
        raise ValueError("candidate acting requires goal and explicit current subtask")
    table = "\n".join(
        f"{i} {center(c['bbox'])[0]} {center(c['bbox'])[1]} {canonical_json(c['text'][:TEXT_CHARACTERS])}"
        for i, c in enumerate(select_candidates(snapshot, instruction=instruction, actor_step=actor_step), 1)
    )
    return (
        "Execute the current subtask using the screenshot. Return the complete ordered JSON action array alone. "
        "Copy requested text exactly; do not continue to later subtasks. "
        "click/double_click/right_click/move: action_type,candidate,dx,dy (integers). "
        "scroll adds scroll_amount (nonzero integer). drag: action_type,start_candidate,start_dx,start_dy,end_candidate,end_dx,end_dy. "
        "type_text: action_type,text; key_press: action_type,key; hotkey: action_type,keys; wait: action_type,duration. "
        "No other fields or x/y output. Point = candidate native center + (dx,dy); use offsets to choose the intended pixel. "
        "Regions are proposals, not proof of interactive controls; inspect the screenshot. "
        "If no region fits, return []. Table: ID X Y OCR-text (at most24characters).\n"
        f"Native size: {snapshot['image_width']}x{snapshot['image_height']}\n{table}\n"
        f"Overall goal: {instruction}\nCurrent subtask: {actor_step}"
    )


def map_pointer_actions(
    generation: str, snapshot: Mapping[str, Any], image_path: str | Path, *,
    instruction: str | None = None, actor_step: str | None = None,
    expected_selection: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Strictly map whole raw arrays; no coordinates, ID guesses or role overrides."""
    validate_snapshot(dict(snapshot), image_path)
    selected = select_candidates(snapshot, instruction=instruction, actor_step=actor_step)
    if expected_selection is not None or snapshot.get("compact_policy", "spatial_v1") == CONTEXT_POLICY:
        actual_binding = selection_binding(snapshot, instruction=instruction, actor_step=actor_step)
        if expected_selection != actual_binding:
            raise ValueError("v3 candidate selection binding differs from the original request")
    actions = strict_json_loads(generation)
    if not isinstance(actions, list) or not actions:
        raise ValueError("pointer selection must be a nonempty bare action array")

    def point(identifier: Any, dx: Any, dy: Any) -> dict[str, int]:
        if type(identifier) is not int or not 1 <= identifier <= len(selected):
            raise ValueError("candidate ID must be an integer in the current selected table")
        if type(dx) is not int or type(dy) is not int:
            raise ValueError("candidate offsets must be integers")
        x, y = center(selected[identifier - 1]["bbox"])
        x, y = x + dx, y + dy
        if not (0 <= x < snapshot["image_width"] and 0 <= y < snapshot["image_height"]):
            raise ValueError("candidate offsets map outside the native screenshot")
        return {"x": x, "y": y}

    mapped = []
    for action in actions:
        if not isinstance(action, dict):
            raise TypeError("pointer action must be an object")
        kind = action.get("action_type")
        if kind in CLICK_TYPES | {"move", "scroll"}:
            expected = {"action_type", "candidate", "dx", "dy"} | ({"scroll_amount"} if kind == "scroll" else set())
            if set(action) != expected:
                raise ValueError("unexpected/missing pointer fields")
            result = {"action_type": kind, **point(action["candidate"], action["dx"], action["dy"])}
            if kind == "scroll":
                if type(action["scroll_amount"]) is not int or not action["scroll_amount"]:
                    raise ValueError("scroll_amount must be a nonzero integer")
                result["scroll_amount"] = action["scroll_amount"]
        elif kind == "drag":
            if set(action) != {"action_type", "start_candidate", "start_dx", "start_dy", "end_candidate", "end_dx", "end_dy"}:
                raise ValueError("unexpected/missing drag fields")
            result = {"action_type": kind,
                      "start": point(action["start_candidate"], action["start_dx"], action["start_dy"]),
                      "end": point(action["end_candidate"], action["end_dx"], action["end_dy"])}
        else:
            result = dict(action)
        mapped.append(result)
    return validate_actions(mapped)


def validate_pointer_sample(sample: Mapping[str, Any]) -> None:
    """Validate full supervision and its frozen label-side semantic review."""
    snapshot = sample["candidate_snapshot"]
    if (snapshot["image_width"], snapshot["image_height"]) != (sample["image_width"], sample["image_height"]):
        raise ValueError("candidate snapshot dimensions differ from the sample")
    target = sample["candidate_target"]
    mapped = map_pointer_actions(canonical_json(target), snapshot, sample["image_path"],
                                 instruction=sample["instruction"], actor_step=sample["actor_step"],
                                 expected_selection=sample.get("candidate_selection_evidence"))
    originals = sample["actions"]
    if len(mapped) != len(originals) or [a["action_type"] for a in mapped] != [a["action_type"] for a in originals]:
        raise ValueError("candidate target must preserve the complete native action sequence")
    evidence = sample["candidate_label_evidence"]
    review_path = Path(evidence["review_path"]).resolve(strict=True)
    if file_sha256(review_path) != evidence["review_sha256"]:
        raise ValueError("candidate semantic review changed")
    review = json.loads(review_path.read_text(encoding="utf8"))
    if review["schema"] != "week5.pointer_semantic_review.v1" or review["status"] != "closed":
        raise ValueError("candidate semantic review is not closed")
    selected = select_candidates(snapshot, instruction=sample["instruction"], actor_step=sample["actor_step"])
    for i, (original, action) in enumerate(zip(originals, target, strict=True)):
        kind = original["action_type"]
        if kind not in MOUSE_TYPES:
            if action != original:
                raise ValueError("nonmouse candidate target must preserve every native field")
            continue
        if kind == "scroll" and action["scroll_amount"] != original["scroll_amount"]:
            raise ValueError("candidate target must preserve the native scroll amount")
        endpoints = [("start_", action["start_candidate"]), ("end_", action["end_candidate"])] if kind == "drag" else [("", action["candidate"])]
        for endpoint, identifier in endpoints:
            records = [r for r in review["reviews"] if r["sample_id"] == sample["sample_id"] and r["action_index"] == i and r.get("endpoint", "") == endpoint]
            if len(records) != 1 or records[0]["decision"] != "accepted":
                raise ValueError("mouse target requires a unique accepted semantic review")
            record = records[0]
            if snapshot.get("compact_policy", "spatial_v1") == "control_balanced_v2":
                required = {"action_type", "current_subtask", "target_point"}
                if not required <= set(record):
                    raise ValueError("v2 semantic review requires source action, subtask and point")
            if snapshot.get("compact_policy", "spatial_v1") == CONTEXT_POLICY:
                required = {"action_type", "current_subtask", "target_point", "instruction"}
                if not required <= set(record):
                    raise ValueError("v3 semantic review requires original request and native target")
                if record["instruction"] != sample["instruction"]:
                    raise ValueError("semantic review goal differs from the request")
            if "action_type" in record and record["action_type"] != kind:
                raise ValueError("semantic review action type differs from the native target")
            if "current_subtask" in record and record["current_subtask"] != sample["actor_step"]:
                raise ValueError("semantic review subtask differs from the request")
            expected_point = original[endpoint.removesuffix("_")] if endpoint else original
            if "target_point" in record and any(record["target_point"][axis] != expected_point[axis] for axis in ("x", "y")):
                raise ValueError("semantic review point differs from the native target")
            if record["preferred_candidate_id"] != selected[identifier - 1]["candidate_id"]:
                raise ValueError("candidate target differs from the reviewed preferred region")
            if record["image_file_sha256"] != snapshot["image_file_sha256"]:
                raise ValueError("semantic review belongs to a different screenshot")
            path = Path(record["snapshot_path"]).resolve(strict=True)
            if file_sha256(path) != record["snapshot_file_sha256"] or snapshot_hash(json.loads(path.read_text(encoding="utf8"))) != snapshot_hash(snapshot):
                raise ValueError("semantic review candidate snapshot changed")
    if file_sha256(review_path) != evidence["review_sha256"]:
        raise ValueError("semantic review changed during label validation")
    if mapped != originals:
        raise ValueError("candidate target must reconstruct every original native field exactly")


def build_pointer_messages(sample: Mapping[str, Any], *, include_target: bool = False) -> list[dict[str, Any]]:
    snapshot = sample["candidate_snapshot"]
    if (snapshot["image_width"], snapshot["image_height"]) != (sample["image_width"], sample["image_height"]):
        raise ValueError("candidate snapshot dimensions differ from the request")
    prompt = pointer_user_prompt(sample["instruction"], sample["actor_step"], snapshot)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if include_target:
        validate_pointer_sample(sample)
        messages.append({"role": "assistant", "content": canonical_json(sample["candidate_target"])})
    return messages


def request_pointer_actions(worker: Any, sample: Mapping[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    """Use a caller-owned HF worker with the exact training prefix, without labels.

    This function neither loads a model nor executes desktop actions. The original
    generation and transport result are retained even when strict mapping fails.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("request timeout must be finite and positive")
    ready = getattr(worker, "ready", {})
    if ready.get("output_constraint") is not None or ready.get("constraint_hook"):
        raise ValueError("candidate inference requires an unconstrained caller-owned worker")
    snapshot = sample["candidate_snapshot"]
    validate_snapshot(snapshot, sample["image_path"])
    binding = (selection_binding(snapshot, instruction=sample["instruction"], actor_step=sample["actor_step"])
               if snapshot.get("compact_policy", "spatial_v1") == CONTEXT_POLICY else None)
    prompt = build_pointer_messages(sample)[0]["content"][1]["text"]
    reply = worker.request(str(sample["image_path"]), prompt, timeout_seconds=timeout_seconds)
    result = {"raw_reply": reply, "prompt": prompt, "snapshot_sha256": snapshot_hash(snapshot),
              "mapped_actions": None, "mapping_error": None}
    if binding is not None:
        result["candidate_selection_evidence"] = binding
        result["snapshot_sha256"] = binding["snapshot_sha256"]
    if reply.get("status") == "ok":
        try:
            current_snapshot = sample["candidate_snapshot"] if binding is not None else snapshot
            if binding is not None and selection_binding(
                current_snapshot, instruction=sample["instruction"], actor_step=sample["actor_step"],
            ) != binding:
                raise ValueError("v3 candidate selection binding differs from the original request")
            result["mapped_actions"] = map_pointer_actions(reply["generation"], current_snapshot, sample["image_path"],
                                                          instruction=sample["instruction"], actor_step=sample["actor_step"],
                                                          expected_selection=binding)
        except (ValueError, TypeError, KeyError) as exc:
            result["mapping_error"] = f"{type(exc).__name__}: {exc}"
    return result
