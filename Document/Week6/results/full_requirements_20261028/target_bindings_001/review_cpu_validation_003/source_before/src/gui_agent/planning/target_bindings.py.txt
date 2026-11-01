"""Frame-local generation candidates; never execution authority or task goals.

Only literal observed labels and their IDs are related here. Existing runtime
grounding, scope and postcondition checks must still validate every native step.
No runtime module is imported while the planning package is initialized.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.schemas import ObservationSnapshot

TARGET_BINDINGS_VERSION = "gui_agent.target_bindings.v1"
TARGET_BINDING_SCHEMA_VERSION = "gui_agent.target_binding_schema.v1"
POINT_BINDING_ACTION_TYPES = ("move", "click", "double_click", "right_click", "scroll")
_RAW_FIELDS = {"element_id", "text", "source", "confidence", "bounding_box", "external_ocr_overlap"}
_CANDIDATE_FIELDS = _RAW_FIELDS | {"classification", "compatible_texts", "overlapping_ocr_ids"}
_FIELDS = {"schema_version", "observation_id", "scope", "min_confidence", "candidates",
           "labels", "anonymous_without_ocr_ids"}


def _number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (max(a["left"], b["left"]) < min(a["right"], b["right"])
            and max(a["top"], b["top"]) < min(a["bottom"], b["bottom"]))


def _candidate_records(raw: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    readable_ocr = [item for item in raw if item["source"] == "ocr"
                    and item["text"].strip() and item["confidence"] >= threshold]
    candidates, labels, no_ocr = [], {}, []
    for item in raw:
        overlapping = []
        compatible = []
        classification = "untrusted_or_other"
        if item["text"].strip() and item["confidence"] >= threshold:
            classification, compatible = "readable_label", [item["text"]]
        elif not item["text"].strip() and item["source"] == "contour":
            overlapping = [other for other in readable_ocr
                           if _overlap(item["bounding_box"], other["bounding_box"])]
            if item["external_ocr_overlap"]:
                classification = "untrusted_or_other"
            elif not overlapping:
                classification = "anonymous_without_ocr"
                no_ocr.append(item["element_id"])
            elif len(overlapping) == 1:
                classification, compatible = "anonymous_with_single_ocr", [overlapping[0]["text"]]
            else:
                classification = "anonymous_mixed_ocr"
        candidates.append({**deepcopy(item), "classification": classification,
                           "compatible_texts": compatible,
                           "overlapping_ocr_ids": [other["element_id"] for other in overlapping]})
        for text in compatible:
            labels.setdefault(text, []).append(item["element_id"])
    return candidates, [{"text": text, "element_ids": ids} for text, ids in labels.items()], no_ocr


def _raw_element(item: Any) -> dict[str, Any]:
    return {"element_id": item.element_id, "text": item.text, "source": item.source,
            "confidence": item.confidence, "bounding_box": item.bounding_box.model_dump(mode="json"),
            "external_ocr_overlap": False}


def derive_target_bindings(
    snapshot: ObservationSnapshot, *, min_confidence: float = 0.35,
    source_snapshot: ObservationSnapshot | None = None,
) -> dict[str, Any]:
    """Derive from the final allowed subset; excluded OCR grants no visual pass."""
    if not _number(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("target bindings need a finite confidence threshold in 0..1")
    raw = [_raw_element(item) for item in snapshot.elements]
    scope = "all_snapshot_candidates"
    if source_snapshot is not None:
        if source_snapshot.observation_id != snapshot.observation_id:
            raise ValueError("target bindings subset belongs to a different observation")
        source = {_item.element_id: _raw_element(_item) for _item in source_snapshot.elements}
        if len(source) != len(source_snapshot.elements):
            raise ValueError("target bindings source contains duplicate element IDs")
        selected = {item["element_id"] for item in raw}
        if any(source.get(item["element_id"]) != item for item in raw):
            raise ValueError("target bindings subset differs from its source elements")
        scope = "authorised_subset"
        external_ocr = [item for key, item in source.items() if key not in selected
                        and item["source"] == "ocr" and item["text"].strip()
                        and item["confidence"] >= min_confidence]
        for item in raw:
            if not item["text"].strip() and item["source"] == "contour":
                item["external_ocr_overlap"] = any(
                    _overlap(item["bounding_box"], other["bounding_box"]) for other in external_ocr)
    candidates, labels, no_ocr = _candidate_records(raw, min_confidence)
    value = {"schema_version": TARGET_BINDINGS_VERSION, "observation_id": snapshot.observation_id,
             "scope": scope, "min_confidence": min_confidence, "candidates": candidates,
             "labels": labels, "anonymous_without_ocr_ids": no_ocr}
    return validate_target_bindings(value, observation_id=snapshot.observation_id)


def validate_target_bindings(value: Any, *, observation_id: Any) -> dict[str, Any]:
    """Reject malformed/stale data; this does not authenticate a caller's pixels."""
    if type(value) is not dict or set(value) != _FIELDS:
        raise ValueError("target_bindings has an invalid object shape")
    if value["schema_version"] != TARGET_BINDINGS_VERSION:
        raise ValueError("unsupported target_bindings version")
    if (type(observation_id) is not str or not observation_id
            or type(value["observation_id"]) is not str or value["observation_id"] != observation_id):
        raise ValueError("target_bindings observation does not match the planning context")
    if value["scope"] not in {"all_snapshot_candidates", "authorised_subset"}:
        raise ValueError("target_bindings has an invalid candidate scope")
    threshold = value["min_confidence"]
    if not _number(threshold) or not 0 <= threshold <= 1:
        raise ValueError("target_bindings has an invalid confidence threshold")
    if type(value["candidates"]) is not list:
        raise ValueError("target_bindings candidates must be a list")
    raw, ids = [], set()
    for item in value["candidates"]:
        if type(item) is not dict or set(item) != _CANDIDATE_FIELDS:
            raise ValueError("target_bindings has an invalid candidate shape")
        if (type(item["element_id"]) is not str or not item["element_id"]
                or item["element_id"] in ids or type(item["text"]) is not str
                or type(item["source"]) is not str or not item["source"]):
            raise ValueError("target_bindings has invalid or duplicate candidate identity/text")
        ids.add(item["element_id"])
        if not _number(item["confidence"]) or not 0 <= item["confidence"] <= 1:
            raise ValueError("target_bindings has invalid candidate confidence")
        box = item["bounding_box"]
        if (type(box) is not dict or set(box) != {"left", "top", "right", "bottom"}
                or not all(_number(number) for number in box.values())
                or box["left"] >= box["right"] or box["top"] >= box["bottom"]):
            raise ValueError("target_bindings has invalid candidate geometry")
        if (type(item["external_ocr_overlap"]) is not bool
                or (item["external_ocr_overlap"] and (value["scope"] != "authorised_subset"
                    or item["text"].strip() or item["source"] != "contour"))):
            raise ValueError("target_bindings has invalid external evidence classification")
        raw.append({key: deepcopy(item[key]) for key in _RAW_FIELDS})
    candidates, labels, no_ocr = _candidate_records(raw, threshold)
    if (value["candidates"] != candidates or value["labels"] != labels
            or value["anonymous_without_ocr_ids"] != no_ocr):
        raise ValueError("target_bindings classifications or label/ID pairs are inconsistent")
    return deepcopy(value)


def apply_target_bindings(step_schema: dict[str, Any], bindings: dict[str, Any]) -> dict[str, Any]:
    """Positive, self-contained unions only; other arguments remain extensible."""
    if type(bindings) is not dict:
        raise ValueError("target_bindings must be an object")
    bindings = validate_target_bindings(bindings, observation_id=bindings.get("observation_id"))
    actions = step_schema["properties"]["action_type"]["enum"]
    point = [action for action in actions if action in POINT_BINDING_ACTION_TYPES]
    other = [action for action in actions if action not in POINT_BINDING_ACTION_TYPES]
    if not point:
        return deepcopy(step_schema)
    empty = [None, ""]

    def branch(
        action_types: list[str], *, query: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None, required_arguments: list[str] | None = None,
        require_query: bool = False,
    ) -> dict[str, Any]:
        result = deepcopy(step_schema)
        result["properties"]["action_type"]["enum"] = action_types
        if query is not None:
            result["properties"]["target_text"] = query
        if arguments is not None:
            result["properties"]["arguments"] = {"type": "object", "additionalProperties": True,
                                                 "properties": arguments}
        required = list(result.get("required", []))
        if required_arguments:
            result["properties"]["arguments"]["required"] = required_arguments
            required.append("arguments")
        if require_query:
            required.append("target_text")
        result["required"] = list(dict.fromkeys(required))
        return result

    variants = [branch(other)] if other else []
    # No ID is synthesized. An unrestricted textual/word-run query remains a
    # native model choice and must pass the existing ambiguity/grounding checks.
    variants.append(branch(point, arguments={"element_id": {"enum": empty},
                                             "target_element": {"enum": empty}}))
    ids = [item["element_id"] for item in bindings["candidates"]]
    for primary, secondary in (("element_id", "target_element"), ("target_element", "element_id")):
        if ids:
            variants.append(branch(point, query={"enum": empty}, arguments={
                primary: {"enum": ids}, secondary: {"enum": [*empty, *ids]}, "target": {"enum": empty},
            }, required_arguments=[primary]))
        visual = bindings["anonymous_without_ocr_ids"]
        if visual:
            variants.append(branch(point, arguments={primary: {"enum": visual}, secondary: {"enum": [*empty, *visual]}},
                                   required_arguments=[primary]))
    for label in bindings["labels"]:
        text, consistent = label["text"], [*empty, *label["element_ids"]]
        args = {"element_id": {"enum": consistent}, "target_element": {"enum": consistent},
                "target": {"enum": [*empty, text]}}
        variants.append(branch(point, query={"enum": [text]}, arguments=args, require_query=True))
        alias_args = {**args, "target": {"enum": [text]}}
        variants.append(branch(point, query={"enum": empty}, arguments=alias_args, required_arguments=["target"]))
    return {"anyOf": variants}
