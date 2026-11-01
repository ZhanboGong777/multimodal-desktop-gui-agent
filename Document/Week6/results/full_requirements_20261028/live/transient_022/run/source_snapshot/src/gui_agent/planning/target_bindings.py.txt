"""Frame-local generation candidates; never execution authority or task goals.

Only literal observed labels and their IDs are related here. Existing runtime
grounding, scope and postcondition checks must still validate every native step.
No runtime module is imported while the planning package is initialized.
"""

from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy
from io import BytesIO
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.schemas import ObservationSnapshot

TARGET_BINDINGS_VERSION = "gui_agent.target_bindings.v1"
TARGET_BINDING_SCHEMA_VERSION = "gui_agent.target_binding_schema.v2"
TARGET_BINDING_DETAIL_SCHEMA_VERSION = "gui_agent.target_binding_schema.v3.current_id_details"
OBSERVED_QUERY_CONTRACT_VERSION = "gui_agent.observed_text_queries.v1"
_QUERY_LIMITS = {"tokens_per_label": 32, "tokens_per_span": 4, "elements_per_run": 4,
                 "extra_query_characters": 160, "extra_queries": 128, "search_steps": 4096}
_QUERY_TOKEN = re.compile(r"[0-9a-z_]+", re.IGNORECASE | re.ASCII)
GENERATION_TARGETING_POLICY_VERSION = "gui_agent.generation_targeting_policy.v1"
GENERATION_TARGETING_POLICY_TEXT = (
    "Generation guidance only; it grants neither input nor success authority. "
    "Use only candidates in the current target_bindings scope, including the final authorised subset. "
    "Use text-only targeting only when current OCR labels or adjacent OCR words can locate the target. "
    "If a target is visible in the screenshot but absent from OCR, choose its matching current ID yourself "
    "from target_bindings.anonymous_without_ocr_ids; never invent or replace an ID. "
    "If no matching allowed candidate is certain, return no steps and explain in errors."
)
POINT_BINDING_ACTION_TYPES = ("move", "click", "double_click", "right_click", "scroll")
_RAW_FIELDS = {"element_id", "text", "source", "confidence", "bounding_box", "external_ocr_overlap"}
_CANDIDATE_FIELDS = _RAW_FIELDS | {"classification", "compatible_texts", "overlapping_ocr_ids"}
_FIELDS = {"schema_version", "observation_id", "scope", "min_confidence", "candidates",
           "labels", "anonymous_without_ocr_ids"}


def generation_targeting_policy() -> dict[str, Any]:
    """A fresh fixed policy; it neither selects a target nor changes bindings."""
    return {"schema_version": GENERATION_TARGETING_POLICY_VERSION,
            "instruction": GENERATION_TARGETING_POLICY_TEXT, "generation_only": True}


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


def observed_query_contract(bindings: dict[str, Any]) -> dict[str, Any]:
    """Finite generation queries, not the complete runtime matching language.

    Full raw labels keep their existing branches. Extra queries are raw token
    spans followed by geometric reading-order runs and spatial paths. One
    candidate consideration or spatial neighbour comparison consumes one search
    step. Qualification uses at most four source elements, proving a matching
    witness exists in the allowed subset without enumerating all runtime matches.
    Runtime still owns whole-frame ambiguity, fresh grounding and dispatch.
    """
    # Local imports keep planning package initialization independent of runtime.
    from ..perception.grounding import find_text
    from ..runtime.action_adapter import _is_next_in_label, _match_adjacent_runs
    from ..runtime.schemas import ElementRef
    from ..schemas import BoundingBox

    if type(bindings) is not dict:
        raise ValueError("target_bindings must be an object")
    packet = validate_target_bindings(bindings, observation_id=bindings.get("observation_id"))
    full = [item["text"] for item in packet["labels"]]
    threshold = packet["min_confidence"]
    source = []
    for item in packet["candidates"]:
        if item["source"] == "ocr" and item["text"].strip() and item["confidence"] >= threshold:
            box = BoundingBox(**item["bounding_box"])
            source.append(ElementRef(element_id=item["element_id"], text=item["text"], source="ocr",
                                     confidence=item["confidence"], bounding_box=box, center=box.center))
    search = {"steps": 0, "candidate_queries_considered": 0, "duplicate_queries": 0,
              "overlength_queries": 0, "unresolvable_queries": 0,
              "labels_with_token_truncation": 0, "count_limit_reached": False,
              "search_limit_reached": False, "truncated": False, "omitted_queries_at_least": 0}
    extra: list[dict[str, Any]] = []
    seen = set(full)
    halted = False

    def consume() -> bool:
        nonlocal halted
        if search["steps"] >= _QUERY_LIMITS["search_steps"]:
            search["search_limit_reached"] = halted = True
            return False
        search["steps"] += 1
        return True

    def consider(query: str, kind: str, witnesses: list[ElementRef]) -> None:
        nonlocal halted
        if halted or not consume():
            return
        search["candidate_queries_considered"] += 1
        if query in seen:
            search["duplicate_queries"] += 1
            return
        if len(query) > _QUERY_LIMITS["extra_query_characters"]:
            search["overlength_queries"] += 1
            return
        if (not query.strip() or not (find_text(witnesses, query, min_confidence=threshold).found
                or _match_adjacent_runs(witnesses, query, threshold))):
            search["unresolvable_queries"] += 1
            return
        if len(extra) >= _QUERY_LIMITS["extra_queries"]:
            search["count_limit_reached"] = halted = True
            search["omitted_queries_at_least"] = 1
            return
        seen.add(query)
        extra.append({"query": query, "source_kind": kind,
                      "source_element_ids": [item.element_id for item in witnesses]})

    # Scan only one token beyond the cap to record truncation without an
    # unbounded token list. Source order is the original final candidate order.
    token_spans = []
    for item in source:
        tokens = list(islice(_QUERY_TOKEN.finditer(item.text), _QUERY_LIMITS["tokens_per_label"] + 1))
        if len(tokens) > _QUERY_LIMITS["tokens_per_label"]:
            search["labels_with_token_truncation"] += 1
        token_spans.append((item, tokens[:_QUERY_LIMITS["tokens_per_label"]]))
    for item, tokens in token_spans:
        if halted:
            break
        for start in range(len(tokens)):
            for size in range(1, min(_QUERY_LIMITS["tokens_per_span"], len(tokens) - start) + 1):
                consider(item.text[tokens[start].start():tokens[start + size - 1].end()],
                         "label_token_span", [item])
                if halted:
                    break
            if halted:
                break

    ordered = sorted(source, key=lambda item: (item.bounding_box.top, item.bounding_box.left))
    for start in range(len(ordered)):
        if halted:
            break
        for size in range(2, min(_QUERY_LIMITS["elements_per_run"], len(ordered) - start) + 1):
            run = ordered[start:start + size]
            # Each candidate consumes a step even when its geometry is invalid.
            if not all(_is_next_in_label(a, b) for a, b in pairwise(run)):
                if consume():
                    search["candidate_queries_considered"] += 1
            else:
                consider(" ".join(item.text.strip() for item in run), "reading_order_run", run)
            if halted:
                break

    def walk(path: list[ElementRef]) -> None:
        if len(path) >= _QUERY_LIMITS["elements_per_run"] or halted:
            return
        for item in ordered:
            if not consume():
                return
            if any(item.element_id == previous.element_id for previous in path):
                continue
            if not _is_next_in_label(path[-1], item):
                continue
            run = [*path, item]
            consider(" ".join(part.text.strip() for part in run), "spatial_run", run)
            if halted:
                return
            walk(run)
            if halted:
                return

    for item in ordered:
        if halted:
            break
        walk([item])
    search["truncated"] = bool(search["labels_with_token_truncation"] or search["overlength_queries"]
                                or search["search_limit_reached"] or search["count_limit_reached"])
    return {"schema_version": OBSERVED_QUERY_CONTRACT_VERSION, "full_label_queries": full,
            "extra_queries": extra, "limits": dict(_QUERY_LIMITS), "search": search}


def apply_target_bindings(step_schema: dict[str, Any], bindings: dict[str, Any], *,
                          current_id_only: bool = False) -> dict[str, Any]:
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
    if current_id_only:
        # A compact generation profile for pixel-detail requests. The provider
        # only chooses a current ID; runtime still checks every text/ID alias,
        # freshness and authorised scope before dispatch. This schema does not
        # claim to establish that a native label and its ID mean the same thing.
        ids = [item["element_id"] for item in bindings["candidates"]]
        if ids:
            variants.insert(0, branch(point, arguments={"element_id": {"enum": ids},
                                                       "target_element": {"enum": empty},
                                                       "target": {"enum": empty}},
                                      required_arguments=["element_id"]))
        return {"anyOf": variants}
    # No free text-only branch: unknown visual text needs a native allowed ID.
    queries = [item["query"] for item in observed_query_contract(bindings)["extra_queries"]]
    if queries:
        no_ids = {"element_id": {"enum": empty}, "target_element": {"enum": empty}}
        variants.append(branch(point, query={"enum": queries}, arguments={
            **no_ids, "target": {"enum": empty}}, require_query=True))
        variants.append(branch(point, query={"enum": empty}, arguments={
            **no_ids, "target": {"enum": queries}}, required_arguments=["target"]))
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


VISUAL_DETAILS_VERSION = "gui_agent.candidate_visual_details.v1"
VISUAL_DETAILS_LIMITS = {"minimum_width": 96, "minimum_height": 24,
                         "maximum_regions": 8, "maximum_total_pixels": 1_000_000}


def candidate_visual_details(bindings: dict[str, Any], image_path: str) -> tuple[dict[str, Any], list[bytes]]:
    """Original-pixel crops, selected only by current scope and geometry.

    No goal, expected text, model output or OCR repair enters selection. Each
    crop is bound to the original image bytes and current ID. The original
    screenshot remains the first image; crops never change coordinate space.
    """
    from pathlib import Path

    from PIL import Image

    if type(bindings) is not dict:
        raise ValueError("visual details require scoped target bindings")
    packet = validate_target_bindings(bindings, observation_id=bindings.get("observation_id"))
    raw = Path(image_path).read_bytes()
    regions, images, pixels, omitted = [], [], 0, 0
    with Image.open(BytesIO(raw)) as image:
        image.load()
        for candidate in sorted(packet["candidates"],
                                key=lambda c: (c["bounding_box"]["top"], c["bounding_box"]["left"])):
            box = candidate["bounding_box"]
            width, height = box["right"] - box["left"], box["bottom"] - box["top"]
            if (candidate["classification"] != "anonymous_without_ocr"
                    or width < VISUAL_DETAILS_LIMITS["minimum_width"]
                    or height < VISUAL_DETAILS_LIMITS["minimum_height"]):
                continue
            if any(int(value) != value for value in box.values()):
                raise ValueError("visual details need integer pixel bounds")
            bounds = tuple(int(box[key]) for key in ("left", "top", "right", "bottom"))
            if not (0 <= bounds[0] < bounds[2] <= image.width
                    and 0 <= bounds[1] < bounds[3] <= image.height):
                raise ValueError("visual detail candidate lies outside its original screenshot")
            if (len(regions) >= VISUAL_DETAILS_LIMITS["maximum_regions"]
                    or pixels + width * height > VISUAL_DETAILS_LIMITS["maximum_total_pixels"]):
                omitted += 1
                continue
            buffer = BytesIO()
            image.crop(bounds).save(buffer, format="PNG")
            payload = buffer.getvalue()
            regions.append({"element_id": candidate["element_id"], "original_box": deepcopy(box),
                            "png_sha256": hashlib.sha256(payload).hexdigest()})
            images.append(payload)
            pixels += int(width * height)
    manifest = {"schema_version": VISUAL_DETAILS_VERSION, "observation_id": packet["observation_id"],
                "source_image_sha256": hashlib.sha256(raw).hexdigest(),
                "limits": dict(VISUAL_DETAILS_LIMITS), "regions": regions,
                "omitted_region_count": omitted, "coordinate_space": "original_screenshot",
                "generation_only": True}
    return manifest, images
