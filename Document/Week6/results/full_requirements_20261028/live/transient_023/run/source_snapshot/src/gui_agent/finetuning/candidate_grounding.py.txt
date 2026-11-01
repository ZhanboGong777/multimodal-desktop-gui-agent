"""Experimental screenshot-only candidates; never a desktop action dispatcher.

Candidates are geometric/text proposals, not semantic proof of an editor/button.
Target annotations are accepted only by the separate coverage/scoring function.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from .evaluation import (
    CLICK_TYPES,
    canonical_json,
    file_sha256,
    strict_json_loads,
    validate_actions,
    value_sha256,
)

EXPERIMENT_VERSION = "week5.candidate-grounding.v1"
ROLES = {"ocr": "text_region", "contour": "contour_region"}


@dataclass(frozen=True)
class DetectorConfig:
    max_candidates: int = 64
    contour_budget: int = 200
    min_area: int = 100
    max_area_ratio: float = 0.25
    min_rectangularity: float = 0.6
    contour_max_aspect: float = 12.0
    ocr: str = "tesseract"
    ocr_language: str = "eng"
    ocr_min_confidence: float = 0.5
    ocr_timeout_seconds: float = 15
    ocr_scale: int = 1
    ocr_merge_aligned: bool = False
    contour_text_lines: bool = False

    def __post_init__(self) -> None:
        if self.ocr not in {"none", "tesseract"}:
            raise ValueError("only installed Tesseract or explicit contour-only is supported")
        if any(
            type(value) is not int or value <= 0
            for value in (self.max_candidates, self.contour_budget, self.min_area)
        ):
            raise ValueError("candidate budgets and area must be positive integers")
        if not 0 < self.max_area_ratio < 1 or not 0 <= self.min_rectangularity <= 1:
            raise ValueError("invalid geometric limits")
        if not 0 <= self.ocr_min_confidence <= 1:
            raise ValueError("invalid OCR confidence")
        if (
            isinstance(self.contour_max_aspect, bool)
            or not math.isfinite(self.contour_max_aspect)
            or self.contour_max_aspect <= 0
        ):
            raise ValueError("contour max aspect must be finite and positive")
        if not math.isfinite(self.ocr_timeout_seconds) or self.ocr_timeout_seconds <= 0:
            raise ValueError("OCR timeout must be finite and positive")
        if type(self.ocr_scale) is not int or self.ocr_scale not in {1, 2}:
            raise ValueError("OCR scale must be the fixed integer 1 or 2")
        if type(self.ocr_merge_aligned) is not bool:
            raise ValueError("OCR aligned merging must be an explicit boolean")
        if type(self.contour_text_lines) is not bool:
            raise ValueError("contour text lines must be an explicit boolean")


def native_box(box: dict[str, Any], width: int, height: int) -> dict[str, int]:
    if set(box) != {"left", "top", "right", "bottom"} or any(
        type(v) is not int for v in box.values()
    ):
        raise ValueError("native bbox requires exactly four integer edges")
    if not (0 <= box["left"] < box["right"] <= width and 0 <= box["top"] < box["bottom"] <= height):
        raise ValueError("candidate bbox is outside the native screenshot")
    return dict(box)


def center(box: dict[str, int]) -> tuple[int, int]:
    return (box["left"] + box["right"]) // 2, (box["top"] + box["bottom"]) // 2


def contains(box: dict[str, int], x: int, y: int) -> bool:
    return box["left"] <= x < box["right"] and box["top"] <= y < box["bottom"]


def _iou(a: dict[str, int], b: dict[str, int]) -> float:
    overlap = max(0, min(a["right"], b["right"]) - max(a["left"], b["left"])) * max(
        0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    )
    area = lambda box: (box["right"] - box["left"]) * (box["bottom"] - box["top"])
    return overlap / (area(a) + area(b) - overlap)


def proposals_to_candidates(
    proposals: list[dict[str, Any]], width: int, height: int, max_candidates: int
) -> list[dict[str, Any]]:
    """Deterministic image-only ordering; no instruction or ground truth used."""
    ordered = sorted(
        proposals,
        key=lambda p: (
            p["source"] != "ocr",
            -p["confidence"],
            p["bbox"]["top"],
            p["bbox"]["left"],
            p["bbox"]["bottom"],
            p["bbox"]["right"],
            p["text"],
        ),
    )
    kept: list[dict[str, Any]] = []

    def direct_parent(first: dict[str, Any], second: dict[str, Any]) -> bool:
        evidence = first.get("evidence", {})
        if evidence.get("kind") == "aligned_ocr_parent":
            return any(
                child["bbox"] == second["bbox"] and child["text"] == second["text"]
                for child in evidence["children"]
            )
        if evidence.get("kind") == "contour_text_line_parent":
            return first["source"] == second["source"] == "contour" and any(
                all(child.get(key) == second[key] for key in ("bbox", "confidence", "text", "source"))
                for child in evidence["children"]
            )
        return False

    for proposal in ordered:
        if proposal["source"] not in ROLES:
            raise ValueError("untrusted candidate source")
        box = native_box(proposal["bbox"], width, height)
        if any(
            _iou(box, existing["bbox"]) >= 0.85
            and not (
                box != existing["bbox"]
                and (direct_parent(proposal, existing) or direct_parent(existing, proposal))
            )
            for existing in kept
        ):
            continue
        candidate = {
                "candidate_id": f"c{len(kept) + 1:03d}",
                "bbox": box,
                "role": ROLES[proposal["source"]],
                "source": proposal["source"],
                "text": proposal["text"],
                "confidence": proposal["confidence"],
            }
        if "evidence" in proposal:
            candidate["evidence"] = proposal["evidence"]
        kept.append(candidate)
        if len(kept) == max_candidates:
            break
    return kept


def _aligned_ocr_parents(
    proposals: list[dict[str, Any]], image_width: int
) -> list[dict[str, Any]]:
    """Add short visible text-line unions using native geometry alone.

    Tesseract's block IDs can split one menu label. Original atomic proposals
    stay available; distant columns and long rows are never joined. A union is
    text evidence, not proof of a button or a menu's clickable row boundary.
    """
    groups: list[list[dict[str, Any]]] = []
    ordered = sorted(
        proposals,
        key=lambda p: (
            p["bbox"]["bottom"], p["bbox"]["left"], p["bbox"]["top"], p["text"],
        ),
    )
    for proposal in ordered:
        box = proposal["bbox"]
        if not proposal["text"].strip() or len(proposal["text"]) > 32:
            continue
        fits = []
        for index, group in enumerate(groups):
            last = group[-1]["bbox"]
            height = max(box["bottom"] - box["top"], last["bottom"] - last["top"])
            overlap = min(box["bottom"], last["bottom"]) - max(box["top"], last["top"])
            min_height = min(box["bottom"] - box["top"], last["bottom"] - last["top"])
            gap = box["left"] - last["right"]
            if (
                0 <= gap <= 1.2 * height
                and abs(box["bottom"] - last["bottom"]) <= 0.35 * height
                and overlap >= 0.6 * min_height
                and box["right"] - group[0]["bbox"]["left"] <= image_width * 0.4
                and len(" ".join(p["text"] for p in [*group, proposal])) <= 32
            ):
                fits.append((gap, index))
        if fits:
            groups[min(fits)[1]].append(proposal)
        else:
            groups.append([proposal])
    parents = []
    for group in groups:
        if len(group) < 2:
            continue
        boxes = [p["bbox"] for p in group]
        parents.append(
            {
                "source": "ocr",
                "text": " ".join(p["text"] for p in group),
                "confidence": min(p["confidence"] for p in group),
                "bbox": {
                    "left": min(b["left"] for b in boxes),
                    "top": min(b["top"] for b in boxes),
                    "right": max(b["right"] for b in boxes),
                    "bottom": max(b["bottom"] for b in boxes),
                },
                "evidence": {
                    "kind": "aligned_ocr_parent",
                    "policy": "native_baseline_gap_v1",
                    "children": [
                        {k: p[k] for k in ("bbox", "text", "confidence")} for p in group
                    ],
                },
            }
        )
    return parents


def _contour_text_line_parents(
    proposals: list[dict[str, Any]], image_width: int, image_height: int
) -> list[dict[str, Any]]:
    """Union short, aligned contour runs without inventing recognized words.

    These remain geometric evidence. A small row of adjacent controls can have
    the same geometry as words; the parent never claims text/control semantics.
    Each atom belongs to one baseline row and one left-to-right run. Reject an
    entire overlong run instead of slicing a dense toolbar into pseudo-labels.
    """
    eligible = []
    for proposal in proposals:
        if proposal["source"] != "contour" or "evidence" in proposal:
            continue
        box = native_box(proposal["bbox"], image_width, image_height)
        height = box["bottom"] - box["top"]
        width = box["right"] - box["left"]
        if 6 <= height <= 32 and 0.3 * height <= width <= 16 * height:
            eligible.append(proposal)

    def aligned(first: dict[str, int], second: dict[str, int]) -> bool:
        heights = (first["bottom"] - first["top"], second["bottom"] - second["top"])
        small, large = min(heights), max(heights)
        overlap = min(first["bottom"], second["bottom"]) - max(first["top"], second["top"])
        return (
            abs(first["bottom"] - second["bottom"]) <= 0.25 * large
            and overlap >= 0.7 * small
            and small >= 0.75 * large
        )

    rows: list[list[dict[str, Any]]] = []
    for proposal in sorted(
        eligible,
        key=lambda p: (
            p["bbox"]["bottom"], p["bbox"]["top"], p["bbox"]["left"], p["bbox"]["right"],
            p["confidence"], p["text"],
        ),
    ):
        box = proposal["bbox"]
        fits = [
            (abs(box["bottom"] - median(p["bbox"]["bottom"] for p in row)), index)
            for index, row in enumerate(rows)
            if all(aligned(box, p["bbox"]) for p in row)
        ]
        if fits:
            rows[min(fits)[1]].append(proposal)
        else:
            rows.append([proposal])

    parents = []

    def append_parent(run: list[dict[str, Any]]) -> None:
        if not 2 <= len(run) <= 8:
            return
        boxes = [p["bbox"] for p in run]
        box = {
            "left": min(b["left"] for b in boxes),
            "top": min(b["top"] for b in boxes),
            "right": max(b["right"] for b in boxes),
            "bottom": max(b["bottom"] for b in boxes),
        }
        typical_height = median(b["bottom"] - b["top"] for b in boxes)
        if (
            box["right"] - box["left"] > min(0.35 * image_width, 24 * typical_height)
            or box["bottom"] - box["top"] > 1.4 * typical_height
        ):
            return
        parents.append(
            {
                "source": "contour",
                "text": "",
                "confidence": 0.5,
                "bbox": native_box(box, image_width, image_height),
                "evidence": {
                    "kind": "contour_text_line_parent",
                    "policy": "native_baseline_gap_v1",
                    "children": [
                        {key: dict(p[key]) if key == "bbox" else p[key]
                         for key in ("bbox", "confidence", "text", "source")}
                        for p in run
                    ],
                },
            }
        )

    for row in rows:
        run: list[dict[str, Any]] = []
        for proposal in sorted(
            row,
            key=lambda p: (
                p["bbox"]["left"], p["bbox"]["right"], p["bbox"]["top"], p["bbox"]["bottom"],
                p["confidence"], p["text"],
            ),
        ):
            if run:
                previous, box = run[-1]["bbox"], proposal["bbox"]
                gap = box["left"] - previous["right"]
                large_height = max(
                    box["bottom"] - box["top"], previous["bottom"] - previous["top"],
                )
                if not 0 <= gap <= 0.6 * large_height:
                    append_parent(run)
                    run = []
            run.append(proposal)
        append_parent(run)
    return parents


def _ocr_proposals(
    image: Image.Image, config: DetectorConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {"requested": config.ocr, "actual_engine": None, "error": None}
    if config.ocr == "none":
        return [], info
    try:
        import pytesseract

        from ..perception.ocr import WordRow, merge_words_into_lines

        info["version"] = str(pytesseract.get_tesseract_version())
        ocr_image = image
        if config.ocr_scale != 1:
            ocr_image = image.resize(
                (image.width * config.ocr_scale, image.height * config.ocr_scale),
                Image.Resampling.LANCZOS,
            )
            info["scale"] = config.ocr_scale
            info["coordinate_space"] = "native_after_outward_integer_rounding"
        data = pytesseract.image_to_data(
            ocr_image,
            lang=config.ocr_language,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
            timeout=config.ocr_timeout_seconds,
        )
        words = []
        for i, text in enumerate(data["text"]):
            confidence = float(data["conf"][i]) / 100
            if not text.strip() or confidence < config.ocr_min_confidence:
                continue
            left, top = int(data["left"][i]), int(data["top"][i])
            right, bottom = left + int(data["width"][i]), top + int(data["height"][i])
            if config.ocr_scale != 1:
                left = max(0, min(image.width, left // config.ocr_scale))
                top = max(0, min(image.height, top // config.ocr_scale))
                right = max(0, min(image.width, (right + config.ocr_scale - 1) // config.ocr_scale))
                bottom = max(0, min(image.height, (bottom + config.ocr_scale - 1) // config.ocr_scale))
                if right <= left or bottom <= top:
                    continue
            words.append(
                WordRow(
                    line_key=(
                        data["page_num"][i],
                        data["block_num"][i],
                        data["par_num"][i],
                        data["line_num"][i],
                    ),
                    text=text,
                    left=left,
                    top=top,
                    width=right - left,
                    height=bottom - top,
                    confidence=min(1.0, confidence),
                )
            )
        proposals = [
            {
                "source": "ocr",
                "text": line.text,
                "confidence": line.confidence,
                "bbox": {
                    "left": line.left,
                    "top": line.top,
                    "right": line.left + line.width,
                    "bottom": line.top + line.height,
                },
            }
            for line in merge_words_into_lines(words)
        ]
        if config.ocr_merge_aligned:
            parents = _aligned_ocr_parents(proposals, image.width)
            info["aligned_parent_policy"] = "native_baseline_gap_v1"
            info["atomic_proposals"] = len(proposals)
            info["aligned_parent_proposals"] = len(parents)
            proposals.extend(parents)
        info["actual_engine"] = "tesseract"
        return proposals, info
    except Exception as exc:  # noqa: BLE001 - recorded explicit contour-only fallback, no model download
        info["error"] = f"{type(exc).__name__}: {exc}"
        return [], info


def detect_candidates(image_path: str | Path, config: DetectorConfig) -> dict[str, Any]:
    """The sole inference candidate producer takes only pixels and config."""
    from ..perception.ui_detection import detect_ui_candidates

    started = time.perf_counter()
    image_path = Path(image_path).resolve(strict=True)
    digest = file_sha256(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    ocr, ocr_info = _ocr_proposals(image, config)
    contours = detect_ui_candidates(
        image,
        min_area=config.min_area,
        max_area_ratio=config.max_area_ratio,
        min_rectangularity=config.min_rectangularity,
        max_candidates=config.contour_budget,
        max_aspect=config.contour_max_aspect,
    )
    contour_proposals = [
        {
            "bbox": item.bounding_box.model_dump(),
            "source": "contour",
            "text": "",
            "confidence": item.confidence,
        }
        for item in contours
    ]
    proposal_counts = {"ocr": len(ocr), "contour": len(contours)}
    proposals = ocr + contour_proposals
    if config.contour_text_lines:
        parents = _contour_text_line_parents(contour_proposals, width, height)
        proposals.extend(parents)
        proposal_counts["contour_text_line_parents"] = len(parents)
    candidates = proposals_to_candidates(proposals, width, height, config.max_candidates)
    if file_sha256(image_path) != digest:
        raise ValueError("image changed while detecting candidates")
    return {
        "image_file_sha256": digest,
        "image_width": width,
        "image_height": height,
        "detector_config": asdict(config),
        "candidates": candidates,
        "proposal_counts": proposal_counts,
        "ocr": ocr_info,
        "mode": "contour+ocr" if ocr_info["actual_engine"] else "contour_only",
        "detection_ms": (time.perf_counter() - started) * 1000,
    }


def validate_snapshot(snapshot: dict[str, Any], image_path: str | Path) -> None:
    if file_sha256(image_path) != snapshot["image_file_sha256"]:
        raise ValueError("candidate snapshot is stale for this image")
    with Image.open(image_path) as image:
        if image.size != (snapshot["image_width"], snapshot["image_height"]):
            raise ValueError("candidate native dimensions differ from the current image")
    ids, boxes = set(), set()
    for candidate in snapshot["candidates"]:
        box = native_box(candidate["bbox"], *image.size)
        geometry = tuple(box[k] for k in ("left", "top", "right", "bottom"))
        if not isinstance(candidate["candidate_id"], str) or not candidate["candidate_id"]:
            raise ValueError("candidate id must be a nonempty string")
        if candidate["candidate_id"] in ids or geometry in boxes:
            raise ValueError("ambiguous duplicate candidate id or bbox")
        if candidate["source"] not in ROLES or candidate["role"] != ROLES[candidate["source"]]:
            raise ValueError("candidate role lacks detection evidence")
        ids.add(candidate["candidate_id"])
        boxes.add(geometry)


def candidate_prompt(instruction: str, snapshot: dict[str, Any]) -> str:
    """No sample/action labels/target boxes can enter this interface."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("nonempty instruction required")
    controls = [
        {k: c[k] for k in ("candidate_id", "bbox", "role", "text")} for c in snapshot["candidates"]
    ]
    return (
        "Choose next executable action(s) for the task from the visible full screenshot. "
        "Candidates are pixel/text proposals, NOT proof of button/editor semantics. "
        "Choose a region only if its visible purpose fits the intended action; do not invent a control. "
        "Return a JSON array alone. click/double_click/right_click/move use only action_type, candidate_id, "
        "candidate_role (copy the detected role). scroll adds nonzero integer scroll_amount. "
        "drag uses action_type, start_candidate_id, start_candidate_role, end_candidate_id, end_candidate_role. "
        "type_text uses only action_type,text; key_press uses action_type,key; hotkey uses action_type,keys; "
        "wait uses action_type,duration. Preserve required text exactly. Never output x/y coordinates. "
        "If no proposal fits, or purpose is ambiguous, return []. Do not repair missing candidates. "
        f"Native screenshot: {snapshot['image_width']}x{snapshot['image_height']}. "
        "Each chosen id will map to its native bbox CENTER; half-open bbox edges.\n"
        f"Detected candidates: {canonical_json(controls)}\nTask: {instruction}"
    )


def map_selection(
    generation: str, snapshot: dict[str, Any], image_path: str | Path
) -> list[dict[str, Any]]:
    validate_snapshot(snapshot, image_path)
    selections = strict_json_loads(generation)
    if not isinstance(selections, list):
        raise TypeError("selection must be a bare JSON array")
    if not selections or not snapshot["candidates"]:
        raise ValueError("no unambiguous candidate selection")
    candidates = {c["candidate_id"]: c for c in snapshot["candidates"]}

    def selected(action: dict[str, Any], prefix: str = "") -> tuple[int, int]:
        identifier = action[prefix + "candidate_id"]
        if not isinstance(identifier, str) or identifier not in candidates:
            raise ValueError("selected id is outside the current candidate set")
        candidate = candidates[identifier]
        if action[prefix + "candidate_role"] != candidate["role"]:
            raise ValueError("selected role differs from detected role")
        return center(candidate["bbox"])

    mapped = []
    for action in selections:
        if not isinstance(action, dict):
            raise TypeError("selection action must be an object")
        kind = action.get("action_type")
        if kind in CLICK_TYPES | {"move", "scroll"}:
            fields = {"action_type", "candidate_id", "candidate_role"}
            if kind == "scroll":
                fields.add("scroll_amount")
            if set(action) != fields:
                raise ValueError("unexpected/missing point selection fields")
            x, y = selected(action)
            result = {"action_type": kind, "x": x, "y": y}
            if kind == "scroll":
                if type(action["scroll_amount"]) is not int or not action["scroll_amount"]:
                    raise ValueError("scroll amount must be a nonzero integer")
                result["scroll_amount"] = action["scroll_amount"]
        elif kind == "drag":
            if set(action) != {
                "action_type",
                "start_candidate_id",
                "start_candidate_role",
                "end_candidate_id",
                "end_candidate_role",
            }:
                raise ValueError("unexpected/missing drag selection fields")
            start, end = selected(action, "start_"), selected(action, "end_")
            result = {
                "action_type": kind,
                "start": dict(zip(("x", "y"), start, strict=True)),
                "end": dict(zip(("x", "y"), end, strict=True)),
            }
        else:
            result = dict(action)
        mapped.append(result)
    return validate_actions(mapped)


def coverage_score(
    actions: list[dict[str, Any]], annotations: list[dict[str, Any]], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Oracle geometric upper bounds ONLY; not model-selected hits or task success."""
    eligible = hit = point_covered = point_exact = 0
    closest_distances = []
    boxes = [c["bbox"] for c in snapshot["candidates"]]
    for i, action in enumerate(actions):
        if action["action_type"] not in CLICK_TYPES:
            continue
        point_covered += any(contains(box, action["x"], action["y"]) for box in boxes)
        point_exact += any(center(box) == (action["x"], action["y"]) for box in boxes)
        closest_distances.append(
            min(
                (
                    math.hypot(
                        (center(box)[0] - action["x"]) / snapshot["image_width"],
                        (center(box)[1] - action["y"]) / snapshot["image_height"],
                    )
                    for box in boxes
                ),
                default=math.sqrt(2),
            )
        )
        annotation = annotations[i] if annotations else {}
        if annotation.get("target_bbox") is None:
            continue
        target = native_box(
            annotation["target_bbox"], snapshot["image_width"], snapshot["image_height"]
        )
        eligible += 1
        hit += any(contains(target, *center(box)) for box in boxes)
    return {
        "click_targets": sum(a["action_type"] in CLICK_TYPES for a in actions),
        "bbox_eligible": eligible,
        "candidate_center_hits_annotation": hit,
        "target_point_inside_candidate": point_covered,
        "exact_candidate_center": point_exact,
        "closest_candidate_center_normalized_distances": closest_distances,
    }


def snapshot_hash(snapshot: dict[str, Any]) -> str:
    return value_sha256({k: v for k, v in snapshot.items() if k != "detection_ms"})
