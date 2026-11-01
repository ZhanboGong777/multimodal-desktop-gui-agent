"""Isolated, bounded joint badge/leader layout; inference evidence only.

This changes CPU layout scheduling, not candidates, glyphs, prompt semantics,
budgets, native coordinates or training admission. Frozen v1 remains intact.
"""
from __future__ import annotations

import copy
import hashlib
import io
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFont

from .candidate_grounding import center, snapshot_hash, validate_snapshot
from .evaluation import file_sha256, value_sha256
from .pointer_protocol import pointer_user_prompt
from .visual_pointer import (
    NEAR_ORDER,
    VISUAL_PREFIX,
    VisualLayoutError,
    VisualPointerConfig,
    _config,
    _leader,
    _native_hash,
    _pixels,
    _positions,
    _raw_protection,
    _rect,
    _sha,
    _visible,
)

VERSION = "week5.visual-pointer.v2"
LAYOUT_POLICY = "joint_domain_mrv_forward_check_dfs"
MAX_RANKED_OPTIONS = 4
MAX_ASSIGNMENT_ATTEMPTS = 256
TIE_ORDER = ("top", "left", "bottom", "right", "raw_candidate_id")
COST_ORDER = ("new_union_ocr_pixels", "new_total_paint_pixels", "raster_leader_pixels", "position_index")


@dataclass
class _Geometry:
    domains: list[dict[str, Any]]
    owners: np.ndarray
    boxes: np.ndarray
    indices: np.ndarray
    valid: np.ndarray
    line_ocr: np.ndarray
    line_lengths: np.ndarray
    badge_areas: np.ndarray
    candidate_areas: list[int]
    ties: list[tuple]
    ocr_mask: np.ndarray
    centre_mask: np.ndarray
    ocr_count: int
    static_counts: list[int]


@dataclass
class _State:
    badge: np.ndarray
    spacing: np.ndarray
    leaders: np.ndarray
    paint: np.ndarray
    painted: int = 0
    ocr_painted: int = 0


def _refuse(reason: str, search: dict[str, Any]) -> None:
    error = VisualLayoutError(f"{reason}; refused entire all-ID v2 layout")
    error.layout_status = reason
    error.layout_search = copy.deepcopy(search)
    raise error


def _geometry(size, selected, raw, font, cfg) -> _Geometry:
    """Cache exact static joint domains and straight raster line pixels once."""
    width, height = size
    ocr_image = Image.new("L", size, 0)
    raw_ocr = [c for c in raw["candidates"] if c["source"] == "ocr"]
    for item in raw_ocr:
        _rect(ImageDraw.Draw(ocr_image), tuple(item["bbox"][k] for k in ("left", "top", "right", "bottom")), 255)
    centre_image = Image.new("L", size, 0)
    centres = [center(c["bbox"]) for c in selected]
    for point in centres:
        centre_image.putpixel(point, 255)
    protected = ImageChops.lighter(ocr_image, centre_image)
    domains, pixel_arrays, areas, ties, counts = [], [], [], [], []
    for owner, (candidate, point) in enumerate(zip(selected, centres, strict=True)):
        identifier = owner + 1
        glyph = tuple(font.getbbox(str(identifier)))
        badge_width = glyph[2] - glyph[0] + cfg.badge_padding * 2
        badge_height = glyph[3] - glyph[1] + cfg.badge_padding * 2
        if badge_width <= 0 or badge_height <= 0:
            raise ValueError("pinned font produced invalid numeric glyph geometry")
        areas.append(badge_width * badge_height)
        bbox = candidate["bbox"]
        raw_id = candidate.get("original_id", candidate["candidate_id"])
        ties.append(tuple(bbox[k] for k in TIE_ORDER[:-1]) + (raw_id,))
        count = 0
        for position_index, (left, top, search) in enumerate(_positions(*point, badge_width, badge_height, cfg)):
            box = (left, top, left + badge_width, top + badge_height)
            if left < 0 or top < 0 or box[2] > width or box[3] > height or protected.crop(box).getbbox() is not None:
                continue
            line = _leader(box, point)
            line_left, line_top = min(p[0] for p in line), min(p[1] for p in line)
            line_right, line_bottom = max(p[0] for p in line) + 1, max(p[1] for p in line) + 1
            local = Image.new("L", (line_right - line_left, line_bottom - line_top), 0)
            ImageDraw.Draw(local).line([(x - line_left, y - line_top) for x, y in line], fill=255, width=1)
            ys, xs = np.nonzero(np.asarray(local))
            pixels = ((ys + line_top) * width + xs + line_left).astype(np.int32)
            if not len(pixels) or np.any(pixels < 0) or np.any(pixels >= width * height):
                raise ValueError("cached straight leader escaped the native canvas")
            pixel_arrays.append(pixels)
            domains.append({"owner": owner, "id": identifier, "box": box, "line": line,
                            "glyph": glyph, "search": search, "position_index": position_index,
                            "line_roi": (line_left, line_top, line_right, line_bottom)})
            count += 1
        counts.append(count)
    lengths = np.array([len(p) for p in pixel_arrays], dtype=np.int32)
    max_line = int(lengths.max()) if len(lengths) else 0
    indices = np.zeros((len(domains), max_line), dtype=np.int32)
    valid = np.zeros(indices.shape, dtype=bool)
    for index, pixels in enumerate(pixel_arrays):
        indices[index, :len(pixels)] = pixels
        valid[index, :len(pixels)] = True
    ocr = np.asarray(ocr_image, dtype=bool)
    owners = np.array([d["owner"] for d in domains], dtype=np.int32)
    boxes = np.array([d["box"] for d in domains], dtype=np.int32).reshape((-1, 4))
    return _Geometry(domains, owners, boxes, indices, valid, valid & ocr.ravel()[indices], lengths,
                     np.array([areas[o] for o in owners], dtype=np.int32), areas, ties, ocr,
                     np.asarray(centre_image, dtype=bool), len(raw_ocr), counts)


def _feasible(g: _Geometry, state: _State, remaining: tuple[int, ...], cfg: VisualPointerConfig):
    """Batch ROI-independent cached pixels and rectangle occupancy queries."""
    blocked = state.spacing | state.leaders
    prefix = np.pad(blocked.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32), ((1, 0), (1, 0)))
    left, top, right, bottom = g.boxes.T
    badge_blocked = prefix[bottom, right] - prefix[top, right] - prefix[bottom, left] + prefix[top, left] > 0
    line_blocked = np.any(g.valid & state.badge.ravel()[g.indices], axis=1)
    new_line = g.valid & ~state.paint.ravel()[g.indices]
    new_total = g.badge_areas + np.count_nonzero(new_line, axis=1)
    new_ocr = np.count_nonzero(new_line & g.line_ocr, axis=1)
    reserved = sum(g.candidate_areas[c] for c in remaining)
    feasible = (np.isin(g.owners, remaining) & ~badge_blocked & ~line_blocked
                & (state.painted + new_total + reserved - g.badge_areas <= state.paint.size * cfg.max_paint_fraction)
                & (state.ocr_painted + new_ocr <= np.count_nonzero(g.ocr_mask) * cfg.max_raw_ocr_paint_fraction))
    counts = np.bincount(g.owners, weights=feasible.astype(np.int32), minlength=len(g.candidate_areas)).astype(np.int32)
    return feasible, counts, new_total, new_ocr


def _assign(g: _Geometry, state: _State, domain_index: int, new_total: int, new_ocr: int, gap: int):
    """Apply a joint assignment with exact reversible local state, not canvas copies."""
    domain = g.domains[domain_index]
    left, top, right, bottom = domain["box"]
    width, height = state.paint.shape[1], state.paint.shape[0]
    expanded = (max(0, left - gap), max(0, top - gap), min(width, right + gap), min(height, bottom + gap))
    el, et, er, eb = expanded
    pixels = g.indices[domain_index, :g.line_lengths[domain_index]]
    undo = (expanded, state.spacing[et:eb, el:er].copy(), state.paint.ravel()[pixels].copy(),
            state.leaders.ravel()[pixels].copy(), state.painted, state.ocr_painted)
    state.badge[top:bottom, left:right] = True
    state.spacing[et:eb, el:er] = True
    state.paint[top:bottom, left:right] = True
    state.paint.ravel()[pixels] = True
    state.leaders.ravel()[pixels] = True
    state.painted += int(new_total)
    state.ocr_painted += int(new_ocr)
    return undo


def _undo(g: _Geometry, state: _State, domain_index: int, undo) -> None:
    left, top, right, bottom = g.domains[domain_index]["box"]
    expanded, spacing, paint, leaders, state.painted, state.ocr_painted = undo
    el, et, er, eb = expanded
    pixels = g.indices[domain_index, :g.line_lengths[domain_index]]
    state.badge[top:bottom, left:right] = False
    state.paint[top:bottom, left:right] = False
    state.spacing[et:eb, el:er] = spacing
    state.paint.ravel()[pixels] = paint
    state.leaders.ravel()[pixels] = leaders


def _solve(g: _Geometry, cfg: VisualPointerConfig):
    height, width = g.ocr_mask.shape
    state = _State(*(np.zeros((height, width), dtype=bool) for _ in range(4)))
    search = {"max_ranked_options": MAX_RANKED_OPTIONS, "max_assignment_attempts": MAX_ASSIGNMENT_ATTEMPTS,
              "assignment_attempts": 0, "forward_checks": 0, "backtracks": 0,
              "static_domain_counts": list(g.static_counts), "cached_joint_domains": len(g.domains),
              "cached_line_pixels": int(g.line_lengths.sum()), "tie_order": list(TIE_ORDER),
              "cost_order": list(COST_ORDER), "placement_id_order": [],
              "optimality_or_impossibility_proven": False}
    if any(n == 0 for n in g.static_counts):
        _refuse("static_zero_domain", search)
    global_cap = width * height * cfg.max_paint_fraction
    ocr_cap = np.count_nonzero(g.ocr_mask) * cfg.max_raw_ocr_paint_fraction
    if sum(g.candidate_areas) > global_cap or np.count_nonzero(g.ocr_mask & g.centre_mask) > ocr_cap:
        _refuse("budget_lower_bound", search)
    chosen: dict[int, int] = {}
    order: list[int] = []

    def dfs(remaining: tuple[int, ...], evaluation=None) -> bool:
        if not remaining:
            return True
        feasible, counts, new_total, new_ocr = evaluation if evaluation is not None else _feasible(g, state, remaining, cfg)
        if any(counts[c] == 0 for c in remaining):
            return False
        owner = min(remaining, key=lambda c: (int(counts[c]), *g.ties[c]))
        domains = np.flatnonzero(feasible & (g.owners == owner)).tolist()
        domains.sort(key=lambda d: (int(new_ocr[d]), int(new_total[d]), int(g.line_lengths[d]), g.domains[d]["position_index"]))
        rest = tuple(c for c in remaining if c != owner)
        for d in domains[:MAX_RANKED_OPTIONS]:
            if search["assignment_attempts"] >= MAX_ASSIGNMENT_ATTEMPTS:
                return False
            search["assignment_attempts"] += 1
            undo = _assign(g, state, d, int(new_total[d]), int(new_ocr[d]), cfg.label_gap)
            chosen[owner] = d
            order.append(owner + 1)
            next_evaluation = None
            forward_ok = True
            if rest:
                search["forward_checks"] += 1
                next_evaluation = _feasible(g, state, rest, cfg)
                forward_ok = all(next_evaluation[1][c] > 0 for c in rest)
            if forward_ok and dfs(rest, next_evaluation):
                return True
            order.pop()
            chosen.pop(owner)
            _undo(g, state, d, undo)
            search["backtracks"] += 1
        return False

    remaining = tuple(range(len(g.candidate_areas)))
    initial = _feasible(g, state, remaining, cfg)
    if any(initial[1][c] == 0 for c in remaining):
        _refuse("budget_lower_bound", search)
    if not dfs(remaining, initial):
        _refuse("bounded_search_exhausted", search)
    search["placement_id_order"] = order
    search["status"] = "all_ids_rendered"
    return chosen, state, search


def _layout_v2(image, selected, raw, font, cfg):
    g = _geometry(image.size, selected, raw, font, cfg)
    chosen, state, search = _solve(g, cfg)
    placements = []
    output = image.copy()
    masks = {name: Image.new("L", image.size, 0) for name in ("badge", "leaders", "paint")}
    for owner in range(len(selected)):
        domain = g.domains[chosen[owner]]
        c = selected[owner]
        placements.append({"id": owner + 1, "candidate_id": c["candidate_id"],
                           "raw_candidate_id": c.get("original_id", c["candidate_id"]),
                           "bbox": copy.deepcopy(c["bbox"]), "native_center": list(center(c["bbox"])),
                           "badge_bbox": dict(zip(("left", "top", "right", "bottom"), domain["box"], strict=True)),
                           "leader": [list(p) for p in domain["line"]], "glyph_bbox": list(domain["glyph"]),
                           "search": domain["search"], "static_position_index": domain["position_index"],
                           "static_position_rank": domain["position_index"] + 1})
        _rect(ImageDraw.Draw(masks["badge"]), domain["box"], 255)
        ImageDraw.Draw(masks["leaders"]).line(domain["line"], fill=255, width=1)
    masks["paint"] = ImageChops.lighter(masks["badge"], masks["leaders"])
    # Reconstruct from endpoints/rectangles independently of cached pixel arrays.
    for name in ("badge", "leaders", "paint"):
        if not np.array_equal(np.asarray(masks[name], dtype=bool), getattr(state, name)):
            raise ValueError("v2 sparse/ROI geometry differs from independent full-canvas mask")
    ocr_image = Image.fromarray(g.ocr_mask.astype(np.uint8) * 255)
    centre_image = Image.fromarray(g.centre_mask.astype(np.uint8) * 255)
    painted = _pixels(masks["paint"])
    painted_ocr = _pixels(ImageChops.multiply(masks["paint"], ocr_image))
    if painted != state.painted or painted_ocr != state.ocr_painted:
        raise ValueError("v2 paint accounting differs from independent full-canvas mask")
    if painted > image.width * image.height * cfg.max_paint_fraction or painted_ocr > _pixels(ocr_image) * cfg.max_raw_ocr_paint_fraction:
        raise ValueError("v2 completed full-canvas mask violates configured engineering budgets")
    overlaps = {"badge_raw_ocr_overlap_pixels": _pixels(ImageChops.multiply(masks["badge"], ocr_image)),
                "badge_candidate_center_overlap_pixels": _pixels(ImageChops.multiply(masks["badge"], centre_image)),
                "badge_leader_overlap_pixels": _pixels(ImageChops.multiply(masks["badge"], masks["leaders"]))}
    if any(overlaps.values()):
        raise ValueError("v2 completed badges overlap protected OCR, centers or leaders")
    for i, first in enumerate(placements):
        a = first["badge_bbox"]
        for second in placements[i + 1:]:
            b = second["badge_bbox"]
            if (a["left"] - cfg.label_gap < b["right"] and b["left"] < a["right"] + cfg.label_gap
                    and a["top"] - cfg.label_gap < b["bottom"] and b["top"] < a["bottom"] + cfg.label_gap):
                raise ValueError("v2 completed badges violate the fixed two-pixel gap")
    draw = ImageDraw.Draw(output)
    for row in placements:
        draw.line([tuple(p) for p in row["leader"]], fill=(0, 255, 255), width=1)
    for row in placements:
        box = tuple(row["badge_bbox"][k] for k in ("left", "top", "right", "bottom"))
        _rect(draw, box, (0, 0, 0))
        glyph = row["glyph_bbox"]
        draw.text((box[0] + cfg.badge_padding - glyph[0], box[1] + cfg.badge_padding - glyph[1]),
                  str(row["id"]), font=font, fill=(255, 255, 255))
    difference = ImageChops.difference(image, output)
    changed = ImageChops.lighter(ImageChops.lighter(*difference.split()[:2]), difference.split()[2]).point([0] + [255] * 255)
    if ImageChops.subtract(changed, masks["paint"]).getbbox() is not None:
        raise ValueError("v2 changed pixels escaped the declared paint mask")
    counts = {"canvas_pixels": image.width * image.height, "painted_pixels": painted,
              "changed_pixels": _pixels(changed), "badge_pixels": _pixels(masks["badge"]),
              "leader_pixels": _pixels(masks["leaders"]), "raw_ocr_candidate_count": g.ocr_count,
              "raw_ocr_union_pixels": _pixels(ocr_image), "raw_ocr_painted_pixels": painted_ocr,
              "raw_ocr_changed_pixels": _pixels(ImageChops.multiply(changed, ocr_image)), **overlaps,
              "paint_fraction": painted / (image.width * image.height),
              "raw_ocr_paint_fraction": painted_ocr / _pixels(ocr_image) if _pixels(ocr_image) else 0.0}
    return output, placements, counts, search


def render_visual_pointer_v2(
    sample: Mapping[str, Any], *, font_path: str | Path, font_sha256: str,
    config: VisualPointerConfig | Mapping[str, Any] | None = None,
    raw_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Same native input contract and prompt as v1, with bounded v2 layout only."""
    cfg = _config(config)
    inputs = _visible(sample)
    path = Path(inputs["image_path"])
    source_bytes = path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    snapshot = inputs["candidate_snapshot"]
    if source_sha != snapshot["image_file_sha256"]:
        raise ValueError("visual candidate snapshot is stale for the original native image")
    if inputs["expected_native_file_sha256"] is not None and _sha(inputs["expected_native_file_sha256"], "native file") != source_sha:
        raise ValueError("native image file identity differs")
    validate_snapshot(snapshot, path)
    raw = _raw_protection(inputs, raw_snapshot)
    with Image.open(io.BytesIO(source_bytes)) as original:
        if getattr(original, "n_frames", 1) != 1 or original.size != (inputs["image_width"], inputs["image_height"]):
            raise ValueError("original native image dimensions or frame count differ")
        rgb, rgba = original.convert("RGB"), original.convert("RGBA")
    rgb_sha = _native_hash(rgb)
    if inputs["expected_native_rgb_sha256"] is not None and _sha(inputs["expected_native_rgb_sha256"], "native RGB") != rgb_sha:
        raise ValueError("original native RGB identity differs")
    font_digest = _sha(font_sha256, "pinned font")
    requested_font_path = Path(font_path)
    font_file = requested_font_path.resolve(strict=True)
    font_bytes = font_file.read_bytes()
    if hashlib.sha256(font_bytes).hexdigest() != font_digest:
        raise ValueError("pinned font file SHA256 differs")
    try:
        font = ImageFont.truetype(io.BytesIO(font_bytes), size=cfg.font_size, layout_engine=ImageFont.Layout.BASIC)
    except OSError as exc:
        raise ValueError("pinned font cannot be loaded; no fallback") from exc
    prompt = VISUAL_PREFIX + pointer_user_prompt(inputs["instruction"], inputs["actor_step"], snapshot)

    def recheck() -> None:
        if _visible(sample) != inputs:
            raise ValueError("visual input/context/selection changed during rendering")
        if snapshot_hash(_raw_protection(inputs, raw_snapshot)) != snapshot_hash(raw):
            raise ValueError("raw OCR protection snapshot changed during rendering")
        if path.resolve(strict=True) != path or file_sha256(path) != source_sha:
            raise ValueError("original native image changed during rendering")
        if requested_font_path.resolve(strict=True) != font_file or file_sha256(font_file) != font_digest:
            raise ValueError("pinned font changed during rendering")

    try:
        model_image, placements, counts, search = _layout_v2(rgb, inputs["selected"], raw, font, cfg)
    except VisualLayoutError:
        recheck()
        raise
    payload = io.BytesIO()
    model_image.save(payload, format="PNG", optimize=False, compress_level=9)
    png_bytes = payload.getvalue()
    recheck()
    metadata = {"version": VERSION, "layout_policy": LAYOUT_POLICY, "layout_status": "all_ids_rendered",
                "layout_search": search, "native_width": rgb.width, "native_height": rgb.height,
                "original_image_path": str(path), "original_image_file_sha256": source_sha,
                "original_native_rgb_sha256": rgb_sha, "original_native_rgba_sha256": _native_hash(rgba),
                "model_png_sha256": hashlib.sha256(png_bytes).hexdigest(),
                "model_native_rgb_sha256": _native_hash(model_image), "model_png_bytes": len(png_bytes),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "context_sha256": inputs["candidate_selection_evidence"]["context_sha256"],
                "candidate_snapshot_sha256": snapshot_hash(snapshot),
                "candidate_selection_evidence": inputs["candidate_selection_evidence"],
                "raw_protection_snapshot_sha256": snapshot_hash(raw), "font_path": str(font_file),
                "font_sha256": font_digest, "pillow_version": PIL.__version__, "numpy_version": np.__version__,
                "config": asdict(cfg), "config_sha256": value_sha256(asdict(cfg)),
                "style": {"badge": "black", "digits": "white", "leader": "cyan", "leader_width": 1},
                "search_order": list(NEAR_ORDER), "search_ring_step": cfg.ring_step,
                "search_max_radius": cfg.max_radius, "maximum_trial_positions_per_id": 264,
                "selected_candidate_count": len(placements), "id_table": placements, "paint": counts,
                "all_selected_ids_rendered": True, "native_coordinate_mapping_preserved": True,
                "annotation_origin": "input-only engineering annotation; no new semantic review or gold",
                "processor_or_token_measurement_performed": False, "training_admission": False,
                "model_quality_proven": False}
    return {"png_bytes": png_bytes, "metadata": metadata, "prompt": prompt}
