"""Input-only native-size visual pointer annotations; no model or detector calls.

This renderer is an engineering preflight, not token admission, new gold or
proof that a processor/model can read its badges. Mapping stays on the original
native screenshot and the original frozen selected candidate table.
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFont

from .candidate_grounding import center, snapshot_hash, validate_snapshot
from .evaluation import file_sha256, value_sha256
from .pointer_protocol import (
    CONTEXT_POLICY,
    build_pointer_messages,
    pointer_user_prompt,
    select_candidates,
    selection_binding,
)
from .visual_pointer_prompts import (
    LEGACY_VISUAL_PREFIX,
    build_visual_user_prompt,
    checked_prompt_version,
    declared_prompt_version,
)

VERSION = "week5.visual-pointer.v1"
VISUAL_PREFIX = LEGACY_VISUAL_PREFIX
NEAR_ORDER = ("NE", "E", "SE", "S", "SW", "W", "NW", "N")


class VisualLayoutError(ValueError):
    """Only whole-layout infeasibility or engineering paint-budget refusal."""


@dataclass(frozen=True)
class VisualPointerConfig:
    font_size: int = 27
    badge_padding: int = 2
    label_gap: int = 2
    leader_width: int = 1
    ring_step: int = 8
    max_radius: int = 256
    max_paint_fraction: float = 0.08
    max_raw_ocr_paint_fraction: float = 0.02


def _config(config: VisualPointerConfig | Mapping[str, Any] | None) -> VisualPointerConfig:
    if config is None:
        result = VisualPointerConfig()
    elif isinstance(config, VisualPointerConfig):
        result = config
    elif isinstance(config, Mapping):
        if set(config) - set(asdict(VisualPointerConfig())):
            raise ValueError("unknown visual configuration field")
        result = VisualPointerConfig(**dict(config))
    else:
        raise TypeError("visual config must be a configuration or mapping")
    defaults = VisualPointerConfig()
    for key in ("font_size", "badge_padding", "label_gap", "leader_width", "ring_step", "max_radius"):
        if type(getattr(result, key)) is not int or getattr(result, key) != getattr(defaults, key):
            raise ValueError("visual geometry must use the fixed registered integer configuration")
    for key in ("max_paint_fraction", "max_raw_ocr_paint_fraction"):
        value = getattr(result, key)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= getattr(defaults, key):
            raise ValueError("visual mask budgets must be finite and cannot exceed fixed caps")
    return result


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA256 hex string")
    return value.lower()


def _visible(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Only access the explicitly allowed inference input fields."""
    width, height = sample["image_width"], sample["image_height"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("native visual dimensions must be positive integers")
    instruction, step = sample["instruction"], sample["actor_step"]
    if any(not isinstance(v, str) or not v.strip() for v in (instruction, step)):
        raise ValueError("visual input requires an explicit goal and current subtask")
    snapshot = sample["candidate_snapshot"]
    if not isinstance(snapshot, Mapping):
        raise TypeError("visual candidate snapshot must be an object")
    snapshot = copy.deepcopy(dict(snapshot))
    if (snapshot["image_width"], snapshot["image_height"]) != (width, height):
        raise ValueError("visual snapshot native dimensions differ")
    expected = sample.get("candidate_selection_evidence")
    binding = selection_binding(snapshot, instruction=instruction, actor_step=step)
    if not isinstance(expected, Mapping) or dict(expected) != binding:
        raise ValueError("visual candidate selection binding differs")
    selected = select_candidates(snapshot, instruction=instruction, actor_step=step)
    if not 0 < len(selected) <= 64:
        raise ValueError("visual table must contain exactly the selected 1..64 candidates")
    image_path = sample["image_path"]
    if not isinstance(image_path, (str, Path)):
        raise TypeError("original native image_path required")
    return {"image_path": str(Path(image_path).resolve(strict=True)), "image_width": width,
            "image_height": height, "instruction": instruction, "actor_step": step,
            "candidate_snapshot": snapshot, "candidate_selection_evidence": binding,
            "selected": selected, "expected_native_rgb_sha256": sample.get("image_sha256"),
            "expected_native_file_sha256": sample.get("image_file_sha256")}


def _raw_protection(inputs: Mapping[str, Any], raw_snapshot: Mapping[str, Any] | None) -> dict:
    snapshot = inputs["candidate_snapshot"]
    numbered = ("numbering_version" in snapshot or snapshot.get("preselection_policy") == CONTEXT_POLICY)
    if numbered and raw_snapshot is None:
        raise ValueError("numbered visual view requires its bound original raw snapshot")
    raw = copy.deepcopy(dict(raw_snapshot)) if isinstance(raw_snapshot, Mapping) else None
    if raw_snapshot is not None and raw is None:
        raise TypeError("raw protection snapshot must be an object")
    if raw is None:
        raw = copy.deepcopy(snapshot)
    expected = snapshot.get("parent_raw_snapshot_sha256") if numbered else snapshot_hash(snapshot)
    if snapshot_hash(raw) != expected:
        raise ValueError("raw OCR protection snapshot binding differs")
    if (raw["image_width"], raw["image_height"], raw["image_file_sha256"]) != (
            inputs["image_width"], inputs["image_height"], snapshot["image_file_sha256"]):
        raise ValueError("raw protection screenshot identity differs")
    validate_snapshot(raw, inputs["image_path"])
    # Validate the complete raw candidate evidence without requiring its selection
    # to become the display table; all raw OCR boxes remain protected.
    raw_selected = select_candidates(raw, instruction=inputs["instruction"], actor_step=inputs["actor_step"])
    if numbered:
        if snapshot.get("parent_context_sha256") != inputs["candidate_selection_evidence"]["context_sha256"]:
            raise ValueError("numbered parent context differs")
        original = {c["candidate_id"]: c for c in raw_selected}
        selected = inputs["selected"]
        if len(selected) != len(original) or {c.get("original_id") for c in selected} != set(original):
            raise ValueError("numbered view does not preserve the whole original selected set")
        for item in selected:
            source = original[item["original_id"]]
            if any(item.get(k) != source.get(k) for k in ("bbox", "source", "role", "text", "confidence")):
                raise ValueError("numbered candidate changed original raw evidence")
    return raw


def _pixels(mask: Image.Image) -> int:
    return mask.histogram()[255]


def _rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: Any) -> None:
    left, top, right, bottom = box
    draw.rectangle((left, top, right - 1, bottom - 1), fill=fill)


def _positions(cx: int, cy: int, width: int, height: int, cfg: VisualPointerConfig):
    gap = cfg.label_gap
    near = ((cx + gap, cy - height - gap), (cx + gap, cy - height // 2),
            (cx + gap, cy + gap), (cx - width // 2, cy + gap),
            (cx - width - gap, cy + gap), (cx - width - gap, cy - height // 2),
            (cx - width - gap, cy - height - gap), (cx - width // 2, cy - height - gap))
    seen = set()
    for index, (left, top) in enumerate(near):
        seen.add((left, top))
        yield left, top, {"kind": "near", "direction": NEAR_ORDER[index], "radius": None}
    directions = ((1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1))
    for radius in range(cfg.ring_step, cfg.max_radius + 1, cfg.ring_step):
        for index, (dx, dy) in enumerate(directions):
            divisor = math.sqrt(2) if dx and dy else 1
            left = cx + round(radius * dx / divisor) - width // 2
            top = cy + round(radius * dy / divisor) - height // 2
            if (left, top) not in seen:
                seen.add((left, top))
                yield left, top, {"kind": "ring", "direction": NEAR_ORDER[index], "radius": radius}


def _leader(box: tuple[int, int, int, int], target: tuple[int, int]):
    left, top, right, bottom = box
    cx, cy = target
    def clamp(value: int, low: int, high: int) -> int:
        return max(low, min(value, high))
    if cy < top:
        anchor = (clamp(cx, left, right - 1), top - 1)
    elif cy >= bottom:
        anchor = (clamp(cx, left, right - 1), bottom)
    elif cx < left:
        anchor = (left - 1, cy)
    else:
        anchor = (right, cy)
    return anchor, target


def _line_trial(line, paint_mask, badge_mask, ocr_mask, prior_painted, prior_ocr, badge_area):
    """Equivalent new-pixel accounting in the line ROI, without canvas-size trials."""
    left, top = min(p[0] for p in line), min(p[1] for p in line)
    right, bottom = max(p[0] for p in line) + 1, max(p[1] for p in line) + 1
    roi = (left, top, right, bottom)
    local = Image.new("L", (right - left, bottom - top), 0)
    ImageDraw.Draw(local).line([(x - left, y - top) for x, y in line], fill=255, width=1)
    if ImageChops.multiply(local, badge_mask.crop(roi)).getbbox() is not None:
        return None
    new_line = ImageChops.subtract(local, paint_mask.crop(roi))
    return (roi, local, prior_painted + badge_area + _pixels(new_line),
            prior_ocr + _pixels(ImageChops.multiply(new_line, ocr_mask.crop(roi))))


def _layout(image: Image.Image, selected: list[dict], raw: dict, font: Any, cfg: VisualPointerConfig):
    size = image.size
    width, height = size
    ocr_mask = Image.new("L", size, 0)
    ocr_draw = ImageDraw.Draw(ocr_mask)
    ocr = [c for c in raw["candidates"] if c["source"] == "ocr"]
    for candidate in ocr:
        b = candidate["bbox"]
        _rect(ocr_draw, tuple(b[k] for k in ("left", "top", "right", "bottom")), 255)
    ocr_pixels = _pixels(ocr_mask)
    centre_mask = Image.new("L", size, 0)
    centres = [center(c["bbox"]) for c in selected]
    for point in centres:
        centre_mask.putpixel(point, 255)
    badge_mask = Image.new("L", size, 0)
    spacing_mask = Image.new("L", size, 0)
    leaders_mask = Image.new("L", size, 0)
    paint_mask = Image.new("L", size, 0)
    prior_painted, prior_ocr = 0, 0
    placements = []
    for identifier, (candidate, point) in enumerate(zip(selected, centres, strict=True), 1):
        glyph = font.getbbox(str(identifier))
        badge_width = glyph[2] - glyph[0] + 2 * cfg.badge_padding
        badge_height = glyph[3] - glyph[1] + 2 * cfg.badge_padding
        if badge_width <= 0 or badge_height <= 0:
            raise ValueError("pinned font produced invalid numeric glyph geometry")
        badge_area = badge_width * badge_height
        # Every feasible badge must be disjoint from all earlier badges and
        # leaders. Its whole area is therefore an unavoidable new paint cost.
        if prior_painted + badge_area > width * height * cfg.max_paint_fraction:
            raise VisualLayoutError(f"visual mask budget exceeded; refused entire layout at ID {identifier}")
        accepted = None
        budget_refused = False
        attempts = 0
        for left, top, search in _positions(*point, badge_width, badge_height, cfg):
            attempts += 1
            box = (left, top, left + badge_width, top + badge_height)
            if left < 0 or top < 0 or box[2] > width or box[3] > height:
                continue
            if any(mask.crop(box).getbbox() is not None for mask in
                   (spacing_mask, centre_mask, ocr_mask, leaders_mask)):
                continue
            line = _leader(box, point)
            trial = _line_trial(line, paint_mask, badge_mask, ocr_mask, prior_painted, prior_ocr, badge_area)
            if trial is None:
                continue
            roi, local_line, painted, ocr_painted = trial
            if painted > width * height * cfg.max_paint_fraction or (
                    ocr_pixels and ocr_painted > ocr_pixels * cfg.max_raw_ocr_paint_fraction):
                budget_refused = True
                continue
            accepted = (box, line, roi, local_line, painted, ocr_painted, search, attempts)
            break
        if accepted is None:
            reason = "visual mask budget exceeded" if budget_refused else "no feasible all-ID visual layout"
            raise VisualLayoutError(f"{reason}; refused entire layout at ID {identifier}")
        box, line, roi, local_line, prior_painted, prior_ocr, search, attempts = accepted
        line_mask = Image.new("L", size, 0)
        line_mask.paste(local_line, roi[:2])
        paint_mask = ImageChops.lighter(paint_mask, line_mask)
        _rect(ImageDraw.Draw(paint_mask), box, 255)
        _rect(ImageDraw.Draw(badge_mask), box, 255)
        gap = cfg.label_gap
        expanded = (max(0, box[0] - gap), max(0, box[1] - gap),
                    min(width, box[2] + gap), min(height, box[3] + gap))
        _rect(ImageDraw.Draw(spacing_mask), expanded, 255)
        leaders_mask = ImageChops.lighter(leaders_mask, line_mask)
        placements.append({"id": identifier, "candidate_id": candidate["candidate_id"],
                           "raw_candidate_id": candidate.get("original_id", candidate["candidate_id"]),
                           "bbox": copy.deepcopy(candidate["bbox"]), "native_center": list(point),
                           "badge_bbox": dict(zip(("left", "top", "right", "bottom"), box, strict=True)),
                           "leader": [list(p) for p in line], "glyph_bbox": list(glyph),
                           "search": search, "placement_attempts": attempts})
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for item in placements:
        draw.line([tuple(p) for p in item["leader"]], fill=(0, 255, 255), width=cfg.leader_width)
    for item in placements:
        b = item["badge_bbox"]
        box = tuple(b[k] for k in ("left", "top", "right", "bottom"))
        _rect(draw, box, (0, 0, 0))
        glyph = item["glyph_bbox"]
        draw.text((box[0] + cfg.badge_padding - glyph[0], box[1] + cfg.badge_padding - glyph[1]),
                  str(item["id"]), font=font, fill=(255, 255, 255))
    difference = ImageChops.difference(image, output)
    changed = ImageChops.lighter(ImageChops.lighter(*difference.split()[:2]), difference.split()[2])
    changed = changed.point([0] + [255] * 255)
    if ImageChops.subtract(changed, paint_mask).getbbox() is not None:
        raise ValueError("rendered pixels escaped the declared annotation paint mask")
    counts = {"canvas_pixels": width * height, "painted_pixels": _pixels(paint_mask),
              "changed_pixels": _pixels(changed), "badge_pixels": _pixels(badge_mask),
              "leader_pixels": _pixels(leaders_mask), "raw_ocr_candidate_count": len(ocr),
              "raw_ocr_union_pixels": ocr_pixels,
              "raw_ocr_painted_pixels": _pixels(ImageChops.multiply(paint_mask, ocr_mask)),
              "raw_ocr_changed_pixels": _pixels(ImageChops.multiply(changed, ocr_mask)),
              "badge_raw_ocr_overlap_pixels": _pixels(ImageChops.multiply(badge_mask, ocr_mask)),
              "badge_leader_overlap_pixels": _pixels(ImageChops.multiply(badge_mask, leaders_mask)),
              "badge_candidate_center_overlap_pixels": _pixels(ImageChops.multiply(badge_mask, centre_mask))}
    if counts["painted_pixels"] != prior_painted or counts["raw_ocr_painted_pixels"] != prior_ocr:
        raise ValueError("visual paint accounting differs from the completed full-canvas mask")
    if counts["painted_pixels"] > width * height * cfg.max_paint_fraction or (
            ocr_pixels and counts["raw_ocr_painted_pixels"] > ocr_pixels * cfg.max_raw_ocr_paint_fraction):
        raise ValueError("completed visual mask violates its configured engineering budgets")
    if any(counts[k] for k in ("badge_raw_ocr_overlap_pixels", "badge_leader_overlap_pixels",
                              "badge_candidate_center_overlap_pixels")):
        raise ValueError("completed visual badges violate protected OCR, leaders or centers")
    counts["paint_fraction"] = counts["painted_pixels"] / counts["canvas_pixels"]
    counts["raw_ocr_paint_fraction"] = counts["raw_ocr_painted_pixels"] / ocr_pixels if ocr_pixels else 0.0
    return output, placements, counts


def _native_hash(image: Image.Image) -> str:
    return hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()


def render_visual_pointer(
    sample: Mapping[str, Any], *, font_path: str | Path, font_sha256: str,
    config: VisualPointerConfig | Mapping[str, Any] | None = None,
    raw_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render every selected ID or refuse the whole layout; never read labels."""
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
        rgb = original.convert("RGB")
        rgba = original.convert("RGBA")
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
        model_image, placements, counts = _layout(rgb, inputs["selected"], raw, font, cfg)
    except VisualLayoutError:
        recheck()  # Source/configuration integrity errors must never become layout failures.
        raise
    payload = io.BytesIO()
    model_image.save(payload, format="PNG", optimize=False, compress_level=9)
    png_bytes = payload.getvalue()
    recheck()
    metadata = {
        "version": VERSION, "native_width": rgb.width, "native_height": rgb.height,
        "original_image_path": str(path), "original_image_file_sha256": source_sha,
        "original_native_rgb_sha256": rgb_sha, "original_native_rgba_sha256": _native_hash(rgba),
        "model_png_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "model_native_rgb_sha256": _native_hash(model_image), "model_png_bytes": len(png_bytes),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "context_sha256": inputs["candidate_selection_evidence"]["context_sha256"],
        "candidate_snapshot_sha256": snapshot_hash(snapshot),
        "candidate_selection_evidence": inputs["candidate_selection_evidence"],
        "raw_protection_snapshot_sha256": snapshot_hash(raw),
        "font_path": str(font_file), "font_sha256": font_digest, "pillow_version": PIL.__version__,
        "config": asdict(cfg), "config_sha256": value_sha256(asdict(cfg)),
        "style": {"badge": "black", "digits": "white", "leader": "cyan", "leader_width": 1},
        "search_order": list(NEAR_ORDER), "search_ring_step": cfg.ring_step,
        "search_max_radius": cfg.max_radius, "maximum_trial_positions_per_id": 264,
        "selected_candidate_count": len(placements), "id_table": placements,
        "paint": counts, "all_selected_ids_rendered": True, "native_coordinate_mapping_preserved": True,
        "annotation_origin": "input-only engineering annotation; no new semantic review or gold",
        "processor_or_token_measurement_performed": False, "training_admission": False,
        "model_quality_proven": False,
    }
    return {"png_bytes": png_bytes, "metadata": metadata, "prompt": prompt}


def build_visual_pointer_messages(
    sample: Mapping[str, Any], *, include_target: bool = False,
    visual_prompt_version: str | None = None,
) -> list[dict[str, Any]]:
    """Same visual user prefix at training/inference; supervision stays label-side."""
    if type(include_target) is not bool:
        raise TypeError("include_target must be an explicit boolean")
    before = _visible(sample)
    version = checked_prompt_version(visual_prompt_version)
    if declared_prompt_version(sample) != version:
        raise ValueError("sample and caller visual_prompt_version differ")
    messages = build_pointer_messages(sample, include_target=include_target)
    messages[0]["content"][1]["text"] = build_visual_user_prompt(
        before["instruction"], before["actor_step"], before["candidate_snapshot"], visual_prompt_version=version)
    if _visible(sample) != before or declared_prompt_version(sample) != version:
        raise ValueError("visual message input/context/selection changed")
    return messages
