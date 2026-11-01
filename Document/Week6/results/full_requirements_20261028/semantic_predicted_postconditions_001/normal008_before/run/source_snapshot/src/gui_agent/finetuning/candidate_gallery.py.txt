"""Deterministic native overview and candidate-gallery input codec.

Native RGB occupies the unchanged top rectangle. Tile positions are shuffled,
but candidate integer aliases and native coordinate meanings never change.
This codec is exclusive to the explicit visual_pointer_gallery_v3 protocol.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFont

from .candidate_grounding import center, snapshot_hash
from .evaluation import value_sha256
from .visual_pointer import _raw_protection, _sha, _visible
from .visual_pointer_prompts import (
    GALLERY_PROMPT_VERSION,
    build_visual_user_prompt,
    declared_prompt_version,
)

VERSION = "week5.candidate-gallery.v3"
PROTOCOL = "visual_pointer_gallery_v3"
MANIFEST_SCHEMA = "week5.visual_pointer_gallery_asset_manifest.v1"
FORBIDDEN_SNAPSHOT_KEYS = frozenset({"actions", "action_annotations", "candidate_target", "candidate_label_evidence", "review",
                                     "targets", "request", "response", "plan", "task_record", "sample_id"})
CONFIG = {"columns": 8, "tile_height": 144, "header_height": 74, "thumbnail_height": 70,
          "font_size": 72, "font_layout": "BASIC", "crop_padding_pixels": 8, "seed": 42,
          "glyph_margin_pixels": 7, "resampling": "BICUBIC", "tile_background": [0, 0, 0],
          "glyph_color": [255, 255, 255], "overview_origin": [0, 0]}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def native_hash(rgb, size):
    return hashlib.sha256(f"{size[0]}x{size[1]}:".encode() + rgb).hexdigest()


def canonical_path(value):
    path = Path(value)
    need(path.is_absolute() and path.resolve(strict=True) == path, "source/font path alias or missing file")
    return path


def resolve_gallery_geometry(native_width: int) -> dict:
    """Keep <=1024 pixels exact; use rational half-up scaling for larger UI.

    Quantize total/header separately and give the thumbnail the remaining
    height. This conserves the tile height when separate rounded parts differ
    by one pixel. Crop padding remains in original native pixels.
    """
    need(type(native_width) is int and native_width > 0, "positive native width required")
    numerator, denominator = max(1024, native_width), 1024

    def scaled(value):
        return (2 * value * numerator + denominator) // (2 * denominator)

    tile_height = scaled(CONFIG["tile_height"])
    header_height = scaled(CONFIG["header_height"])
    return {"scale_numerator": numerator, "scale_denominator": denominator,
            "rounding": "positive_rational_half_up; thumbnail=tile_height-header_height",
            "tile_width": native_width // CONFIG["columns"], "tile_height": tile_height,
            "header_height": header_height, "thumbnail_height": tile_height - header_height,
            "font_size": scaled(CONFIG["font_size"]), "glyph_margin_pixels": scaled(CONFIG["glyph_margin_pixels"]),
            "crop_padding_pixels": CONFIG["crop_padding_pixels"]}


def compose_gallery_rgb(native_rgb_bytes: bytes, native_size: tuple[int, int], candidates: Sequence[Mapping], *,
                        font_bytes: bytes, font_sha256: str) -> dict:
    """Pure pixels plus selected input geometry; candidate text/GT is never read.

    crop_affine maps original pixel-centre coordinates to the displayed
    thumbnail. Resizing uses actual rounded x/y ratios, including half pixels.
    """
    need(type(native_rgb_bytes) is bytes and type(native_size) is tuple and len(native_size) == 2
         and all(type(v) is int and v > 0 for v in native_size), "immutable native RGB/positive dimensions required")
    width, height = native_size
    need(len(native_rgb_bytes) == width * height * 3, "native RGB byte length differs")
    need(isinstance(candidates, Sequence) and 0 < len(candidates) <= 64, "complete selected1..64 table required")
    # Only these input fields are read, never dict(candidate) or opaque values.
    records = [{"candidate_id": item["candidate_id"], "raw_candidate_id": item.get("original_id", item["candidate_id"]),
                "native_bbox": {key: item["bbox"][key] for key in ("left", "top", "right", "bottom")}}
               for item in candidates]
    need(all(isinstance(item["candidate_id"], str) and item["candidate_id"] for item in records)
         and len({item["candidate_id"] for item in records}) == len(records), "unique input candidate IDs required")
    for item in records:
        box = item["native_bbox"]
        need(all(type(v) is int for v in box.values()) and 0 <= box["left"] < box["right"] <= width
             and 0 <= box["top"] < box["bottom"] <= height, "candidate native bbox outside image")
    need(type(font_bytes) is bytes and hashlib.sha256(font_bytes).hexdigest() == _sha(font_sha256, "gallery font"),
         "pinned gallery font bytes differ")
    resolved = resolve_gallery_geometry(width)
    font = ImageFont.truetype(io.BytesIO(font_bytes), size=resolved["font_size"], layout_engine=ImageFont.Layout.BASIC)
    native = Image.frombytes("RGB", native_size, native_rgb_bytes)
    columns, tile_w, tile_h = CONFIG["columns"], resolved["tile_width"], resolved["tile_height"]
    need(tile_w > 0, "canvas too narrow for fixed gallery columns")
    canvas = Image.new("RGB", (width, height + math.ceil(len(records) / columns) * tile_h), (0, 0, 0))
    canvas.paste(native, (0, 0))
    draw = ImageDraw.Draw(canvas)
    seed_material = {"base_seed": CONFIG["seed"], "native_rgb_sha256": native_hash(native_rgb_bytes, native_size),
                     "ordered_selected_geometry": records}
    seed_sha256 = value_sha256(seed_material)
    order = list(range(1, len(records) + 1))
    random.Random(int(seed_sha256, 16)).shuffle(order)
    table = []
    for tile_index, identifier in enumerate(order):
        item = records[identifier - 1]
        tile_x, tile_y = tile_index % columns * tile_w, height + tile_index // columns * tile_h
        glyph = font.getbbox(str(identifier))
        glyph_w, glyph_h = glyph[2] - glyph[0], glyph[3] - glyph[1]
        margin = resolved["glyph_margin_pixels"]
        need(0 < glyph_w <= tile_w - 2 * margin and 0 < glyph_h <= resolved["header_height"] - 2 * margin,
             "fixed Arial72 digits/caption margin do not fit; refuse whole gallery")
        glyph_x = tile_x + (tile_w - glyph_w) // 2
        glyph_y = tile_y + (resolved["header_height"] - glyph_h) // 2
        draw.text((glyph_x - glyph[0], glyph_y - glyph[1]), str(identifier), font=font, fill=(255, 255, 255))
        box, padding = item["native_bbox"], CONFIG["crop_padding_pixels"]
        crop_box = (max(0, box["left"] - padding), max(0, box["top"] - padding),
                    min(width, box["right"] + padding), min(height, box["bottom"] + padding))
        crop = native.crop(crop_box)
        fit = min(tile_w / crop.width, resolved["thumbnail_height"] / crop.height)
        tw, th = min(tile_w, max(1, round(crop.width * fit))), min(resolved["thumbnail_height"], max(1, round(crop.height * fit)))
        thumb = crop.resize((tw, th), resample=Image.Resampling.BICUBIC)
        tx = tile_x + (tile_w - tw) // 2
        ty = tile_y + resolved["header_height"] + (resolved["thumbnail_height"] - th) // 2
        canvas.paste(thumb, (tx, ty))
        sx, sy = tw / crop.width, th / crop.height
        table.append({"id": identifier, **item, "native_center": list(center(box)),
                      "tile_index": tile_index, "tile_bbox": [tile_x, tile_y, tile_x + tile_w, tile_y + tile_h],
                      "header_bbox": [tile_x, tile_y, tile_x + tile_w, tile_y + resolved["header_height"]],
                      "id_glyph_bbox": [glyph_x, glyph_y, glyph_x + glyph_w, glyph_y + glyph_h],
                      "crop_bbox": list(crop_box), "thumbnail_bbox": [tx, ty, tx + tw, ty + th],
                      "crop_affine": {"scale_x": sx, "scale_y": sy,
                                      "translate_x": tx - crop_box[0] * sx + (sx - 1) / 2,
                                      "translate_y": ty - crop_box[1] * sy + (sy - 1) / 2,
                                      "coordinates": "native_pixel_centres_to_canvas_pixel_centres"}})
    need(canvas.crop((0, 0, width, height)).tobytes() == native_rgb_bytes, "native overview was modified")
    table.sort(key=lambda item: item["id"])
    need([item["id"] for item in table] == list(range(1, len(records) + 1)), "gallery dropped/duplicated candidate aliases")
    return {"rgb_bytes": canvas.tobytes(), "size": canvas.size,
            "metadata": {"version": VERSION, "status": "input_only_prototype_not_actor_admission",
                         "config": copy.deepcopy(CONFIG), "config_sha256": value_sha256(CONFIG),
                         "resolved_geometry": resolved,
                         "font_sha256": font_sha256, "pillow_version": PIL.__version__,
                         "native_size": list(native_size), "canvas_size": list(canvas.size),
                         "overview_bbox": [0, 0, width, height], "overview_rgb_byteexact": True,
                         "native_rgb_sha256": native_hash(native_rgb_bytes, native_size),
                         "gallery_native_rgb_sha256": native_hash(canvas.tobytes(), canvas.size),
                         "selected_candidate_count": len(records), "tile_display_order": order, "id_table": table,
                         "shuffle_seed_derivation": {"scheme": "week5.candidate-gallery.shuffle.v1",
                                                     "input": seed_material, "input_sha256": seed_sha256,
                                                     "effective_seed_hex": seed_sha256}}}


def render_candidate_gallery(sample: Mapping, *, font_path: str | Path, font_sha256: str,
                             raw_snapshot: Mapping | None = None,
                             visual_prompt_version: str = GALLERY_PROMPT_VERSION) -> dict:
    """Source and all-policy selection guards around the reusable pure renderer."""
    # No whole sample copy/hash: secrets outside this visible list are untouched.
    need(visual_prompt_version == GALLERY_PROMPT_VERSION, "gallery renderer requires its fixed gallery.v4 prompt")
    need("input_image_protocol" not in sample or sample["input_image_protocol"] == PROTOCOL,
         "gallery renderer refuses mixed input protocol")
    need("visual_prompt_version" not in sample or declared_prompt_version(sample) == visual_prompt_version,
         "gallery renderer refuses mixed prompt version")
    for snapshot in (sample["candidate_snapshot"], raw_snapshot):
        if snapshot is not None:
            need(isinstance(snapshot, Mapping) and not FORBIDDEN_SNAPSHOT_KEYS.intersection(snapshot),
                 "gallery snapshot contains forbidden label/request keys")
    path, font_file = canonical_path(sample["image_path"]), canonical_path(font_path)
    inputs = _visible(sample)
    raw = _raw_protection(inputs, raw_snapshot)
    source_bytes, font_bytes = path.read_bytes(), font_file.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    need(source_sha == inputs["candidate_snapshot"]["image_file_sha256"]
         and source_sha == _sha(sample["image_file_sha256"], "native source"), "gallery native source SHA differs")
    with Image.open(io.BytesIO(source_bytes)) as image:
        need(getattr(image, "n_frames", 1) == 1 and image.size == (inputs["image_width"], inputs["image_height"]),
             "gallery original frame/dimensions differ")
        rgb = image.convert("RGB").tobytes()
    native_size = (inputs["image_width"], inputs["image_height"])
    need(native_hash(rgb, native_size) == _sha(sample["image_sha256"], "native RGB"), "gallery native RGB SHA differs")
    result = compose_gallery_rgb(rgb, native_size, inputs["selected"], font_bytes=font_bytes, font_sha256=font_sha256)
    png = io.BytesIO()
    Image.frombytes("RGB", result["size"], result["rgb_bytes"]).save(png, format="PNG", optimize=False)
    png_bytes = png.getvalue()
    need(canonical_path(path) == path and path.read_bytes() == source_bytes and canonical_path(font_file) == font_file
         and font_file.read_bytes() == font_bytes, "gallery original/font changed while rendering")
    final_inputs = _visible(sample)
    final_raw = _raw_protection(final_inputs, raw_snapshot)
    need(final_inputs == inputs and final_raw == raw, "gallery context/selection changed while rendering")
    need(("input_image_protocol" not in sample or sample["input_image_protocol"] == PROTOCOL)
         and ("visual_prompt_version" not in sample or declared_prompt_version(sample) == visual_prompt_version),
         "gallery declared input/prompt version changed while rendering")
    prompt = build_visual_user_prompt(inputs["instruction"], inputs["actor_step"], inputs["candidate_snapshot"],
                                    visual_prompt_version=visual_prompt_version)
    metadata = {**result["metadata"], "original_image_path": str(path), "original_image_file_sha256": source_sha,
                "original_native_rgb_sha256": native_hash(rgb, native_size), "font_path": str(font_file),
                "gallery_png_sha256": hashlib.sha256(png_bytes).hexdigest(), "gallery_png_bytes": len(png_bytes),
                "context_sha256": inputs["candidate_selection_evidence"]["context_sha256"],
                "candidate_snapshot_sha256": snapshot_hash(inputs["candidate_snapshot"]),
                "raw_snapshot_sha256": snapshot_hash(raw), "candidate_selection_evidence": inputs["candidate_selection_evidence"],
                "input_image_protocol": PROTOCOL, "visual_prompt_version": visual_prompt_version,
                "status": "rendered_not_training_admission", "prompt_sha256": hashlib.sha256(prompt.encode("utf8")).hexdigest(),
                "model_png_sha256": hashlib.sha256(png_bytes).hexdigest(), "model_native_rgb_sha256": result["metadata"]["gallery_native_rgb_sha256"],
                "model_png_bytes": len(png_bytes)}
    return {"png_bytes": png_bytes, "metadata": metadata, "prompt": prompt}
