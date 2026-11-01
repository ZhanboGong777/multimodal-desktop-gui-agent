"""Pure input-only geometry recipe; no file, image, processor or model access.

Cells describe nominal geometric footprints and location anchors. Bicubic resize
and vision/language attention can mix information across those footprints.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from fractions import Fraction

SCHEMA = "week5.native_grid_geometry.recipe.v1"
ORDER = "time, merged_row, merged_col; raster after Qwen inverse window_index"
SCOPE = "nominal geometric cell/anchor; not exact resampler or model receptive field"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def integers(value, size, name, *, positive=False):
    require(isinstance(value, (list, tuple)) and len(value) == size
            and all(type(item) is int for item in value), f"{name}: exact integer array required")
    require(not positive or all(item > 0 for item in value), f"{name}: positive dimensions required")
    return tuple(value)


def box(value, bounds, name):
    result = integers(value, 4, name)
    left, top, right, bottom = result
    width, height = bounds
    require(0 <= left < right <= width and 0 <= top < bottom <= height,
            f"{name}: nonempty pixel-edge rectangle outside its domain")
    return result


def contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def overlaps(first, second):
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(first[3], second[3])


def _input_geometry(metadata):
    """Explicit whitelisted reads only; opaque GT/target fields remain untouched."""
    require(isinstance(metadata, Mapping), "metadata mapping required")
    native = integers(metadata["native_size"], 2, "native_size", positive=True)
    canvas = integers(metadata["canvas_size"], 2, "canvas_size", positive=True)
    overview = box(metadata["overview_bbox"], canvas, "overview_bbox")
    require(canvas[0] == native[0] and canvas[1] >= native[1]
            and overview == (0, 0, *native) and metadata["overview_rgb_byteexact"] is True,
            "preapproved byteexact native overview at canvas origin required")
    table = metadata["id_table"]
    require(isinstance(table, list) and len(table) <= 64, "complete input table0..64 required")
    clean, tiles = [], []
    for entry in table:
        require(isinstance(entry, Mapping), "table entry mapping required")
        identifier, tile_index = entry["id"], entry["tile_index"]
        require(type(identifier) is type(tile_index) is int, "integer input id/tile_index required")
        tile = box(entry["tile_bbox"], canvas, "tile_bbox")
        header = box(entry["header_bbox"], canvas, "header_bbox")
        thumbnail = box(entry["thumbnail_bbox"], canvas, "thumbnail_bbox")
        crop = box(entry["crop_bbox"], native, "crop_bbox")
        require(tile[1] >= native[1] and contains(tile, header) and contains(tile, thumbnail)
                and header[0] == tile[0] and header[1] == tile[1] and header[2] == tile[2]
                and header[3] <= thumbnail[1] and not overlaps(header, thumbnail),
                "thumbnail/header must remain in one nonnative tile with no overlap")
        require(all(not overlaps(tile, prior) for prior in tiles), "overlapping input tiles are ambiguous")
        tiles.append(tile)
        sx = Fraction(thumbnail[2] - thumbnail[0], crop[2] - crop[0])
        sy = Fraction(thumbnail[3] - thumbnail[1], crop[3] - crop[1])
        affine = entry["crop_affine"]
        require(isinstance(affine, Mapping), "crop_affine mapping required")
        require(affine["coordinates"] == "native_pixel_centres_to_canvas_pixel_centres", "affine pixel-centre convention differs")
        # Reproduce renderer float arithmetic to audit saved metadata; geometry
        # calculations below use the independent exact integer ratios instead.
        expected = {"scale_x": float(sx), "scale_y": float(sy),
                    "translate_x": thumbnail[0] - crop[0] * float(sx) + (float(sx) - 1) / 2,
                    "translate_y": thumbnail[1] - crop[1] * float(sy) + (float(sy) - 1) / 2}
        actual = {}
        for key, wanted in expected.items():
            observed = affine[key]
            require(type(observed) in (int, float) and math.isfinite(observed)
                    and math.isclose(observed, wanted, rel_tol=1e-12, abs_tol=1e-10),
                    "stored rounded half-pixel affine differs from integer crop/thumbnail geometry")
            actual[key] = observed
        clean.append({"id": identifier, "tile_index": tile_index, "tile_bbox": list(tile),
                      "header_bbox": list(header), "thumbnail_bbox": list(thumbnail),
                      "crop_bbox": list(crop), "crop_affine": {**actual, "coordinates": affine["coordinates"]}})
    require([entry["id"] for entry in clean] == list(range(1, len(clean) + 1))
            and sorted(entry["tile_index"] for entry in clean) == list(range(len(clean))),
            "complete ordered ids and unique raster tile indices required")
    return {"native_size": list(native), "canvas_size": list(canvas), "overview_bbox": list(overview),
            "overview_rgb_byteexact": True, "id_table": clean}


def _floating(values):
    return [float(value) for value in values]


def _unit(value, size):
    return Fraction(0) if size == 1 else value / (size - 1)


def build_recipe(metadata, grid_thw, *, patch_size=14, merge_size=2):
    """Generate every token independent of target labels and proposal relevance.

    Cell edges use half-open pixel-edge coordinates; centres use zero-based
    pixel-centre indices. Static bounds are a descriptive feature prior, never
    a runtime coordinate correction or evidence of learned model outputs.
    """
    geometry = _input_geometry(metadata)
    t, grid_h, grid_w = integers(grid_thw, 3, "grid_thw", positive=True)
    require(t == 1, "exactly one still-image temporal grid required")
    require(type(patch_size) is type(merge_size) is int and patch_size > 0 and merge_size > 0
            and grid_h % merge_size == grid_w % merge_size == 0,
            "positive patch/merge and divisible nonmerged grid required")
    width, height = geometry["native_size"]
    canvas_w, canvas_h = geometry["canvas_size"]
    regions = [("overview", 0, geometry["overview_bbox"], None)]
    regions.extend(("thumbnail", entry["id"], entry["thumbnail_bbox"], entry["crop_bbox"])
                   for entry in geometry["id_table"])
    tokens, merged_h, merged_w = [], grid_h // merge_size, grid_w // merge_size
    native_keys = ("native_center", "native_unit_center", "native_cell_edges", "native_cell_span",
                   "native_unit_span", "static_bounds_native_xy", "static_bounds_native_unit_xy")
    for row in range(merged_h):
        for col in range(merged_w):
            cell = (Fraction(col * merge_size * canvas_w, grid_w), Fraction(row * merge_size * canvas_h, grid_h),
                    Fraction((col + 1) * merge_size * canvas_w, grid_w), Fraction((row + 1) * merge_size * canvas_h, grid_h))
            centre = ((cell[0] + cell[2]) / 2 - Fraction(1, 2), (cell[1] + cell[3]) / 2 - Fraction(1, 2))
            candidates = [region for region in regions if contains(region[2], cell)]
            intersected = [region for region in regions if overlaps(region[2], cell)]
            token = {"token_index": len(tokens), "time_index": 0, "merged_row": row, "merged_col": col,
                     "canvas_center": _floating(centre), "canvas_cell_edges": _floating(cell), "valid": False,
                     "mask_reason": "cross_region" if len(intersected) > 1 else "partial_region" if intersected else "no_native_region",
                     "region_kind": None, "region_id": None, **dict.fromkeys(native_keys)}
            if len(candidates) == 1:
                kind, identifier, region, crop = candidates[0]
                if kind == "overview":
                    native_cell = cell
                else:
                    sx = Fraction(region[2] - region[0], crop[2] - crop[0])
                    sy = Fraction(region[3] - region[1], crop[3] - crop[1])
                    native_cell = (crop[0] + (cell[0] - region[0]) / sx, crop[1] + (cell[1] - region[1]) / sy,
                                   crop[0] + (cell[2] - region[0]) / sx, crop[1] + (cell[3] - region[1]) / sy)
                native_center = ((native_cell[0] + native_cell[2]) / 2 - Fraction(1, 2),
                                 (native_cell[1] + native_cell[3]) / 2 - Fraction(1, 2))
                # A one-pixel axis has one representable native coordinate.
                native_center = tuple(Fraction(0) if size == 1 else value for value, size in zip(native_center, (width, height), strict=True))
                if all(0 <= value <= size - 1 for value, size in zip(native_center, (width, height), strict=True)):
                    span = (native_cell[2] - native_cell[0], native_cell[3] - native_cell[1])
                    lower = tuple(max(Fraction(0), value - extent) for value, extent in zip(native_center, span, strict=True))
                    upper = tuple(min(Fraction(size - 1), value + extent) for value, extent, size in zip(native_center, span, (width, height), strict=True))
                    bounds = (*lower, *upper)
                    token.update(valid=True, mask_reason=None, region_kind=kind, region_id=identifier,
                                 native_center=_floating(native_center), native_cell_edges=_floating(native_cell),
                                 native_cell_span=_floating(span), native_unit_center=_floating([_unit(value, size) for value, size in zip(native_center, (width, height), strict=True)]),
                                 native_unit_span=_floating([_unit(value, size) for value, size in zip(span, (width, height), strict=True)]),
                                 static_bounds_native_xy=_floating(bounds),
                                 static_bounds_native_unit_xy=_floating([_unit(value, size) for value, size in zip(bounds, (width, height, width, height), strict=True)]))
                else:
                    token["mask_reason"] = "native_center_outside_pixel_domain"
            tokens.append(token)
    config = {"geometry": geometry, "grid_thw": [t, grid_h, grid_w], "patch_size": patch_size, "merge_size": merge_size}
    return {"schema": SCHEMA, "geometry_input_sha256": value_sha(config), "grid_thw": [t, grid_h, grid_w],
            "native_size": [width, height], "canvas_size": [canvas_w, canvas_h],
            "resized_size": [grid_w * patch_size, grid_h * patch_size], "merged_shape": [t, merged_h, merged_w],
            "token_order": ORDER, "scope": SCOPE, "geometry_only_no_pixel_validation": True,
            "static_bounds_policy": "anchor +/- one cell native span, intersect representable native-coordinate domain; descriptive only",
            "token_count": len(tokens), "tokens": tokens}
