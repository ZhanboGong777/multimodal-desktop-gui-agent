"""Synthetic metadata/grid tests only; no pixels, labels, files or model calls."""
from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from gui_agent.finetuning import native_spatial_grid_geometry as recipe_module

NATIVE_FIELDS = ("native_center", "native_unit_center", "native_cell_edges", "native_cell_span",
                 "native_unit_span", "static_bounds_native_xy", "static_bounds_native_unit_xy")


def tile(identifier, index, tile_box, header_box, crop_box, thumbnail_box):
    sx = (thumbnail_box[2] - thumbnail_box[0]) / (crop_box[2] - crop_box[0])
    sy = (thumbnail_box[3] - thumbnail_box[1]) / (crop_box[3] - crop_box[1])
    return {"id": identifier, "tile_index": index, "tile_bbox": tile_box, "header_bbox": header_box,
            "crop_bbox": crop_box, "thumbnail_bbox": thumbnail_box,
            "crop_affine": {"scale_x": sx, "scale_y": sy,
                            "translate_x": thumbnail_box[0] - crop_box[0] * sx + (sx - 1) / 2,
                            "translate_y": thumbnail_box[1] - crop_box[1] * sy + (sy - 1) / 2,
                            "coordinates": "native_pixel_centres_to_canvas_pixel_centres"}}


def scene():
    return {"native_size": [112, 56], "canvas_size": [112, 112], "overview_bbox": [0, 0, 112, 56],
            "overview_rgb_byteexact": True,
            "id_table": [tile(1, 0, [0, 56, 56, 112], [0, 56, 56, 70], [0, 0, 112, 56], [0, 70, 56, 112])]}


def odd_scene():
    return {"native_size": [101, 73], "canvas_size": [101, 137], "overview_bbox": [0, 0, 101, 73],
            "overview_rgb_byteexact": True,
            "id_table": [tile(1, 0, [0, 73, 101, 137], [0, 73, 101, 85], [7, 11, 38, 58], [37, 85, 60, 120])]}


def test_default_grid_is_merged_raster_with_whole_cell_masks():
    result = recipe_module.build_recipe(scene(), [1, 8, 8])
    assert result["merged_shape"] == [1, 4, 4]
    assert result["resized_size"] == [112, 112]
    assert result["token_count"] == 16
    assert [(item["token_index"], item["time_index"], item["merged_row"], item["merged_col"])
            for item in result["tokens"]] == [(row * 4 + col, 0, row, col) for row in range(4) for col in range(4)]
    assert {item["token_index"] for item in result["tokens"] if item["valid"]} == {*range(8), 12, 13}
    first = result["tokens"][0]
    assert first["canvas_cell_edges"] == [0.0, 0.0, 28.0, 28.0]
    assert first["canvas_center"] == [13.5, 13.5]
    assert first["region_kind"] == "overview" and first["region_id"] == 0
    assert result["tokens"][12]["region_kind"] == "thumbnail" and result["tokens"][12]["region_id"] == 1
    for item in result["tokens"]:
        if not item["valid"]:
            assert item["region_kind"] is item["region_id"] is None
            assert all(item[field] is None for field in NATIVE_FIELDS)
    assert result["geometry_only_no_pixel_validation"] is True
    assert "nominal" in result["scope"] and "not exact resampler" in result["scope"]


def test_center_inside_thumbnail_is_insufficient_when_cell_crosses_header():
    metadata = scene()
    metadata["id_table"] = [tile(1, 0, [0, 56, 56, 112], [0, 56, 56, 68], [0, 0, 112, 56], [0, 68, 56, 112])]
    item = recipe_module.build_recipe(metadata, [1, 8, 8])["tokens"][8]
    assert item["canvas_center"] == [13.5, 69.5]
    assert 0 <= item["canvas_center"][0] < 56 and 68 <= item["canvas_center"][1] < 112
    assert item["valid"] is False and item["mask_reason"] == "partial_region"
    assert all(item[field] is None for field in NATIVE_FIELDS)


def test_cell_intersecting_overview_and_thumbnail_is_masked():
    item = recipe_module.build_recipe(scene(), [1, 6, 8])["tokens"][4]
    assert item["valid"] is False and item["mask_reason"] == "cross_region"
    assert all(item[field] is None for field in NATIVE_FIELDS)


def test_odd_canvas_resize_keeps_fractional_pixel_centers():
    result = recipe_module.build_recipe(odd_scene(), [1, 10, 8])
    first = result["tokens"][0]
    assert result["resized_size"] == [112, 140]
    assert first["canvas_cell_edges"] == pytest.approx([0, 0, Fraction(101, 4), Fraction(137, 5)])
    assert first["canvas_center"] == pytest.approx([Fraction(97, 8), Fraction(66, 5)])
    assert first["native_unit_center"] == pytest.approx([Fraction(97, 800), Fraction(11, 60)])


def test_rounded_thumbnail_axes_use_independent_inverse_scales_and_static_bounds():
    item = recipe_module.build_recipe(odd_scene(), [1, 20, 16])["tokens"][59]
    assert (item["merged_row"], item["merged_col"]) == (7, 3)
    assert item["valid"] and item["region_kind"] == "thumbnail"
    assert item["native_center"] == pytest.approx([Fraction(259, 16), Fraction(4807, 140)])
    assert item["native_unit_center"] == pytest.approx([Fraction(259, 1600), Fraction(4807, 10080)])
    assert item["native_cell_edges"] == pytest.approx([Fraction(1505, 184), Fraction(8973, 350),
                                                       Fraction(1159, 46), Fraction(7706, 175)])
    assert item["native_cell_span"] == pytest.approx([Fraction(3131, 184), Fraction(6439, 350)])
    assert item["native_unit_span"] == pytest.approx([Fraction(3131, 18400), Fraction(6439, 25200)])
    assert item["static_bounds_native_xy"] == pytest.approx([0, Fraction(11157, 700),
                                                             Fraction(12219, 368), Fraction(36913, 700)])
    assert item["static_bounds_native_unit_xy"] == pytest.approx([0, Fraction(11157, 50400),
                                                                  Fraction(12219, 36800), Fraction(36913, 50400)])


@pytest.mark.parametrize("native,grid,axis", [([1, 56], [1, 4, 2], 0), ([56, 1], [1, 2, 4], 1)])
def test_degenerate_native_axis_has_zero_unit_span_and_bounds(native, grid, axis):
    metadata = {"native_size": native, "canvas_size": native, "overview_bbox": [0, 0, *native],
                "overview_rgb_byteexact": True, "id_table": []}
    result = recipe_module.build_recipe(metadata, grid)
    assert result["token_count"] == 2
    for item in result["tokens"]:
        assert item["valid"]
        assert item["native_center"][axis] == item["native_unit_center"][axis] == item["native_unit_span"][axis] == 0.0
        assert item["static_bounds_native_xy"][axis] == item["static_bounds_native_xy"][axis + 2] == 0.0
        assert item["static_bounds_native_unit_xy"][axis] == item["static_bounds_native_unit_xy"][axis + 2] == 0.0


def test_oversampled_pixel_anchor_outside_domain_is_masked_without_correction():
    metadata = {"native_size": [2, 2], "canvas_size": [2, 2], "overview_bbox": [0, 0, 2, 2],
                "overview_rgb_byteexact": True, "id_table": []}
    result = recipe_module.build_recipe(metadata, [1, 8, 8])
    assert result["tokens"][0]["canvas_center"] == [-0.25, -0.25]
    assert result["tokens"][0]["mask_reason"] == "native_center_outside_pixel_domain"
    assert all(result["tokens"][0][field] is None for field in NATIVE_FIELDS)
    assert {item["token_index"] for item in result["tokens"] if item["valid"]} == {5, 6, 9, 10}


class OpaquePoison:
    def refused(self, *args, **kwargs):
        raise AssertionError("opaque label/GT/pixel field was inspected")
    __repr__ = __str__ = __iter__ = __deepcopy__ = __bool__ = refused


class WhitelistMapping(dict):
    """Fail on whole mapping copies/enumeration and nonwhitelisted key reads."""
    def __init__(self, visible, opaque):
        super().__init__({**visible, **opaque})
        self.allowed = frozenset(visible)
    def __getitem__(self, key):
        if key not in self.allowed:
            raise AssertionError(f"unapproved field read: {key}")
        return super().__getitem__(key)
    def refused(self, *args, **kwargs):
        raise AssertionError("whole mapping enumeration/copy is not input-only")
    __iter__ = items = keys = values = refused
    def get(self, key, default=None):
        return self[key] if key in self.allowed else self.refused()


def test_gt_pixel_and_label_poison_are_not_read_or_in_geometry_digest():
    clean = odd_scene()
    expected = recipe_module.build_recipe(clean, [1, 20, 16])
    entry = clean["id_table"][0]
    affine = WhitelistMapping(entry["crop_affine"], {"GT_affine_label": OpaquePoison()})
    entry = WhitelistMapping({**entry, "crop_affine": affine}, {"native_bbox": OpaquePoison(), "native_center": OpaquePoison(),
                                                               "candidate_target": OpaquePoison(), "text": OpaquePoison()})
    poisoned = WhitelistMapping({**clean, "id_table": [entry]}, {"actions": OpaquePoison(), "source_annotation": OpaquePoison(),
                                                               "GT": OpaquePoison(), "rgb_bytes": OpaquePoison(), "image_path": OpaquePoison()})
    assert recipe_module.build_recipe(poisoned, [1, 20, 16]) == expected


def test_tolerated_stored_affine_rounding_is_audited_but_not_used_for_features():
    metadata = odd_scene()
    original = recipe_module.build_recipe(metadata, [1, 20, 16])
    metadata["id_table"][0]["crop_affine"]["translate_x"] += 1e-11
    perturbed = recipe_module.build_recipe(metadata, [1, 20, 16])
    assert perturbed["geometry_input_sha256"] != original["geometry_input_sha256"]
    assert perturbed["tokens"] == original["tokens"]


@pytest.mark.parametrize("grid,options", [([1, 8], {}), ([1, 8, 8, 1], {}), ([0, 8, 8], {}),
                                         ([2, 8, 8], {}), ([True, 8, 8], {}), ([1, 0, 8], {}),
                                         ([1, 7, 8], {}), ([1, 8, -2], {}), ([1, 8, 8.0], {}),
                                         ([1, 8, 8], {"merge_size": 0}), ([1, 8, 8], {"merge_size": True}),
                                         ([1, 8, 8], {"merge_size": 3}), ([1, 8, 8], {"patch_size": 0}),
                                         ([1, 8, 8], {"patch_size": False})])
def test_invalid_grid_patch_merge_and_temporal_contract_are_rejected(grid, options):
    with pytest.raises(ValueError):
        recipe_module.build_recipe(scene(), grid, **options)


@pytest.mark.parametrize("problem", ["native_bool", "canvas_float", "overview_size", "not_byteexact",
                                     "crop_outside", "thumbnail_outside", "header_overlap", "affine_nan",
                                     "affine_inf", "affine_no_halfpixel", "affine_bool"])
def test_malformed_geometry_and_saved_affine_are_rejected(problem):
    metadata = scene()
    entry = metadata["id_table"][0]
    if problem == "native_bool":
        metadata["native_size"][0] = True
    elif problem == "canvas_float":
        metadata["canvas_size"][0] = 112.0
    elif problem == "overview_size":
        metadata["overview_bbox"][2] = 111
    elif problem == "not_byteexact":
        metadata["overview_rgb_byteexact"] = False
    elif problem == "crop_outside":
        entry["crop_bbox"][2] = 113
    elif problem == "thumbnail_outside":
        entry["thumbnail_bbox"][2] = 57
    elif problem == "header_overlap":
        entry["header_bbox"][3] = 71
    elif problem == "affine_nan":
        entry["crop_affine"]["scale_x"] = float("nan")
    elif problem == "affine_inf":
        entry["crop_affine"]["translate_y"] = float("inf")
    elif problem == "affine_no_halfpixel":
        entry["crop_affine"]["translate_x"] = 0.0
    elif problem == "affine_bool":
        entry["crop_affine"]["scale_y"] = True
    with pytest.raises(ValueError):
        recipe_module.build_recipe(metadata, [1, 8, 8])


@pytest.mark.parametrize("problem", ["id_order", "duplicate_id", "duplicate_tile", "id_bool", "overlapping_tiles"])
def test_table_order_uniqueness_and_panel_ambiguity_are_rejected(problem):
    metadata = scene()
    metadata["id_table"].append(tile(2, 1, [56, 56, 112, 112], [56, 56, 112, 70], [0, 0, 112, 56], [56, 70, 112, 112]))
    if problem == "id_order":
        metadata["id_table"].reverse()
    elif problem == "duplicate_id":
        metadata["id_table"][1]["id"] = 1
    elif problem == "duplicate_tile":
        metadata["id_table"][1]["tile_index"] = 0
    elif problem == "id_bool":
        metadata["id_table"][0]["id"] = True
    elif problem == "overlapping_tiles":
        metadata["id_table"][1] = {**copy.deepcopy(metadata["id_table"][0]), "id": 2, "tile_index": 1}
    with pytest.raises(ValueError):
        recipe_module.build_recipe(metadata, [1, 8, 8])


def test_whitelisted_grid_change_changes_input_digest():
    first = recipe_module.build_recipe(scene(), [1, 8, 8])
    second = recipe_module.build_recipe(scene(), [1, 10, 8])
    assert first["geometry_input_sha256"] != second["geometry_input_sha256"]
