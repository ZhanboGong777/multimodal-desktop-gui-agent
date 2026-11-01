"""Gallery geometry/pixel oracles using synthetic RGB and a pinned test font."""
from __future__ import annotations

import copy
import hashlib
import io
from collections.abc import Mapping
from pathlib import Path

import pytest
import test_week5_visual_pointer as fixtures
from PIL import Image, ImageFont

from gui_agent.finetuning import candidate_gallery as gallery


@pytest.fixture
def font(tmp_path, monkeypatch):
    fonts = {size: ImageFont.load_default(size=size) for size in (72, 154)}
    path = tmp_path / "synthetic_font.ttf"
    path.write_bytes(b"SYNTHETIC_FONT_FIXTURE_NO_FALLBACK")
    calls = []

    def load(data, *, size, layout_engine):
        assert data.getvalue() == path.read_bytes() and size in fonts and layout_engine == ImageFont.Layout.BASIC
        calls.append(size)
        return fonts[size]

    monkeypatch.setattr(gallery.ImageFont, "truetype", load)
    return {"font_path": path, "font_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, calls


def sample(tmp_path, n=64, size=(1024, 768)):
    return fixtures.make_sample(tmp_path, size=size, candidates=[fixtures.candidate(f"raw{i:03}",
                                8 + i % 8 * 125, 8 + i // 8 * 90, width=40, height=35)
                                for i in range(n)])


class Secrets(Mapping):
    def __init__(self, values, forbidden):
        self.values, self.forbidden = values, forbidden

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"secret value read: {key}")
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def render(row, font):
    return gallery.render_candidate_gallery(row, **font[0])


def test_all64_native_RGB_byteexact_independent_crops_and_affine(tmp_path, font):
    row = sample(tmp_path)
    original_bytes = Path(row["image_path"]).read_bytes()
    result = render(row, font)
    metadata = result["metadata"]
    canvas = Image.open(io.BytesIO(result["png_bytes"])).convert("RGB")
    original = Image.open(row["image_path"]).convert("RGB")
    assert canvas.size == (1024, 1920) and metadata["overview_bbox"] == [0, 0, 1024, 768]
    assert canvas.crop((0, 0, 1024, 768)).tobytes() == original.tobytes()
    assert Path(row["image_path"]).read_bytes() == original_bytes
    assert [item["id"] for item in metadata["id_table"]] == list(range(1, 65))
    assert set(metadata["tile_display_order"]) == set(range(1, 65))
    assert metadata["tile_display_order"] != list(range(1, 65))
    assert sorted(item["tile_index"] for item in metadata["id_table"]) == list(range(64))
    for item, selected in zip(metadata["id_table"], gallery._visible(row)["selected"], strict=True):
        assert item["candidate_id"] == selected["candidate_id"] and item["native_bbox"] == selected["bbox"]
        assert item["native_center"] == list(gallery.center(selected["bbox"]))
        tile, header, glyph, thumb = (item[key] for key in ("tile_bbox", "header_bbox", "id_glyph_bbox", "thumbnail_bbox"))
        assert tile[2] - tile[0] == 128 and tile[3] - tile[1] == 144 and header[3] - header[1] == 74
        assert header[0] + 7 <= glyph[0] < glyph[2] <= header[2] - 7
        assert header[1] + 7 <= glyph[1] < glyph[3] <= header[3] - 7
        crop = original.crop(item["crop_bbox"])
        expected = crop.resize((thumb[2] - thumb[0], thumb[3] - thumb[1]), Image.Resampling.BICUBIC)
        assert canvas.crop(thumb).tobytes() == expected.tobytes()
        affine = item["crop_affine"]
        mapped_x = item["crop_bbox"][0] * affine["scale_x"] + affine["translate_x"]
        mapped_y = item["crop_bbox"][1] * affine["scale_y"] + affine["translate_y"]
        assert mapped_x == pytest.approx(thumb[0] + (affine["scale_x"] - 1) / 2)
        assert mapped_y == pytest.approx(thumb[1] + (affine["scale_y"] - 1) / 2)
    assert metadata["version"] == gallery.VERSION and "model_image_path" not in metadata


def test_deterministic_aliases_ignore_all_sample_and_candidate_secrets(tmp_path, font):
    row = sample(tmp_path, 8)
    first = render(row, font)
    secret_names = {"actions", "candidate_target", "review", "sample_id", "chosen_printed_id"}
    poisoned = {**row, **dict.fromkeys(secret_names, object())}
    second = render(Secrets(poisoned, secret_names), font)
    assert first == second
    rgb = Image.open(row["image_path"]).convert("RGB")
    selected = gallery._visible(row)["selected"]
    candidates = [Secrets({**c, "actions": object(), "review": object(), "text": object()},
                          {"actions", "review", "text"}) for c in selected]
    pure = gallery.compose_gallery_rgb(rgb.tobytes(), rgb.size, candidates,
                    font_bytes=font[0]["font_path"].read_bytes(), font_sha256=font[0]["font_sha256"])
    assert pure["rgb_bytes"] == Image.open(io.BytesIO(first["png_bytes"])).tobytes()


def test_per_input_shuffle_uses_only_native_RGB_ordered_geometry_and_rawIDs(tmp_path, font):
    row = sample(tmp_path, 64)
    rgb = Image.open(row["image_path"]).convert("RGB")
    selected = gallery._visible(row)["selected"]
    kwargs = {"font_bytes": font[0]["font_path"].read_bytes(), "font_sha256": font[0]["font_sha256"]}
    first = gallery.compose_gallery_rgb(rgb.tobytes(), rgb.size, selected, **kwargs)
    assert first == gallery.compose_gallery_rgb(rgb.tobytes(), rgb.size, selected, **kwargs)
    altered = copy.deepcopy(selected)
    altered[0]["bbox"]["left"] += 1
    geometry = gallery.compose_gallery_rgb(rgb.tobytes(), rgb.size, altered, **kwargs)
    view = gallery.compose_gallery_rgb(rgb.tobytes(), rgb.size, list(reversed(selected)), **kwargs)
    rgb_mutation = bytearray(rgb.tobytes())
    rgb_mutation[0] ^= 1
    pixels = gallery.compose_gallery_rgb(bytes(rgb_mutation), rgb.size, selected, **kwargs)
    first_order = first["metadata"]["tile_display_order"]
    for result in (geometry, view, pixels):
        assert result["metadata"]["tile_display_order"] != first_order
        assert set(result["metadata"]["tile_display_order"]) == set(range(1, 65))
        proof = result["metadata"]["shuffle_seed_derivation"]
        assert proof["input_sha256"] == gallery.value_sha256(proof["input"])
        assert set(proof["input"]) == {"base_seed", "native_rgb_sha256", "ordered_selected_geometry"}


@pytest.mark.parametrize("problem", ["zero", "over64", "bbox", "fontSHA", "dimensions"])
def test_pure_whole_gallery_refusals(tmp_path, font, problem):
    row = sample(tmp_path, 1)
    rgb = Image.open(row["image_path"]).convert("RGB")
    candidates = copy.deepcopy(row["candidate_snapshot"]["candidates"])
    if problem == "zero":
        candidates = []
    elif problem == "over64":
        candidates *= 65
    elif problem == "bbox":
        candidates[0]["bbox"]["right"] = 9999
    with pytest.raises(ValueError):
        gallery.compose_gallery_rgb(rgb.tobytes(), (1, 1) if problem == "dimensions" else rgb.size, candidates,
                      font_bytes=font[0]["font_path"].read_bytes(),
                      font_sha256="0" * 64 if problem == "fontSHA" else font[0]["font_sha256"])


def test_snapshot_context_stale_binding_and_forbidden_snapshot_fail_before_secret_value(tmp_path, font):
    row = sample(tmp_path, 2)
    changed = {**row, "actor_step": "Different visible step"}
    with pytest.raises(ValueError, match="binding differs"):
        render(changed, font)
    poisoned = copy.deepcopy(row)
    poisoned["candidate_snapshot"] = Secrets({**poisoned["candidate_snapshot"], "actions": object()}, {"actions"})
    with pytest.raises(ValueError, match="forbidden label"):
        render(poisoned, font)


def test_numbered_raw_parent_required_and_all_IDs_retained(tmp_path, font):
    row = sample(tmp_path, 8)
    raw = copy.deepcopy(row["candidate_snapshot"])
    from gui_agent.finetuning.pointer_numbering import numbered_view

    row["candidate_snapshot"] = numbered_view(row, view_index=0)["candidate_snapshot"]
    fixtures.bind(row)
    with pytest.raises(ValueError, match="bound original raw snapshot"):
        render(row, font)
    result = gallery.render_candidate_gallery(row, **font[0], raw_snapshot=raw)
    assert [r["candidate_id"] for r in result["metadata"]["id_table"]] == [f"v{i:03}" for i in range(1, 9)]
    assert len({r["raw_candidate_id"] for r in result["metadata"]["id_table"]}) == 8
    bad = copy.deepcopy(raw)
    bad["candidates"][0]["text"] = "mutated"
    with pytest.raises(ValueError, match="raw OCR protection snapshot binding"):
        gallery.render_candidate_gallery(row, **font[0], raw_snapshot=bad)


def test_source_mutation_during_font_load_refused(tmp_path, font, monkeypatch):
    row = sample(tmp_path, 2)
    load = gallery.ImageFont.truetype

    def mutate(*args, **kwargs):
        result = load(*args, **kwargs)
        Path(row["image_path"]).write_bytes(b"mutated")
        return result

    monkeypatch.setattr(gallery.ImageFont, "truetype", mutate)
    with pytest.raises(ValueError, match="original/font changed"):
        render(row, font)


@pytest.mark.parametrize("field,value", [("input_image_protocol", "visual_pointer_v2"),
                                      ("visual_prompt_version", "week5.visual-pointer-prompt.compact.v3")])
def test_gallery_explicit_protocol_and_prompt_mixing_refused(tmp_path, font, field, value):
    row = sample(tmp_path, 2)
    row[field] = value
    with pytest.raises(ValueError, match="mixed"):
        render(row, font)


def test_gallery_descriptor_metadata_prompt_and_native_canvas_identity(tmp_path, font):
    row = sample(tmp_path, 2)
    row.update(input_image_protocol=gallery.PROTOCOL, visual_prompt_version=gallery.GALLERY_PROMPT_VERSION)
    result = render(row, font)
    metadata = result["metadata"]
    assert metadata["status"] == "rendered_not_training_admission"
    assert metadata["input_image_protocol"] == gallery.PROTOCOL and metadata["visual_prompt_version"] == gallery.GALLERY_PROMPT_VERSION
    assert metadata["native_size"] == [1024, 768] and metadata["canvas_size"] == [1024, 912]
    assert metadata["model_png_sha256"] == metadata["gallery_png_sha256"] == hashlib.sha256(result["png_bytes"]).hexdigest()
    assert metadata["model_native_rgb_sha256"] == metadata["gallery_native_rgb_sha256"]
    assert metadata["prompt_sha256"] == hashlib.sha256(result["prompt"].encode()).hexdigest()
    assert "UI above; shuffled candidate crop proposals below." in result["prompt"]


@pytest.mark.parametrize("width", [883, 1024])
def test_responsive_base_widths_keep_existing_raster_parameters_and_bytes(tmp_path, font, monkeypatch, width):
    row = fixtures.make_sample(tmp_path, size=(width, 768), candidates=[fixtures.candidate("A", 100, 120)])
    first = render(row, font)
    resolved = gallery.resolve_gallery_geometry(width)
    assert tuple(resolved[k] for k in ("tile_width", "tile_height", "header_height", "thumbnail_height", "font_size",
                                      "glyph_margin_pixels", "crop_padding_pixels")) == (width // 8, 144, 74, 70, 72, 7, 8)
    # Force the former fixed profile independently of the responsive resolver.
    baseline = {**resolved, "tile_height": 144, "header_height": 74, "thumbnail_height": 70,
                "font_size": 72, "glyph_margin_pixels": 7, "crop_padding_pixels": 8}
    monkeypatch.setattr(gallery, "resolve_gallery_geometry", lambda value: baseline)
    assert render(row, font)["png_bytes"] == first["png_bytes"]


def test_responsive_2190_keeps_all64_native_RGB_and_scales_glyphs_crops(tmp_path, font):
    row = sample(tmp_path, size=(2190, 768))
    result = render(row, font)
    canvas = Image.open(io.BytesIO(result["png_bytes"])).convert("RGB")
    original = Image.open(row["image_path"]).convert("RGB")
    resolved = result["metadata"]["resolved_geometry"]
    assert (resolved["tile_width"], resolved["tile_height"], resolved["header_height"],
            resolved["thumbnail_height"], resolved["font_size"], resolved["glyph_margin_pixels"],
            resolved["crop_padding_pixels"]) == (273, 308, 158, 150, 154, 15, 8)
    assert canvas.size == (2190, 768 + 8 * 308)
    assert canvas.crop((0, 0, 2190, 768)).tobytes() == original.tobytes()
    assert [r["id"] for r in result["metadata"]["id_table"]] == list(range(1, 65))
    for item in result["metadata"]["id_table"]:
        header, glyph, tile, thumb = (item[k] for k in ("header_bbox", "id_glyph_bbox", "tile_bbox", "thumbnail_bbox"))
        assert header[0] + 15 <= glyph[0] < glyph[2] <= header[2] - 15
        assert header[1] + 15 <= glyph[1] < glyph[3] <= header[3] - 15
        assert header[3] <= thumb[1] < thumb[3] <= tile[3]
        crop = original.crop(item["crop_bbox"])
        assert canvas.crop(thumb).tobytes() == crop.resize((thumb[2] - thumb[0], thumb[3] - thumb[1]), Image.Resampling.BICUBIC).tobytes()
    assert font[1] == [154]


def test_responsive_width_quantization_is_deterministic_and_conserves_tile_height():
    for width in (1024, 1025, 1830, 2190, 2191):
        resolved = gallery.resolve_gallery_geometry(width)
        assert resolved == gallery.resolve_gallery_geometry(width)
        assert resolved["thumbnail_height"] + resolved["header_height"] == resolved["tile_height"]
        assert resolved["tile_width"] == width // 8 and resolved["crop_padding_pixels"] == 8
