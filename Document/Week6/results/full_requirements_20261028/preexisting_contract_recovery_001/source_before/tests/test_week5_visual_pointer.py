"""Synthetic visual-layout guards; no actual sources, processors, models or GUI."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFont

from gui_agent.finetuning import pointer_protocol as pp
from gui_agent.finetuning import visual_pointer as vp
from gui_agent.finetuning.candidate_grounding import center, snapshot_hash
from gui_agent.finetuning.evaluation import canonical_json, file_sha256
from gui_agent.finetuning.pointer_numbering import numbered_view


def bind(sample):
    sample["candidate_selection_evidence"] = pp.selection_binding(
        sample["candidate_snapshot"], instruction=sample["instruction"], actor_step=sample["actor_step"])
    return sample


def candidate(identifier, x, y, *, source="contour", width=40, height=40):
    return {"candidate_id": identifier, "source": source,
            "role": "text_region" if source == "ocr" else "contour_region",
            "text": "Visible Alpha" if source == "ocr" else "", "confidence": .8,
            "bbox": {"left": x, "top": y, "right": x + width, "bottom": y + height}}


def make_sample(tmp_path, *, candidates=None, size=(1024, 768), policy="query_relations_v3"):
    path = tmp_path / "synthetic_native.png"
    image = Image.new("RGB", size, (89, 113, 137))
    ImageDraw.Draw(image).rectangle((0, 0, 15, 15), fill=(10, 20, 30))
    image.save(path)
    if candidates is None:
        candidates = [candidate(f"raw{i:03}", 80 + i % 4 * 200, 80 + i // 4 * 220,
                                source="ocr" if i % 2 else "contour") for i in range(8)]
    sample = {"image_path": str(path), "image_width": size[0], "image_height": size[1],
              "image_file_sha256": file_sha256(path),
              "image_sha256": hashlib.sha256(f"{size[0]}x{size[1]}:".encode() + image.tobytes()).hexdigest(),
              "instruction": "Complete the requested Alpha field.", "actor_step": "Click Alpha.",
              "candidate_snapshot": {"image_width": size[0], "image_height": size[1],
                                     "image_file_sha256": file_sha256(path),
                                     "compact_policy": policy, "candidates": candidates},
              "sample_id": "SYNTHETIC_ONLY"}
    return bind(sample)


@pytest.fixture
def sample(tmp_path):
    return make_sample(tmp_path)


@pytest.fixture
def font_asset(tmp_path, monkeypatch):
    # An embedded synthetic font is injected only in tests. Production must load
    # the exact registered font bytes and has no fallback.
    font = ImageFont.load_default(size=27)
    path = tmp_path / "synthetic_pinned_font.ttf"
    path.write_bytes(b"synthetic font fixture; never a production fallback")
    digest = file_sha256(path)
    calls = []

    def load(data, *, size, layout_engine):
        calls.append((data.getvalue(), size, layout_engine))
        assert data.getvalue() == path.read_bytes()
        assert size == 27 and layout_engine == ImageFont.Layout.BASIC
        return font

    monkeypatch.setattr(vp.ImageFont, "truetype", load)
    return {"font_path": path, "font_sha256": digest}, calls


def render(sample, font_asset, **kwargs):
    return vp.render_visual_pointer(sample, **font_asset[0], **kwargs)


def intersects(a, b):
    return a["left"] < b["right"] and b["left"] < a["right"] and a["top"] < b["bottom"] and b["top"] < a["bottom"]


def assert_all_visible(result, sample):
    metadata = result["metadata"]
    image = Image.open(io.BytesIO(result["png_bytes"])).convert("RGB")
    assert image.size == (sample["image_width"], sample["image_height"])
    selected = pp.select_candidates(sample["candidate_snapshot"], instruction=sample["instruction"], actor_step=sample["actor_step"])
    table = metadata["id_table"]
    assert [row["id"] for row in table] == list(range(1, len(selected) + 1))
    assert len(table) == metadata["selected_candidate_count"] == len(selected)
    for row, source in zip(table, selected, strict=True):
        assert row["candidate_id"] == source["candidate_id"]
        assert row["bbox"] == source["bbox"]
        assert row["native_center"] == list(center(source["bbox"]))
        assert row["leader"][-1] == row["native_center"]
        assert image.getpixel(tuple(row["native_center"])) == (0, 255, 255)
        b = row["badge_bbox"]
        pixels = list(image.crop(tuple(b[k] for k in ("left", "top", "right", "bottom"))).get_flattened_data())
        assert (0, 0, 0) in pixels and (255, 255, 255) in pixels
        assert 0 < row["placement_attempts"] <= 264
        for other in table:
            point = other["native_center"]
            assert not (b["left"] <= point[0] < b["right"] and b["top"] <= point[1] < b["bottom"])
            if other is not row:
                expanded = {"left": b["left"] - 2, "top": b["top"] - 2,
                            "right": b["right"] + 2, "bottom": b["bottom"] + 2}
                assert not intersects(expanded, other["badge_bbox"])
    paint = metadata["paint"]
    assert 0 < paint["changed_pixels"] <= paint["painted_pixels"] <= paint["canvas_pixels"] * .08
    assert paint["raw_ocr_painted_pixels"] <= paint["raw_ocr_union_pixels"] * .02
    assert paint["raw_ocr_changed_pixels"] <= paint["raw_ocr_painted_pixels"]
    assert paint["badge_raw_ocr_overlap_pixels"] == paint["badge_candidate_center_overlap_pixels"] == 0
    assert paint["badge_leader_overlap_pixels"] == 0


def test_every_id_visible_native_mapping_masks_hashes_and_source_unchanged(sample, font_asset):
    before = copy.deepcopy(sample)
    native_before = Path(sample["image_path"]).read_bytes()
    result = render(sample, font_asset)
    assert_all_visible(result, sample)
    m = result["metadata"]
    assert m["original_image_file_sha256"] == sample["image_file_sha256"]
    assert m["original_native_rgb_sha256"] == sample["image_sha256"]
    assert m["model_png_sha256"] == hashlib.sha256(result["png_bytes"]).hexdigest()
    assert m["model_png_sha256"] != m["original_image_file_sha256"]
    assert m["model_native_rgb_sha256"] != m["original_native_rgb_sha256"]
    assert m["prompt_sha256"] == hashlib.sha256(result["prompt"].encode()).hexdigest()
    assert m["raw_protection_snapshot_sha256"] == snapshot_hash(sample["candidate_snapshot"])
    assert m["font_sha256"] == font_asset[0]["font_sha256"]
    assert m["pillow_version"] and m["config"] == vp.VisualPointerConfig().__dict__
    assert m["search_order"] == list(vp.NEAR_ORDER) and m["search_ring_step"] == 8
    assert "not UI controls" in result["prompt"] and "native center plus dx/dy" in result["prompt"]
    assert m["processor_or_token_measurement_performed"] is False
    assert m["training_admission"] is False and m["model_quality_proven"] is False
    assert sample == before and Path(sample["image_path"]).read_bytes() == native_before


class VisibleOnly(dict):
    forbidden = frozenset({"sample_id", "actions", "candidate_target", "candidate_label_evidence", "review", "targets"})

    def __getitem__(self, key):
        assert key not in self.forbidden, f"renderer leaked label-side field {key}"
        return super().__getitem__(key)

    def get(self, key, default=None):
        assert key not in self.forbidden, f"renderer leaked label-side field {key}"
        return super().get(key, default)


def test_deterministic_and_secret_targets_reviews_sample_id_independent(sample, font_asset):
    first = render(sample, font_asset)
    hidden = VisibleOnly(copy.deepcopy(sample))
    hidden.update(sample_id=object(), actions=object(), candidate_target=object(),
                  candidate_label_evidence=object(), review=object(), targets=object())
    assert render(hidden, font_asset) == first
    assert vp.build_visual_pointer_messages(hidden)[0]["content"][1]["text"] == first["prompt"]
    assert render(sample, font_asset) == first


def derived_sample(parent, view_index):
    derived = copy.deepcopy(parent)
    derived["candidate_snapshot"] = numbered_view(parent, view_index=view_index)["candidate_snapshot"]
    return bind(derived)


def test_relabel_deterministic_original_centers_unchanged_exact_selected_table(sample, font_asset):
    raw = sample["candidate_snapshot"]
    zero, shuffled = derived_sample(sample, 0), derived_sample(sample, 2)
    first = render(zero, font_asset, raw_snapshot=raw)
    second = render(shuffled, font_asset, raw_snapshot=raw)
    assert render(shuffled, font_asset, raw_snapshot=raw) == second
    assert_all_visible(first, zero)
    assert_all_visible(second, shuffled)
    maps = [{r["raw_candidate_id"]: (r["native_center"], r["bbox"]) for r in result["metadata"]["id_table"]}
            for result in (first, second)]
    assert maps[0] == maps[1]
    assert [r["raw_candidate_id"] for r in first["metadata"]["id_table"]] != [r["raw_candidate_id"] for r in second["metadata"]["id_table"]]
    assert first["png_bytes"] != second["png_bytes"]
    assert second["metadata"]["raw_protection_snapshot_sha256"] == snapshot_hash(raw)


def test_more_than_64_raw_candidates_never_number_raw_pool(tmp_path, font_asset):
    candidates = [candidate(f"pool{i:03}", 32 + i % 12 * 80, 32 + i // 12 * 70,
                            width=20, height=16) for i in range(96)]
    parent = make_sample(tmp_path, candidates=candidates)
    expected = pp.select_candidates(parent["candidate_snapshot"], instruction=parent["instruction"], actor_step=parent["actor_step"])
    assert len(expected) == 64
    result = render(parent, font_asset)
    assert_all_visible(result, parent)
    assert [r["candidate_id"] for r in result["metadata"]["id_table"]] == [c["candidate_id"] for c in expected]
    derived = derived_sample(parent, 1)
    numbered = render(derived, font_asset, raw_snapshot=parent["candidate_snapshot"])
    assert len(numbered["metadata"]["id_table"]) == 64
    assert {r["raw_candidate_id"] for r in numbered["metadata"]["id_table"]} == {c["candidate_id"] for c in expected}


def test_numbered_protects_ocr_not_in_selected_snapshot(tmp_path, font_asset):
    candidates = [candidate(f"ocr{i:03}", 30 + i % 12 * 160, 30 + i // 12 * 180,
                            source="ocr", width=60, height=80) for i in range(96)]
    parent = make_sample(tmp_path, candidates=candidates, size=(2048, 1536))
    derived = derived_sample(parent, 1)
    assert len(derived["candidate_snapshot"]["candidates"]) == 64
    result = render(derived, font_asset, raw_snapshot=parent["candidate_snapshot"])
    m = result["metadata"]
    assert m["paint"]["raw_ocr_candidate_count"] == 96
    assert m["paint"]["raw_ocr_union_pixels"] == 96 * 60 * 80
    selected_ids = {c["original_id"] for c in derived["candidate_snapshot"]["candidates"]}
    unseen_ocr = [c for c in candidates if c["candidate_id"] not in selected_ids]
    assert len(unseen_ocr) == 32
    for row in m["id_table"]:
        for raw in unseen_ocr:
            assert not intersects(row["badge_bbox"], raw["bbox"])
    assert_all_visible(result, derived)


@pytest.mark.parametrize("change", ["missing", "hash", "image", "geometry", "context", "whole_set"])
def test_numbered_raw_missing_or_mutated_binding_fatal(sample, font_asset, change):
    derived = derived_sample(sample, 1)
    raw = copy.deepcopy(sample["candidate_snapshot"])
    if change == "missing":
        raw = None
    elif change == "hash":
        raw["candidates"][0]["text"] = "changed"
    elif change == "image":
        raw["image_file_sha256"] = "0" * 64
    elif change == "geometry":
        derived["candidate_snapshot"]["candidates"][0]["bbox"]["left"] += 1
        bind(derived)
    elif change == "context":
        derived["actor_step"] += " changed"
        bind(derived)
    else:
        derived["candidate_snapshot"]["candidates"].pop()
        bind(derived)
    with pytest.raises(ValueError) as exc:
        render(derived, font_asset, raw_snapshot=raw)
    assert not isinstance(exc.value, vp.VisualLayoutError)


@pytest.mark.parametrize("policy", ["spatial_v1", "query_relations_v3", "control_balanced_v2"])
@pytest.mark.parametrize("field", ["actor_step", "instruction", "candidate_selection_evidence", "snapshot"])
def test_all_policies_require_current_explicit_selection_binding(sample, font_asset, policy, field):
    sample["candidate_snapshot"]["compact_policy"] = policy
    bind(sample)
    if field == "snapshot":
        sample["candidate_snapshot"]["candidates"][0]["confidence"] = .1
    elif field == "candidate_selection_evidence":
        del sample[field]
    else:
        sample[field] += " altered"
    with pytest.raises(ValueError) as exc:
        render(sample, font_asset)
    assert not isinstance(exc.value, vp.VisualLayoutError)


@pytest.mark.parametrize("change", ["snapshot_file", "image_bytes", "native_rgb", "native_file", "dimensions", "empty", "invalid_box", "invalid_confidence", "unknown_policy"])
def test_stale_or_invalid_visible_inputs_fatal(sample, font_asset, change):
    if change == "snapshot_file":
        sample["candidate_snapshot"]["image_file_sha256"] = "0" * 64
        bind(sample)
    elif change == "image_bytes":
        Image.new("RGB", (1024, 768), "red").save(sample["image_path"])
    elif change == "native_rgb":
        sample["image_sha256"] = "0" * 64
    elif change == "native_file":
        sample["image_file_sha256"] = "0" * 64
    elif change == "dimensions":
        sample["image_width"] = 1000
    elif change == "empty":
        sample["candidate_snapshot"]["candidates"] = []
        bind(sample)
    elif change == "invalid_box":
        sample["candidate_snapshot"]["candidates"][0]["bbox"]["right"] = 2000
    elif change == "invalid_confidence":
        sample["candidate_snapshot"]["candidates"][0]["confidence"] = float("nan")
    else:
        sample["candidate_snapshot"]["compact_policy"] = "unsupported"
    with pytest.raises(ValueError) as exc:
        render(sample, font_asset)
    assert not isinstance(exc.value, vp.VisualLayoutError)


@pytest.mark.parametrize("config", [{"font_size": True}, {"font_size": 26}, {"badge_padding": 0},
                                     {"label_gap": 1}, {"leader_width": 2}, {"ring_step": 16},
                                     {"max_radius": 257}, {"max_paint_fraction": .081},
                                     {"max_raw_ocr_paint_fraction": .021}, {"max_paint_fraction": True},
                                     {"max_paint_fraction": float("nan")}, {"max_paint_fraction": "0.08"},
                                     {"max_raw_ocr_paint_fraction": -1}, {"unknown": 1}, "bad"])
def test_fixed_config_types_and_budgets_cannot_be_loosened(sample, font_asset, config):
    with pytest.raises((ValueError, TypeError)) as exc:
        render(sample, font_asset, config=config)
    assert not isinstance(exc.value, vp.VisualLayoutError)


def test_zero_budget_refuses_whole_layout_no_partial_or_raw_fallback(sample, font_asset):
    before = Path(sample["image_path"]).read_bytes()
    with pytest.raises(vp.VisualLayoutError, match="mask budget exceeded.*entire layout"):
        render(sample, font_asset, config={"max_paint_fraction": 0})
    assert Path(sample["image_path"]).read_bytes() == before


def test_crowded_all_ocr_layout_refuses_whole_image(tmp_path, font_asset):
    sample = make_sample(tmp_path, size=(128, 128), candidates=[candidate("full", 0, 0, source="ocr", width=128, height=128)])
    with pytest.raises(vp.VisualLayoutError, match="no feasible all-ID visual layout.*entire layout"):
        render(sample, font_asset)


def test_native_edges_placed_inside_canvas_without_resizing(tmp_path, font_asset):
    candidates = [candidate("left", 0, 0, width=2, height=2),
                  candidate("right", 510, 382, width=2, height=2)]
    sample = make_sample(tmp_path, candidates=candidates, size=(512, 384))
    result = render(sample, font_asset)
    assert_all_visible(result, sample)
    for row in result["metadata"]["id_table"]:
        b = row["badge_bbox"]
        assert 0 <= b["left"] < b["right"] <= 512 and 0 <= b["top"] < b["bottom"] <= 384


@pytest.mark.parametrize("change", ["font_hash", "invalid_hash", "missing", "unloadable"])
def test_font_strict_file_pin_and_no_fallback(sample, font_asset, monkeypatch, change):
    kwargs = dict(font_asset[0])
    if change == "font_hash":
        kwargs["font_sha256"] = "0" * 64
    elif change == "invalid_hash":
        kwargs["font_sha256"] = "bad"
    elif change == "missing":
        kwargs["font_path"] = Path(kwargs["font_path"]).with_name("missing.ttf")
    else:
        def fail(*args, **kwargs):
            raise OSError("synthetic invalid font bytes")
        monkeypatch.setattr(vp.ImageFont, "truetype", fail)
    with pytest.raises((ValueError, FileNotFoundError)) as exc:
        vp.render_visual_pointer(sample, **kwargs)
    assert not isinstance(exc.value, vp.VisualLayoutError)
    assert font_asset[1] == []


@pytest.mark.parametrize("change", ["native", "font", "context", "spatial_binding", "raw"])
@pytest.mark.parametrize("layout_failure", [False, True])
def test_post_render_mutations_fatal_even_if_layout_refuses(sample, font_asset, monkeypatch, change, layout_failure):
    parent_raw = copy.deepcopy(sample["candidate_snapshot"])
    if change == "raw":
        sample = derived_sample(sample, 1)
    elif change == "spatial_binding":
        sample["candidate_snapshot"]["compact_policy"] = "spatial_v1"
        bind(sample)
    original_layout = vp._layout

    def changed_layout(*args, **kwargs):
        result = None if layout_failure else original_layout(*args, **kwargs)
        if change == "native":
            Image.new("RGB", (1024, 768), "red").save(sample["image_path"])
        elif change == "font":
            Path(font_asset[0]["font_path"]).write_bytes(b"changed font")
        elif change == "context":
            sample["actor_step"] += " changed"
        elif change == "spatial_binding":
            sample["candidate_selection_evidence"]["context_sha256"] = "0" * 64
        else:
            parent_raw["candidates"][0]["text"] = "changed raw"
        if layout_failure:
            raise vp.VisualLayoutError("no feasible all-ID visual layout; synthetic refusal")
        return result

    monkeypatch.setattr(vp, "_layout", changed_layout)
    with pytest.raises(ValueError) as exc:
        render(sample, font_asset, raw_snapshot=parent_raw if change == "raw" else None)
    assert not isinstance(exc.value, vp.VisualLayoutError)


def test_train_and_inference_share_visual_prefix_original_complete_json_target(sample, font_asset, tmp_path):
    target = [{"action_type": "type_text", "text": "secret exact text \n中文"},
              {"action_type": "hotkey", "keys": ["ctrl", "a"]},
              {"action_type": "wait", "duration": .25}]
    sample["candidate_target"] = copy.deepcopy(target)
    sample["actions"] = copy.deepcopy(target)
    review = tmp_path / "synthetic_label_side_review.json"
    review.write_text(canonical_json({"schema": "week5.pointer_semantic_review.v1", "status": "closed", "reviews": []}), encoding="utf8")
    sample["candidate_label_evidence"] = {"review_path": str(review), "review_sha256": file_sha256(review)}
    inference = vp.build_visual_pointer_messages(sample)
    training = vp.build_visual_pointer_messages(sample, include_target=True)
    assert inference == training[:1]
    assert inference[0]["content"][1]["text"] == render(sample, font_asset)["prompt"]
    assert "secret exact text" not in inference[0]["content"][1]["text"]
    assert training[-1] == {"role": "assistant", "content": canonical_json(target)}
    assert json.loads(training[-1]["content"]) == target == sample["actions"]
    sample["candidate_target"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="preserve every native field"):
        vp.build_visual_pointer_messages(sample, include_target=True)
    with pytest.raises(TypeError):
        vp.build_visual_pointer_messages(sample, include_target=1)


def test_messages_detect_context_mutation_after_old_builder(sample, monkeypatch):
    original = vp.build_pointer_messages

    def mutating_builder(sample, **kwargs):
        result = original(sample, **kwargs)
        sample["actor_step"] += " changed"
        return result

    monkeypatch.setattr(vp, "build_pointer_messages", mutating_builder)
    with pytest.raises(ValueError):
        vp.build_visual_pointer_messages(sample)


@pytest.mark.parametrize("line", [((1, 3), (29, 27)), ((29, 27), (1, 3)), ((1, 27), (29, 3)),
                                   ((16, 1), (16, 30)), ((1, 16), (30, 16)), ((16, 16), (16, 16))])
def test_roi_accounting_matches_independent_full_canvas_union_with_crossing_lines(line):
    size = (32, 32)
    paint, badge, ocr = (Image.new("L", size, 0) for _ in range(3))
    ImageDraw.Draw(paint).line(((0, 16), (31, 16)), fill=255)
    ImageDraw.Draw(paint).rectangle((0, 0, 3, 3), fill=255)
    ImageDraw.Draw(badge).rectangle((0, 0, 3, 3), fill=255)
    ImageDraw.Draw(ocr).rectangle((8, 8, 23, 23), fill=255)
    full_line = Image.new("L", size, 0)
    ImageDraw.Draw(full_line).line(line, fill=255)
    prior_painted = paint.histogram()[255]
    prior_ocr = ImageChops.multiply(paint, ocr).histogram()[255]
    result = vp._line_trial(line, paint, badge, ocr, prior_painted, prior_ocr, 12)
    if ImageChops.multiply(full_line, badge).getbbox() is not None:
        assert result is None
    else:
        roi, local, painted, painted_ocr = result
        reconstructed = Image.new("L", size, 0)
        reconstructed.paste(local, roi[:2])
        assert reconstructed.tobytes() == full_line.tobytes()
        union = ImageChops.lighter(paint, full_line)
        assert painted == union.histogram()[255] + 12
        assert painted_ocr == ImageChops.multiply(union, ocr).histogram()[255]


def test_impossible_global_badge_lower_bound_does_not_try_positions(sample, font_asset, monkeypatch):
    def forbidden_trial(*args, **kwargs):
        raise AssertionError("a badge cannot fit even before adding a leader")

    monkeypatch.setattr(vp, "_positions", forbidden_trial)
    with pytest.raises(vp.VisualLayoutError, match="mask budget exceeded"):
        render(sample, font_asset, config={"max_paint_fraction": 0})


@pytest.mark.parametrize("counter", ["paint", "ocr"])
def test_corrupt_roi_cached_count_is_fatal_not_expected_layout_refusal(sample, font_asset, monkeypatch, counter):
    original = vp._line_trial

    def undercount(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is None:
            return None
        roi, local, painted, ocr = result
        return roi, local, painted - 1 if counter == "paint" else painted, ocr - 1 if counter == "ocr" else ocr

    monkeypatch.setattr(vp, "_line_trial", undercount)
    with pytest.raises(ValueError, match="accounting differs") as exc:
        render(sample, font_asset)
    assert not isinstance(exc.value, vp.VisualLayoutError)
