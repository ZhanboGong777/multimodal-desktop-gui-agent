"""Synthetic bounded joint-domain layouts; no actual404, model or GUI use."""
from __future__ import annotations

import copy
import io
import time
from pathlib import Path

import numpy as np
import pytest
import test_week5_visual_pointer as fixtures
from PIL import Image, ImageDraw, ImageFont

from gui_agent.finetuning import visual_pointer as v1
from gui_agent.finetuning import visual_pointer_v2 as v2

sample = fixtures.sample
font_asset = fixtures.font_asset


def render(sample, font_asset, **kwargs):
    return v2.render_visual_pointer_v2(sample, **font_asset[0], **kwargs)


def assert_v2_visible(result, sample):
    # Reuse the independent pixel/mapping oracle; its old field bound checks
    # static enumeration rank here, never the v2 DFS assignment-attempt count.
    adapted = copy.deepcopy(result)
    for row in adapted["metadata"]["id_table"]:
        assert "placement_attempts" not in row
        assert row["static_position_rank"] == row["static_position_index"] + 1
        row["placement_attempts"] = row["static_position_rank"]
    fixtures.assert_all_visible(adapted, sample)


def font_from_fixture(font_asset):
    return ImageFont.truetype(io.BytesIO(Path(font_asset[0]["font_path"]).read_bytes()), size=27,
                              layout_engine=ImageFont.Layout.BASIC)


def raw_complement(width, height, holes):
    """Synthetic OCR rectangles cover everything except exact designated holes."""
    rectangles = [(0, 0, width, height)]
    for hl, ht, hr, hb in holes:
        result = []
        for left, top, right, bottom in rectangles:
            il, it, ir, ib = max(left, hl), max(top, ht), min(right, hr), min(bottom, hb)
            if il >= ir or it >= ib:
                result.append((left, top, right, bottom))
                continue
            result.extend([(left, top, right, it), (left, ib, right, bottom),
                           (left, it, il, ib), (ir, it, right, ib)])
        rectangles = [r for r in result if r[0] < r[2] and r[1] < r[3]]
    return {"candidates": [fixtures.candidate(f"barrier{i}", r[0], r[1], source="ocr",
                                             width=r[2] - r[0], height=r[3] - r[1])
                           for i, r in enumerate(rectangles)]}


def test_v2_native_input_contract_exact_prompt_all_ids_and_determinism(sample, font_asset):
    native = Path(sample["image_path"]).read_bytes()
    before = copy.deepcopy(sample)
    result = render(sample, font_asset)
    assert_v2_visible(result, sample)
    m = result["metadata"]
    assert m["version"] == "week5.visual-pointer.v2"
    assert m["layout_policy"] == v2.LAYOUT_POLICY and m["layout_status"] == "all_ids_rendered"
    assert m["numpy_version"] == np.__version__
    search = m["layout_search"]
    assert search["max_ranked_options"] == 4 and search["max_assignment_attempts"] == 256
    assert len(search["placement_id_order"]) == len(set(search["placement_id_order"])) == 8
    assert search["tie_order"] == ["top", "left", "bottom", "right", "raw_candidate_id"]
    assert search["optimality_or_impossibility_proven"] is False
    assert result["prompt"] == v1.build_visual_pointer_messages(sample)[0]["content"][1]["text"]
    assert m["training_admission"] is m["model_quality_proven"] is False
    assert m["config"] == v1.VisualPointerConfig().__dict__
    assert render(sample, font_asset) == result
    assert sample == before and Path(sample["image_path"]).read_bytes() == native


def test_hidden_actions_targets_reviews_and_sample_id_never_read(sample, font_asset):
    expected = render(sample, font_asset)
    hidden = fixtures.VisibleOnly(copy.deepcopy(sample))
    hidden.update(sample_id=object(), actions=object(), candidate_target=object(), candidate_label_evidence=object())
    assert render(hidden, font_asset) == expected


def test_geometry_scheduling_relabel_keeps_original_mapping_and_id_sorted_table(sample, font_asset):
    raw = sample["candidate_snapshot"]
    results = [render(fixtures.derived_sample(sample, view), font_asset, raw_snapshot=raw) for view in (0, 2)]
    maps = [{r["raw_candidate_id"]: r["native_center"] for r in result["metadata"]["id_table"]}
            for result in results]
    assert maps[0] == maps[1]
    schedules = []
    for result in results:
        m = result["metadata"]
        assert [r["id"] for r in m["id_table"]] == list(range(1, 9))
        schedules.append([m["id_table"][i - 1]["raw_candidate_id"] for i in m["layout_search"]["placement_id_order"]])
    assert schedules[0] == schedules[1]


def test_first_fit_consumes_only_slot_but_mrv_joint_domain_succeeds(font_asset):
    font = font_from_fixture(font_asset)
    glyph = font.getbbox("1")
    width, height = glyph[2] - glyph[0] + 4, glyph[3] - glyph[1] + 4
    assert font.getbbox("1") == font.getbbox("2")
    ax, ay = 350, 350
    bx, by = ax + 2 + width // 2 + 248, ay
    selected = [fixtures.candidate("flexible", ax - 1, ay - 1, width=2, height=2),
                fixtures.candidate("constrained", bx - 1, by - 1, width=2, height=2)]
    shared = (ax + 2, ay - height // 2, ax + 2 + width, ay - height // 2 + height)
    private = (ax - 128 - width // 2, ay - height // 2,
               ax - 128 - width // 2 + width, ay - height // 2 + height)
    raw = raw_complement(1024, 768, [shared, private])
    native = Image.new("RGB", (1024, 768), (90, 110, 130))
    with pytest.raises(v1.VisualLayoutError):
        v1._layout(native, selected, raw, font, v1.VisualPointerConfig())
    output, placements, counts, search = v2._layout_v2(native, selected, raw, font, v1.VisualPointerConfig())
    assert search["static_domain_counts"] == [2, 1]
    assert search["placement_id_order"] == [2, 1]
    assert tuple(placements[1]["badge_bbox"].values()) == shared
    assert tuple(placements[0]["badge_bbox"].values()) == private
    assert len(placements) == 2 and output.size == native.size
    assert counts["badge_raw_ocr_overlap_pixels"] == counts["badge_leader_overlap_pixels"] == 0


def test_cost_prefers_less_new_ocr_than_first_fit_without_cap_relaxation(font_asset):
    font = font_from_fixture(font_asset)
    selected = [fixtures.candidate("only", 299, 299, width=2, height=2)]
    raw = {"candidates": [fixtures.candidate("leader_ocr", 301, 298, source="ocr", width=2, height=2),
                           fixtures.candidate("far_ocr", 0, 0, source="ocr", width=100, height=100)]}
    image = Image.new("RGB", (1024, 768), (90, 110, 130))
    _, _, old_counts = v1._layout(image, selected, raw, font, v1.VisualPointerConfig())
    _, placements, counts, _ = v2._layout_v2(image, selected, raw, font, v1.VisualPointerConfig())
    assert old_counts["raw_ocr_painted_pixels"] > counts["raw_ocr_painted_pixels"] == 0
    assert placements[0]["static_position_index"] != 0
    assert counts["raw_ocr_painted_pixels"] <= counts["raw_ocr_union_pixels"] * .02


def synthetic_geometry(domains, size=(128, 128), ocr=None, ties=None):
    """Explicit algorithm-only joint domains with actual rasterized line caches."""
    width, height = size
    rows, pixel_arrays = [], []
    for owner, box, line in domains:
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).line(line, fill=255)
        pixels = np.flatnonzero(np.asarray(mask).ravel()).astype(np.int32)
        rows.append({"owner": owner, "box": box, "line": line, "position_index": len(rows)})
        pixel_arrays.append(pixels)
    lengths = np.array([len(p) for p in pixel_arrays], dtype=np.int32)
    indices = np.zeros((len(rows), int(lengths.max())), dtype=np.int32)
    valid = np.zeros(indices.shape, dtype=bool)
    for i, pixels in enumerate(pixel_arrays):
        indices[i, :len(pixels)] = pixels
        valid[i, :len(pixels)] = True
    owners = np.array([r["owner"] for r in rows], dtype=np.int32)
    count = int(owners.max()) + 1
    areas = [(next(r["box"] for r in rows if r["owner"] == owner)[2] - next(r["box"] for r in rows if r["owner"] == owner)[0])
             * (next(r["box"] for r in rows if r["owner"] == owner)[3] - next(r["box"] for r in rows if r["owner"] == owner)[1])
             for owner in range(count)]
    ocr_mask = np.zeros((height, width), dtype=bool) if ocr is None else ocr
    return v2._Geometry(rows, owners, np.array([r["box"] for r in rows], dtype=np.int32), indices, valid,
                        valid & ocr_mask.ravel()[indices], lengths, np.array([areas[o] for o in owners]),
                        areas, ties or [(i, 0, i + 1, 1, f"raw{i}") for i in range(count)], ocr_mask,
                        np.zeros((height, width), dtype=bool), 0,
                        np.bincount(owners, minlength=count).tolist())


def empty_state(size=(128, 128)):
    return v2._State(*(np.zeros(size[::-1], dtype=bool) for _ in range(4)))


def test_forward_check_rejects_last_joint_slot_consumption_and_exact_undo():
    # A and B each have two static joint options. A's cheap line crosses both
    # possible B badges. B's cheap line crosses A's safe badge. Forward checking
    # must undo A's cheap choice and take the safe branch before placing B.
    domains = [(0, (10, 10, 20, 20), ((21, 15), (90, 15))),
               (0, (10, 90, 20, 100), ((21, 95), (90, 95))),
               (1, (40, 10, 50, 20), ((45, 21), (15, 95))),
               (1, (70, 10, 80, 20), ((75, 21), (75, 50)))]
    g = synthetic_geometry(domains)
    # The cheapest A branch is the shorter top line; B cannot use either badge.
    chosen, state, search = v2._solve(g, v1.VisualPointerConfig())
    assert chosen[0] == 1 and chosen[1] == 3
    assert search["backtracks"] >= 1 and search["forward_checks"] >= 2
    expected_badge = np.zeros_like(state.badge)
    expected_line = Image.new("L", (128, 128), 0)
    for d in chosen.values():
        left, top, right, bottom = g.domains[d]["box"]
        expected_badge[top:bottom, left:right] = True
        ImageDraw.Draw(expected_line).line(g.domains[d]["line"], fill=255)
    assert np.array_equal(state.badge, expected_badge)
    assert np.array_equal(state.leaders, np.asarray(expected_line, dtype=bool))
    assert state.painted == np.count_nonzero(state.paint)


def test_shared_line_cost_counts_union_and_bidirectional_badge_leader_conflicts():
    domains = [(0, (10, 10, 20, 20), ((20, 15), (70, 15))),
               (1, (80, 10, 90, 20), ((79, 15), (30, 15))),
               (1, (40, 10, 50, 20), ((40, 21), (70, 21))),
               (1, (80, 30, 90, 40), ((79, 35), (10, 15)))]
    ocr = np.ones((128, 128), dtype=bool)
    for _, (left, top, right, bottom), _ in domains:
        ocr[top:bottom, left:right] = False
    g = synthetic_geometry(domains, ocr=ocr)
    state = empty_state()
    original = (state.badge.copy(), state.spacing.copy(), state.leaders.copy(), state.paint.copy())
    _, _, total, ocr_new = v2._feasible(g, state, (0, 1), v1.VisualPointerConfig())
    undo = v2._assign(g, state, 0, int(total[0]), int(ocr_new[0]), 2)
    feasible, _, total, ocr_new = v2._feasible(g, state, (1,), v1.VisualPointerConfig())
    assert feasible[1]
    assert int(ocr_new[1]) == 9 < g.line_lengths[1]
    assert not feasible[2]  # Its new badge covers the old leader.
    assert not feasible[3]  # Its new leader crosses the old badge.
    v2._undo(g, state, 0, undo)
    assert all(np.array_equal(actual, expected) for actual, expected in zip(
        (state.badge, state.spacing, state.leaders, state.paint), original, strict=True))
    assert state.painted == state.ocr_painted == 0


def test_reserved_all_remaining_badge_area_rejects_long_first_leader():
    g = synthetic_geometry([(0, (0, 0, 20, 20), ((21, 10), (80, 10))),
                            (1, (80, 80, 100, 100), ((79, 90), (70, 90)))], size=(100, 100))
    state = empty_state((100, 100))
    feasible, _, total, _ = v2._feasible(g, state, (0, 1), v1.VisualPointerConfig())
    assert total[0] < 800  # This partial first label alone would fit.
    assert not feasible[0]  # The remaining required badge also needs400px.
    with pytest.raises(v1.VisualLayoutError, match="budget_lower_bound"):
        v2._solve(g, v1.VisualPointerConfig())


def test_bounded_failure_classifies_search_not_proof_or_partial_fallback():
    g = synthetic_geometry([(0, (10, 10, 20, 20), ((21, 15), (90, 15))),
                            (1, (40, 10, 50, 20), ((45, 21), (90, 21)))])
    with pytest.raises(v1.VisualLayoutError, match="bounded_search_exhausted") as exc:
        v2._solve(g, v1.VisualPointerConfig())
    assert exc.value.layout_status == "bounded_search_exhausted"
    assert 0 < exc.value.layout_search["assignment_attempts"] <= 256
    assert exc.value.layout_search["optimality_or_impossibility_proven"] is False


def test_assignment_budget_stops_at_exact_256_without_claiming_impossibility():
    domains = []
    for owner in range(8):
        x = 10 + owner * 60
        domains.extend([(owner, (x, 10, x + 10, 20), ((x + 10, 15), (x + 20, 15))),
                        (owner, (x, 70, x + 10, 80), ((x + 10, 75), (x + 20, 75)))])
    domains.extend([(8, (510, 10, 520, 20), ((520, 15), (750, 15))),
                    (8, (530, 10, 540, 20), ((540, 15), (750, 15))),
                    (9, (600, 10, 610, 20), ((605, 21), (605, 35))),
                    (9, (640, 10, 650, 20), ((645, 21), (645, 35)))])
    g = synthetic_geometry(domains, size=(1024, 768))
    with pytest.raises(v1.VisualLayoutError, match="bounded_search_exhausted") as exc:
        v2._solve(g, v1.VisualPointerConfig())
    assert exc.value.layout_search["assignment_attempts"] == 256
    assert exc.value.layout_search["max_ranked_options"] == 4
    assert exc.value.layout_search["optimality_or_impossibility_proven"] is False


def test_corrupt_cached_geometry_rejected_by_independent_mask_reconstruction(sample, font_asset, monkeypatch):
    original = v2._solve

    def corrupt(*args, **kwargs):
        chosen, state, search = original(*args, **kwargs)
        state.leaders[0, 0] = True
        return chosen, state, search

    monkeypatch.setattr(v2, "_solve", corrupt)
    with pytest.raises(ValueError, match="independent full-canvas mask") as exc:
        render(sample, font_asset)
    assert not isinstance(exc.value, v1.VisualLayoutError)


@pytest.mark.parametrize("change", ["native", "font", "context", "raw", "selection"])
@pytest.mark.parametrize("layout_failure", [False, True])
def test_v2_post_render_integrity_fatal_even_during_expected_refusal(sample, font_asset, monkeypatch, change, layout_failure):
    raw = copy.deepcopy(sample["candidate_snapshot"])
    if change == "raw":
        sample = fixtures.derived_sample(sample, 2)
    original = v2._layout_v2

    def altered(*args, **kwargs):
        result = None if layout_failure else original(*args, **kwargs)
        if change == "native":
            Image.new("RGB", (1024, 768), "red").save(sample["image_path"])
        elif change == "font":
            Path(font_asset[0]["font_path"]).write_bytes(b"changed font")
        elif change == "context":
            sample["actor_step"] += " altered"
        elif change == "raw":
            raw["candidates"][0]["text"] = "changed raw"
        else:
            sample["candidate_selection_evidence"]["context_sha256"] = "0" * 64
        if layout_failure:
            raise v1.VisualLayoutError("bounded_search_exhausted")
        return result

    monkeypatch.setattr(v2, "_layout_v2", altered)
    with pytest.raises(ValueError) as exc:
        render(sample, font_asset, raw_snapshot=raw if change == "raw" else None)
    assert not isinstance(exc.value, v1.VisualLayoutError)


@pytest.mark.parametrize("change", ["stale_image", "font_hash", "context", "native_rgb", "missing_raw", "config", "empty"])
def test_v2_input_guard_failures_not_layout_refusals(sample, font_asset, change):
    kwargs = {}
    if change == "stale_image":
        Image.new("RGB", (1024, 768), "red").save(sample["image_path"])
    elif change == "font_hash":
        kwargs["font_sha256"] = "0" * 64
    elif change == "context":
        sample["actor_step"] += " stale"
    elif change == "native_rgb":
        sample["image_sha256"] = "0" * 64
    elif change == "missing_raw":
        sample = fixtures.derived_sample(sample, 2)
    elif change == "config":
        kwargs["config"] = {"max_paint_fraction": .09}
    else:
        sample["candidate_snapshot"]["candidates"] = []
        fixtures.bind(sample)
    with pytest.raises(ValueError) as exc:
        v2.render_visual_pointer_v2(sample, **(font_asset[0] | kwargs))
    assert not isinstance(exc.value, v1.VisualLayoutError)


@pytest.mark.parametrize("counter", ["paint", "ocr"])
def test_v2_final_full_canvas_detects_poisoned_increment_counters(sample, font_asset, monkeypatch, counter):
    original = v2._feasible

    def poisoned(*args, **kwargs):
        feasible, counts, total, ocr = original(*args, **kwargs)
        return feasible, counts, total - 1 if counter == "paint" else total, ocr - 1 if counter == "ocr" else ocr

    monkeypatch.setattr(v2, "_feasible", poisoned)
    with pytest.raises(ValueError, match="accounting differs") as exc:
        render(sample, font_asset)
    assert not isinstance(exc.value, v1.VisualLayoutError)


def test_v2_static_zero_domain_and_zero_budget_whole_refusal(tmp_path, font_asset):
    sample = fixtures.make_sample(tmp_path, size=(128, 128), candidates=[
        fixtures.candidate("full_ocr", 0, 0, source="ocr", width=128, height=128)])
    with pytest.raises(v1.VisualLayoutError, match="static_zero_domain"):
        render(sample, font_asset)
    other = fixtures.make_sample(tmp_path)
    with pytest.raises(v1.VisualLayoutError, match="budget_lower_bound"):
        render(other, font_asset, config={"max_paint_fraction": 0})


def test_synthetic64_cpu_performance_diagnostic(tmp_path, font_asset):
    candidates = [fixtures.candidate(f"raw{i:03}", 40 + i % 8 * 120, 40 + i // 8 * 85,
                                    width=20, height=16) for i in range(64)]
    sample = fixtures.make_sample(tmp_path, candidates=candidates)
    start = time.perf_counter()
    result = render(sample, font_asset)
    elapsed = time.perf_counter() - start
    assert_v2_visible(result, sample)
    print(f"SYNTHETIC64_CPU_SECONDS={elapsed:.9f}; ASSIGNMENT_ATTEMPTS={result['metadata']['layout_search']['assignment_attempts']}; "
          f"CACHED_JOINT_DOMAINS={result['metadata']['layout_search']['cached_joint_domains']}")
    assert len(result["metadata"]["id_table"]) == 64
