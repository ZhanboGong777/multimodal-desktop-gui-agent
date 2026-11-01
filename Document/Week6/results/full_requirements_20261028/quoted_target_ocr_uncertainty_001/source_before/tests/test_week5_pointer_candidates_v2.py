"""CPU checks for versioned proposal selection, not grounding accuracy."""

from __future__ import annotations

import copy

import pytest

from gui_agent.finetuning.pointer_protocol import select_candidates


def item(identifier, source, x, y, w, h, *, text="", confidence=0.5):
    return {"candidate_id": identifier, "source": source,
            "role": "text_region" if source == "ocr" else "contour_region",
            "bbox": {"left": x, "top": y, "right": x+w, "bottom": y+h},
            "text": text, "confidence": confidence}


def snapshot(candidates):
    return {"image_width": 1024, "image_height": 768,
            "compact_policy": "control_balanced_v2", "candidates": candidates}


def test_control_tier_is_applied_before_contour_spatial_round_robin():
    panels = [item(f"p{i:03}", "contour", i, i, 600, 200) for i in range(40)]
    fields = [item(f"f{i:03}", "contour", i*20, 350, 30, 20) for i in range(32)]
    chosen = select_candidates(snapshot(panels + fields))
    assert {c["candidate_id"] for c in chosen} >= {c["candidate_id"] for c in fields}
    assert len(chosen) == 64


def test_complete_short_text_precedes_higher_confidence_glyph_noise():
    noise = [item(f"n{i:03}", "ocr", i*20, 10, 6, 10, text="»", confidence=0.99)
             for i in range(40)]
    labels = [item(f"l{i:03}", "ocr", i*20, 50, 35, 12, text="Button", confidence=0.6)
              for i in range(32)]
    chosen = select_candidates(snapshot(noise + labels))
    assert {c["candidate_id"] for c in chosen} >= {c["candidate_id"] for c in labels}


def test_selection_is_order_independent_and_never_reads_hidden_targets():
    evidence = [item(f"c{i:03}", "contour", i*10, 350, 30, 20) for i in range(80)]
    original = snapshot(evidence)
    selected = select_candidates(original)
    poisoned = copy.deepcopy(original)
    poisoned.update(instruction="Select c079", actions=[{"x": 800, "y": 350}],
                    target_bbox={"left": 700}, candidate_target="HIDDEN")
    poisoned["candidates"].reverse()
    assert select_candidates(poisoned) == selected


def test_v1_remains_default_and_unknown_policy_fails():
    raw = snapshot([item("c001", "contour", 10, 10, 30, 20)])
    raw.pop("compact_policy")
    assert select_candidates(raw) == select_candidates({**raw, "compact_policy": "spatial_v1"})
    with pytest.raises(ValueError, match="unknown compact"):
        select_candidates({**raw, "compact_policy": "anything_else"})


def test_empty_pools_and_backfill_keep_all_available_evidence():
    assert select_candidates(snapshot([])) == []
    evidence = [item(f"c{i:03}", "ocr", i*10, 350, 30, 20, text="Label") for i in range(50)]
    assert select_candidates(snapshot(evidence)) == evidence


def test_parent_line_preference_does_not_merge_or_invent_new_geometry():
    child = item("c002", "ocr", 10, 10, 24, 12, text="Open", confidence=0.95)
    parent = item("c001", "ocr", 10, 10, 90, 12, text="Open settings", confidence=0.6)
    parent["evidence"] = {"kind": "aligned_ocr_parent", "children": []}
    original = snapshot([child, parent])
    selected = select_candidates(original)
    assert selected == [parent, child]
    assert selected[0] is parent and selected[1] is child
