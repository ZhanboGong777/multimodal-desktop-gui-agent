"""Synthetic query/relationship checks without model or dataset evidence."""

from __future__ import annotations

import copy
import inspect
import random
from collections import Counter

import pytest

from gui_agent.finetuning.pointer_candidates_v3 import (
    POLICY,
    _query,
    _score,
    _tokens,
    select_query_relations,
)


def item(identifier, source, x, y, width, height, *, text="", confidence=0.5):
    return {"candidate_id": identifier, "source": source,
            "role": "text_region" if source == "ocr" else "contour_region",
            "bbox": {"left": x, "top": y, "right": x+width, "bottom": y+height},
            "text": text, "confidence": confidence}


def ocr_noise(count=80):
    return [item(f"o{i:03}", "ocr", 10+i*8, 20, 30, 12,
                 text=f"Noise{i}", confidence=0.99) for i in range(count)]


def panels(count=80):
    # All centers share the field/button bucket; large areas win old v1 order.
    return [item(f"p{i:03}", "contour", 300+i%20, 380+i//20, 350, 100)
            for i in range(count)]


def choose(candidates, *, instruction="Choose the requested item", actor_step="Choose Alpha"):
    return select_query_relations(candidates, 1024, 768,
                                  instruction=instruction, actor_step=actor_step)


def ids(candidates):
    return {c["candidate_id"] for c in candidates}


def test_low_confidence_query_label_survives_more_than_32_ocr_proposals():
    label = item("target", "ocr", 15, 22, 70, 12, text="Alpha", confidence=0.05)
    raw = ocr_noise()+panels()+[label]
    selected = choose(raw)
    assert "target" in ids(selected)
    assert len(selected) == 64
    assert sum(c["source"] == "ocr" for c in selected) == 32
    assert next(c for c in selected if c["candidate_id"] == "target") is label


def test_actor_step_has_twice_goal_token_weight_and_whole_label_matches():
    query = _query("Choose Alpha", "Choose Beta")
    assert _score("Beta", query) == 2*_score("Alpha", query)
    query = _query("Open product", 'Open "Microsoft Edge"')
    assert _score("Microsoft Edge", query) > _score("Edge", query)
    assert _score("in", _query("pin item", "pin item")) == 0


def test_long_matching_raw_title_precedes_short_nonmatching_labels():
    title = "A complete long document title with additional descriptive words and project details"
    assert len(title) > 64
    label = item("long_title", "ocr", 15, 22, 350, 12, text=title, confidence=0.01)
    selected = choose(ocr_noise()+panels()+[label], actor_step=f"Open {title}")
    assert "long_title" in ids(selected)
    assert next(c for c in selected if c["candidate_id"] == "long_title")["text"] == title


def test_single_quoted_product_uses_tokens_without_treating_apostrophe_as_quote():
    query = _query("it's a shortcut", "Choose 'Ａlpha'")
    assert query[0][2] == [{"alpha"}]
    assert query[1][2] == []
    assert _score("Alpha", query) > 0


def test_context_change_changes_membership_but_preserves_raw_ids():
    alpha = item("alpha", "ocr", 15, 22, 60, 12, text="Alpha", confidence=0.01)
    beta = item("beta", "ocr", 75, 22, 60, 12, text="Beta", confidence=0.01)
    raw = ocr_noise()+panels()+[alpha, beta]
    first = choose(raw, actor_step="Choose Alpha")
    second = choose(raw, actor_step="Choose Beta")
    assert "alpha" in ids(first) and "beta" not in ids(first)
    assert "beta" in ids(second) and "alpha" not in ids(second)
    assert ids(first) | ids(second) <= ids(raw)


def test_empty_field_pair_enters_together_after_large_spatial_proposals():
    first = item("field_a", "contour", 380, 430, 60, 20)
    second = item("field_b", "contour", 455, 430, 60, 20)
    selected = choose(panels()+ocr_noise()+[first, second])
    assert {"field_a", "field_b"} <= ids(selected)
    assert len(selected) == 64


def test_independent_buttons_precede_parent_covering_multiple_button_frames():
    first = item("button_a", "contour", 390, 430, 45, 20)
    second = item("button_b", "contour", 460, 430, 45, 20)
    parent = item("whole_buttons", "contour", 380, 425, 130, 30)
    extra = [item(f"small{i}", "contour", 300+i*12, 450, 12, 20) for i in range(8)]
    labels = [item("label_a", "ocr", 394, 434, 38, 12, text="Launch", confidence=0.5),
              item("label_b", "ocr", 463, 434, 38, 12, text="Cancel", confidence=0.5)]
    selected = choose(panels()+ocr_noise()+[first, second, parent]+extra+labels,
                      actor_step="Choose Launch or Cancel")
    assert {"button_a", "button_b"} <= ids(selected)
    assert "whole_buttons" not in ids(selected)


def test_existing_contour_text_line_parent_can_use_relationship_slot():
    parent = item("line_parent", "contour", 380, 430, 100, 20, text="Confirm action")
    parent["evidence"] = {"kind": "contour_text_line_parent", "children": ["raw_a", "raw_b"]}
    selected = choose(panels()+ocr_noise()+[parent], actor_step="Confirm action")
    assert "line_parent" in ids(selected)
    assert next(c for c in selected if c["candidate_id"] == "line_parent") is parent


def test_text_covered_long_word_contours_are_not_empty_field_pair_proposals():
    covered = [item("covered_a", "contour", 380, 430, 60, 20),
               item("covered_b", "contour", 455, 430, 60, 20)]
    labels = [item("word_a", "ocr", 385, 434, 50, 12, text="Toolbar"),
              item("word_b", "ocr", 460, 434, 50, 12, text="Region")]
    fields = [item("empty_a", "contour", 380, 470, 60, 20),
              item("empty_b", "contour", 455, 470, 60, 20)]
    bounded = [item(f"bounded{i}", "contour", 300+i*12, 450, 12, 20) for i in range(8)]
    selected = choose(panels()+ocr_noise()+covered+labels+fields+bounded)
    assert {"empty_a", "empty_b"} <= ids(selected)
    assert not {"covered_a", "covered_b"} & ids(selected)


def test_zero_query_text_line_parents_do_not_displace_field_pair_or_bounded_frames():
    lines = [item(f"line{i}", "contour", 320+i*2, 450, 80, 20, text="Toolbar")
             for i in range(20)]
    for line in lines:
        line["evidence"] = {"kind": "contour_text_line_parent"}
    fields = [item("empty_a", "contour", 380, 470, 60, 20),
              item("empty_b", "contour", 455, 470, 60, 20)]
    bounded = [item(f"bounded{i}", "contour", 300+i*12, 450, 12, 20) for i in range(8)]
    selected = choose(panels()+ocr_noise()+lines+fields+bounded)
    assert {"empty_a", "empty_b"} <= ids(selected)
    assert not ids(lines) & ids(selected)


def test_same_geometry_ocr_parent_does_not_duplicate_an_atomic_slot():
    atomic = item("atomic", "ocr", 10, 10, 90, 12, text="Open", confidence=0.99)
    parent = item("parent", "ocr", 10, 10, 90, 12, text="Open settings", confidence=0.4)
    parent["evidence"] = {"kind": "aligned_ocr_parent"}
    selected = choose([atomic, parent], actor_step="Open settings")
    assert selected == [parent]


def test_unicode_chinese_bigrams_numbers_and_symbol_noise():
    assert _tokens("ＢＵＴＴＯＮ、设置编号１２") == {"button", "设置", "置编", "编号", "12"}
    assert _tokens("» + =") == set()
    noise = [item(f"n{i:03}", "ocr", 10+i*8, 20, 8, 12, text="»", confidence=1)
             for i in range(80)]
    label = item("chinese", "ocr", 15, 22, 80, 12, text="语言设置", confidence=0.05)
    number = item("number", "ocr", 95, 22, 50, 12, text="２０２７", confidence=0.05)
    selected = choose(noise+panels()+[label, number],
                      instruction="使用编号2027", actor_step="选择语言设置")
    assert {"chinese", "number"} <= ids(selected)


class GuardedCandidate(dict):
    def __getitem__(self, key):
        if key in {"actions", "preferred", "target_point", "reviews"}:
            raise AssertionError(f"forbidden label read: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in {"actions", "preferred", "target_point", "reviews"}:
            raise AssertionError(f"forbidden label read: {key}")
        return super().get(key, default)


def test_interface_and_candidate_access_exclude_action_and_label_inputs():
    assert list(inspect.signature(select_query_relations).parameters) == [
        "candidates", "width", "height", "instruction", "actor_step"]
    original = item("raw", "ocr", 10, 10, 60, 12, text="Alpha")
    poisoned = GuardedCandidate({**original, "actions": object(), "preferred": object(),
                                "target_point": object(), "reviews": object()})
    assert choose([poisoned])[0] is poisoned
    with pytest.raises(TypeError, match="actions"):
        select_query_relations([original], 1024, 768,
                               instruction="Choose item", actor_step="Choose Alpha", actions=[])


def test_reordering_is_invariant_selection_is_id_sorted_and_inputs_unchanged():
    fields = [item("fa", "contour", 380, 430, 60, 20),
              item("fb", "contour", 455, 430, 60, 20)]
    raw = ocr_noise()+panels()+fields
    original = copy.deepcopy(raw)
    first = choose(raw)
    shuffled = list(reversed(raw))
    random.Random(7).shuffle(shuffled)
    assert choose(shuffled) == first
    assert raw == original
    assert [c["candidate_id"] for c in first] == sorted(ids(first))


def test_per_source_quota_and_cross_source_backfill_are_fixed():
    selected = choose(ocr_noise()+panels())
    assert len(selected) == 64
    assert Counter(c["source"] for c in selected) == {"ocr": 32, "contour": 32}
    sparse = ocr_noise(20)+panels()
    selected = choose(sparse)
    assert len(selected) == 64
    assert sum(c["source"] == "ocr" for c in selected) == 20
    assert sum(c["source"] == "contour" for c in selected) == 44
    assert choose(ocr_noise(50)) == sorted(ocr_noise(50), key=lambda c: c["candidate_id"])
    assert choose([]) == []
    assert POLICY == "query_relations_v3"


@pytest.mark.parametrize("instruction,actor_step", [("", "Choose"), ("Choose", ""), (None, "Choose")])
def test_context_must_be_explicit_nonempty_text(instruction, actor_step):
    with pytest.raises(ValueError, match="nonempty"):
        choose([], instruction=instruction, actor_step=actor_step)
