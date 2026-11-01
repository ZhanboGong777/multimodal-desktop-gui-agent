"""Contextual candidate IDs must mean the same thing in every protocol path."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning import pointer_protocol as pp
from gui_agent.finetuning.evaluation import canonical_json, file_sha256
from gui_agent.finetuning.training import build_messages, load_prepared_records


@pytest.fixture
def contextual_sample(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 80), "white").save(image)
    snapshot = {"image_width": 100, "image_height": 80,
                "image_file_sha256": file_sha256(image), "compact_policy": pp.CONTEXT_POLICY,
                "candidates": [{"candidate_id": "c001", "source": "contour", "role": "contour_region",
                                "bbox": {"left": 10, "top": 10, "right": 50, "bottom": 30},
                                "text": "", "confidence": 0.5}]}
    path = tmp_path / "snapshot.json"
    path.write_text(canonical_json(snapshot), encoding="utf8")
    row = {"sample_id": "context", "image_path": str(image), "image_width": 100, "image_height": 80,
           "group_id": "synthetic-context", "source": "synthetic", "source_split": "train",
           "instruction": "Fill the form", "actor_step": "Focus the visible field and type the requested text",
           "actions": [{"action_type": "click", "x": 20, "y": 19},
                       {"action_type": "type_text", "text": "中文 \n"}],
           "candidate_target": [{"action_type": "click", "candidate": 1, "dx": -10, "dy": -1},
                                {"action_type": "type_text", "text": "中文 \n"}],
           "candidate_snapshot": snapshot}
    record = {"sample_id": row["sample_id"], "action_index": 0, "endpoint": "",
              "action_type": "click", "current_subtask": row["actor_step"], "instruction": row["instruction"],
              "target_point": {"x": 20, "y": 19}, "decision": "accepted", "preferred_candidate_id": "c001",
              "image_file_sha256": file_sha256(image), "snapshot_path": str(path),
              "snapshot_file_sha256": file_sha256(path)}
    review = tmp_path / "review.json"
    review.write_text(canonical_json({"schema": "week5.pointer_semantic_review.v1", "status": "closed",
                                     "reviews": [record]}), encoding="utf8")
    row["candidate_label_evidence"] = {"review_path": str(review), "review_sha256": file_sha256(review)}
    row["candidate_selection_evidence"] = pp.selection_binding(
        snapshot, instruction=row["instruction"], actor_step=row["actor_step"])
    return row


def mapped(row, **changes):
    args = {"instruction": row["instruction"], "actor_step": row["actor_step"],
            "expected_selection": row["candidate_selection_evidence"]} | changes
    return pp.map_pointer_actions(canonical_json(row["candidate_target"]), row["candidate_snapshot"],
                                  row["image_path"], **args)


def test_contextual_training_prefix_inference_and_whole_native_mapping_share_ids(contextual_sample):
    row = contextual_sample
    native = copy.deepcopy(row["actions"])
    messages = build_messages(row, "conditional_actor", include_target=True, target_protocol="candidate_compact")
    pp.validate_pointer_sample(row)
    assert mapped(row) == native
    calls = []

    class Worker:
        def request(self, image, prompt, *, timeout_seconds):
            calls.append((image, prompt, timeout_seconds))
            return {"status": "ok", "generation": messages[1]["content"]}

    result = pp.request_pointer_actions(Worker(), row, timeout_seconds=10)
    assert calls == [(row["image_path"], messages[0]["content"][1]["text"], 10)]
    assert result["mapped_actions"] == native and result["mapping_error"] is None
    assert result["candidate_selection_evidence"] == row["candidate_selection_evidence"]
    assert row["actions"] == native


@pytest.mark.parametrize("change", ["instruction", "actor_step", "numbering", "snapshot", "missing"])
def test_changed_or_missing_selection_binding_is_refused(contextual_sample, change):
    row = contextual_sample
    if change in {"instruction", "actor_step"}:
        row[change] += " changed"
    elif change == "numbering":
        row["candidate_selection_evidence"]["selected_candidate_ids"] = ["c999"]
    elif change == "snapshot":
        row["candidate_snapshot"]["candidates"][0]["confidence"] = 0.6
    else:
        row.pop("candidate_selection_evidence")
    with pytest.raises(ValueError, match="selection binding"):
        pp.validate_pointer_sample(row)


@pytest.mark.parametrize("field", ["instruction", "actor_step"])
def test_missing_explicit_context_is_refused_before_inference(contextual_sample, field):
    row = contextual_sample
    kwargs = {"instruction": row["instruction"], "actor_step": row["actor_step"]}
    kwargs[field] = None
    with pytest.raises(ValueError, match="explicit goal"):
        pp.select_candidates(row["candidate_snapshot"], **kwargs)


@pytest.mark.parametrize("change", ["missing_instruction", "different_instruction"])
def test_v3_review_binds_the_original_goal(contextual_sample, change):
    row = contextual_sample
    path = Path(row["candidate_label_evidence"]["review_path"])
    review = json.loads(path.read_text("utf8"))
    if change == "missing_instruction":
        review["reviews"][0].pop("instruction")
    else:
        review["reviews"][0]["instruction"] = "Another goal"
    path.write_text(canonical_json(review), encoding="utf8")
    row["candidate_label_evidence"]["review_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="semantic review"):
        pp.validate_pointer_sample(row)


def test_model_reply_cannot_be_remapped_after_context_changes(contextual_sample):
    row = contextual_sample

    class Worker:
        def request(self, *args, **kwargs):
            row["actor_step"] = "A different step"
            return {"status": "ok", "generation": canonical_json(row["candidate_target"])}

    result = pp.request_pointer_actions(Worker(), row, timeout_seconds=10)
    assert result["mapped_actions"] is None and "selection binding" in result["mapping_error"]
    assert result["raw_reply"]["generation"] == canonical_json(row["candidate_target"])


@pytest.mark.parametrize("change", ["spatial_v1", "control_balanced_v2", "candidate_geometry", "replacement"])
def test_worker_cannot_downgrade_policy_or_replace_candidate_numbering(contextual_sample, change):
    row = contextual_sample
    original_binding = copy.deepcopy(row["candidate_selection_evidence"])

    class Worker:
        def request(self, *args, **kwargs):
            if change in {"candidate_geometry", "replacement"}:
                if change == "replacement":
                    row["candidate_snapshot"] = copy.deepcopy(row["candidate_snapshot"])
                row["candidate_snapshot"]["candidates"][0]["bbox"]["right"] += 2
            else:
                row["candidate_snapshot"]["compact_policy"] = change
            return {"status": "ok", "generation": canonical_json(row["candidate_target"])}

    result = pp.request_pointer_actions(Worker(), row, timeout_seconds=10)
    assert result["mapped_actions"] is None and "selection binding" in result["mapping_error"]
    assert result["candidate_selection_evidence"] == original_binding
    assert result["snapshot_sha256"] == original_binding["snapshot_sha256"]


def test_label_fields_cannot_enter_contextual_prompt(contextual_sample):
    row = contextual_sample
    prefix = pp.build_pointer_messages(row)
    row.update(actions="HIDDEN_GT", candidate_target="HIDDEN_TARGET", candidate_label_evidence="HIDDEN_REVIEW",
               candidate_selection_evidence="HIDDEN_BINDING")
    assert pp.build_pointer_messages(row) == prefix
    assert "HIDDEN" not in canonical_json(prefix)


def test_actual_pointer_loader_keeps_context_binding_and_every_native_field(contextual_sample, tmp_path):
    row = contextual_sample
    path = tmp_path / "records.jsonl"
    path.write_text(canonical_json(row) + "\n", encoding="utf8")
    loaded = load_prepared_records(path, target_protocol="candidate_compact")
    assert len(loaded) == 1
    assert loaded[0]["actions"] == row["actions"]
    assert loaded[0]["candidate_selection_evidence"] == row["candidate_selection_evidence"]
