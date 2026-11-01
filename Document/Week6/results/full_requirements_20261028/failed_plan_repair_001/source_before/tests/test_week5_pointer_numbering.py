"""Synthetic numbering invariants; no real datasets, encoding or model calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning import pointer_protocol as pp
from gui_agent.finetuning.evaluation import canonical_json, file_sha256
from gui_agent.finetuning.pointer_numbering import numbered_view, remap_target, verify_numbered_view
from gui_agent.finetuning.training import load_prepared_records


@pytest.fixture
def prepare_module():
    path = Path(__file__).resolve().parents[1] / "scripts/week5_augment_pointer_ids.py"
    spec = importlib.util.spec_from_file_location("numbering_prepare_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_parent(tmp_path):
    image_path = tmp_path / "native.png"
    image = Image.new("RGB", (512, 384), "white")
    image.save(image_path)
    candidates = []
    for index in range(8):
        x, y = 20 + index * 40, 40 + index * 20
        source = "ocr" if index % 2 else "contour"
        candidates.append({"candidate_id": f"raw{index + 1:03}", "source": source,
                           "role": "text_region" if source == "ocr" else "contour_region",
                           "text": "Alpha" if source == "ocr" else "", "confidence": .5,
                           "bbox": {"left": x, "top": y, "right": x + 20, "bottom": y + 12}})
    snapshot = {"image_width": 512, "image_height": 384, "image_file_sha256": file_sha256(image_path),
                "compact_policy": "query_relations_v3", "candidates": candidates}
    sample = {"sample_id": "screenagent:synthetic_parent:001", "group_id": "screenagent:synthetic_parent",
              "source": "screenagent", "source_split": "train", "record_path": "synthetic_parent/001.json",
              "image_path": str(image_path), "image_width": 512, "image_height": 384,
              "image_file_sha256": file_sha256(image_path),
              "image_sha256": hashlib.sha256(b"512x384:" + image.tobytes()).hexdigest(),
              "instruction": "Complete the requested Alpha field actions.",
              "actor_step": "Click Alpha, scroll, drag, and enter the exact text.",
              "candidate_snapshot": snapshot}
    sample["candidate_selection_evidence"] = pp.selection_binding(snapshot, instruction=sample["instruction"], actor_step=sample["actor_step"])
    target = [{"action_type": kind, "candidate": index, "dx": 3 - index, "dy": index - 3}
              for index, kind in enumerate(("click", "double_click", "right_click", "move", "scroll"), 1)]
    target[-1]["scroll_amount"] = -3
    target += [{"action_type": "drag", "start_candidate": 6, "start_dx": 2, "start_dy": -1,
                "end_candidate": 7, "end_dx": -3, "end_dy": 4},
               {"action_type": "type_text", "text": "x \n中"},
               {"action_type": "hotkey", "keys": ["ctrl", "a"]},
               {"action_type": "key_press", "key": "enter"},
               {"action_type": "wait", "duration": .25}]
    sample["candidate_target"] = target
    sample["actions"] = pp.map_pointer_actions(canonical_json(target), snapshot, image_path,
                                               instruction=sample["instruction"], actor_step=sample["actor_step"],
                                               expected_selection=sample["candidate_selection_evidence"])
    snapshot_path = tmp_path / "original_snapshot.json"
    snapshot_path.write_text(canonical_json(snapshot), encoding="utf8")
    selected = pp.select_candidates(snapshot, instruction=sample["instruction"], actor_step=sample["actor_step"])
    reviews = []
    for index, action in enumerate(target):
        if action["action_type"] not in pp.MOUSE_TYPES:
            continue
        for endpoint in (("start_", "end_") if action["action_type"] == "drag" else ("",)):
            point = sample["actions"][index][endpoint[:-1]] if endpoint else sample["actions"][index]
            reviews.append({"sample_id": sample["sample_id"], "action_index": index, "endpoint": endpoint,
                            "decision": "accepted", "preferred_candidate_id": selected[action[endpoint + "candidate"] - 1]["candidate_id"],
                            "action_type": action["action_type"], "current_subtask": sample["actor_step"],
                            "instruction": sample["instruction"], "target_point": {k: point[k] for k in ("x", "y")},
                            "image_file_sha256": sample["image_file_sha256"], "snapshot_path": str(snapshot_path),
                            "snapshot_file_sha256": file_sha256(snapshot_path)})
    review_path = tmp_path / "closed_parent_review.json"
    review_path.write_text(canonical_json({"schema": "week5.pointer_semantic_review.v1", "status": "closed", "reviews": reviews}), encoding="utf8")
    sample["candidate_label_evidence"] = {"review_path": str(review_path), "review_sha256": file_sha256(review_path)}
    return sample


def write_train(tmp_path, parent, name="train.jsonl"):
    path = tmp_path / name
    path.write_text(canonical_json(parent) + "\n", encoding="utf8")
    return path


def write_plan(tmp_path, train_path, parent):
    review_path = Path(parent["candidate_label_evidence"]["review_path"])
    plan_path = tmp_path / "preparation_plan.json"
    plan = {"schema": "week5.train_only_numbering_preparation_plan.v1",
            "status": "registered_before_actual_augmentation",
            "bound_sha256": {str(train_path.resolve()): file_sha256(train_path),
                             str(review_path.resolve()): file_sha256(review_path)},
            "actual_preparation_budget": {"source_train_rows": 1, "source_train_groups": 1,
                                          "views_per_parent": 4, "total_derived_rows": 4,
                                          "unique_native_images": 1, "seed": 42}}
    plan_path.write_text(canonical_json(plan), encoding="utf8")
    return plan_path


def test_four_views_preserve_whole_native_array_offsets_and_nonmouse(synthetic_parent):
    original = copy.deepcopy(synthetic_parent)
    selected = pp.select_candidates(original["candidate_snapshot"], instruction=original["instruction"], actor_step=original["actor_step"])
    mappings = []
    for index in range(4):
        view = numbered_view(synthetic_parent, view_index=index)
        assert numbered_view(synthetic_parent, view_index=index) == view
        verify_numbered_view(synthetic_parent, view)
        snapshot = view["candidate_snapshot"]
        assert snapshot["compact_policy"] == "spatial_v1"
        derived = pp.select_candidates(snapshot, instruction=original["instruction"], actor_step=original["actor_step"])
        assert len(derived) == len(selected) and derived == snapshot["candidates"]
        for old, new in view["old_to_new"].items():
            assert view["new_to_old"][new] == old
            assert derived[new - 1]["original_id"] == selected[old - 1]["candidate_id"]
            for field in ("bbox", "text", "source", "role", "confidence"):
                assert derived[new - 1][field] == selected[old - 1][field]
        target = remap_target(original["candidate_target"], view["old_to_new"])
        assert remap_target(target, view["new_to_old"]) == original["candidate_target"]
        binding = pp.selection_binding(snapshot, instruction=original["instruction"], actor_step=original["actor_step"])
        assert pp.map_pointer_actions(canonical_json(target), snapshot, original["image_path"],
                                      instruction=original["instruction"], actor_step=original["actor_step"],
                                      expected_selection=binding) == original["actions"]
        for before, after in zip(original["candidate_target"], target, strict=True):
            assert {k: v for k, v in before.items() if k not in {"candidate", "start_candidate", "end_candidate"}} == {
                k: v for k, v in after.items() if k not in {"candidate", "start_candidate", "end_candidate"}}
        mappings.append(view["old_to_new"])
    assert mappings[0] == {i: i for i in range(1, len(selected) + 1)}
    assert any(mapping != mappings[0] for mapping in mappings[1:])
    assert synthetic_parent == original


def test_visible_view_does_not_read_gt_labels_or_review(synthetic_parent):
    changed = copy.deepcopy(synthetic_parent)
    changed.update(actions="HIDDEN_GT", candidate_target="HIDDEN_TARGET", candidate_label_evidence="HIDDEN_REVIEW")
    for index in range(4):
        assert numbered_view(changed, view_index=index) == numbered_view(synthetic_parent, view_index=index)


def test_raw_pool_is_v3_selected_once_then_every_numbered_item_is_visible(synthetic_parent):
    sample = copy.deepcopy(synthetic_parent)
    pool = []
    for index in range(96):
        source = "ocr" if index < 48 else "contour"
        x, y = 8 + index % 16 * 25, 12 + index // 16 * 25
        pool.append({"candidate_id": f"large{index:03}", "source": source,
                     "role": "text_region" if source == "ocr" else "contour_region",
                     "text": f"Alpha {index}" if source == "ocr" else "", "confidence": .5,
                     "bbox": {"left": x, "top": y, "right": x + 12, "bottom": y + 10}})
    sample["candidate_snapshot"]["candidates"] = pool
    sample["candidate_selection_evidence"] = pp.selection_binding(sample["candidate_snapshot"], instruction=sample["instruction"], actor_step=sample["actor_step"])
    original = pp.select_candidates(sample["candidate_snapshot"], instruction=sample["instruction"], actor_step=sample["actor_step"])
    assert len(original) == 64
    for index in range(4):
        view = numbered_view(sample, view_index=index)
        snapshot = view["candidate_snapshot"]
        assert len(snapshot["candidates"]) == 64
        assert {c["original_id"] for c in snapshot["candidates"]} == {c["candidate_id"] for c in original}
        assert pp.select_candidates(snapshot) == snapshot["candidates"]


@pytest.mark.parametrize("change", ["context", "original_snapshot", "parent_binding", "derived_snapshot", "bijection"])
def test_original_or_derived_binding_mutation_is_refused(synthetic_parent, change):
    sample = copy.deepcopy(synthetic_parent)
    view = numbered_view(sample, view_index=2)
    if change == "context":
        sample["actor_step"] += " altered"
    elif change == "original_snapshot":
        sample["candidate_snapshot"]["candidates"][0]["text"] = "changed"
    elif change == "parent_binding":
        sample["candidate_selection_evidence"]["context_sha256"] = "0" * 64
    elif change == "derived_snapshot":
        view["candidate_snapshot"]["candidates"][0]["bbox"]["left"] += 1
    else:
        view["old_to_new"][1] = view["old_to_new"][2]
    with pytest.raises(ValueError):
        verify_numbered_view(sample, view)


@pytest.mark.parametrize("target,mapping", [
    ([{"action_type": "click", "candidate": True, "dx": 0, "dy": 0}], {1: 1}),
    ([{"action_type": "click", "candidate": 2, "dx": 0, "dy": 0}], {1: 1}),
    ([{"action_type": "click", "candidate": 1, "dx": 0, "dy": 0, "x": 1}], {1: 1}),
    ([{"action_type": "scroll", "candidate": 1, "dx": 0, "dy": 0, "scroll_amount": 0}], {1: 1}),
    ([{"action_type": "hotkey", "keys": ["ctrl", "a"], "candidate": 1}], {1: 1}),
    ([{"action_type": "click", "candidate": 1, "dx": 0, "dy": 0}], {1: 1, 2: 1}),
])
def test_conflicting_or_malformed_remapping_cannot_repair_labels(target, mapping):
    with pytest.raises(ValueError):
        remap_target(target, mapping)


def test_prepare_all_views_have_inherited_review_and_existing_loader_accepts(synthetic_parent, tmp_path, prepare_module):
    train = write_train(tmp_path, synthetic_parent)
    plan = write_plan(tmp_path, train, synthetic_parent)
    summary = prepare_module.prepare_dataset(train, tmp_path / "prepared", workspace_root=tmp_path, preparation_plan=plan)
    assert summary["parent_rows"] == 1 and summary["derived_rows"] == 4 and summary["groups"] == 1
    assert summary["training_admission"] is False and summary["model_processor_detector_GUI_calls"] == 0
    loaded = load_prepared_records(tmp_path / "prepared/train.jsonl", target_protocol="candidate_compact")
    assert len(loaded) == len({row["sample_id"] for row in loaded}) == 4
    assert all(row["actions"] == synthetic_parent["actions"] for row in loaded)
    assert all(row["instruction"] == synthetic_parent["instruction"] and row["actor_step"] == synthetic_parent["actor_step"] for row in loaded)
    assert all(row["image_sha256"] == synthetic_parent["image_sha256"] and row["parent_sample_id"] == synthetic_parent["sample_id"] for row in loaded)
    review = json.loads((tmp_path / "prepared/derived_semantic_review.json").read_text("utf8"))
    assert review["status"] == "closed" and review["fresh_image_reviewed"] is False
    assert review["validation_annotations_used"] == 0
    assert len(review["reviews"]) == 7 * 4
    assert all(row["annotation_origin"] == "inherited_remapped_original_AI_review" and not row["fresh_image_reviewed"] for row in review["reviews"])


@pytest.mark.parametrize("change", ["official_test", "explicit_validation_split", "validation_filename", "outside_image", "output_dev_path", "collision"])
def test_prepare_refuses_test_dev_escape_and_output_collision(synthetic_parent, tmp_path, prepare_module, change):
    parent = copy.deepcopy(synthetic_parent)
    filename, output = "train.jsonl", tmp_path / "prepared"
    if change == "official_test":
        parent["source_split"] = "test"
    elif change == "explicit_validation_split":
        parent["split"] = "validation"
    elif change == "validation_filename":
        filename = "validation.jsonl"
    elif change == "outside_image":
        parent["image_path"] = str(tmp_path.parent / "outside.png")
    elif change == "output_dev_path":
        output = tmp_path / "test" / "prepared"
    else:
        output.mkdir()
        (output / "keep.txt").write_text("existing", encoding="utf8")
    train = write_train(tmp_path, parent, filename)
    plan = write_plan(tmp_path, train, parent)
    with pytest.raises((ValueError, FileExistsError)):
        prepare_module.prepare_dataset(train, output, workspace_root=tmp_path, preparation_plan=plan)
    if change == "collision":
        assert (output / "keep.txt").read_text("utf8") == "existing"


def test_source_mutation_during_numbering_cannot_publish_success(synthetic_parent, tmp_path, prepare_module, monkeypatch):
    train = write_train(tmp_path, synthetic_parent)
    plan = write_plan(tmp_path, train, synthetic_parent)
    original = prepare_module.numbered_view
    mutated = False
    def mutate_once(*args, **kwargs):
        nonlocal mutated
        view = original(*args, **kwargs)
        if not mutated:
            train.write_text(train.read_text("utf8") + "\n", encoding="utf8")
            mutated = True
        return view
    monkeypatch.setattr(prepare_module, "numbered_view", mutate_once)
    with pytest.raises(ValueError, match="changed|mutation"):
        prepare_module.prepare_dataset(train, tmp_path / "prepared", workspace_root=tmp_path, preparation_plan=plan)
    assert not (tmp_path / "prepared/preparation_receipt.json").exists()


def test_source_change_between_parse_and_binding_is_refused(synthetic_parent, tmp_path, prepare_module, monkeypatch):
    train = write_train(tmp_path, synthetic_parent)
    plan = write_plan(tmp_path, train, synthetic_parent)
    original = prepare_module.strict_json_loads
    mutated = False
    def mutate_after_first_parse(*args, **kwargs):
        nonlocal mutated
        value = original(*args, **kwargs)
        if not mutated and isinstance(value, dict) and value.get("sample_id") == synthetic_parent["sample_id"]:
            train.write_text(train.read_text("utf8") + "\n", encoding="utf8")
            mutated = True
        return value
    monkeypatch.setattr(prepare_module, "strict_json_loads", mutate_after_first_parse)
    with pytest.raises(ValueError, match="changed|mutation"):
        prepare_module.prepare_dataset(train, tmp_path / "prepared", workspace_root=tmp_path, preparation_plan=plan)
    assert not (tmp_path / "prepared/preparation_receipt.json").exists()
