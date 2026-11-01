"""CPU-only diagnostic identities, leakage guards and lexical/annotation evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning.diagnostics import (
    audit_rows,
    bilingual_goal_input,
    diagnose_dataset,
    ensure_development_isolation,
    file_sha256,
    load_development_rows,
    review_raw_fields,
    select_diagnostic_rows,
    value_sha256,
)


def sample(tmp_path, name, *, actions=None, instruction="Type  exact  now", **changes):
    color = hashlib.sha256(name.encode()).digest()[:3]
    image = tmp_path / f"{name}.png"
    Image.new("RGB", (32, 24), tuple(color)).save(image)
    row = {"sample_id": name, "group_id": f"screenagent:{name}", "image_path": str(image),
           "image_width": 32, "image_height": 24, "source": "screenagent",
           "source_split": "train", "instruction": instruction,
           "actions": actions or [{"action_type": "click", "x": 3, "y": 5}],
           "action_annotations": [{}]}
    row.update(changes)
    return row


def dataset(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def loaded(tmp_path, name, rows, split="train"):
    return load_development_rows(dataset(tmp_path, name, rows), split=split)


def pair(tmp_path):
    train = dataset(tmp_path, "train.jsonl", [sample(tmp_path, f"train-{i}") for i in range(4)])
    validation = dataset(tmp_path, "validation.jsonl", [sample(tmp_path, f"val-{i}") for i in range(3)])
    return train, validation


def test_native_dimensions_bytes_pixels_and_original_row_identity(tmp_path):
    row = sample(tmp_path, "point", actions=[{"action_type": "click", "x": 31, "y": 23}])
    result, = loaded(tmp_path, "train.jsonl", [row])
    assert result["source_sample_sha256"] == value_sha256(row)
    assert result["image_file_sha256"] == file_sha256(row["image_path"])
    with Image.open(row["image_path"]) as image:
        digest = hashlib.sha256(b"32x24:" + image.convert("RGB").tobytes()).hexdigest()
    assert result["image_sha256"] == digest
    assert result["image_sha256"] != result["image_file_sha256"]
    assert result["diagnostic_split"] == "train"


@pytest.mark.parametrize("changes,reason", [
    ({"image_width": 31}, "dimensions"),
    ({"image_sha256": "0" * 64}, "decoded RGB"),
    ({"actions": [{"action_type": "click", "x": 32, "y": 0}]}, "outside_image"),
    ({"actions": [{"action_type": "click", "x": True, "y": 0}]}, "invalid_integer"),
    ({"source_split": "test"}, "official test"),
    ({"source_split": "test_public"}, "official test"),
    ({"source_split": "validation"}, "training-source"),
    ({"actions": [{"action_type": "key_press", "key": "Return"}]}, "not canonical"),
    ({"action_annotations": [{}, {}]}, "aligned"),
    ({"action_annotations": [{"target_bbox": {"left": 0, "top": 0, "right": 33, "bottom": 10}}]},
     "bbox outside"),
    ({"action_annotations": [{"target_bbox": {"left": 0, "top": 0, "right": True, "bottom": 10}}]},
     "integer"),
    ({"action_annotations": [{"target_bbox": {"left": 3, "top": 0, "right": 3, "bottom": 10}}]},
     "bbox outside"),
    ({"action_annotations": [{"expected_text": "leak"}]}, "unknown annotation"),
])
def test_malformed_development_data_is_refused(tmp_path, changes, reason):
    row = sample(tmp_path, "bad", **changes)
    with pytest.raises(ValueError, match=reason):
        loaded(tmp_path, "train.jsonl", [row])


def test_validation_origin_can_be_official_train_but_not_test(tmp_path):
    result, = loaded(tmp_path, "validation.jsonl", [sample(tmp_path, "a")], "validation")
    assert result["source_split"] == "train"
    assert result["diagnostic_split"] == "validation"
    row = sample(tmp_path, "b", source_split="validation")
    assert loaded(tmp_path, "validation.jsonl", [row], "validation")[0]["source_split"] == "validation"
    with pytest.raises(ValueError, match="train/validation only"):
        load_development_rows(tmp_path / "unread.jsonl", split="test")


def test_named_test_file_refused_before_opening(tmp_path):
    # It need not exist: no test bytes are consumed before this refusal.
    with pytest.raises(ValueError, match="test file"):
        load_development_rows(tmp_path / "test.jsonl", split="validation")


def test_duplicate_ids_and_json_keys_refused(tmp_path):
    row = sample(tmp_path, "a")
    with pytest.raises(ValueError, match="duplicate sample_id"):
        loaded(tmp_path, "train.jsonl", [row, row])
    path = tmp_path / "train.jsonl"
    path.write_text('{"source_split":"train","source_split":"test"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_development_rows(path, split="train")


@pytest.mark.parametrize("field", ["sample_id", "group_id", "image_sha256"])
def test_cross_split_leakage_refused(tmp_path, field):
    train = loaded(tmp_path, "train.jsonl", [sample(tmp_path, "a")])
    validation = loaded(tmp_path, "validation.jsonl", [sample(tmp_path, "b")], "validation")
    validation[0][field] = train[0][field]
    with pytest.raises(ValueError, match=field):
        ensure_development_isolation(train, validation)


def test_literal_absence_not_unlearnability_and_no_text_normalization(tmp_path):
    exact = {"action_type": "type_text", "text": " exact "}
    absent = {"action_type": "type_text", "text": " exact \n"}
    rows = loaded(tmp_path, "train.jsonl", [
        sample(tmp_path, "present", actions=[exact]),
        sample(tmp_path, "absent", actions=[absent]),
    ])
    audit = audit_rows(rows)
    assert audit["type_text"]["literal_present_in_instruction"] == 1
    assert audit["type_text"]["literal_absent_from_instruction"] == 1
    assert "absence does not establish unlearnability" in audit["type_text"]["interpretation"]
    assert "unlearnable_samples" not in audit
    assert rows[1]["actions"][0]["text"] == exact["text"]  # sorted sample ids: present is last


def test_bbox_half_open_disagreement_recorded_without_fabricated_bbox(tmp_path):
    box = {"left": 1, "top": 2, "right": 10, "bottom": 12}
    rows = loaded(tmp_path, "train.jsonl", [
        sample(tmp_path, "inside", actions=[{"action_type": "click", "x": 1, "y": 2}],
               action_annotations=[{"target_bbox": box}]),
        sample(tmp_path, "edge", actions=[{"action_type": "click", "x": 10, "y": 11}],
               action_annotations=[{"target_bbox": box}]),
        sample(tmp_path, "missing"),
    ])
    audit = audit_rows(rows)
    assert audit["clicks"]["actions"] == 3
    assert audit["clicks"]["native_bbox_present"] == 2
    assert audit["clicks"]["native_bbox_absent"] == 1
    assert audit["clicks"]["annotated_target_point_outside_bbox"] == 1
    evidence = {row["sample_id"]: row for row in audit["per_sample"]}
    assert evidence["missing"]["click_annotation_evidence"][0]["target_point_inside_annotated_bbox"] is None


def test_multi_action_lengths_types_and_ambiguous_same_input_labels(tmp_path):
    actions = [{"action_type": "click", "x": 3, "y": 5},
               {"action_type": "hotkey", "keys": ["ctrl", "a"]},
               {"action_type": "type_text", "text": "orange"}]
    one = sample(tmp_path, "one", actions=actions, action_annotations=[{}, {}, {}])
    two = sample(tmp_path, "two", image_path=one["image_path"])
    rows = loaded(tmp_path, "train.jsonl", [one, two])
    audit = audit_rows(rows)
    assert audit["sequence_length_counts"] == {"1": 1, "3": 1}
    assert audit["multi_action_samples"] == 1
    assert audit["action_type_counts"] == {"click": 2, "hotkey": 1, "type_text": 1}
    assert audit["identical_image_and_instruction_with_different_targets"] == 1
    conflict, = audit["input_label_conflicts_for_manual_review"]
    assert [item["sample_id"] for item in conflict["samples"]] == ["one", "two"]
    assert len({item["target_sha256"] for item in conflict["samples"]}) == 2


def test_fixed_selection_is_order_independent_result_independent_and_stratified(tmp_path):
    rows = [sample(tmp_path, f"click-{i}") for i in range(9)]
    rows += [sample(tmp_path, f"text-{i}", actions=[{"action_type": "type_text", "text": "x"}])
             for i in range(3)]
    first = select_diagnostic_rows(rows, seed=42, limit=4)
    second = select_diagnostic_rows(list(reversed(rows)), seed=42, limit=4)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert counter_kinds(first) == {"click", "type_text"}
    assert len(select_diagnostic_rows(rows, limit=0)) == len(rows)
    contaminated = [dict(row, generation="whatever", score=0) for row in rows]
    assert [row["sample_id"] for row in select_diagnostic_rows(contaminated, limit=4)] == [
        row["sample_id"] for row in first]


def counter_kinds(rows):
    return {action["action_type"] for row in rows for action in row["actions"]}


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_selection_limits_refused(tmp_path, limit):
    with pytest.raises(ValueError, match="integers"):
        select_diagnostic_rows([sample(tmp_path, "a")], limit=limit)


def raw_record(tmp_path, row, **changes):
    root = tmp_path / "raw_train"
    session = row["group_id"].split(":", 1)[1]
    path = root / session / "step.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_actions = []
    for action in row["actions"]:
        if action["action_type"] == "click":
            raw_actions.append({"action_type": "MouseAction", "mouse_action_type": "click",
                                "mouse_button": "left", "mouse_position": {
                                    "width": action["x"], "height": action["y"]}, "element": "Type SECRET"})
        else:
            raw_actions.append({"action_type": "KeyboardAction", "keyboard_action_type": "text",
                                "keyboard_text": action["text"], "element": "Type SECRET"})
    raw = {"session_id": session, "task_prompt_en": row["instruction"],
           "send_prompt": 'example [{"action_type":"KeyboardAction","keyboard_text":"SECRET"}]',
           "LLM_response_editer": 'CURRENT SECRET LABEL', "current_task": "Type SECRET",
           "actions": raw_actions}
    raw.update(changes)
    path.write_text(json.dumps(raw), encoding="utf-8")
    row["record_path"] = f"{session}\\step.json"
    return root, path


def test_raw_review_excludes_responses_prompt_action_description_and_unknown_fields(tmp_path):
    row = sample(tmp_path, "a", actions=[{"action_type": "type_text", "text": "SECRET"}])
    root, path = raw_record(tmp_path, row, future_action="LEAK_FUTURE")
    review = review_raw_fields([row], root)
    assert review["records"][0]["sha256"] == file_sha256(path)
    assert review["current_task_exactly_matches_action_element_records"] == 1
    assert review["field_action_schema_marker_counts"]["send_prompt"] == 1
    assert review["field_current_type_text_literal_mentions"]["send_prompt"] == 1
    assert review["unknown_fields_require_review"] == ["future_action"]
    assert review["model_inputs_exported"] is False
    text = json.dumps({key: value for key, value in review.items() if key != "goal_translation_examples"})
    assert "CURRENT SECRET LABEL" not in text
    assert "LEAK_FUTURE" not in text
    assert "Type SECRET" not in text
    policies = {field: item for item in review["field_policies"] for field in item["fields"]}
    assert not policies["send_prompt"]["current_model_input"]
    assert not policies["current_task"]["current_model_input"]
    assert not policies["LLM_response_editer"]["current_model_input"]


@pytest.mark.parametrize("change,reason", [
    ({"session_id": "another"}, "session provenance"),
    ({"task_prompt_en": "different goal"}, "instruction provenance"),
])
def test_raw_wrong_provenance_refused(tmp_path, change, reason):
    row = sample(tmp_path, "a")
    root, _ = raw_record(tmp_path, row, **change)
    with pytest.raises(ValueError, match=reason):
        review_raw_fields([row], root)


def test_raw_path_escape_refused(tmp_path):
    row = sample(tmp_path, "a")
    root, path = raw_record(tmp_path, row)
    outside = tmp_path / "outside.json"
    outside.write_bytes(path.read_bytes())
    row["record_path"] = "../outside.json"
    with pytest.raises(ValueError, match="escapes"):
        review_raw_fields([row], root)


def test_raw_changed_action_label_refused_even_when_goal_and_session_match(tmp_path):
    row = sample(tmp_path, "a", actions=[{"action_type": "type_text", "text": "SECRET"}])
    root, _ = raw_record(tmp_path, row, actions=[{
        "action_type": "KeyboardAction", "keyboard_action_type": "text", "keyboard_text": "DIFFERENT"}])
    with pytest.raises(ValueError, match="label provenance"):
        review_raw_fields([row], root)


def test_raw_test_directory_refused_without_reading_records(tmp_path):
    root = tmp_path / "test"
    root.mkdir()
    with pytest.raises(ValueError, match="test raw source"):
        review_raw_fields([], root)


@pytest.mark.parametrize("changed", ["code", "image", "data"])
def test_change_during_audit_refused_without_writing_evidence(tmp_path, monkeypatch, changed):
    from gui_agent.finetuning import diagnostics

    train, validation = pair(tmp_path)
    code = tmp_path / "source.py"
    code.write_text("version = 1", encoding="utf-8")
    real_audit = diagnostics.audit_rows
    altered = False

    def mutate_after_loading(rows):
        nonlocal altered
        result = real_audit(rows)
        if not altered:
            altered = True
            if changed == "code":
                code.write_text("version = 2", encoding="utf-8")
            elif changed == "image":
                Image.new("RGB", (32, 24), (0, 0, 0)).save(rows[0]["image_path"])
            else:
                train.write_text(train.read_text() + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(diagnostics, "audit_rows", mutate_after_loading)
    output = tmp_path / "new_audit"
    with pytest.raises(ValueError, match="changed during audit"):
        diagnose_dataset(train, validation, output, source_files=[code])
    assert not output.exists()


def test_original_english_literal_evidence_counts_are_per_action_and_split(tmp_path):
    train = sample(tmp_path, "a", actions=[{"action_type": "type_text", "text": "冯诺依曼"}],
                   instruction="Search for von Neumann", diagnostic_split="train")
    root, _ = raw_record(tmp_path, train, task_prompt="搜索冯诺依曼资料")
    validation = sample(tmp_path, "b", actions=[{"action_type": "type_text", "text": "orange"}],
                        instruction="Search orange", diagnostic_split="validation")
    raw_record(tmp_path, validation, task_prompt="搜索orange")
    review = review_raw_fields([train, validation], root)
    stats = review["goal_translation_lexical_evidence_by_split"]
    assert stats["train"]["type_text_actions"] == 1
    assert stats["train"]["english_goal_literal_present"] == 0
    assert stats["train"]["original_goal_literal_present"] == 1
    assert stats["train"]["either_goal_literal_present"] == 1
    assert stats["train"]["original_only_literal_present"] == 1
    assert stats["validation"]["english_goal_literal_present"] == 1
    assert stats["validation"]["original_goal_literal_present"] == 1
    assert review["scoring_side_examples_contain_labels_not_safe_prompt_payload"] is True
    assert {example["category"] for example in review["goal_translation_examples"]} == {"both", "original_only"}
    assert review["goal_translation_examples"] == review_raw_fields([validation, train], root)["goal_translation_examples"]


@pytest.mark.parametrize("original,english,expected", [
    ("搜索冯诺依曼", "Search for von Neumann", "Original task: 搜索冯诺依曼\nEnglish task: Search for von Neumann"),
    (" same ", "same", "same"),
    (None, "English only", "English only"),
    ("Original only", " ", "Original only"),
])
def test_bilingual_v2_policy_uses_two_goals_only(original, english, expected):
    assert bilingual_goal_input(original, english) == expected


def test_bilingual_v2_never_fills_missing_goals_from_labels():
    with pytest.raises(ValueError, match="task goal"):
        bilingual_goal_input(None, " ")
    with pytest.raises(TypeError, match="strings"):
        bilingual_goal_input({"actions": "secret label"}, "goal")


def test_freeze_all_validation_train_cap_and_hashes_no_test_consumption(tmp_path):
    train, validation = pair(tmp_path)
    # An unusable test file is deliberately present and must not affect this audit.
    (tmp_path / "test.jsonl").write_text("NOT JSON", encoding="utf-8")
    output = tmp_path / "audit"
    manifest = diagnose_dataset(train, validation, output, train_limit=2,
                                source_files=[Path(__file__)])
    assert manifest["selected_counts"] == {"train": 2, "validation": 3}
    assert manifest["test_data_consumed"] is False
    assert manifest["input_data"]["train"]["sha256"] == file_sha256(train)
    assert manifest["source_files"][0]["sha256"] == file_sha256(__file__)
    for item in manifest["files"]:
        assert item["sha256"] == file_sha256(output / item["path"])
    selection = json.loads((output / "selection.json").read_text())
    assert selection["input_data_sha256"]["validation"] == file_sha256(validation)
    assert len(selection["samples"]) == 5
    assert all("actions" not in row and "instruction" not in row for row in selection["samples"])
    second = diagnose_dataset(train, validation, tmp_path / "audit2", train_limit=2,
                              frozen_selection=output / "selection.json")
    assert second["selection_sha256"] == manifest["selection_sha256"]
    with pytest.raises(ValueError, match="not empty"):
        diagnose_dataset(train, validation, output)


def test_frozen_selection_refuses_target_change_even_same_sample_and_image(tmp_path):
    train, validation = pair(tmp_path)
    output = tmp_path / "audit"
    diagnose_dataset(train, validation, output)
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    rows[0]["actions"][0]["x"] += 1
    dataset(tmp_path, "train.jsonl", rows)
    with pytest.raises(ValueError, match="frozen selection differs"):
        diagnose_dataset(train, validation, tmp_path / "new", frozen_selection=output / "selection.json")
    assert not (tmp_path / "new").exists()


def test_frozen_selection_refuses_pixel_change_even_same_size_and_filename(tmp_path):
    train, validation = pair(tmp_path)
    output = tmp_path / "audit"
    diagnose_dataset(train, validation, output)
    row = json.loads(train.read_text().splitlines()[0])
    Image.new("RGB", (32, 24), (0, 0, 0)).save(row["image_path"])
    with pytest.raises(ValueError, match="frozen selection differs"):
        diagnose_dataset(train, validation, tmp_path / "new", frozen_selection=output / "selection.json")


def test_cli_cpu_audit_and_no_test_flag(tmp_path):
    train, validation = pair(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "week5_diagnose.py"
    result = subprocess.run([sys.executable, str(script), "--train", str(train),
                             "--validation", str(validation), "--output-dir", str(tmp_path / "cli")],
                            capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["selected_counts"] == {"train": 4, "validation": 3}
    help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True,
                                 text=True, encoding="utf-8", timeout=30, check=False)
    assert "--test" not in help_result.stdout
    assert "--validation-limit" in help_result.stdout


def test_unknown_or_extra_target_fields_are_refused_not_partially_scored(tmp_path):
    row = sample(tmp_path, "a")
    row["actions"] += [{"action_type": "click", "x": 2, "y": 2, "text": "leak"}]
    row["action_annotations"] += [{}]
    with pytest.raises(ValueError, match="extra_action_fields"):
        loaded(tmp_path, "train.jsonl", [row])


def test_value_hash_preserves_array_order_and_exact_literal_but_not_object_order():
    one = [{"action_type": "type_text", "text": " exact "}, {"action_type": "key_press", "key": "enter"}]
    reordered = [{"text": " exact ", "action_type": "type_text"}, one[1]]
    assert value_sha256(one) == value_sha256(reordered)
    assert value_sha256(one) != value_sha256(list(reversed(one)))
    changed = copy.deepcopy(one)
    changed[0]["text"] = changed[0]["text"].strip()
    assert value_sha256(one) != value_sha256(changed)
