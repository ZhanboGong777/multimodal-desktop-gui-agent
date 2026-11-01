"""CPU-only supervision and split invariants; no downloaded data is required."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning.data import (
    DataPreparationError,
    SourceSpec,
    action_target_json,
    prepare_dataset,
    to_gui_action_steps,
    validate_actions,
)


def click(x=12, y=9):
    return {"action_type": "MouseAction", "mouse_action_type": "click",
            "mouse_button": "left", "mouse_position": {"width": x, "height": y}}


def record(root, session, name, *, color=(12, 24, 36), actions=None, **overrides):
    folder = root / session
    (folder / "images").mkdir(parents=True, exist_ok=True)
    image = folder / "images" / f"{name}.png"
    Image.new("RGB", (1024, 768), color).save(image)
    payload = {"saved_image_name": image.name, "task_prompt_en": "Click the field",
               "video_width": 1024, "video_height": 768,
               "actions": [click()] if actions is None else actions,
               "send_prompt": "SECRET HISTORY AND LABELS MUST NOT BE TRAINED"}
    payload.update(overrides)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def rows(output, split=None):
    paths = [output / f"{split}.jsonl"] if split else list(output.glob("*.jsonl"))
    return [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines()]


def prepare(root, output, **kwargs):
    return prepare_dataset([SourceSpec("screenagent", "train", root)], output, **kwargs)


def test_real_width_height_coordinates_text_and_annotations(tmp_path):
    raw = tmp_path / "raw"
    mouse = click(487, 193)
    mouse["clickable_area"] = {"upper_left_position": {"width": 450, "height": 170},
                               "lower_right_position": {"width": 530, "height": 220}}
    text = "  冯诺依曼\n "
    record(raw, "session", "step", actions=[
        {"action_type": "PlanAction", "plan": "hidden plan"}, mouse,
        {"action_type": "KeyboardAction", "keyboard_action_type": "text", "keyboard_text": text},
        {"action_type": "KeyboardAction", "keyboard_action_type": "press", "keyboard_key": "Return"},
        {"action_type": "EvaluateSubTaskAction", "evaluation": "hidden judgement"},
    ])
    output = tmp_path / "prepared"
    manifest = prepare(raw, output)
    sample, = rows(output)
    assert sample["actions"] == [{"action_type": "click", "x": 487, "y": 193},
                                 {"action_type": "type_text", "text": text},
                                 {"action_type": "key_press", "key": "enter"}]
    assert sample["action_annotations"] == [{"target_bbox": {"left": 450, "top": 170,
                                                               "right": 530, "bottom": 220}}, {}, {}]
    assert (sample["image_width"], sample["image_height"]) == (1024, 768)
    assert sample["image_sha256"] == hashlib.sha256(
        b"1024x768:" + Image.new("RGB", (1024, 768), (12, 24, 36)).tobytes()).hexdigest()
    assert json.loads(action_target_json(sample)) == sample["actions"]
    assert "hidden" not in action_target_json(sample)
    assert "target_bbox" not in action_target_json(sample)
    assert "send_prompt" not in sample
    steps = to_gui_action_steps(sample["actions"])
    assert steps[0].coordinates.x == 487
    assert steps[1].input_text == text
    assert manifest["retained_samples"] == 1
    assert manifest["ready_for_training"] is False  # one session cannot make an independent validation split


@pytest.mark.parametrize("action", [
    {"action_type": "click", "x": 0, "y": 0},
    {"action_type": "double_click", "x": 31, "y": 23},
    {"action_type": "right_click", "x": 1, "y": 2},
    {"action_type": "move", "x": 1, "y": 2},
    {"action_type": "scroll", "x": 1, "y": 2, "scroll_amount": -3},
    {"action_type": "drag", "start": {"x": 1, "y": 2}, "end": {"x": 31, "y": 23}},
    {"action_type": "type_text", "text": "  exact\n "},
    {"action_type": "key_press", "key": "enter"},
    {"action_type": "hotkey", "keys": ["ctrl", "a"]},
    {"action_type": "wait", "duration": 0.5},
])
def test_complete_canonical_actions_validate(action):
    assert validate_actions([action], 32, 24) == [action]


@pytest.mark.parametrize("action,reason", [
    ({"action_type": "click", "x": True, "y": 2}, "invalid_integer"),
    ({"action_type": "click", "x": 1.0, "y": 2}, "invalid_integer"),
    ({"action_type": "click", "x": 32, "y": 2}, "coordinates_outside_image"),
    ({"action_type": "click", "x": -1, "y": 2}, "coordinates_outside_image"),
    ({"action_type": "click", "x": 1}, "incomplete_or_extra_action_fields"),
    ({"action_type": "click", "x": 1, "y": 2, "text": "label"}, "incomplete_or_extra_action_fields"),
    ({"action_type": "scroll", "x": 1, "y": 2, "scroll_amount": 0}, "empty_scroll"),
    ({"action_type": "type_text", "text": " \n "}, "empty_keyboard_text"),
    ({"action_type": "key_press", "key": "unknown_key"}, "unsupported_keyboard_key"),
    ({"action_type": "hotkey", "keys": ["ctrl"]}, "incomplete_hotkey"),
    ({"action_type": "wait", "duration": True}, "invalid_wait_duration"),
    ({"action_type": "wait", "duration": float("nan")}, "invalid_wait_duration"),
    ({"action_type": "wait", "duration": 0}, "invalid_wait_duration"),
    ({"action_type": "drag", "start": {"x": 1, "y": 2, "label": "private"},
      "end": {"x": 3, "y": 4}}, "incomplete_or_extra_point_fields"),
    ({"action_type": "plan", "text": "invent an action"}, "unsupported_action_type"),
])
def test_invalid_canonical_actions_never_coerce(action, reason):
    with pytest.raises(DataPreparationError, match=reason):
        validate_actions([action], 32, 24)


@pytest.mark.parametrize("actions,overrides,reason", [
    ([], {}, "no_executable_actions"),
    ([{"action_type": "PlanAction"}], {}, "no_executable_actions"),
    ([click(), {"action_type": "MouseAction", "mouse_action_type": "drag",
                "mouse_position": {"width": 40, "height": 50}}], {}, "unsupported_or_incomplete_mouse_action"),
    ([{"action_type": "KeyboardAction", "keyboard_action_type": "down", "keyboard_key": "Control_L"}], {},
     "unsupported_or_incomplete_keyboard_action"),
    ([click(1024, 0)], {}, "coordinates_outside_image"),
    ([click()], {"video_width": 1023}, "declared_image_dimensions_mismatch"),
    ([click()], {"video_height": "768"}, "invalid_integer:video_height"),
    ([click()], {"saved_image_name": None}, "missing_screenshot"),
    ([click()], {"saved_image_name": "../elsewhere.png"}, "unsafe_image_reference"),
    ([click()], {"task_prompt_en": "", "task_prompt": None}, "missing_instruction"),
])
def test_bad_source_labels_are_accounted_for_without_partial_action_lists(tmp_path, actions, overrides, reason):
    raw = tmp_path / "raw"
    record(raw, "session", "invalid", actions=actions, **overrides)
    output = tmp_path / "prepared"
    manifest = prepare(raw, output)
    assert rows(output) == []
    assert manifest["input_records"] == manifest["excluded_samples"] == 1
    assert manifest["exclusion_counts"] == {reason: 1}
    assert manifest["ready_for_heldout_comparison"] is False


def test_negative_variants_and_broken_images_do_not_enter_supervision(tmp_path):
    raw = tmp_path / "raw"
    record(raw, "session", "step_translate_neg_plan")
    path = record(raw, "session", "corrupt")
    (path.parent / "images" / "corrupt.png").write_bytes(b"not image pixels")
    record(raw, "session", "explicit_negative", is_negative=True)
    manifest = prepare(raw, tmp_path / "prepared")
    assert manifest["exclusion_counts"] == {"negative_variant": 2, "invalid_screenshot": 1}


def test_cross_session_duplicates_and_shared_images_cannot_leak(tmp_path):
    raw = tmp_path / "raw"
    record(raw, "a", "shared", color=(1, 2, 3), actions=[click(10, 10)])
    record(raw, "b", "shared", color=(1, 2, 3), actions=[click(20, 20)])
    record(raw, "a", "duplicate", color=(1, 2, 3), actions=[click(10, 10)])
    record(raw, "a", "unique", color=(10, 20, 30))
    record(raw, "b", "unique", color=(11, 21, 31))
    record(raw, "c", "independent", color=(12, 22, 32))
    output = tmp_path / "prepared"
    manifest = prepare(raw, output, validation_fraction=0.5)
    assignments = manifest["group_assignments"]
    assert assignments["screenagent:a"] == assignments["screenagent:b"]
    assert assignments["screenagent:c"] != assignments["screenagent:a"]
    assert manifest["exclusion_counts"] == {"duplicate_image_actions": 1}
    train, validation = rows(output, "train"), rows(output, "validation")
    for field in ("group_id", "image_sha256"):
        assert not ({row[field] for row in train} & {row[field] for row in validation})
    assert manifest["ready_for_training"] is True


def test_official_test_overlap_removes_entire_training_trajectory(tmp_path):
    train, test = tmp_path / "raw_train", tmp_path / "raw_test"
    record(train, "overlap", "same_pixels", color=(1, 2, 3))
    record(train, "overlap", "other_step", color=(4, 5, 6))
    record(train, "same_id", "other_pixels", color=(8, 9, 10))
    record(train, "safe_a", "step", color=(11, 12, 13))
    record(train, "safe_b", "step", color=(14, 15, 16))
    record(test, "test_session", "official", color=(1, 2, 3))
    record(test, "same_id", "official", color=(17, 18, 19))
    output = tmp_path / "prepared"
    manifest = prepare_dataset([SourceSpec("screenagent", "train", train),
                                SourceSpec("screenagent", "test", test)], output)
    assert manifest["exclusion_counts"] == {"official_test_overlap_trajectory": 3}
    assert len(rows(output, "test")) == 2
    assert {row["source_split"] for row in rows(output, "test")} == {"test"}
    assert {row["source_split"] for row in rows(output, "validation")} == {"train"}
    assert not {"screenagent:overlap", "screenagent:same_id"} & set(manifest["group_assignments"])
    assert manifest["ready_for_heldout_comparison"] is True


def test_validation_source_overlap_is_protected_at_trajectory_level(tmp_path):
    train, validation = tmp_path / "raw_train", tmp_path / "raw_val"
    record(train, "overlap", "shared", color=(1, 2, 3))
    record(train, "overlap", "private", color=(4, 5, 6))
    record(train, "safe", "private", color=(7, 8, 9))
    record(validation, "heldout", "shared", color=(1, 2, 3))
    output = tmp_path / "prepared"
    manifest = prepare_dataset([SourceSpec("screenagent", "train", train),
                                SourceSpec("screenagent", "validation", validation)], output)
    assert manifest["exclusion_counts"] == {"heldout_overlap_trajectory": 2}
    assert {row["group_id"] for row in rows(output, "train")} == {"screenagent:safe"}
    assert {row["source_split"] for row in rows(output, "validation")} == {"validation"}


def test_fixed_seed_assignments_are_independent_of_input_source_order(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for number in range(6):
        record(first if number % 2 else second, f"session{number}", "step", color=(number, 20, 30))
    specs = [SourceSpec("screenagent", "train", first), SourceSpec("screenagent", "train", second)]
    one = prepare_dataset(specs, tmp_path / "one", seed=113)
    two = prepare_dataset(list(reversed(specs)), tmp_path / "two", seed=113)
    assert one["group_assignments"] == two["group_assignments"]
    assert {row["sample_id"] for row in rows(tmp_path / "one")} == {row["sample_id"] for row in rows(tmp_path / "two")}


def test_unified_screenagent_uses_unified_fields_not_raw_parser(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 24), (1, 2, 3)).save(image)
    payload = {"dataset_name": "screenagent", "sample_id": "old_id", "instruction": "Enter exact text",
               "image_path": str(image), "metadata": {"session_id": "actual_session"},
               "actions": [{"step_index": 0, "action_type": "type_text", "input_text": "  text\n",
                            "raw_action": {"private_label": "ignored"}}]}
    source = tmp_path / "unified.jsonl"
    source.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "prepared"
    manifest = prepare_dataset([SourceSpec("unified", "train", source)], output)
    sample, = rows(output)
    assert sample["source"] == "screenagent"
    assert sample["group_id"] == "screenagent:actual_session"
    assert sample["actions"] == [{"action_type": "type_text", "text": "  text\n"}]
    assert manifest["retained_samples"] == 1


def test_webarena_intent_is_never_invented_as_action(tmp_path):
    source = tmp_path / "webarena.json"
    source.write_text(json.dumps({"dataset_name": "webarena", "instruction": "Click submit",
                                   "actions": [{"action_type": "click", "x": 1, "y": 2}]}), encoding="utf-8")
    manifest = prepare_dataset([SourceSpec("unified", "train", source)], tmp_path / "prepared")
    assert manifest["exclusion_counts"] == {"webarena_has_no_recorded_actions": 1}
    assert manifest["retained_samples"] == 0


def test_zip_native_layout_and_provenance_hashes(tmp_path):
    raw = tmp_path / "raw"
    record(raw, "real_session", "step")
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in raw.rglob("*"):
            if path.is_file():
                handle.write(path, "test/" + str(path.relative_to(raw)).replace("\\", "/"))
    output = tmp_path / "prepared"
    manifest = prepare_dataset([SourceSpec("screenagent", "test", archive, "official-url", "pinned-sha")], output)
    sample, = rows(output, "test")
    assert sample["group_id"] == "screenagent:real_session"
    assert sample["source_split"] == "test"
    source, = manifest["sources"]
    assert source["origin_url"] == "official-url"
    assert source["revision"] == "pinned-sha"
    assert source["files"][0]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    for artifact in manifest["outputs"].values():
        assert artifact["sha256"] == hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()


def test_malformed_records_are_not_silently_dropped(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text('{"instruction":"ok"}\nnot JSON\nnull\n', encoding="utf-8")
    manifest = prepare_dataset([SourceSpec("unified", "train", source)], tmp_path / "prepared")
    assert manifest["input_records"] == manifest["excluded_samples"] == 3
    assert manifest["exclusion_counts"] == {"missing_trajectory_id": 1, "invalid_json": 1, "invalid_record": 1}


def test_existing_output_is_preserved_and_missing_source_is_clear(tmp_path):
    raw = tmp_path / "raw"
    record(raw, "session", "step")
    output = tmp_path / "prepared"
    output.mkdir()
    original = output / "keep.txt"
    original.write_text("keep evidence", encoding="utf-8")
    with pytest.raises(DataPreparationError, match="not empty"):
        prepare(raw, output)
    assert original.read_text(encoding="utf-8") == "keep evidence"
    with pytest.raises(DataPreparationError, match="source does not exist"):
        prepare(tmp_path / "missing", tmp_path / "new")


def test_bbox_without_true_area_is_absent_and_invalid_bbox_excluded(tmp_path):
    raw = tmp_path / "raw"
    record(raw, "valid", "step")
    mouse = click()
    mouse["clickable_area"] = {"upper_left_position": {"width": 0, "height": 0},
                               "lower_right_position": {"width": 1025, "height": 768}}
    record(raw, "invalid", "step", actions=[mouse], color=(1, 2, 3))
    output = tmp_path / "prepared"
    manifest = prepare(raw, output)
    sample, = rows(output)
    assert sample["action_annotations"] == [{}]
    assert manifest["exclusion_counts"] == {"invalid_bbox_annotation": 1}


def test_complete_source_keyboard_scroll_right_click_and_wait_mapping(tmp_path):
    raw = tmp_path / "raw"
    right = click()
    right["mouse_button"] = "right"
    record(raw, "session", "step", actions=[
        right,
        {"action_type": "KeyboardAction", "keyboard_action_type": "press", "keyboard_key": ["Return"]},
        {"action_type": "KeyboardAction", "keyboard_action_type": "press", "keyboard_key": "Control_L+a"},
        {"action_type": "MouseAction", "mouse_action_type": "scroll_down", "scroll_repeat": 3,
         "mouse_position": {"width": 10, "height": 20}},
        {"action_type": "WaitAction", "wait_time": 0.5},
    ])
    output = tmp_path / "prepared"
    prepare(raw, output)
    sample, = rows(output)
    assert sample["actions"] == [
        {"action_type": "right_click", "x": 12, "y": 9},
        {"action_type": "key_press", "key": "enter"},
        {"action_type": "hotkey", "keys": ["ctrl", "a"]},
        {"action_type": "scroll", "x": 10, "y": 20, "scroll_amount": -3},
        {"action_type": "wait", "duration": 0.5},
    ]


def test_cli_reports_unready_single_session_and_refuses_overwrite(tmp_path):
    raw = tmp_path / "raw"
    record(raw, "session", "step")
    output = tmp_path / "prepared"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "week5_prepare_data.py"),
               "--source", "screenagent", "train", str(raw), "--output-dir", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    summary = json.loads(result.stdout)
    assert summary["splits"] == {"train": 1, "validation": 0, "test": 0}
    assert summary["ready_for_training"] is False
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "not empty" in result.stderr


def test_equivalent_pixels_and_reordered_action_keys_are_duplicates(tmp_path):
    png, bmp = tmp_path / "image.png", tmp_path / "image.bmp"
    image = Image.new("RGB", (32, 24), (1, 2, 3))
    image.save(png)
    image.save(bmp)
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        {"dataset_name": "screenagent", "group_id": "one", "instruction": "Click field",
         "image_path": str(png), "actions": [{"action_type": "click", "x": 1, "y": 2}]},
        {"dataset_name": "screenagent", "group_id": "two", "instruction": "Another wording",
         "image_path": str(bmp), "actions": [{"y": 2, "x": 1, "action_type": "click"}]},
    ]), encoding="utf-8")
    output = tmp_path / "prepared"
    manifest = prepare_dataset([SourceSpec("unified", "train", source)], output)
    assert manifest["retained_samples"] == 1
    assert manifest["exclusion_counts"] == {"duplicate_image_actions": 1}
    assert len(rows(output)) == 1
