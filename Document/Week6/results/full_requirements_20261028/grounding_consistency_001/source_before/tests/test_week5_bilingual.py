"""Independent V2 task inputs, unchanged supervision and no test/model access."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning.prompts import build_user_prompt

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "week5_prepare_bilingual.py"
spec = importlib.util.spec_from_file_location("week5_prepare_bilingual", SCRIPT)
bilingual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bilingual)


def source_sample(tmp_path, name, *, original="搜索冯诺依曼", english="Search for von Neumann",
                  text="冯诺依曼", **raw_changes):
    folder = tmp_path / "raw_train" / name
    folder.mkdir(parents=True, exist_ok=True)
    raw = {"session_id": name, "task_prompt": original, "task_prompt_en": english,
           "current_task": "CURRENT ACTION SECRET", "send_prompt": "HISTORY SECRET",
           "LLM_response_editer": "CORRECTED LABEL SECRET",
           "actions": [{"element": "ACTION DESCRIPTION SECRET", "keyboard_text": text}]}
    raw.update(raw_changes)
    (folder / "step.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    image = tmp_path / f"{name}.png"
    Image.new("RGB", (32, 24), (len(name), ord(name[0]), 45)).save(image)
    row = {"sample_id": f"screenagent:{name}:step", "group_id": f"screenagent:{name}",
           "source": "screenagent", "source_split": "train", "instruction": (english or original).strip(),
           "image_path": str(image), "image_width": 32, "image_height": 24,
           "actions": [{"action_type": "type_text", "text": text}], "action_annotations": [{}],
           "record_path": f"{name}\\step.json", "additional_original_metadata": {"keep": True}}
    return row


def write_data(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def pair(tmp_path):
    one, two = source_sample(tmp_path, "train-a"), source_sample(tmp_path, "val-b")
    return write_data(tmp_path, "train.jsonl", [one]), write_data(tmp_path, "validation.jsonl", [two])


def run(tmp_path, train=None, validation=None, output="v2", **kwargs):
    if train is None:
        train, validation = pair(tmp_path)
    return bilingual.prepare_bilingual(train, validation, tmp_path / "raw_train", tmp_path / output, **kwargs)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_only_instruction_and_explicit_source_metadata_change(tmp_path):
    train, validation = pair(tmp_path)
    before = {path: path.read_bytes() for path in (train, validation)}
    images = {row["image_path"]: Path(row["image_path"]).read_bytes()
              for path in (train, validation) for row in read_rows(path)}
    (tmp_path / "test.jsonl").write_text("UNREADABLE TEST LABELS", encoding="utf-8")
    manifest = run(tmp_path, train, validation, source_files=[SCRIPT])
    for split, path in (("train", train), ("validation", validation)):
        original, = read_rows(path)
        new, = read_rows(tmp_path / "v2" / f"{split}.jsonl")
        metadata = new.pop("instruction_source")
        assert new == {**original, "instruction": "Original task: 搜索冯诺依曼\nEnglish task: Search for von Neumann"}
        assert metadata["fields"] == ["task_prompt", "task_prompt_en"]
        assert metadata["raw_record_sha256"] == bilingual.file_sha256(
            tmp_path / "raw_train" / original["group_id"].split(":")[1] / "step.json")
        assert manifest["outputs"][split]["instruction_changed_samples"] == 1
        assert before[path] == path.read_bytes()
    assert all(Path(path).read_bytes() == content for path, content in images.items())
    assert manifest["test_data_consumed"] is False
    assert manifest["images_copied_or_modified"] is False
    assert manifest["model_or_processor_loaded"] is False
    assert manifest["token_length_preflight"] == {
        **manifest["token_length_preflight"], "performed": False, "max_seq_length": 1536,
        "required_before_training": True}


def test_label_and_action_description_changes_cannot_change_instruction(tmp_path):
    train, validation = pair(tmp_path)
    run(tmp_path, train, validation, output="first")
    rows = read_rows(train)
    rows[0]["actions"] = [{"action_type": "key_press", "key": "enter"}]
    write_data(tmp_path, "train.jsonl", rows)
    raw_file = tmp_path / "raw_train" / "train-a" / "step.json"
    raw = json.loads(raw_file.read_text(encoding="utf-8"))
    raw.update(actions=[{"element": "OTHER TARGET LABEL"}], current_task="FUTURE SECRET",
               send_prompt="EXPOSE TARGET JSON", LLM_response_editer="UNKNOWN CURRENT LABEL")
    raw_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    run(tmp_path, train, validation, output="second")
    one, = read_rows(tmp_path / "first" / "train.jsonl")
    two, = read_rows(tmp_path / "second" / "train.jsonl")
    assert one["instruction"] == two["instruction"]
    assert one["actions"] != two["actions"]
    assert "SECRET" not in build_user_prompt(two)
    assert "OTHER TARGET LABEL" not in build_user_prompt(two)
    assert "instruction_source" not in build_user_prompt(two)


def test_literal_match_changes_do_not_control_included_task_fields(tmp_path):
    train_row = source_sample(tmp_path, "train-a", text="COMPLETELY ABSENT")
    val_row = source_sample(tmp_path, "val-b", text="冯诺依曼")
    train = write_data(tmp_path, "train.jsonl", [train_row])
    validation = write_data(tmp_path, "validation.jsonl", [val_row])
    manifest = run(tmp_path, train, validation)
    one, = read_rows(tmp_path / "v2" / "train.jsonl")
    two, = read_rows(tmp_path / "v2" / "validation.jsonl")
    assert one["instruction"] == two["instruction"]
    assert manifest["literal_lexical_evidence"]["train"]["new_bilingual_instruction_literal_present"] == 0
    assert manifest["literal_lexical_evidence"]["validation"]["new_bilingual_instruction_literal_present"] == 1


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json", "C:\\outside.json", "train-a/step.txt"])
def test_unsafe_raw_lookup_refused_before_output(tmp_path, path):
    train, validation = pair(tmp_path)
    rows = read_rows(train)
    rows[0]["record_path"] = path
    write_data(tmp_path, "train.jsonl", rows)
    with pytest.raises(ValueError, match="unsafe source"):
        run(tmp_path, train, validation)
    assert not (tmp_path / "v2").exists()


@pytest.mark.parametrize("change,reason", [
    ({"session_id": "wrong"}, "session/group"),
    ({"task_prompt_en": "different task"}, "instruction provenance"),
    ({"task_prompt": ["wrong type"]}, "strings"),
])
def test_raw_provenance_or_goal_type_refused(tmp_path, change, reason):
    train, validation = pair(tmp_path)
    path = tmp_path / "raw_train" / "train-a" / "step.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(change)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=reason):
        run(tmp_path, train, validation)


def test_test_path_refused_before_hash_or_read(tmp_path, monkeypatch):
    def no_hash(path):
        pytest.fail(f"should not hash or read official test: {path}")
    monkeypatch.setattr(bilingual, "file_sha256", no_hash)
    with pytest.raises(ValueError, match="before reading"):
        bilingual.prepare_bilingual(tmp_path / "test.jsonl", tmp_path / "validation.jsonl",
                                     tmp_path / "missing_raw", tmp_path / "v2")


def test_mislabeled_test_rows_refused(tmp_path):
    train, validation = pair(tmp_path)
    rows = read_rows(validation)
    rows[0]["source_split"] = "test"
    write_data(tmp_path, "validation.jsonl", rows)
    with pytest.raises(ValueError, match="test/unknown"):
        run(tmp_path, train, validation)


def test_empty_output_and_prior_v2_metadata_are_refused(tmp_path):
    train, validation = pair(tmp_path)
    run(tmp_path, train, validation)
    with pytest.raises(ValueError, match="not empty"):
        run(tmp_path, train, validation)
    rows = read_rows(train)
    rows[0]["instruction_source"] = {"version": "already v2"}
    write_data(tmp_path, "train.jsonl", rows)
    with pytest.raises(ValueError, match="already has"):
        run(tmp_path, train, validation, output="another")


def test_source_audit_hashes_and_exact_sample_set_verified(tmp_path):
    train, validation = pair(tmp_path)
    records = []
    for path in (train, validation):
        row, = read_rows(path)
        source = tmp_path / "raw_train" / row["record_path"].replace("\\", "/")
        records.append({"sample_id": row["sample_id"], "record_path": row["record_path"],
                        "sha256": bilingual.file_sha256(source)})
    audit = tmp_path / "raw_field_review.json"
    audit.write_text(json.dumps({"records": records}), encoding="utf-8")
    manifest = run(tmp_path, train, validation, source_audit=audit)
    assert manifest["source_audit"]["sha256"] == bilingual.file_sha256(audit)
    records[0]["sha256"] = "0" * 64
    audit.write_text(json.dumps({"records": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source audit"):
        run(tmp_path, train, validation, output="another", source_audit=audit)
    records.pop()
    audit.write_text(json.dumps({"records": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly these"):
        run(tmp_path, train, validation, output="another", source_audit=audit)


def test_all_original_actions_and_native_annotations_preserved(tmp_path):
    train, validation = pair(tmp_path)
    rows = read_rows(train)
    rows[0]["actions"] = [{"action_type": "click", "x": 3, "y": 5},
                          {"action_type": "type_text", "text": " exact \n"}]
    rows[0]["action_annotations"] = [{"target_bbox": {"left": 1, "top": 2, "right": 8, "bottom": 9}}, {}]
    write_data(tmp_path, "train.jsonl", rows)
    run(tmp_path, train, validation)
    new, = read_rows(tmp_path / "v2" / "train.jsonl")
    assert new["actions"] == rows[0]["actions"]
    assert new["action_annotations"] == rows[0]["action_annotations"]


def test_duplicate_raw_goal_keys_refused(tmp_path):
    train, validation = pair(tmp_path)
    path = tmp_path / "raw_train" / "train-a" / "step.json"
    path.write_text('{"session_id":"train-a","task_prompt":"first","task_prompt":"second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        run(tmp_path, train, validation)


def test_cli_cpu_generation_counts_and_preflight_not_claimed(tmp_path):
    train, validation = pair(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), "--train", str(train),
                             "--validation", str(validation), "--raw-train-root", str(tmp_path / "raw_train"),
                             "--output-dir", str(tmp_path / "cli")], check=False,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["outputs"]["train"]["samples"] == 1
    assert report["tokenized"] is False
    assert report["test_data_consumed"] is False
