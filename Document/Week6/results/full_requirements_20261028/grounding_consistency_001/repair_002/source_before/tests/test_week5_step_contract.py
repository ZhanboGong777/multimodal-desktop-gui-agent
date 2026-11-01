"""Original-goal source provenance, unchanged targets and non-oracle model input."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning import step_contract
from gui_agent.finetuning.prompts import build_user_prompt


def fixture_pair(tmp_path):
    files = []
    for name, color in [("a", (10, 20, 30)), ("b", (40, 50, 60))]:
        folder = tmp_path / "raw_train" / name
        folder.mkdir(parents=True)
        raw = {"session_id": name, "task_prompt": '输入“Angemon”',
               "task_prompt_en": 'Enter "Angel"', "current_task": "TEACHER_STEP_SECRET",
               "send_prompt": "TEACHER_HISTORY_SECRET", "LLM_response_editer": "LABEL_REPLY_SECRET",
               "actions": [{"action_type": "KeyboardAction", "keyboard_action_type": "input",
                            "keyboard_text": "Angemon"},
                           {"action_type": "KeyboardAction", "keyboard_action_type": "press",
                            "keyboard_key": "Ctrl+s"}]}
        (folder / "step.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (32, 24), color).save(image)
        row = {"sample_id": f"screenagent:{name}:step", "group_id": f"screenagent:{name}",
               "source": "screenagent", "source_split": "train", "instruction": raw["task_prompt_en"],
               "image_path": str(image), "image_width": 32, "image_height": 24,
               "record_path": f"{name}/step.json", "arbitrary_original_metadata": {"keep": True},
               "actions": [{"action_type": "type_text", "text": "Angemon"},
                           {"action_type": "hotkey", "keys": ["ctrl", "s"]}],
               "action_annotations": [{}, {}]}
        path = tmp_path / ("train.jsonl" if name == "a" else "validation.jsonl")
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        files.append(path)
    return files


def prepare(tmp_path, files=None, output="derived"):
    train, validation = files or fixture_pair(tmp_path)
    return step_contract.prepare_original_goals(train, validation, tmp_path / "raw_train", tmp_path / output)


@pytest.mark.parametrize("original,english,fallback,expected", [
    ('  输入“Angemon”\n', 'Enter "Angel"', None, ('  输入“Angemon”\n', "task_prompt")),
    ("", "English only", None, ("English only", "task_prompt_en")),
    (None, "", "fallback", ("fallback", "instruction")),
    ("goal", "translated goal", "another goal", ("goal", "task_prompt")),
])
def test_source_priority_preserves_original_literal_without_translation(original, english, fallback, expected):
    assert step_contract.original_goal_input(original, english, fallback) == expected


@pytest.mark.parametrize("values", [([], "goal", None), ("goal", 42, None), (None, None, {})])
def test_invalid_source_field_types_are_not_silently_coerced(values):
    with pytest.raises(TypeError):
        step_contract.original_goal_input(*values)


def test_empty_source_goals_refused():
    with pytest.raises(ValueError, match="nonempty"):
        step_contract.original_goal_input("  ", None, "\n")


def test_conversion_preserves_every_target_identity_and_image(tmp_path):
    files = fixture_pair(tmp_path)
    original_bytes = {path: path.read_bytes() for path in files}
    images = {path: path.read_bytes() for path in tmp_path.glob("*.png")}
    manifest = prepare(tmp_path, files)
    assert manifest["outputs"]["train"]["samples"] == 1
    assert manifest["labels_preserved_and_raw_verified"] and not manifest["token_length_preflight_performed"]
    assert not manifest["model_or_processor_loaded"] and not manifest["test_data_consumed"]
    for split, path in zip(("train", "validation"), files, strict=True):
        old = json.loads(path.read_text(encoding="utf-8"))
        new = json.loads((tmp_path / "derived" / f"{split}.jsonl").read_text(encoding="utf-8"))
        assert {k: v for k, v in new.items() if k not in {"instruction", "instruction_source"}} == {
            k: v for k, v in old.items() if k != "instruction"}
        assert new["instruction"] == '输入“Angemon”'
        assert new["instruction_source"]["selected_field"] == "task_prompt"
        prompt = build_user_prompt(new, "current_step")
        assert '输入“Angemon”' in prompt
        assert all(secret not in prompt for secret in ["TEACHER_STEP_SECRET", "TEACHER_HISTORY_SECRET",
                                                       "LABEL_REPLY_SECRET", "instruction_source"])
    assert all(path.read_bytes() == content for path, content in original_bytes.items())
    assert all(path.read_bytes() == content for path, content in images.items())


def test_teacher_context_and_current_action_changes_cannot_choose_goal(tmp_path):
    files = fixture_pair(tmp_path)
    prepare(tmp_path, files, output="first")
    raw_path = tmp_path / "raw_train/a/step.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw.update(current_task="DIFFERENT STEP", send_prompt="DIFFERENT HISTORY",
               LLM_response_editer="DIFFERENT RESPONSE")
    raw["actions"][0]["keyboard_text"] = "COMPLETELY ABSENT TEXT"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    row = json.loads(files[0].read_text(encoding="utf-8"))
    row["actions"][0]["text"] = "COMPLETELY ABSENT TEXT"
    files[0].write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    prepare(tmp_path, files, output="second")
    one = json.loads((tmp_path / "first/train.jsonl").read_text(encoding="utf-8"))
    two = json.loads((tmp_path / "second/train.jsonl").read_text(encoding="utf-8"))
    assert one["instruction"] == two["instruction"]
    assert one["actions"] != two["actions"]


def test_raw_label_mismatch_is_rejected_instead_of_corrected(tmp_path):
    files = fixture_pair(tmp_path)
    row = json.loads(files[0].read_text(encoding="utf-8"))
    row["actions"][0]["text"] = "CHANGED TARGET"
    files[0].write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="labels differ"):
        prepare(tmp_path, files)
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("record", ["../outside.json", "C:/outside.json", "/outside.json"])
def test_unsafe_source_paths_fail_before_writing_dataset(tmp_path, record):
    files = fixture_pair(tmp_path)
    row = json.loads(files[0].read_text(encoding="utf-8"))
    row["record_path"] = record
    files[0].write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe source"):
        prepare(tmp_path, files)
    assert not (tmp_path / "derived").exists()


def test_official_test_path_is_rejected_before_any_hash_or_read(tmp_path, monkeypatch):
    monkeypatch.setattr(step_contract, "file_sha256", lambda p: pytest.fail(f"unexpected read {p}"))
    with pytest.raises(ValueError, match="before reading"):
        step_contract.prepare_original_goals(tmp_path / "test.jsonl", tmp_path / "validation.jsonl",
                                              tmp_path / "missing", tmp_path / "derived")


def test_existing_data_is_not_overwritten(tmp_path):
    files = fixture_pair(tmp_path)
    prepare(tmp_path, files)
    before = (tmp_path / "derived/train.jsonl").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path, files)
    assert (tmp_path / "derived/train.jsonl").read_bytes() == before


@pytest.mark.parametrize("stage", ["train.jsonl", "manifest.json"])
@pytest.mark.parametrize("kind", ["raw", "prepared", "image", "code"])
def test_source_changes_during_output_writes_cannot_report_success(tmp_path, monkeypatch, stage, kind):
    files = fixture_pair(tmp_path)
    changed = False
    old_write = Path.write_text
    old_sources = step_contract._conversion_source_hashes

    def source_hashes():
        return {**old_sources(), "synthetic_source": "changed" if changed and kind == "code" else "stable"}

    def write_and_change(path, text, *args, **kwargs):
        nonlocal changed
        result = old_write(path, text, *args, **kwargs)
        if path.parent == tmp_path / "derived" and path.name == stage and not changed:
            changed = True
            if kind == "raw":
                raw_path = tmp_path / "raw_train/a/step.json"
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                raw["task_prompt"] = "modified during output write"
                old_write(raw_path, json.dumps(raw), encoding="utf-8")
            elif kind == "prepared":
                old_write(files[0], files[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
            elif kind == "image":
                Image.new("RGB", (32, 24), (90, 80, 70)).save(tmp_path / "a.png")
        return result

    monkeypatch.setattr(step_contract, "_conversion_source_hashes", source_hashes)
    monkeypatch.setattr(Path, "write_text", write_and_change)
    with pytest.raises(ValueError, match="changed during conversion"):
        prepare(tmp_path, files)
    assert changed
    assert not (tmp_path / "derived/manifest.json").exists()
    assert json.loads((tmp_path / "derived/failure.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_cli_prepares_only_source_tasks_and_reports_actual_counts(tmp_path):
    train, validation = fixture_pair(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/week5_prepare_original_goals.py"
    result = subprocess.run([sys.executable, str(script), "--train", str(train), "--validation", str(validation),
                             "--raw-train-root", str(tmp_path / "raw_train"),
                             "--output-dir", str(tmp_path / "cli")], check=False,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["selected_fields"] == {"train:task_prompt": 1, "validation:task_prompt": 1}
    assert not output["tokenized"] and not output["test_data_consumed"]


def test_cancellation_after_manifest_write_revokes_success(tmp_path, monkeypatch):
    files = fixture_pair(tmp_path)
    old_write = Path.write_text

    def write_then_cancel(path, text, *args, **kwargs):
        result = old_write(path, text, *args, **kwargs)
        if path.parent == tmp_path / "derived" and path.name == "manifest.json":
            raise KeyboardInterrupt("cancelled after writing manifest")
        return result

    monkeypatch.setattr(Path, "write_text", write_then_cancel)
    with pytest.raises(KeyboardInterrupt):
        prepare(tmp_path, files)
    assert not (tmp_path / "derived/manifest.json").exists()
    failure = json.loads((tmp_path / "derived/failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "cancelled"
