"""Conditional actor input provenance, contradictory requests, and immutable labels."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning import actor_contract as contract
from gui_agent.finetuning.prompts import build_user_prompt


def prompt(goal='输入“Angemon”', step='输入“Angemon”并保存', width=32, height=24):
    return "\n".join([
        contract.INTRO,
        f'您现在的目标是操作计算机屏幕，视频宽度：{width}，视频高度：{height}，总体目标是 "{goal}"。',
        "", contract.PLAN_LINE, "    1. PLAN_SECRET_NOT_INPUT", "",
        f'{contract.ACTIVE_PREFIX}{step}{contract.ACTIVE_END}', contract.OPERATIONS,
        '```json\n[{"keyboard_text":"EXAMPLE_NOT_INPUT"}]\n```',
        f'{contract.ACTIVE_PREFIX}{step}{contract.FINAL_END}',
    ])


def raw_record():
    return {"session_id": "a", "task_prompt": '输入“Angemon”', "task_prompt_en": 'Enter "Angel"',
            "current_task": '输入“Angemon”并保存', "send_prompt": prompt(),
            "send_prompt_en": "TRANSLATED_SECRET_NOT_INPUT", "advice": "ADVICE_NOT_INPUT",
            "video_width": 32, "video_height": 24, "saved_image_name": "screenshot.png",
            "LLM_response_editer": "RESPONSE_SECRET_NOT_INPUT",
            "actions": [{"action_type": "KeyboardAction", "keyboard_action_type": "text",
                         "keyboard_text": "Angemon", "description": "DESCRIPTION_NOT_INPUT"},
                        {"action_type": "KeyboardAction", "keyboard_action_type": "press",
                         "keyboard_key": "Ctrl+s"}]}


def pair(tmp_path):
    paths = []
    for name, color in (("a", (10, 20, 30)), ("b", (50, 60, 70))):
        folder = tmp_path / "raw_train" / name
        (folder / "images").mkdir(parents=True)
        raw = raw_record()
        raw["session_id"] = name
        (folder / "step.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf8")
        source_image = folder / "images/screenshot.png"
        Image.new("RGB", (32, 24), color).save(source_image)
        image = tmp_path / f"{name}.png"
        image.write_bytes(source_image.read_bytes())
        row = {"sample_id": f"screenagent:{name}:step", "group_id": f"screenagent:{name}",
               "source": "screenagent", "source_split": "train", "instruction": raw["task_prompt_en"],
               "image_path": str(image), "image_width": 32, "image_height": 24,
               "record_path": f"{name}/step.json", "original_extra_metadata": {"unchanged": True},
               "actions": [{"action_type": "type_text", "text": "Angemon"},
                           {"action_type": "hotkey", "keys": ["ctrl", "s"]}],
               "action_annotations": [{}, {}]}
        path = tmp_path / ("train.jsonl" if name == "a" else "validation.jsonl")
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf8")
        paths.append(path)
    return paths


def prepare(tmp_path, paths, output="derived"):
    return contract.prepare_actor_contract(*paths, tmp_path / "raw_train", tmp_path / output)


def replace_json(path, mutation):
    row = json.loads(path.read_text(encoding="utf8"))
    mutation(row)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf8")


def test_fixed_declarations_are_parsed_before_raw_field_comparison():
    raw = raw_record()
    decision = contract.verify_actor_request(raw, 32, 24)
    assert decision["eligible"] and decision["actor_step"] == raw["current_task"]
    assert decision["instruction"] == raw["task_prompt"]
    assert set(decision) == {"eligible", "reason", "instruction", "goal_field", "actor_step",
                             "request_prompt_sha256", "declaration_line_indices", "request_template",
                             "source_revision"}


def test_only_input_fields_can_affect_goal_and_actor_step():
    raw = raw_record()
    original = contract.verify_actor_request(raw, 32, 24)
    changed = copy.deepcopy(raw)
    changed["actions"] = [{"action_type": "WaitAction", "wait_time": 2}]
    changed.update(LLM_response="COMPLETELY DIFFERENT", LLM_response_editer="LABEL_SECRET",
                   send_prompt_en="WRONG TRANSLATION", send_prompt_zh="ANOTHER TRANSLATION",
                   sub_task_list=["A DIFFERENT GOLD PLAN"])
    actual = contract.verify_actor_request(changed, 32, 24)
    assert actual == original


def test_preserves_unicode_and_whitespace_of_goal_and_step():
    raw = raw_record()
    raw["task_prompt"] = '  输入“原文”  '
    raw["current_task"] = '  输入“原文”后按 Ctrl+S  '
    raw["send_prompt"] = prompt(raw["task_prompt"], raw["current_task"])
    decision = contract.verify_actor_request(raw, 32, 24)
    assert decision["eligible"]
    assert decision["instruction"] == raw["task_prompt"]
    assert decision["actor_step"] == raw["current_task"]


def test_official_linux_template_uses_the_same_source_only_activity_policy():
    raw = raw_record()
    raw["send_prompt"] = raw["send_prompt"].replace(contract.INTRO, contract.LINUX_INTRO)
    decision = contract.verify_actor_request(raw, 32, 24)
    assert decision["eligible"] and decision["request_template"] == "actor_agent_linux_zh"
    assert decision["actor_step"] == raw["current_task"]


@pytest.mark.parametrize("current", [None, "", "  ", 42])
def test_missing_current_task_is_an_explicit_exclusion(current):
    raw = raw_record()
    raw["current_task"] = current
    assert contract.verify_actor_request(raw, 32, 24) == {
        "eligible": False, "reason": "missing_or_empty_current_task"}


def test_conflicting_declarations_do_not_choose_the_matching_one():
    raw = raw_record()
    raw["send_prompt"] = raw["send_prompt"].replace(
        f'{contract.ACTIVE_PREFIX}{raw["current_task"]}{contract.FINAL_END}',
        f'{contract.ACTIVE_PREFIX}旧子任务{contract.FINAL_END}')
    assert contract.verify_actor_request(raw, 32, 24)["reason"] == "conflicting_request_activity_declarations"


def test_current_field_matching_only_plan_or_example_cannot_pass():
    raw = raw_record()
    raw["current_task"] = "EXAMPLE_NOT_INPUT"
    assert contract.verify_actor_request(raw, 32, 24)["reason"] == "request_activity_differs_from_current_task"


@pytest.mark.parametrize("family", ["PlanAction", "EvaluateSubTaskAction", "UNKNOWN"])
def test_non_actor_or_mixed_phase_exports_fail_closed(family):
    raw = raw_record()
    raw["actions"].append({"action_type": family, "element": raw["current_task"]})
    assert contract.verify_actor_request(raw, 32, 24)["reason"] == "not_exclusive_actor_phase"


@pytest.mark.parametrize("change,reason", [
    (lambda raw: raw.update(send_prompt="not an actor request"), "unsupported_actor_template"),
    (lambda raw: raw.update(send_prompt=None), "missing_original_send_prompt"),
    (lambda raw: raw.update(current_task="first\nsecond"), "unsupported_multiline_current_task"),
    (lambda raw: raw.update(task_prompt="different source goal"), "request_goal_differs_from_source_goal"),
    (lambda raw: raw.update(video_width=31), "request_dimensions_differ_from_native_image"),
    (lambda raw: raw.update(video_width=True), "invalid_native_dimensions"),
    (lambda raw: raw.update(send_prompt=raw["send_prompt"] + '\n现在的子任务是 "a"。'),
     "missing_or_ambiguous_activity_declarations"),
    (lambda raw: raw.update(send_prompt=raw["send_prompt"].replace(contract.PLAN_LINE, "other context")),
     "unsupported_actor_template"),
    (lambda raw: raw.update(send_prompt=raw["send_prompt"] + '\nHISTORY_AFTER_FINAL'),
     "activity_declarations_outside_actor_positions"),
])
def test_unknown_or_inconsistent_request_evidence_is_rejected(change, reason):
    raw = raw_record()
    change(raw)
    assert contract.verify_actor_request(raw, 32, 24)["reason"] == reason


def test_conversion_changes_only_input_and_context_provenance(tmp_path):
    paths = pair(tmp_path)
    original = {path: path.read_bytes() for path in paths}
    images = {path: path.read_bytes() for path in tmp_path.rglob("*.png")}
    result = prepare(tmp_path, paths)
    for split, path in zip(("train", "validation"), paths, strict=True):
        old = json.loads(original[path])
        new = json.loads((tmp_path / "derived" / f"{split}.jsonl").read_text(encoding="utf8"))
        assert {key: value for key, value in new.items()
                if key not in {"instruction", "instruction_source", "actor_step", "actor_step_source"}} == {
                    key: value for key, value in old.items() if key != "instruction"}
        assert new["instruction"] == '输入“Angemon”' and new["actor_step"] == '输入“Angemon”并保存'
        rendered = build_user_prompt(new, "conditional_actor")
        assert new["actor_step"] in rendered and new["instruction"] in rendered
        assert all(secret not in rendered for secret in (
            "PLAN_SECRET", "EXAMPLE_NOT_INPUT", "TRANSLATED_SECRET", "ADVICE_NOT_INPUT",
            "RESPONSE_SECRET", "DESCRIPTION_NOT_INPUT", "actor_step_source", "raw_record"))
    assert result["labels_and_annotations_preserved_and_raw_verified"]
    assert result["split_isolation_verified"] and not result["test_data_consumed"]
    assert not result["model_or_processor_loaded"] and not result["token_length_preflight_performed"]
    assert all(path.read_bytes() == content for path, content in original.items())
    assert all(path.read_bytes() == content for path, content in images.items())


def test_excluded_rows_remain_in_manifest_with_source_only_reason(tmp_path):
    paths = pair(tmp_path)
    row = json.loads(paths[0].read_text(encoding="utf8"))
    second = copy.deepcopy(row)
    second.update(sample_id="screenagent:a:missing", record_path="a/missing.json")
    raw = raw_record()
    raw["current_task"] = None
    (tmp_path / "raw_train/a/missing.json").write_text(json.dumps(raw), encoding="utf8")
    with paths[0].open("a", encoding="utf8") as stream:
        stream.write(json.dumps(second) + "\n")
    result = prepare(tmp_path, paths)
    assert result["outputs"]["train"]["samples"] == 1
    rejected = [row for row in result["records"] if not row["eligible"]]
    assert len(rejected) == 1 and rejected[0]["reason"] == "missing_or_empty_current_task"


def test_empty_derived_split_cannot_receive_success_manifest(tmp_path):
    paths = pair(tmp_path)
    replace_json(tmp_path / "raw_train/b/step.json", lambda row: row.update(current_task=None))
    with pytest.raises(ValueError, match="both train and validation"):
        prepare(tmp_path, paths)
    assert not (tmp_path / "derived").exists()


def test_target_changes_never_choose_a_different_input(tmp_path):
    paths = pair(tmp_path)
    prepare(tmp_path, paths, output="first")
    replace_json(tmp_path / "raw_train/a/step.json", lambda raw: raw["actions"][0].update(
        keyboard_text="NOT IN ANY INSTRUCTION"))
    replace_json(paths[0], lambda row: row["actions"][0].update(text="NOT IN ANY INSTRUCTION"))
    prepare(tmp_path, paths, output="second")
    first = json.loads((tmp_path / "first/train.jsonl").read_text(encoding="utf8"))
    second = json.loads((tmp_path / "second/train.jsonl").read_text(encoding="utf8"))
    assert (first["instruction"], first["actor_step"]) == (second["instruction"], second["actor_step"])
    assert first["actions"] != second["actions"]


def test_prepared_label_mismatch_is_fatal_and_not_repaired(tmp_path):
    paths = pair(tmp_path)
    replace_json(paths[0], lambda row: row["actions"][0].update(text="WRONG_LABEL"))
    with pytest.raises(ValueError, match="labels differ"):
        prepare(tmp_path, paths)
    assert not (tmp_path / "derived").exists()


def test_source_native_image_identity_is_required(tmp_path):
    paths = pair(tmp_path)
    Image.new("RGB", (32, 24), (200, 10, 20)).save(tmp_path / "raw_train/a/images/screenshot.png")
    with pytest.raises(ValueError, match="source screenshot differs"):
        prepare(tmp_path, paths)


def test_source_image_change_after_validation_cannot_redefine_baseline(tmp_path, monkeypatch):
    paths = pair(tmp_path)
    original_check = contract._source_image
    changed = False

    def mutate_after_validated(root, record, raw, row):
        nonlocal changed
        image = original_check(root, record, raw, row)
        if not changed:
            changed = True
            Image.new("RGB", (32, 24), (200, 40, 10)).save(image)
        return image

    monkeypatch.setattr(contract, "_source_image", mutate_after_validated)
    with pytest.raises(ValueError, match="source screenshot changed"):
        prepare(tmp_path, paths)
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("source_path", ["../outside.json", "C:/outside.json", "/outside.json", "test/x.json"])
def test_source_record_escape_or_official_test_rejected(tmp_path, source_path):
    paths = pair(tmp_path)
    replace_json(paths[0], lambda row: row.update(record_path=source_path))
    with pytest.raises(ValueError, match="unsafe source|official test"):
        prepare(tmp_path, paths)
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("which", ["train", "validation", "raw"])
def test_official_test_path_is_rejected_before_any_content_read(tmp_path, monkeypatch, which):
    monkeypatch.setattr(contract, "file_sha256", lambda path: pytest.fail(f"unexpected hash/read: {path}"))
    arguments = [tmp_path / "train.jsonl", tmp_path / "validation.jsonl", tmp_path / "raw_train"]
    arguments[{"train": 0, "validation": 1, "raw": 2}[which]] = tmp_path / "test" / "data.jsonl"
    with pytest.raises(ValueError, match="before reading"):
        contract.prepare_actor_contract(*arguments, tmp_path / "derived")


def test_existing_output_refused_before_data_read(tmp_path, monkeypatch):
    paths = pair(tmp_path)
    (tmp_path / "derived").mkdir()
    monkeypatch.setattr(contract, "file_sha256", lambda path: pytest.fail(f"unexpected read: {path}"))
    with pytest.raises(ValueError, match="already exists"):
        prepare(tmp_path, paths)


def test_declared_official_test_screenshot_refused_before_image_open(tmp_path, monkeypatch):
    paths = pair(tmp_path)
    replace_json(paths[0], lambda row: row.update(image_path=str(tmp_path / "test/private.png")))
    monkeypatch.setattr(Image, "open", lambda path: pytest.fail(f"unexpected image read: {path}"))
    with pytest.raises(ValueError, match="before reading"):
        prepare(tmp_path, paths)


def test_group_leakage_is_rejected_before_source_record_lookup(tmp_path):
    paths = pair(tmp_path)
    replace_json(paths[1], lambda row: row.update(group_id="screenagent:a"))
    with pytest.raises(ValueError, match="leakage: group_id"):
        prepare(tmp_path, paths)


def test_derived_context_is_not_accepted_as_original_source(tmp_path):
    paths = pair(tmp_path)
    replace_json(paths[0], lambda row: row.update(actor_step="already derived"))
    with pytest.raises(ValueError, match="original V1"):
        prepare(tmp_path, paths)


@pytest.mark.parametrize("failure", ["raw_change", "output_change", "source_change", "cancel", "write_error"])
def test_postwrite_failure_revokes_success_manifest_and_preserves_failure(tmp_path, monkeypatch, failure):
    paths = pair(tmp_path)
    original_write = Path.write_text

    def altered_write(path, *args, **kwargs):
        if path == tmp_path / "derived/manifest.json" and failure == "write_error":
            raise OSError("disk write failed")
        value = original_write(path, *args, **kwargs)
        if path == tmp_path / "derived/manifest.json":
            if failure == "raw_change":
                original_write(tmp_path / "raw_train/a/step.json", "{}", encoding="utf8")
            elif failure == "output_change":
                original_write(tmp_path / "derived/train.jsonl", "{}\n", encoding="utf8")
            elif failure == "source_change":
                monkeypatch.setattr(contract, "_conversion_sources", lambda: {"changed": "source"})
            elif failure == "cancel":
                raise KeyboardInterrupt()
        return value

    monkeypatch.setattr(Path, "write_text", altered_write)
    exception = KeyboardInterrupt if failure == "cancel" else (OSError, ValueError)
    with pytest.raises(exception):
        prepare(tmp_path, paths)
    assert not (tmp_path / "derived/manifest.json").exists()
    failed = json.loads((tmp_path / "derived/failure.json").read_text(encoding="utf8"))
    assert failed["status"] == ("cancelled" if failure == "cancel" else "failed")
    assert (tmp_path / "derived/train.jsonl").exists()


def test_cli_prepares_only_given_train_and_dev_pair(tmp_path):
    paths = pair(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/week5_prepare_actor_contract.py"
    result = subprocess.run([sys.executable, str(script), "--train", str(paths[0]),
                             "--validation", str(paths[1]), "--raw-train-root", str(tmp_path / "raw_train"),
                             "--output-dir", str(tmp_path / "derived")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["outputs"]["validation"]["samples"] == 1
