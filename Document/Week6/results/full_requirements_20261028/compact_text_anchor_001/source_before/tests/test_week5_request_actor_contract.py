"""Synthetic request-only source contracts; no production source or model access."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning import actor_contract as legacy
from gui_agent.finetuning import request_actor_contract as contract
from gui_agent.finetuning.prompts import build_user_prompt


def request(goal='  输入“原文”  ', step='  输入“原文”并保存  ', width=32, height=24):
    return "\n".join([
        legacy.INTRO,
        f'您现在的目标是操作计算机屏幕，视频宽度：{width}，视频高度：{height}，总体目标是 "{goal}"。',
        "", legacy.PLAN_LINE, "1. PLAN_SECRET_MUST_NOT_SELECT_STEP", "",
        f'{legacy.ACTIVE_PREFIX}{step}{legacy.ACTIVE_END}', legacy.OPERATIONS,
        '```json\n[{"keyboard_text":"EXAMPLE_SECRET"}]\n```',
        f'{legacy.ACTIVE_PREFIX}{step}{legacy.FINAL_END}',
    ])


def raw_record():
    return {
        "session_id": "a", "task_prompt": '  输入“原文”  ', "task_prompt_en": "ENGLISH ORIGINAL GOAL",
        "send_prompt": request(), "send_prompt_en": "TRANSLATION_SECRET", "sub_task_list": ["PLAN_SECRET"],
        "video_width": 32, "video_height": 24, "saved_image_name": "screen.png",
        "LLM_response": "RESPONSE_SECRET", "actions": [
            {"action_type": "KeyboardAction", "keyboard_action_type": "text",
             "keyboard_text": "原文", "description": "ACTION_DESCRIPTION_SECRET"},
            {"action_type": "KeyboardAction", "keyboard_action_type": "press", "keyboard_key": "Ctrl+s"},
        ],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def mutate_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    write_json(path, data)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    train = tmp_path / "data/processed/v1/train.jsonl"
    raw_root = tmp_path / "raw/train"
    output = tmp_path / "outputs/prepared"
    monkeypatch.setattr(contract, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(contract, "V1_TRAIN_PATH", train)
    monkeypatch.setattr(contract, "RAW_TRAIN_ROOT", raw_root)
    monkeypatch.setattr(contract, "TRAIN_ROWS", 2)
    rows = []
    train.parent.mkdir(parents=True)
    for name, color in (("a", (10, 20, 30)), ("b", (40, 50, 60))):
        folder = raw_root / name
        (folder / "images").mkdir(parents=True)
        raw = raw_record()
        raw["session_id"] = name
        if name == "b":
            raw["current_task"] = '  输入“原文”并保存  '
        write_json(folder / "step.json", raw)
        native = folder / "images/screen.png"
        Image.new("RGB", (32, 24), color).save(native)
        prepared = train.parent / f"{name}.png"
        prepared.write_bytes(native.read_bytes())
        rgb_sha = hashlib.sha256(b"32x24:" + bytes(color) * (32 * 24)).hexdigest()
        rows.append({
            "sample_id": f"screenagent:{name}:step", "group_id": f"screenagent:{name}",
            "source": "screenagent", "source_split": "train", "instruction": raw["task_prompt_en"],
            "image_path": str(prepared), "image_width": 32, "image_height": 24,
            "image_sha256": rgb_sha, "image_file_sha256": contract.file_sha256(prepared),
            "record_path": f"{name}/step.json", "original_extra_metadata": {"preserve": True},
            "actions": [{"action_type": "type_text", "text": "原文"},
                        {"action_type": "hotkey", "keys": ["ctrl", "s"]}],
            "action_annotations": [{}, {}],
        })
    train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan = {"schema": contract.PLAN_SCHEMA, "status": "registered", "contract_version": contract.VERSION,
            "logical_split": "train", "train_input": {"path": str(train), "sha256": contract.file_sha256(train), "rows": 2},
            "raw_train_root": str(raw_root), "output_dir": str(output),
            "budgets": {"source_rows": 2, **{key: 0 for key in contract.ZERO_BUDGETS}}}
    write_json(plan_path, plan)
    return {"train": train, "raw_root": raw_root, "output": output, "plan": plan_path, "rows": rows}


def execute(fixture):
    return contract.prepare_request_actor_train(fixture["plan"], contract.file_sha256(fixture["plan"]))


def refresh_plan(fixture):
    mutate_json(fixture["plan"], lambda plan: plan["train_input"].update(sha256=contract.file_sha256(fixture["train"])))


def test_missing_metadata_is_new_version_and_legacy_still_refuses():
    raw = raw_record()
    before = copy.deepcopy(raw)
    decision = contract.verify_request_actor_request(raw, 32, 24)
    assert decision["eligible"] and decision["version"] == contract.VERSION
    assert decision["actor_step"] == '  输入“原文”并保存  '
    assert decision["instruction"] == '  输入“原文”  '
    assert decision["independent_current_task_status"] == "not_provided"
    assert decision["legacy_contract_result_not_claimed"]
    assert legacy.verify_actor_request(raw, 32, 24)["reason"] == "missing_or_empty_current_task"
    assert raw == before and "current_task" not in raw


@pytest.mark.parametrize("current", [None, "", "  ", 4, [], {}, False])
def test_present_invalid_metadata_is_never_treated_as_missing(current):
    raw = raw_record()
    raw["current_task"] = current
    assert contract.verify_request_actor_request(raw, 32, 24) == {
        "eligible": False, "reason": "empty_or_invalid_independent_current_task"}


def test_present_metadata_requires_verbatim_agreement():
    raw = raw_record()
    raw["current_task"] = '  输入“原文”并保存  '
    assert contract.verify_request_actor_request(raw, 32, 24)["independent_current_task_status"] == "provided_concordant"
    raw["current_task"] = raw["current_task"].strip()
    assert contract.verify_request_actor_request(raw, 32, 24)["reason"] == "request_activity_differs_from_current_task"


def test_conflicting_declarations_cannot_select_a_matching_plan_or_metadata():
    raw = raw_record()
    raw["send_prompt"] = raw["send_prompt"].replace(
        f'{legacy.ACTIVE_PREFIX}  输入“原文”并保存  {legacy.FINAL_END}',
        f'{legacy.ACTIVE_PREFIX}PLAN_SECRET{legacy.FINAL_END}')
    raw["current_task"] = "PLAN_SECRET"
    assert contract.verify_request_actor_request(raw, 32, 24)["reason"] == "conflicting_request_activity_declarations"


def test_target_response_translations_and_plan_items_do_not_select_step():
    raw = raw_record()
    expected = contract.verify_request_actor_request(raw, 32, 24)
    raw.update(LLM_response="CHANGED_RESPONSE", sub_task_list=["CHANGED_PLAN"], send_prompt_en="CHANGED_TRANSLATION")
    raw["actions"] = [{"action_type": "WaitAction", "wait_time": 2, "description": "CHANGED_TARGET"}]
    actual = contract.verify_request_actor_request(raw, 32, 24)
    assert actual == expected
    raw["send_prompt"] = raw["send_prompt"].replace("PLAN_SECRET_MUST_NOT_SELECT_STEP", "DIFFERENT_PLAN_ITEM")
    changed = contract.verify_request_actor_request(raw, 32, 24)
    assert (changed["instruction"], changed["actor_step"]) == (expected["instruction"], expected["actor_step"])


@pytest.mark.parametrize("family", ["PlanAction", "EvaluateSubTaskAction", "unknown"])
def test_exclusive_actor_family_filter_does_not_admit_mixed_phase(family):
    raw = raw_record()
    raw["actions"].append({"action_type": family})
    assert contract.verify_request_actor_request(raw, 32, 24)["reason"] == "not_exclusive_actor_phase"


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.update(send_prompt=None), "missing_original_send_prompt"),
    (lambda r: r.update(send_prompt=r["send_prompt"].replace(legacy.INTRO, "unknown")), "unsupported_actor_template"),
    (lambda r: r.update(send_prompt=r["send_prompt"] + "\nAFTER_FINAL"), "activity_declarations_outside_actor_positions"),
    (lambda r: r.update(send_prompt=r["send_prompt"] + '\n现在的子任务是 "third"。'), "missing_or_ambiguous_activity_declarations"),
    (lambda r: r.update(send_prompt=request(step=" ")), "empty_or_invalid_request_activity"),
    (lambda r: r.update(send_prompt=request(step="bad\x00step")), "empty_or_invalid_request_activity"),
    (lambda r: r.update(task_prompt="different"), "request_goal_differs_from_source_goal"),
    (lambda r: r.update(video_width=True), "invalid_native_dimensions"),
    (lambda r: r.update(video_height=23), "request_dimensions_differ_from_native_image"),
    (lambda r: r.update(current_task="first\nsecond"), "unsupported_multiline_current_task"),
])
def test_fixed_source_contract_rejection(change, reason):
    raw = raw_record()
    change(raw)
    assert contract.verify_request_actor_request(raw, 32, 24)["reason"] == reason


def test_linux_original_template_is_supported_without_translation():
    raw = raw_record()
    raw["send_prompt"] = raw["send_prompt"].replace(legacy.INTRO, legacy.LINUX_INTRO)
    assert contract.verify_request_actor_request(raw, 32, 24)["request_template"] == "actor_agent_linux_zh"


def test_derivation_preserves_whole_labels_identity_and_all_source_bytes(registered):
    before = {path: path.read_bytes() for path in registered["train"].parent.rglob("*") if path.is_file()}
    before.update({path: path.read_bytes() for path in registered["raw_root"].rglob("*") if path.is_file()})
    result = execute(registered)
    new_rows = [json.loads(line) for line in (registered["output"] / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    for old, new in zip(registered["rows"], new_rows, strict=True):
        assert {key: value for key, value in new.items() if key not in contract.DERIVED_FIELDS and key != "instruction"} == {
            key: value for key, value in old.items() if key != "instruction"}
        assert new["instruction"] == '  输入“原文”  ' and new["actor_step"] == '  输入“原文”并保存  '
        rendered = build_user_prompt(new, "conditional_actor")
        assert all(secret not in rendered for secret in ("PLAN_SECRET", "RESPONSE_SECRET", "ACTION_DESCRIPTION_SECRET", "EXAMPLE_SECRET", "raw_record"))
    assert result["original_rows"] == 2 and len(result["records"]) == 2
    assert result["independent_current_task_absence_count"] == 1
    assert result["logical_split"] == "train" and set(result["outputs"]) == {"train"}
    assert result["legacy_contract_result_not_claimed"] and not result["new_sources_are_independent_holdout"]
    assert not result["training_admission_performed"] and not result["token_length_preflight_performed"]
    assert not result["candidate_semantic_admission_performed"] and not result["literal_admission_performed"]
    assert all(path.read_bytes() == value for path, value in before.items())


def test_request_rejected_row_stays_in_original_denominator(registered):
    raw_path = registered["raw_root"] / "a/step.json"
    mutate_json(raw_path, lambda raw: raw.update(current_task=None))
    result = execute(registered)
    assert result["original_rows"] == len(result["records"]) == 2
    assert result["outputs"]["train"]["samples"] == 1
    assert result["records"][0]["reason"] == "empty_or_invalid_independent_current_task"
    assert result["records"][0]["independent_current_task_present"]


@pytest.mark.parametrize("name", ["test", "dev", "validation", "test.jsonl", "dev.jsonl", "validation.jsonl"])
def test_forbidden_paths_refused_before_plan_or_data_read(tmp_path, monkeypatch, name):
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"unexpected content read: {self}"))
    with pytest.raises(ValueError, match="forbidden before reading"):
        contract.prepare_request_actor_train(tmp_path / name / "plan.json", "0" * 64)


@pytest.mark.parametrize("split", ["test", "dev", "validation"])
def test_logical_nontrain_split_refused_before_any_read(tmp_path, monkeypatch, split):
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unexpected read"))
    with pytest.raises(ValueError, match="before reading any"):
        contract.prepare_request_actor_train(tmp_path / "plan.json", "0" * 64, logical_split=split)


@pytest.mark.parametrize("change,reason", [
    (lambda p: p.update(status="draft"), "registered train-only"),
    (lambda p: p.update(logical_split="validation"), "registered train-only"),
    (lambda p: p["budgets"].update(source_rows=1), "whole V1 train denominator"),
    (lambda p: p["budgets"].update(source_rows=True), "whole V1 train denominator"),
    (lambda p: p["budgets"].update(model_calls=1), "zero model"),
    (lambda p: p["budgets"].update(GUI_calls=False), "zero model"),
    (lambda p: p["train_input"].update(rows=1), "whole V1 train denominator"),
    (lambda p: p["train_input"].update(path="relative/train.jsonl"), "absolute input"),
])
def test_unregistered_or_invalid_budget_refuses_before_train_read(registered, monkeypatch, change, reason):
    mutate_json(registered["plan"], change)
    original_read = Path.read_bytes
    def guarded(path):
        if path == registered["train"]:
            pytest.fail("train read before plan validation")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match=reason):
        execute(registered)


def test_plan_sha_is_mandatory_and_wrong_sha_stops_source_read(registered, monkeypatch):
    monkeypatch.setattr(contract, "_conversion_sources", lambda: pytest.fail("dependencies read before plan validated"))
    with pytest.raises(ValueError, match="plan SHA256 mismatch"):
        contract.prepare_request_actor_train(registered["plan"], "0" * 64)
    with pytest.raises(ValueError, match="64 hex"):
        contract.prepare_request_actor_train(registered["plan"], "")


@pytest.mark.parametrize("target", ["dev.jsonl", "other_train.jsonl"])
def test_plan_cannot_redirect_unique_v1_entry(registered, target):
    mutate_json(registered["plan"], lambda p: p["train_input"].update(path=str(registered["train"].with_name(target))))
    with pytest.raises(ValueError, match="forbidden before reading|only the original"):
        execute(registered)


def test_existing_output_refuses_before_source_read(registered, monkeypatch):
    registered["output"].mkdir(parents=True)
    monkeypatch.setattr(contract, "_conversion_sources", lambda: pytest.fail("source dependency read after collision"))
    with pytest.raises(ValueError, match="collision refused"):
        execute(registered)


def test_train_sha_and_denominator_are_not_refitted(registered):
    registered["train"].write_text(registered["train"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="V1 train SHA256 mismatch"):
        execute(registered)
    registered["train"].write_text(json.dumps(registered["rows"][0], ensure_ascii=False) + "\n", encoding="utf-8")
    refresh_plan(registered)
    with pytest.raises(ValueError, match="whole V1 train row denominator"):
        execute(registered)
    assert not registered["output"].exists()


@pytest.mark.parametrize("field,value,reason", [
    ("source_split", "validation", "only original training-source"),
    ("actor_step", "DERIVED", "original V1 data"),
    ("record_path", "../outside.json", "unsafe source"),
    ("record_path", "dev/step.json", "forbidden before reading"),
    ("image_path", "dev/private.png", "forbidden before reading"),
])
def test_all_paths_and_context_checked_before_image_open(registered, monkeypatch, field, value, reason):
    rows = copy.deepcopy(registered["rows"])
    rows[0][field] = value
    registered["train"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    refresh_plan(registered)
    monkeypatch.setattr(Image, "open", lambda path: pytest.fail(f"image opened before precheck: {path}"))
    with pytest.raises(ValueError, match=reason):
        execute(registered)


def test_prepared_whole_label_mismatch_is_fatal_and_never_repaired(registered):
    rows = copy.deepcopy(registered["rows"])
    rows[0]["actions"][0]["text"] = "WRONG"
    registered["train"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    refresh_plan(registered)
    with pytest.raises(ValueError, match="whole labels/annotations differ"):
        execute(registered)
    assert not registered["output"].exists()


def test_refused_request_still_requires_whole_native_labels(registered):
    mutate_json(registered["raw_root"] / "a/step.json", lambda raw: raw.update(current_task=None))
    rows = copy.deepcopy(registered["rows"])
    rows[0]["actions"][0]["text"] = "WRONG"
    registered["train"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    refresh_plan(registered)
    with pytest.raises(ValueError, match="whole labels/annotations differ"):
        execute(registered)
    assert not registered["output"].exists()


def test_raw_native_rgb_identity_must_match_prepared(registered):
    Image.new("RGB", (32, 24), (200, 10, 20)).save(registered["raw_root"] / "a/images/screen.png")
    with pytest.raises(ValueError, match="source screenshot differs"):
        execute(registered)


def test_source_image_toctou_cannot_become_a_new_baseline(registered, monkeypatch):
    original = legacy._source_image
    done = False
    def alter(root, record, raw, row):
        nonlocal done
        image = original(root, record, raw, row)
        if not done:
            Image.new("RGB", (32, 24), (200, 30, 10)).save(image)
            done = True
        return image
    monkeypatch.setattr(legacy, "_source_image", alter)
    with pytest.raises(ValueError, match="source or screenshot changed"):
        execute(registered)
    assert not registered["output"].exists()


def test_output_parent_replaced_after_mkdir_refuses_before_first_open(registered, monkeypatch):
    original_mkdir, original_resolve, original_open = Path.mkdir, Path.resolve, Path.open
    replaced, dataset_opened = False, False
    outside = registered["output"].parents[2] / "outside_repository"
    def replaced_mkdir(path, *args, **kwargs):
        nonlocal replaced
        result = original_mkdir(path, *args, **kwargs)
        if path == registered["output"]:
            replaced = True
        return result
    def replaced_resolve(path, *args, **kwargs):
        # Model a directory link replacement deterministically, without requiring
        # Windows symlink privileges or ever writing outside the synthetic root.
        if replaced and path.is_relative_to(registered["output"]):
            return outside / path.relative_to(registered["output"])
        return original_resolve(path, *args, **kwargs)
    def guarded_open(path, *args, **kwargs):
        nonlocal dataset_opened
        if path == registered["output"] / "train.jsonl":
            dataset_opened = True
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", replaced_mkdir)
    monkeypatch.setattr(Path, "resolve", replaced_resolve)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="escapes repository"):
        execute(registered)
    assert replaced and not dataset_opened
    assert not (registered["output"] / "train.jsonl").exists()
    assert not (registered["output"] / "manifest.json").exists()
    assert not outside.exists()


@pytest.mark.parametrize("failure", ["raw", "train", "plan", "code", "output", "cancel", "write"])
def test_postwrite_failure_revokes_owned_success_manifest(registered, monkeypatch, failure):
    original = contract._write_new
    def alter(path, text, **kwargs):
        if path.name == "manifest.json" and failure == "write":
            raise OSError("disk full")
        original(path, text, **kwargs)
        if path.name != "manifest.json":
            return
        if failure == "raw":
            write_json(registered["raw_root"] / "a/step.json", {})
        elif failure == "train":
            registered["train"].write_text("changed", encoding="utf-8")
        elif failure == "plan":
            write_json(registered["plan"], {})
        elif failure == "code":
            monkeypatch.setattr(contract, "_conversion_sources", lambda: {"changed": "code"})
        elif failure == "output":
            (registered["output"] / "train.jsonl").write_text("changed", encoding="utf-8")
        elif failure == "cancel":
            raise KeyboardInterrupt()
    monkeypatch.setattr(contract, "_write_new", alter)
    error = KeyboardInterrupt if failure == "cancel" else (ValueError, OSError)
    with pytest.raises(error):
        execute(registered)
    assert not (registered["output"] / "manifest.json").exists()
    failed = json.loads((registered["output"] / "failure.json").read_text(encoding="utf-8"))
    assert failed["status"] == ("cancelled" if failure == "cancel" else "failed")
    assert (registered["output"] / "train.jsonl").exists()


def test_cli_passes_only_bound_plan_and_explicit_train_split(monkeypatch, capsys):
    script = Path(contract.__file__).resolve().parents[3] / "scripts/week5_prepare_request_actor_train.py"
    spec = importlib.util.spec_from_file_location("request_train_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = []
    def fake(plan, sha256, *, logical_split):
        received.append((plan, sha256, logical_split))
        return {"version": contract.VERSION, "original_rows": 2, "outputs": {}, "reason_counts": {}}
    monkeypatch.setattr(module, "prepare_request_actor_train", fake)
    assert module.main(["--plan", "synthetic-plan.json", "--plan-sha256", "a" * 64, "--logical-split", "train"]) == 0
    assert received == [("synthetic-plan.json", "a" * 64, "train")]
    assert json.loads(capsys.readouterr().out)["training_admission_performed"] is False
    with pytest.raises(SystemExit):
        module.main(["--plan", "synthetic-plan.json", "--plan-sha256", "a" * 64, "--logical-split", "validation"])
