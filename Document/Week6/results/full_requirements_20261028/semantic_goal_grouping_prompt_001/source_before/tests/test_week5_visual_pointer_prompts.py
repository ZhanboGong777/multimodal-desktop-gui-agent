"""Versioned prompt semantics using synthetic evidence only; no processors."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
import test_week5_visual_pointer as fixtures
import test_week5_visual_pointer_inference as inference_fixtures
import test_week5_visual_pointer_input as input_fixtures

from gui_agent.finetuning.candidate_grounding import center
from gui_agent.finetuning.evaluation import canonical_json, file_sha256
from gui_agent.finetuning.pointer_protocol import pointer_user_prompt, select_candidates
from gui_agent.finetuning.training import TrainingConfig
from gui_agent.finetuning.visual_pointer import VISUAL_PREFIX, build_visual_pointer_messages
from gui_agent.finetuning.visual_pointer_prompts import (
    COMPACT_HEADER,
    COMPACT_PROMPT_VERSION,
    DEDUP_PROMPT_VERSION,
    GALLERY_HEADER,
    GALLERY_INSTRUCTION,
    GALLERY_PROMPT_VERSION,
    build_visual_user_prompt,
    declared_prompt_version,
)

font_asset = fixtures.font_asset


def test_dedup_v2_retains_pre_additive_exact_byte_contract():
    snapshot = {"image_width": 100, "image_height": 100, "candidates": [
        {"candidate_id": "A", "source": "ocr", "role": "text_region", "confidence": .8,
         "text": "abcdefghijklmnopqrstuvwxyz", "bbox": {"left": 10, "top": 10, "right": 30, "bottom": 30}}]}
    prompt = build_visual_user_prompt("原始目标\n  空格", "原始当前子任务", snapshot, visual_prompt_version=DEDUP_PROMPT_VERSION)
    assert hashlib.sha256(prompt.encode("utf8")).hexdigest() == "2a78060ba6625634dde9605b1c540ab21df6ec3d2191cae9a40f35ef84001899"


def test_compact005_preserves_exact_suffix64_and_all_rules(tmp_path):
    assert hashlib.sha256(COMPACT_HEADER.encode("utf8")).hexdigest() == "ac019aee39eb165815e403eb0f935649ba6b4f87e3aa09b051095134a15c0d31"
    candidates = [fixtures.candidate(f"c{i:03}", 15 + i % 16 * 61, 15 + i // 16 * 180, source="ocr") |
                  {"text": "中英文 OCR abcdefghijklmnopqrstuvwxyz"} for i in range(64)]
    row = fixtures.make_sample(tmp_path, candidates=candidates)
    row.update(instruction="完整原目标\n  Preserve spaces", actor_step="当前原子任务\n 不截断")
    fixtures.bind(row)
    legacy = build_visual_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"])
    compact = build_visual_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"],
                                       visual_prompt_version=COMPACT_PROMPT_VERSION)
    assert compact.startswith(COMPACT_HEADER)
    assert compact[compact.index("Native size:"):] == legacy[legacy.index("Native size:"):]
    for text in ("complete ordered JSON action array", "exact requested text", "Current subtask only",
                 "action_type", "click double_click right_click move: candidate dx dy",
                 "scroll: candidate dx dy scroll_amount(nonzero integer)",
                 "drag: start_candidate start_dx start_dy end_candidate end_dx end_dy",
                 "type_text:text; key_press:key; hotkey:keys; wait:duration", "only listed fields, no x/y",
                 "IDs/offsets integers", "mouse/both drag endpoints=center+(dx,dy)", "intended offsets in bounds",
                 "annotations, not UI controls", "Badge ID=table ID", "leader tip=native candidate center",
                 "never click badges", "Candidate proposals; inspect screenshot; no region fits: []"):
        assert text in compact


def visual_config(tmp_path, version):
    return {"input_image_protocol": "visual_pointer_v2", "target_protocol": "candidate_compact",
            "prompt_variant": "conditional_actor", "visual_prompt_version": version,
            "visual_asset_manifest_path": str(tmp_path / "approved.json"), "visual_asset_manifest_sha256": "a" * 64,
            "visual_original_root": str(tmp_path / "native"), "visual_asset_root": str(tmp_path / "assets")}


def test_training_config_accepts_caller_compact_and_preserves_legacy_null(tmp_path):
    assert TrainingConfig(**visual_config(tmp_path, COMPACT_PROMPT_VERSION)).visual_prompt_version == COMPACT_PROMPT_VERSION
    assert TrainingConfig(**visual_config(tmp_path, None)).visual_prompt_version is None
    with pytest.raises(ValueError):
        declared_prompt_version({"visual_prompt_version": None})


@pytest.mark.parametrize("value", ["", "unknown", False, 1])
def test_training_config_rejects_unknown_prompt_versions(tmp_path, value):
    with pytest.raises(ValueError):
        TrainingConfig(**visual_config(tmp_path, value))


def test_compact_rebind_same_png_approved_version_and_mixed_registry_refusal(tmp_path, font_asset):
    asset = input_fixtures.Asset(tmp_path, font_asset)
    png = asset.model.read_bytes()
    input_fixtures.dedup_asset(asset)
    asset.row["visual_prompt_version"] = COMPACT_PROMPT_VERSION
    with pytest.raises(ValueError):
        asset.validate()  # caller-approved v2 registry cannot approve a v3 row
    asset.metadata["visual_prompt_version"] = COMPACT_PROMPT_VERSION
    asset.prompt = build_visual_user_prompt(asset.row["instruction"], asset.row["actor_step"], asset.row["candidate_snapshot"],
                                           visual_prompt_version=COMPACT_PROMPT_VERSION)
    asset.metadata["prompt_sha256"] = hashlib.sha256(asset.prompt.encode("utf8")).hexdigest()
    asset.register()
    descriptor = asset.validate()
    assert descriptor.visual_prompt_version == COMPACT_PROMPT_VERSION and descriptor.prompt == asset.prompt
    assert asset.model.read_bytes() == png
    assert build_visual_pointer_messages(asset.row, visual_prompt_version=COMPACT_PROMPT_VERSION)[0]["content"][1]["text"] == asset.prompt
    with pytest.raises(ValueError):
        build_visual_pointer_messages(asset.row, visual_prompt_version=DEDUP_PROMPT_VERSION)


def test_compact_child_receives_exact_approved_header_and_version_ack(tmp_path, font_asset, monkeypatch):
    asset = input_fixtures.dedup_asset(input_fixtures.Asset(tmp_path, font_asset))
    asset.row["visual_prompt_version"] = asset.metadata["visual_prompt_version"] = COMPACT_PROMPT_VERSION
    asset.prompt = build_visual_user_prompt(asset.row["instruction"], asset.row["actor_step"], asset.row["candidate_snapshot"],
                                           visual_prompt_version=COMPACT_PROMPT_VERSION)
    asset.metadata["prompt_sha256"] = hashlib.sha256(asset.prompt.encode("utf8")).hexdigest()
    asset.register()
    state = inference_fixtures.install_child_spies(monkeypatch)
    descriptor, envelope = inference_fixtures.inference.prepare_visual_request(asset.row, registry=asset.registry)
    result = inference_fixtures.child_responses(inference_fixtures.config_for(asset), [envelope])
    assert result[0]["visual_prompt_version"] == COMPACT_PROMPT_VERSION
    assert result[1]["status"] == "ok" and result[1]["input_ack"]["visual_prompt_version"] == COMPACT_PROMPT_VERSION
    assert result[1]["input_ack"] == inference_fixtures.inference.input_ack(descriptor, asset.registry)
    assert state.texts == [asset.prompt] and state.processor_calls == state.generate_calls == 1


def test_missing_version_keeps_exact_legacy_prompt(tmp_path):
    row = fixtures.make_sample(tmp_path)
    assert build_visual_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"]) == (
        VISUAL_PREFIX + pointer_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"]))


def test_gallery004_only_changes_annotation_description_preserves_grammar_and_native_suffix64(tmp_path):
    row = fixtures.make_sample(tmp_path, candidates=[fixtures.candidate(f"c{i:03}",
            15 + i % 16 * 61, 15 + i // 16 * 180, source="ocr") |
            {"text": "中英文 OCR abcdefghijklmnopqrstuvwxyz"} for i in range(64)])
    row.update(instruction="完整原目标\n  Preserve spaces", actor_step="当前原子任务\n 不截断")
    fixtures.bind(row)
    compact = build_visual_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"],
                                       visual_prompt_version=COMPACT_PROMPT_VERSION)
    gallery = build_visual_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"],
                                       visual_prompt_version=GALLERY_PROMPT_VERSION)
    assert gallery.startswith(GALLERY_HEADER) and GALLERY_INSTRUCTION in gallery
    assert GALLERY_HEADER[:GALLERY_HEADER.index("UI above;")] == COMPACT_HEADER[:COMPACT_HEADER.index("Black badges/")]
    assert gallery[gallery.index("Native size:"):] == compact[compact.index("Native size:"):]
    assert "Table ID X Y OCR<=24chars\n" in GALLERY_HEADER
    assert "never gallery. No fit: []." in gallery and "leader" not in gallery and "badge" not in gallery


def test_new_prompt_keeps_all64_table_entries_and_verbatim_context(tmp_path):
    candidates = [fixtures.candidate(f"native-{i:03}", 15 + i % 16 * 61, 15 + i // 16 * 180,
                                     source="ocr") | {"text": "原始OCR文字abcdefghijklmnopqrstuv"} for i in range(64)]
    row = fixtures.make_sample(tmp_path, candidates=candidates)
    row.update(instruction="完整目标\n包含中文  空格", actor_step="当前子任务\n不由标签决定")
    fixtures.bind(row)
    prompt = build_visual_user_prompt(row["instruction"], row["actor_step"], row["candidate_snapshot"],
                                      visual_prompt_version=DEDUP_PROMPT_VERSION)
    table = "\n".join(f"{i} {center(c['bbox'])[0]} {center(c['bbox'])[1]} {canonical_json(c['text'][:24])}"
                      for i, c in enumerate(select_candidates(row["candidate_snapshot"], instruction=row["instruction"],
                                                            actor_step=row["actor_step"]), 1))
    assert f"Native size: 1024x768\n{table}\nOverall goal: {row['instruction']}\nCurrent subtask: {row['actor_step']}" in prompt
    for rule in ("complete ordered JSON action array", "copy requested text exactly", "no later subtasks",
                 "action_type", "click/double_click/right_click/move: candidate,dx,dy",
                 "scroll: candidate,dx,dy,scroll_amount (nonzero integer)",
                 "drag: start_candidate,start_dx,start_dy,end_candidate,end_dx,end_dy",
                 "type_text: text; key_press: key; hotkey: keys; wait: duration",
                 "No other fields/x/y", "IDs/offsets are integers", "both drag endpoints", "native bounds",
                 "annotations, not UI controls", "Badge number=Table ID", "original native center",
                 "never click badges", "proposals, not control evidence", "If no region fits, return []"):
        assert rule in prompt


@pytest.mark.parametrize("version", [DEDUP_PROMPT_VERSION, COMPACT_PROMPT_VERSION])
def test_new_prompt_preserves_full_assistant_target_and_never_reads_it_for_input(tmp_path, version):
    row = fixtures.make_sample(tmp_path)
    target = [{"action_type": "type_text", "text": "不可泄漏的GT\n  Unicode"},
              {"action_type": "hotkey", "keys": ["ctrl", "a"]}, {"action_type": "wait", "duration": .25}]
    row.update(visual_prompt_version=version, actions=copy.deepcopy(target), candidate_target=copy.deepcopy(target))
    review = tmp_path / "closed.json"
    review.write_text(canonical_json({"schema": "week5.pointer_semantic_review.v1", "status": "closed", "reviews": []}), encoding="utf8")
    row["candidate_label_evidence"] = {"review_path": str(review), "review_sha256": file_sha256(review)}
    full = build_visual_pointer_messages(row, include_target=True, visual_prompt_version=version)
    assert full[:1] == build_visual_pointer_messages(row, visual_prompt_version=version)
    assert "不可泄漏" not in full[0]["content"][1]["text"]
    assert json.loads(full[-1]["content"]) == target
    changed = copy.deepcopy(row)
    changed["candidate_target"] = object()
    changed["actions"] = object()
    changed["candidate_label_evidence"] = object()
    assert build_visual_pointer_messages(changed, visual_prompt_version=version) == full[:1]


@pytest.mark.parametrize("value", ["", "unknown", False, 1, None])
def test_explicit_invalid_or_mixed_version_refused(tmp_path, value):
    row = fixtures.make_sample(tmp_path)
    row["visual_prompt_version"] = value
    with pytest.raises(ValueError):
        build_visual_pointer_messages(row, visual_prompt_version=DEDUP_PROMPT_VERSION)
