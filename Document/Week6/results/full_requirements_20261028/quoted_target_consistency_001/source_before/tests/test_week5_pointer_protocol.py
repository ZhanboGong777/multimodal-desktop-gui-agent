"""CPU contract tests; synthetic requests are not model-effect measurements."""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest
from PIL import Image

from gui_agent.finetuning import pointer_protocol as pp
from gui_agent.finetuning.evaluation import HFConfig, canonical_json, file_sha256


@pytest.fixture
def pointer_sample(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 80), "white").save(image)
    snapshot = {"image_width": 100, "image_height": 80, "image_file_sha256": file_sha256(image),
                "candidates": [{"candidate_id": "c001", "source": "contour", "role": "contour_region",
                                "bbox": {"left": 10, "top": 10, "right": 50, "bottom": 30},
                                "text": "", "confidence": 0.5}]}
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf8")
    record = {"sample_id": "s", "action_index": 0, "decision": "accepted",
              "preferred_candidate_id": "c001", "image_file_sha256": file_sha256(image),
              "snapshot_path": str(snapshot_path), "snapshot_file_sha256": file_sha256(snapshot_path)}
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({"schema": "week5.pointer_semantic_review.v1", "status": "closed",
                                      "reviews": [record]}), encoding="utf8")
    return {"sample_id": "s", "image_path": str(image), "image_width": 100, "image_height": 80,
            "instruction": "Enter the requested text", "actor_step": "Focus field and type 中文 ",
            "actions": [{"action_type": "click", "x": 20, "y": 19}, {"action_type": "type_text", "text": "中文 \n"}],
            "candidate_target": [{"action_type": "click", "candidate": 1, "dx": -10, "dy": -1}, {"action_type": "type_text", "text": "中文 \n"}],
            "candidate_snapshot": snapshot,
            "candidate_label_evidence": {"review_path": str(review_path), "review_sha256": file_sha256(review_path)}}


def test_source_quotas_prevent_ocr_starvation_and_keep_spatial_diversity():
    candidates = []
    for source in ("ocr", "contour"):
        for index in range(80):
            cell, offset = index % 16, index // 16
            x, y = cell % 4 * 100 + offset * 3, cell // 4 * 100 + (source == "contour") * 20
            candidates.append({"candidate_id": f"c{len(candidates)+1:03d}", "source": source,
                               "role": pp.ROLES[source], "confidence": 0.9 if source == "ocr" else 0.5,
                               "text": "Label" if source == "ocr" else "",
                               "bbox": {"left": x, "top": y, "right": x+2, "bottom": y+2}})
    snapshot = {"image_width": 400, "image_height": 400, "candidates": candidates}
    selected = pp.select_candidates(snapshot)
    assert len(selected) == 64
    assert Counter(c["source"] for c in selected) == {"ocr": 32, "contour": 32}
    for source in pp.ROLES:
        assert len({(pp.center(c["bbox"])[0]//100, pp.center(c["bbox"])[1]//100)
                    for c in selected if c["source"] == source}) == 16
    assert pp.select_candidates({**snapshot, "candidates": list(reversed(candidates))}) == selected
    assert pp.select_candidates({**snapshot, "actions": "HIDDEN", "instruction": "HIDDEN"}) == selected


def test_small_pool_keeps_all_candidates_and_prompt_uses_only_request_evidence(pointer_sample):
    sample = pointer_sample
    sample.update(actions=[{"text": "HIDDEN_GT"}], candidate_target="HIDDEN_TARGET",
                  candidate_label_evidence="HIDDEN_REVIEW", source="HIDDEN_SOURCE")
    sample["candidate_snapshot"]["annotations"] = "HIDDEN_BOX"
    sample["candidate_snapshot"]["candidates"][0]["text"] = "a" * 40
    assert len(pp.select_candidates(sample["candidate_snapshot"])) == 1
    prefix = pp.build_pointer_messages(sample)
    text = prefix[0]["content"][1]["text"]
    assert "HIDDEN" not in text and '1 30 20 "'+"a"*24+'"' in text
    assert "a"*25 not in text
    assert sample["actor_step"] in text


def test_whole_target_mapping_and_label_provenance_do_not_replace_native_gt(pointer_sample):
    original = copy.deepcopy(pointer_sample["actions"])
    pp.validate_pointer_sample(pointer_sample)
    messages = pp.build_pointer_messages(pointer_sample, include_target=True)
    assert json.loads(messages[1]["content"]) == pointer_sample["candidate_target"]
    assert pp.map_pointer_actions(messages[1]["content"], pointer_sample["candidate_snapshot"], pointer_sample["image_path"]) == [
        {"action_type": "click", "x": 20, "y": 19}, {"action_type": "type_text", "text": "中文 \n"}]
    assert pointer_sample["actions"] == original


@pytest.mark.parametrize("raw", [
    "[]", "{}", '```json\n[]\n```', '[{"action_type":"click","candidate":true,"dx":0,"dy":0}]',
    '[{"action_type":"click","candidate":0,"dx":0,"dy":0}]', '[{"action_type":"click","candidate":65,"dx":0,"dy":0}]',
    '[{"action_type":"click","candidate":1.0,"dx":0,"dy":0}]', '[{"action_type":"click","candidate":"1","dx":0,"dy":0}]',
    '[{"action_type":"click","candidate":1,"candidate":1}]',
    '[{"action_type":"click","candidate":1,"dx":0,"dy":0,"x":30}]',
    '[{"action_type":"scroll","candidate":1,"dx":0,"dy":0,"scroll_amount":0}]',
    '[{"action_type":"drag","start_candidate":1}]',
    '[{"action_type":"click","candidate":1,"dx":true,"dy":0}]',
    '[{"action_type":"click","candidate":1,"dx":0.0,"dy":0}]',
    '[{"action_type":"click","candidate":1,"dx":0}]',
    '[{"action_type":"click","candidate":1,"dx":-31,"dy":0}]',
])
def test_strict_no_repair_mapping(pointer_sample, raw):
    with pytest.raises((ValueError, TypeError)):
        pp.map_pointer_actions(raw, pointer_sample["candidate_snapshot"], pointer_sample["image_path"])


@pytest.mark.parametrize("change", ["partial", "text", "offset", "review_abstain", "review_bytes", "snapshot_bytes", "image_bytes"])
def test_incomplete_or_changed_supervision_is_refused(pointer_sample, change):
    from pathlib import Path

    sample = pointer_sample
    evidence = sample["candidate_label_evidence"]
    review_path = Path(evidence["review_path"])
    review = json.loads(review_path.read_text("utf8"))
    if change == "partial":
        sample["candidate_target"] = sample["candidate_target"][:1]
    elif change == "text":
        sample["candidate_target"][1]["text"] = "中文"
    elif change == "offset":
        sample["candidate_target"][0]["dx"] += 1
    elif change == "review_abstain":
        review["reviews"][0]["decision"] = "abstain"
        review_path.write_text(json.dumps(review), encoding="utf8")
        evidence["review_sha256"] = file_sha256(review_path)
    elif change == "review_bytes":
        review_path.write_text("{}", encoding="utf8")
    elif change == "snapshot_bytes":
        Path(review["reviews"][0]["snapshot_path"]).write_text("{}", encoding="utf8")
    else:
        Image.new("RGB", (100, 80), "black").save(sample["image_path"])
    with pytest.raises((ValueError, TypeError)):
        pp.validate_pointer_sample(sample)


def test_shared_inference_prefix_accepts_no_label_fields_and_preserves_bad_raw(pointer_sample):
    sample = {k: v for k, v in pointer_sample.items() if k not in {"actions", "candidate_target", "candidate_label_evidence"}}
    prefix = pp.build_pointer_messages(sample)[0]["content"][1]["text"]
    raw = '[{"action_type":"click","candidate":999}]'
    calls = []

    class Worker:
        def request(self, image, prompt, *, timeout_seconds):
            calls.append((image, prompt, timeout_seconds))
            return {"status": "ok", "generation": raw}

    result = pp.request_pointer_actions(Worker(), sample, timeout_seconds=10)
    assert calls == [(sample["image_path"], prefix, 10)]
    assert result["raw_reply"]["generation"] == raw
    assert result["mapped_actions"] is None and result["mapping_error"]


def test_pointer_request_rejects_native_constraint_before_model_call(pointer_sample):
    class Worker:
        def __init__(self):
            self.ready = {"output_constraint": {"mode": "json_actions"}, "constraint_hook": "prefix_allowed_tokens_fn"}

        def request(self, *args, **kwargs):
            pytest.fail("incompatible worker must be refused before a request")

    with pytest.raises(ValueError, match="unconstrained"):
        pp.request_pointer_actions(Worker(), pointer_sample, timeout_seconds=10)


@pytest.mark.parametrize("field,value", [("action_type", "double_click"), ("current_subtask", "Changed task"), ("target_point", {"x": 21, "y": 19})])
def test_review_source_context_must_match_native_target(pointer_sample, field, value):
    from pathlib import Path

    evidence = pointer_sample["candidate_label_evidence"]
    path = Path(evidence["review_path"])
    review = json.loads(path.read_text("utf8"))
    review["reviews"][0][field] = value
    path.write_text(json.dumps(review), encoding="utf8")
    evidence["review_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="semantic review"):
        pp.validate_pointer_sample(pointer_sample)


def test_candidate_protocol_refuses_native_generation_constraint():
    HFConfig({"target_protocol": "candidate_compact"}, output_constraint="none")
    with pytest.raises(ValueError, match="candidate_compact"):
        HFConfig({"target_protocol": "candidate_compact"}, output_constraint="json_actions")


def test_v2_requires_all_semantic_source_fields(pointer_sample):
    from pathlib import Path

    sample = pointer_sample
    sample["candidate_snapshot"]["compact_policy"] = "control_balanced_v2"
    evidence = sample["candidate_label_evidence"]
    review_path = Path(evidence["review_path"])
    review = json.loads(review_path.read_text("utf8"))
    snapshot_path = Path(review["reviews"][0]["snapshot_path"])
    snapshot_path.write_text(json.dumps(sample["candidate_snapshot"]), encoding="utf8")
    review["reviews"][0]["snapshot_file_sha256"] = file_sha256(snapshot_path)
    review_path.write_text(json.dumps(review), encoding="utf8")
    evidence["review_sha256"] = file_sha256(review_path)
    with pytest.raises(ValueError, match="v2 semantic review"):
        pp.validate_pointer_sample(sample)
    review["reviews"][0].update(action_type="click", current_subtask=sample["actor_step"],
                                target_point={"x": 20, "y": 19})
    review_path.write_text(json.dumps(review), encoding="utf8")
    evidence["review_sha256"] = file_sha256(review_path)
    pp.validate_pointer_sample(sample)


def test_scroll_supervision_preserves_amount(pointer_sample):
    pointer_sample["actions"][0] = {"action_type": "scroll", "x": 20, "y": 19, "scroll_amount": -3}
    pointer_sample["candidate_target"][0] = {"action_type": "scroll", "candidate": 1, "dx": -10, "dy": -1, "scroll_amount": -2}
    with pytest.raises(ValueError, match="scroll amount"):
        pp.validate_pointer_sample(pointer_sample)


def test_drag_scroll_and_no_candidate_nonmouse_mapping(pointer_sample):
    snapshot, image = pointer_sample["candidate_snapshot"], pointer_sample["image_path"]
    assert pp.map_pointer_actions(canonical_json([
        {"action_type": "drag", "start_candidate": 1, "start_dx": -2, "start_dy": 1,
         "end_candidate": 1, "end_dx": 3, "end_dy": -4},
        {"action_type": "scroll", "candidate": 1, "dx": 0, "dy": 0, "scroll_amount": -2}]), snapshot, image) == [
        {"action_type": "drag", "start": {"x": 28, "y": 21}, "end": {"x": 33, "y": 16}},
        {"action_type": "scroll", "x": 30, "y": 20, "scroll_amount": -2}]
    assert pp.map_pointer_actions('[{"action_type":"key_press","key":"ENTER"}]',
                                  {**snapshot, "candidates": []}, image) == [{"action_type": "key_press", "key": "ENTER"}]
