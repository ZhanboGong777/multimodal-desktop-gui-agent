"""Synthetic CPU tests for preparation stages; no detector/model is executed."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gui_agent.finetuning.candidate_grounding import DetectorConfig
from gui_agent.finetuning.evaluation import file_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/week5_prepare_pointer.py"
SPEC = importlib.util.spec_from_file_location("week5_prepare_pointer_test_module", SCRIPT)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Other agents may edit protocol sources concurrently; synthetic tests bind
    # their script, while real invocations bind the production modules too.
    monkeypatch.setattr(prep, "_sources", lambda: {str(SCRIPT): file_sha256(SCRIPT)})
    train_actions = [
        {"action_type": "click", "x": 20, "y": 19},
        {"action_type": "double_click", "x": 22, "y": 20},
        {"action_type": "right_click", "x": 24, "y": 21},
        {"action_type": "move", "x": 26, "y": 22},
        {"action_type": "scroll", "x": 28, "y": 23, "scroll_amount": -3},
        {"action_type": "drag", "start": {"x": 30, "y": 24}, "end": {"x": 70, "y": 55}},
        {"action_type": "type_text", "text": "中文 \n"},
        {"action_type": "hotkey", "keys": ["ctrl", "s"]},
        {"action_type": "wait", "duration": 0.25},
    ]
    paths = {}
    for split, color, actions in (
        ("train", "white", train_actions),
        ("validation", "black", [{"action_type": "type_text", "text": "free answer"}]),
    ):
        image_path = tmp_path / f"{split}.png"
        image = Image.new("RGB", (100, 80), color)
        image.save(image_path)
        pixel_sha = hashlib.sha256(b"100x80:" + image.tobytes()).hexdigest()
        row = {"sample_id": split + ":sample", "group_id": split + ":group",
               "image_path": str(image_path), "image_width": 100, "image_height": 80,
               "image_sha256": pixel_sha, "instruction": "Test complete actions",
               "actor_step": "Use the shown controls and type 中文 \n",
               "source": "synthetic", "source_split": "train", "actions": actions,
               "action_annotations": [{} for _ in actions]}
        paths[split] = tmp_path / f"{split}.jsonl"
        paths[split].write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    config = DetectorConfig(max_candidates=512, contour_budget=200, contour_max_aspect=64)
    calls = []

    def detector(image_path, actual_config):
        calls.append((image_path, actual_config))
        return {"image_width": 100, "image_height": 80,
                "image_file_sha256": file_sha256(image_path),
                "detector_config": asdict(actual_config), "candidates": [
                    {"candidate_id": "c001", "source": "contour", "role": "contour_region",
                     "bbox": {"left": 10, "top": 10, "right": 50, "bottom": 30},
                     "confidence": 0.5, "text": ""},
                    {"candidate_id": "c002", "source": "ocr", "role": "text_region",
                     "bbox": {"left": 60, "top": 40, "right": 80, "bottom": 60},
                     "confidence": 0.9, "text": "Field"},
                ]}

    return SimpleNamespace(paths=paths, config=config, detector=detector, calls=calls,
                           detection=tmp_path / "detection", tmp_path=tmp_path)


def detect(data, *, resume=False, detector=None, **kwargs):
    return prep.detect_phase(data.paths["train"], data.paths["validation"], data.detection,
                             config=kwargs.pop("config", data.config), detector=detector or data.detector,
                             resume=resume, **kwargs)


def closed_review(data):
    review = json.loads((data.detection / "semantic_review_template.json").read_text("utf-8"))
    review["status"] = "closed"
    for record in review["reviews"]:
        record.update(decision="accepted", preferred_candidate_id="c001")
    path = data.tmp_path / "review.json"
    path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    return path, review


class FakeProcessor:
    def __init__(self, *, overflow=False, omit_eos=False, mismatch=False):
        self.calls = []
        self.overflow, self.omit_eos, self.mismatch = overflow, omit_eos, mismatch
        self.tokenizer = SimpleNamespace(
            pad_token_id=0,
            convert_tokens_to_ids=lambda token: {
                "<|vision_start|>": 2, "<|image_pad|>": 3,
                "<|video_pad|>": 5, "<|vision_end|>": 4, "<|im_end|>": 9,
            }[token],
        )

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        return "PREFIX" if add_generation_prompt else "FULL:" + messages[-1]["content"]

    def __call__(self, *, text, images, padding, return_tensors):
        assert len(images) == 1 and padding is False and return_tensors == "pt"
        self.calls.append(text[0])
        ids = [10, 2, 3, 4, 11]
        if text[0].startswith("FULL:"):
            if self.mismatch:
                ids[0] = 99
            ids += [12] * (1536 if self.overflow else 1)
            ids += [8 if self.omit_eos else 9, 13]
        return {"input_ids": [ids], "image_grid_thw": [[1, 2, 2]]}


def finish(data, review_path, *, processor=None, output_name="finished"):
    processor = processor or FakeProcessor()
    directory = data.tmp_path / "processor"
    directory.mkdir(exist_ok=True)
    (directory / "processor_config.json").write_text("{}", encoding="utf-8")
    result = prep.finish_phase(data.detection, review_path, data.tmp_path / output_name,
                              processor_path=directory, processor_factory=lambda _: processor)
    return result, processor


def test_detect_has_no_auto_labels_and_emits_all_mouse_endpoint_slots(dataset):
    result = detect(dataset)
    assert result["rows"] == result["unique_images"] == len(dataset.calls) == 2
    assert result["mouse_review_slots"] == 7
    assert result["automatic_semantic_acceptances"] == result["model_loads"] == 0
    template = json.loads((dataset.detection / "semantic_review_template.json").read_text("utf-8"))
    assert template["status"] == "draft"
    assert all(r["decision"] == "pending" and r["preferred_candidate_id"] is None
               for r in template["reviews"])
    drag = [r for r in template["reviews"] if r["action_type"] == "drag"]
    assert [r["endpoint"] for r in drag] == ["start_", "end_"]
    rows = [json.loads(line) for line in (dataset.detection / "rows_and_snapshots.jsonl").read_text("utf-8").splitlines()]
    for entry in rows:
        original = json.loads(dataset.paths[entry["split"]].read_text("utf-8"))
        assert entry["sample"] == original
        assert "candidate_target" not in entry["sample"]


def test_finish_exact_whole_roundtrip_full_ids_masks_and_literal_diagnostics(dataset):
    detect(dataset)
    review_path, _ = closed_review(dataset)
    result, processor = finish(dataset, review_path)
    assert result["status"] == "closed_CPU_preparation_only"
    assert result["actual_encode_calls"] == len(processor.calls) == 4
    assert result["splits"]["train"]["admitted"] == result["splits"]["validation"]["admitted"] == 1
    prepared = [json.loads(line) for line in (dataset.tmp_path / "finished/all_rows.jsonl").read_text("utf-8").splitlines()]
    for entry in prepared:
        row = entry["sample"]
        assert row["actions"] == json.loads(dataset.paths[entry["split"]].read_text("utf-8"))["actions"]
        assert prep.map_pointer_actions(json.dumps(row["candidate_target"]), row["candidate_snapshot"], row["image_path"]) == row["actions"]
    target = prepared[0]["sample"]["candidate_target"]
    assert target[4]["scroll_amount"] == -3
    assert set(target[5]) == {"action_type", "start_candidate", "start_dx", "start_dy",
                              "end_candidate", "end_dx", "end_dy"}
    assert target[6]["text"] == "中文 \n"
    tokens = json.loads((dataset.tmp_path / "finished/actual_token_ids.json").read_text("utf-8"))
    assert all(t["labels"] == [-100] * 5 + [12, 9, 13] for t in tokens)
    decisions = json.loads((dataset.tmp_path / "finished/decisions.json").read_text("utf-8"))
    assert not decisions[1]["literal_visibility"][0]["exact_substring_in_step"]
    assert decisions[1]["literal_visibility"][0]["unique_literal_contract_proven"] is False


@pytest.mark.parametrize("field", ["action_type", "current_subtask", "target_point",
                                  "image_file_sha256", "snapshot_file_sha256", "endpoint"])
def test_closed_review_requires_all_strict_bound_fields(dataset, field):
    detect(dataset)
    path, review = closed_review(dataset)
    del review["reviews"][0][field]
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        finish(dataset, path)
    assert not (dataset.tmp_path / "finished/result.json").exists()


@pytest.mark.parametrize("change", ["duplicate", "missing", "point", "subtask", "foreign_dataset", "bool_index"])
def test_review_duplicates_missing_or_tampered_context_are_rejected(dataset, change):
    detect(dataset)
    path, review = closed_review(dataset)
    if change == "duplicate":
        review["reviews"].append(copy.deepcopy(review["reviews"][0]))
    elif change == "missing":
        review["reviews"].pop()
    elif change == "point":
        review["reviews"][0]["target_point"]["x"] += 1
    elif change == "subtask":
        review["reviews"][0]["current_subtask"] = "Different"
    elif change == "foreign_dataset":
        review["dataset_rows_sha256"] = "bad"
    else:
        review["reviews"][0]["action_index"] = True
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError):
        finish(dataset, path)


@pytest.mark.parametrize("reason", ["abstain", "not_selected"])
def test_whole_blocked_row_is_kept_without_partial_target_or_gt_fallback(dataset, monkeypatch, reason):
    detect(dataset)
    path, review = closed_review(dataset)
    if reason == "abstain":
        review["reviews"][5].update(decision="abstain", preferred_candidate_id=None)
    else:
        review["reviews"][5]["preferred_candidate_id"] = "c002"
        monkeypatch.setattr(prep, "select_candidates", lambda snapshot: [snapshot["candidates"][0]])
    path.write_text(json.dumps(review), encoding="utf-8")
    result, processor = finish(dataset, path)
    assert result["splits"]["train"]["all_rows"] == 1
    assert result["splits"]["train"]["semantic_complete"] == result["splits"]["train"]["admitted"] == 0
    assert result["actual_encode_calls"] == len(processor.calls) == 3
    row = json.loads((dataset.tmp_path / "finished/all_rows.jsonl").read_text("utf-8").splitlines()[0])["sample"]
    assert "candidate_target" not in row
    assert len(row["actions"]) == 9 and row["actions"][6]["text"] == "中文 \n"


def test_overlong_full_sequences_are_saved_without_truncation_or_admission(dataset):
    detect(dataset)
    path, _ = closed_review(dataset)
    result, _ = finish(dataset, path, processor=FakeProcessor(overflow=True))
    assert result["splits"]["train"]["known_full_overflow"] == 1
    assert result["splits"]["validation"]["admitted"] == 0
    tokens = json.loads((dataset.tmp_path / "finished/actual_token_ids.json").read_text("utf-8"))
    assert all(len(t["full"]["input_ids"]) > 1536 for t in tokens)


@pytest.mark.parametrize("bad", ["omit_eos", "mismatch"])
def test_failed_mask_or_end_token_does_not_write_success_receipt(dataset, bad):
    detect(dataset)
    path, _ = closed_review(dataset)
    with pytest.raises(ValueError):
        finish(dataset, path, processor=FakeProcessor(**{bad: True}))
    output = dataset.tmp_path / "finished"
    assert not (output / "result.json").exists()
    failures = list(output.glob("failure_finish_*.json"))
    assert len(failures) == 1 and json.loads(failures[0].read_text("utf-8"))["accepted_as_success"] is False


def test_detect_resume_uses_immutable_completed_cache_without_redetection(dataset):
    attempted = []

    def interrupted(image, config):
        attempted.append(image)
        if len(attempted) == 2:
            raise RuntimeError("synthetic interruption")
        return dataset.detector(image, config)

    with pytest.raises(RuntimeError):
        detect(dataset, detector=interrupted)
    assert not (dataset.detection / "result.json").exists()
    result = detect(dataset, resume=True)
    assert result["detector_calls_this_invocation"] == result["cache_reuses_this_invocation"] == 1
    assert len(dataset.calls) == 2
    with pytest.raises(FileExistsError):
        detect(dataset, resume=True)


def test_resume_rejects_changed_cached_snapshot_without_repair(dataset):
    def interrupted(image, config):
        if dataset.calls:
            raise RuntimeError("stop")
        return dataset.detector(image, config)

    with pytest.raises(RuntimeError):
        detect(dataset, detector=interrupted)
    snapshot = next((dataset.detection / "snapshots").glob("*.json"))
    snapshot.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        detect(dataset, resume=True)
    assert len(dataset.calls) == 1


def test_test_records_and_group_or_image_overlap_refused_before_detection(dataset):
    original = json.loads(dataset.paths["validation"].read_text("utf-8"))
    original["source_split"] = "official_test"
    dataset.paths["validation"].write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="official test"):
        detect(dataset)
    assert not dataset.calls
    original["source_split"] = "train"
    original["group_id"] = "train:group"
    dataset.paths["validation"].write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        detect(dataset)
    assert not dataset.calls


def test_no_overwrite_of_closed_finish_outputs(dataset):
    detect(dataset)
    path, _ = closed_review(dataset)
    finish(dataset, path)
    before = file_sha256(dataset.tmp_path / "finished/result.json")
    with pytest.raises(FileExistsError):
        finish(dataset, path)
    assert file_sha256(dataset.tmp_path / "finished/result.json") == before


def v3_config(*, language="eng"):
    return DetectorConfig(**prep.DETECTOR_OPTIONS, ocr_language=language, contour_text_lines=True)


@pytest.mark.parametrize("policy", prep.COMPACT_POLICIES)
def test_detect_freezes_requested_policy_without_computing_selection(dataset, monkeypatch, policy):
    def forbidden_selection(*args, **kwargs):
        pytest.fail("detection must not compute selected candidate retention before review")

    monkeypatch.setattr(prep, "select_candidates", forbidden_selection)
    config = v3_config() if policy == "query_relations_v3" else dataset.config
    result = detect(dataset, compact_policy=policy, config=config)
    assert result["compact_policy"] == policy
    template = json.loads((dataset.detection / "semantic_review_template.json").read_text("utf-8"))
    assert all(("instruction" in r) == (policy == "query_relations_v3") for r in template["reviews"])
    for path in (dataset.detection / "snapshots").glob("*.json"):
        snapshot = json.loads(path.read_text("utf-8"))
        assert snapshot["compact_policy"] == policy
        assert snapshot["ocr_language"] == snapshot["detector_config"]["ocr_language"] == "eng"


@pytest.mark.parametrize("previous", [None, "existing-parent-prefix"])
@pytest.mark.parametrize("fail", [False, True])
def test_local_assets_bound_and_environment_restored_on_success_and_failure(dataset, monkeypatch, previous, fail):
    if previous is None:
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    else:
        monkeypatch.setenv("TESSDATA_PREFIX", previous)
    directory = dataset.tmp_path / "tessdata"
    directory.mkdir()
    for name in ("eng", "chi_sim"):
        (directory / f"{name}.traineddata").write_bytes(("synthetic-" + name).encode())
    config = DetectorConfig(max_candidates=512, contour_budget=200, contour_max_aspect=64,
                            ocr_language="eng+chi_sim")

    def detector(image, actual_config):
        import os

        assert os.environ["TESSDATA_PREFIX"] == str(directory.resolve())
        assert actual_config.ocr_language == "eng+chi_sim"
        if fail:
            raise RuntimeError("synthetic OCR failure")
        return dataset.detector(image, actual_config)

    if fail:
        with pytest.raises(RuntimeError, match="synthetic"):
            detect(dataset, detector=detector, config=config, ocr_language="eng+chi_sim",
                   tessdata_dir=directory)
    else:
        result = detect(dataset, detector=detector, config=config, ocr_language="eng+chi_sim",
                        tessdata_dir=directory)
        assert result["tessdata_directory"] == str(directory.resolve())
        assert result["traineddata_files_sha256"] == {
            str(directory.resolve() / f"{name}.traineddata"): file_sha256(directory / f"{name}.traineddata")
            for name in ("eng", "chi_sim")}
    import os

    assert os.environ.get("TESSDATA_PREFIX") == previous
    assert ("TESSDATA_PREFIX" in os.environ) == (previous is not None)


def test_missing_local_traineddata_refused_before_detector_or_environment_change(dataset, monkeypatch):
    monkeypatch.setenv("TESSDATA_PREFIX", "existing")
    directory = dataset.tmp_path / "incomplete-tessdata"
    directory.mkdir()
    (directory / "eng.traineddata").write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="traineddata"):
        detect(dataset, ocr_language="eng+chi_sim", tessdata_dir=directory)
    import os

    assert os.environ["TESSDATA_PREFIX"] == "existing" and not dataset.calls
    assert not dataset.detection.exists()


@pytest.mark.parametrize("change", ["policy", "language", "source", "asset"])
def test_resume_rejects_changed_policy_language_source_or_asset(dataset, monkeypatch, change):
    directory = dataset.tmp_path / "tessdata"
    directory.mkdir()
    (directory / "eng.traineddata").write_bytes(b"synthetic-eng")
    (directory / "chi_sim.traineddata").write_bytes(b"synthetic-chi")

    def interrupted(image, config):
        if dataset.calls:
            raise RuntimeError("synthetic stop")
        return dataset.detector(image, config)

    with pytest.raises(RuntimeError):
        detect(dataset, detector=interrupted, tessdata_dir=directory)
    kwargs = {"tessdata_dir": directory}
    if change == "policy":
        kwargs["compact_policy"] = "spatial_v1"
    elif change == "language":
        kwargs.update(ocr_language="eng+chi_sim", config=DetectorConfig(
            max_candidates=512, contour_budget=200, contour_max_aspect=64,
            ocr_language="eng+chi_sim"))
    elif change == "source":
        monkeypatch.setattr(prep, "_sources", lambda: {str(SCRIPT): "changed-source"})
    else:
        (directory / "eng.traineddata").write_bytes(b"changed-asset")
    with pytest.raises(ValueError, match="resume"):
        detect(dataset, resume=True, **kwargs)
    assert len(dataset.calls) == 1


def test_v3_finish_binds_shared_context_and_preserves_every_native_field(dataset):
    detect(dataset, compact_policy="query_relations_v3", config=v3_config())
    path, _ = closed_review(dataset)
    result, _ = finish(dataset, path)
    assert result["compact_policy"] == "query_relations_v3"
    rows = [json.loads(line)["sample"] for line in
            (dataset.tmp_path / "finished/all_rows.jsonl").read_text("utf-8").splitlines()]
    for row in rows:
        context = {"instruction": row["instruction"], "actor_step": row["actor_step"]}
        binding = prep.selection_binding(row["candidate_snapshot"], **context)
        assert row["candidate_selection_evidence"] == binding
        assert prep.map_pointer_actions(json.dumps(row["candidate_target"]), row["candidate_snapshot"],
                                        row["image_path"], **context, expected_selection=binding) == row["actions"]
        with pytest.raises(ValueError):
            prep.map_pointer_actions(json.dumps(row["candidate_target"]), row["candidate_snapshot"],
                                     row["image_path"], **context)
        changed = {**binding, "context_sha256": "different"}
        with pytest.raises(ValueError):
            prep.map_pointer_actions(json.dumps(row["candidate_target"]), row["candidate_snapshot"],
                                     row["image_path"], **context, expected_selection=changed)


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_v3_external_review_instruction_is_mandatory_and_bound(dataset, change):
    detect(dataset, compact_policy="query_relations_v3", config=v3_config())
    path, review = closed_review(dataset)
    if change == "missing":
        del review["reviews"][0]["instruction"]
    else:
        review["reviews"][0]["instruction"] = "Different goal"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="instruction"):
        finish(dataset, path)
    assert not (dataset.tmp_path / "finished/result.json").exists()


def test_unknown_policy_wrong_raw_language_and_missing_v3_context_fail_closed(dataset):
    with pytest.raises(ValueError, match="unknown"):
        detect(dataset, compact_policy="unknown")
    assert not dataset.calls

    def wrong_language(image, config):
        snapshot = dataset.detector(image, config)
        snapshot["detector_config"]["ocr_language"] = "other"
        return snapshot

    with pytest.raises(ValueError, match="raw snapshot OCR language"):
        detect(dataset, detector=wrong_language)
    assert not (dataset.detection / "result.json").exists()
    entries = [{"split": "train", "sample": json.loads(dataset.paths["train"].read_text("utf-8"))}]
    entries[0]["sample"]["actor_step"] = ""
    dataset.paths["train"].write_text(json.dumps(entries[0]["sample"]), encoding="utf-8")
    with pytest.raises(ValueError, match="actor_step"):
        prep.detect_phase(dataset.paths["train"], dataset.paths["validation"],
                          dataset.tmp_path / "missing-context", config=v3_config(),
                          detector=dataset.detector, compact_policy="query_relations_v3")


def test_v3_rejects_nonfixed_producer_configuration_before_detection(dataset):
    config = DetectorConfig(max_candidates=512, contour_budget=200, contour_max_aspect=64,
                            contour_text_lines=True, ocr_scale=1, ocr_merge_aligned=False)
    with pytest.raises(ValueError, match="fixed raw detector"):
        detect(dataset, compact_policy="query_relations_v3", config=config)
    assert not dataset.calls


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_policy_language_and_asset_flags_keep_old_defaults(dataset, monkeypatch, explicit):
    captured = {}
    output = dataset.tmp_path / "cli-output"

    def fake_detect(train, validation, directory, **kwargs):
        captured.update(train=train, validation=validation, **kwargs)
        target = Path(directory)
        target.mkdir()
        result = {"status": "closed_detection_only", "inputs_sha256": {},
                  "source_sha256": {}, "artifacts_sha256": {}}
        prep._json(target / "result.json", result)
        return result

    monkeypatch.setattr(prep, "detect_phase", fake_detect)
    argv = ["detect", "--train", str(dataset.paths["train"]), "--validation",
            str(dataset.paths["validation"]), "--output-dir", str(output)]
    if explicit:
        argv += ["--compact-policy", "query_relations_v3", "--ocr-language", "eng+chi_sim",
                 "--tessdata-dir", "local-assets"]
    assert prep.main(argv) == 0
    assert captured["compact_policy"] == ("query_relations_v3" if explicit else "control_balanced_v2")
    assert captured["ocr_language"] == ("eng+chi_sim" if explicit else "eng")
    assert captured["tessdata_dir"] == ("local-assets" if explicit else None)
