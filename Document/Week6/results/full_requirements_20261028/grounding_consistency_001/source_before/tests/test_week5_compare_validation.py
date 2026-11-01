"""Small synthetic CPU provenance/paired-scoring tests; no model or desktop use."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.finetuning.training import TrainingConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/week5_compare_validation.py"
spec = importlib.util.spec_from_file_location("week5_compare_validation_test", SCRIPT)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _seal(directory, metadata):
    metadata["study_id"] = compare.value_sha256(metadata["frozen_before_generation"])
    metadata["raw_sha256"] = compare.file_sha256(directory / "raw.jsonl")
    _json(directory / "metadata.json", metadata)


def _raw(directory):
    return [json.loads(line) for line in (directory / "raw.jsonl").read_text(encoding="utf-8").splitlines()]


def _write_raw(directory, rows):
    (directory / "raw.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    _seal(directory, compare.read_json(directory / "metadata.json"))


def _study(tmp_path, name, *, bilingual=False, budgets=(384,), faults=None, modified_target=False):
    directory = tmp_path / name
    directory.mkdir()
    image = tmp_path / "common-image.png"
    if not image.exists():
        Image.new("RGB", (16, 16), "red").save(image)
    dataset = directory / "validation.jsonl"
    train = directory / "train.jsonl"
    _json(train, {"sample_id": "train-only", "group_id": "train-only", "image_sha256": "train-only"})
    targets = [{"action_type": "click", "x": 4, "y": 5}, {"action_type": "type_text", "text": "typed"}]
    if modified_target:
        targets[1]["text"] = "different target"
    base_rows = [{"sample_id": f"sample-{i:03d}", "group_id": f"group-{i // 4:03d}",
                  "image_path": str(image), "instruction": "点击后输入" if bilingual else "click then type",
                  "actions": targets, "action_annotations": [{"target_bbox": {"left": 0, "top": 0, "right": 8, "bottom": 8}}, {}],
                  "image_width": 16, "image_height": 16, "source": "synthetic", "source_split": "train"}
                 for i in range(128)]
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in base_rows), encoding="utf-8")
    samples = compare.load_prepared_records(dataset)
    selection = [{**{key: row[key] for key in compare.IDENTITY_FIELDS}, "source_sample_sha256": compare.value_sha256(row)} for row in samples]
    _json(directory / "selection.json", selection)
    artifacts = {role: {"path": f"/recorded/{role}", "files": {"saved": compare.value_sha256(role)}} for role in ("base", "adapter", "processor")}
    for role in artifacts.values():
        role["sha256"] = compare.value_sha256(role["files"])
    sources = {key: compare.file_sha256(compare.REPOSITORY / key) for key in compare.SCORING_SOURCES}
    sources["scripts/week5_validation_ablation.py"] = compare.value_sha256(name)  # Historical CLI versions may differ.
    frozen = {"scope": compare.SCOPE, "selection": selection, "dataset_sha256": compare.file_sha256(dataset),
              "source_files": sources, **artifacts, "training_config": {
                  **{key: getattr(TrainingConfig(), key) for key in compare.INFERENCE_FIELDS},
                  "validation_jsonl": str(dataset), "train_jsonl": str(train)},
              "pixel_budgets": list(budgets), "max_new_tokens": 256, "seed": 42}
    metadata = {"status": "completed", "frozen_before_generation": frozen,
                "provenance_after": {key: copy.deepcopy(frozen[key]) for key in ("dataset_sha256", "source_files", "base", "adapter", "processor")}}
    sid = compare.value_sha256(frozen)
    records = []
    for budget in budgets:
        for i, row in enumerate(samples):
            generation = json.dumps((faults or {}).get(i, row["actions"]), ensure_ascii=False)
            records.append({"record_type": "request", "study_id": sid, "variant": f"adapter_original_pixels_{budget}",
                            "sample_id": row["sample_id"], "group_id": row["group_id"], "image_path": row["image_path"],
                            "image_file_sha256": row["image_file_sha256"], "instruction": row["instruction"],
                            "target": {key: row.get(key, []) for key in compare.TARGET_FIELDS},
                            "status": "ok", "generation": generation, "inference_attempted": True, "image_unchanged": True,
                            "prompt": compare.build_user_prompt(compare.safe_prompt_input(row), "original")})
    records.append({"record_type": "provenance_verdict", "unchanged": True})
    (directory / "raw.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    _seal(directory, metadata)
    _json(directory / "summary.json", {"scope": compare.SCOPE, "complete": True, "interrupted": False,
          "provenance_unchanged": True, "planned_samples_per_cell": 128, "expected_cells": len(budgets),
          "cells": [{"variant": f"adapter_original_pixels_{budget}", "requests": 128, "inference_attempted": 128} for budget in budgets]})
    return directory


def test_full128_legacy_two_budgets_align_to_new_single384_and_keep_input_hash(tmp_path):
    old = _study(tmp_path, "old", budgets=(384, 1280))
    new = _study(tmp_path, "new", bilingual=True)
    result = compare.compare([compare.load_study("old_en", old), compare.load_study("new_cn", new)])
    assert result["scope"] == compare.SCOPE and result["sample_count"] == 128
    assert result["cells"][0]["input_sha256"] != result["cells"][1]["input_sha256"]
    assert result["cells"][0]["legacy_fields_absent"] == ["training_config_sha256", "image_file_sha256"]
    assert result["paired"][0]["instruction_changed_samples"] == 128
    assert result["paired"][0]["whole_action_exact"]["both_correct"] == 128
    assert result["cells"][0]["prediction_action_count_distribution"] == {"2": 128}


def test_paired_exact_type_sequence_and_text_slot_transitions_use_original_scorer(tmp_path):
    wrong_text = [{"action_type": "click", "x": 4, "y": 5}, {"action_type": "type_text", "text": "typed "}]
    wrong_kind = [{"action_type": "key_press", "key": "enter"}]
    omitted = [{"action_type": "click", "x": 4, "y": 5}]
    left = _study(tmp_path, "left", faults={0: wrong_text, 1: wrong_kind})
    right = _study(tmp_path, "right", faults={1: wrong_kind, 2: omitted})
    result = compare.compare([compare.load_study("left", left), compare.load_study("right", right)])
    paired = result["paired"][0]
    assert paired["whole_action_exact"]["improved"] == paired["whole_action_exact"]["worsened"] == 1
    assert paired["type_sequence_exact"]["improved"] == 0 and paired["type_sequence_exact"]["worsened"] == 1
    assert paired["text_exact_slots"]["improved"] == paired["text_exact_slots"]["worsened"] == 1
    assert paired["text_exact_slots"]["eligible"] == 128
    assert result["cells"][1]["prediction_action_count_distribution"] == {"2": 126, "1": 2}


@pytest.mark.parametrize("failure", ["incomplete", "missing_raw", "duplicate", "target", "verdict", "image", "source", "after", "selection", "official_test"])
def test_missing_or_unprovable_full128_evidence_refuses_comparison(tmp_path, failure):
    directory = _study(tmp_path, "study")
    metadata = compare.read_json(directory / "metadata.json")
    rows = _raw(directory)
    if failure == "incomplete":
        metadata["status"] = "incomplete"
        _json(directory / "metadata.json", metadata)
    elif failure in {"missing_raw", "duplicate", "target", "verdict"}:
        if failure == "missing_raw":
            rows.pop(0)
        elif failure == "duplicate":
            rows.insert(0, rows[0])
        elif failure == "target":
            rows[0]["target"]["actions"][1]["text"] = "changed target"
        else:
            rows[-1]["unchanged"] = False
        _write_raw(directory, rows)
    elif failure == "image":
        Image.new("RGB", (16, 16), "blue").save(tmp_path / "common-image.png")
    elif failure == "source":
        metadata["frozen_before_generation"]["source_files"][compare.SCORING_SOURCES[0]] = "wrong"
        metadata["provenance_after"]["source_files"] = copy.deepcopy(metadata["frozen_before_generation"]["source_files"])
        _seal(directory, metadata)
    elif failure == "after":
        metadata["provenance_after"]["dataset_sha256"] = "changed"
        _json(directory / "metadata.json", metadata)
    elif failure == "selection":
        selected = compare.read_json(directory / "selection.json")
        selected[0]["group_id"] = "changed"
        _json(directory / "selection.json", selected)
    else:
        dataset = Path(metadata["frozen_before_generation"]["training_config"]["validation_jsonl"])
        data = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
        data[0]["source_split"] = "test"
        dataset.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
        metadata["frozen_before_generation"]["dataset_sha256"] = compare.file_sha256(dataset)
        metadata["provenance_after"]["dataset_sha256"] = metadata["frozen_before_generation"]["dataset_sha256"]
        _seal(directory, metadata)
    with pytest.raises(ValueError):
        compare.load_study("study", directory)


def test_raw_bytes_tamper_is_rejected_even_when_128_records_remain(tmp_path):
    directory = _study(tmp_path, "study")
    with (directory / "raw.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="raw hash"):
        compare.load_study("study", directory)


@pytest.mark.parametrize("key", ["sample_id", "group_id", "image_sha256"])
def test_training_overlap_prevents_development_validation_comparison(tmp_path, key):
    directory = _study(tmp_path, "study")
    sample = compare.load_prepared_records(directory / "validation.jsonl")[0]
    train = {"sample_id": "train", "group_id": "train", "image_sha256": "train", key: sample[key]}
    _json(directory / "train.jsonl", train)
    with pytest.raises(ValueError, match="overlap"):
        compare.load_study("study", directory)


def test_cross_study_changed_target_cannot_be_paired_despite_complete_local_provenance(tmp_path):
    first = compare.load_study("first", _study(tmp_path, "first"))
    second = compare.load_study("second", _study(tmp_path, "second", modified_target=True))
    with pytest.raises(ValueError, match="actions/annotations differ"):
        compare.compare([first, second])


def test_failed_request_retains_denominator_and_action_count_failure_bucket(tmp_path):
    directory = _study(tmp_path, "study")
    rows = _raw(directory)
    rows[0]["status"] = "timeout"
    _write_raw(directory, rows)
    other = _study(tmp_path, "other")
    result = compare.compare([compare.load_study("study", directory), compare.load_study("other", other)])
    assert result["cells"][0]["whole_action_exact"] == 127
    assert result["cells"][0]["prediction_action_count_distribution"]["invalid_or_failed"] == 1
    assert result["paired"][0]["whole_action_exact"]["eligible"] == 128


def test_duplicate_independent_labels_are_refused(tmp_path):
    study = compare.load_study("same", _study(tmp_path, "study"))
    with pytest.raises(ValueError, match="distinct study labels"):
        compare.compare([study, study])


@pytest.mark.parametrize("key", ["training_config_sha256", "image_file_sha256"])
def test_new_hardening_fields_when_present_must_match_recorded_after(tmp_path, key):
    directory = _study(tmp_path, "study")
    metadata = compare.read_json(directory / "metadata.json")
    metadata["frozen_before_generation"][key] = "recorded-before"
    metadata["provenance_after"][key] = "changed-after"
    _seal(directory, metadata)
    with pytest.raises(ValueError, match="hardened before/after"):
        compare.load_study("study", directory)


def test_current_hardened_image_map_cannot_hide_one_missing_image(tmp_path):
    directory = _study(tmp_path, "study")
    metadata = compare.read_json(directory / "metadata.json")
    metadata["frozen_before_generation"]["image_file_sha256"] = {}
    metadata["provenance_after"]["image_file_sha256"] = {}
    _seal(directory, metadata)
    with pytest.raises(ValueError, match="hardened full image map"):
        compare.load_study("study", directory)


def test_other_quantization_cannot_be_silently_compared_as_the_same_inference(tmp_path):
    first = compare.load_study("first", _study(tmp_path, "first"))
    second = compare.load_study("second", _study(tmp_path, "second"))
    second["frozen"]["training_config"]["load_in_4bit"] = False
    with pytest.raises(ValueError, match="inference loading/processor"):
        compare.compare([first, second])


def _training_sources(tmp_path, change="writer"):
    before = "import time\n\ndef _write_json(path, value):\n    path.write_text(value)\n\ndef forward(batch):\n    return batch['labels']\n\ndef mask(labels):\n    return labels\n"
    after = before.replace("path.write_text(value)", "time.sleep(0.1)\n    path.write_text(value)")
    if change == "forward":
        after = after.replace("return batch['labels']", "return batch['prompt']")
    elif change == "mask":
        after = after.replace("return labels", "return labels[1:]")
    elif change == "other_function":
        after += "\ndef extra():\n    return 1\n"
    archive, current = tmp_path / "archived.py", tmp_path / "current.py"
    archive.write_text(before, encoding="utf-8")
    current.write_text(after, encoding="utf-8")
    return archive, current


def test_only_verified_writer_change_is_allowed_and_records_actual_shas(tmp_path):
    archive, current = _training_sources(tmp_path)
    result = compare.verify_writer_only_training_change(archive, current, compare.file_sha256(archive))
    assert result["ast_outside_write_json_identical"] is True
    assert result["historical_sha256"] == result["archived_sha256"]
    assert result["current_sha256"] != result["archived_sha256"]
    assert result["allowed_function"] == "_write_json"


@pytest.mark.parametrize("change", ["forward", "mask", "other_function"])
def test_training_change_outside_writer_cannot_hide_behind_io_fix(tmp_path, change):
    archive, current = _training_sources(tmp_path, change)
    with pytest.raises(ValueError, match="outside the sole allowed"):
        compare.verify_writer_only_training_change(archive, current, compare.file_sha256(archive))


def test_wrong_historical_archive_sha_is_refused_before_ast_exception(tmp_path):
    archive, current = _training_sources(tmp_path)
    with pytest.raises(ValueError, match="archive SHA"):
        compare.verify_writer_only_training_change(archive, current, "wrong-historical-sha")
