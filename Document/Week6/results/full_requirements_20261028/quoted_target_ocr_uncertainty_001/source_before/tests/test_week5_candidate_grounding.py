"""CPU synthetic safety tests; no replies here count as model measurements."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from gui_agent.finetuning import candidate_grounding as cg
from gui_agent.finetuning.evaluation import file_sha256


@pytest.fixture
def scene(tmp_path):
    path = tmp_path / "screen.png"
    image = Image.new("RGB", (120, 80), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 40, 30), outline="black", width=2)
    image.save(path)
    snapshot = {
        "image_file_sha256": file_sha256(path),
        "image_width": 120,
        "image_height": 80,
        "candidates": [
            {
                "candidate_id": "c001",
                "role": "contour_region",
                "source": "contour",
                "text": "",
                "confidence": 0.5,
                "bbox": {"left": 10, "top": 10, "right": 40, "bottom": 30},
            }
        ],
    }
    return path, snapshot


def selection(**changes):
    return json.dumps(
        [
            {
                "action_type": "click",
                "candidate_id": "c001",
                "candidate_role": "contour_region",
                **changes,
            }
        ]
    )


def test_detect_pixels_only_and_record_explicit_ocr_fallback(scene, monkeypatch):
    path, _snapshot = scene
    monkeypatch.setattr(
        cg,
        "_ocr_proposals",
        lambda image, config: (
            [],
            {"requested": "tesseract", "actual_engine": None, "error": "not installed"},
        ),
    )
    detected = cg.detect_candidates(path, cg.DetectorConfig())
    assert detected["mode"] == "contour_only" and detected["ocr"]["error"] == "not installed"
    assert detected["candidates"] and all(c["source"] == "contour" for c in detected["candidates"])
    with pytest.raises(TypeError):
        cg.detect_candidates(path, cg.DetectorConfig(), action_annotations=[{"target_bbox": {}}])


def test_contour_detection_is_deterministic_with_native_boxes(scene):
    path, _snapshot = scene
    a = cg.detect_candidates(path, cg.DetectorConfig(ocr="none"))
    b = cg.detect_candidates(path, cg.DetectorConfig(ocr="none"))
    assert a["candidates"] == b["candidates"]
    assert cg.snapshot_hash(a) == cg.snapshot_hash(b)
    cg.validate_snapshot(a, path)


def test_mapping_uses_current_native_center_and_never_label_coordinates(scene):
    path, snapshot = scene
    assert cg.map_selection(selection(), snapshot, path) == [
        {"action_type": "click", "x": 25, "y": 20}
    ]
    assert cg.map_selection(selection(action_type="scroll", scroll_amount=-2), snapshot, path) == [
        {"action_type": "scroll", "x": 25, "y": 20, "scroll_amount": -2}
    ]


@pytest.mark.parametrize(
    "reply",
    [
        "[]",
        "{}",
        "```json\n[]\n```",
        '[{"action_type":"click","candidate_id":"c001","candidate_id":"c099","candidate_role":"contour_region"}]',
        selection(candidate_id="c099"),
        selection(candidate_id=1),
        selection(candidate_role="editor"),
        selection(x=25),
        selection(description="looks correct"),
        selection(action_type="scroll", scroll_amount=True),
        selection(action_type="scroll", scroll_amount=0),
    ],
)
def test_no_id_guess_repair_role_override_or_free_coordinates(scene, reply):
    path, snapshot = scene
    with pytest.raises((ValueError, TypeError)):
        cg.map_selection(reply, snapshot, path)


def test_stale_image_no_candidates_and_duplicate_id_are_refused(scene):
    path, snapshot = scene
    empty = {**snapshot, "candidates": []}
    with pytest.raises(ValueError, match="no unambiguous"):
        cg.map_selection(selection(), empty, path)
    duplicate = copy.deepcopy(snapshot)
    duplicate["candidates"].append(copy.deepcopy(duplicate["candidates"][0]))
    with pytest.raises(ValueError, match="ambiguous"):
        cg.map_selection(selection(), duplicate, path)
    Image.new("RGB", (120, 80), "black").save(path)
    with pytest.raises(ValueError, match="stale"):
        cg.map_selection(selection(), snapshot, path)


def test_candidate_role_is_detection_evidence_not_model_description(scene):
    path, snapshot = scene
    snapshot["candidates"][0]["role"] = "button"
    with pytest.raises(ValueError, match="role lacks"):
        cg.map_selection(selection(candidate_role="button"), snapshot, path)


def test_bbox_bounds_and_integer_types_are_strict(scene):
    path, snapshot = scene
    for box in [
        {"left": True, "top": 10, "right": 40, "bottom": 30},
        {"left": -1, "top": 10, "right": 40, "bottom": 30},
        {"left": 10, "top": 10, "right": 121, "bottom": 30},
    ]:
        bad = copy.deepcopy(snapshot)
        bad["candidates"][0]["bbox"] = box
        with pytest.raises(ValueError):
            cg.map_selection(selection(), bad, path)


def test_prompt_excludes_annotation_actions_sample_metadata_even_if_extra_keys_present(scene):
    _path, snapshot = scene
    snapshot.update(
        actions=[{"text": "SECRET_LABEL"}],
        action_annotations=[{"target_bbox": "SECRET_BOX"}],
        sample_id="SECRET_ID",
    )
    prompt = cg.candidate_prompt("Click the visible control", snapshot)
    assert "SECRET" not in prompt
    assert "c001" in prompt and "contour_region" in prompt and "10" in prompt


def test_coverage_without_annotation_never_invents_target_bbox(scene):
    _path, snapshot = scene
    scored = cg.coverage_score([{"action_type": "click", "x": 20, "y": 20}], [{}], snapshot)
    assert scored["click_targets"] == scored["target_point_inside_candidate"] == 1
    assert scored["bbox_eligible"] == scored["candidate_center_hits_annotation"] == 0
    assert scored["exact_candidate_center"] == 0
    assert scored["closest_candidate_center_normalized_distances"] == [5 / 120]


def test_real_annotation_coverage_is_oracle_not_model_selected_success(scene):
    _path, snapshot = scene
    scored = cg.coverage_score(
        [{"action_type": "click", "x": 26, "y": 21}],
        [{"target_bbox": {"left": 20, "top": 15, "right": 30, "bottom": 25}}],
        snapshot,
    )
    assert scored["candidate_center_hits_annotation"] == scored["bbox_eligible"] == 1
    assert "strict_action_exact" not in scored


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location(
        "week5_candidate_test_cli",
        Path(__file__).parents[1] / "scripts/week5_candidate_grounding.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coverage_cpu_full_denominator_and_no_model_or_annotation_input(
    scene, tmp_path, monkeypatch, cli
):
    path, snapshot = scene
    samples = [
        {
            "sample_id": "s1",
            "group_id": "g1",
            "source_sample_sha256": "src",
            "image_path": str(path),
            "image_file_sha256": file_sha256(path),
            "image_sha256": "pixels",
            "image_width": 120,
            "image_height": 80,
            "actions": [{"action_type": "click", "x": 20, "y": 20}],
            "action_annotations": [{}],
        }
    ]
    monkeypatch.setattr(cli, "samples", lambda _: samples)
    config = tmp_path / "config.json"
    cli.write_json(config, cli.TrainingConfig().to_dict())
    binding = {
        "training_dataset": {"path": str(path), "sha256": file_sha256(path)},
        "validation_dataset": {"path": str(path), "sha256": file_sha256(path)},
        "training_image_files": {},
    }
    monkeypatch.setattr(cli, "development_binding", lambda *_args: binding)
    calls = []

    def detect(image_path, config):
        calls.append((image_path, asdict(config)))
        return {**snapshot, "mode": "contour_only", "detection_ms": 1}

    monkeypatch.setattr(cli, "detect_candidates", detect)
    monkeypatch.setattr(
        cli, "HFWorker", lambda *args, **kwargs: pytest.fail("CPU coverage must never load a model")
    )
    assert (
        cli.main(
            [
                "coverage",
                "--dataset",
                str(path),
                "--training-config",
                str(config),
                "--ocr",
                "none",
                "--output",
                str(tmp_path / "coverage"),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "coverage/summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 1 and summary["totals"]["target_point_inside_candidate"] == 1
    assert summary["candidate_center_annotation_coverage_rate"] is None
    assert calls == [(str(path), asdict(cg.DetectorConfig(ocr="none")))]


def test_duplicate_bbox_is_refused_independently_of_dictionary_order(scene):
    path, snapshot = scene
    candidate = copy.deepcopy(snapshot["candidates"][0])
    candidate["candidate_id"] = "c002"
    candidate["bbox"] = {key: candidate["bbox"][key] for key in ("bottom", "right", "top", "left")}
    snapshot["candidates"].append(candidate)
    with pytest.raises(ValueError, match="ambiguous"):
        cg.validate_snapshot(snapshot, path)


def test_no_candidates_has_finite_missing_distance_penalty(scene):
    _path, snapshot = scene
    snapshot["candidates"] = []
    score = cg.coverage_score([{"action_type": "click", "x": 20, "y": 20}], [{}], snapshot)
    assert score["closest_candidate_center_normalized_distances"] == [math.sqrt(2)]
    assert score["target_point_inside_candidate"] == 0


@pytest.fixture
def experiment(scene, tmp_path, monkeypatch, cli):
    path, snapshot = scene
    snapshot["detection_ms"] = 500
    snapshot["detector_config"] = asdict(cg.DetectorConfig())
    records = [
        {
            "sample_id": f"s{i}",
            "group_id": f"g{i}",
            "source_sample_sha256": f"src{i}",
            "image_path": str(path),
            "image_file_sha256": file_sha256(path),
            "image_sha256": "rgb",
            "image_width": 120,
            "image_height": 80,
            "instruction": "Click the visible control",
            "actions": [{"action_type": "click", "x": 20, "y": 20}],
            "action_annotations": [{}],
        }
        for i in range(3)
    ]
    monkeypatch.setattr(cli, "samples", lambda _: records)
    # Synthetic dependencies are frozen separately from the concurrently edited worktree.
    monkeypatch.setattr(cli, "code_manifest", lambda _: {"fixture.py": "frozen"})
    monkeypatch.setattr(cli, "producer_sources", lambda: {"fixture.py": "producer"})
    dataset = tmp_path / "validation.jsonl"
    dataset.write_text("CPU synthetic fixture, not real inference data", encoding="utf-8")
    binding = {
        "training_dataset": {"path": str(dataset), "sha256": file_sha256(dataset)},
        "validation_dataset": {"path": str(dataset), "sha256": file_sha256(dataset)},
        "training_image_files": {},
    }
    monkeypatch.setattr(cli, "development_binding", lambda *_args: binding)
    cache = tmp_path / "cached"
    cache.mkdir()
    cli.write_json(
        cache / "metadata.json",
        {
            "schema_version": cg.EXPERIMENT_VERSION,
            "scope": "validation_candidate_coverage_only",
            "samples": cli.identities(records),
            "dataset_sha256": file_sha256(dataset),
            "source_files": cli.producer_sources(),
            "detector_config": asdict(cg.DetectorConfig()),
            "development_binding": binding,
        },
    )
    cli.write_json(cache / "summary.json", {"provenance_unchanged": True, "samples": 3})
    for sample in records:
        cli.append(
            cache / "candidates.jsonl",
            {
                "sample_id": sample["sample_id"],
                "source_sample_sha256": sample["source_sample_sha256"],
                "status": "ok",
                "snapshot": snapshot,
                "snapshot_sha256": cg.snapshot_hash(snapshot),
            },
        )
    artifact_paths = {}
    for name in ("base", "adapter", "processor"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "artifact.json").write_text("CPU fixture", encoding="utf-8")
        artifact_paths[name] = directory
    config = tmp_path / "config.json"
    cli.write_json(
        config,
        cli.TrainingConfig(
            base_model_path=str(artifact_paths["base"]),
            processor_path=str(artifact_paths["processor"]),
        ).to_dict(),
    )
    output = tmp_path / "experiment"
    argv = [
        "run",
        "--dataset",
        str(dataset),
        "--coverage",
        str(cache),
        "--training-config",
        str(config),
        "--adapter",
        str(artifact_paths["adapter"]),
        "--output",
        str(output),
        "--mode",
        "candidate",
        "--max-new-tokens",
        "16",
    ]
    return {
        "records": records,
        "cache": cache,
        "config": config,
        "output": output,
        "argv": argv,
        "image": path,
        "snapshot": snapshot,
        "artifacts": artifact_paths,
    }


def reply(status="ok", **changes):
    return {
        "status": status,
        "generation": selection(),
        "duration_ms": 5,
        "generated_tokens": 10,
        **changes,
    }


def install_worker(cli, monkeypatch, responses, *, constructor=None, cleanup=None):
    instances = []
    pending = iter(responses)

    class CPUFixtureWorker:
        def __init__(self, *_args, **_kwargs):
            instances.append(self)
            self.ready = {"CPU_synthetic_fixture": True}
            self.closed = 0
            self.calls = 0
            if constructor:
                constructor()

        def request(self, image_path, prompt, **_kwargs):
            assert "SECRET_LABEL" not in prompt and "target_bbox" not in prompt
            self.calls += 1
            # Every new process warms before a measured request.
            result = reply() if self.calls == 1 else next(pending)
            if isinstance(result, BaseException):
                raise result
            return result() if callable(result) else result

        def close(self):
            self.closed += 1
            if cleanup:
                cleanup()

    monkeypatch.setattr(cli, "HFWorker", CPUFixtureWorker)
    return instances


def evidence(cli, experiment):
    rows = [
        json.loads(line)
        for line in (experiment["output"] / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((experiment["output"] / "summary.json").read_text(encoding="utf-8"))
    return [row for row in rows if row["record_type"] == "request"], summary, rows


def test_bad_candidate_id_preserves_generation_and_never_uses_labels(cli, experiment, monkeypatch):
    bad = selection(candidate_id="c099")
    instances = install_worker(cli, monkeypatch, [reply(generation=bad)] * 3)
    assert cli.main(experiment["argv"]) == 0  # Completed study; semantic errors are results.
    rows, summary, _ = evidence(cli, experiment)
    assert len(instances) == 1 and instances[0].closed == 1
    assert all(row["generation"] == bad and row["mapped_generation"] == "" for row in rows)
    assert summary["requests"] == 3 and summary["totals"]["strict_action_exact"] == 0
    assert summary["statuses"] == {"error": 3}


def test_completed_typed_token_limits_stay_resident_but_all_scores_fail(
    cli, experiment, monkeypatch
):
    instances = install_worker(
        cli, monkeypatch, [reply("error", error_kind="token_limit", generated_tokens=16)] * 3
    )
    assert cli.main(experiment["argv"]) == 0
    rows, summary, _ = evidence(cli, experiment)
    assert len(instances) == 1 and instances[0].calls == 4
    assert summary["statuses"] == {"error": 3} and summary["totals"]["strict_json_valid"] == 0
    assert all(
        row["generation"] and row["model_response"]["error_kind"] == "token_limit" for row in rows
    )


@pytest.mark.parametrize(
    "failure",
    [
        reply("timeout"),
        reply("error"),
        reply("error", error_kind="token_limit", generated_tokens=True),
        RuntimeError("transport broken"),
    ],
)
def test_timeouts_unknown_errors_and_transport_close_then_reload(
    cli, experiment, monkeypatch, failure
):
    instances = install_worker(cli, monkeypatch, [failure, reply(), reply()])
    assert cli.main(experiment["argv"]) == 1
    rows, summary, _ = evidence(cli, experiment)
    assert len(instances) == 2 and all(worker.closed == 1 for worker in instances)
    assert summary["requests"] == 3 and rows[0]["status"] != "ok"
    assert sum(row["status"] == "ok" for row in rows) == 2


def test_cancellation_records_unmeasured_denominator_without_fabricated_latency(
    cli, experiment, monkeypatch
):
    instances = install_worker(cli, monkeypatch, [KeyboardInterrupt()])
    assert cli.main(experiment["argv"]) == 130
    rows, summary, _ = evidence(cli, experiment)
    assert len(instances) == 1 and instances[0].closed == 1
    assert summary["requests"] == 3 and summary["statuses"] == {"cancelled": 1, "not_started": 2}
    assert all(row["duration_ms"] is None and not row["inference_attempted"] for row in rows[1:])
    assert summary["duration_missing"] == summary["unmeasured_samples"] == 2


def test_startup_failure_does_not_repeatedly_load_model_for_remaining_samples(
    cli, experiment, monkeypatch
):
    def fail():
        raise RuntimeError("load failed")

    instances = install_worker(cli, monkeypatch, [], constructor=fail)
    assert cli.main(experiment["argv"]) == 1
    _rows, summary, _ = evidence(cli, experiment)
    assert len(instances) == 1 and summary["statuses"] == {"load_or_warmup_error": 3}
    assert summary["inference_attempted"] == 0 and summary["model_request_ms"]["n"] == 0


def test_postrun_provenance_exception_retains_raw_and_summary(cli, experiment, monkeypatch):
    install_worker(cli, monkeypatch, [reply()] * 3)
    calls = [0]

    def manifest(_root):
        calls[0] += 1
        if calls[0] > 1:
            raise OSError("source unavailable")
        return {"fixture.py": "frozen"}

    monkeypatch.setattr(cli, "code_manifest", manifest)
    assert cli.main(experiment["argv"]) == 1
    rows, summary, raw = evidence(cli, experiment)
    assert len(rows) == 3 and all(row["status"] == "ok" for row in rows)
    assert (
        not summary["provenance_unchanged"] and "source unavailable" in summary["provenance_error"]
    )
    assert (
        summary["statuses"] == {"provenance_changed": 3}
        and summary["totals"]["strict_json_valid"] == 0
    )
    assert raw[-1]["record_type"] == "provenance_verdict"


@pytest.mark.parametrize("interrupted_scan", ["source", "artifact"])
def test_interrupt_during_final_provenance_preserves_measured_outputs_and_summary(
    cli, experiment, monkeypatch, interrupted_scan
):
    instances = install_worker(cli, monkeypatch, [reply()] * 3)
    if interrupted_scan == "source":
        calls = [0]

        def manifest(_root):
            calls[0] += 1
            if calls[0] > 1:
                raise KeyboardInterrupt
            return {"fixture.py": "frozen"}

        monkeypatch.setattr(cli, "code_manifest", manifest)
    else:
        original = cli.artifact_manifest
        calls = [0]

        def artifacts(path):
            calls[0] += 1
            # The first three reads freeze base/adapter/processor before load.
            if calls[0] == 4:
                raise KeyboardInterrupt
            return original(path)

        monkeypatch.setattr(cli, "artifact_manifest", artifacts)
    assert cli.main(experiment["argv"]) == 130
    rows, summary, raw = evidence(cli, experiment)
    assert len(rows) == summary["requests"] == 3
    assert all(row["status"] == "ok" and row["generation"] == selection() for row in rows)
    assert instances[0].closed == 1
    assert summary["interrupted"] and not summary["provenance_unchanged"]
    assert summary["statuses"] == {"provenance_changed": 3}
    assert summary["totals"]["strict_action_exact"] == 0
    assert "KeyboardInterrupt" in summary["provenance_error"]
    assert raw[-1]["record_type"] == "provenance_verdict" and not raw[-1]["unchanged"]


@pytest.mark.parametrize("changed", ["config", "processor", "cache", "image"])
def test_mutated_inputs_cannot_score_as_success_and_original_generation_is_kept(
    cli, experiment, monkeypatch, changed
):
    calls = [0]

    def mutate():
        calls[0] += 1
        if calls[0] == 3:
            if changed == "config":
                experiment["config"].write_text("modified", encoding="utf-8")
            elif changed == "processor":
                (experiment["artifacts"]["processor"] / "artifact.json").write_text(
                    "changed", encoding="utf-8"
                )
            elif changed == "cache":
                with (experiment["cache"] / "candidates.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write("\n")
            else:
                Image.new("RGB", (120, 80), "black").save(experiment["image"])
        return reply()

    install_worker(cli, monkeypatch, [mutate] * 3)
    assert cli.main(experiment["argv"]) == 1
    rows, summary, _ = evidence(cli, experiment)
    assert rows[-1]["generation"] == selection()
    assert not summary["provenance_unchanged"] and summary["totals"]["strict_json_valid"] == 0


def test_duplicate_cache_samples_reject_before_worker_or_output(cli, experiment, monkeypatch):
    cache_file = experiment["cache"] / "candidates.jsonl"
    first = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
    cli.append(cache_file, first)
    monkeypatch.setattr(
        cli, "HFWorker", lambda *_args, **_kwargs: pytest.fail("must reject before GPU load")
    )
    assert cli.main(experiment["argv"]) == 2 and not experiment["output"].exists()


def test_worker_cleanup_failure_cannot_discard_request_ledger(cli, experiment, monkeypatch):
    def fail():
        raise RuntimeError("cleanup failed")

    install_worker(cli, monkeypatch, [reply()] * 3, cleanup=fail)
    assert cli.main(experiment["argv"]) == 1
    rows, summary, raw = evidence(cli, experiment)
    assert len(rows) == summary["requests"] == 3 and summary["provenance_unchanged"]
    assert any(row["record_type"] == "worker_cleanup_error" for row in raw)


def test_cache_detection_is_explicit_estimate_not_live_e2e(cli, experiment, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(cli.time, "perf_counter", lambda: clock[0])
    original_prompt = cli.candidate_prompt

    def prompt(*args):
        clock[0] += 1
        return original_prompt(*args)

    monkeypatch.setattr(cli, "candidate_prompt", prompt)
    instances = install_worker(cli, monkeypatch, [reply()] * 3)
    original_request = instances  # Installation does not construct a process.
    worker_type = cli.HFWorker
    original_init, original_call = worker_type.__init__, worker_type.request

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        clock[0] += 10

    def request(self, *args, **kwargs):
        clock[0] += 20 if self.calls == 0 else 5
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(worker_type, "__init__", initialize)
    monkeypatch.setattr(worker_type, "request", request)
    assert cli.main(experiment["argv"]) == 0
    rows, summary, _ = evidence(cli, experiment)
    assert len(original_request) == 1
    assert rows[0]["duration_ms"] == 6000 and rows[0]["excluded_model_load_warmup_ms"] == 30000
    assert (
        rows[0]["cached_detection_ms"] == 500 and rows[0]["estimated_recompute_pipeline_ms"] == 6500
    )
    assert summary["observed_request_wall_ms"]["p50"] == 6000
    metadata = json.loads((experiment["output"] / "metadata.json").read_text(encoding="utf-8"))
    assert "NOT observed live E2E" in metadata["estimated_recompute_latency_scope"]


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_invalid_contour_aspect_configuration_is_refused(value):
    with pytest.raises(ValueError, match="aspect"):
        cg.DetectorConfig(contour_max_aspect=value)


def test_aspect_configuration_is_local_and_recovers_pixel_rectangle(tmp_path):
    path = tmp_path / "wide-control.png"
    image = Image.new("RGB", (1000, 200), "white")
    ImageDraw.Draw(image).rectangle((20, 50, 900, 80), outline="black", width=2)
    image.save(path)
    normal = cg.detect_candidates(path, cg.DetectorConfig(ocr="none", max_candidates=200))
    wide = cg.detect_candidates(
        path, cg.DetectorConfig(ocr="none", max_candidates=200, contour_max_aspect=64)
    )
    assert normal["detector_config"]["contour_max_aspect"] == 12
    assert wide["detector_config"]["contour_max_aspect"] == 64
    assert not any(cg.contains(c["bbox"], 450, 65) for c in normal["candidates"])
    assert any(cg.contains(c["bbox"], 450, 65) for c in wide["candidates"])
    assert all(c["role"] == "contour_region" for c in wide["candidates"])


def test_development_binding_refuses_train_or_missing_saved_validation(cli, tmp_path, monkeypatch):
    train, val = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text("train", encoding="utf-8")
    val.write_text("validation", encoding="utf-8")
    config = cli.TrainingConfig(train_jsonl=str(train), validation_jsonl=str(val))
    monkeypatch.setattr(
        cli,
        "load_prepared_records",
        lambda *_args: pytest.fail("must reject before reading other records"),
    )
    with pytest.raises(ValueError, match="saved isolated validation"):
        cli.development_binding(config, str(train), [])
    with pytest.raises(ValueError, match="saved train and validation"):
        cli.development_binding(cli.TrainingConfig(), str(val), [])


@pytest.mark.parametrize("duplicate_key", ["sample_id", "group_id", "image_sha256"])
def test_validation_binding_checks_sample_group_and_decoded_image_isolation(
    cli, tmp_path, monkeypatch, duplicate_key
):
    train, val = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text("train", encoding="utf-8")
    val.write_text("validation", encoding="utf-8")
    train_row = {
        "sample_id": "train",
        "group_id": "train-session",
        "image_sha256": "train-rgb",
        "image_path": "fixture",
        "image_file_sha256": "fixture",
    }
    val_row = {"sample_id": "val", "group_id": "val-session", "image_sha256": "val-rgb"}
    val_row[duplicate_key] = train_row[duplicate_key]
    monkeypatch.setattr(cli, "load_prepared_records", lambda *_args: [train_row])
    config = cli.TrainingConfig(train_jsonl=str(train), validation_jsonl=str(val))
    with pytest.raises(ValueError, match="leakage"):
        cli.development_binding(config, str(val), [val_row])


@pytest.mark.parametrize("bad", ["summary", "source", "binding", "detector"])
def test_untrusted_candidate_cache_rejects_before_worker_or_output(
    cli, experiment, monkeypatch, bad
):
    if bad == "summary":
        cli.write_json(
            experiment["cache"] / "summary.json", {"provenance_unchanged": False, "samples": 3}
        )
    elif bad == "detector":
        path = experiment["cache"] / "candidates.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["snapshot"]["detector_config"]["contour_max_aspect"] = 64
        rows[0]["snapshot_sha256"] = cg.snapshot_hash(rows[0]["snapshot"])
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    else:
        path = experiment["cache"] / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["source_files" if bad == "source" else "development_binding"] = {"wrong": True}
        cli.write_json(path, metadata)
    monkeypatch.setattr(
        cli,
        "HFWorker",
        lambda *_args, **_kwargs: pytest.fail("untrusted cache must not start a GPU worker"),
    )
    assert cli.main(experiment["argv"]) == 2 and not experiment["output"].exists()


def test_fixed_probe_subset_uses_only_seed_and_ids_same_for_both_routes(
    cli, experiment, monkeypatch
):
    instances = install_worker(cli, monkeypatch, [reply()] * 4)
    assert cli.main(experiment["argv"] + ["--limit", "2", "--seed", "42"]) == 0
    candidate_rows, summary, _ = evidence(cli, experiment)
    assert summary["planned_samples"] == summary["requests"] == 2
    baseline_output = experiment["output"].with_name("baseline")
    argv = list(experiment["argv"])
    argv[argv.index("--mode") + 1] = "baseline"
    argv[argv.index("--output") + 1] = str(baseline_output)
    # Changing scoring labels must not alter sample selection or candidate prompt.
    for row in experiment["records"]:
        row["actions"] = [{"action_type": "key_press", "key": "Enter"}]
    assert cli.main(argv + ["--limit", "2", "--seed", "42"]) == 0
    baseline_rows, _, _ = evidence(cli, {**experiment, "output": baseline_output})
    assert [r["sample_id"] for r in candidate_rows] == [r["sample_id"] for r in baseline_rows]
    assert len(instances) == 2


def install_ocr_rows(monkeypatch, rows, calls):
    """Stub only the external OCR call; production native geometry runs normally."""
    columns = ("text", "left", "top", "width", "height", "conf", "block_num")
    data = {key: [row[i] for row in rows] for i, key in enumerate(columns)}
    data.update(page_num=[1] * len(rows), par_num=[1] * len(rows), line_num=[1] * len(rows))

    def image_to_data(image, **kwargs):
        calls.append((image, kwargs))
        return data

    module = SimpleNamespace(
        get_tesseract_version=lambda: "CPU_fixture",
        image_to_data=image_to_data,
        Output=SimpleNamespace(DICT="DICT"),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", module)


@pytest.mark.parametrize("value", [0, 3, True, 1.0, "2"])
def test_ocr_scale_accepts_only_the_two_registered_integer_sizes(value):
    with pytest.raises(ValueError, match="OCR scale"):
        cg.DetectorConfig(ocr_scale=value)


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_ocr_aligned_merge_is_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        cg.DetectorConfig(ocr_merge_aligned=value)


def test_default_ocr_keeps_the_native_call_and_original_atomic_proposals(monkeypatch):
    calls = []
    install_ocr_rows(
        monkeypatch,
        [("Pin", 10, 10, 18, 12, "90", 1), ("taskbar", 32, 10, 39, 12, "80", 2)],
        calls,
    )
    image = Image.new("RGB", (120, 80), "white")
    proposals, info = cg._ocr_proposals(image, cg.DetectorConfig())
    assert calls[0][0] is image
    assert calls[0][1] == {
        "lang": "eng", "config": "--psm 11", "output_type": "DICT", "timeout": 15,
    }
    assert [p["text"] for p in proposals] == ["Pin", "taskbar"]
    assert all("evidence" not in p for p in proposals)
    assert info == {
        "requested": "tesseract", "actual_engine": "tesseract", "error": None,
        "version": "CPU_fixture",
    }


def test_scaled_ocr_maps_outward_native_edges_before_cross_block_parent_union(monkeypatch):
    calls = []
    install_ocr_rows(
        monkeypatch,
        [
            ("Pin", 21, 21, 35, 23, "90", 1),
            ("to", 60, 24, 19, 20, "85", 2),
            ("taskbar", 83, 21, 77, 23, "80", 3),
        ],
        calls,
    )
    image = Image.new("RGB", (240, 80), "white")
    proposals, info = cg._ocr_proposals(
        image, cg.DetectorConfig(ocr_scale=2, ocr_merge_aligned=True)
    )
    assert calls[0][0].size == (480, 160)
    assert [p["text"] for p in proposals] == ["Pin", "to", "taskbar", "Pin to taskbar"]
    parent = proposals[-1]
    assert parent["bbox"] == {"left": 10, "top": 10, "right": 80, "bottom": 22}
    assert parent["confidence"] == 0.8
    assert parent["evidence"]["kind"] == "aligned_ocr_parent"
    assert parent["evidence"]["children"] == [
        {k: p[k] for k in ("bbox", "text", "confidence")} for p in proposals[:3]
    ]
    assert info["scale"] == 2 and info["atomic_proposals"] == 3
    assert info["aligned_parent_proposals"] == 1


def ocr_proposal(text, left, top, right, bottom):
    return {
        "source": "ocr", "text": text, "confidence": 0.9,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
    }


def test_aligned_parent_refuses_separate_columns_rows_long_labels_and_wide_spans():
    separate = [
        ocr_proposal("File", 10, 10, 30, 22),
        ocr_proposal("Edit", 80, 10, 100, 22),
        ocr_proposal("Below", 32, 40, 50, 52),
        ocr_proposal("a" * 30, 150, 10, 170, 22),
        ocr_proposal("word", 174, 10, 196, 22),
    ]
    assert cg._aligned_ocr_parents(separate, 1000) == []
    # Nearby labels covering over 40% of this image stay independent.
    assert cg._aligned_ocr_parents(
        [ocr_proposal("First", 1, 10, 30, 22), ocr_proposal("Second", 31, 10, 60, 22)],
        100,
    ) == []


def test_aligned_parent_provenance_survives_dedup_without_discarding_its_atomic_child():
    proposals = [ocr_proposal("Longlabel", 10, 10, 90, 22), ocr_proposal("x", 91, 10, 94, 22)]
    parent = cg._aligned_ocr_parents(proposals, 1000)[0]
    assert cg._iou(parent["bbox"], proposals[0]["bbox"]) >= 0.85
    candidates = cg.proposals_to_candidates([*proposals, parent], 1000, 100, 512)
    assert {c["text"] for c in candidates} == {"Longlabel", "x", "Longlabel x"}
    assert next(c for c in candidates if c["text"] == "Longlabel x")["evidence"] == parent["evidence"]
    # Exact geometry remains unambiguous even when a repeated parent arrives.
    repeated = cg.proposals_to_candidates([*proposals, parent, parent], 1000, 100, 512)
    assert len(repeated) == 3


def test_scaled_ocr_drops_degenerate_boxes_and_clips_only_to_visible_native_pixels(monkeypatch):
    calls = []
    install_ocr_rows(
        monkeypatch,
        [("edge", 235, 155, 10, 10, "90", 1), ("outside", 260, 20, 10, 10, "90", 2)],
        calls,
    )
    proposals, _info = cg._ocr_proposals(
        Image.new("RGB", (120, 80), "white"), cg.DetectorConfig(ocr_scale=2)
    )
    assert len(proposals) == 1
    assert proposals[0]["bbox"] == {"left": 117, "top": 77, "right": 120, "bottom": 80}


def contour_proposal(left, top, right, bottom, confidence=0.5):
    return {
        "source": "contour", "text": "", "confidence": confidence,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
    }


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_contour_text_lines_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="contour text lines.*explicit boolean"):
        cg.DetectorConfig(contour_text_lines=value)


def test_contour_parent_reorders_baseline_cluster_left_to_right_and_keeps_atoms():
    # Baseline sorting alone would produce middle, right, left.
    atoms = [
        contour_proposal(35, 10, 50, 22, 0.6),
        contour_proposal(55, 11, 80, 23, 0.4),
        contour_proposal(10, 12, 30, 24),
    ]
    original = copy.deepcopy(atoms)
    parents = cg._contour_text_line_parents(atoms, 400, 100)
    assert atoms == original
    assert len(parents) == 1
    parent = parents[0]
    assert parent["bbox"] == {"left": 10, "top": 10, "right": 80, "bottom": 24}
    assert parent["source"] == "contour" and parent["text"] == "" and parent["confidence"] == 0.5
    assert parent["evidence"] == {
        "kind": "contour_text_line_parent",
        "policy": "native_baseline_gap_v1",
        "children": [atoms[index] for index in (2, 0, 1)],
    }
    assert cg._contour_text_line_parents(list(reversed(atoms)), 400, 100) == parents
    candidates = cg.proposals_to_candidates([*atoms, *parents], 400, 100, 512)
    assert len(candidates) == 4 and all(c["role"] == "contour_region" for c in candidates)
    # Evidence owns native bbox copies; it cannot mutate the original atoms.
    parent["evidence"]["children"][0]["bbox"]["left"] = 0
    assert atoms == original


def test_contour_parents_keep_menu_rows_and_distant_columns_separate():
    atoms = [
        contour_proposal(10, 10, 30, 22), contour_proposal(33, 10, 53, 22),
        contour_proposal(100, 10, 120, 22), contour_proposal(123, 10, 143, 22),
        contour_proposal(10, 34, 30, 46), contour_proposal(33, 34, 53, 46),
    ]
    parents = cg._contour_text_line_parents(atoms, 1000, 100)
    assert [p["bbox"] for p in parents] == [
        {"left": 10, "top": 10, "right": 53, "bottom": 22},
        {"left": 100, "top": 10, "right": 143, "bottom": 22},
        {"left": 10, "top": 34, "right": 53, "bottom": 46},
    ]
    assert sum(len(p["evidence"]["children"]) for p in parents) == len(atoms)


@pytest.mark.parametrize(
    "first,second",
    [
        # Size mismatch, baseline displacement, and insufficient vertical overlap.
        ((10, 10, 30, 22), (33, 5, 53, 22)),
        ((10, 10, 30, 22), (33, 14, 53, 26)),
        ((10, 14, 30, 26), (33, 6, 53, 22)),
        # Excessive spacing and overlapping contours do not form a text run.
        ((10, 10, 30, 22), (38, 10, 58, 22)),
        ((10, 10, 30, 22), (29, 10, 49, 22)),
        # Out-of-range word-like height/width; raw form controls stay independent.
        ((10, 10, 30, 15), (33, 10, 53, 15)),
        ((10, 10, 30, 43), (33, 10, 53, 43)),
        ((10, 10, 13, 22), (16, 10, 36, 22)),
        ((10, 10, 203, 22), (206, 10, 226, 22)),
    ],
)
def test_contour_parent_refuses_unaligned_overlapping_or_non_word_geometry(first, second):
    assert cg._contour_text_line_parents(
        [contour_proposal(*first), contour_proposal(*second)], 1000, 100,
    ) == []


def test_contour_parent_rows_do_not_accumulate_transitive_baseline_drift():
    atoms = [
        contour_proposal(10, 8, 20, 20),
        contour_proposal(23, 10, 33, 22),
        contour_proposal(36, 12, 46, 24),
    ]
    parents = cg._contour_text_line_parents(atoms, 400, 100)
    assert len(parents) == 1
    assert parents[0]["evidence"]["children"] == atoms[:2]


def test_contour_parent_rejects_a_long_dense_control_run_without_slicing_it():
    dense_controls = [contour_proposal(10 + i * 93, 10, 100 + i * 93, 22) for i in range(4)]
    assert cg._contour_text_line_parents(dense_controls, 1200, 100) == []
    # The image-width limit also applies to a short two-item run.
    assert cg._contour_text_line_parents(
        [contour_proposal(10, 10, 60, 22), contour_proposal(63, 10, 113, 22)], 200, 100,
    ) == []


def test_contour_parent_eight_child_limit_does_not_truncate_a_ninth_atom():
    atoms = [contour_proposal(i * 6, 10, i * 6 + 5, 22) for i in range(9)]
    assert len(cg._contour_text_line_parents(atoms[:8], 400, 100)[0]["evidence"]["children"]) == 8
    assert cg._contour_text_line_parents(atoms, 400, 100) == []
    assert len(cg.proposals_to_candidates(atoms, 400, 100, 512)) == 9


def test_contour_parent_preserves_bounds_and_requires_valid_native_children():
    atoms = [contour_proposal(0, 0, 30, 12), contour_proposal(33, 0, 60, 12)]
    parent = cg._contour_text_line_parents(atoms, 200, 100)[0]
    assert cg.native_box(parent["bbox"], 200, 100) == parent["bbox"]
    for child in parent["evidence"]["children"]:
        assert cg.native_box(child["bbox"], 200, 100) == child["bbox"]
    with pytest.raises(ValueError, match="outside the native screenshot"):
        cg._contour_text_line_parents([contour_proposal(190, 0, 201, 12)], 200, 100)


@pytest.mark.parametrize("height", [6, 32])
def test_contour_parent_accepts_the_inclusive_child_height_boundaries(height):
    atoms = [contour_proposal(0, 0, 16, height), contour_proposal(18, 0, 34, height)]
    parent = cg._contour_text_line_parents(atoms, 200, 100)[0]
    assert parent["bbox"] == {"left": 0, "top": 0, "right": 34, "bottom": height}


def test_contour_parent_accepts_exact_width_gap_and_font_ratio_thresholds():
    # h=10: widths .3h and 16h, gap .6h, all exactly on their limits.
    atoms = [contour_proposal(0, 0, 3, 10), contour_proposal(9, 0, 169, 10)]
    assert len(cg._contour_text_line_parents(atoms, 1000, 100)) == 1
    # The smaller font is exactly 75% of the larger font's height.
    assert len(cg._contour_text_line_parents(
        [contour_proposal(0, 4, 20, 16), contour_proposal(24, 0, 44, 16)], 400, 100,
    )) == 1


def test_contour_parent_rejects_excess_union_height_even_when_each_pair_aligns():
    atoms = [contour_proposal(0, 0, 10, 8)] + [
        contour_proposal(12 + i * 12, 3, 22 + i * 12, 10) for i in range(3)
    ]
    # Pairwise height ratio, bottom delta and overlap pass; 10 > 1.4*median(8,7,7,7).
    assert cg._contour_text_line_parents(atoms, 400, 100) == []


def test_contour_parent_iou_exemption_is_only_for_its_actual_children():
    atoms = [contour_proposal(10, 10, 90, 22), contour_proposal(91, 10, 96, 22)]
    parent = cg._contour_text_line_parents(atoms, 1000, 100)[0]
    assert cg._iou(parent["bbox"], atoms[0]["bbox"]) >= 0.85
    assert len(cg.proposals_to_candidates([*atoms, parent, copy.deepcopy(parent)], 1000, 100, 512)) == 3
    unrelated = contour_proposal(9, 10, 90, 22, confidence=0.9)
    kept = cg.proposals_to_candidates([*atoms, parent, unrelated], 1000, 100, 512)
    assert all("evidence" not in candidate for candidate in kept)
    # Even identical bbox/text is insufficient if it is not the recorded source/confidence.
    impostor = contour_proposal(10, 10, 90, 22, confidence=0.9)
    kept = cg.proposals_to_candidates([*atoms, parent, impostor], 1000, 100, 512)
    assert all("evidence" not in candidate for candidate in kept)
    unsupported = copy.deepcopy(parent)
    unsupported["evidence"]["kind"] = "unrecognized_parent"
    assert len(cg.proposals_to_candidates([*atoms, unsupported], 1000, 100, 512)) == 2


def test_contour_line_integration_is_opt_in_and_keeps_default_producer_behavior(tmp_path, monkeypatch):
    from gui_agent.perception import ui_detection

    path = tmp_path / "synthetic-word-contours.png"
    Image.new("RGB", (400, 100), "white").save(path)
    atoms = [contour_proposal(10, 10, 30, 22), contour_proposal(33, 10, 53, 22)]
    detector_calls, parent_calls = [], []

    def contours(image, **kwargs):
        detector_calls.append((image.size, kwargs))
        return [
            SimpleNamespace(
                bounding_box=SimpleNamespace(model_dump=lambda box=p["bbox"]: dict(box)),
                confidence=p["confidence"],
            )
            for p in atoms
        ]

    original_parent = cg._contour_text_line_parents

    def parents(*args):
        parent_calls.append(args)
        return original_parent(*args)

    monkeypatch.setattr(ui_detection, "detect_ui_candidates", contours)
    monkeypatch.setattr(cg, "_contour_text_line_parents", parents)
    default = cg.detect_candidates(path, cg.DetectorConfig(ocr="none"))
    explicit_false = cg.detect_candidates(path, cg.DetectorConfig(ocr="none", contour_text_lines=False))
    assert not parent_calls and default["detector_config"]["contour_text_lines"] is False
    assert default["candidates"] == cg.proposals_to_candidates(atoms, 400, 100, 64)
    assert default["candidates"] == explicit_false["candidates"]
    assert default["proposal_counts"] == {"ocr": 0, "contour": 2}
    enabled = cg.detect_candidates(path, cg.DetectorConfig(ocr="none", contour_text_lines=True))
    assert len(parent_calls) == 1 and len(detector_calls) == 3
    assert detector_calls[0] == detector_calls[1] == detector_calls[2]
    assert len(enabled["candidates"]) == 3
    assert enabled["proposal_counts"] == {"ocr": 0, "contour": 2, "contour_text_line_parents": 1}
    assert all(c["text"] == "" for c in enabled["candidates"])
    cg.validate_snapshot(enabled, path)
