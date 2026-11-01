"""Regression checks for development ledger scoring and label isolation."""

import importlib.util
import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/week5_validation_ablation.py"
spec = importlib.util.spec_from_file_location("week5_validation_ablation_test", SCRIPT)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def request(*, status="ok", generation='[{"action_type":"click","x":4,"y":5}]', **extra):
    return {
        "record_type": "request", "variant": "test", "sample_id": "one",
        "status": status, "generation": generation, "inference_attempted": True,
        "duration_ms": 12,
        "target": {
            "actions": [{"action_type": "click", "x": 4, "y": 5}],
            "image_width": 20, "image_height": 20,
            "action_annotations": [{"target_bbox": {"left": 3, "top": 4, "right": 8, "bottom": 9}}],
        }, **extra,
    }


def test_scoring_preserves_original_exact_and_bbox_distinction():
    row = request(generation='[{"action_type":"click","x":6,"y":6}]')
    cell = ablation.summarize([row], unchanged=True)["cells"][0]
    assert cell["metrics"]["whole_action_exact"]["correct"] == 0
    assert cell["metrics"]["click_bbox_hit"]["correct"] == 1


@pytest.mark.parametrize("status", ["timeout", "error", "not_started"])
def test_failures_remain_in_denominators_without_credit(status):
    row = request(status=status, inference_attempted=status != "not_started", duration_ms=None)
    cell = ablation.summarize([row], unchanged=True)["cells"][0]
    assert cell["metrics"]["strict_json"] == {"correct": 0, "total": 1, "rate": 0}
    assert cell["metrics"]["click_bbox_hit"]["total"] == 1
    assert cell["percall_dispatch_to_reply_ms"]["n"] == 0


@pytest.mark.parametrize("unchanged,image_unchanged", [(False, True), (True, False)])
def test_changed_sources_or_images_cannot_receive_success(unchanged, image_unchanged):
    cell = ablation.summarize([request(image_unchanged=image_unchanged)], unchanged=unchanged)["cells"][0]
    assert cell["metrics"]["whole_action_exact"]["correct"] == 0
    assert cell["metrics"]["strict_json"]["correct"] == 0


def test_multiaction_omission_does_not_pass_or_change_target_slots():
    row = request()
    row["target"]["actions"].append({"action_type": "type_text", "text": "a b"})
    row["target"]["action_annotations"].append({})
    cell = ablation.summarize([row], unchanged=True)["cells"][0]
    assert cell["metrics"]["type_accuracy"]["total"] == 2
    assert cell["metrics"]["text_exact"] == {"correct": 0, "total": 1, "rate": 0}
    assert cell["metrics"]["whole_action_exact"]["correct"] == 0


def test_warmup_and_load_events_are_not_scored_as_requests():
    result = ablation.summarize([{"record_type": "warmup"}, {"record_type": "load"}, request()], unchanged=True)
    assert result["warmup_calls"] == 1
    assert result["cells"][0]["requests"] == 1


@pytest.mark.parametrize("extra", [{"generated_tokens": True}, {"error_kind": "transport"}, {"generated_tokens": 2}])
def test_only_typed_completed_token_cap_can_reuse_worker(extra):
    result = {"status": "error", "error_kind": "token_limit", "generated_tokens": 256, **extra}
    assert not ablation.completed_token_limit(result, 256)


def test_completed_token_cap_is_failed_answer_but_resident_worker():
    assert ablation.completed_token_limit({"status": "error", "error_kind": "token_limit", "generated_tokens": 256}, 256)


def test_test_or_train_path_is_rejected_before_dataset_loading(tmp_path, monkeypatch):
    config = ablation.TrainingConfig(validation_jsonl=str(tmp_path / "validation.jsonl"))
    monkeypatch.setattr(ablation, "load_prepared_records", lambda _: pytest.fail("test must never be opened"))
    with pytest.raises(ValueError, match="only the saved validation"):
        ablation.validate_dataset(tmp_path / "test.jsonl", config)


@pytest.mark.parametrize("key", ["sample_id", "group_id", "image_sha256"])
def test_training_overlap_is_rejected(key, tmp_path, monkeypatch):
    val = {"sample_id": "v", "group_id": "v", "image_sha256": "v",
           "actions": [{"action_type": "key_press", "key": "enter"}],
           "image_width": 10, "image_height": 10}
    train = {"sample_id": "t", "group_id": "t", "image_sha256": "t", key: val[key]}
    config = ablation.TrainingConfig(validation_jsonl=str(tmp_path / "validation.jsonl"), train_jsonl="train.jsonl")
    monkeypatch.setattr(ablation, "load_prepared_records", lambda p: [train] if str(p) == "train.jsonl" else [val])
    with pytest.raises(ValueError, match=key):
        ablation.validate_dataset(tmp_path / "validation.jsonl", config)


def test_ledger_is_parseable_json_and_keeps_actual_failure(tmp_path):
    rows = []
    path = tmp_path / "raw.jsonl"
    ablation.append(path, request(status="timeout", generation=""), rows)
    assert json.loads(path.read_text(encoding="utf8"))["status"] == "timeout"
    assert len(rows) == 1


@pytest.fixture
def cpu_run(tmp_path, monkeypatch):
    """128 synthetic images/targets; loader/worker/artifacts are CPU fixtures."""
    from PIL import Image

    dataset = tmp_path / "validation.jsonl"
    dataset.write_text("frozen CPU fixture dataset\n", encoding="utf8")
    config_path = tmp_path / "config.json"
    config = ablation.TrainingConfig(
        base_model_path=str(tmp_path / "base"), train_jsonl=str(tmp_path / "train.jsonl"),
        validation_jsonl=str(dataset), processor_path=str(tmp_path / "processor"),
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf8")
    samples = []
    for index in range(128):
        image = tmp_path / f"image-{index:03d}.png"
        Image.new("RGB", (20, 20), (index, 30, 45)).save(image)
        samples.append({
            "sample_id": f"sample-{index:03d}", "group_id": f"group-{index:03d}",
            "image_path": str(image), "image_file_sha256": ablation.file_sha256(image),
            "image_sha256": f"synthetic-pixel-id-{index}", "source_sample_sha256": f"synthetic-row-{index}",
            "instruction": "Click the visible control", "image_width": 20, "image_height": 20,
            "actions": [{"action_type": "click", "x": 4, "y": 5}], "action_annotations": [{}],
        })
    monkeypatch.setattr(ablation, "validate_dataset", lambda *_: samples)
    monkeypatch.setattr(ablation, "sources", lambda: {"CPU fixture": "frozen"})
    monkeypatch.setattr(ablation, "artifact_manifest", lambda path: {"path": str(path), "sha256": "fixture"})
    args = Namespace(dataset=str(dataset), training_config=str(config_path), adapter=str(tmp_path / "adapter"),
                     output=str(tmp_path / "output"), timeout=1, max_new_tokens=256, pixel_budget=[384, 1280])
    state = {"instances": [], "measured_calls": 0}

    def install(*, on_request=None, on_warmup=None, on_create=None, on_close=None):
        class FixtureWorker:
            def __init__(self, *_args, **_kwargs):
                self.ready = {"scope": "CPU fixture, not actual model inference"}
                self.calls = self.closed = 0
                state["instances"].append(self)
                if on_create:
                    on_create()

            def request(self, image, prompt, **_kwargs):
                self.calls += 1
                assert '"actions"' not in prompt and "target_bbox" not in prompt
                if self.calls == 1:
                    if on_warmup:
                        return on_warmup(self)
                else:
                    state["measured_calls"] += 1
                    if on_request:
                        result = on_request(self, state["measured_calls"])
                        if result is not None:
                            return result
                return {"status": "ok", "generation": '[{"action_type":"click","x":4,"y":5}]',
                        "duration_ms": 5, "generated_tokens": 20, "error_kind": None}

            def close(self):
                self.closed += 1
                if on_close:
                    on_close()
        monkeypatch.setattr(ablation, "HFWorker", FixtureWorker)

    def results():
        root = Path(args.output)
        raw = [json.loads(line) for line in (root / "raw.jsonl").read_text(encoding="utf8").splitlines()]
        return raw, json.loads((root / "summary.json").read_text(encoding="utf8")), \
            json.loads((root / "metadata.json").read_text(encoding="utf8"))

    return args, samples, state, install, results


@pytest.mark.parametrize("stage", ["startup", "warmup", "request", "cleanup"])
def test_interruption_retains_all_128_per_cell_and_no_invented_request_times(cpu_run, stage):
    args, _, state, install, results = cpu_run

    def interrupt(*_):
        raise KeyboardInterrupt

    hooks = {"startup": {"on_create": interrupt}, "warmup": {"on_warmup": interrupt},
             "request": {"on_request": interrupt}, "cleanup": {"on_close": interrupt}}
    install(**hooks[stage])
    assert ablation.run(args) == 1
    raw, summary, metadata = results()
    requests = [row for row in raw if row["record_type"] == "request"]
    assert len(requests) == len({(row["variant"], row["sample_id"]) for row in requests}) == 256
    assert all(cell["requests"] == cell["metrics"]["whole_action_exact"]["total"] == 128
               for cell in summary["cells"])
    assert summary["interrupted"] and not summary["complete"] and metadata["status"] == "incomplete"
    not_started = [row for row in requests if row["status"] == "not_started"]
    assert not_started and all(row["duration_ms"] is None and not row["inference_attempted"] for row in not_started)
    if stage in {"startup", "warmup", "request"}:
        cancelled, = [row for row in requests if row["status"] == "cancelled"]
        assert cancelled["duration_ms"] is None
        assert cancelled["inference_attempted"] == (stage == "request")
        assert sum(cell["percall_dispatch_to_reply_ms"]["n"] for cell in summary["cells"]) == 0
    if stage != "startup":
        assert all(worker.closed == 1 for worker in state["instances"])


@pytest.mark.parametrize("failure", ["sources", "artifacts", "dataset"])
def test_post_provenance_read_error_keeps_256_requests_summary_metadata_and_revokes_credit(cpu_run, monkeypatch, failure):
    args, _, state, install, results = cpu_run
    install()
    if failure == "sources":
        calls = 0
        def failed_sources():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("source tree unavailable after generation")
            return {"CPU fixture": "frozen"}
        monkeypatch.setattr(ablation, "sources", failed_sources)
    elif failure == "artifacts":
        calls = 0
        def failed_artifact(path):
            nonlocal calls
            calls += 1
            if calls > 3:
                raise OSError("model identity unavailable after generation")
            return {"path": str(path), "sha256": "fixture"}
        monkeypatch.setattr(ablation, "artifact_manifest", failed_artifact)
    else:
        sha = ablation.file_sha256
        def failed_dataset(path):
            if str(path) == args.dataset and state["measured_calls"] == 256:
                raise OSError("dataset identity unavailable after generation")
            return sha(path)
        monkeypatch.setattr(ablation, "file_sha256", failed_dataset)
    assert ablation.run(args) == 1
    raw, summary, metadata = results()
    requests = [row for row in raw if row["record_type"] == "request"]
    assert len(requests) == 256 and all(row["status"] == "ok" for row in requests)
    assert summary["provenance_unchanged"] is False and summary["provenance_errors"]
    assert metadata["status"] == "incomplete" and metadata["provenance_errors"]
    assert all(cell["metrics"]["whole_action_exact"] == {"correct": 0, "total": 128, "rate": 0}
               for cell in summary["cells"])
    verdict, = [row for row in raw if row["record_type"] == "provenance_verdict"]
    assert verdict["unchanged"] is False and verdict["errors"]


def test_final_check_detects_prior_image_changed_after_last_request_not_only_active_frame(cpu_run):
    args, samples, _, install, results = cpu_run
    first = Path(samples[0]["image_path"])
    before = first.read_bytes()
    def mutate_earlier_frame(_worker, number):
        if number == 256:
            first.write_bytes(before + b"changed after the last request")
    install(on_request=mutate_earlier_frame)
    assert ablation.run(args) == 1
    raw, summary, metadata = results()
    requests = [row for row in raw if row["record_type"] == "request"]
    assert len(requests) == 256 and all(row["image_unchanged"] for row in requests)
    assert not summary["provenance_unchanged"]
    assert metadata["provenance_after"]["image_file_sha256"][samples[0]["sample_id"]] != samples[0]["image_file_sha256"]
    assert all(cell["metrics"]["strict_json"]["correct"] == 0 for cell in summary["cells"])


def test_startup_errors_keep_full_target_denominator_and_missing_latency(cpu_run):
    args, _, state, install, results = cpu_run
    def unavailable():
        raise RuntimeError("CPU synthetic startup failure")
    install(on_create=unavailable)
    assert ablation.run(args) == 1
    raw, summary, metadata = results()
    assert len(state["instances"]) == 6  # three infrastructure failures per cell
    statuses = Counter(row["status"] for row in raw if row["record_type"] == "request")
    assert statuses == {"error": 6, "not_started": 250}
    assert summary["unmeasured_samples"] == 256 and metadata["status"] == "incomplete"
    assert all(cell["metrics"]["whole_action_exact"]["total"] == 128
               and cell["percall_dispatch_to_reply_ms"]["n"] == 0 for cell in summary["cells"])


def test_cleanup_exception_does_not_duplicate_request_or_claim_completed_run(cpu_run):
    args, _, _, install, results = cpu_run
    def broken_close():
        raise RuntimeError("CPU synthetic close failure")
    install(on_close=broken_close)
    assert ablation.run(args) == 1
    raw, summary, metadata = results()
    requests = [row for row in raw if row["record_type"] == "request"]
    assert len(requests) == len({(row["variant"], row["sample_id"]) for row in requests}) == 256
    assert len(summary["worker_cleanup_errors"]) == 2
    assert summary["provenance_unchanged"] and not summary["complete"] and metadata["status"] == "incomplete"


def test_typed_token_limit_retains_raw_failure_but_reuses_only_completed_workers(cpu_run):
    args, _, state, install, results = cpu_run
    install(on_request=lambda *_: {"status": "error", "error_kind": "token_limit", "generated_tokens": 256,
                                   "generation": "[incomplete actual generation", "duration_ms": 5})
    assert ablation.run(args) == 0
    raw, summary, _ = results()
    requests = [row for row in raw if row["record_type"] == "request"]
    assert len(state["instances"]) == 2 and all(worker.closed == 1 for worker in state["instances"])
    assert all(row["error_kind"] == "token_limit" and row["generation"] for row in requests)
    assert all(cell["metrics"]["strict_json"] == {"correct": 0, "total": 128, "rate": 0}
               for cell in summary["cells"])
