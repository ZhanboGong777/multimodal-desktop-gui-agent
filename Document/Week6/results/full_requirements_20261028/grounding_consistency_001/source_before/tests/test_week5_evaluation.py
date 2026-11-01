"""CPU-only metric/ledger tests. Stub replies here are never measured model runs."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gui_agent.finetuning.evaluation import (
    CELLS,
    HFConfig,
    HFWorker,
    distribution,
    load_prepared,
    nearest_rank,
    parse_generation,
    safe_prompt_input,
    score_generation,
    select_samples,
    summarize_records,
    validate_actions,
)


def target():
    return [
        {"action_type": "click", "x": 12, "y": 15},
        {"action_type": "type_text", "text": "  中文 punctuation!  "},
    ]


def annotations():
    return [{"target_bbox": {"left": 10, "top": 10, "right": 20, "bottom": 20}}, {}]


def score(reply, **kwargs):
    return score_generation(
        target(), reply, width=100, height=80, action_annotations=annotations(), **kwargs
    )


def record(reply=None, *, cell="base_original", status="ok", duration=20, sample_id="s1"):
    return {
        "record_type": "request",
        "run_id": "r1",
        "study_id": "study",
        "cell": cell,
        "sample_id": sample_id,
        "image_sha256": "pixels",
        "source_sample_sha256": "source",
        "instruction": "fixture instruction",
        "target": {
            "actions": target(),
            "image_width": 100,
            "image_height": 80,
            "action_annotations": annotations(),
        },
        "generation": json.dumps(target()) if reply is None else reply,
        "status": status,
        "duration_ms": duration,
        "worker_inference_ms": duration,
        "inference_attempted": True,
    }


def verdict(unchanged=True):
    return {"record_type": "provenance_verdict", "run_id": "r1", "unchanged": unchanged}


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n[{"action_type":"click","x":1,"y":2}]\n```',
        '[{"action_type":"click","x":1,"x":2,"y":2}]',
        '[{"action_type":"wait","duration":NaN}]',
        '[{"action_type":"wait","duration":Infinity}]',
        "[] trailing",
    ],
)
def test_no_json_repair_or_nonstandard_constants(reply):
    actions, syntax, error = parse_generation(reply)
    assert actions is None and not syntax and error


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"actions": target()},
        [{"action_type": "CLICK", "x": 1, "y": 2}],
        [{"action_type": "click", "x": True, "y": 2}],
        [{"action_type": "click", "x": 1.5, "y": 2}],
        [{"action_type": "click", "x": "1", "y": 2}],
        [{"action_type": "click", "x": 1, "y": 2, "target_bbox": [1, 2, 3, 4]}],
        [{"action_type": "click", "x": 1, "y": 2, "duration": 1}],
        [{"action_type": "scroll", "x": 1, "y": 2}],
        [{"action_type": "drag", "start": {"x": 1, "y": 2}, "end": {"x": True, "y": 2}}],
        [{"action_type": "hotkey", "keys": "ctrl+c"}],
        [{"action_type": "wait", "duration": -1}],
    ],
)
def test_action_schema_is_strict(payload):
    with pytest.raises(ValueError):
        validate_actions(payload)


def test_exact_actions_text_and_native_bbox():
    metrics = score(json.dumps(target()))
    assert metrics["strict_action_exact"] and metrics["strict_json_valid"]
    assert metrics["action_type_correct"] == metrics["action_type_slots"] == 2
    assert metrics["text_exact_correct"] == metrics["text_exact_total"] == 1
    assert metrics["click_bbox_hits"] == metrics["click_bbox_eligible"] == 1
    assert metrics["click_normalized_distances_observed"] == [0]


def test_whitespace_not_normalized_and_bbox_success_not_strict_success():
    changed = target()
    changed[0]["x"] = 19
    changed[1]["text"] = changed[1]["text"].strip()
    metrics = score(json.dumps(changed))
    assert not metrics["strict_action_exact"]
    assert metrics["click_bbox_hits"] == 1 and metrics["text_exact_correct"] == 0
    assert metrics["click_normalized_distances_observed"] == pytest.approx([0.07])


def test_half_open_bbox_right_edge_is_miss():
    changed = target()
    changed[0]["x"] = 20
    assert score(json.dumps(changed))["click_bbox_hits"] == 0


def test_extra_actions_penalize_type_denominator():
    metrics = score(json.dumps(target() + [{"action_type": "wait", "duration": 1}]))
    assert metrics["action_type_correct"] == 2 and metrics["action_type_slots"] == 3
    assert not metrics["action_type_sequence_exact"]
    assert metrics["extra_predicted_actions"] == 1


@pytest.mark.parametrize("status", ["error", "timeout", "cancelled", "provenance_changed"])
def test_error_partial_json_cannot_receive_success_credit(status):
    metrics = score(json.dumps(target()), status=status)
    assert not metrics["strict_json_valid"] and not metrics["strict_action_exact"]
    assert metrics["text_exact_correct"] == metrics["click_bbox_hits"] == 0
    assert metrics["click_distance_missing"] == 1
    assert metrics["click_normalized_distances_observed"] == []
    assert metrics["click_normalized_distances_all_targets"] == [math.sqrt(2)]


def test_bbox_absence_remains_unscored_instead_of_inventing_point_box():
    metrics = score_generation(target(), json.dumps(target()), width=100, height=80)
    assert metrics["click_bbox_eligible"] == 0 and metrics["click_bbox_absent"] == 1
    assert metrics["click_normalized_distances_observed"] == [0]


def test_out_of_bounds_prediction_is_not_clipped():
    changed = target()
    changed[0]["x"] = 500
    metrics = score(json.dumps(changed))
    assert metrics["click_bbox_hits"] == 0 and metrics["click_prediction_out_of_bounds"] == 1
    assert metrics["click_normalized_distances_observed"][0] > math.sqrt(2)


def test_real_bbox_geometry_must_be_valid():
    with pytest.raises(ValueError, match="outside"):
        score_generation(
            target(),
            "",
            width=100,
            height=80,
            action_annotations=[
                {"target_bbox": {"left": 1, "top": 1, "right": 200, "bottom": 20}},
                {},
            ],
        )


def test_nearest_rank_small_n_and_empty_not_zero():
    assert nearest_rank([60, 180], 0.5) == 60
    assert nearest_rank([60, 180], 0.9) == nearest_rank([60, 180], 0.95) == 180
    assert distribution([]) == {"n": 0, "p50": None, "p90": None, "p95": None, "mean": None}
    with pytest.raises(ValueError):
        nearest_rank([float("nan")], 0.5)


def test_summary_recomputes_cached_metrics_and_keeps_error_denominator():
    good = record(duration=10)
    bad = record(status="timeout", duration=100, sample_id="s2")
    bad["metrics"] = {"strict_action_exact": True}
    summary = summarize_records([good, bad, verdict()])
    cell = summary["cells"][0]
    assert cell["strict_action_exact"] == {"correct": 1, "total": 2, "rate": 0.5}
    assert cell["percall_e2e_ms_all_recorded_requests"]["p95"] == 100
    assert cell["percall_e2e_ms_strict_action_success"]["n"] == 1
    assert cell["statuses"] == {"ok": 1, "timeout": 1}
    assert not summary["desktop_task_e2e"]


def test_startup_failure_missing_latency_not_zero_latency():
    failed = record(status="load_or_warmup_error", duration=None)
    failed["inference_attempted"] = False
    summary = summarize_records([failed, verdict()])["cells"][0]
    assert summary["requests"] == 1 and summary["duration_missing"] == 1
    assert summary["inference_attempted"] == 0
    assert summary["percall_e2e_ms_all_recorded_requests"]["p50"] is None
    assert summary["percall_e2e_ms_strict_action_success"]["n"] == 0


def test_pending_or_changed_provenance_stays_in_denominator():
    pending = summarize_records([record()])["cells"][0]
    changed = summarize_records([record(), verdict(False)])["cells"][0]
    assert pending["statuses"] == {"provenance_pending": 1}
    assert changed["statuses"] == {"provenance_changed": 1}
    assert pending["strict_action_exact"]["total"] == 1
    assert changed["strict_action_exact"]["correct"] == 0


def test_no_duplicate_jobs_or_conflicting_final_verdict():
    with pytest.raises(ValueError, match="duplicate evaluation"):
        summarize_records([record(), record(), verdict()])
    with pytest.raises(ValueError, match="duplicate final"):
        summarize_records([record(), verdict(), verdict(False)])


def test_full_2x2_same_samples_and_no_cross_cell_target_changes():
    rows = [record(cell=cell) for cell in CELLS]
    comparison = summarize_records([*rows, verdict()])["comparisons"][0]
    assert comparison["complete_2x2_same_samples"] and comparison["shared_sample_count"] == 1
    rows[-1]["target"]["actions"][0]["x"] = 50
    with pytest.raises(ValueError, match="differs across"):
        summarize_records([*rows, verdict()])


def test_paired_adapter_and_prompt_effects_use_identical_samples_and_errors():
    rows = [
        record(cell=cell, status="timeout" if cell == "base_original" else "ok") for cell in CELLS
    ]
    summary = summarize_records([*rows, verdict()])
    pairs = {pair["comparison"]: pair for pair in summary["paired_comparisons"]}
    assert pairs["adapter_effect_original"]["outcomes"] == {"improved": 1}
    assert pairs["adapter_effect_original"]["strict_success_rate_delta_after_minus_before"] == 1
    assert pairs["adapter_effect_improved"]["outcomes"] == {"both_success": 1}
    assert pairs["prompt_effect_base"]["comparison_eligible"]
    rows[0]["inference_attempted"] = False
    summary = summarize_records([*rows, verdict()])
    assert not summary["comparisons"][0]["complete_2x2_same_samples"]
    assert not summary["paired_comparisons"][0]["comparison_eligible"]


def test_prompt_input_contains_no_label_source_or_filename():
    prepared = {
        "instruction": "Locate Save",
        "image_width": 100,
        "image_height": 80,
        "sample_id": "label-name",
        "actions": target(),
        "image_path": "answer.jpg",
        "group_id": "session",
        "source": "label",
        "action_annotations": annotations(),
    }
    assert safe_prompt_input(prepared) == {
        "instruction": "Locate Save",
        "image_width": 100,
        "image_height": 80,
    }


def test_selection_is_seed_fixed_order_independent_and_capped():
    samples = [{"sample_id": f"{index:04d}"} for index in range(100)]
    selected = select_samples(samples)
    assert len(selected) == 48 and selected == select_samples(list(reversed(samples)))
    assert selected == sorted(selected, key=lambda sample: sample["sample_id"])
    assert len(select_samples(samples, max_samples=64)) == 64
    with pytest.raises(ValueError):
        select_samples(samples, max_samples=1)


def prepared_file(tmp_path):
    image = tmp_path / "screen.png"
    Image.new("RGB", (100, 80), "white").save(image)
    sample = {
        "sample_id": "fixture-sample",
        "group_id": "fixture-group",
        "image_path": image.name,
        "instruction": "Locate the control",
        "actions": target(),
        "image_width": 100,
        "image_height": 80,
        "source": {"dataset": "unit-test-fixture"},
        "source_split": "test",
        "action_annotations": annotations(),
    }
    path = tmp_path / "heldout.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    return path, sample


def test_prepared_native_image_and_input_provenance(tmp_path):
    path, _ = prepared_file(tmp_path)
    samples = load_prepared(path)
    assert Path(samples[0]["image_path"]).is_absolute()
    assert len(samples[0]["image_sha256"]) == len(samples[0]["source_sample_sha256"]) == 64


@pytest.mark.parametrize(
    "change,pattern",
    [
        ({"source_split": "train"}, "training-split"),
        ({"image_width": 101}, "dimensions"),
        ({"actions": [{"action_type": "click", "x": 100, "y": 2}]}, "outside"),
    ],
)
def test_prepared_refuses_invalid_or_training_data(tmp_path, change, pattern):
    path, sample = prepared_file(tmp_path)
    sample.update(change)
    if "actions" in change:
        sample.pop("action_annotations")
    path.write_text(json.dumps(sample), encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        load_prepared(path)


def test_duplicate_prepared_samples_are_not_extra_measurements(tmp_path):
    path, sample = prepared_file(tmp_path)
    path.write_text((json.dumps(sample) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sample"):
        load_prepared(path)


def test_prepared_decoded_rgb_hash_is_checked_not_overwritten(tmp_path):
    path, sample = prepared_file(tmp_path)
    sample["image_sha256"] = "incorrect"
    path.write_text(json.dumps(sample), encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        load_prepared(path)


def test_hard_timeout_covers_blocked_recv_after_partial_transport_header():
    release = threading.Event()

    class Connection:
        def send(self, value):
            pass

        def recv(self):
            release.wait(1)
            raise EOFError("closed")

        def close(self):
            release.set()

    class Process:
        pid = 123
        alive = True
        terminated = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, **kwargs):
            pass

    worker = HFWorker.__new__(HFWorker)
    worker.connection, worker.process, worker.request_count = Connection(), Process(), 0
    response = worker.request("fixture.png", "fixture prompt", timeout_seconds=0.01)
    assert response["status"] == "timeout" and response["worker_inference_ms"] is None
    assert worker.process.terminated and 0 < response["duration_ms"] < 1000


def startup_transport(monkeypatch, *, receive_error=None, start_error=None, partial_start=False):
    """No subprocess or model: simulate a partial ready frame and resource ownership."""
    from gui_agent.finetuning import evaluation

    release = threading.Event()

    class Connection:
        closed = False

        def poll(self, timeout):
            return True  # The old implementation would enter an unbounded recv.

        def recv(self):
            if receive_error is not None:
                raise receive_error
            release.wait(1)
            raise EOFError("closed")

        def close(self):
            self.closed = True
            release.set()

    class ChildConnection:
        closed = False

        def close(self):
            self.closed = True

    class Process:
        pid = None
        terminated = False
        alive = False

        def start(self):
            if partial_start:
                self.pid, self.alive = 123, True
            if start_error is not None:
                raise start_error
            self.pid, self.alive = 123, True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated, self.alive = True, False

        def join(self, **kwargs):
            assert self.pid is not None

    parent, child, process = Connection(), ChildConnection(), Process()
    context = SimpleNamespace(Pipe=lambda: (parent, child), Process=lambda **kwargs: process)
    monkeypatch.setattr(evaluation.multiprocessing, "get_context", lambda _: context)
    return evaluation, parent, child, process


def test_startup_partial_ready_recv_has_hard_deadline_and_cleans_child(monkeypatch):
    _module, parent, child, process = startup_transport(monkeypatch)
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="loading timed out"):
        HFWorker(HFConfig({}), load_timeout_seconds=0.01)
    assert time.perf_counter() - started < 0.5
    assert parent.closed and child.closed and process.terminated


def test_startup_keyboard_interrupt_while_waiting_closes_unassigned_worker(monkeypatch):
    module, parent, child, process = startup_transport(monkeypatch)

    class CancelledQueue:
        def __init__(self, **kwargs):
            pass

        def put(self, value):
            pass

        def get(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(module.queue, "Queue", CancelledQueue)
    with pytest.raises(KeyboardInterrupt):
        HFWorker(HFConfig({}), load_timeout_seconds=1)
    assert parent.closed and child.closed and process.terminated


@pytest.mark.parametrize("error", [OSError("ready transport failed"), EOFError("child exited")])
def test_startup_receive_exception_closes_unassigned_worker(monkeypatch, error):
    _module, parent, child, process = startup_transport(monkeypatch, receive_error=error)
    with pytest.raises((OSError, RuntimeError)):
        HFWorker(HFConfig({}), load_timeout_seconds=1)
    assert parent.closed and child.closed and process.terminated


def test_startup_start_exception_closes_both_pipes_without_join_before_start(monkeypatch):
    _module, parent, child, process = startup_transport(
        monkeypatch, start_error=RuntimeError("spawn failed")
    )
    with pytest.raises(RuntimeError, match="spawn failed"):
        HFWorker(HFConfig({}), load_timeout_seconds=1)
    assert parent.closed and child.closed and not process.terminated


def test_startup_cancelled_partial_spawn_terminates_created_child(monkeypatch):
    _module, parent, child, process = startup_transport(
        monkeypatch,
        start_error=KeyboardInterrupt(),
        partial_start=True,
    )
    with pytest.raises(KeyboardInterrupt):
        HFWorker(HFConfig({}), load_timeout_seconds=1)
    assert parent.closed and child.closed and process.terminated


@pytest.fixture
def cli():
    path = Path(__file__).parents[1] / "scripts/week5_evaluate.py"
    spec = importlib.util.spec_from_file_location("week5_evaluate_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_inputs(tmp_path, monkeypatch, cli):
    path, sample = prepared_file(tmp_path)
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "fixture.txt").write_text("not a model", encoding="utf-8")
    (adapter / "fixture.txt").write_text("not an adapter", encoding="utf-8")
    configuration = {
        "base_model_path": str(base),
        "base_model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "base_model_revision": "895c3a49bc3fa70a340399125c650a463535e71c",
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration), encoding="utf-8")

    class Config:
        @staticmethod
        def from_dict(value):
            return SimpleNamespace(to_dict=lambda: value)

    monkeypatch.setitem(
        sys.modules, "gui_agent.finetuning.training", SimpleNamespace(TrainingConfig=Config)
    )
    monkeypatch.setattr(cli, "code_manifest", lambda _: {"code.py": "frozen"})
    monkeypatch.setattr(cli, "runtime_versions", lambda: {"fixture": "unit-test-only"})
    args = [
        "run",
        "--dataset",
        str(path),
        "--training-config",
        str(config),
        "--adapter",
        str(adapter),
        "--output",
        str(tmp_path / "evaluation"),
    ]
    return args, sample


def test_cli_ledger_all_cells_warmups_separate_and_model_never_sees_labels(
    tmp_path, monkeypatch, cli
):
    args, sample = cli_inputs(tmp_path, monkeypatch, cli)
    calls = []

    class StubWorker:
        """Unit-test transport only; cannot be selected from the real CLI."""

        def __init__(self, *args, **kwargs):
            self.ready = {"versions": {"fixture": "unit-test-only"}, "load_ms": 50}
            self.request_count = 0

        def request(self, image_path, prompt, **kwargs):
            self.request_count += 1
            calls.append(prompt)
            assert sample["sample_id"] not in prompt and sample["group_id"] not in prompt
            assert sample["actions"][1]["text"] not in prompt
            return {
                "status": "ok",
                "generation": json.dumps(sample["actions"]),
                "duration_ms": 10,
                "worker_inference_ms": 8,
                "rendered_prompt": prompt,
            }

        def close(self):
            pass

    monkeypatch.setattr(cli, "HFWorker", StubWorker)
    assert cli.main(args) == 0
    output = tmp_path / "evaluation"
    raw = [
        json.loads(line) for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len([row for row in raw if row["record_type"] == "request"]) == 4
    assert len([row for row in raw if row["record_type"] == "warmup"]) == 4
    assert len(calls) == 8
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["comparisons"][0]["complete_2x2_same_samples"]
    assert summary["warmup_calls"] == 4
    assert cli.main(args) == 2  # Existing measurement cannot be overwritten.


def test_cli_post_measurement_provenance_error_keeps_raw_generation(tmp_path, monkeypatch, cli):
    args, sample = cli_inputs(tmp_path, monkeypatch, cli)
    args += ["--cell", "base_original", "--warmup-calls", "0"]

    class StubWorker:
        def __init__(self, *args, **kwargs):
            self.ready, self.request_count = (
                {"load_ms": 1, "versions": {"fixture": "unit-test-only"}},
                0,
            )

        def request(self, *args, **kwargs):
            self.request_count += 1
            return {
                "status": "ok",
                "generation": json.dumps(sample["actions"]),
                "duration_ms": 10,
                "worker_inference_ms": 8,
            }

        def close(self):
            pass

    calls = 0

    def source(_):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("post-measurement source read failed")
        return {"code.py": "frozen"}

    monkeypatch.setattr(cli, "HFWorker", StubWorker)
    monkeypatch.setattr(cli, "code_manifest", source)
    assert cli.main(args) == 1
    raw = [
        json.loads(line)
        for line in (tmp_path / "evaluation/raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    request = next(row for row in raw if row["record_type"] == "request")
    assert request["status"] == "ok" and json.loads(request["generation"]) == sample["actions"]
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text(encoding="utf-8"))
    assert summary["cells"][0]["statuses"] == {"provenance_changed": 1}
    assert summary["cells"][0]["strict_action_exact"]["correct"] == 0


def test_cli_load_error_counts_all_samples_without_repeated_load(tmp_path, monkeypatch, cli):
    args, sample = cli_inputs(tmp_path, monkeypatch, cli)
    args += ["--cell", "base_original"]
    second = {**sample, "sample_id": "fixture-second"}
    path = tmp_path / "heldout.jsonl"
    path.write_text(json.dumps(sample) + "\n" + json.dumps(second), encoding="utf-8")
    loads = 0

    def failing_worker(*args, **kwargs):
        nonlocal loads
        loads += 1
        raise RuntimeError("unit-test load error")

    monkeypatch.setattr(cli, "HFWorker", failing_worker)
    assert cli.main(args) == 1 and loads == 1
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text(encoding="utf-8"))
    cell = summary["cells"][0]
    assert cell["requests"] == 2 and cell["inference_attempted"] == 0
    assert cell["strict_action_exact"]["total"] == 2
    assert cell["percall_e2e_ms_all_recorded_requests"]["n"] == 0


def test_cli_cancellation_preserves_planned_denominator_without_fake_measurements(
    tmp_path, monkeypatch, cli
):
    args, _sample = cli_inputs(tmp_path, monkeypatch, cli)
    args += ["--warmup-calls", "0"]

    class CancelledWorker:
        def __init__(self, *args, **kwargs):
            self.ready, self.request_count = (
                {"load_ms": 1, "versions": {"fixture": "unit-test-only"}},
                0,
            )

        def request(self, *args, **kwargs):
            self.request_count += 1
            raise KeyboardInterrupt

        def close(self):
            pass

    monkeypatch.setattr(cli, "HFWorker", CancelledWorker)
    assert cli.main(args) == 130
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text(encoding="utf-8"))
    assert sum(cell["requests"] for cell in summary["cells"]) == 4
    assert sum(cell["inference_attempted"] for cell in summary["cells"]) == 1
    assert summary["cells"][0]["strict_action_exact"]["correct"] == 0
    assert all(cell["percall_e2e_ms_strict_action_success"]["n"] == 0 for cell in summary["cells"])
    assert {status for cell in summary["cells"] for status in cell["statuses"]} == {
        "cancelled",
        "not_started",
    }


def test_cli_checks_training_group_and_image_leakage(tmp_path, monkeypatch, cli):
    args, sample = cli_inputs(tmp_path, monkeypatch, cli)
    training = tmp_path / "train.jsonl"
    training.write_text(
        json.dumps(
            {"sample_id": "train-id", "group_id": sample["group_id"], "image_sha256": "different"}
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["train_jsonl"] = str(training)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert cli.main(args) == 2
    assert not (tmp_path / "evaluation").exists()


def test_cli_completed_token_limits_stay_resident_but_all_fail_scoring(tmp_path, monkeypatch, cli):
    args, sample = cli_inputs(tmp_path, monkeypatch, cli)
    args += ["--cell", "base_original", "--max-new-tokens", "2"]
    second = {**sample, "sample_id": "fixture-second"}
    (tmp_path / "heldout.jsonl").write_text(
        json.dumps(sample) + "\n" + json.dumps(second), encoding="utf-8"
    )
    workers = []

    class TokenLimitWorker:
        def __init__(self, *args, **kwargs):
            self.ready = {"versions": {"fixture": "unit-test-only"}, "load_ms": 50}
            self.request_count, self.closes = 0, 0
            workers.append(self)

        def request(self, *args, **kwargs):
            self.request_count += 1
            # Even a syntactically exact target receives zero success credit
            # after a truncated generation. The first call is separate warmup.
            return {
                "status": "error",
                "error": "max_new_tokens reached without EOS",
                "error_kind": "token_limit",
                "generated_tokens": 2,
                "generation": json.dumps(sample["actions"]),
                "duration_ms": 10 * self.request_count,
                "worker_inference_ms": 8,
            }

        def close(self):
            self.closes += 1

    monkeypatch.setattr(cli, "HFWorker", TokenLimitWorker)
    assert cli.main(args) == 0  # Measurement completed; no task-success claim.
    assert len(workers) == 1 and workers[0].request_count == 3 and workers[0].closes == 1
    output = tmp_path / "evaluation"
    raw = [
        json.loads(line) for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    requests = [row for row in raw if row["record_type"] == "request"]
    assert len(requests) == 2 and all(row["status"] == "error" for row in requests)
    assert all(row["error_kind"] == "token_limit" for row in requests)
    assert all(row["generation"] == json.dumps(sample["actions"]) for row in requests)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    cell = summary["cells"][0]
    assert cell["statuses"] == {"error": 2}
    assert (
        cell["strict_json_valid"]
        == cell["strict_action_exact"]
        == {"correct": 0, "total": 2, "rate": 0.0}
    )
    assert cell["percall_e2e_ms_all_recorded_requests"]["p95"] == 30
    assert cell["percall_e2e_ms_strict_action_success"]["n"] == 0
    assert summary["warmup_calls"] == 1
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    frozen = next(row["metadata"] for row in raw if row["record_type"] == "study")
    assert (
        metadata["worker_recovery_policy"]
        == frozen["worker_recovery_policy"]
        == cli.WORKER_RECOVERY_POLICY
    )
    assert cli.value_sha256(frozen) == metadata["study_id"]
    without_policy = {
        key: value for key, value in frozen.items() if key != "worker_recovery_policy"
    }
    assert cli.value_sha256(without_policy) != metadata["study_id"]


@pytest.mark.parametrize(
    ("status", "error_kind", "generated_tokens"),
    [
        ("error", None, 2),
        ("error", "unknown", 2),
        ("timeout", "token_limit", 2),
        ("cancelled", "token_limit", 2),
        ("error", "token_limit", None),
        ("error", "token_limit", True),
        ("error", "token_limit", 1),
    ],
)
def test_cli_other_or_unproved_errors_close_reload_and_warmup(
    tmp_path, monkeypatch, cli, status, error_kind, generated_tokens
):
    args, sample = cli_inputs(tmp_path, monkeypatch, cli)
    args += ["--cell", "base_original", "--max-new-tokens", "2"]
    second = {**sample, "sample_id": "fixture-second"}
    (tmp_path / "heldout.jsonl").write_text(
        json.dumps(sample) + "\n" + json.dumps(second), encoding="utf-8"
    )
    workers = []

    class FailedWorker:
        def __init__(self, *args, **kwargs):
            self.ready = {"versions": {"fixture": "unit-test-only"}, "load_ms": 50}
            self.request_count, self.closes = 0, 0
            workers.append(self)

        def request(self, *args, **kwargs):
            self.request_count += 1
            if self.request_count == 1:
                return {"status": "ok", "generation": "warmup", "duration_ms": 1}
            return {
                "status": status,
                "error": "max_new_tokens reached without EOS",  # Text alone is not authority.
                "error_kind": error_kind,
                "generated_tokens": generated_tokens,
                "generation": "original failure preserved",
                "duration_ms": 20,
            }

        def close(self):
            self.closes += 1

    monkeypatch.setattr(cli, "HFWorker", FailedWorker)
    assert cli.main(args) == 1
    assert len(workers) == 2 and all(
        worker.request_count == 2 and worker.closes == 1 for worker in workers
    )
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text(encoding="utf-8"))
    assert summary["warmup_calls"] == 2
    assert summary["cells"][0]["statuses"] == {status: 2}
    assert summary["cells"][0]["strict_action_exact"]["correct"] == 0


@pytest.mark.parametrize("outcome", ["token_limit", "eos", "inference_exception"])
def test_cpu_worker_labels_only_completed_no_eos_generation_nonfatal(
    tmp_path, monkeypatch, outcome
):
    from gui_agent.finetuning import evaluation

    _path, sample = prepared_file(tmp_path)
    seeds, responses = [], []
    fake_torch = SimpleNamespace(
        manual_seed=seeds.append,
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Inputs(dict):
        def to(self, device):
            return self

    class Suffix:
        shape = (1, 2)

        def __getitem__(self, index):
            return 99 if outcome == "eos" else 42

    class Generated:
        def __getitem__(self, index):
            return Suffix()

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[99])

        def eval(self):
            pass

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 2
            if outcome == "inference_exception":
                raise RuntimeError("synthetic inference failure")
            return Generated()

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "rendered fixture"

        def __call__(self, **kwargs):
            return Inputs(input_ids=SimpleNamespace(shape=(1, 1)))

        def batch_decode(self, *args, **kwargs):
            return ["original decoded fixture"]

    class Config:
        @staticmethod
        def from_dict(value):
            return value

    monkeypatch.setitem(
        sys.modules,
        "gui_agent.finetuning.training",
        SimpleNamespace(
            TrainingConfig=Config,
            load_model_and_processor=lambda *args, **kwargs: (Model(), Processor()),
        ),
    )
    monkeypatch.setattr(evaluation, "runtime_versions", lambda: {"fixture": "CPU-only"})
    pending = iter(
        [{"image_path": str(tmp_path / sample["image_path"]), "prompt": "fixture"}] * 2 + [None]
    )

    class Connection:
        closed = False

        def recv(self):
            return next(pending)

        def send(self, response):
            responses.append(response)

        def close(self):
            self.closed = True

    connection = Connection()
    evaluation._hf_worker(connection, HFConfig({}, max_new_tokens=2), None)
    assert connection.closed and seeds == [42, 42] and len(responses) == 3
    for response in responses[1:]:
        if outcome == "token_limit":
            assert response["status"] == "error" and response["error_kind"] == "token_limit"
            assert (
                response["generated_tokens"] == 2
                and response["generation"] == "original decoded fixture"
            )
        elif outcome == "eos":
            assert response["status"] == "ok" and response["error_kind"] is None
        else:
            assert response["status"] == "error" and response.get("error_kind") is None
            assert response["error"] == "RuntimeError: synthetic inference failure"


def test_cli_unknown_warmup_error_with_token_count_still_closes_worker(tmp_path, monkeypatch, cli):
    args, _sample = cli_inputs(tmp_path, monkeypatch, cli)
    args += ["--cell", "base_original", "--max-new-tokens", "2"]
    workers = []

    class FailedWarmupWorker:
        def __init__(self, *args, **kwargs):
            self.ready = {"versions": {"fixture": "unit-test-only"}, "load_ms": 50}
            self.request_count, self.closes = 0, 0
            workers.append(self)

        def request(self, *args, **kwargs):
            self.request_count += 1
            return {
                "status": "error",
                "error_kind": "unknown",
                "error": "synthetic failure",
                "generated_tokens": 2,
                "duration_ms": 10,
            }

        def close(self):
            self.closes += 1

    monkeypatch.setattr(cli, "HFWorker", FailedWarmupWorker)
    assert cli.main(args) == 1
    assert len(workers) == 1 and workers[0].request_count == 1 and workers[0].closes == 1
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text(encoding="utf-8"))
    assert summary["cells"][0]["statuses"] == {"load_or_warmup_error": 1}
    assert summary["cells"][0]["inference_attempted"] == 0
    assert summary["cells"][0]["duration_missing"] == 1
