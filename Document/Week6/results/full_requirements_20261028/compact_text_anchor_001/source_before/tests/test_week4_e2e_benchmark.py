"""The outer harness must retain failed requests and never infer success from prose."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "week4_e2e_benchmark.py"
SPEC = importlib.util.spec_from_file_location("week4_e2e_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def record(identity, elapsed, status="failed", *, commit="abc", warm="warm"):
    return {
        "schema": benchmark.SCHEMA,
        "run_id": identity,
        "case_id": "T04",
        "status": status,
        "outer_e2e_ms": elapsed,
        "cohort": {"commit": commit, "warm_state": warm, "execute": True},
    }


def measurement(returncode=0, **extra):
    return {
        "returncode": returncode,
        "outer_timeout": False,
        "cancelled": False,
        "launch_error": None,
        "output_error": None,
        **extra,
    }


def sent_summary(marker="WEEK4_MESSAGE_CHECK_test"):
    return {
        "case_id": "T04",
        "execute": True,
        "status": "succeeded",
        "verification": {
            "outcome": "passed",
            "evidence": {
                "vision_assessment": {
                    "marker": marker,
                    "conversation": "文件传输助手",
                    "outgoing": True,
                    "composer_empty": True,
                    "status": "sent",
                    "send_state": "sent",
                }
            },
        },
    }


def test_all_requests_include_failures_and_timeouts_success_distribution_is_separate():
    result = benchmark.aggregate_records(
        [
            record("one", 100, "succeeded"),
            record("two", 200, "failed"),
            record("three", 900, "outer_timed_out"),
            record("four", 400, "blocked"),
        ]
    )
    group = result["groups"][0]
    assert group["requests"] == 4
    assert group["success_rate"] == 0.25
    assert group["timeout_count"] == 1
    assert group["all_request_end_latency"] == {
        "n": 4,
        "p50_ms": 200,
        "p90_ms": 900,
        "p95_ms": 900,
        "max_ms": 900,
    }
    assert group["success_completion_latency"]["n"] == 1
    assert group["success_completion_latency"]["p95_ms"] == 100
    assert group["by_status"]["outer_timed_out"]["n"] == 1
    assert group["all_request_system_latency"]["n"] == 0


def test_version_and_warm_state_do_not_mix():
    result = benchmark.aggregate_records(
        [
            record("one", 100),
            record("two", 100, commit="other"),
            record("three", 100, warm="cold"),
        ]
    )
    assert len([g for g in result["groups"] if g["case_id"] == "ALL"]) == 3


def test_in_flight_source_change_keeps_attempt_but_never_credits_fixed_version_success():
    request = record("one", 200, "succeeded")
    request.update({"provenance_changed": True, "observed_task_status": "succeeded"})
    benchmark.apply_provenance_verdict(request)
    group = benchmark.aggregate_records([request])["groups"][0]
    assert group["requests"] == 1
    assert group["succeeded"] == 0
    assert group["success_completion_latency"]["n"] == 0
    assert group["all_request_end_latency"]["n"] == 1
    assert request["observed_task_status"] == "succeeded"
    assert request["status_before_provenance_check"] == "succeeded"


def test_provenance_change_does_not_hide_a_real_timeout():
    request = record("one", 200, "outer_timed_out")
    request.update({"provenance_changed": True, "outer_timeout": True})
    benchmark.apply_provenance_verdict(request)
    assert benchmark.aggregate_records([request])["groups"][0]["timeout_count"] == 1


def test_aggregate_normalizes_old_v1_provenance_flags_without_mutating_input():
    success = record("one", 100, "succeeded")
    timeout = record("two", 900, "timed_out")
    success["provenance_changed"] = timeout["provenance_changed"] = True
    group = benchmark.aggregate_records([success, timeout])["groups"][0]
    assert group["requests"] == 2
    assert group["succeeded"] == 0
    assert group["success_completion_latency"]["n"] == 0
    assert group["all_request_end_latency"]["n"] == 2
    assert group["status_counts"] == {"provenance_changed": 2}
    assert group["timeout_count"] == 1
    assert success["status"] == "succeeded"
    assert timeout["status"] == "timed_out"


def test_external_config_changes_are_detected_before_launch(tmp_path):
    original = Path(__file__).resolve().parents[1] / "configs" / "week4.yaml"
    config = tmp_path / "config.yaml"
    config.write_bytes(original.read_bytes())
    args = benchmark.parse_args(["plan", "--config", str(config)])
    effective, _ = benchmark.resolve_configuration(args)
    cohort = {
        "config_sha256": benchmark.canonical_hash(effective),
        "config_file_sha256": benchmark.hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    assert benchmark.configuration_unchanged(args, cohort)
    config.write_bytes(original.read_bytes() + b"\n# external edit\n")
    assert not benchmark.configuration_unchanged(args, cohort)


def test_post_run_source_and_config_errors_preserve_measured_request(monkeypatch, tmp_path):
    request = record("one", 200, "succeeded")
    request["observed_task_status"] = "succeeded"
    args = benchmark.parse_args(["run", "--config", str(tmp_path / "deleted.yaml")])

    def broken_source(_):
        raise OSError("source disappeared")

    monkeypatch.setattr(benchmark, "source_snapshot", broken_source)
    benchmark.check_provenance_after(request, args, {}, tmp_path / "deleted.json", "hash")
    benchmark.apply_provenance_verdict(request)
    assert "source disappeared" in request["provenance_error"]
    assert "config:" in request["provenance_error"]
    assert request["status"] == "provenance_changed"
    assert request["observed_task_status"] == "succeeded"
    group = benchmark.aggregate_records([request])["groups"][0]
    assert group["requests"] == 1
    assert group["all_request_end_latency"]["n"] == 1
    assert group["success_completion_latency"]["n"] == 0


def test_run_freezes_config_and_preserves_request_after_post_run_git_failure(monkeypatch, tmp_path):
    original = tmp_path / "original.json"
    config = {
        "model": {"provider": "mock", "model_name": "mock", "base_url": None},
        "output": {"directory": "ignored"},
        "logging": {"directory": "ignored"},
    }
    original.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "resolve_configuration", lambda _: (config, {}))
    samples = []

    def sampled_source(_):
        samples.append(1)
        if len(samples) == 3:
            raise OSError("post-run git unavailable")
        return {"commit": "abc", "dirty": False}

    monkeypatch.setattr(benchmark, "source_snapshot", sampled_source)

    def measured_stub(command, **kwargs):
        frozen = Path(command[command.index("--config") + 1])
        assert frozen != original
        assert json.loads(frozen.read_text(encoding="utf-8"))["model"] == config["model"]
        directory = Path(command[command.index("--output-directory") + 1])
        identity = command[command.index("--run-id") + 1]
        (directory / identity).mkdir(parents=True)
        (directory / identity / "task_summary.json").write_text(
            json.dumps(
                {
                    "case_id": "T04",
                    "execute": False,
                    "status": "dry_run_completed",
                    "confirmation_ms": 0,
                }
            ),
            encoding="utf-8",
        )
        return {
            **measurement(),
            "outer_e2e_ms": 150,
            "started_at": benchmark.utc_now(),
            "finished_at": benchmark.utc_now(),
        }

    monkeypatch.setattr(benchmark, "measure_process", measured_stub)
    output = tmp_path / "output"
    assert (
        benchmark.main(["run", "--config", str(original), "--output-directory", str(output)]) == 2
    )
    saved = list(output.rglob("e2e_run.json"))
    assert len(saved) == 1
    request = json.loads(saved[0].read_text(encoding="utf-8"))
    assert request["outer_e2e_ms"] == 150
    assert request["status"] == "provenance_changed"
    assert request["observed_task_status"] == "dry_run_completed"
    assert "post-run git unavailable" in request["provenance_error"]
    result = json.loads(next(output.rglob("aggregate.json")).read_text(encoding="utf-8"))
    assert result["record_count"] == 1


@pytest.mark.parametrize("value", [None, float("nan"), -1, True])
def test_invalid_latency_cannot_be_silently_removed(value):
    with pytest.raises(ValueError, match="every request"):
        benchmark.aggregate_records([record("one", value)])


def test_duplicate_or_old_runtime_records_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        benchmark.aggregate_records([record("same", 1), record("same", 2)])
    with pytest.raises(ValueError, match="runtime-only"):
        benchmark.aggregate_records([{"elapsed_ms": 2, "run_id": "old"}])


def test_outer_deadline_overrides_a_written_success_summary():
    status, _ = benchmark.classify_summary(
        sent_summary(),
        measurement(outer_timeout=True),
        case_id="T04",
        execute=True,
        marker="WEEK4_MESSAGE_CHECK_test",
    )
    assert status == "outer_timed_out"


def test_success_requires_summary_exit_and_exact_sent_marker():
    marker = "WEEK4_MESSAGE_CHECK_test"
    classify = lambda summary, result: benchmark.classify_summary(
        summary,
        result,
        case_id="T04",
        execute=True,
        marker=marker,
    )[0]
    assert classify(None, measurement()) == "missing_summary"
    assert classify(sent_summary(), measurement(1)) == "invalid_summary"
    assert classify(sent_summary("old marker"), measurement()) == "invalid_summary"
    summary = sent_summary()
    summary["verification"]["evidence"]["vision_assessment"]["composer_empty"] = False
    assert classify(summary, measurement()) == "invalid_summary"
    assert classify(sent_summary(), measurement()) == "succeeded"


def test_unique_short_markers_even_when_runs_share_a_second():
    identities = [benchmark.new_identity("T04") for _ in range(100)]
    assert len({identity[0] for identity in identities}) == 100
    assert len({identity[1] for identity in identities}) == 100
    assert all(len(marker) <= 48 for _, marker in identities)


def test_warm_claim_requires_matching_ready_record(tmp_path):
    config = {
        "model": {
            "provider": "openai_compatible",
            "model_name": "qwen",
            "base_url": "http://localhost/v1",
        }
    }
    path = tmp_path / "warmup.json"
    assert (
        benchmark.warmup_metadata(None, config, "warm")["warm_state_evidence"]
        == "declared_unverified"
    )
    path.write_text(
        json.dumps(
            {
                "ready": True,
                "provider": "openai_compatible",
                "model_name": "other",
                "base_url": "http://localhost/v1",
            }
        ),
        encoding="utf-8",
    )
    assert (
        benchmark.warmup_metadata(path, config, "warm")["warm_state_evidence"]
        == "declared_unverified"
    )
    path.write_text(
        json.dumps(
            {
                "ready": True,
                "provider": "openai_compatible",
                "model_name": "qwen",
                "base_url": "http://localhost/v1",
            }
        ),
        encoding="utf-8",
    )
    assert (
        benchmark.warmup_metadata(path, config, "warm")["warm_state_evidence"]
        == "matching_ready_probe_record"
    )


def test_environment_cannot_redirect_repeated_t04_messages(monkeypatch):
    monkeypatch.setenv("GUI_AGENT_MESSAGE_CONVERSATION", "Another Person")
    monkeypatch.setenv("GUI_AGENT_MESSAGE_INSTRUCTION", "Send anything to another person")
    args = benchmark.parse_args(["plan", "--provider", "mock", "--model", "mock-vision-model"])
    _, env = benchmark.resolve_configuration(args)
    assert env["GUI_AGENT_MESSAGE_CONVERSATION"] == "文件传输助手"
    assert "Another Person" not in env["GUI_AGENT_MESSAGE_INSTRUCTION"]
    assert "WEEK4_MESSAGE_CHECK_001" in env["GUI_AGENT_MESSAGE_INSTRUCTION"]


def test_measured_child_captures_final_output_and_startup_shutdown_cost(tmp_path):
    result = benchmark.measure_process(
        [sys.executable, "-c", "import time; time.sleep(.08); print('succeeded is just prose')"],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=5,
        output=tmp_path / "terminal.log",
        tee=False,
    )
    assert result["returncode"] == 0
    assert result["outer_e2e_ms"] >= 80
    assert b"succeeded is just prose" in (tmp_path / "terminal.log").read_bytes()
    assert (
        benchmark.classify_summary(
            None,
            result,
            case_id="T04",
            execute=True,
            marker="marker",
        )[0]
        == "missing_summary"
    )


def test_measured_child_timeout_is_not_dropped(tmp_path):
    result = benchmark.measure_process(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=0.1,
        output=tmp_path / "terminal.log",
        tee=False,
    )
    assert result["outer_timeout"] is True
    assert result["returncode"] is not None
    assert result["outer_e2e_ms"] >= 100


def test_launch_error_still_has_measured_request_latency(tmp_path):
    result = benchmark.measure_process(
        [str(tmp_path / "does-not-exist")],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=1,
        output=tmp_path / "terminal.log",
        tee=False,
    )
    assert result["outer_e2e_ms"] >= 0
    assert (
        benchmark.classify_summary(
            None,
            result,
            case_id="T04",
            execute=True,
            marker="marker",
        )[0]
        == "harness_error"
    )


@pytest.mark.parametrize(
    "answer, expected_requests, exit_code", [("STOP", 1, 130), ("READY", 2, 0)]
)
def test_preparation_gate_preserves_completed_requests_before_the_next_launch(
    monkeypatch,
    tmp_path,
    answer,
    expected_requests,
    exit_code,
):
    config = {
        "model": {"provider": "mock", "model_name": "mock", "base_url": None},
        "output": {"directory": "ignored"},
        "logging": {"directory": "ignored"},
    }
    original = tmp_path / "config.json"
    original.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(benchmark, "resolve_configuration", lambda _: (config, {}))
    monkeypatch.setattr(benchmark, "source_snapshot", lambda _: {"commit": "fixed"})
    monkeypatch.setattr(benchmark, "configuration_unchanged", lambda *_: True)
    monkeypatch.setattr(
        benchmark,
        "check_provenance_after",
        lambda row, *_: row.update(provenance_changed=False),
    )
    launched = []

    def measured(command, **kwargs):
        launched.append(command)
        return {
            **measurement(2),
            "outer_e2e_ms": 10.0,
            "started_at": benchmark.utc_now(),
            "finished_at": benchmark.utc_now(),
        }

    monkeypatch.setattr(benchmark, "measure_process", measured)
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert (
        benchmark.main(
            [
                "run",
                "--case",
                "T04",
                "--repeats",
                "2",
                "--pause-between-runs",
                "--config",
                str(original),
                "--output-directory",
                str(tmp_path / "records"),
            ]
        )
        == exit_code
    )
    records = list((tmp_path / "records").rglob("e2e_run.json"))
    assert len(launched) == len(records) == expected_requests
    aggregate_path = next((tmp_path / "records").rglob("aggregate.json"))
    group = json.loads(aggregate_path.read_text(encoding="utf-8"))["groups"][0]
    assert group["requests"] == expected_requests
    assert group["succeeded"] == 0
