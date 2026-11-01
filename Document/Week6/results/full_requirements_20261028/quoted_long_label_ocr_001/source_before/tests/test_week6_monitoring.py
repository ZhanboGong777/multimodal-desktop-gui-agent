"""Passive monitoring: evidence integrity, privacy and concurrent readers."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import get_args

import pytest

from gui_agent.runtime import monitoring
from gui_agent.runtime.monitoring import RunMonitor, read_run_status
from gui_agent.runtime.schemas import TaskStatus

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/week6_status.py"


class Clock:
    value = 50.0

    def __call__(self):
        return self.value


@pytest.fixture(autouse=True)
def process_identity(monkeypatch):
    monkeypatch.setattr(monitoring, "_process_info", lambda _pid: (True, "windows:1234"))


def start(tmp_path, *, execute=True, subgoals=0, clock=None):
    writer = RunMonitor(tmp_path, clock=clock or Clock())
    writer.start("T06", execute, max_actions=10, max_planning_attempts=3,
                 subgoals_total=subgoals)
    return writer


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "status_events.jsonl").read_text("utf-8").splitlines()]


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_lifecycle_counts_only_verified_subgoals_and_keeps_every_event(tmp_path):
    clock = Clock()
    writer = start(tmp_path, subgoals=2, clock=clock)
    clock.value += 1
    writer.update("planning", planning_pass=1)
    clock.value += 2
    writer.update("executing", actions_used=1, subgoal_id="goal_1")
    writer.update("subgoal_verification", subgoals_completed=1, detail_code="verified")
    clock.value += 1
    final = writer.finish("failed", actions_used=2, subgoals_completed=1)
    history = events(tmp_path)
    assert [event["seq"] for event in history] == [1, 2, 3, 4, 5]
    assert history[2]["subgoals_completed"] == 0
    assert history[-1] == final == json.loads((tmp_path / "run_status.json").read_text("utf-8"))
    assert final["elapsed_ms"] == 4000
    result = read_run_status(tmp_path)
    assert result["state"] == "finished" and result["trusted"]
    assert result["snapshot"]["task_status"] == "failed"
    assert result["snapshot"]["subgoals_completed"] == 1
    with pytest.raises(ValueError, match="finished"):
        writer.update("executing")
    with pytest.raises(ValueError, match="finished"):
        writer.finish("succeeded")


def test_terminal_status_set_tracks_runtime_schema():
    assert monitoring.TASK_STATUSES == frozenset(get_args(TaskStatus))


@pytest.mark.parametrize("execute,status", [(False, "succeeded"), (True, "dry_run_completed"),
                                           (True, "complete"), (True, "running")])
def test_finish_refuses_invented_or_misleading_task_status(tmp_path, execute, status):
    writer = start(tmp_path, execute=execute)
    before = events(tmp_path)
    with pytest.raises(ValueError):
        writer.finish(status)
    assert events(tmp_path) == before
    assert read_run_status(tmp_path)["state"] == "active"


def test_dry_run_finish_is_explicit_and_never_reports_success(tmp_path):
    writer = start(tmp_path, execute=False)
    writer.finish("dry_run_completed", actions_used=3)
    result = read_run_status(tmp_path)
    assert result["state"] == "finished"
    assert result["snapshot"]["task_status"] == "dry_run_completed"
    assert result["snapshot"]["execute"] is False


@pytest.mark.parametrize("changes", [
    {"stage": "this contains private task prose"},
    {"stage": "planning", "detail_code": "password=secret with spaces"},
    {"stage": "resolving", "subgoal_id": "click (100, 200)"},
    {"stage": "planning", "planning_pass": 4},
    {"stage": "executing", "actions_used": 11},
    {"stage": "subgoal_verification", "subgoals_completed": 2},
    {"stage": "finished"},
    {"stage": "executing", "actions_used": True},
])
def test_invalid_or_sensitive_fields_never_reach_any_monitor_file(tmp_path, changes):
    writer = start(tmp_path, subgoals=1)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        writer.update(**changes)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_progress_and_clock_cannot_go_backwards(tmp_path):
    clock = Clock()
    writer = start(tmp_path, subgoals=2, clock=clock)
    clock.value += 1
    writer.update("executing", actions_used=3, planning_pass=2, subgoals_completed=1)
    for changes in ({"actions_used": 2}, {"planning_pass": 1}, {"subgoals_completed": 0}):
        with pytest.raises(ValueError):
            writer.update("executing", **changes)
    clock.value -= 1
    with pytest.raises(ValueError, match="elapsed"):
        writer.update("executing")
    assert len(events(tmp_path)) == 2


def test_does_not_overwrite_monitor_evidence_and_accepts_existing_legacy_directory(tmp_path):
    (tmp_path / "steps.jsonl").write_text('legacy steps stay here\n', encoding="utf-8")
    start(tmp_path)
    before = (tmp_path / "run_status.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        start(tmp_path)
    assert (tmp_path / "run_status.json").read_bytes() == before
    assert (tmp_path / "steps.jsonl").read_text("utf-8") == 'legacy steps stay here\n'


@pytest.mark.parametrize("phase", ["before", "partial", "after"])
def test_interrupted_append_cannot_invent_trusted_completion(tmp_path, phase):
    writer = start(tmp_path)
    append = writer._append_event

    def interrupt(status):
        if phase == "partial":
            with (tmp_path / "status_events.jsonl").open("ab") as stream:
                stream.write(b'{"schema":')
        elif phase == "after":
            append(status)
        raise KeyboardInterrupt

    writer._append_event = interrupt
    with pytest.raises(KeyboardInterrupt):
        writer.update("executing", actions_used=1)
    observed = read_run_status(tmp_path)
    if phase == "before":
        assert observed["state"] == "active"  # only the previous complete snapshot exists
    else:
        assert observed["state"] == "inconsistent" and not observed["trusted"]
    writer._append_event = append
    writer.finish("cancelled", actions_used=1)
    observed = read_run_status(tmp_path)
    if phase == "after":
        assert observed["state"] == "finished"
        assert observed["snapshot"]["task_status"] == "cancelled"
    else:
        assert observed["state"] == "inconsistent" and not observed["trusted"]


def test_atomic_latest_is_whole_during_replace_and_reader_never_repairs_files(tmp_path, monkeypatch):
    writer = start(tmp_path)
    replace = monitoring.os.replace
    observed = []

    def inspect(source, destination):
        observed.append(json.loads(Path(destination).read_text("utf-8")))
        observed.append(read_run_status(tmp_path))
        replace(source, destination)

    monkeypatch.setattr(monitoring.os, "replace", inspect)
    writer.update("planning", planning_pass=1)
    assert observed[0]["seq"] == 1
    assert observed[1]["state"] == "inconsistent"
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert read_run_status(tmp_path)["state"] == "active"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_readers_never_observe_torn_latest_or_false_finished(tmp_path, monkeypatch):
    writer = start(tmp_path)
    done = threading.Event()
    observations = []

    def read():
        while not done.is_set():
            observations.append(read_run_status(tmp_path))

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for _ in range(20):
            writer.update("observing")
    finally:
        done.set()
        reader.join(timeout=5)
    assert not reader.is_alive() and observations
    for item in observations:
        if item["state"] == "active":
            assert item["trusted"] and item["diagnostics"] == []
        elif item["state"] == "inconsistent":
            assert not item["trusted"] and item["snapshot"] is None
            assert item["diagnostics"] and set(item["diagnostics"]) <= {
                "latest_journal_mismatch", "journal_incomplete_line", "journal_invalid_record",
            }
        elif item["state"] == "unavailable":
            # A Windows sharing denial can affect the read itself as well as
            # replacement. It must never expose a trusted or partial snapshot.
            assert not item["trusted"] and item["snapshot"] is None
            assert item["diagnostics"] == ["monitor_files_unreadable"]
        else:
            # On Windows a real sharing denial can outlast the bounded replace
            # retries. It is an honest persistence failure, not a torn snapshot
            # or permission to claim that the run is still active.
            assert item["state"] == "degraded" and not item["trusted"]
            assert item["diagnostics"] == item["snapshot"]["write_failures"]
            assert item["diagnostics"] and set(item["diagnostics"]) <= set(writer.write_failures)
        if item["snapshot"] is not None:
            assert item["snapshot"]["event"] != "finished"
            assert item["snapshot"]["task_status"] is None
    writer.finish("cancelled")
    final = read_run_status(tmp_path)
    assert events(tmp_path)[-1]["task_status"] == "cancelled"
    if final["state"] == "unavailable":
        assert not final["trusted"] and final["snapshot"] is None
        assert final["diagnostics"] == ["monitor_files_unreadable"]
    elif not writer.write_failures:
        assert final["state"] == "finished" and final["trusted"]
        assert final["snapshot"]["task_status"] == "cancelled"
    else:
        assert not final["trusted"]
        if final["state"] == "degraded":
            assert final["diagnostics"] == list(writer.write_failures)
        else:
            assert final["state"] == "inconsistent"
            assert final["diagnostics"] == ["latest_journal_mismatch"]

    # Exercise the same denial deterministically even when this run's real
    # concurrent reads happened not to encounter Windows sharing contention.
    def denied(_path):
        raise PermissionError("simulated reader sharing denial")

    with monkeypatch.context() as context:
        context.setattr(monitoring, "_read_latest", denied)
        unavailable = read_run_status(tmp_path)
    assert unavailable["state"] == "unavailable" and not unavailable["trusted"]
    assert unavailable["snapshot"] is None
    assert unavailable["diagnostics"] == ["monitor_files_unreadable"]


def test_transient_replace_permission_error_retries_only_the_same_snapshot(tmp_path, monkeypatch):
    writer = start(tmp_path)
    replace = monitoring.os.replace
    attempts, waits = [], []

    def busy(source, destination):
        attempts.append((source, destination))
        if len(attempts) <= 2:
            raise PermissionError("temporary sharing denial; private error text")
        replace(source, destination)

    monkeypatch.setattr(monitoring.os, "replace", busy)
    monkeypatch.setattr(monitoring.time, "sleep", waits.append)
    writer.update("planning", planning_pass=1)
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert waits == list(monitoring._REPLACE_RETRY_DELAYS[:2])
    assert writer.write_failures == ()
    assert [event["seq"] for event in events(tmp_path)] == [1, 2]
    assert read_run_status(tmp_path)["state"] == "active"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_persistent_replace_denial_is_bounded_visible_and_never_falsely_active(tmp_path, monkeypatch, caplog, error):
    writer = start(tmp_path)
    replace = monitoring.os.replace
    original = (tmp_path / "run_status.json").read_bytes()
    attempts, waits = [], []

    def denied(_source, _destination):
        attempts.append(1)
        raise error("private filesystem error text")

    monkeypatch.setattr(monitoring.os, "replace", denied)
    monkeypatch.setattr(monitoring.time, "sleep", waits.append)
    state = writer.update("executing", actions_used=1)
    assert len(attempts) == (4 if error is PermissionError else 1)
    assert waits == (list(monitoring._REPLACE_RETRY_DELAYS) if error is PermissionError else [])
    assert state["write_failures"] == ["latest_write_failed"]
    assert "latest_write_failed" in caplog.text and "private filesystem" not in caplog.text
    assert (tmp_path / "run_status.json").read_bytes() == original
    assert [event["seq"] for event in events(tmp_path)] == [1, 2]
    current = read_run_status(tmp_path)
    assert current["state"] == "inconsistent" and not current["trusted"]
    assert current["diagnostics"] == ["latest_journal_mismatch"]
    assert not list(tmp_path.glob("*.tmp"))

    # A later successful write must not erase the earlier evidence failure or
    # turn a recorded terminal verdict into a falsely trusted completion.
    monkeypatch.setattr(monitoring.os, "replace", replace)
    writer.finish("failed", actions_used=1)
    final = read_run_status(tmp_path)
    assert final["state"] == "degraded" and not final["trusted"]
    assert final["diagnostics"] == ["latest_write_failed"]
    assert final["snapshot"]["task_status"] == "failed"


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing behavior")
def test_real_windows_reader_handle_prevents_replace_without_torn_snapshot(tmp_path):
    writer = start(tmp_path)
    latest = tmp_path / "run_status.json"
    original = latest.read_bytes()
    # CPython's regular Windows read handle lacks FILE_SHARE_DELETE. Holding it
    # longer than the bounded retries deterministically reproduces the race.
    with latest.open("rb") as reader:
        result = writer.update("observing")
        assert result["write_failures"] == ["latest_write_failed"]
        assert reader.read() == original
        assert latest.read_bytes() == original
    assert events(tmp_path)[-1]["seq"] == 2
    observed = read_run_status(tmp_path)
    assert observed["state"] == "inconsistent" and not observed["trusted"]
    assert observed["diagnostics"] == ["latest_journal_mismatch"]
    writer.update("planning", planning_pass=1)
    observed = read_run_status(tmp_path)
    assert observed["state"] == "degraded" and not observed["trusted"]
    assert observed["diagnostics"] == ["latest_write_failed"]
    assert observed["snapshot"]["event"] != "finished"


def test_replace_retry_wait_is_not_subtracted_from_the_callers_clock(tmp_path, monkeypatch):
    clock = Clock()
    writer = start(tmp_path, clock=clock)
    replace = monitoring.os.replace
    attempts = []
    started = clock.value

    def briefly_busy(source, destination):
        attempts.append(1)
        if len(attempts) == 1:
            raise PermissionError("sharing denial")
        replace(source, destination)

    def wait(seconds):
        clock.value += seconds

    monkeypatch.setattr(monitoring.os, "replace", briefly_busy)
    monkeypatch.setattr(monitoring.time, "sleep", wait)
    writer.update("executing")
    assert clock.value - started == pytest.approx(monitoring._REPLACE_RETRY_DELAYS[0])
    writer.finish("timed_out")
    assert read_run_status(tmp_path)["snapshot"]["elapsed_ms"] == pytest.approx(5.0)


@pytest.mark.parametrize("damage", ["tail", "bad_utf8", "wrong_sequence", "finished_only_in_latest",
                                   "changed_budget", "duplicate_key", "sensitive_extra"])
def test_reader_rejects_corrupt_or_inconsistent_evidence_without_echoing_it(tmp_path, damage):
    writer = start(tmp_path)
    writer.update("planning", planning_pass=1)
    latest = tmp_path / "run_status.json"
    journal = tmp_path / "status_events.jsonl"
    if damage == "tail":
        with journal.open("ab") as stream:
            stream.write(b'{"seq":3')
    elif damage == "bad_utf8":
        latest.write_bytes(b'\xff')
    elif damage == "duplicate_key":
        latest.write_text('{"seq":1,"seq":2}', encoding="utf-8")
    else:
        current = json.loads(latest.read_text("utf-8"))
        if damage == "wrong_sequence":
            current["seq"] += 2
        elif damage == "finished_only_in_latest":
            current.update(event="finished", stage="finished", task_status="succeeded")
        elif damage == "changed_budget":
            current["max_actions"] = 9
        else:
            current["private_instruction"] = "secret should never be echoed"
        dump(latest, current)
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = read_run_status(tmp_path)
    assert result["state"] in {"corrupt", "inconsistent"} and not result["trusted"]
    assert result["snapshot"] is None
    assert "secret should never be echoed" not in json.dumps(result)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_stale_dead_reused_and_unverifiable_writer_never_report_active(tmp_path, monkeypatch):
    start(tmp_path)
    current = read_run_status(tmp_path)["snapshot"]
    recorded = datetime.fromisoformat(current["updated_at_utc"])
    assert read_run_status(tmp_path, now=recorded + timedelta(seconds=31))["state"] == "stale"
    for info, state in (((False, None), "inactive"), ((True, "windows:9999"), "inactive"),
                        ((None, None), "unconfirmed")):
        monkeypatch.setattr(monitoring, "_process_info", lambda _pid, value=info: value)
        result = read_run_status(tmp_path, now=recorded)
        assert result["state"] == state and not result["trusted"]
    assert read_run_status(tmp_path, now=recorded - timedelta(seconds=30))["state"] == "unconfirmed"


def test_old_terminal_status_does_not_become_a_stale_active_run(tmp_path):
    writer = start(tmp_path)
    final = writer.finish("succeeded")
    recorded = datetime.fromisoformat(final["updated_at_utc"])
    result = read_run_status(tmp_path, now=recorded + timedelta(days=30))
    assert result["state"] == "finished" and result["trusted"]


@pytest.mark.parametrize("missing", ["run_status.json", "status_events.jsonl"])
def test_missing_half_of_evidence_is_unconfirmed(tmp_path, missing):
    start(tmp_path)
    (tmp_path / missing).unlink()
    assert read_run_status(tmp_path)["state"] == "inconsistent"


def test_legacy_directory_is_safe_but_never_claims_verified_monitoring(tmp_path):
    dump(tmp_path / "task_summary.json", {
        "case_id": "T01", "status": "succeeded", "execute": True,
        "instruction": "secret instruction", "error": "private exception text",
        "steps": [{"resolved": {"x": 123, "text": "private typed text"}}],
    })
    before = (tmp_path / "task_summary.json").read_bytes()
    result = read_run_status(tmp_path)
    assert result["state"] == "legacy" and not result["trusted"]
    assert result["snapshot"] == {"case_id": "T01", "task_status": "succeeded", "execute": True}
    assert "secret" not in json.dumps(result) and "private" not in json.dumps(result)
    assert (tmp_path / "task_summary.json").read_bytes() == before
    assert read_run_status(tmp_path / "missing")["state"] == "missing"


@pytest.mark.parametrize("target", ["journal", "latest"])
def test_write_failures_are_visible_and_never_retry_or_raise_into_actions(tmp_path, monkeypatch, caplog, target):
    writer = start(tmp_path)
    calls = []

    def unavailable(_status):
        calls.append(1)
        raise PermissionError("private OS path and raw exception")

    monkeypatch.setattr(writer, "_append_event" if target == "journal" else "_replace_latest", unavailable)
    state = writer.update("executing", actions_used=1)
    assert len(calls) == 1
    assert state["write_failures"] == [f"{target}_write_failed"]
    assert writer.write_failures == (f"{target}_write_failed",)
    assert "persistence failure" in caplog.text and "private OS path" not in caplog.text
    result = read_run_status(tmp_path)
    assert result["state"] == "inconsistent" and not result["trusted"]


def test_repaired_latest_failure_is_still_visibly_degraded(tmp_path, monkeypatch):
    writer = start(tmp_path)
    original = writer._replace_latest

    def unavailable(_status):
        raise OSError("not recorded")

    monkeypatch.setattr(writer, "_replace_latest", unavailable)
    writer.update("observing")
    monkeypatch.setattr(writer, "_replace_latest", original)
    writer.finish("failed")
    result = read_run_status(tmp_path)
    assert result["state"] == "degraded" and not result["trusted"]
    assert result["diagnostics"] == ["latest_write_failed"]


def load_cli():
    spec = importlib.util.spec_from_file_location("_test_week6_status_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_works_with_stdlib_only_and_has_no_model_or_gui_import(tmp_path):
    writer = start(tmp_path, execute=False)
    writer.finish("dry_run_completed")
    result = subprocess.run([sys.executable, "-S", str(CLI), "--run-dir", str(tmp_path), "--json"],
                            capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["state"] == "finished"
    assert payload["snapshot"]["task_status"] == "dry_run_completed"
    # -S excludes all site packages: dependency-heavy runtime imports would fail.
    assert not result.stderr


def test_cli_human_watch_exits_on_terminal_and_keyboard_interrupt(tmp_path, monkeypatch, capsys):
    cli = load_cli()
    replies = iter([
        {"state": "active", "snapshot": {"case_id": "T06", "stage": "planning"}, "diagnostics": []},
        {"state": "finished", "snapshot": {"task_status": "failed"}, "diagnostics": []},
    ])
    monkeypatch.setattr(cli, "_reader", lambda: lambda _path: next(replies))
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", lambda interval: sleeps.append(interval))
    assert cli.main(["--run-dir", str(tmp_path), "--watch", "--interval", "0.05"]) == 0
    assert sleeps == [0.05]
    assert "active case=T06 stage=planning" in capsys.readouterr().out

    monkeypatch.setattr(cli, "_reader", lambda: lambda _path: {"state": "missing", "diagnostics": []})

    def interrupt(_interval):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt)
    assert cli.main(["--run-dir", str(tmp_path), "--watch"]) == 130


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "0.001", "61", "text"])
def test_cli_rejects_invalid_or_unbounded_polling(tmp_path, interval):
    cli = load_cli()
    with pytest.raises(SystemExit) as exc:
        cli.main(["--run-dir", str(tmp_path), "--interval", interval])
    assert exc.value.code == 2
