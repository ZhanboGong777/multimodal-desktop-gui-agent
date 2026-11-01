"""Semantic-plan revisions stay visible without rewriting verified progress."""

from __future__ import annotations

import json

import pytest

from gui_agent.runtime import monitoring
from gui_agent.runtime.monitoring import RunMonitor, read_run_status


@pytest.fixture(autouse=True)
def process_identity(monkeypatch):
    monkeypatch.setattr(monitoring, "_process_info", lambda _pid: (True, "windows:1234"))


def start(path, *, dynamic=True, total=0):
    writer = RunMonitor(path, clock=lambda: 50.0)
    writer.start("SEMANTIC", True, 10, 12, total, dynamic_subgoals=dynamic)
    return writer


def events(path):
    return [json.loads(line) for line in (path / "status_events.jsonl").read_text("utf-8").splitlines()]


def rewrite(path, records):
    (path / "status_events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (path / "run_status.json").write_text(json.dumps(records[-1]), encoding="utf-8")


def test_revision_counts_remaining_plan_and_keeps_verified_goals(tmp_path):
    writer = start(tmp_path)
    writer.update("decomposing", planning_pass=1, subgoals_total=3)
    writer.update("executing", actions_used=1, subgoal_id="project")
    writer.update("subgoal_verification", subgoals_completed=1, subgoal_id="project")
    # A replacement plan combines the two remaining objectives into one.
    writer.update("decomposing", planning_pass=2, subgoals_total=2)
    writer.update("executing", actions_used=2, subgoal_id="finish")
    writer.update("subgoal_verification", subgoals_completed=2, subgoal_id="finish")
    final = writer.finish("succeeded", subgoals_completed=2)
    result = read_run_status(tmp_path)
    assert result["state"] == "finished" and result["trusted"]
    assert result["snapshot"] == final
    assert final["subgoals_total"] == final["subgoals_completed"] == 2
    history = events(tmp_path)
    assert [record["subgoals_total"] for record in history] == [0, 3, 3, 3, 2, 2, 2, 2]
    assert [record["subgoals_completed"] for record in history] == [0, 0, 0, 1, 1, 1, 2, 2]


def test_legacy_schema_and_fields_stay_compatible(tmp_path):
    writer = start(tmp_path, dynamic=False, total=2)
    writer.update("decomposing", subgoals_total=2)
    writer.finish("failed", subgoals_completed=0)
    record = read_run_status(tmp_path)["snapshot"]
    assert record["schema"] == monitoring.STATUS_SCHEMA
    assert set(record) == monitoring._FIELDS
    assert "dynamic_subgoals" not in record


@pytest.mark.parametrize("stage", ["decomposing", "planning", "executing"])
def test_legacy_total_cannot_change_even_during_decomposition(tmp_path, stage):
    writer = start(tmp_path, dynamic=False, total=2)
    before = events(tmp_path)
    with pytest.raises(ValueError, match="total change"):
        writer.update(stage, subgoals_total=3)
    assert events(tmp_path) == before
    assert read_run_status(tmp_path)["trusted"]


@pytest.mark.parametrize("stage", ["planning", "executing", "subgoal_verification", "recovering"])
def test_semantic_total_change_is_restricted_to_accepted_plan_stage(tmp_path, stage):
    writer = start(tmp_path)
    before = events(tmp_path)
    with pytest.raises(ValueError, match="total change"):
        writer.update(stage, subgoals_total=3)
    assert events(tmp_path) == before


@pytest.mark.parametrize("total", [True, -1, 10_001, 1.5, "3"])
def test_invalid_dynamic_total_never_persists(tmp_path, total):
    writer = start(tmp_path)
    before = events(tmp_path)
    with pytest.raises(ValueError):
        writer.update("decomposing", subgoals_total=total)
    assert events(tmp_path) == before


@pytest.mark.parametrize("dynamic", [1, "true", None])
def test_schema_opt_in_must_be_boolean(tmp_path, dynamic):
    with pytest.raises(ValueError, match="dynamic_subgoals"):
        start(tmp_path, dynamic=dynamic)
    assert not (tmp_path / "status_events.jsonl").exists()


def test_replan_cannot_erase_completed_work_or_regress_progress(tmp_path):
    writer = start(tmp_path, total=3)
    writer.update("subgoal_verification", subgoals_completed=2)
    before = events(tmp_path)
    with pytest.raises(ValueError):
        writer.update("decomposing", subgoals_total=1)
    with pytest.raises(ValueError, match="nonmonotonic"):
        writer.update("decomposing", subgoals_total=3, subgoals_completed=1)
    assert events(tmp_path) == before
    writer.update("decomposing", subgoals_total=4)
    assert read_run_status(tmp_path)["snapshot"]["subgoals_completed"] == 2


@pytest.mark.parametrize("tamper", ["schema", "legacy_total", "wrong_stage", "terminal_total"])
def test_reader_refuses_tampered_plan_history_even_when_latest_matches(tmp_path, tamper):
    writer = start(tmp_path)
    writer.update("decomposing", subgoals_total=2)
    writer.finish("failed")
    history = events(tmp_path)
    if tamper == "schema":
        history[1]["schema"] = monitoring.STATUS_SCHEMA
    elif tamper == "legacy_total":
        for record in history:
            record["schema"] = monitoring.STATUS_SCHEMA
    elif tamper == "wrong_stage":
        history[1]["stage"] = "executing"
    else:
        history[-1]["subgoals_total"] = 3
    rewrite(tmp_path, history)
    result = read_run_status(tmp_path)
    assert result["state"] == "inconsistent" and not result["trusted"]
    assert "journal_identity_mismatch" in result["diagnostics"]
