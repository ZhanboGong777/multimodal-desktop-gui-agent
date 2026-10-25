"""Natural-language CLI remains dry-run/confirmation guarded and budgets include decomposition."""

from __future__ import annotations

import json

import pytest
import test_week6_cli as legacy
from test_week6_automatic_decomposition import ScriptedClient, ready
from test_week6_cli import forbid_runtime

from gui_agent.runtime.monitoring import read_run_status

cli = legacy.cli
offline = legacy.offline


def args(tmp_path, *extra):
    return ["--instruction", "Open the editor and prepare a local draft", "--workflow-id", "AutoEditor",
            "--expect-text", "AllDone", "--forbid-text", "ERROR", "--run-id", "automatic_run",
            "--output-directory", str(tmp_path / "runs"), *extra]


def test_validate_natural_language_only_does_not_generate_any_subgoal_or_touch_runtime(cli, monkeypatch, tmp_path, capsys):
    forbid_runtime(monkeypatch, cli)
    monkeypatch.setattr(cli, "AutomaticWorkflowRunner", lambda *_a, **_k: pytest.fail("no automatic execution"))
    assert cli.main(args(tmp_path, "--validate-only")) == 0
    assert json.loads(capsys.readouterr().out) == {
        "workflow_id": "AutoEditor", "subgoals": None, "valid": True, "desktop_actions": 0,
        "model_calls": 0, "mode": "automatic", "decomposition_pending": True,
    }
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("change", ["no_checks", "message", "huge_goal_count", "one_planning_call"])
def test_invalid_automatic_input_is_rejected_before_configuration_or_model(cli, monkeypatch, tmp_path, change):
    forbid_runtime(monkeypatch, cli)
    argv = args(tmp_path)
    if change == "no_checks":
        argv.remove("--expect-text")
        argv.remove("AllDone")
        argv.remove("--forbid-text")
        argv.remove("ERROR")
    elif change == "message":
        argv[1] = "Send a message to a colleague"
    elif change == "huge_goal_count":
        argv += ["--max-subgoals", "33"]
    else:
        argv += ["--planning-attempts", "1"]
    assert cli.main(argv) == 2 and not (tmp_path / "runs").exists()


def test_execute_automatic_still_requires_interactive_confirmation_before_decomposition(cli, monkeypatch, tmp_path):
    forbid_runtime(monkeypatch, cli)
    monkeypatch.setattr(cli, "_has_interactive_stdin", lambda: False)
    assert cli.main(args(tmp_path, "--execute")) == 2
    assert not (tmp_path / "runs").exists()


def test_natural_language_real_orchestration_dry_run_counts_decomposition_in_shared_planning(cli, offline, monkeypatch, tmp_path):
    client = ScriptedClient([ready("Open editor", "Complete the draft")])
    monkeypatch.setattr(cli, "create_model_client", lambda _config: client)
    assert cli.main(args(tmp_path, "--planning-attempts", "4", "--model-requests", "5")) == 0
    directory = tmp_path / "runs/automatic_run"
    result = json.loads((directory / "automatic_workflow_summary.json").read_text("utf-8"))
    assert result["status"] == "dry_run_completed" and result["planning_calls"] == 3
    assert result["model_requests"] == 1  # stub children deliberately make no inference
    assert len(client.calls) == 1 and len(offline.runs) == 2
    assert [options.max_planning_attempts for _, options, _ in offline.runs] == [3, 2]
    assert all(not options.execute and options.confirm for _, options, _ in offline.runs)
    assert all(0 < options.task_timeout_seconds < 90 for _, options, _ in offline.runs)
    assert offline.observed_frames == 1 and offline.confirms == []  # initial grounding only
    status = read_run_status(directory)
    assert status["trusted"] and status["snapshot"]["task_status"] == "dry_run_completed"
    assert status["snapshot"]["planning_pass"] == 3
    assert status["snapshot"]["subgoals_total"] == 0  # parent budget monitor; graph has exact counts


def test_natural_language_execution_confirms_every_generated_stage_and_final_check_is_independent(cli, offline, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "create_model_client", lambda _config: ScriptedClient([ready("Open editor", "Complete draft")]))
    assert cli.main(args(tmp_path, "--execute")) == 0
    assert offline.confirms == ["G01", "G02"]
    assert offline.risk_confirms == [("G01", "G01"), ("G02", "G02")]
    assert offline.countdowns == [3, 3] and offline.observed_frames == 2
    raw = json.loads((tmp_path / "runs/automatic_run/automatic_workflow_summary.json").read_text("utf-8"))
    assert raw["decomposition"]["workflow"]["final_check"]["expect_text"] == ["AllDone"]
    assert raw["decomposition"]["workflow"]["final_check"]["forbid_text"] == ["ERROR"]
    assert raw["workflow_result"]["final_verification"]["outcome"] == "passed"


def test_last_request_spent_decomposing_prevents_child_creation_and_returns_failure(cli, offline, monkeypatch, tmp_path):
    client = ScriptedClient([ready("Open editor")])
    monkeypatch.setattr(cli, "create_model_client", lambda _config: client)
    assert cli.main(args(tmp_path, "--model-requests", "1")) == 1
    assert len(client.calls) == 1 and offline.children == []
    assert len(offline.observers) == 1  # initial screen, no child observer
    raw = json.loads((tmp_path / "runs/automatic_run/automatic_workflow_summary.json").read_text("utf-8"))
    assert raw["status"] == "blocked" and raw["workflow_result"] is None


def test_model_clarification_cannot_become_dry_run_success(cli, offline, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "create_model_client", lambda _config: ScriptedClient([
        json.dumps({"status": "needs_clarification", "reason": "which local editor?", "subgoals": []}),
    ]))
    assert cli.main(args(tmp_path)) == 1 and offline.children == []
    raw = json.loads((tmp_path / "runs/automatic_run/automatic_workflow_summary.json").read_text("utf-8"))
    assert raw["status"] == "needs_clarification" and raw["decomposition"]["error"] == "which local editor?"
