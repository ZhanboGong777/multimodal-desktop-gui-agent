"""Week 6 CLI wiring through the real workflow, with no desktop/model work."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_runner import _frame

from gui_agent.config import Config
from gui_agent.planning.schemas import PlanStep, TaskPlan
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.schemas import StepRecord, TaskRunResult, TaskSpec, VerificationResult
from gui_agent.runtime.workflow import WorkflowSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_week6_cli_integration", ROOT / "scripts/week6_agent_cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workflow_file(tmp_path):
    spec = WorkflowSpec(
        workflow_id="demo_workflow", instruction="Finish two local stages",
        subgoals=[
            TaskSpec(case_id=f"G{index}", instruction=f"Reach local stage {index}",
                     expect_text=[f"Done{index}"], risk="medium") for index in (1, 2)
        ],
        final_check=TaskSpec(case_id="FINAL", instruction="Read the final local state", expect_text=["AllDone"]),
    )
    path = tmp_path / "workflow.json"
    path.write_text(spec.model_dump_json(), encoding="utf-8-sig")
    return path


def argv(workflow_file, tmp_path, *extra):
    return ["--workflow", str(workflow_file), "--output-directory", str(tmp_path / "runs"),
            "--run-id", "integration_run", *extra]


def no_work(*_args, **_kwargs):
    pytest.fail("validation/refusal must not construct a model, observer, executor or run")


def forbid_runtime(monkeypatch, cli):
    for name in ("load_environment", "load_config", "create_model_client", "ObservationService",
                 "TaskPlanner", "ActionExecutor", "TaskRunner", "WorkflowRunner", "RunMonitor"):
        monkeypatch.setattr(cli, name, no_work)


def test_validate_only_has_no_runtime_calls_or_run_directory(cli, workflow_file, tmp_path, monkeypatch, capsys):
    forbid_runtime(monkeypatch, cli)
    assert cli.main(argv(workflow_file, tmp_path, "--validate-only")) == 0
    assert json.loads(capsys.readouterr().out) == {
        "workflow_id": "demo_workflow", "subgoals": 2, "valid": True,
        "desktop_actions": 0, "model_calls": 0,
    }
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("damage", ["missing", "json", "missing_check", "duplicate_goal"])
def test_invalid_workflow_is_refused_before_runtime_or_output(cli, workflow_file, tmp_path, monkeypatch, capsys, damage):
    if damage == "missing":
        workflow_file.unlink()
    elif damage == "json":
        workflow_file.write_text('{"subgoals":', encoding="utf-8")
    else:
        raw = json.loads(workflow_file.read_text("utf-8-sig"))
        if damage == "missing_check":
            raw["subgoals"][0]["expect_text"] = []
        else:
            raw["subgoals"][1]["case_id"] = raw["subgoals"][0]["case_id"].lower()
        workflow_file.write_text(json.dumps(raw), encoding="utf-8")
    forbid_runtime(monkeypatch, cli)
    assert cli.main(argv(workflow_file, tmp_path)) == 2
    assert "Invalid workflow" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_execute_requires_interactive_confirmation_before_model_or_directory(cli, workflow_file, tmp_path, monkeypatch, capsys):
    forbid_runtime(monkeypatch, cli)
    monkeypatch.setattr(cli, "_has_interactive_stdin", lambda: False)
    assert cli.main(argv(workflow_file, tmp_path, "--execute")) == 2
    assert "interactive terminal" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("extra", [["--yes"], ["--max-actions", "0"], ["--planning-attempts", "-1"],
                                  ["--task-timeout", "0"], ["--ocr-min-confidence", "1.1"],
                                  ["--run-id", "../old_run"]])
def test_cli_has_no_yes_bypass_and_rejects_invalid_limits_or_paths(cli, workflow_file, tmp_path, monkeypatch, extra):
    forbid_runtime(monkeypatch, cli)
    with pytest.raises(SystemExit) as exc:
        cli.main(argv(workflow_file, tmp_path, *extra))
    assert exc.value.code == 2
    assert not (tmp_path / "runs").exists()


@pytest.fixture
def offline(cli, monkeypatch):
    """Keep the real orchestration/records while replacing all work boundaries."""
    state = SimpleNamespace(
        config=Config(), environment_calls=0, config_paths=[], clients=[],
        observers=[], planners=[], executors=[], children=[], runs=[],
        behaviors={}, observed_frames=0, confirms=[], risk_confirms=[], countdowns=[], shown=[],
    )
    state.config.execution.max_actions = 12
    state.config.execution.max_planning_attempts = 6
    state.config.execution.task_timeout_seconds = 90
    state.config.execution.verification_timeout_seconds = 0
    monkeypatch.delenv("GUI_AGENT_MODEL", raising=False)
    monkeypatch.delenv("GUI_AGENT_BASE_URL", raising=False)

    def environment():
        state.environment_calls += 1

    def configuration(path):
        state.config_paths.append(path)
        return state.config.model_copy(deep=True)

    def client(config):
        state.clients.append(config.model_copy(deep=True))
        return SimpleNamespace(complete=no_work, generate_multimodal=no_work)

    class Observer:
        def __init__(self, config, **kwargs):
            self.config = config
            self.kwargs = kwargs
            state.observers.append(self)

        def observe(self):
            state.observed_frames += 1
            return _frame(f"fresh-final-{state.observed_frames}", ("AllDone",))

    def planner(client, **kwargs):
        state.planners.append((client, kwargs))
        return SimpleNamespace(plan=no_work)

    def executor(config):
        state.executors.append(config)
        return SimpleNamespace(execute=no_work)

    class Child:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            state.children.append(self)

        def run(self, task, options, **callbacks):
            state.runs.append((task, options, callbacks))
            plan = TaskPlan(
                task_id=task.case_id, instruction=task.instruction, summary="A synthetic local plan",
                steps=[PlanStep(step_id="wait", description="synthetic step", action_type="wait",
                                arguments={"duration": 0})],
            )
            if "on_plan" in callbacks:
                callbacks["on_plan"](plan)
            behavior = state.behaviors.get(task.case_id, {})
            cancelled = False
            if options.execute:
                cancelled = not callbacks["confirm"](plan)
                if not cancelled and task.risk in {"medium", "high"}:
                    cancelled = not callbacks["high_risk_confirm"](plan)
                if not cancelled:
                    callbacks["countdown"](3)
            status = behavior.get("status", "cancelled" if cancelled else (
                "succeeded" if options.execute else "dry_run_completed"
            ))
            passed = status == "succeeded"
            attempted = status not in {"cancelled", "blocked"}
            outcome = TaskRunResult(
                run_id=self.recorder.run_id, case_id=task.case_id, instruction=task.instruction,
                status=status, execute=options.execute, planning_calls=1,
                steps=[StepRecord(
                    index=1, step_id="wait", description="synthetic validation", action_type="wait",
                    action_result={"success": True, "dry_run": not options.execute},
                )] if attempted else [],
                verification=VerificationResult(outcome="passed" if passed else "inconclusive"),
            )
            self.recorder.write_summary(outcome)
            return outcome

    def confirm(plan):
        state.confirms.append(plan.task_id)
        return state.behaviors.get(plan.task_id, {}).get("confirm", True)

    def high_risk(task, plan):
        state.risk_confirms.append((task.case_id, plan.task_id))
        return state.behaviors.get(task.case_id, {}).get("high_risk_confirm", True)

    monkeypatch.setattr(cli, "load_environment", environment)
    monkeypatch.setattr(cli, "load_config", configuration)
    monkeypatch.setattr(cli, "create_model_client", client)
    monkeypatch.setattr(cli, "ObservationService", Observer)
    monkeypatch.setattr(cli, "TaskPlanner", planner)
    monkeypatch.setattr(cli, "ActionExecutor", executor)
    monkeypatch.setattr(cli, "TaskRunner", Child)
    monkeypatch.setattr(cli, "_has_interactive_stdin", lambda: True)
    monkeypatch.setattr(cli, "_confirm", confirm)
    monkeypatch.setattr(cli, "_confirm_high_risk", high_risk)
    monkeypatch.setattr(cli, "_countdown", lambda seconds: state.countdowns.append(seconds))
    monkeypatch.setattr(cli, "_show_plan", lambda plan: state.shown.append(plan.task_id))
    return state


def read_summary(tmp_path):
    return json.loads((tmp_path / "runs/integration_run/workflow_summary.json").read_text("utf-8"))


def test_existing_directory_is_refused_and_old_evidence_remains_unchanged(cli, offline, workflow_file, tmp_path, capsys):
    directory = tmp_path / "runs/integration_run"
    directory.mkdir(parents=True)
    marker = directory / "old_weight_reference.json"
    marker.write_bytes(b'{"historical":"keep exact bytes"}\n')
    before = marker.read_bytes()
    assert cli.main(argv(workflow_file, tmp_path)) == 2
    assert "Workflow refused" in capsys.readouterr().err
    assert marker.read_bytes() == before and list(directory.iterdir()) == [marker]
    assert offline.children == offline.observers == offline.runs == []


def test_cli_overrides_reach_real_shared_workflow_budgets_and_perception(cli, offline, workflow_file, tmp_path):
    assert cli.main(argv(
        workflow_file, tmp_path, "--config", "chosen.yaml", "--max-actions", "7",
        "--planning-attempts", "5", "--task-timeout", "12.5", "--provider", "mock",
        "--model", "prepared-model", "--base-url", "http://127.0.0.1:9",
        "--ocr-engine", "paddleocr", "--ocr-min-confidence", "0",
    )) == 0
    assert offline.environment_calls == 1 and offline.config_paths == ["chosen.yaml"]
    assert len(offline.clients) == 1
    assert offline.clients[0].model_name == "prepared-model"
    assert offline.clients[0].provider == "mock"
    assert offline.clients[0].base_url == "http://127.0.0.1:9"
    assert [options.max_actions for _, options, _ in offline.runs] == [7, 6]
    assert [options.max_planning_attempts for _, options, _ in offline.runs] == [5, 4]
    assert all(0 < options.task_timeout_seconds <= 12.5 for _, options, _ in offline.runs)
    assert all(not options.execute and options.confirm for _, options, _ in offline.runs)
    assert all(observer.config.perception.ocr.engine == "paddleocr"
               and observer.config.perception.ocr.fallback_engine == "paddleocr"
               and observer.config.perception.ocr.min_confidence == 0 for observer in offline.observers)
    assert all(kwargs["allow_real_execution"] is False for _, kwargs in offline.planners)
    assert all(observer.kwargs["save_images"] is True for observer in offline.observers)
    assert offline.confirms == offline.risk_confirms == offline.countdowns == []


def test_real_workflow_dry_run_never_becomes_success_or_verified_subgoals(cli, offline, workflow_file, tmp_path):
    assert cli.main(argv(workflow_file, tmp_path)) == 0
    result = read_summary(tmp_path)
    assert result["status"] == "dry_run_completed" and result["execute"] is False
    assert result["total_actions"] == result["planning_calls"] == 2
    assert [goal["status"] for goal in result["subgoals"]] == ["dry_run_completed"] * 2
    assert result["final_verification"] is None
    assert offline.observed_frames == 0  # final live verification is not attempted in a dry run
    assert offline.shown == ["G1", "G2"]
    directory = tmp_path / "runs/integration_run"
    checkpoints = [json.loads(line) for line in (directory / "checkpoints.jsonl").read_text("utf-8").splitlines()]
    assert all(event["completed"] == 0 for event in checkpoints)
    observed = read_run_status(directory)
    assert observed["state"] == "finished" and observed["trusted"]
    snapshot = observed["snapshot"]
    assert snapshot["task_status"] == "dry_run_completed" and snapshot["subgoals_completed"] == 0
    assert snapshot["actions_used"] == snapshot["planning_pass"] == 2
    events = [json.loads(line) for line in (directory / "status_events.jsonl").read_text("utf-8").splitlines()]
    assert events[-1] == snapshot
    assert sum(event["event"] == "finished" for event in events) == 1
    assert all(event["task_status"] != "succeeded" for event in events)


def test_execute_callbacks_follow_each_actual_subgoal_and_monitor_final_verification(cli, offline, workflow_file, tmp_path):
    assert cli.main(argv(workflow_file, tmp_path, "--execute")) == 0
    assert offline.confirms == offline.shown == ["G1", "G2"]
    assert offline.risk_confirms == [("G1", "G1"), ("G2", "G2")]
    assert offline.countdowns == [3, 3]
    assert offline.observed_frames == 1  # real WorkflowRunner requires a new final frame
    assert all(options.execute for _, options, _ in offline.runs)
    assert all(kwargs["allow_real_execution"] is True for _, kwargs in offline.planners)
    result = read_summary(tmp_path)
    assert result["status"] == "succeeded" and result["final_verification"]["outcome"] == "passed"
    assert all(goal["status"] == "completed" for goal in result["subgoals"])
    final = read_run_status(tmp_path / "runs/integration_run")
    assert final["state"] == "finished" and final["snapshot"]["task_status"] == "succeeded"
    assert final["snapshot"]["subgoals_completed"] == 2


@pytest.mark.parametrize("gate", ["confirm", "high_risk_confirm"])
def test_declined_confirmation_stops_before_next_subgoal_and_records_cancelled(cli, offline, workflow_file, tmp_path, gate):
    offline.behaviors["G1"] = {gate: False}
    assert cli.main(argv(workflow_file, tmp_path, "--execute")) == 1
    assert len(offline.runs) == 1 and offline.countdowns == []
    result = read_summary(tmp_path)
    assert result["status"] == "cancelled" and result["total_actions"] == 0
    assert result["subgoals"][1]["status"] == "not_started"
    final = read_run_status(tmp_path / "runs/integration_run")
    assert final["state"] == "finished" and final["snapshot"]["task_status"] == "cancelled"
    assert final["snapshot"]["subgoals_completed"] == final["snapshot"]["actions_used"] == 0


def test_failed_child_has_nonzero_exit_and_terminal_monitor(cli, offline, workflow_file, tmp_path):
    offline.behaviors["G1"] = {"status": "failed"}
    assert cli.main(argv(workflow_file, tmp_path)) == 1
    assert len(offline.runs) == 1
    assert read_summary(tmp_path)["status"] == "failed"
    final = read_run_status(tmp_path / "runs/integration_run")
    assert final["state"] == "finished" and final["snapshot"]["task_status"] == "failed"


def test_unknown_child_dispatch_is_visible_in_root_terminal_monitor(
    cli, offline, workflow_file, tmp_path, monkeypatch,
):
    import week6_status

    original_child = cli.TaskRunner

    class UnknownChild(original_child):
        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            result.status = "failed"
            result.steps = []
            result.unconfirmed_action_attempts = 1
            self.recorder.write_summary(result)
            return result

    monkeypatch.setattr(cli, "TaskRunner", UnknownChild)
    assert cli.main(argv(workflow_file, tmp_path, "--execute")) == 1
    assert len(offline.runs) == 1
    summary = read_summary(tmp_path)
    assert summary["total_actions"] == 0 and summary["unconfirmed_action_attempts"] == 1
    assert summary["accounting_complete"] is False
    assert summary["subgoals"][1]["status"] == "not_started"
    final = read_run_status(tmp_path / "runs/integration_run")
    assert final["state"] == "finished" and final["snapshot"]["task_status"] == "failed"
    assert final["snapshot"]["actions_used"] == 1
    assert final["snapshot"]["detail_code"] == "unconfirmed_dispatch"
    assert "detail=unconfirmed_dispatch" in week6_status._human(final)


def test_configuration_error_is_refused_before_model_and_run(cli, workflow_file, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_environment", lambda: None)

    def invalid(_path):
        raise OSError("prepared missing configuration")

    monkeypatch.setattr(cli, "load_config", invalid)
    monkeypatch.setattr(cli, "create_model_client", no_work)
    monkeypatch.setattr(cli, "WorkflowRunner", no_work)
    assert cli.main(argv(workflow_file, tmp_path)) == 2
    assert "Configuration failed" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_infinite_whole_workflow_budget_is_refused_without_child_or_run(cli, offline, workflow_file, tmp_path, capsys):
    assert cli.main(argv(workflow_file, tmp_path, "--task-timeout", "inf")) == 2
    assert "Workflow refused" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists() and offline.runs == []
