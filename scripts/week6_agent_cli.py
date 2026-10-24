"""Run a checked sequence of subgoals with one shared budget (dry run by default)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from week4_agent_cli import (
    _apply_environment,
    _confidence,
    _confirm,
    _confirm_high_risk,
    _countdown,
    _has_interactive_stdin,
    _positive_float,
    _positive_int,
    _run_id,
    _show_plan,
    load_environment,
)

from gui_agent.config import load_config
from gui_agent.control.executor import ActionExecutor
from gui_agent.models import CLIENTS, create_model_client
from gui_agent.planning import TaskPlanner
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.monitoring import RunMonitor
from gui_agent.runtime.observation import ObservationService
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import ExecutionOptions
from gui_agent.runtime.verification import Verifier
from gui_agent.runtime.workflow import WorkflowRunner, WorkflowSpec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true", help="validate the spec without model, capture or input")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/week4.yaml"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--provider", choices=sorted(CLIENTS))
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-actions", type=_positive_int)
    parser.add_argument("--planning-attempts", type=_positive_int)
    parser.add_argument("--task-timeout", type=_positive_float)
    parser.add_argument("--ocr-engine", choices=("tesseract", "paddleocr"))
    parser.add_argument("--ocr-min-confidence", type=_confidence)
    parser.add_argument("--output-directory", type=Path, default=REPO_ROOT / "outputs/week6")
    parser.add_argument("--run-id", type=_run_id)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        spec = WorkflowSpec.model_validate_json(args.workflow.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        print(f"Invalid workflow: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(json.dumps({"workflow_id": spec.workflow_id, "subgoals": len(spec.subgoals),
                          "valid": True, "desktop_actions": 0, "model_calls": 0}))
        return 0
    if args.execute and not _has_interactive_stdin():
        print("--execute requires an interactive terminal for plan confirmations", file=sys.stderr)
        return 2
    load_environment()
    try:
        config = load_config(args.config)
        _apply_environment(config, args)
        if args.ocr_engine:
            config.perception.ocr.engine = config.perception.ocr.fallback_engine = args.ocr_engine
        if args.ocr_min_confidence is not None:
            config.perception.ocr.min_confidence = args.ocr_min_confidence
        options = ExecutionOptions(
            execute=args.execute, confirm=True,
            max_actions=args.max_actions or config.execution.max_actions,
            max_planning_attempts=args.planning_attempts or config.execution.max_planning_attempts,
            task_timeout_seconds=args.task_timeout or config.execution.task_timeout_seconds,
            verification_timeout_seconds=config.execution.verification_timeout_seconds,
            verification_poll_interval_seconds=config.execution.verification_poll_interval_seconds,
        )
        client = create_model_client(config.model)
    except (OSError, ValueError) as exc:
        print(f"Configuration failed: {exc}", file=sys.stderr)
        return 2
    from datetime import UTC, datetime

    run_id = args.run_id or f"{spec.workflow_id}_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}"
    directory = args.output_directory / run_id
    active = {"task": spec.subgoals[0], "monitor": None}

    def factory(task, index, childdir):
        active["task"] = task
        session = RunSession(session_id=childdir.name, directory=childdir)
        recorder = TaskRecorder(session)
        recorder.save_config(config.model_dump(mode="json"))
        return TaskRunner(
            observer=ObservationService(config, max_elements=config.execution.max_elements,
                                        output_directory=childdir, save_images=True),
            planner=TaskPlanner(client, max_steps=config.planning.max_steps,
                                require_structured_output=config.planning.require_structured_output,
                                allow_real_execution=args.execute),
            adapter=ActionAdapter(max_wait_seconds=config.execution.max_wait_seconds),
            executor=ActionExecutor(config.control),
            verifier=Verifier(poll_interval_seconds=config.execution.verification_poll_interval_seconds),
            recorder=recorder,
        )

    def checkpoint(event):
        if active["monitor"] is None:
            active["monitor"] = RunMonitor(directory)
            active["monitor"].start(
                case_id=spec.workflow_id, execute=options.execute, max_actions=options.max_actions,
                max_planning_attempts=options.max_planning_attempts, subgoals_total=len(spec.subgoals),
            )
        active["monitor"].update(
            "decomposing" if event["status"] == "running" else "subgoal_verification",
            subgoal_id=event["case_id"], subgoals_completed=event["completed"],
            actions_used=event["actions"], planning_pass=event["planning_calls"],
        )
        print(f"{event['case_id']}: {event['status']} | verified {event['completed']}/{event['total']} | actions {event['actions']}/{options.max_actions}", flush=True)

    callbacks = {"on_checkpoint": checkpoint, "on_plan": _show_plan}
    if args.execute:
        callbacks.update(confirm=_confirm, countdown=_countdown,
                         high_risk_confirm=lambda plan: _confirm_high_risk(active["task"], plan))
    print(f"Records: {directory}\nMode: {'execute' if args.execute else 'dry run'}", flush=True)
    try:
        result = WorkflowRunner(directory, factory).run(spec, options, **callbacks)
    except (OSError, ValueError) as exc:
        print(f"Workflow refused: {exc}", file=sys.stderr)
        return 2
    if active["monitor"] is not None:
        active["monitor"].finish(
            result.status,
            actions_used=result.total_actions + result.unconfirmed_action_attempts,
            detail_code="unconfirmed_dispatch" if result.unconfirmed_action_attempts else "",
        )
    print(result.model_dump_json(indent=2))
    return 0 if result.status in {"succeeded", "dry_run_completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
