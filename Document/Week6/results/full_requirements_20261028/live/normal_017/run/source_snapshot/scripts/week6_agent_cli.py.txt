"""Decompose an overall request into verified semantic subtasks, or run an explicit workflow."""
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
from gui_agent.planning.workflow_decomposer import AutomaticWorkflowRunner, WorkflowRequest
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--workflow", type=Path, help="use an existing explicitly checked workflow")
    mode.add_argument("--instruction", help="plan and execute this overall natural-language task under shared limits")
    parser.add_argument("--automatic-protocol", choices=("rolling_actions", "observed_departure", "semantic_subtasks"),
                        default="semantic_subtasks",
                        help="automatic mode: model-generated semantic subtasks (default), or legacy rolling/departure checks")
    parser.add_argument("--workflow-id", type=_run_id, default="automatic_workflow")
    parser.add_argument("--expect-text", action="append", default=[], help="caller-owned final required text")
    parser.add_argument("--forbid-text", action="append", default=[], help="caller-owned final forbidden text")
    parser.add_argument("--model-requests", type=_positive_int, default=24,
                        help="shared HTTP request cap across all automatic planning")
    parser.add_argument("--max-subgoals", type=_positive_int, default=8,
                        help="semantic/departure proposal limit; rolling_actions plans one next action")
    parser.add_argument("--decomposition-rounds", type=_positive_int, default=4,
                        help="bounded semantic revisions or fresh-screen departure rounds, 1..8")
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
    request = None
    spec = None
    try:
        if args.workflow:
            if args.expect_text or args.forbid_text:
                raise ValueError("--expect-text/--forbid-text require --instruction; workflow final checks are unchanged")
            spec = WorkflowSpec.model_validate_json(args.workflow.read_text(encoding="utf-8-sig"))
        else:
            if args.max_subgoals > 32:
                raise ValueError("--max-subgoals must be at most 32")
            if args.automatic_protocol == "semantic_subtasks" and args.max_subgoals < 2:
                raise ValueError("semantic_subtasks requires at least two initial semantic objectives")
            if args.decomposition_rounds > 8:
                raise ValueError("--decomposition-rounds must be at most 8")
            if (args.automatic_protocol in {"observed_departure", "semantic_subtasks"}
                    and args.planning_attempts is not None and args.planning_attempts < 2):
                raise ValueError("the selected protocol needs at least two planning calls including decomposition")
            request = WorkflowRequest(workflow_id=args.workflow_id, instruction=args.instruction,
                                      expect_text=args.expect_text, forbid_text=args.forbid_text)
    except (OSError, ValueError) as exc:
        print(f"Invalid workflow: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(json.dumps({"workflow_id": spec.workflow_id if spec else request.workflow_id,
                          "subgoals": len(spec.subgoals) if spec else None,
                          "valid": True, "desktop_actions": 0, "model_calls": 0,
                          **({"mode": "automatic", "decomposition_pending": True} if request else {})}))
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
        if request and args.automatic_protocol in {"observed_departure", "semantic_subtasks"} and options.max_planning_attempts < 2:
            raise ValueError("the selected protocol needs at least two planning calls including decomposition")
        client = create_model_client(config.model)
    except (OSError, ValueError) as exc:
        print(f"Configuration failed: {exc}", file=sys.stderr)
        return 2
    from datetime import UTC, datetime

    workflow_id = spec.workflow_id if spec else request.workflow_id
    run_id = args.run_id or f"{workflow_id}_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}"
    directory = args.output_directory / run_id
    active = {"task": spec.subgoals[0] if spec else None, "monitor": None}

    def factory(task, index, childdir, budgeted_client=None):
        active["task"] = task
        session = RunSession(session_id=childdir.name, directory=childdir)
        recorder = TaskRecorder(session)
        recorder.save_config(config.model_dump(mode="json"))
        return TaskRunner(
            observer=ObservationService(config, max_elements=config.execution.max_elements,
                                        output_directory=childdir, save_images=True),
            planner=TaskPlanner(budgeted_client or client, max_steps=config.planning.max_steps,
                                require_structured_output=config.planning.require_structured_output,
                                allow_real_execution=args.execute),
            adapter=ActionAdapter(max_wait_seconds=config.execution.max_wait_seconds),
            executor=ActionExecutor(config.control),
            verifier=Verifier(poll_interval_seconds=config.execution.verification_poll_interval_seconds),
            recorder=recorder,
            absolute_deadline=getattr(budgeted_client, "deadline", None),
        )

    def initial_observer(observation_directory):
        return ObservationService(
            config, max_elements=config.execution.max_elements,
            output_directory=observation_directory, save_images=True,
        ).observe()

    def ensure_monitor(total):
        if active["monitor"] is None:
            active["monitor"] = RunMonitor(directory)
            active["monitor"].start(
                case_id=workflow_id, execute=options.execute, max_actions=options.max_actions,
                max_planning_attempts=options.max_planning_attempts, subgoals_total=total,
            )

    def decomposition_started(event):
        # The goal count is unknown until the model returns. Keep decomposition
        # status in its own evidence; the fixed-count run monitor starts once the
        # WorkflowRunner supplies the actual count, not an invented denominator.
        print("Decomposing the overall instruction; caller final checks remain fixed", flush=True)

    def checkpoint(event):
        if request is None:  # AutomaticWorkflowRunner owns its overall parent monitor.
            ensure_monitor(event["total"])
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
        if request:
            result = AutomaticWorkflowRunner(
                directory, client, factory, max_model_requests=args.model_requests,
                max_subgoals=args.max_subgoals, initial_observer=initial_observer,
                max_decomposition_rounds=args.decomposition_rounds,
                protocol_mode=args.automatic_protocol,
            ).run(request, options, on_decomposition=decomposition_started, **callbacks)
        else:
            result = WorkflowRunner(directory, factory).run(spec, options, **callbacks)
    except (OSError, ValueError) as exc:
        print(f"Workflow refused: {exc}", file=sys.stderr)
        return 2
    if active["monitor"] is not None:
        outcome = result.workflow_result if request else result
        active["monitor"].finish(
            result.status,
            actions_used=(outcome.total_actions + outcome.unconfirmed_action_attempts) if outcome else 0,
            detail_code="unconfirmed_dispatch" if outcome and outcome.unconfirmed_action_attempts else "",
        )
    print(result.model_dump_json(indent=2))
    return 0 if result.status in {"succeeded", "dry_run_completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
