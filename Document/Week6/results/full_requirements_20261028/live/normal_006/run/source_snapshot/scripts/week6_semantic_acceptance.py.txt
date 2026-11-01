"""Fresh semantic-subtask acceptance in the isolated offline Week6 window.

The model sees only its request, screenshot and production OCR elements. The lab
journal is consulted after execution, solely for independent outcome/attribution.
OS inputs remain individually mediated by the window-bound Computer Use bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from week6_lab_acceptance import BoundObserver, SkyBridgeExecutor, write

from gui_agent.config import load_config
from gui_agent.models.openai_compatible import OpenAICompatibleClient
from gui_agent.planning import TaskPlanner
from gui_agent.planning.workflow_decomposer import AutomaticWorkflowRunner, WorkflowRequest
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import ExecutionOptions
from gui_agent.runtime.verification import Verifier


def source_paths(config: Path) -> list[Path]:
    return sorted({
        *(ROOT / "src").rglob("*.py"), config.resolve(), Path(__file__).resolve(),
        ROOT / "scripts/week6_lab_acceptance.py", ROOT / "scripts/week6_local_lab.py",
        ROOT / "scripts/week6_agent_cli.py", ROOT / "pyproject.toml",
    })


def independent_attribution(events: list[dict], receipts: list[dict]) -> tuple[list[dict], bool]:
    transitions = {"project_selected", "theme_selected", "review_rejected", "review_accepted", "local_commit"}
    attributed = []
    for event in events:
        if event["event"] not in transitions:
            continue
        event_time = datetime.fromisoformat(event["utc"])
        owners = [receipt["id"] for receipt in receipts if receipt.get("success") and
                  datetime.fromisoformat(receipt["started_utc"]) <= event_time <=
                  datetime.fromisoformat(receipt["finished_utc"])]
        attributed.append({"event": event["event"], "utc": event["utc"], "input_receipts": owners})
    owners = [item["input_receipts"][0] for item in attributed if len(item["input_receipts"]) == 1]
    return attributed, bool(len(attributed) >= 4 and len(owners) == len(attributed)
                            and len(set(owners)) == len(owners))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixture-directory", type=Path, required=True)
    parser.add_argument("--project", choices=("ALPHA", "BETA"), required=True)
    parser.add_argument("--colour", choices=("BLUE", "GREEN"), required=True)
    parser.add_argument("--mode", choices=("normal", "transient"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--max-model-requests", type=int, default=32)
    parser.add_argument("--task-timeout", type=float, default=7200)
    parser.add_argument("--model-timeout", type=float, default=900)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    bridge, activation = args.directory / "sky_bridge", args.directory / "sky_activation"
    bridge.mkdir()
    activation.mkdir()
    paths = source_paths(args.config)
    pins = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        destination = args.directory / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination.with_suffix(destination.suffix + ".txt"))
    config = load_config(args.config)
    options = ExecutionOptions(execute=True, confirm=False, max_actions=20,
                               max_planning_attempts=24, task_timeout_seconds=args.task_timeout,
                               verification_timeout_seconds=.8,
                               verification_poll_interval_seconds=.2)
    request = WorkflowRequest(
        workflow_id="semantic_lab",
        instruction=f"In Week6 Local Lab, prepare a {args.colour} report for project {args.project}, "
                    "review it and commit it locally. Use the visible controls. This is an offline test "
                    "with no external communication. If review is temporarily unavailable, retry within budget.",
        expect_text=[f"Project {args.project}", f"Theme {args.colour}", "LOCAL COMPLETE"],
    )
    client = OpenAICompatibleClient(model_name=args.model, base_url="http://127.0.0.1:11434/v1",
                                   api_key="ollama-local", timeout_seconds=args.model_timeout, max_retries=0,
                                   temperature=0, max_tokens=4096)
    contract = {
        "schema": "week6.semantic_live.contract.v1", "request": request.model_dump(mode="json"),
        "options": options.model_dump(mode="json"), "model": args.model,
        "max_model_requests": args.max_model_requests, "automatic_protocol": "semantic_subtasks",
        "model_transport": {"base_url": "http://127.0.0.1:11434/v1", "temperature": 0,
                            "max_tokens": 4096, "max_retries": 0,
                            "timeout_seconds": args.model_timeout},
        "fixture_mode": args.mode, "configuration": config.model_dump(mode="json"),
        "config_path": args.config.resolve().relative_to(ROOT).as_posix(),
        "input_transport": "Individually observed window-bound Sky inputs; unchanged production adapter coordinates",
        "source_sha256": pins,
        "independent_expected": {"project": args.project, "colour": args.colour,
                                 "stage": "completed", "committed": True,
                                 "event_order": ["project_selected", "theme_selected", "review_accepted", "local_commit"],
                                 "review_rejections": int(args.mode == "transient")},
        "semantic_gate": "Native model-generated distinct subgoals and pre-input postconditions must be audited separately; raw final success alone is insufficient.",
    }
    write(args.directory / "contract.json", contract)
    executor = SkyBridgeExecutor(bridge, time.monotonic() + options.task_timeout_seconds)
    monitors = []

    def observer(directory):
        instance = BoundObserver(config, output_directory=directory, save_images=True)
        instance.activation_directory = activation
        return instance

    def factory(task, index, directory, budgeted_client):
        return TaskRunner(observer=observer(directory),
                          planner=TaskPlanner(budgeted_client, max_steps=4, allow_real_execution=True),
                          adapter=ActionAdapter(), executor=executor,
                          absolute_deadline=budgeted_client.deadline,
                          verifier=Verifier(poll_interval_seconds=.2),
                          recorder=TaskRecorder(RunSession(session_id=directory.name, directory=directory)))

    automatic = args.directory / "automatic"
    result = AutomaticWorkflowRunner(
        automatic, client, factory, max_model_requests=args.max_model_requests,
        initial_observer=lambda d: observer(d).observe(), protocol_mode="semantic_subtasks",
    ).run(request, options, on_checkpoint=lambda event: monitors.append(read_run_status(automatic)))
    monitors.append(read_run_status(automatic))
    write(args.directory / "runtime_result.json", result.model_dump(mode="json"))
    # This is intentionally the first fixture-journal read in this process.
    state = json.loads((args.fixture_directory / "independent_state.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in
              (args.fixture_directory / "independent_events.jsonl").read_text(encoding="utf-8").splitlines()]
    order = ["project_selected", "theme_selected", "review_accepted", "local_commit"]
    selected = [event["event"] for event in events if event["event"] in order]
    independent = bool(state["project"] == args.project and state["colour"] == args.colour
                       and state["stage"] == "completed" and state["committed"]
                       and selected == order
                       and sum(event["event"] == "review_rejected" for event in events)
                       == int(args.mode == "transient"))
    receipts = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(bridge.glob("*_receipt.json"))]
    attributed, attribution_pass = independent_attribution(events, receipts)
    source_unchanged = all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
                           for path, digest in pins.items())
    known_gate = bool(result.status == "succeeded" and independent and attribution_pass and source_unchanged
                      and result.accounting_complete and not result.unconfirmed_action_attempts)
    acceptance = {
        "schema": "week6.semantic_live.observed_result.v1", "status": result.status,
        "runtime_pass": result.status == "succeeded", "independent_pass": independent,
        "attribution_pass": attribution_pass, "source_unchanged": source_unchanged,
        "runtime_independent_gates_passed": known_gate,
        "semantic_audit_pending": True,
        "meaning": "Observed evidence only. A separate read-only native-plan/semantic-goal audit and real owner exit receipt are required for final acceptance.",
        "transition_attribution": attributed, "input_receipts": receipts,
        "state": state, "independent_events": events,
        "automatic_result": result.model_dump(mode="json"),
        "status_reads": [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in monitors],
    }
    write(args.directory / "acceptance_observed.json", acceptance)
    print(json.dumps({"directory": str(args.directory), "status": result.status,
                      "runtime_independent_gates_passed": known_gate, "semantic_audit_pending": True}))
    return 0 if known_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
