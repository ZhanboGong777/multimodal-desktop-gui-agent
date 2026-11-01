"""Real model/OCR/runtime acceptance with a window-bound Sky input bridge.

This harness never sends OS input itself. Every requested, adapter-resolved click
waits for a separately observed and approved Sky action receipt. The lab's hidden
state is read only after the run, for independent acceptance, never for planning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.models.openai_compatible import OpenAICompatibleClient
from gui_agent.planning import TaskPlanner
from gui_agent.planning.workflow_decomposer import AutomaticWorkflowRunner, WorkflowRequest
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.observation import ObservationService
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import ExecutionOptions
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class BoundObserver(ObservationService):
    """Refuse other foreground windows; the fixture covers the primary display."""
    activation_directory: Path | None = None

    def observe(self, **kwargs):
        if self.activation_directory is not None:
            # Sky brings only the already selected, dedicated test window to
            # the foreground. This is harness orchestration, not a model input.
            token = f"capture_{time.monotonic_ns()}"
            request = self.activation_directory / f"{token}_request.json"
            write(request, {"id": token, "required_window_title": "Week6 Local Lab"})
            receipt = self.activation_directory / f"{token}_receipt.json"
            deadline = time.monotonic() + 90
            while not receipt.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("dedicated-window activation receipt missing")
                time.sleep(.05)
            if not json.loads(receipt.read_text(encoding="utf-8"))["success"]:
                raise RuntimeError("dedicated-window activation refused")
        snapshot = super().observe(**kwargs)
        if snapshot.window_title != "Week6 Local Lab" or not snapshot.foreground_stable:
            raise RuntimeError("dedicated fixture is not the stable foreground window")
        return snapshot


class SkyBridgeExecutor:
    def __init__(self, directory, deadline):
        self.directory = directory
        self.deadline = deadline
        self.index = 0

    def update_screen(self, screen):
        self.screen = screen

    def execute(self, action, *, dry_run=False, screen=None):
        start = datetime.now(UTC)
        if dry_run:
            return ActionResult(action=action, success=True, dry_run=True,
                                started_at=start, finished_at=datetime.now(UTC))
        # The lab accepts single or double left clicks. Any request for typing,
        # shortcuts, other windows or other input is a recorded refusal.
        if action.action_type not in {"click", "double_click"}:
            return ActionResult(action=action, success=False, dry_run=False,
                                started_at=start, finished_at=datetime.now(UTC),
                                error="fixture bridge permits only grounded single/double left clicks")
        self.index += 1
        token = f"input_{self.index:03d}"
        active_screen = screen or self.screen
        write(self.directory / f"{token}_request.json", {
            "id": token, "action": action.model_dump(mode="json"),
            "screen": active_screen.model_dump(mode="json"),
            "required_window_title": "Week6 Local Lab", "created_utc": start.isoformat(),
            "deadline_monotonic": self.deadline,
        })
        receipt = self.directory / f"{token}_receipt.json"
        deadline = min(time.monotonic() + 180, self.deadline)
        while not receipt.exists():
            if time.monotonic() >= deadline:
                # Unknown dispatch never becomes a fabricated failed/no-op input.
                raise RuntimeError("Sky input receipt missing; dispatch outcome unknown")
            time.sleep(.15)
        value = json.loads(receipt.read_text(encoding="utf-8"))
        if value["id"] != token or value["action"] != action.model_dump(mode="json"):
            raise RuntimeError("Sky input receipt mismatch; dispatch outcome unknown")
        return ActionResult(action=action, success=value["success"], dry_run=False,
                            started_at=start, finished_at=datetime.now(UTC),
                            error=value.get("error"), metadata={"bridge": "window-bound Sky", "receipt": token})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixture-directory", type=Path, required=True)
    parser.add_argument("--project", choices=("ALPHA", "BETA"), required=True)
    parser.add_argument("--colour", choices=("BLUE", "GREEN"), required=True)
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--activation-bridge", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    bridge = args.directory / "sky_bridge"
    bridge.mkdir()
    activation = args.directory / "sky_activation"
    if args.activation_bridge:
        activation.mkdir()
    source_paths = [p for p in (ROOT / "src").rglob("*.py")]
    source_paths += [Path(__file__), ROOT / "scripts/week6_local_lab.py"]
    source_pins = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in source_paths}
    for p in source_paths:
        destination = args.directory / "source_snapshot" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination.with_suffix(".py.txt"))
    config = load_config(ROOT / "configs/week4.yaml")
    config.perception.ocr.engine = config.perception.ocr.fallback_engine = "tesseract"
    config.perception.ocr.languages = ["en"]
    config.perception.ocr.min_confidence = .45
    client = OpenAICompatibleClient(model_name=args.model, base_url="http://127.0.0.1:11434/v1",
                                   api_key="ollama-local", timeout_seconds=180, max_retries=0,
                                   temperature=0, max_tokens=2200)
    options = ExecutionOptions(execute=True, confirm=False, max_actions=12,
                               max_planning_attempts=12, task_timeout_seconds=1200,
                               verification_timeout_seconds=.8,
                               verification_poll_interval_seconds=.2)
    request = WorkflowRequest(
        workflow_id="lab_auto",
        instruction=f"In Week6 Local Lab, prepare a {args.colour} report for project {args.project}, "
                    "review it and commit it locally. Use the visible controls. This is an offline test, "
                    "with no external communication. If review is temporarily unavailable, retry within budget.",
        expect_text=[f"Project {args.project}", f"Theme {args.colour}", "LOCAL COMPLETE"],
    )
    write(args.directory / "contract.json", {
        "request": request.model_dump(mode="json"), "options": options.model_dump(mode="json"),
        "max_model_requests": 24, "independent_pass": {
            "project": args.project, "colour": args.colour, "stage": "completed", "committed": True,
            "event_order": ["project_selected", "theme_selected", "review_accepted", "local_commit"],
            "project_and_theme_selected_exactly_once": True,
        },
        "input_transport": "Sky bridge; production adapter coordinates unchanged; no PyAutoGUI input",
        "automatic_protocol": "rolling_actions",
        "intermediate_checks": "Model-generated one-step action plans are execution proposals, not verified semantic subgoals or stage-departure proofs. Fresh unchanged caller final rules establish overall success.",
        "model": args.model,
        "baseline": "prior explicit-subgoal workflow cannot accept this overall instruction",
        "source_sha256": source_pins,
    })
    executor = SkyBridgeExecutor(bridge, time.monotonic() + options.task_timeout_seconds)
    monitors = []

    def observer(directory):
        instance = BoundObserver(config, output_directory=directory, save_images=True)
        instance.activation_directory = activation if args.activation_bridge else None
        return instance

    def factory(task, index, directory, budgeted_client):
        # The automatic runner constrains each real model plan to one step.
        # Caller-owned final checks remain unchanged across bounded recovery.
        return TaskRunner(observer=observer(directory),
                          planner=TaskPlanner(budgeted_client, max_steps=4, allow_real_execution=True),
                          adapter=ActionAdapter(), executor=executor,
                          absolute_deadline=budgeted_client.deadline,
                          verifier=Verifier(poll_interval_seconds=.2),
                          recorder=TaskRecorder(RunSession(session_id=directory.name, directory=directory)))

    workflow_dir = args.directory / "automatic"
    def checkpoint(event):
        # The production automatic runner owns the cumulative monitor; each
        # workflow round's callback counters are intentionally local to it.
        monitors.append(read_run_status(workflow_dir))

    result = AutomaticWorkflowRunner(workflow_dir, client, factory, max_model_requests=24,
                                     initial_observer=lambda d: observer(d).observe(),
                                     protocol_mode="rolling_actions").run(
        request, options, on_checkpoint=checkpoint,
    )
    monitors.append(read_run_status(workflow_dir))
    state = json.loads((args.fixture_directory / "independent_state.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (args.fixture_directory / "independent_events.jsonl").read_text().splitlines()]
    expected_order = ["project_selected", "theme_selected", "review_accepted", "local_commit"]
    accepted_events = [event["event"] for event in events if event["event"] in expected_order]
    independent_pass = (state["project"] == args.project and state["colour"] == args.colour
                        and state["stage"] == "completed" and state["committed"]
                        and accepted_events == expected_order)
    receipts = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(bridge.glob("*_receipt.json"))]
    attributed = []
    for event in events:
        if event["event"] not in {*expected_order, "review_rejected"}:
            continue
        event_time = datetime.fromisoformat(event["utc"])
        owners = [receipt["id"] for receipt in receipts if receipt.get("success") and
                  datetime.fromisoformat(receipt["started_utc"]) <= event_time <=
                  datetime.fromisoformat(receipt["finished_utc"])]
        attributed.append({"event": event["event"], "utc": event["utc"], "input_receipts": owners})
    # Each semantic transition must come from one distinct model-selected input.
    # A cross-page double click or external input cannot receive completion credit.
    transition_owners = [item["input_receipts"][0] for item in attributed if len(item["input_receipts"]) == 1]
    attribution_pass = (len(attributed) >= 4 and len(transition_owners) == len(attributed)
                        and len(set(transition_owners)) == len(transition_owners))
    write(args.directory / "acceptance.json", {
        "status": result.status, "runtime_pass": result.status == "succeeded",
        "independent_pass": independent_pass,
        "passed": result.status == "succeeded" and independent_pass and attribution_pass,
        "attribution_pass": attribution_pass, "transition_attribution": attributed,
        "input_receipts": receipts,
        "state": state, "independent_events": events,
        "automatic_result": result.model_dump(mode="json"),
        "source_unchanged": all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
                                for path, digest in source_pins.items()),
        "status_reads": [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in monitors],
    })
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
