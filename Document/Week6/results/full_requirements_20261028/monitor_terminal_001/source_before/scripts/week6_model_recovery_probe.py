"""Real local model planning on saved synthetic GUI images; dispatches no input.

This is a planning/grounding probe, not a real desktop task success benchmark.
Each predeclared condition invokes TaskPlanner exactly once. Its one existing
format retry is recorded; no condition is tuned or repeated to select successes.
The endpoint and model are fixed, and existing output directories are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.models.base import ModelError
from gui_agent.models.openai_compatible import OpenAICompatibleClient
from gui_agent.planning.parser import PlanParseError, extract_json_object, parse_plan
from gui_agent.planning.planner import TaskPlanner
from gui_agent.planning.prompts import SYSTEM_PROMPT
from gui_agent.runtime.action_adapter import ActionAdapter, ActionResolutionError, keys_for_platform
from gui_agent.runtime.observation import describe_elements
from gui_agent.runtime.recovery import build_recovery_feedback
from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot, StepRecord
from gui_agent.schemas import BoundingBox, ScreenInfo

MODEL = "qwen2.5vl:3b"
BASE_URL = "http://127.0.0.1:11434/v1"
CONDITION_TIMEOUT_SECONDS = 120.0
SOURCE_PATHS = (
    "scripts/week6_model_recovery_probe.py", "src/gui_agent/models/base.py",
    "src/gui_agent/models/openai_compatible.py", "src/gui_agent/planning/planner.py",
    "src/gui_agent/planning/prompts.py", "src/gui_agent/planning/parser.py",
    "src/gui_agent/planning/schemas.py", "src/gui_agent/runtime/action_adapter.py",
    "src/gui_agent/runtime/recovery.py", "src/gui_agent/runtime/schemas.py",
    "src/gui_agent/runtime/observation.py",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_get(path: str) -> Any:
    with urlopen(f"http://127.0.0.1:11434{path}", timeout=10) as response:
        return json.load(response)


def font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"),
                      Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    raise FileNotFoundError("a readable Arial or DejaVuSans font is required")


def make_snapshot(directory: Path, number: int, buttons: list[tuple[str, tuple[int, ...]]]) -> ObservationSnapshot:
    image = Image.new("RGB", (800, 520), "#eef2f6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 770, 490), fill="white", outline="#334155", width=2)
    draw.rectangle((30, 30, 770, 92), fill="#17324d")
    draw.text((52, 48), "LOCAL ANALYSIS DEMO", font=font(24), fill="white")
    draw.text((70, 125), "Choose an action", font=font(28), fill="#162539")
    draw.text((70, 178), "Status: Ready", font=font(23), fill="#334155")
    refs = []
    oid = f"obs-{number:04d}"
    for index, (label, bounds) in enumerate(buttons):
        left, top, right, bottom = bounds
        draw.rounded_rectangle(bounds, radius=8, fill="#1e5d91", outline="#153d5e", width=2)
        draw.text(((left + right) / 2, (top + bottom) / 2), label,
                  font=font(23), fill="white", anchor="mm")
        box = BoundingBox(left=left, top=top, right=right, bottom=bottom)
        refs.append(ElementRef(element_id=f"{oid}-e{index:03d}", text=label,
                               bounding_box=box, center=box.center, confidence=1.0, source="manual"))
    image_path = directory / "frames" / f"f{number:03d}.png"
    image.save(image_path)
    return ObservationSnapshot(
        observation_id=oid, captured_at=datetime.now(UTC), image_path=str(image_path.resolve()),
        screen_info=ScreenInfo(screenshot_width=800, screenshot_height=520,
                               control_width=800, control_height=520),
        elements=refs, ocr_engine="synthetic_ground_truth_no_ocr",
        window_title="Local analysis demo", window_class="SyntheticFixture",
        window_id="synthetic-local-analysis", window_bounds=BoundingBox(
            left=30, top=30, right=770, bottom=490,
        ), foreground_stable=True,
    )


def conditions(directory: Path) -> list[dict[str, Any]]:
    snapshots = [
        make_snapshot(directory, 1, [("Run analysis", (90, 280, 350, 356)),
                                     ("Cancel", (465, 280, 725, 356))]),
        make_snapshot(directory, 2, [("Cancel", (90, 280, 350, 356)),
                                     ("Run analysis", (465, 380, 725, 456))]),
        make_snapshot(directory, 3, [("Run analysis", (90, 280, 350, 356)),
                                     ("Run analysis", (465, 280, 725, 356))]),
    ]
    instruction = "Click the Run analysis button in this local demo, then finish. Do not type or use keyboard shortcuts."
    records = []
    for index, snapshot in enumerate(snapshots):
        context = {
            "platform": "win32", "observation_id": snapshot.observation_id,
            "screen": "800x520", "target_app": "Local analysis demo",
            "allowed_keys": keys_for_platform("win32"),
            "limits": "at most 2 actions and 120 s for this planning probe",
            "visible_text": describe_elements(snapshot),
            "success_rules": ["the uniquely identified Run analysis button is activated"],
        }
        if index == 1:
            context["recovery"] = build_recovery_feedback(
                [StepRecord(index=0, step_id="previous-click", description="previous target lookup",
                            action_type="click", error="no element matches a previous target")],
                previous_pass=1, verification=None, actions_used=0,
                max_actions=2, remaining_ms=CONDITION_TIMEOUT_SECONDS * 1000,
            ).model_dump(mode="json")
        expected = snapshot.elements[0 if index == 0 else 1] if index < 2 else None
        records.append({
            "condition": ("initial_unique", "recovery_relayout", "ambiguous_same_labels")[index],
            "instruction": instruction, "context": context, "snapshot": snapshot,
            "ground_truth": {
                "expected_behavior": "one_current_click" if expected else "refuse_and_request_clarification",
                "expected_element_id": expected.element_id if expected else None,
                "expected_screenshot_point": expected.center.model_dump() if expected else None,
                "old_frame_ids": [item.element_id for previous in snapshots[:index]
                                  for item in previous.elements],
                "old_unique_button_point": snapshots[0].elements[0].center.model_dump() if index else None,
                "ambiguous": index == 2,
                "ground_truth_is_not_in_model_context": True,
            },
        })
    return records


class RecordingLocalClient(OpenAICompatibleClient):
    """Use the project's exact backend while retaining every SDK response before parsing."""

    def __init__(self, directory: Path):
        super().__init__(model_name=MODEL, base_url=BASE_URL, api_key="ollama-local",
                         timeout_seconds=CONDITION_TIMEOUT_SECONDS, max_retries=0,
                         temperature=0.0, max_tokens=2048, environ={})
        self.directory = directory
        self.deadline = 0.0
        self.condition_name = ""
        self.condition_instruction = ""
        self.condition_task_id = ""
        self.calls: list[dict[str, Any]] = []
        self._sdk: Any = None
        self._proxy: Any = None

    def _ensure_client(self) -> Any:
        if self._proxy is None:
            self._sdk = super()._ensure_client()
            self._proxy = SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=self._create),
            ))
        return self._proxy

    def _create(self, **request: Any) -> Any:
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise ModelError("the condition's 120-second deadline was exhausted before an HTTP request")
        number = len(self.calls) + 1
        record: dict[str, Any] = {
            "request_number": number, "condition": self.condition_name,
            "started_at_utc": utc_now(), "timeout_seconds": remaining,
            "model": request["model"], "temperature": request["temperature"],
            "max_tokens": request.get("max_tokens"), "transport_retries": 0,
        }
        self.calls.append(record)
        path = self.directory / f"request_{number:03d}.json"
        write_json(path, record)
        started = time.perf_counter()
        try:
            response = self._sdk.chat.completions.create(**request, timeout=remaining)
            record["raw_sdk_response"] = response.model_dump(mode="json")
            choices = record["raw_sdk_response"].get("choices") or []
            raw_content = choices[0].get("message", {}).get("content", "") if choices else ""
            try:
                parsed = parse_plan(raw_content, instruction=self.condition_instruction,
                                    max_steps=3, task_id=self.condition_task_id)
                record["independent_raw_parse_check"] = {"ok": True, "plan": parsed.as_dict()}
            except PlanParseError as exc:
                record["independent_raw_parse_check"] = {"ok": False, "error": str(exc)}
            return response
        except Exception as exc:
            record["transport_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_at_utc"] = utc_now()
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            write_json(path, record)


def evaluate(plan_result: Any, condition: dict[str, Any]) -> dict[str, Any]:
    snapshot = condition["snapshot"]
    ground_truth = condition["ground_truth"]
    raw = plan_result.response.content if plan_result.response is not None else ""
    try:
        payload = extract_json_object(raw)
    except PlanParseError:
        payload = None
    plan = plan_result.plan
    executable = plan.executable_steps if plan is not None else []
    raw_steps = payload.get("steps") if isinstance(payload, dict) else None
    raw_errors = payload.get("errors") if isinstance(payload, dict) else None
    refusal = bool(
        isinstance(raw_errors, list) and any(str(item).strip() for item in raw_errors)
        and isinstance(raw_steps, list)
        and not any(isinstance(step, dict) and step.get("action_type") != "finish" for step in raw_steps)
    )
    resolutions = []
    adapter = ActionAdapter(platform="win32")
    for step in executable:
        record: dict[str, Any] = {"step_id": step.step_id, "action_type": step.action_type}
        try:
            resolved = adapter.resolve(step, snapshot)
            current = snapshot.element(resolved.element_id) if resolved.element_id else None
            record.update(
                adapter_accepted=True, resolved=resolved.model_dump(mode="json"),
                uses_current_listed_geometry=bool(
                    current is not None and resolved.screenshot_point == current.center
                ),
            )
        except ActionResolutionError as exc:
            record.update(adapter_accepted=False, resolution_error=str(exc))
        resolutions.append(record)
    serialized_steps = json.dumps([step.model_dump(mode="json") for step in executable])
    stale_ids = [value for value in ground_truth["old_frame_ids"] if value in serialized_steps]
    coordinate_keys = {"x", "y", "start", "end", "coordinates", "coordinate", "point"}
    coordinate_arguments = any(coordinate_keys & step.arguments.keys() for step in executable)
    expected = ground_truth["expected_element_id"]
    if ground_truth["ambiguous"]:
        target_check_passed = refusal
    else:
        target_check_passed = bool(
            plan_result.ok and plan is not None and not plan.errors and len(executable) == 1
            and executable[0].action_type == "click" and not stale_ids and not coordinate_arguments
            and resolutions[0].get("adapter_accepted")
            and resolutions[0]["resolved"]["element_id"] == expected
            and resolutions[0]["resolved"]["screenshot_point"] == ground_truth["expected_screenshot_point"]
        )
    return {
        "target_check_passed": target_check_passed,
        "model_refusal_observed": refusal, "executable_step_count": len(executable),
        "adapter_resolutions": resolutions, "stale_id_references": stale_ids,
        "explicit_coordinate_arguments": coordinate_arguments,
        "runtime_plan_errors_present": bool(plan is not None and plan.errors),
        "desktop_actions_dispatched": 0,
        "note": "Adapter acceptance validates a current candidate, not semantic disambiguation or desktop success.",
    }


def run_probe(directory: Path, prepare_only: bool = False) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "frames").mkdir()
    prepared = conditions(directory)
    source_before = {path: sha256(REPO_ROOT / path) for path in SOURCE_PATHS}
    summary: dict[str, Any] = {
        "schema": "week6.real_local_model_saved_synthetic_gui_probe.v1",
        "scope": "Real local model + saved synthetic GUI images; no desktop input dispatched.",
        "started_at_utc": utc_now(), "provider": "openai_compatible", "model": MODEL,
        "base_url": BASE_URL, "python_version": platform.python_version(),
        "platform": platform.platform(), "desktop_actions_dispatched": 0,
        "synthetic_images": True, "real_desktop_screenshots": False,
        "ocr_was_run": False, "mock_client_used": False,
        "accepted_C11_weights_loaded_or_modified": False,
        "temperature": 0.0, "max_tokens": 2048, "model_transport_max_retries": 0,
        "max_format_retries": 1, "condition_timeout_seconds": CONDITION_TIMEOUT_SECONDS,
        "initial_planner_invocations_per_condition": 1,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "source_sha256_before": source_before, "conditions": [],
        "actual_http_requests": 0, "conditions_attempted": 0,
        "declared_conditions": [condition["condition"] for condition in prepared],
        "limitations": [
            "Three hand-authored English synthetic frames; no real GUI captures or OCR errors.",
            "Each condition is one planner invocation; format retries are retained, not selected for success.",
            "Recovery feedback is supplied to planning; no desktop failure/recovery was executed.",
            "The installed Ollama base model is separate from the accepted Week5 C11 LoRA.",
            "No desktop task success rate or model-quality improvement is inferred.",
        ],
    }
    summary_path = directory / "summary.json"
    for index, condition in enumerate(prepared, 1):
        write_json(directory / f"input_{index:02d}.json", {
            "condition": condition["condition"], "instruction": condition["instruction"],
            "context": condition["context"],
            "snapshot": condition["snapshot"].model_dump(mode="json"),
            "image_sha256": sha256(Path(condition["snapshot"].image_path)),
            "ground_truth": condition["ground_truth"],
        })
    write_json(summary_path, summary)
    if prepare_only:
        summary["status"] = "inputs_prepared_no_model_requests"
        summary["actual_http_requests"] = 0
        write_json(summary_path, summary)
        return summary
    try:
        tags = local_get("/api/tags")
        selected = [item for item in tags.get("models", []) if item.get("name") == MODEL]
        if len(selected) != 1:
            raise RuntimeError("the one fixed model is not installed; no download or substitute is allowed")
        summary["selected_installed_model"] = selected[0]
        summary["ollama_running_before"] = local_get("/api/ps")
        try:
            summary["git_commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                text=True, check=True, timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            summary["git_commit"] = None
        client = RecordingLocalClient(directory)
        planner = TaskPlanner(client, max_steps=3, allow_real_execution=False, max_format_retries=1)
        for index, condition in enumerate(prepared, 1):
            client.condition_name = condition["condition"]
            client.condition_instruction = condition["instruction"]
            client.condition_task_id = f"W6_SAVED_{index}"
            started = time.perf_counter()
            client.deadline = started + CONDITION_TIMEOUT_SECONDS
            before_http = len(client.calls)
            before_project_count = client.request_count
            planned = planner.plan(
                condition["instruction"], context=condition["context"],
                image_path=condition["snapshot"].image_path, task_id=f"W6_SAVED_{index}",
            )
            record = {
                "condition": condition["condition"], "planner_invocations": 1,
                "actual_http_requests": len(client.calls) - before_http,
                "project_client_request_count": client.request_count - before_project_count,
                "condition_elapsed_ms": (time.perf_counter() - started) * 1000,
                "plan_result": planned.as_dict(),
                "last_model_raw_content": planned.response.content if planned.response else None,
                "target_comparison": evaluate(planned, condition),
                "request_numbers": [call["request_number"] for call in client.calls[before_http:]],
            }
            summary["conditions"].append(record)
            write_json(directory / f"condition_{index:02d}.json", record)
            write_json(summary_path, summary)
            print(json.dumps({"condition": record["condition"], "requests": record["actual_http_requests"],
                              "target_check_passed": record["target_comparison"]["target_check_passed"],
                              "elapsed_ms": record["condition_elapsed_ms"]}), flush=True)
            if any(call.get("transport_error") for call in client.calls[before_http:]):
                summary["stop_reason"] = "actual local transport failure; remaining conditions not attempted"
                break
            if planned.response is not None and planned.response.error:
                summary["stop_reason"] = "local model/backend error; remaining conditions not attempted"
                break
        summary["actual_http_requests"] = len(client.calls)
        summary["project_client_request_count"] = client.request_count
        summary["conditions_attempted"] = len(summary["conditions"])
        summary["target_checks_passed"] = sum(
            item["target_comparison"]["target_check_passed"] for item in summary["conditions"]
        )
        summary["status"] = "all_predeclared_conditions_attempted" if len(summary["conditions"]) == 3 else "stopped"
        summary["ollama_running_after"] = local_get("/api/ps")
    except Exception as exc:  # noqa: BLE001 - retain the one actual failed run; no blind retry
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        attempted = {item["condition"] for item in summary["conditions"]}
        summary["conditions_not_attempted"] = [
            item["condition"] for item in prepared if item["condition"] not in attempted
        ]
        summary["finished_at_utc"] = utc_now()
        source_after = {path: sha256(REPO_ROOT / path) for path in SOURCE_PATHS}
        summary["source_sha256_after"] = source_after
        summary["source_unchanged_during_probe"] = source_before == source_after
        write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "Document/Week6/results/model_recovery_20261024_001")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    summary = run_probe(args.output, args.prepare_only)
    print(json.dumps({"status": summary["status"], "summary": str(args.output / "summary.json")},
                     ensure_ascii=False), flush=True)
    return 0 if summary["status"] in {
        "inputs_prepared_no_model_requests", "all_predeclared_conditions_attempted",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
