"""Minimal per-run recording.

Week 2 does not build a full trajectory or monitoring system. Each demo run gets
its own session directory plus a small JSON summary, which is enough to prove
what was captured, what was recognised and what was executed.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .control.safety import redact_action, redact_result
from .schemas import ActionResult, DesktopAction, PerceptionResult, UIElement

WEEK2_OUTPUT_DIR = Path("outputs/week2")
PERCEPTION_FILE = "perception.json"
ACTION_FILE = "action.json"
SUMMARY_FILE = "run_summary.json"
BEFORE_IMAGE = "before.png"
AFTER_IMAGE = "after.png"
ANNOTATED_IMAGE = "annotated.png"


@dataclass
class RunSession:
    """One demo run: a directory plus helpers for the artefacts inside it."""

    session_id: str
    directory: Path
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def create(
        cls,
        root: str | Path = WEEK2_OUTPUT_DIR,
        session_id: str | None = None,
    ) -> RunSession:
        identifier = session_id or datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
        directory = Path(root) / identifier
        directory.mkdir(parents=True, exist_ok=True)
        return cls(session_id=identifier, directory=directory)

    def path_for(self, name: str) -> Path:
        return self.directory / name

    def save_json(self, name: str, payload: Any) -> Path:
        target = self.path_for(name)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return target

    def write_text(self, name: str, text: str) -> Path:
        target = self.path_for(name)
        target.write_text(text, encoding="utf-8")
        return target

    def save_log(self, lines: Sequence[str]) -> Path:
        return self.write_text("run.log", "\n".join(lines) + "\n")

    def save_perception(self, result: PerceptionResult) -> Path:
        return self.save_json(PERCEPTION_FILE, result.model_dump(mode="json"))

    def save_action(self, action: DesktopAction, result: ActionResult) -> Path:
        # Redact on the way out: a typed credential must never reach the file.
        payload = {
            "action": redact_action(action),
            "result": redact_result(result),
        }
        return self.save_json(ACTION_FILE, payload)

    def save_summary(self, summary: dict[str, Any]) -> Path:
        return self.save_json(SUMMARY_FILE, summary)


def build_run_summary(
    *,
    selected_target: str | None,
    action: DesktopAction | None,
    result: ActionResult | None,
    capture_time_ms: float,
    ocr_time_ms: float,
    total_time_ms: float,
    element_count: int = 0,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the minimal run record required by the Week 2 hand-off."""
    summary: dict[str, Any] = {
        # Local time with an explicit offset, matching the session directory name.
        "timestamp": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "selected_target": selected_target,
        "action": redact_action(action) if action else None,
        "dry_run": None if result is None else result.dry_run,
        "execution_result": None if result is None else redact_result(result),
        "capture_time_ms": round(capture_time_ms, 3),
        "ocr_time_ms": round(ocr_time_ms, 3),
        "total_time_ms": round(total_time_ms, 3),
        "element_count": element_count,
        "error": error,
    }
    if extra:
        summary.update(extra)
    return summary


def elements_to_payload(elements: Sequence[UIElement]) -> list[dict[str, Any]]:
    return [element.model_dump(mode="json") for element in elements]
