"""Per-run recording for a multi-step task.

``RunSession`` from Week 2 writes one ``before.png`` and one ``after.png`` per
session. A task that takes twenty actions would overwrite those twenty times, so
this recorder keeps every step in its own numbered directory and appends to a
single step log instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..recording import RunSession
from .schemas import ObservationSnapshot, StepRecord, TaskRunResult

REDACTED = "<redacted>"

#: Argument keys whose values must never reach the log.
SENSITIVE_KEYS = frozenset({"text", "clipboard", "password", "message", "content"})


def redact(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Replace typed content with a marker before it is written down.

    The length is kept so a failed step can still be diagnosed without storing
    what the user actually typed.
    """
    if not arguments:
        return {}
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if key.casefold() in SENSITIVE_KEYS and isinstance(value, str):
            safe[key] = f"{REDACTED}({len(value)} chars)"
        else:
            safe[key] = value
    return safe


class TaskRecorder:
    """Owns the output directory for one task run."""

    def __init__(self, session: RunSession) -> None:
        self.session = session
        self.directory = Path(session.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._step_log = self.directory / "steps.jsonl"

    @property
    def run_id(self) -> str:
        return Path(self.directory).name

    # ── observations ───────────────────────────────────────────────────
    def save_observation(self, snapshot: ObservationSnapshot) -> Path:
        """Write the element list for one frame next to its image.

        The element list is always written, even when no image was saved: it is
        what makes a resolved coordinate traceable back to the frame it came from,
        and losing it would leave a step record pointing at nothing.
        """
        target = self.directory / f"{snapshot.observation_id}.json"
        target.write_text(
            json.dumps(
                {
                    "observation_id": snapshot.observation_id,
                    "captured_at": snapshot.captured_at.isoformat(),
                    "image_path": snapshot.image_path,
                    "screen_info": snapshot.screen_info.model_dump(),
                    "ocr_engine": snapshot.ocr_engine,
                    "processing_time_ms": round(snapshot.processing_time_ms, 3),
                    "errors": snapshot.errors,
                    "elements": [item.model_dump() for item in snapshot.elements],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    # ── steps ──────────────────────────────────────────────────────────
    def append_step(self, record: StepRecord) -> None:
        """Append one line; steps are never rewritten, so a crash keeps the history."""
        with self._step_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def read_steps(self) -> list[dict[str, Any]]:
        if not self._step_log.exists():
            return []
        return [
            json.loads(line)
            for line in self._step_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # ── summary ────────────────────────────────────────────────────────
    def write_summary(self, result: TaskRunResult) -> Path:
        target = self.directory / "task_summary.json"
        payload = result.model_dump(mode="json")
        payload["action_count"] = result.action_count
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
