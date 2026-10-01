"""Per-run recording for a multi-step task.

``RunSession`` from Week 2 writes one ``before.png`` and one ``after.png`` per
session. A task that takes twenty actions would overwrite those twenty times, so
this recorder keeps every step in its own numbered directory and appends to a
single step log instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..control.safety import REDACTED, is_sensitive_text  # the same marker the action log uses
from ..recording import RunSession
from .schemas import ObservationSnapshot, StepRecord, TaskRunResult

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


def _mask(text: str | None) -> str | None:
    """Blank credential-shaped on-screen text, and leave everything else alone.

    Deliberately narrow: the element list is how a resolved coordinate is traced
    back to the frame it came from, so masking ordinary interface text would cost
    the record its purpose. A password field's contents are the case this is for.
    """
    if text and is_sensitive_text(text):
        return REDACTED
    return text


#: Keys whose value is a credential wherever it appears in the configuration.
_SECRET_KEYS = frozenset({"api_key", "apikey", "key", "token", "secret", "password"})


def _mask_config(payload: Any) -> Any:
    """Recursively mask anything whose key name says it is a credential."""
    if isinstance(payload, Mapping):
        return {
            key: (REDACTED if str(key).casefold() in _SECRET_KEYS else _mask_config(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_mask_config(item) for item in payload]
    return payload


def _mask_step(payload: dict[str, Any]) -> dict[str, Any]:
    """Mask typed text wherever one step record carries it.

    A step record names the action twice: ``resolved`` is what the adapter decided
    to do, ``action_result`` is what the executor was handed. Only the second was
    masked, so the raw text was written to ``steps.jsonl`` - and to the summary
    that embeds the same records - while the usage guide said typed text is never
    stored. Found by reading the file a real run produced rather than the object
    the code returned.
    """
    masked = dict(payload)
    for key in ("resolved", "action_result"):
        block = masked.get(key)
        if isinstance(block, Mapping) and isinstance(block.get("action"), Mapping):
            masked[key] = {**block, "action": redact(dict(block["action"]))}
    return masked


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
                    "ocr_notices": snapshot.notices,
                    # 16.2 asks for sensitive text to be kept out of the records.
                    # Typed text is masked by the action redactor; this is the other
                    # way a credential reaches a file - OCR reading it off the
                    # screen. Only credential-shaped text is masked, so the element
                    # list still traces a coordinate back to the words it came from.
                    "elements": [
                        {**item.model_dump(), "text": _mask(item.text)}
                        for item in snapshot.elements
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return target

    def save_config(self, config: Mapping[str, Any]) -> Path:
        """Write the effective configuration this run used, with values masked.

        14.1 names ``run_config.json`` and nothing wrote it: a run recorded the
        model name and the limits in its summary, so a reader could not tell which
        ``max_elements``, OCR engine or verification timeouts produced the frames
        in front of them. The environment table in the test report then asks for
        `timeout_seconds` "in the config the run used", which is this file.

        Values are masked by key name rather than trusted to be safe: the model
        config holds no credential by design - it is read from the environment -
        but a key pasted into a YAML field would otherwise be written to disk and
        then copied into the repository by the evidence collector.
        """
        target = self.directory / "run_config.json"
        target.write_text(
            json.dumps(_mask_config(config), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    # ── steps ──────────────────────────────────────────────────────────
    def append_step(self, record: StepRecord) -> None:
        """Append one line; steps are never rewritten, so a crash keeps the history."""
        payload = _mask_step(record.model_dump(mode="json"))
        with self._step_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
        # The summary embeds the same step records, so it needs the same masking -
        # and it is the file the evidence collector copies into the repository.
        payload["steps"] = [_mask_step(step) for step in payload.get("steps", [])]
        payload["action_count"] = result.action_count
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
