"""Small, passive run-status records, independent of GUI and model dependencies.

The runner remains the authority for actions and verification. This module only
records bounded codes and counters. A reader requires agreement between the
append-only journal and the atomically replaced snapshot before trusting it.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

STATUS_SCHEMA = "week6.run_status.v1"
SEMANTIC_STATUS_SCHEMA = "week6.run_status.semantic.v1"
READ_SCHEMA = "week6.status_read.v1"
STAGES = frozenset({
    "initializing", "preconditions", "decomposing", "observing", "planning",
    "confirmation", "resolving", "executing", "step_verification",
    "subgoal_verification", "task_verification", "recovering", "finished",
})
# Kept local so the standalone reader does not import runtime.__init__, which
# imports perception and control. Tests compare this set with schemas.TaskStatus.
TASK_STATUSES = frozenset({
    "planned", "dry_run_completed", "succeeded", "failed", "blocked",
    "cancelled", "timed_out",
})
_CODE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,63}\Z", re.ASCII)
_RUN_ID = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_IDENTITY = re.compile(r"(?:windows|proc):[0-9]{1,24}\Z", re.ASCII)
_WRITE_CODES = frozenset({"directory_write_failed", "journal_write_failed", "latest_write_failed"})
_MAX_ACTIONS = 100_000
_MAX_PASSES = 10_000
_MAX_SUBGOALS = 10_000
_MAX_SEQUENCE = 1_000_000
_MAX_ELAPSED_MS = 366 * 24 * 60 * 60 * 1000
_MAX_RECORD_BYTES = 16_384
_MAX_JOURNAL_BYTES = 64 * 1024 * 1024
_MAX_EVENTS = 100_000
# A Windows reader may briefly hold the destination without FILE_SHARE_DELETE.
# Retry only the atomic file replacement, never an event append or an action.
# Waiting uses real time and remains part of the caller's overall task budget.
_REPLACE_RETRY_DELAYS = (0.005, 0.01, 0.02)
_FIELDS = frozenset({
    "schema", "run_id", "case_id", "execute", "pid", "process_identity",
    "seq", "event", "stage", "task_status", "updated_at_utc", "elapsed_ms",
    "max_actions", "max_planning_attempts", "planning_pass", "actions_used",
    "subgoals_total", "subgoals_completed", "subgoal_id", "detail_code", "write_failures",
})
_LOG = logging.getLogger(__name__)


def _integer(value: Any, name: str, maximum: int, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def _code(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or not (empty and value == "" or _CODE.fullmatch(value)):
        raise ValueError(f"invalid {name}")
    return value


def _utc(value: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("invalid timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _json(data: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_value: str) -> Any:
        raise ValueError("nonfinite JSON value")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def _validate(value: Any) -> dict[str, Any]:
    """Validate an entire record; unknown fields are never echoed by the reader."""
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("unexpected status fields")
    if value["schema"] not in {STATUS_SCHEMA, SEMANTIC_STATUS_SCHEMA} or not isinstance(value["run_id"], str):
        raise ValueError("invalid schema or run identity")
    if not _RUN_ID.fullmatch(value["run_id"]):
        raise ValueError("invalid run identity")
    _code(value["case_id"], "case_id")
    if type(value["execute"]) is not bool:
        raise ValueError("invalid execute")
    _integer(value["pid"], "pid", 2**32 - 1, minimum=1)
    identity = value["process_identity"]
    if identity is not None and (not isinstance(identity, str) or not _IDENTITY.fullmatch(identity)):
        raise ValueError("invalid process identity")
    _integer(value["seq"], "sequence", _MAX_SEQUENCE, minimum=1)
    if value["event"] not in {"started", "updated", "finished"} or value["stage"] not in STAGES:
        raise ValueError("invalid event or stage")
    terminal = value["event"] == "finished"
    if terminal:
        if value["task_status"] not in TASK_STATUSES or value["stage"] != "finished":
            raise ValueError("invalid terminal status")
        if not value["execute"] and value["task_status"] == "succeeded":
            raise ValueError("dry run cannot succeed")
        if value["execute"] and value["task_status"] == "dry_run_completed":
            raise ValueError("executing run cannot be a dry run")
    elif value["task_status"] is not None or value["stage"] == "finished":
        raise ValueError("nonterminal record claims completion")
    if value["event"] == "started" and (value["seq"] != 1 or value["stage"] != "initializing"):
        raise ValueError("invalid initial record")
    _utc(value["updated_at_utc"])
    elapsed = value["elapsed_ms"]
    if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or not 0 <= elapsed <= _MAX_ELAPSED_MS:
        raise ValueError("invalid elapsed time")
    actions = _integer(value["max_actions"], "max_actions", _MAX_ACTIONS, minimum=1)
    passes = _integer(value["max_planning_attempts"], "max_planning_attempts", _MAX_PASSES, minimum=1)
    goals = _integer(value["subgoals_total"], "subgoals_total", _MAX_SUBGOALS)
    _integer(value["actions_used"], "actions_used", actions)
    _integer(value["planning_pass"], "planning_pass", passes)
    _integer(value["subgoals_completed"], "subgoals_completed", goals)
    if value["subgoal_id"] is not None:
        _code(value["subgoal_id"], "subgoal_id")
    _code(value["detail_code"], "detail_code", empty=True)
    failures = value["write_failures"]
    if not isinstance(failures, list) or len(failures) > len(_WRITE_CODES):
        raise ValueError("invalid write diagnostics")
    if any(not isinstance(code, str) or code not in _WRITE_CODES for code in failures):
        raise ValueError("invalid write diagnostic")
    if len(set(failures)) != len(failures):
        raise ValueError("duplicate write diagnostic")
    return value


def _process_info(pid: int) -> tuple[bool | None, str | None]:
    """Read process identity without signals, process control, psutil or GUI APIs."""
    if os.name == "nt":
        from ctypes import wintypes

        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.GetExitCodeProcess.restype = wintypes.BOOL
            kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            kernel.GetProcessTimes.restype = wintypes.BOOL
            handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not handle:
                # Invalid parameter means no such PID; access denial is unknown.
                return (False if ctypes.get_last_error() == 87 else None), None
            try:
                exit_code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return None, None
                if exit_code.value != 259:  # STILL_ACTIVE
                    return False, None
                created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
                if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in (created, exited, system, user))):
                    return None, None
                stamp = (created.dwHighDateTime << 32) | created.dwLowDateTime
                return True, f"windows:{stamp}"
            finally:
                kernel.CloseHandle(handle)
        except (OSError, AttributeError):
            return None, None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = raw.rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return False, None
        return True, f"proc:{int(fields[19])}"
    except FileNotFoundError:
        return (False if Path("/proc").is_dir() else None), None
    except (OSError, ValueError, IndexError):
        return None, None


class RunMonitor:
    """One writer for one run directory; all methods only persist status.

    Invalid programmer input raises ValueError. File-write failures are exposed
    as fixed diagnostic codes in returned snapshots, write_failures and logging;
    only transient PermissionError during atomic replacement has a small bounded
    file retry. No event append, model request, or desktop action is retried.
    """

    def __init__(self, directory: str | Path, *, clock: Callable[[], float] = time.monotonic):
        self.directory = Path(directory)
        self.clock = clock
        self._status: dict[str, Any] | None = None
        self._started: float | None = None
        self._finished = False
        self._failures: list[str] = []
        self._lock = threading.RLock()

    @property
    def write_failures(self) -> tuple[str, ...]:
        return tuple(self._failures)

    def _failure(self, code: str) -> None:
        if code not in self._failures:
            self._failures.append(code)
            _LOG.warning("Run monitor persistence failure: %s", code)

    def _append_event(self, status: dict[str, Any]) -> None:
        mode = "x" if status["seq"] == 1 else "a"
        with (self.directory / "status_events.jsonl").open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(status, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()

    def _replace_latest(self, status: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=self.directory,
                                             prefix=".run_status.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(status, stream, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
            for attempt in range(len(_REPLACE_RETRY_DELAYS) + 1):
                try:
                    os.replace(temporary, self.directory / "run_status.json")
                    break
                except PermissionError:
                    if attempt == len(_REPLACE_RETRY_DELAYS):
                        raise
                    time.sleep(_REPLACE_RETRY_DELAYS[attempt])
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _persist(self, status: dict[str, Any]) -> dict[str, Any]:
        # Consume the sequence before I/O. If Ctrl+C lands after a complete
        # append, the cancellation event must not reuse that sequence number.
        self._status = status
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._failure("directory_write_failed")
        status["write_failures"] = list(self._failures)
        try:
            self._append_event(status)
        except OSError:
            self._failure("journal_write_failed")
        status["write_failures"] = list(self._failures)
        try:
            self._replace_latest(status)
        except OSError:
            self._failure("latest_write_failed")
        status["write_failures"] = list(self._failures)
        self._status = status
        return {**status, "write_failures": list(status["write_failures"])}

    def start(self, case_id: str, execute: bool, max_actions: int, max_planning_attempts: int,
              subgoals_total: int = 0, *, dynamic_subgoals: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._status is not None:
                raise ValueError("monitor already started")
            if any((self.directory / name).exists() for name in ("run_status.json", "status_events.jsonl")):
                raise ValueError("refusing to overwrite existing monitor evidence")
            if type(dynamic_subgoals) is not bool:
                raise ValueError("invalid dynamic_subgoals")
            started = self.clock()
            if not math.isfinite(started):
                raise ValueError("invalid monotonic clock")
            self._started = started
            _, identity = _process_info(os.getpid())
            status = {
                "schema": SEMANTIC_STATUS_SCHEMA if dynamic_subgoals else STATUS_SCHEMA,
                "run_id": uuid4().hex, "case_id": case_id,
                "execute": execute, "pid": os.getpid(), "process_identity": identity,
                "seq": 1, "event": "started", "stage": "initializing", "task_status": None,
                "updated_at_utc": datetime.now(UTC).isoformat(), "elapsed_ms": 0.0,
                "max_actions": max_actions, "max_planning_attempts": max_planning_attempts,
                "planning_pass": 0, "actions_used": 0, "subgoals_total": subgoals_total,
                "subgoals_completed": 0, "subgoal_id": None, "detail_code": "", "write_failures": [],
            }
            return self._persist(_validate(status))

    def _next(self, stage: str, *, planning_pass: int | None, actions_used: int | None,
              subgoal_id: str | None, subgoals_completed: int | None,
              subgoals_total: int | None, detail_code: str,
              task_status: str | None = None) -> dict[str, Any]:
        if self._status is None or self._started is None:
            raise ValueError("monitor has not started")
        if self._finished:
            raise ValueError("monitor already finished")
        status = dict(self._status)
        if subgoals_total is not None:
            _integer(subgoals_total, "subgoals_total", _MAX_SUBGOALS)
            if subgoals_total != status["subgoals_total"]:
                # Only the explicit semantic schema may accept a new remaining
                # plan. Completed goals never regress; abandoned plans remain
                # in the semantic ledger rather than inflating this total.
                if status["schema"] != SEMANTIC_STATUS_SCHEMA or stage != "decomposing" or task_status:
                    raise ValueError("subgoal total change requires semantic decomposition")
                status["subgoals_total"] = subgoals_total
        for name, value in (("planning_pass", planning_pass), ("actions_used", actions_used),
                            ("subgoals_completed", subgoals_completed)):
            if value is not None:
                if type(value) is not int or value < status[name]:
                    raise ValueError(f"nonmonotonic {name}")
                status[name] = value
        elapsed = (self.clock() - self._started) * 1000.0
        if not math.isfinite(elapsed) or elapsed < status["elapsed_ms"]:
            raise ValueError("nonmonotonic elapsed time")
        status.update(seq=status["seq"] + 1, event="finished" if task_status else "updated",
                      stage=stage, task_status=task_status, subgoal_id=subgoal_id,
                      detail_code=detail_code, updated_at_utc=datetime.now(UTC).isoformat(),
                      elapsed_ms=elapsed, write_failures=list(self._failures))
        return _validate(status)

    def update(self, stage: str, *, planning_pass: int | None = None, actions_used: int | None = None,
               subgoal_id: str | None = None, subgoals_completed: int | None = None,
               subgoals_total: int | None = None, detail_code: str = "") -> dict[str, Any]:
        """Record a phase; None clears the current subgoal ID, without verifying it."""
        with self._lock:
            status = self._next(stage, planning_pass=planning_pass, actions_used=actions_used,
                                subgoal_id=subgoal_id, subgoals_completed=subgoals_completed,
                                subgoals_total=subgoals_total, detail_code=detail_code)
            return self._persist(status)

    def finish(self, status: str, *, actions_used: int | None = None,
               subgoals_completed: int | None = None, detail_code: str = "") -> dict[str, Any]:
        with self._lock:
            if status not in TASK_STATUSES:
                raise ValueError("unknown task status")
            result = self._next("finished", planning_pass=None, actions_used=actions_used,
                                subgoal_id=None, subgoals_completed=subgoals_completed,
                                subgoals_total=None, detail_code=detail_code, task_status=status)
            snapshot = self._persist(result)
            self._finished = True
            return snapshot


def _read_latest(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = stream.read(_MAX_RECORD_BYTES + 1)
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("oversized latest record")
    return _validate(_json(data))


def _read_journal(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if path.stat().st_size > _MAX_JOURNAL_BYTES:
        raise ValueError("oversized journal")
    previous: dict[str, Any] | None = None
    diagnostics: list[str] = []
    with path.open("rb") as stream:
        for number in range(1, _MAX_EVENTS + 2):
            line = stream.readline(_MAX_RECORD_BYTES + 1)
            if not line:
                break
            if number > _MAX_EVENTS or len(line) > _MAX_RECORD_BYTES:
                raise ValueError("journal exceeds limits")
            if not line.endswith(b"\n"):
                diagnostics.append("journal_incomplete_line")
            try:
                event = _validate(_json(line))
            except (ValueError, TypeError, UnicodeError):
                diagnostics.append("journal_invalid_record")
                continue
            if previous is None:
                if event["seq"] != 1 or event["event"] != "started":
                    diagnostics.append("journal_missing_start")
            else:
                if event["run_id"] != previous["run_id"] or event["seq"] != previous["seq"] + 1:
                    diagnostics.append("journal_sequence_mismatch")
                if previous["event"] == "finished":
                    diagnostics.append("journal_after_finish")
                for key in ("schema", "case_id", "execute", "pid", "process_identity", "max_actions",
                            "max_planning_attempts"):
                    if event[key] != previous[key]:
                        diagnostics.append("journal_identity_mismatch")
                if (event["subgoals_total"] != previous["subgoals_total"]
                        and (event["schema"] != SEMANTIC_STATUS_SCHEMA
                            or previous["schema"] != SEMANTIC_STATUS_SCHEMA
                            or event["stage"] != "decomposing" or event["event"] != "updated")):
                    diagnostics.append("journal_identity_mismatch")
                for key in ("elapsed_ms", "planning_pass", "actions_used", "subgoals_completed"):
                    if event[key] < previous[key]:
                        diagnostics.append("journal_progress_regressed")
            previous = event
    return previous, sorted(set(diagnostics))


def _legacy(directory: Path) -> dict[str, Any]:
    path = directory / "task_summary.json"
    if not path.exists():
        return {"state": "missing", "snapshot": None, "diagnostics": ["monitor_files_missing"]}
    try:
        # Legacy summaries contain prose and actions. Read a bounded file but
        # expose only validated identifiers and terminal-status metadata.
        if path.stat().st_size > _MAX_JOURNAL_BYTES:
            raise ValueError("legacy summary exceeds limit")
        raw = _json(path.read_bytes())
        if not isinstance(raw, dict) or raw.get("status") not in TASK_STATUSES:
            raise ValueError("invalid legacy status")
        case_id = _code(raw.get("case_id"), "legacy case_id")
        execute = raw.get("execute", False)
        if type(execute) is not bool or (not execute and raw["status"] == "succeeded"):
            raise ValueError("invalid legacy execution status")
        snapshot = {"case_id": case_id, "task_status": raw["status"], "execute": execute}
        return {"state": "legacy", "snapshot": snapshot, "diagnostics": ["legacy_status_unmonitored"]}
    except (ValueError, TypeError, UnicodeError):
        return {"state": "corrupt", "snapshot": None, "diagnostics": ["legacy_summary_invalid"]}
    except OSError:
        return {"state": "unavailable", "snapshot": None, "diagnostics": ["legacy_summary_unreadable"]}


def read_run_status(directory: str | Path, *, stale_after_seconds: float = 30.0,
                    now: datetime | None = None) -> dict[str, Any]:
    """Read evidence only; damaged, stale or unverifiable runs are never active.

    state=finished means the final monitor event is consistent, not that the
    task succeeded. task_status supplies the actual terminal verdict. Legacy
    summaries remain available with state=legacy and trusted=False.
    """
    if type(stale_after_seconds) not in {int, float} or not math.isfinite(stale_after_seconds) or not 0 < stale_after_seconds <= 86_400:
        raise ValueError("invalid stale_after_seconds")
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("now must have a timezone")
    directory = Path(directory)
    result: dict[str, Any] = {"schema": READ_SCHEMA, "trusted": False, "age_seconds": None}
    latest_path, events_path = directory / "run_status.json", directory / "status_events.jsonl"
    try:
        if not latest_path.exists() and not events_path.exists():
            return {**result, **_legacy(directory)}
        if not latest_path.exists() or not events_path.exists():
            return {**result, "state": "inconsistent", "snapshot": None,
                    "diagnostics": ["latest_missing" if not latest_path.exists() else "journal_missing"]}
        # Event append precedes replacement, so briefly different versions are
        # expected during writes. Bounded read-only retries never repair files.
        latest = None
        diagnostics: list[str] = []
        for _ in range(3):
            latest = _read_latest(latest_path)
            final, diagnostics = _read_journal(events_path)
            if latest == final:
                break
        if latest != final:
            diagnostics.append("latest_journal_mismatch")
        assert latest is not None
        if diagnostics:
            return {**result, "state": "inconsistent", "snapshot": None,
                    "diagnostics": sorted(set(diagnostics))}
        age = (moment.astimezone(UTC) - _utc(latest["updated_at_utc"])).total_seconds()
        result["age_seconds"] = max(0.0, round(age, 3))
        if age < -5:
            return {**result, "state": "unconfirmed", "snapshot": latest,
                    "diagnostics": ["timestamp_in_future"]}
        if latest["write_failures"]:
            return {**result, "state": "degraded", "snapshot": latest,
                    "diagnostics": list(latest["write_failures"])}
        if latest["event"] == "finished":
            return {**result, "state": "finished", "trusted": True, "snapshot": latest, "diagnostics": []}
        if age > stale_after_seconds:
            return {**result, "state": "stale", "snapshot": latest, "diagnostics": ["status_not_recent"]}
        alive, identity = _process_info(latest["pid"])
        if alive is False or identity is not None and identity != latest["process_identity"]:
            return {**result, "state": "inactive", "snapshot": latest,
                    "diagnostics": ["writer_process_not_current"]}
        if alive is not True or identity is None or latest["process_identity"] is None:
            return {**result, "state": "unconfirmed", "snapshot": latest,
                    "diagnostics": ["writer_process_unconfirmed"]}
        return {**result, "state": "active", "trusted": True, "snapshot": latest, "diagnostics": []}
    except (ValueError, TypeError, UnicodeError):
        return {**result, "state": "corrupt", "snapshot": None, "diagnostics": ["monitor_record_invalid"]}
    except OSError:
        return {**result, "state": "unavailable", "snapshot": None, "diagnostics": ["monitor_files_unreadable"]}
