"""Read one run's status without importing a model or desktop-control package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path


def _reader():
    # Importing gui_agent.runtime would execute its perception/control exports.
    # This file is deliberately dependency-free and safe to load directly.
    path = Path(__file__).resolve().parents[1] / "src/gui_agent/runtime/monitoring.py"
    spec = importlib.util.spec_from_file_location("_week6_passive_monitoring", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("status reader is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_run_status


def _interval(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("interval must be a number") from exc
    if not math.isfinite(number) or not 0.05 <= number <= 60.0:
        raise argparse.ArgumentTypeError("interval must be between 0.05 and 60 seconds")
    return number


def _human(result: dict) -> str:
    snapshot = result.get("snapshot") or {}
    fields = [result["state"]]
    for key, label in (("case_id", "case"), ("stage", "stage"), ("task_status", "status")):
        if snapshot.get(key) is not None:
            fields.append(f"{label}={snapshot[key]}")
    for used, maximum, label in (("actions_used", "max_actions", "actions"),
                                 ("planning_pass", "max_planning_attempts", "passes"),
                                 ("subgoals_completed", "subgoals_total", "verified_subgoals")):
        if used in snapshot:
            fields.append(f"{label}={snapshot[used]}/{snapshot[maximum]}")
    if result.get("diagnostics"):
        fields.append("diagnostics=" + ",".join(result["diagnostics"]))
    if snapshot.get("detail_code"):
        fields.append("detail=" + snapshot["detail_code"])
    return " ".join(fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--watch", action="store_true", help="poll until a verified terminal status")
    parser.add_argument("--interval", type=_interval, default=1.0)
    parser.add_argument("--json", action="store_true", help="print one JSON object per read")
    args = parser.parse_args(argv)
    read_run_status = _reader()
    try:
        while True:
            result = read_run_status(args.run_dir)
            print(json.dumps(result, ensure_ascii=True, allow_nan=False) if args.json else _human(result), flush=True)
            if not args.watch or result["state"] == "finished":
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
