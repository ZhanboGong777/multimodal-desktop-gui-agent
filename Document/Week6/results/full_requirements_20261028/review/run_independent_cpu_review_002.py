"""Execute only the independent offline review target and preserve raw evidence."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "cpu_validation_002"
PYTHON = Path("D:/Developer/multimodal-desktop-gui-agent/.venv-week5/Scripts/python.exe")
PINS = [
    "src/gui_agent/planning/semantic_workflow.py",
    "src/gui_agent/planning/workflow_decomposer.py",
    "src/gui_agent/planning/planner.py",
    "src/gui_agent/runtime/monitoring.py",
    "src/gui_agent/runtime/workflow.py",
    "src/gui_agent/runtime/runner.py",
    "src/gui_agent/runtime/schemas.py",
    "src/gui_agent/runtime/verification.py",
    "tests/test_week6_full_requirements_review.py",
    "tests/test_week6_semantic_subtasks.py",
    "tests/test_week6_automatic_decomposition.py",
    "tests/test_week6_rolling_actions.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    before = {name: sha(ROOT / name) for name in PINS}
    command = [str(PYTHON), "-m", "pytest", "-q", "tests/test_week6_full_requirements_review.py",
               "--basetemp=.pytest-tmp-fullrequirements-review002", "--tb=short",
               f"--junitxml={OUT / 'junit.xml'}"]
    started_utc, started = datetime.now(UTC).isoformat(), time.perf_counter()
    with (OUT / "stdout.txt").open("wb") as stdout, (OUT / "stderr.txt").open("wb") as stderr:
        completed = subprocess.run(command, cwd=ROOT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            stdout=stdout, stderr=stderr, timeout=120, check=False)
    elapsed = time.perf_counter() - started
    after = {name: sha(ROOT / name) for name in PINS}
    suite = ET.parse(OUT / "junit.xml").getroot()
    cases = list(suite.iter("testcase"))
    failures = [case.attrib["name"] for case in cases if case.find("failure") is not None or case.find("error") is not None]
    result = {
        "schema": "week6.full_requirements.independent_cpu_review.v1",
        "reviewer": "/root/week6_replanning_review", "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(), "command": command, "cwd": str(ROOT),
        "actual_returncode": completed.returncode, "wrapper_wall_seconds": elapsed,
        "collected_testcases": len(cases), "failed_testcases": failures,
        "selected_source_before": before, "selected_source_after": after,
        "selected_source_unchanged": before == after,
        "artifacts": {name: sha(OUT / name) for name in ("stdout.txt", "stderr.txt", "junit.xml")},
        "runner_sha256": sha(Path(__file__)),
        "scope": "Offline scripted observations/model replies/CPU executor doubles using real runtime. No real model requests, GUI operations, OCR measurements or task-success claims.",
        "passed": completed.returncode == 0 and not failures and before == after,
    }
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"directory": str(OUT), "returncode": completed.returncode,
                      "testcases": len(cases), "failures": failures,
                      "selected_source_unchanged": before == after,
                      "audit_sha256": sha(OUT / "audit.json")}, ensure_ascii=False))
    return completed.returncode if before == after else 2


if __name__ == "__main__":
    raise SystemExit(main())
