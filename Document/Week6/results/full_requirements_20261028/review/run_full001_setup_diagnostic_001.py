"""One exact-node offline reproduction of full001's external basetemp shape."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "full001_setup_diagnostic_001"
PYTHON = "D:/Developer/multimodal-desktop-gui-agent/.venv-week5/Scripts/python.exe"
BASE = ROOT.parent / "tmp/week6_full001_setup_diagnostic_001/pytest"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins():
    selected = json.loads((ROOT / "Document/Week6/results/full_requirements_20261028/full_validation_001/tested_code_pins_before.json").read_text("utf8"))
    return {name: digest(ROOT / name) for name in selected}


def main():
    OUT.mkdir(exist_ok=False)
    if BASE.parent.exists():
        raise RuntimeError("Unique missing-parent reproduction directory already exists; refusing reuse")
    before = pins()
    command = [PYTHON, "-m", "pytest", "-q", "tests/test_config.py::test_an_unknown_top_level_key_is_rejected",
               "--basetemp", str(BASE), "--tb=long"]
    started, tick = datetime.now(UTC).isoformat(), time.perf_counter()
    with (OUT / "stdout.txt").open("wb") as stdout, (OUT / "stderr.txt").open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
            stdout=stdout, stderr=stderr, timeout=90, check=False)
    after = pins()
    audit = {"schema": "week6.full001.single_node_setup_reproduction.v1",
        "reviewer": "/root/week6_replanning_review", "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(), "wall_seconds": time.perf_counter() - tick,
        "command": command, "actual_returncode": result.returncode,
        "before_source_pins": before, "after_source_pins": after, "source_unchanged": before == after,
        "new_base_parent_existed_before": False, "new_base_parent_exists_after": BASE.parent.exists(),
        "artifacts": {name: digest(OUT / name) for name in ("stdout.txt", "stderr.txt")},
        "original_full001_paths_or_processes_modified": False,
        "scope": "One CPU node only. No config, source, test, script or production edits; no GUI/model/whole-suite rerun."}
    target = OUT / "audit.json"
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"actual_returncode": result.returncode, "source_unchanged": before == after,
                      "audit": str(target), "audit_sha256": digest(target)}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
