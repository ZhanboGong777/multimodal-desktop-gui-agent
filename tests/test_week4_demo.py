"""The runnable demonstration, run.

16.3 asks for a closed-loop demonstration and the hand-back manual's step 3 is
`python scripts/week4_offline_demo.py`, with a stated expectation: four actions
dispatched, `succeeded`, `verification: passed`. Nothing ran it. The suite covered
the runner's loop with its own scripted frames, so when real runs began stopping on
a step whose expected result was not observed, the demo stopped at its own second
step - its scripted screens did not contain the words its scripted plan expected -
and no test failed. The manual would have told the reviewer to expect a success and
shown them a failure.

So this file runs the script and checks what the operator is promised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "week4_offline_demo.py"


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output-directory", str(tmp_path), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def test_the_demo_completes_the_loop_the_manual_promises(tmp_path: Path) -> None:
    """Step 3's stated expectation, asserted rather than hoped for."""
    result = _run(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status      : succeeded" in result.stdout
    assert "actions     : 4 dispatched" in result.stdout
    assert "verification: passed" in result.stdout


def test_the_demo_leaves_the_records_a_finished_run_leaves(tmp_path: Path) -> None:
    """A demonstration that leaves no evidence is a claim, not a demonstration."""
    _run(tmp_path)

    sessions = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(sessions) == 1, sessions
    summary = json.loads((sessions[0] / "task_summary.json").read_text(encoding="utf-8"))

    assert summary["status"] == "succeeded"
    assert summary["verification"]["outcome"] == "passed"
    assert summary["action_count"] == 4
    assert summary["steps"], "one record per step"
    assert (sessions[0] / "steps.jsonl").is_file()


def test_the_demo_can_be_made_to_fail_on_purpose(tmp_path: Path) -> None:
    """`--fail-at` is how the stop-on-failure path is demonstrated rather than described."""
    result = _run(tmp_path, "--fail-at", "2")

    assert "status      : failed" in result.stdout
    assert "action failed" in result.stdout
    # The third action never runs: a failed step ends the run.
    assert "key_press" not in result.stdout
