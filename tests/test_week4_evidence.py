"""Evidence has to travel with the repository.

`outputs/` is not tracked, so a run id quoted in the test report resolves to
nothing once the report is read on another machine. These tests fix what "collect
the evidence" copies - and, more importantly, what it refuses to copy.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "week4_collect_evidence.py"


def _load():
    spec = importlib.util.spec_from_file_location("week4_collect_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = _load()


def _session(root: Path, name: str = "T01_20261001_130401", **extra: str) -> Path:
    session = root / name
    session.mkdir(parents=True, exist_ok=True)
    (session / "task_summary.json").write_text(
        json.dumps({"run_id": name, "status": "succeeded", "execute": True}), encoding="utf-8"
    )
    (session / "steps.jsonl").write_text('{"step_id": "s1"}\n', encoding="utf-8")
    (session / "obs-0001.json").write_text('{"elements": []}', encoding="utf-8")
    (session / "monitor1_20261001_130401.png").write_bytes(b"\x89PNG fake")
    for filename, content in extra.items():
        (session / filename).write_text(content, encoding="utf-8")
    return session


def test_collect_copies_the_summary_and_the_step_log(tmp_path: Path) -> None:
    session = _session(tmp_path / "outputs")
    destination = tmp_path / "evidence"

    copied = evidence.collect(session, destination)

    assert sorted(path.name for path in copied) == ["steps.jsonl", "task_summary.json"]
    assert (destination / session.name / "task_summary.json").is_file()


def test_collect_leaves_the_screen_behind(tmp_path: Path) -> None:
    """Screenshots and per-frame observations are pictures of the whole desktop.

    They are the one thing that must not end up in a public repository, so the
    collector copies named text records rather than the session directory.
    """
    session = _session(tmp_path / "outputs")
    destination = tmp_path / "evidence"

    evidence.collect(session, destination)

    written = {path.name for path in (destination / session.name).iterdir()}
    assert written == {"task_summary.json", "steps.jsonl"}
    assert not any(name.endswith(".png") for name in written)
    assert not any(name.startswith("obs-") for name in written)


def test_collect_can_skip_the_step_log(tmp_path: Path) -> None:
    session = _session(tmp_path / "outputs")
    destination = tmp_path / "evidence"

    copied = evidence.collect(session, destination, include_steps=False)

    assert [path.name for path in copied] == ["task_summary.json"]


def test_a_session_without_a_summary_is_not_evidence(tmp_path: Path) -> None:
    """A run with no summary did not finish, and half a run proves nothing."""
    session = tmp_path / "T01_unfinished"
    session.mkdir()
    with pytest.raises(evidence.EvidenceError, match="did not finish"):
        evidence.collect(session, tmp_path / "evidence")


def test_a_missing_session_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(evidence.EvidenceError, match="not a directory"):
        evidence.collect(tmp_path / "nope", tmp_path / "evidence")


def test_latest_session_picks_the_newest_for_that_case(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _session(root, "T01_20260101_000000")
    _session(root, "T02_20260101_000000")
    newest = _session(root, "T01_20260202_000000")

    assert evidence.latest_session("T01", root) == newest
    assert evidence.latest_session("t01", root) == newest


def test_latest_session_says_so_when_there_is_nothing(tmp_path: Path) -> None:
    with pytest.raises(evidence.EvidenceError, match="no T03 session"):
        evidence.latest_session("T03", tmp_path)


def test_run_id_is_read_from_the_summary(tmp_path: Path) -> None:
    session = _session(tmp_path / "outputs")
    assert evidence.run_id_of(session) == session.name
    assert evidence.run_id_of(tmp_path) == ""


def test_the_command_line_collects_and_prints_the_run_id(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _session(root, "T01_20260202_000000")
    destination = tmp_path / "evidence"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--latest",
            "T01",
            "--session-root",
            str(root),
            "--destination",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "T01_20260202_000000" in result.stdout
    assert (destination / "T01_20260202_000000" / "task_summary.json").is_file()


def test_the_command_line_reports_nothing_to_collect(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--latest",
            "T04",
            "--session-root",
            str(tmp_path),
            "--destination",
            str(tmp_path / "evidence"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 2
    assert "no T04 session" in result.stderr
