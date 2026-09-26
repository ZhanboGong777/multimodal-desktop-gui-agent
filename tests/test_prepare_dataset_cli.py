"""End-to-end tests for the dataset preparation CLI.

The CLI is what the Week 3 deliverable is judged on, so it is exercised as a real
process rather than by importing ``main``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "week3_prepare_dataset.py"

WEBARENA = {
    "tasks": [
        {"task_id": "1", "intent": "What is the top rated product?", "start_url": "http://a.test"},
        {"task_id": "2", "intent": "Post a message", "start_url": "http://b.test"},
        {"task_id": "3", "intent": "Find the cheapest flight", "start_url": "http://c.test"},
        {"task_id": "4", "no_intent": True},
    ]
}


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "webarena.json"
    path.write_text(json.dumps(WEBARENA, indent=2), encoding="utf-8")
    return path


def test_converts_and_reads_back(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = run_cli(
        "--dataset", "webarena", "--input", str(source), "--output", str(out), "--limit", "3"
    )
    assert result.returncode == 0, result.stderr
    assert "成功转换数: 3" in result.stdout
    assert "回读验证: 3 个样本通过 Schema 校验" in result.stdout

    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in lines] == ["1", "2", "3"]


def test_validate_only_writes_nothing(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = run_cli(
        "--dataset",
        "webarena",
        "--input",
        str(source),
        "--output",
        str(out),
        "--limit",
        "2",
        "--validate-only",
    )
    assert result.returncode == 0, result.stderr
    assert "--validate-only" in result.stdout
    assert not out.exists()


def test_a_record_without_an_intent_is_counted_not_fatal(source: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = run_cli(
        "--dataset", "webarena", "--input", str(source), "--output", str(out), "--limit", "9"
    )
    assert result.returncode == 0, result.stderr
    assert "成功转换数: 3" in result.stdout
    assert "跳过样本数: 1" in result.stdout
    assert "no intent" in result.stdout


def test_stats_json_is_written(source: Path, tmp_path: Path) -> None:
    stats = tmp_path / "stats.json"
    result = run_cli(
        "--dataset",
        "webarena",
        "--input",
        str(source),
        "--output",
        str(tmp_path / "out.jsonl"),
        # --limit counts converted samples, so it must be large enough to reach
        # the deliberately broken fourth record.
        "--limit",
        "9",
        "--stats-json",
        str(stats),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(stats.read_text(encoding="utf-8"))
    assert payload["valid"] == 3
    assert payload["invalid"] == 1


def test_unknown_dataset_is_rejected(source: Path, tmp_path: Path) -> None:
    result = run_cli(
        "--dataset", "nope", "--input", str(source), "--output", str(tmp_path / "o.jsonl")
    )
    assert result.returncode == 2
    assert "unknown dataset" in result.stderr


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    result = run_cli(
        "--dataset",
        "webarena",
        "--input",
        str(tmp_path / "absent.json"),
        "--output",
        str(tmp_path / "o.jsonl"),
    )
    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_output_is_required_unless_validating(source: Path) -> None:
    result = run_cli("--dataset", "webarena", "--input", str(source), "--limit", "1")
    assert result.returncode == 2
    assert "--output is required" in result.stderr


def test_zero_limit_is_rejected(source: Path, tmp_path: Path) -> None:
    result = run_cli(
        "--dataset",
        "webarena",
        "--input",
        str(source),
        "--output",
        str(tmp_path / "o.jsonl"),
        "--limit",
        "0",
    )
    assert result.returncode == 2
    assert "--limit" in result.stderr
