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


# ─────────────────── Parquet input (the Mind2Web case) ───────────────────
def _write_parquet(path: Path, records: list[dict[str, object]]) -> None:
    """Smallest real Parquet file, so the reader is tested on the real format."""
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pyarrow.table(
        {
            "annotation_id": [r["annotation_id"] for r in records],
            "confirmed_task": [r["confirmed_task"] for r in records],
            "action_reprs": [r["action_reprs"] for r in records],
            "target_action_index": [r["target_action_index"] for r in records],
        }
    )
    pq.write_table(table, path)


def test_reads_a_parquet_shard(tmp_path: Path) -> None:
    """Mind2Web ships as Parquet, which is binary.

    Reading it as text produced a few lines of mojibake, converted nothing, wrote an
    empty file and still exited zero - so the documented Mind2Web command appeared to
    work. This is the regression test for that.
    """
    from gui_agent.datasets.base import read_records

    shard = tmp_path / "shard.parquet"
    _write_parquet(
        shard,
        [
            {
                "annotation_id": "m2w-1",
                "confirmed_task": "Rent a truck",
                "action_reprs": ["[link]  Budget Truck -> CLICK"],
                "target_action_index": "0",
            }
        ],
    )

    records = list(read_records(shard))

    assert len(records) == 1
    assert records[0]["confirmed_task"] == "Rent a truck"


def test_parquet_flows_through_the_cli(tmp_path: Path) -> None:
    shard = tmp_path / "shard.parquet"
    _write_parquet(
        shard,
        [
            {
                "annotation_id": "m2w-1",
                "confirmed_task": "Rent a truck",
                "action_reprs": ["[link]  Budget Truck -> CLICK"],
                "target_action_index": "0",
            }
        ],
    )
    out = tmp_path / "out.jsonl"

    result = run_cli("--dataset", "mind2web", "--input", str(shard), "--output", str(out))

    assert result.returncode == 0, result.stderr
    assert "成功转换数: 1" in result.stdout
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_a_run_that_converts_nothing_fails(tmp_path: Path) -> None:
    """Exit zero on an empty export reads as success to whoever ran the command."""
    source = tmp_path / "wrong.json"
    source.write_text(json.dumps({"unrelated": True}), encoding="utf-8")
    out = tmp_path / "out.jsonl"

    result = run_cli("--dataset", "mind2web", "--input", str(source), "--output", str(out))

    assert result.returncode == 1
    assert "no samples were written" in result.stderr
