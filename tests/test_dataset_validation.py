"""Tests for validation, statistics and the JSONL round trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui_agent.datasets.schemas import GUIActionStep, GUITaskSample
from gui_agent.datasets.validation import (
    DatasetExportError,
    collect_stats,
    read_jsonl,
    validate_sample,
    write_jsonl,
)


def make_sample(sample_id: str = "s1", **overrides: object) -> GUITaskSample:
    payload: dict[str, object] = {
        "sample_id": sample_id,
        "dataset_name": "screenagent",
        "instruction": "Open the browser",
    }
    payload.update(overrides)
    return GUITaskSample(**payload)  # type: ignore[arg-type]


def test_a_minimal_sample_is_valid() -> None:
    assert validate_sample(make_sample()) == []


def test_a_sample_without_actions_is_still_valid() -> None:
    """WebArena ships task definitions with no trajectory."""
    assert validate_sample(make_sample(actions=[])) == []


def test_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    samples = [
        make_sample("a", actions=[GUIActionStep(step_index=0, action_type="click")]),
        make_sample("b"),
    ]
    assert write_jsonl(samples, path) == 2

    back = read_jsonl(path)
    assert [s.sample_id for s in back] == ["a", "b"]
    assert back[0].actions[0].action_type == "click"


def test_each_line_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_jsonl([make_sample("a"), make_sample("b")], path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["dataset_name"] == "screenagent"


def test_read_jsonl_rejects_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"sample_id": "a"}\n', encoding="utf-8")  # missing fields
    with pytest.raises(DatasetExportError):
        read_jsonl(path)


def test_read_jsonl_missing_file() -> None:
    with pytest.raises(DatasetExportError):
        read_jsonl("/nonexistent/export.jsonl")


def test_collect_stats_counts_valid_invalid_and_actions() -> None:
    stats = collect_stats(
        [
            make_sample("a", actions=[GUIActionStep(step_index=0, action_type="click")]),
            make_sample("b", actions=[GUIActionStep(step_index=0, action_type="click")]),
            make_sample("c", actions=[GUIActionStep(step_index=0, action_type="type_text")]),
        ]
    )
    assert (stats.total, stats.valid, stats.invalid) == (3, 3, 0)
    assert stats.action_types == {"click": 2, "type_text": 1}
    assert stats.skip_ratio == 0.0


def test_collect_stats_survives_an_invalid_sample() -> None:
    good = make_sample("ok")
    broken = make_sample("bad")
    object.__setattr__(broken, "instruction", "")  # simulate a bad adapter output
    stats = collect_stats([good, broken])

    assert (stats.total, stats.valid, stats.invalid) == (2, 1, 1)
    assert any("instruction" in key for key in stats.missing_fields)
    assert "成功转换数: 1" in stats.report()
    assert stats.skip_ratio == 0.5


def test_stats_dict_is_json_serialisable() -> None:
    stats = collect_stats([make_sample()])
    json.dumps(stats.as_dict())
