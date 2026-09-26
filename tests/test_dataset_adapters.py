"""Adapter tests. Every adapter is exercised on hand-written fixtures: to_sample
is pure, so no download and no network are involved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui_agent.datasets import (
    Mind2WebAdapter,
    ScreenAgentAdapter,
    WebArenaAdapter,
    get_adapter,
    read_records,
)
from gui_agent.datasets.base import DatasetError
from gui_agent.datasets.normalize import as_int, clean_text, normalize_action


# ───────────────────────── normalize ─────────────────────────
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CLICK", "click"),
        ("  left_click ", "click"),
        ("TYPE", "type_text"),
        ("select_option", "click"),
        ("HOVER", "move"),
        ("scroll_down", "scroll"),
        ("DONE", "finish"),
        ("", "unknown"),
        (None, "unknown"),
        ("SomeWeird-VERB", "someweird_verb"),
    ],
)
def test_normalize_action(raw: object, expected: str) -> None:
    assert normalize_action(raw) == expected


def test_clean_text_and_as_int() -> None:
    assert clean_text("  a \n b  ") == "a b"
    assert clean_text(None) == ""
    assert clean_text(12) == "12"
    assert as_int("3.0") == 3
    assert as_int("x") is None
    assert as_int(True) is None


# ───────────────────────── ScreenAgent ─────────────────────────
def test_screenagent_maps_a_real_record() -> None:
    """Shape taken from data/ScreenAgent/test.zip, which was inspected directly."""
    sample = ScreenAgentAdapter().to_sample(
        {
            "task_prompt": "Change font to Noto Sans",
            "current_task": "Click 'Enter' to apply.",
            "saved_image_name": "2024-01-12_16-50-31-036068.jpg",
            "video_width": 1024,
            "video_height": 768,
            "status": "evaluating",
            "actions": [
                {
                    "action_type": "MouseAction",
                    "mouse_action_type": "left_click",
                    "mouse_button": "left",
                    "mouse_position": [512, 384],
                    "clickable_area": [500, 370, 540, 400],
                },
                {
                    "action_type": "KeyboardAction",
                    "keyboard_action_type": "type",
                    "keyboard_text": "Noto Sans",
                },
                {
                    "action_type": "EvaluateSubTaskAction",
                    "situation": "sub_task_success",
                },
            ],
        },
        split="test",
    )

    assert sample.instruction == "Change font to Noto Sans"
    assert sample.image_path == "2024-01-12_16-50-31-036068.jpg"
    assert sample.observation == "Click 'Enter' to apply."
    assert sample.source_split == "test"
    assert sample.metadata["video_width"] == 1024
    assert sample.metadata["status"] == "evaluating"

    # The evaluation is not a step: counting it would inflate every trajectory.
    assert sample.step_count == 2
    assert len(sample.metadata["evaluations"]) == 1
    assert sample.metadata["evaluations"][0]["situation"] == "sub_task_success"

    click = sample.actions[0]
    assert click.action_type == "click"
    assert click.coordinates is not None
    assert (click.coordinates.x, click.coordinates.y) == (512, 384)
    assert click.target_bbox is not None and click.target_bbox.width == 40

    typing = sample.actions[1]
    assert typing.action_type == "type_text"
    assert typing.input_text == "Noto Sans"


def test_screenagent_keeps_an_unseen_action_verb() -> None:
    """Unknown verbs survive rather than failing the sample."""
    sample = ScreenAgentAdapter().to_sample(
        {
            "task_prompt": "Do something unusual",
            "actions": [{"action_type": "WaitAction"}],
        }
    )
    assert sample.actions[0].action_type == "wait"
    assert sample.actions[0].raw_action["action_type"] == "WaitAction"


def test_screenagent_still_accepts_a_record_without_actions() -> None:
    sample = ScreenAgentAdapter().to_sample(
        {"task_prompt": "Open the browser", "saved_image_name": "a.png"}
    )
    assert sample.instruction == "Open the browser"
    assert sample.image_path == "a.png"
    assert sample.step_count == 0
    assert sample.sample_id == "a"


def test_screenagent_rejects_a_record_without_task_prompt() -> None:
    with pytest.raises(ValueError):
        ScreenAgentAdapter().to_sample({"id": "x", "actions": []})


# ───────────────────────── Mind2Web ─────────────────────────
def test_mind2web_maps_steps_and_website() -> None:
    sample = Mind2WebAdapter().to_sample(
        {
            "annotation_id": "m2w-1",
            "confirmed_task": "Find a hotel in Paris",
            "website": "booking.com",
            "actions": [
                {"operation": "CLICK", "element": {"text": "Hotels"}, "value": ""},
                {"operation": "TYPE", "value": "Paris"},
            ],
        }
    )
    assert sample.sample_id == "m2w-1"
    assert sample.metadata["website"] == "booking.com"
    assert sample.step_count == 2
    assert sample.actions[0].target_text == "Hotels"
    assert sample.actions[1].input_text == "Paris"


def test_mind2web_accepts_string_action_reprs() -> None:
    sample = Mind2WebAdapter().to_sample(
        {"confirmed_task": "Do something", "action_reprs": ["CLICK [Submit]", "TYPE [Paris]"]}
    )
    assert [s.action_type for s in sample.actions] == ["click", "type_text"]
    assert sample.actions[0].target_text == "Submit"


# ───────────────────────── WebArena ─────────────────────────
def test_webarena_keeps_intent_and_start_url() -> None:
    sample = WebArenaAdapter().to_sample(
        {
            "task_id": 42,
            "intent": "What is the top rated product?",
            "start_url": "http://shop.test",
            "sites": ["shopping"],
            "eval": {"reference_answers": {"exact_match": "Widget"}},
        }
    )
    assert sample.sample_id == "42"
    assert sample.actions == []
    assert sample.metadata["start_url"] == "http://shop.test"
    assert sample.metadata["sites"] == ["shopping"]
    assert sample.metadata["eval"] is not None


def test_webarena_rejects_a_record_without_intent() -> None:
    with pytest.raises(ValueError):
        WebArenaAdapter().to_sample({"task_id": 1})


# ───────────────────────── factory + reader ─────────────────────────
def test_get_adapter_returns_each_adapter() -> None:
    for name in ("screenagent", "mind2web", "webarena"):
        assert get_adapter(name).name == name


def test_get_adapter_rejects_an_unknown_name() -> None:
    with pytest.raises(DatasetError):
        get_adapter("nope")


def test_read_records_handles_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in read_records(path)] == [1, 2]


def test_read_records_handles_a_wrapped_json_list(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"tasks": [{"a": 1}, {"a": 2}]}), encoding="utf-8")
    assert len(list(read_records(path))) == 2


def test_read_records_missing_file() -> None:
    with pytest.raises(DatasetError):
        list(read_records("/nonexistent/path.json"))


def test_read_records_handles_pretty_printed_json(tmp_path: Path) -> None:
    """Regression: a formatted JSON file is not JSONL.

    An earlier reader switched to line-by-line parsing whenever the text held a
    newline, so every inner line of a pretty-printed document carried a trailing
    comma and was silently dropped.
    """
    path = tmp_path / "pretty.json"
    path.write_text(
        json.dumps({"tasks": [{"a": 1}, {"a": 2}, {"a": 3}]}, indent=2),
        encoding="utf-8",
    )
    assert [r["a"] for r in read_records(path)] == [1, 2, 3]


def test_read_records_handles_a_bare_json_list(tmp_path: Path) -> None:
    path = tmp_path / "bare.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}], indent=4), encoding="utf-8")
    assert len(list(read_records(path))) == 2
