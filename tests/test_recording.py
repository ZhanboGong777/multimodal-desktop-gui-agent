"""Tests for the minimal run recording."""

from __future__ import annotations

import json
from pathlib import Path

from gui_agent.recording import (
    ACTION_FILE,
    PERCEPTION_FILE,
    SUMMARY_FILE,
    RunSession,
    build_run_summary,
    elements_to_payload,
)
from gui_agent.schemas import (
    ActionResult,
    BoundingBox,
    DesktopAction,
    PerceptionResult,
    ScreenInfo,
    UIElement,
)


def make_screen() -> ScreenInfo:
    return ScreenInfo(
        screenshot_width=100, screenshot_height=50, control_width=100, control_height=50
    )


def make_element() -> UIElement:
    return UIElement(
        text="OK",
        bounding_box=BoundingBox(left=1, top=2, right=11, bottom=12),
        confidence=0.9,
    )


def test_session_creates_its_directory(tmp_path: Path) -> None:
    session = RunSession.create(tmp_path, "run1")
    assert session.directory == tmp_path / "run1"
    assert session.directory.is_dir()
    assert session.session_id == "run1"


def test_session_generates_a_timestamped_identifier(tmp_path: Path) -> None:
    session = RunSession.create(tmp_path)
    assert session.session_id
    assert session.directory.is_dir()


def test_save_json_writes_utf8(tmp_path: Path) -> None:
    session = RunSession.create(tmp_path, "run")
    path = session.save_json("x.json", {"中文": "值"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"中文": "值"}


def test_perception_round_trips_as_json(tmp_path: Path) -> None:
    session = RunSession.create(tmp_path, "run")
    result = PerceptionResult(
        screen_info=make_screen(), elements=[make_element()], image_path="before.png"
    )
    path = session.save_perception(result)
    assert path.name == PERCEPTION_FILE

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["elements"][0]["text"] == "OK"
    assert payload["screen_info"]["screenshot_width"] == 100
    assert payload["image_path"] == "before.png"


def test_action_and_result_are_saved_together(tmp_path: Path) -> None:
    session = RunSession.create(tmp_path, "run")
    action = DesktopAction(action_type="click", x=1, y=2)
    result = ActionResult(success=True, dry_run=True, action=action)
    path = session.save_action(action, result)
    assert path.name == ACTION_FILE

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["action"]["action_type"] == "click"
    assert payload["result"]["dry_run"] is True


def test_summary_contains_every_required_field(tmp_path: Path) -> None:
    action = DesktopAction(action_type="move", x=1, y=2, target_description="OK")
    result = ActionResult(success=True, dry_run=True)
    summary = build_run_summary(
        selected_target="OK",
        action=action,
        result=result,
        capture_time_ms=1.5,
        ocr_time_ms=2.5,
        total_time_ms=4.0,
        element_count=3,
    )
    for key in (
        "timestamp",
        "platform",
        "selected_target",
        "action",
        "dry_run",
        "execution_result",
        "capture_time_ms",
        "ocr_time_ms",
        "total_time_ms",
        "error",
    ):
        assert key in summary
    assert summary["dry_run"] is True
    assert summary["capture_time_ms"] == 1.5
    assert summary["element_count"] == 3

    path = RunSession.create(tmp_path, "run").save_summary(summary)
    assert path.name == SUMMARY_FILE
    assert json.loads(path.read_text(encoding="utf-8"))["selected_target"] == "OK"


def test_summary_accepts_a_run_without_an_action() -> None:
    summary = build_run_summary(
        selected_target=None,
        action=None,
        result=None,
        capture_time_ms=1.0,
        ocr_time_ms=2.0,
        total_time_ms=3.0,
        error="target not found",
    )
    assert summary["action"] is None
    assert summary["dry_run"] is None
    assert summary["execution_result"] is None
    assert summary["error"] == "target not found"


def test_summary_merges_extra_fields() -> None:
    summary = build_run_summary(
        selected_target="x",
        action=None,
        result=None,
        capture_time_ms=0.0,
        ocr_time_ms=0.0,
        total_time_ms=0.0,
        extra={"candidate_count": 2},
    )
    assert summary["candidate_count"] == 2


def test_elements_to_payload_is_json_serialisable() -> None:
    payload = elements_to_payload([make_element()])
    assert payload[0]["source"] == "ocr"
    assert json.loads(json.dumps(payload))[0]["text"] == "OK"


def test_log_file_is_written(tmp_path: Path) -> None:
    session = RunSession.create(tmp_path, "run")
    path = session.save_log(["first", "second"])
    assert path.read_text(encoding="utf-8") == "first\nsecond\n"
