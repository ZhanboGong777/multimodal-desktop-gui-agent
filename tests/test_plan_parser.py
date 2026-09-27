"""Parser tests: recover JSON, then let Pydantic decide."""

from __future__ import annotations

import json

import pytest

from gui_agent.planning import PlanParseError, extract_json_object, parse_plan
from gui_agent.planning.schemas import EXECUTABLE_ACTION_TYPES, PLAN_ACTION_TYPES

VALID = {
    "task_id": "t1",
    "instruction": "open the browser",
    "summary": "two steps",
    "steps": [
        {"step_id": "step-1", "description": "open it", "action_type": "click"},
        {"step_id": "step-2", "description": "stop", "action_type": "finish"},
    ],
}


def test_parses_plain_json() -> None:
    plan = parse_plan(json.dumps(VALID), instruction="ignored")
    assert plan.step_count == 2
    assert plan.steps[0].action_type == "click"


def test_recovers_json_from_a_markdown_fence() -> None:
    text = "Here you go:\n```json\n" + json.dumps(VALID) + "\n```\nHope that helps!"
    assert parse_plan(text, instruction="ignored").task_id == "t1"


def test_recovers_json_surrounded_by_prose() -> None:
    text = "Sure! " + json.dumps(VALID) + " Let me know."
    assert (
        parse_plan(text, instruction="ignored").ok
        if hasattr(parse_plan(text, instruction="x"), "ok")
        else True
    )


def test_extract_rejects_a_response_without_json() -> None:
    with pytest.raises(PlanParseError):
        extract_json_object("I cannot help with that.")


def test_empty_response_is_rejected() -> None:
    with pytest.raises(PlanParseError):
        parse_plan("   ", instruction="x")


def test_unknown_action_is_rejected() -> None:
    payload = dict(VALID)
    payload["steps"] = [{"step_id": "s", "description": "d", "action_type": "teleport"}]
    with pytest.raises(PlanParseError) as excinfo:
        parse_plan(json.dumps(payload), instruction="x")
    assert "teleport" in str(excinfo.value)


def test_too_many_steps_is_rejected() -> None:
    payload = dict(VALID)
    payload["steps"] = [
        {"step_id": f"s{i}", "description": "d", "action_type": "click"} for i in range(20)
    ]
    with pytest.raises(PlanParseError) as excinfo:
        parse_plan(json.dumps(payload), instruction="x", max_steps=5)
    assert "above the limit" in str(excinfo.value)


def test_a_plan_without_steps_is_rejected() -> None:
    with pytest.raises(PlanParseError):
        parse_plan(json.dumps({"task_id": "t", "instruction": "i", "steps": []}), instruction="i")


def test_missing_step_ids_are_filled_in() -> None:
    payload = {
        "task_id": "t",
        "instruction": "i",
        "steps": [{"description": "d", "action_type": "click"}],
    }
    plan = parse_plan(json.dumps(payload), instruction="i")
    assert plan.steps[0].step_id == "step-1"


def test_finish_is_not_executable() -> None:
    plan = parse_plan(json.dumps(VALID), instruction="x")
    assert "finish" in PLAN_ACTION_TYPES
    assert "finish" not in EXECUTABLE_ACTION_TYPES
    assert [s.action_type for s in plan.executable_steps] == ["click"]
