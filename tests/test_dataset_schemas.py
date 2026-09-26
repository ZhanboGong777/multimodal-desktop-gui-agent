"""Tests for the unified GUI dataset schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gui_agent.datasets import GUIActionStep, GUITaskSample
from gui_agent.schemas import BoundingBox, Point


def make_step(**overrides: object) -> GUIActionStep:
    payload: dict[str, object] = {"step_index": 0, "action_type": "click"}
    payload.update(overrides)
    return GUIActionStep(**payload)  # type: ignore[arg-type]


def make_sample(**overrides: object) -> GUITaskSample:
    payload: dict[str, object] = {
        "sample_id": "s1",
        "dataset_name": "screenagent",
        "instruction": "Open the browser",
    }
    payload.update(overrides)
    return GUITaskSample(**payload)  # type: ignore[arg-type]


def test_minimal_sample_is_valid() -> None:
    sample = make_sample()
    assert sample.actions == []
    assert sample.step_count == 0
    assert sample.image_path is None


@pytest.mark.parametrize("field", ["sample_id", "instruction", "dataset_name"])
def test_missing_required_text_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_sample(**{field: "   "})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_sample(unexpected="x")


def test_action_type_is_casefolded_and_trimmed() -> None:
    assert make_step(action_type="  CLICK  ").action_type == "click"


def test_empty_action_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_step(action_type="   ")


def test_unknown_action_keeps_the_original_payload() -> None:
    """An action verb this project does not implement must not break the sample."""
    raw = {"action": "SELECT", "element_id": 42}
    step = make_step(action_type="select", raw_action=raw)
    sample = make_sample(actions=[step])

    assert sample.actions[0].action_type == "select"
    assert sample.actions[0].raw_action == raw


def test_step_index_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        make_step(step_index=-1)


def test_geometry_uses_the_shared_types() -> None:
    step = make_step(
        target_bbox=BoundingBox(left=1, top=2, right=11, bottom=12),
        coordinates=Point(x=6, y=7),
    )
    assert step.target_bbox is not None and step.target_bbox.width == 10
    assert step.coordinates is not None and step.coordinates.x == 6


def test_action_type_counts_summarises_a_trajectory() -> None:
    sample = make_sample(
        actions=[
            make_step(step_index=0, action_type="click"),
            make_step(step_index=1, action_type="click"),
            make_step(step_index=2, action_type="type_text"),
        ]
    )
    assert sample.step_count == 3
    assert sample.action_type_counts() == {"click": 2, "type_text": 1}
