"""Planner tests: the mock backend makes the whole path deterministic."""

from __future__ import annotations

import json

from gui_agent.models import MockModelClient
from gui_agent.models.base import ModelClient, ModelError, ModelResponse
from gui_agent.planning import TaskPlanner


def make_planner(**kwargs: object) -> TaskPlanner:
    return TaskPlanner(MockModelClient(), **kwargs)  # type: ignore[arg-type]


def test_plan_is_produced_and_validated() -> None:
    result = make_planner().plan("Open the browser and search for GUI agents")
    assert result.ok
    assert result.plan is not None
    assert result.plan.instruction == "Open the browser and search for GUI agents"
    assert result.plan.steps[-1].action_type == "finish"


def test_planning_is_deterministic() -> None:
    first = make_planner().plan("Close the settings window")
    second = make_planner().plan("Close the settings window")
    assert first.plan is not None and second.plan is not None
    assert first.plan.model_dump() == second.plan.model_dump()


def test_context_and_image_are_accepted() -> None:
    result = make_planner().plan(
        "Click the save button",
        context={"elements": [{"text": "Save"}]},
        image_path="shots/a.png",
    )
    assert result.ok
    assert result.metadata["provider"] == "mock"


def test_planning_never_executes_anything() -> None:
    """The result records the policy and nothing in the module touches control."""
    result = make_planner(allow_real_execution=False).plan("Open the browser")
    assert result.metadata["allow_real_execution"] is False
    assert result.ok


def test_max_steps_from_the_config_is_enforced() -> None:
    result = make_planner(max_steps=1).plan("Open the browser then close the tab")
    assert not result.ok
    assert "above the limit" in (result.error or "")


class _BadJson(ModelClient):
    name = "bad-json"

    def __init__(self, reply: str, **kwargs: object) -> None:
        super().__init__(model_name="bad")  # type: ignore[arg-type]
        self.reply = reply
        self.calls = 0

    def complete(self, messages: object, **kwargs: object) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self.reply, model_name="bad", provider=self.name)


class _Dead(ModelClient):
    name = "dead"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(model_name="dead")  # type: ignore[arg-type]

    def complete(self, messages: object, **kwargs: object) -> ModelResponse:
        raise ModelError("connection refused")


def test_unparseable_output_is_retried_then_reported() -> None:
    client = _BadJson("not json at all")
    result = TaskPlanner(client, max_format_retries=1).plan("do something")
    assert not result.ok
    assert client.calls == 2  # one initial attempt plus one controlled retry
    assert "no JSON object" in (result.error or "")


def test_a_dead_backend_returns_an_error_result() -> None:
    result = TaskPlanner(_Dead()).plan("do something")
    assert not result.ok
    assert "connection refused" in (result.error or "")


def test_result_as_dict_is_json_serialisable() -> None:
    json.dumps(make_planner().plan("Open the browser").as_dict())
