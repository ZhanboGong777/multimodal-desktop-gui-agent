"""Tests for the offline mock backend."""

from __future__ import annotations

import json

from gui_agent.models import MockModelClient
from gui_agent.models.base import ModelClient, ModelError, ModelResponse


def test_mock_returns_a_stable_plan() -> None:
    first = MockModelClient().generate_multimodal("Open the browser and search for GUI agents")
    second = MockModelClient().generate_multimodal("Open the browser and search for GUI agents")
    assert first.content == second.content
    assert first.ok


def test_mock_plan_shape() -> None:
    response = MockModelClient().generate_text("Open the browser, search for GUI agents")
    plan = json.loads(response.content)
    assert plan["instruction"] == "Open the browser, search for GUI agents"
    assert len(plan["steps"]) >= 2
    assert plan["steps"][-1]["action_type"] == "finish"
    assert all("step_id" in step for step in plan["steps"])


def test_mock_splits_a_compound_instruction() -> None:
    plan = json.loads(
        MockModelClient().generate_text("Open the browser then close the tab").content
    )
    descriptions = [step["description"] for step in plan["steps"]]
    assert "Open the browser" in descriptions[0]
    assert len(descriptions) >= 3  # two clauses plus finish


def test_mock_works_without_network_or_key() -> None:
    client = MockModelClient()
    assert client.health_check() is True
    assert client.calls


def test_mock_never_needs_an_api_key() -> None:
    response = MockModelClient(temperature=0.0, max_retries=0).generate_text("do nothing")
    assert response.error is None
    assert response.provider == "mock"


def test_response_as_dict_hides_the_raw_payload() -> None:
    response = ModelResponse(
        content="secret-ish",
        model_name="m",
        provider="mock",
        raw_response={"api_key": "should-not-be-logged"},
    )
    dumped = json.dumps(response.as_dict())
    assert "should-not-be-logged" not in dumped
    assert "content_length" in dumped


def test_ok_is_false_for_an_empty_response() -> None:
    assert ModelResponse(content="  ", model_name="m", provider="p").ok is False
    assert ModelResponse(content="x", model_name="m", provider="p", error="boom").ok is False


class _AlwaysFails(ModelClient):
    name = "failing"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(model_name="failing", **kwargs)  # type: ignore[arg-type]
        self.attempts = 0

    def complete(self, messages: object, **kwargs: object) -> ModelResponse:
        self.attempts += 1
        raise ModelError("backend is down")


def test_retries_are_bounded_and_reported() -> None:
    client = _AlwaysFails(max_retries=2)  # type: ignore[arg-type]
    response = client.generate_text("hello")
    assert client.attempts == 3  # one initial attempt plus two retries
    assert response.ok is False
    assert "backend is down" in (response.error or "")


def test_health_check_never_raises() -> None:
    assert _AlwaysFails().health_check() is False  # type: ignore[arg-type]


def test_generate_multimodal_carries_the_image_path() -> None:
    client = MockModelClient()
    client.generate_multimodal("Close the window", image_path="shots/a.png")
    payload = json.loads(client.calls[-1]["messages"][0]["content"])
    assert payload["image_path"] == "shots/a.png"
    assert payload["instruction"] == "Close the window"
