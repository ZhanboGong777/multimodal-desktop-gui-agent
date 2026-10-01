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


# ───────── the mock reads the observation it is given ─────────
def _prompt_with_elements(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    lines = "\n".join(
        f"{element_id}  '{text}'  conf=0.90  center=(250,78)  box=(100,60,400,96)"
        for element_id, text in pairs
    )
    return [
        {"role": "system", "content": "You plan desktop GUI actions."},
        {"role": "user", "content": f"Instruction: Open the browser\n\nScreen context:\n{lines}"},
    ]


def test_the_mock_aims_at_an_element_that_is_really_on_screen() -> None:
    """A rule-based backend that ignores the frame names elements that do not exist.

    That is what made every dry run stop at step one: the adapter refused a target
    the mock had invented. Reading the supplied element list fixes the mock without
    pretending it understands the screen.
    """
    client = MockModelClient()
    response = client.generate_multimodal(
        "Open the browser",
        context={"visible_text": "obs-0001-e000  'Browser'  conf=0.90  center=(250,78)"},
    )
    plan = json.loads(response.content)
    click = next(step for step in plan["steps"] if step["action_type"] != "finish")
    assert click["arguments"]["element_id"] == "obs-0001-e000"
    assert click["target_text"] == "Browser"


def test_the_mock_prefers_the_element_the_clause_names() -> None:
    client = MockModelClient()
    rendered = "\n".join(
        f"{i}  '{t}'  conf=0.9  center=(1,2)"
        for i, t in (("obs-0001-e000", "Files"), ("obs-0001-e001", "Browser"))
    )
    response = client.generate_multimodal("Click Browser", context={"visible_text": rendered})
    plan = json.loads(response.content)
    assert plan["steps"][0]["arguments"]["element_id"] == "obs-0001-e001"


def test_the_mock_still_works_with_no_observation() -> None:
    """Nothing to aim at: fall back to the clause's words rather than inventing an id."""
    client = MockModelClient()
    plan = json.loads(client.generate_multimodal("Open the browser").content)
    assert "element_id" not in plan["steps"][0]["arguments"]
    assert plan["steps"][0]["target_text"]


def test_the_mock_records_what_it_was_given() -> None:
    client = MockModelClient()
    client.generate_multimodal(
        "x", context={"visible_text": "obs-0001-e000  'Browser'  conf=0.9  center=(1,2)"}
    )
    assert client.calls[-1]["elements"] == [("obs-0001-e000", "Browser")]


def test_the_mock_avoids_a_target_the_adapter_would_call_ambiguous() -> None:
    """A text contained in another element cannot resolve to a single target.

    Reproduced from the Windows review machine: the mock picked a word-level
    element ("the") that also occurred inside five other elements, so the adapter
    refused with `6 elements match 'the'` and the dry run stopped for a reason that
    had nothing to do with the pipeline.
    """
    client = MockModelClient()
    response = client.generate_multimodal(
        "the search",
        context={
            "visible_text": (
                "obs-0002-e000  'the'  conf=0.91  center=(1,1)\n"
                "obs-0002-e001  'Other'  conf=0.93  center=(2,2)\n"
                "obs-0002-e002  'the search box'  conf=0.94  center=(3,3)"
            )
        },
    )
    step = json.loads(response.content)["steps"][0]

    # The adapter looks a target up with a substring match, so the chosen text has
    # to occur in exactly one element. Which unambiguous element it lands on is the
    # mock's business; not creating an ambiguity is the point.
    chosen = step["target_text"]
    texts = ["the", "Other", "the search box"]
    assert sum(1 for text in texts if chosen in text) == 1
    assert step["arguments"]["element_id"] != "obs-0002-e000"


def test_the_mock_still_aims_somewhere_when_every_element_overlaps() -> None:
    """With nothing unambiguous the first element is as good as any other.

    A dry run that stops on a genuinely ambiguous screen is the correct outcome and
    the adapter's own tests cover it. What must not happen is a crash.
    """
    client = MockModelClient()
    response = client.generate_multimodal(
        "Open the browser",
        context={
            "visible_text": (
                "obs-0002-e000  'Open'  conf=0.91  center=(1,1)\n"
                "obs-0002-e001  'Open the browser'  conf=0.93  center=(2,2)"
            )
        },
    )
    step = json.loads(response.content)["steps"][0]
    assert step["arguments"]["element_id"].startswith("obs-0002-e")


def test_the_mock_prefers_a_readable_target_over_ocr_noise() -> None:
    """Aiming at "@" makes a dry run die on a token that is gone by the next frame.

    That says nothing about the pipeline, so a candidate with real letters in it
    wins. The noise is still used when there is nothing else on screen.
    """
    client = MockModelClient()
    response = client.generate_multimodal(
        "Open the browser",
        context={
            "visible_text": (
                "obs-0002-e000  '@'  conf=0.91  center=(1,1)\n"
                "obs-0002-e001  'HO'  conf=0.93  center=(2,2)\n"
                "obs-0002-e002  'Browser'  conf=0.88  center=(3,3)"
            )
        },
    )
    step = json.loads(response.content)["steps"][0]
    assert step["arguments"]["element_id"] == "obs-0002-e002"


def test_the_mock_still_uses_noise_when_that_is_all_there_is() -> None:
    client = MockModelClient()
    response = client.generate_multimodal(
        "Open it",
        context={"visible_text": "obs-0002-e000  '@'  conf=0.91  center=(1,1)"},
    )
    step = json.loads(response.content)["steps"][0]
    assert step["arguments"]["element_id"] == "obs-0002-e000"
