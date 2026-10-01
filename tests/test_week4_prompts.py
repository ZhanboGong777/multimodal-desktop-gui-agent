"""The prompt has to tell the model how to aim at something it can actually hit.

Week 3's prompt named no arguments and no element ids, so a plan could be valid
JSON and still be unresolvable against the frame. These checks pin the parts the
runtime depends on, and pin the length: a prompt that grows back past the point
where the 7B model truncates its own answer would break the loop again.

They also pin the *shape* of the user turn. A function that rendered a plain-text
turn lived here for a while and nothing called it - the planner goes through
`ModelRequest.to_messages` - so the tests were guarding a prompt no model ever
saw. The checks below go through the objects the planner actually uses.
"""

from __future__ import annotations

import json

from gui_agent.models import MockModelClient, ModelRequest
from gui_agent.planning import TaskPlanner
from gui_agent.planning.prompts import SYSTEM_PROMPT

#: The prompt that caused truncated replies measured 1 199 characters; the one
#: that worked measured 792. Stay near the working end.
MAX_PROMPT_CHARS = 1000


def test_the_prompt_fits_the_budget_that_keeps_the_model_on_task() -> None:
    assert len(SYSTEM_PROMPT) <= MAX_PROMPT_CHARS, (
        f"{len(SYSTEM_PROMPT)} characters; the 7B model rambled past the JSON at 1 199"
    )


def test_the_prompt_shows_how_to_target_an_element() -> None:
    assert "element_id" in SYSTEM_PROMPT
    assert "obs-0001-e" in SYSTEM_PROMPT, "the example must look like a real frame-local id"


def test_the_prompt_forbids_invented_coordinates() -> None:
    assert "never invent coordinates" in SYSTEM_PROMPT


def test_the_prompt_names_the_arguments_for_each_action() -> None:
    for token in ("type_text", '"key"', '"keys"', "scroll_amount", '"duration"'):
        assert token in SYSTEM_PROMPT, f"{token} is not described"


def test_the_prompt_requires_finish_and_says_when() -> None:
    assert '"finish"' in SYSTEM_PROMPT
    assert "LAST step" in SYSTEM_PROMPT


def test_the_prompt_points_at_this_platform_s_keys() -> None:
    assert "platform" in SYSTEM_PROMPT, "the model must not hard-code another OS's shortcuts"


def test_the_user_turn_is_one_json_object() -> None:
    """The shape a backend receives: instruction, context and image in one turn."""
    payload = json.loads(
        ModelRequest(
            instruction="open it",
            context={"platform": "darwin", "visible_text": "obs-0001-e000  'Browser'"},
            image_path="/tmp/shot.png",
        )
        .to_messages()[0]["content"]
    )

    assert payload["instruction"] == "open it"
    assert payload["context"]["platform"] == "darwin"
    assert payload["image_path"] == "/tmp/shot.png"


def test_nothing_is_omitted_when_there_is_nothing_to_say() -> None:
    """A run with no context or image sends the instruction and nothing else."""
    payload = json.loads(ModelRequest(instruction="go").to_messages()[0]["content"])

    assert payload == {"instruction": "go"}


def test_the_element_list_reaches_the_model_whole() -> None:
    """8.2.10: the observation is filtered structurally, never cut mid-field.

    There is no length truncation anywhere on this path - the list is capped by
    `execution.max_elements`, one whole element at a time.
    """
    elements = "\n".join(_element_line(index) for index in range(200))

    payload = json.loads(
        ModelRequest(instruction="go", context={"visible_text": elements})
        .to_messages()[0]["content"]
    )

    assert payload["context"]["visible_text"] == elements


def test_the_planner_sends_that_shape_and_not_a_rendered_template() -> None:
    """Through the planner's own call, so the two cannot drift apart again."""
    client = MockModelClient()
    planner = TaskPlanner(client, max_steps=7)

    planner.plan(
        "Open the browser",
        context={"visible_text": "obs-0001-e000  'Browser'"},
        image_path="/tmp/s.png",
        task_id="T01",
    )

    sent = client.calls[-1]["messages"]
    payload = json.loads(sent[-1]["content"])
    assert [message["role"] for message in sent] == ["system", "user"]
    assert payload["instruction"] == "Open the browser"
    assert payload["context"]["max_steps"] == 7, "8.2.9: the cap travels with the task"
    assert payload["image_path"] == "/tmp/s.png"


def _element_line(index: int) -> str:
    return (
        f"obs-0002-e{index:03d}  'Element {index} with a reasonably long label'  "
        f"conf=0.90  center=(100,{20 * index})  box=(10,10,400,40)"
    )
