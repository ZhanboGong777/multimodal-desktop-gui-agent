"""The prompt has to tell the model how to aim at something it can actually hit.

Week 3's prompt named no arguments and no element ids, so a plan could be valid
JSON and still be unresolvable against the frame. These checks pin the parts the
runtime depends on, and pin the length: a prompt that grows back past the point
where the 7B model truncates its own answer would break the loop again.
"""

from __future__ import annotations

import re

from gui_agent.planning.prompts import SYSTEM_PROMPT, build_user_prompt

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


def test_the_user_turn_carries_the_platform_when_it_is_known() -> None:
    rendered = build_user_prompt("open it", context={"platform": "darwin"}, max_steps=10)
    assert "Platform: darwin" in rendered


def test_the_user_turn_still_works_without_a_platform() -> None:
    assert "Platform:" not in build_user_prompt("open it", max_steps=5)


def _element_line(index: int) -> str:
    return (
        f"obs-0002-e{index:03d}  'Element {index} with a reasonably long label'  "
        f"conf=0.90  center=(100,{20 * index})  box=(10,10,400,40)"
    )


def test_the_element_list_reaches_the_model_whole() -> None:
    line = _element_line(7)
    rendered = build_user_prompt("do it", context={"visible_text": line})
    assert line in rendered


def test_a_long_element_list_is_trimmed_by_whole_lines() -> None:
    """8.2.10 forbids cutting the serialized observation mid-field.

    An element id with half its text is unusable in both directions: the model
    cannot copy a label it cannot read, and the adapter cannot match one that was
    chopped. So the list is trimmed a whole element at a time, and the prompt says
    how many were left out.
    """
    elements = "\n".join(_element_line(index) for index in range(200))
    rendered = build_user_prompt("do it", context={"visible_text": elements})
    block = rendered.split("Screen elements:\n", 1)[1]

    assert len(block) < 4400, "the element block is not bounded"
    assert "further elements omitted" in block

    for line in block.splitlines():
        if line.startswith("..."):
            continue
        assert re.fullmatch(r"obs-\d+-e\d+\s+'[^']*'\s+conf=.*", line), line


def test_a_short_element_list_is_passed_through_whole() -> None:
    elements = "\n".join(_element_line(index) for index in range(3))
    rendered = build_user_prompt("do it", context={"visible_text": elements})

    for index in range(3):
        assert f"obs-0002-e{index:03d}" in rendered
    assert "omitted" not in rendered


def test_only_the_descriptive_fields_are_length_limited() -> None:
    """8.2.3: typed text, file paths and test markers are preserved verbatim.

    A blanket "every string under 60 characters" rule invites the model to
    abbreviate the very marker the task is verified against.
    """
    assert "Keep every string under 60 characters" not in SYSTEM_PROMPT
    assert "description and summary under 60 chars" in SYSTEM_PROMPT

    marker = "WEEK4-OPEN-FILE-OK-with-a-deliberately-long-tail-so-it-exceeds-sixty"
    rendered = build_user_prompt(f"open {marker}", context={"platform": "win32"})
    assert marker in rendered
    assert len(marker) > 60
