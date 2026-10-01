"""The prompt has to tell the model how to aim at something it can actually hit.

Week 3's prompt named no arguments and no element ids, so a plan could be valid
JSON and still be unresolvable against the frame. These checks pin the parts the
runtime depends on, and pin the length: a prompt that grows back past the point
where the 7B model truncates its own answer would break the loop again.
"""

from __future__ import annotations

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
