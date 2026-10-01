"""The five case definitions are the experiment's control conditions.

Each case has to carry a rule the verifier can evaluate and a starting state the
runner can check. A case missing either one is not "not yet measured" - it is
unrunnable, or worse, winnable without doing anything, and the report would quote
a number that means nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime

from gui_agent.runtime.tasks import (
    CASES,
    MESSAGE_MARKER,
    SAMPLE_FILE,
    case_ids,
    get_case,
    get_case_for_run,
    new_message_marker,
)


def test_the_five_cases_are_present() -> None:
    assert case_ids() == ["T01", "T02", "T03", "T04", "T05"]


def test_every_case_has_a_rule_the_verifier_can_evaluate() -> None:
    """`check_task` reads expect_text and forbid_text, nothing else.

    The prose in `success_rules` is the standard a human applies; it is not
    machine-checkable. A case with prose and no text rule is reported inconclusive
    and the runner refuses to start it, so it would never produce a result at all.
    """
    for case_id, case in CASES.items():
        assert case.expect_text or case.forbid_text, f"{case_id} has no checkable rule"


def test_every_case_declares_a_precondition() -> None:
    """The runner only checks a starting state when the task declares one.

    T01 and T05 are both satisfiable by doing nothing - a browser that was already
    open, a window that was never opened - so a case added without preconditions
    would silently lose the check that stops a run being credited for a state it
    did not create.
    """
    for case_id, case in CASES.items():
        assert case.preconditions, f"{case_id} declares no preconditions"


def test_the_rule_text_is_distinctive_enough_to_be_evidence() -> None:
    """Verifying a search by looking for the word "search" would prove nothing."""
    for marker in (MESSAGE_MARKER, "WEEK4-OPEN-FILE-OK"):
        assert len(marker) >= 12, marker
        assert " " not in marker, marker
    assert SAMPLE_FILE.endswith(".txt")
    assert CASES["T04"].expect_text == [MESSAGE_MARKER]
    assert CASES["T05"].forbid_text == ["WEEK4-OPEN-FILE-OK"]


def test_the_send_message_case_is_marked_high_risk() -> None:
    """Risk drives the second confirmation, so it is not decoration."""
    assert CASES["T04"].risk == "high"
    assert CASES["T05"].risk == "medium"
    assert CASES["T01"].risk == "low"


def test_get_case_hands_back_a_copy() -> None:
    """A caller may edit a case; the shared definition must not move with it."""
    case = get_case("t01")
    assert case is not None
    case.expect_text.append("mutated")
    case.preconditions.clear()
    assert "mutated" not in CASES["T01"].expect_text
    assert CASES["T01"].preconditions
    assert get_case("T99") is None


def test_a_run_of_the_send_message_case_gets_a_fresh_marker() -> None:
    """15.4 asks for a new identifier on every run, and a fixed one breaks T04.

    After one attempt the previous message is still in the conversation, so the
    rule is already satisfied and the precondition - which requires no earlier
    message carrying the marker - refuses the retry. The case becomes single-use.
    """
    first, first_marker = get_case_for_run("T04")
    second, second_marker = get_case_for_run(
        "T04", now=datetime(2026, 10, 2, 9, 0, 0, tzinfo=UTC)
    )

    assert first is not None and second is not None
    assert first_marker and second_marker
    assert first_marker != second_marker
    assert first.expect_text == [first_marker]
    assert first_marker in first.instruction
    assert MESSAGE_MARKER not in first.expect_text
    # The preconditions speak of "this marker", so they read correctly for any
    # marker; what guards the retry is the rule, which now carries the fresh one.
    assert first.preconditions == CASES["T04"].preconditions


def test_the_marker_is_not_left_in_the_shared_definition() -> None:
    """`get_case` hands back a copy; the minted marker must not leak into CASES."""
    get_case_for_run("T04")
    assert CASES["T04"].expect_text == [MESSAGE_MARKER]


def test_the_other_cases_are_handed_back_untouched() -> None:
    task, marker = get_case_for_run("T01")

    assert marker == ""
    assert task is not None
    assert task.expect_text == CASES["T01"].expect_text
    assert task.instruction == CASES["T01"].instruction


def test_the_marker_carries_the_time_it_was_minted() -> None:
    marker = new_message_marker(datetime(2026, 10, 2, 9, 30, 15, tzinfo=UTC))

    assert marker.startswith("WEEK4_MESSAGE_CHECK_")
    assert "20261002" in marker
