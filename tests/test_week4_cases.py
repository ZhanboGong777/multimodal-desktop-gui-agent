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
    sample_directory,
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


def test_the_sample_directory_is_named_for_each_platform() -> None:
    """The fold the operator has to create, not a description of it.

    `SAMPLE_DIRECTORY` was defined and read by nothing, so the constant that knows
    where the file goes never reached the precondition that tells the operator.
    """
    assert sample_directory("darwin") == "~/Desktop/week4_test"
    assert sample_directory("win32") == r"%USERPROFILE%\Desktop\week4_test"


def test_the_send_message_default_names_the_button_and_forbids_enter() -> None:
    """The default instruction has to be the phrasing that is known to work.

    Two measurements on the Windows node, in opposite directions:

    * `"... and send it"` produced a plan ending in `key_press Enter`, and the marker stayed in
      the message box with the send button beside it unclicked - this client is configured for
      "Enter for a new line";
    * an instruction that explicitly *asked* for Enter produced a plan whose `key_press enter`
      opened the client's file-picker dialog, leaving the box untouched.

    So the default names the control to click and rules the key out, and this asserts both. It
    matters because the default is what every machine runs: leaving the known-failing sentence in
    place and relying on `GUI_AGENT_MESSAGE_INSTRUCTION` to avoid it is how a case stays broken on
    every node nobody configured.
    """
    instruction = get_case("T04").instruction

    assert "send button" in instruction, instruction
    assert "Do not press Enter" in instruction, instruction


def test_only_the_send_message_case_turns_the_step_gate_off() -> None:
    """T04 is the one case whose steps cannot be judged, and the flag says so explicitly.

    Measured across four passes of four separate runs: T04's typing step verifies against
    `expected_result` reading "The message is typed into the message box", whose tokens include
    `typed` - a word that names what was done rather than anything a screen displays - and whose
    remaining nouns name a box the OCR cannot read. So the check fails on a step that *worked*, and
    a failed step verification stops the run before its send. The same sentence came back even from
    runs whose instruction explicitly forbade mentioning the box.

    What is asserted here is that the exception stays an exception: if another case ever turns this
    off, that should be a decision someone makes on purpose rather than a default spreading.
    """
    off = [
        case_id
        for case_id in case_ids()
        if get_case(case_id) is not None and not get_case(case_id).gate_on_step_verification
    ]

    assert off == ["T04"], f"expected only T04 to disable the gate, got {off}"


def test_the_open_file_case_names_where_the_file_must_be() -> None:
    """T03 cannot start without the file, so its precondition says where to put it."""
    task = get_case("T03")
    assert task is not None

    first = task.preconditions[0]
    assert SAMPLE_FILE in first
    assert sample_directory() in first, first


def test_the_search_case_needs_evidence_of_a_results_page_not_just_the_query() -> None:
    """T02's rule passed on a query that was never submitted, and this is what stops that.

    Measured: a run typed 'GUI agent research' into the address bar, never pressed Enter,
    and the task verified as **passed** - because the rule asked only for the query text to
    be on screen, and the autocomplete dropdown was showing it. A second capture of the same
    untouched screen still read the string, so the false positive was reproducible rather
    than a one-off.

    The second entry is the results-page marker the engine writes on a real submission. It
    was absent on the frame that falsely passed and present on a real results page, as
    'google.com/search?q=GUI+agent+research&oq=GUI+agent+'. Both are required - the list is
    an AND - so the rule now means "a results page carrying the query", which is what it
    says, rather than "the query is somewhere on screen".
    """
    task = get_case("T02")
    assert task is not None

    assert "GUI agent research" in task.expect_text
    assert any("google.com" in item for item in task.expect_text), (
        "the query alone is satisfied by text sitting in an input box; T02 needs evidence "
        "that a page loaded, which the address field carries and a new tab's does not"
    )
    # Short rather than precise, and that is the measurement: a frame carrying
    # 'google.com/search?q=GUI+agent+research&oq=...&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIG...'
    # was read by the OCR engine as *not* containing 'google.com/search' - the field is
    # narrow, the string is long, and the engine gave up on it. `google.com` came through
    # on that same frame.
    assert not any("/" in item or "?" in item for item in task.expect_text if "." in item), (
        "a marker with a path or query in it is longer than the OCR reads reliably in a "
        "browser's address field"
    )
