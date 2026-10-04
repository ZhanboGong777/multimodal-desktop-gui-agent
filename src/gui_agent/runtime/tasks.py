"""The five controlled task cases from the Week 4 spec.

Each case carries its own success rule. That is the point: the runner refuses to
call a task complete unless something on screen can be checked, so "the model said
finish" or "the plan ran out" never counts as success.

The text markers are deliberately unique and improbable. Verifying a search by
looking for the word "search" proves nothing; verifying a sent message by looking
for a string that only this run could have produced does.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from .schemas import TaskSpec

#: Prefix of the send-message marker. The full marker carries a timestamp, because
#: 15.4 asks for a new identifier on every run - and a fixed one breaks the case
#: after a single attempt: the previous run's message is still in the conversation,
#: so the rule is already satisfied and the precondition ("holds no earlier message
#: with this marker") refuses the retry. `MESSAGE_MARKER` stays as the marker the
#: case is defined with, for `--list-cases` and for tests.
MESSAGE_MARKER_PREFIX = "WEEK4_MESSAGE_CHECK_"
MESSAGE_MARKER = f"{MESSAGE_MARKER_PREFIX}001"

#: Sample file used by the open-file case; created by the test setup.
SAMPLE_FILE = "week4_sample.txt"

#: Directory that holds the sample file on each machine.
SAMPLE_DIRECTORY = {
    "darwin": "~/Desktop/week4_test",
    "win32": r"%USERPROFILE%\Desktop\week4_test",
}


def sample_directory(platform: str | None = None) -> str:
    """Where the sample file lives on this machine.

    The case definitions are read by the operator as much as by the code, so T03's
    precondition names the actual folder rather than "the week4 test folder". The
    constant was defined and read by nothing at all before this.
    """
    key = platform or ("darwin" if sys.platform == "darwin" else "win32")
    return SAMPLE_DIRECTORY[key]


def _cases() -> dict[str, TaskSpec]:
    return {
        "T01": TaskSpec(
            case_id="T01",
            instruction="Open the web browser from the desktop",
            target_app="browser",
            preconditions=[
                "the desktop is visible and no other window covers the launch entry",
                (
                    "no browser window is open. A browser that was already running carries "
                    "the text this rule looks for, so on its own it would not show that "
                    "this run opened anything - and a real run is blocked rather than "
                    "credited for it"
                ),
            ],
            success_rules=[
                "a browser window is in the foreground with an address bar or tab strip",
            ],
            # Read from a screenshot of the Windows run's last frame, which is what
            # settled this: the browser *had* opened (Chrome's new-tab page, with its
            # address bar and the page title '新标签页'), and the rule still failed
            # because Chrome on a Chinese install shows '在 Google 中搜索，或输入网址',
            # never the English word 'search'. The only marks present in both locales are
            # the scheme in a URL tile and the verb in the search box, so those two are
            # what the rule asks for. The list is an AND: a rule that listed four marks
            # failed on the three the frame did not carry.
            expect_text=["http", "搜索"],
            # The machine-checkable half of "no browser window is open". Text cannot
            # answer it: a browser behind another window, or minimised, contributes none
            # of the text this rule looks for, so a text check called it absent while it
            # was running - and T01 then planned against the window that already
            # existed, which is how it failed three times on the Windows node while both
            # its words and the model's plan were correct.
            forbids_processes=["browser"],
            risk="low",
        ),
        "T02": TaskSpec(
            case_id="T02",
            instruction="Search the web for GUI agent research",
            target_app="browser",
            preconditions=["a browser window is open and focused"],
            success_rules=[
                "a results page is loaded and the query text is visible",
            ],
            # Both entries are required - the list is an AND - and the second is what makes
            # the rule mean "a results page" rather than "somewhere on screen".
            #
            # Measured, because the difference produced a false success: a run typed the
            # query into the address bar and never submitted it, and the rule passed on
            # 'GUI agent research' alone, which the autocomplete dropdown was showing.
            # `&oq=` is added by the engine on a real submission and appears in the URL
            # field, so it distinguishes a submitted search from text waiting in an input
            # box. Confirmed both ways: absent on the frame that falsely passed, present on
            # a real results page as 'google.com/search?q=GUI+agent+research&oq=GUI+agent+'.
            expect_text=["GUI agent research", "google.com/search"],
            # The other direction of the same check: T02 is meaningless without a
            # browser to search in, and "is one running" is a fact about the machine
            # rather than something to read off the screen.
            requires_processes=["browser"],
            risk="low",
        ),
        "T03": TaskSpec(
            case_id="T03",
            instruction=f"Open the file {SAMPLE_FILE} from the week4 test folder",
            target_app="text editor",
            preconditions=[
                f"{SAMPLE_FILE} exists in {sample_directory()}",
                "no file with the same name is open already",
            ],
            success_rules=[
                "the file is open in an application, not merely selected in a file list",
            ],
            expect_text=["WEEK4-OPEN-FILE-OK"],
            risk="low",
        ),
        "T04": TaskSpec(
            case_id="T04",
            instruction=f"Send {MESSAGE_MARKER} to the week4 test conversation",
            target_app="messaging",
            preconditions=[
                "the test conversation is open and contains no other message with this marker",
                "the operator has agreed that a real message may be sent",
            ],
            success_rules=[
                "the marker appears as a sent message in the correct conversation",
            ],
            expect_text=[MESSAGE_MARKER],
            risk="high",
        ),
        "T05": TaskSpec(
            case_id="T05",
            instruction="Close the week4 test application without affecting other applications",
            target_app="text editor",
            preconditions=[
                (
                    f"{SAMPLE_FILE} is open in the test application, so the marker text is on "
                    "screen before the run starts. The rule only asks that the marker be "
                    "gone, which is already true while the window is shut - so a run that "
                    "closed nothing would otherwise look like a success"
                ),
                "the window is focused, so the close control is reachable",
            ],
            success_rules=[
                "the application's window is gone and unrelated applications are untouched",
            ],
            forbid_text=["WEEK4-OPEN-FILE-OK"],
            # "is open in the test application" is about the machine, not the screen. The
            # text check cannot tell "the window was closed" from "the window was behind
            # something", and that ambiguity cost three blocked runs on the Windows node:
            # each reported "the success rule already holds on the untouched screen" while
            # Notepad was open but not in the capture. Requiring the editor to be running
            # makes the honest cases proceed and the dishonest ones say so.
            requires_processes=["editor"],
            risk="medium",
        ),
    }


#: All five cases by id.
CASES: dict[str, TaskSpec] = _cases()


def get_case(case_id: str) -> TaskSpec | None:
    """Return a copy of the case, so a caller cannot mutate the shared spec."""
    found = CASES.get(case_id.upper())
    return found.model_copy(deep=True) if found else None


def case_ids() -> list[str]:
    return sorted(CASES)


def new_message_marker(now: datetime | None = None) -> str:
    """A marker no earlier run could have used."""
    stamp = (now or datetime.now(UTC)).astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{MESSAGE_MARKER_PREFIX}{stamp}"


def get_case_for_run(case_id: str, now: datetime | None = None) -> tuple[TaskSpec | None, str]:
    """A case ready for one run, plus the message marker it will use.

    15.4 asks for a fresh identifier on every run, and a fixed one breaks the case
    after a single attempt: the previous run's message is still in the conversation,
    so T04's rule is already satisfied and its own precondition - which requires no
    earlier message with that marker - refuses the retry. Only the send-message case
    is touched; the marker is returned so the CLI can show what to look for.
    """
    task = get_case(case_id)
    if task is None or MESSAGE_MARKER not in task.expect_text:
        return task, ""
    marker = new_message_marker(now)
    task.expect_text = [marker if text == MESSAGE_MARKER else text for text in task.expect_text]
    task.instruction = task.instruction.replace(MESSAGE_MARKER, marker)
    # The preconditions say "this marker" rather than naming the literal, so they
    # read correctly for any marker and need no rewriting. What guards a retry is
    # `check_task` against `expect_text`, which now carries the fresh one.
    return task, marker
