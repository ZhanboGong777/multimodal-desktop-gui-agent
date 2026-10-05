"""The five controlled task cases from the Week 4 spec.

Each case carries its own success rule. That is the point: the runner refuses to
call a task complete unless something on screen can be checked, so "the model said
finish" or "the plan ran out" never counts as success.

The text markers are deliberately unique and improbable. Verifying a search by
looking for the word "search" proves nothing; verifying a sent message by looking
for a string that only this run could have produced does.
"""

from __future__ import annotations

import os
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

#: The conversation the send-message case is aimed at, as the screen names it.
#:
#: It has to be named, and it was not - measured on the Windows node: T04's plan opened with
#: `click "Open the week4 test conversation"` and the step died with
#: `no element matches 'week4 test conversation' in obs-0002`. The instruction said "to the
#: week4 test conversation" while naming nothing the screen could be searched for, so the
#: model went looking for a box with that label. Which conversation is open is a
#: *precondition* - the operator prepares it - and a precondition the model is asked to
#: re-establish is one the case cannot rely on.
#:
#: Overridable per machine, because the conversation is the operator's to choose and a
#: project should not hard-code somebody's chat list. The default is the one this node uses.
MESSAGE_CONVERSATION = os.environ.get("GUI_AGENT_MESSAGE_CONVERSATION", "文件传输助手")

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
            # Both entries are required - the list is an AND - and they are read from two
            # different places on purpose.
            #
            # 'GUI agent research' is the task's own query. Alone it is not evidence: a run
            # that typed the query and never submitted it verified as **passed** because the
            # autocomplete dropdown was showing the string, which is a rule for "the goal was
            # reached" being satisfied by the goal's input.
            #
            # 'Google 搜索' is part of the window title, and it comes from the window rather
            # than from the pixels - the observation records `window_title`, and the verifier
            # searches it alongside the OCR text. That matters here because this case's real
            # evidence lives in browser chrome, where the OCR engine is unreliable: measured
            # on a run that did reach a Google results page, the element list held neither the
            # URL nor the tab title, and the title was read on one frame and missed on the
            # next, while `foreground_window_title()` returned
            # 'GUI agent research - Google 搜索 - Google Chrome' for that same screen.
            #
            # The second entry is the only honest marker available, and it is read through
            # OCR - which is this case's unsolved problem rather than a solved one. Three
            # alternatives were measured and each is present in the unsubmitted state too,
            # so each would have been a false discriminator rather than a fix:
            #
            #   'Google Chrome'  every Chrome window carries it;
            #   'Google 搜索'    the autocomplete dropdown shows a suggestion reading
            #                    'GUI agent research - Google 搜索', character for character
            #                    what the results page's own title says;
            #   'http'           the toolbar shows URL-ish text on a new tab as well.
            #
            # The window title is now recorded on every observation (`window_title`) and the
            # verifier searches it, which is a real gain for rules that can use it, and it is
            # how the three candidates above were checked. It does not settle *this* rule,
            # because what distinguishes a submitted search from an unsubmitted one is the
            # address field, and the two states carry the same title and the same suggestion
            # text. Recorded here so the next attempt does not re-test the same three.
            expect_text=["GUI agent research", "google.com"],
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
            # "already open" is the whole point, and saying it removes a step the model
            # otherwise invents. Measured twice on the Windows node: with the instruction
            # naming the conversation but not its state, the plan opened with
            # `click "Open the conversation named '文件传输助手'"` and died on
            # `2 elements match '文件传输助手' (e034, e054); refusing to pick one
            # arbitrarily` - the name is on screen twice, in the list and in the open chat's
            # header, which is itself proof that the conversation was already open. The
            # precondition says the operator prepares it, so the instruction must not ask the
            # model to do the operator's half.
            # The default lives in code; the wording a machine needs is settable, because what
            # sends a message is a property of the client rather than of the case. Measured on
            # this node: "and send it" produced a plan ending in `key_press Enter`, the marker
            # stayed in the message box, and the frame showed the send button beside it
            # unclicked - that WeChat has "Enter for a new line" set, so Enter types a newline.
            # The rule that had told the model otherwise was reverted; the way to aim it at the
            # button is to say so, and to say so per machine.
            #
            # So the *default* now says to click the button, because the phrasing it replaced is
            # the one measured not to work. Leaving a known-failing sentence as the default and
            # requiring an environment variable to avoid it is how a case stays broken on every
            # machine nobody configured - and this one has already cost several runs.
            #
            # Naming a key is also unsafe in the other direction, measured later on the same
            # node: an instruction that *asked* for Enter produced a plan whose `key_press enter`
            # opened WeChat's file-picker dialog (the frame afterwards reports
            # `foreground: '选择文件'`), leaving the message box untouched. An instruction is
            # executable policy, so this one names the control to click and forbids the key.
            instruction=os.environ.get(
                "GUI_AGENT_MESSAGE_INSTRUCTION",
                f"Type {MESSAGE_MARKER} into the message box of the conversation named "
                f'"{MESSAGE_CONVERSATION}", which is already open, then click the send button. '
                f"Do not press Enter at any point: in this client Enter opens a file dialog "
                f"instead of sending.",
            ),
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
            # Its typing step cannot be judged from a screenshot and its task can; see the
            # field for the four runs that established it. The task rule still runs.
            gate_on_step_verification=False,
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
