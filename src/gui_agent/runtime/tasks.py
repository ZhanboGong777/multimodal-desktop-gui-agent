"""The five controlled task cases from the Week 4 spec.

Each case carries its own success rule. That is the point: the runner refuses to
call a task complete unless something on screen can be checked, so "the model said
finish" or "the plan ran out" never counts as success.

The text markers are deliberately unique and improbable. Verifying a search by
looking for the word "search" proves nothing; verifying a sent message by looking
for a string that only this run could have produced does.
"""

from __future__ import annotations

from .schemas import TaskSpec

#: Unique marker for the send-message case. Changing it changes the evidence.
MESSAGE_MARKER = "WEEK4_MESSAGE_CHECK_001"

#: Sample file used by the open-file case; created by the test setup.
SAMPLE_FILE = "week4_sample.txt"

#: Directory that holds the sample file on each machine.
SAMPLE_DIRECTORY = {
    "darwin": "~/Desktop/week4_test",
    "win32": r"%USERPROFILE%\Desktop\week4_test",
}


def _cases() -> dict[str, TaskSpec]:
    return {
        "T01": TaskSpec(
            case_id="T01",
            instruction="Open the web browser from the desktop",
            target_app="browser",
            preconditions=[
                "the desktop is visible and no other window covers the launch entry",
                "the browser may already be running; that is recorded, not hidden",
            ],
            success_rules=[
                "a browser window is in the foreground with an address bar or tab strip",
            ],
            expect_text=["http", "search"],
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
            expect_text=["GUI agent research"],
            risk="low",
        ),
        "T03": TaskSpec(
            case_id="T03",
            instruction=f"Open the file {SAMPLE_FILE} from the week4 test folder",
            target_app="text editor",
            preconditions=[
                f"{SAMPLE_FILE} exists in the week4 test folder",
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
            preconditions=["the week4 test application is open and focused"],
            success_rules=[
                "the application's window is gone and unrelated applications are untouched",
            ],
            forbid_text=["WEEK4-OPEN-FILE-OK"],
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
