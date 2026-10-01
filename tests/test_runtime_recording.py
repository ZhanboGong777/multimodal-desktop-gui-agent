"""Recording must survive a long run without losing or leaking anything."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from gui_agent.recording import RunSession
from gui_agent.runtime.recorder import TaskRecorder, redact
from gui_agent.runtime.schemas import (
    ElementRef,
    ExecutionOptions,
    ObservationSnapshot,
    StepRecord,
    TaskRunResult,
)
from gui_agent.schemas import BoundingBox, Point, ScreenInfo


def _frame(observation_id: str) -> ObservationSnapshot:
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=10, screenshot_height=10, control_width=10, control_height=10
        ),
        elements=[
            ElementRef(
                element_id=f"{observation_id}-e000",
                text="hi",
                bounding_box=BoundingBox(left=0, top=0, right=5, bottom=5),
                center=Point(x=2, y=2),
            )
        ],
    )


def test_typed_text_is_redacted_before_it_is_written() -> None:
    """The log records that something was typed, not what."""
    safe = redact({"text": "hunter2", "key": "enter"})
    assert safe["text"] == "<redacted>(7 chars)"
    assert safe["key"] == "enter"


def test_redaction_handles_an_empty_argument_mapping() -> None:
    assert redact(None) == {}
    assert redact({}) == {}


def test_steps_accumulate_instead_of_overwriting(tmp_path: Path) -> None:
    recorder = TaskRecorder(RunSession.create(tmp_path, session_id="r1"))
    for index in range(3):
        recorder.append_step(
            StepRecord(index=index, step_id=f"s{index}", description="d", action_type="click")
        )
    steps = recorder.read_steps()
    assert [s["step_id"] for s in steps] == ["s0", "s1", "s2"]


def test_each_observation_gets_its_own_file(tmp_path: Path) -> None:
    """RunSession writes one before/after pair; twenty steps would clobber it."""
    recorder = TaskRecorder(RunSession.create(tmp_path, session_id="r1"))
    for index in range(5):
        recorder.save_observation(_frame(f"obs-{index:04d}"))
    written = sorted(p.name for p in (tmp_path / "r1").glob("obs-*.json"))
    assert written == [f"obs-{i:04d}.json" for i in range(5)]


def test_an_observation_without_an_image_is_still_recorded(tmp_path: Path) -> None:
    """The element list is what makes a coordinate traceable; keep it regardless."""
    recorder = TaskRecorder(RunSession.create(tmp_path, session_id="r1"))
    path = recorder.save_observation(_frame("obs-0001"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["observation_id"] == "obs-0001"
    assert payload["image_path"] is None
    assert payload["elements"][0]["element_id"] == "obs-0001-e000"


def test_the_summary_is_written_and_readable(tmp_path: Path) -> None:
    recorder = TaskRecorder(RunSession.create(tmp_path, session_id="r1"))
    result = TaskRunResult(
        run_id="r1",
        case_id="T01",
        instruction="open it",
        status="failed",
        execute=True,
        elapsed_ms=12.5,
    )
    path = recorder.write_summary(result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["execute"] is True
    assert payload["action_count"] == 0
    assert ExecutionOptions().max_actions == 20


def test_reading_steps_from_a_run_that_never_acted_is_empty(tmp_path: Path) -> None:
    """A run blocked before its first action has no step log at all.

    Reading one must return nothing rather than raise: the evidence collector reads
    these, and a blocked run is exactly the kind of run whose record gets shipped.
    """
    session = RunSession.create(tmp_path, "blocked-run")
    recorder = TaskRecorder(session)

    assert recorder.read_steps() == []


# ───────── where the summary's provenance comes from ─────────
def test_the_commit_can_be_supplied_by_the_environment(monkeypatch) -> None:
    """A checkout with no git metadata records nothing otherwise.

    A copy unpacked onto the Windows node has no `.git`, and a provenance field
    that silently reads "" is worse than one the operator can set.
    """
    from gui_agent.runtime import provenance

    monkeypatch.setenv(provenance.COMMIT_ENV, "deadbee")

    assert provenance.git_commit() == "deadbee"


def test_a_screen_description_of_no_frame_is_empty() -> None:
    """A run blocked before it captured anything still has to build a summary."""
    from gui_agent.runtime import provenance

    assert provenance.screen_description(None) == ""


def test_the_screen_description_carries_both_coordinate_spaces() -> None:
    """A click that landed wrong cannot be re-read without knowing the scale."""
    from gui_agent.runtime import provenance
    from gui_agent.runtime.schemas import ObservationSnapshot
    from gui_agent.schemas import ScreenInfo

    snapshot = ObservationSnapshot(
        observation_id="obs-0001",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=2940, screenshot_height=1912, control_width=1470, control_height=956
        ),
    )

    assert provenance.screen_description(snapshot) == (
        "screenshot 2940x1912, control 1470x956"
    )


def test_the_summary_names_the_operating_system_and_its_build() -> None:
    """`os_description` was written and never wired: the field it was for did not
    exist, so the function was reachable only from its own test.

    The task report's environment table asks for the OS, and this is where that
    row can come from instead of the operator typing it in afterwards.
    """
    from gui_agent.runtime import provenance

    described = provenance.os_description()

    assert described
    assert described == platform.platform(), "the build, not just the family"
    assert len(described.split()) == 1, "one token, so it survives a table cell"


# ───────── the documents' own claims ─────────
def _spell(number: int) -> str:
    ones = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if number < 20:
        return ones[number].capitalize()
    word = tens[number // 10]
    if number % 10:
        word += "-" + ones[number % 10]
    return word.capitalize()


def test_the_troubleshooting_guide_counts_its_own_rows() -> None:
    """The header said twenty-eight while the tables held forty-two.

    It was hand-maintained and bumped only in the rounds where someone remembered,
    so it drifted by fourteen without anything noticing. A count that a reader is
    invited to trust has to be checked by something other than the person editing
    it - which is the same reason the test counts in the report are now measured
    rather than carried forward.
    """
    guide = REPO_ROOT / "Document" / "Week4" / "Week4_Troubleshooting.md"
    lines = guide.read_text(encoding="utf-8").splitlines()

    rows = [
        line
        for line in lines
        if line.startswith("| ")
        and not line.startswith("| ---")
        and not line.startswith("| Symptom")
    ]
    header = lines[2]
    assert _spell(len(rows)) in header, f"{len(rows)} rows, header says: {header!r}"
