"""What counts as done, and - more importantly - what does not."""

from __future__ import annotations

from datetime import UTC, datetime

from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot, TaskSpec
from gui_agent.runtime.verification import Verifier
from gui_agent.schemas import ActionResult, BoundingBox, DesktopAction, Point, ScreenInfo


def _frame(observation_id: str, texts: tuple[str, ...]) -> ObservationSnapshot:
    elements = [
        ElementRef(
            element_id=f"{observation_id}-e{index:03d}",
            text=text,
            bounding_box=BoundingBox(left=0, top=index * 20, right=100, bottom=index * 20 + 18),
            center=Point(x=50, y=index * 20 + 9),
            confidence=0.9,
        )
        for index, text in enumerate(texts)
    ]
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=100, screenshot_height=100, control_width=100, control_height=100
        ),
        elements=elements,
    )


TASK = TaskSpec(
    case_id="T",
    instruction="search",
    expect_text=["Results ready"],
    success_rules=["results loaded"],
)


def test_the_task_passes_only_when_the_rule_matches() -> None:
    assert Verifier().check_task(TASK, _frame("o1", ("Results ready",))).passed


def test_a_changed_screen_without_the_rule_fails() -> None:
    """The dangerous near-miss: something happened, but it was not the goal."""
    result = Verifier().check_task(TASK, _frame("o1", ("An error page",)))
    assert result.outcome == "failed"
    assert "not found" in result.detail


def test_forbidden_text_keeps_the_task_open() -> None:
    task = TaskSpec(
        case_id="T", instruction="close it", forbid_text=["Unsaved"], success_rules=["r"]
    )
    assert Verifier().check_task(task, _frame("o1", ("Desktop",))).passed
    assert Verifier().check_task(task, _frame("o1", ("Unsaved",))).outcome == "failed"


def test_a_task_with_no_rule_is_inconclusive_never_passed() -> None:
    vague = TaskSpec(case_id="T", instruction="do something")
    result = Verifier().check_task(vague, _frame("o1", ("anything",)))
    assert result.outcome == "inconclusive"
    assert not result.passed


def test_a_failed_action_fails_the_step() -> None:
    action = DesktopAction(action_type="click", x=1, y=1)
    result = Verifier().check_step(
        expected_result=None,
        before=_frame("o1", ("a",)),
        after=_frame("o2", ("a",)),
        action_result=ActionResult(action=action, success=False, dry_run=False, error="boom"),
    )
    assert result.outcome == "failed"
    assert "boom" in result.detail


def test_a_step_whose_screen_did_not_change_is_inconclusive() -> None:
    action = DesktopAction(action_type="click", x=1, y=1)
    frame = _frame("o1", ("a", "b"))
    result = Verifier().check_step(
        expected_result=None,
        before=frame,
        after=frame,
        action_result=ActionResult(action=action, success=True, dry_run=False),
    )
    assert result.outcome == "inconclusive"


def test_an_observation_with_errors_cannot_pass_a_step() -> None:
    """A degraded frame is not evidence; say so instead of guessing."""
    action = DesktopAction(action_type="click", x=1, y=1)
    after = _frame("o2", ("a",)).model_copy(update={"errors": ["ocr unavailable"]})
    result = Verifier().check_step(
        expected_result=None,
        before=_frame("o1", ("a",)),
        after=after,
        action_result=ActionResult(action=action, success=True, dry_run=False),
    )
    assert result.outcome == "inconclusive"
    assert "ocr unavailable" in result.detail


def test_step_verification_matches_on_expected_result_words() -> None:
    action = DesktopAction(action_type="click", x=1, y=1)
    result = Verifier().check_step(
        expected_result="the results page appears",
        before=_frame("o1", ("start",)),
        after=_frame("o2", ("Results page loaded",)),
        action_result=ActionResult(action=action, success=True, dry_run=False),
    )
    assert result.outcome == "passed"


# ───── the five cases' rules against the screens that must not count ─────
def test_a_selected_file_does_not_satisfy_the_open_file_case() -> None:
    """16.2's "file merely selected" case, and T03's rule does distinguish it.

    The rule looks for the file's *contents*. A file list shows the name, not the
    contents, so a run that only managed to select the file does not pass.
    """
    from gui_agent.runtime.tasks import get_case

    task = get_case("T03")
    assert task is not None

    verdict = Verifier().check_task(
        task, _frame("o1", ("week4_sample.txt", "Desktop", "week4_test"))
    )

    assert verdict.outcome == "failed"
    assert "WEEK4-OPEN-FILE-OK" in verdict.detail


def test_the_close_case_only_proves_the_marker_is_gone() -> None:
    """16.2's "closed the window but the app is still running" case.

    T05's rule asks that the marker be gone, and a minimised window satisfies that
    just as a closed one does. The case does not claim the process exited - 15.5
    says such a claim needs separate process evidence - so the honest thing is to
    pin what the rule can and cannot see rather than imply more.
    """
    from gui_agent.runtime.tasks import get_case

    task = get_case("T05")
    assert task is not None

    assert Verifier().check_task(task, _frame("o1", ("Desktop", "Other window"))).passed
    assert Verifier().check_task(task, _frame("o1", ("WEEK4-OPEN-FILE-OK",))).outcome == "failed"
