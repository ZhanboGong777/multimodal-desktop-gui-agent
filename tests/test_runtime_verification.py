"""What counts as done, and - more importantly - what does not."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from gui_agent.models.base import ModelResponse
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


def test_an_expectation_written_as_a_description_of_the_action_still_passes() -> None:
    """The sentence T04's model wrote, and the word that made it fail.

    expected_result='The message is typed into the message box' produced the tokens
    ['message', 'typed', 'into', 'message'] and no hits, because 	yped names what had just
    been done rather than anything a screen displays. A token that can never match fails the
    check, and a failed step verification stops the pass - so a step whose typing had worked was
    recorded as failing and the run stopped one step short of its send, four passes running.

    The assertion is that the action words are gone and the nouns are not: message and \box
    stay, because a screen can show them and a case may expect it to.
    """
    from gui_agent.runtime.verification import _ACTION_WORDS, _words

    tokens = [t for t in _words('The message is typed into the message box') if len(t) >= 4]

    assert 'typed' in tokens, 'the word is present before filtering, or this proves nothing'
    kept = [t for t in tokens if t not in _ACTION_WORDS]
    assert 'typed' not in kept
    assert 'message' in kept and 'into' in kept
    assert 'box' in [t for t in _words('the box') if t not in _ACTION_WORDS]


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


def test_no_observation_is_inconclusive_rather_than_failed() -> None:
    """Nothing to look at is not the same as looking and finding the goal unmet.

    Reporting `failed` would say the task was attempted and did not work; the truth
    is that nobody looked.
    """
    verdict = Verifier().check_task(TASK, None)

    assert verdict.outcome == "inconclusive"
    assert "no observation" in verdict.detail


def test_the_screen_going_away_while_polling_is_inconclusive() -> None:
    """The polling loop is the one observer call the runner does not guard.

    Before this it let the exception out of `run()` entirely, so a display that
    slept between the last action and the verdict ended the run as a traceback.
    """

    def blind() -> ObservationSnapshot:
        raise RuntimeError("monitor_index 1 is out of range (available 1..0)")

    verdict, latest = Verifier().check_task_with_polling(
        TASK, blind, deadline_seconds=10.0, sleep=lambda _s: None, clock=lambda: 0.0
    )

    assert verdict.outcome == "inconclusive"
    assert "could not look at the screen" in verdict.detail
    assert "monitor_index 1 is out of range" in verdict.detail
    assert latest is None


def test_a_changed_screen_with_nothing_to_check_is_not_a_pass() -> None:
    """11.1.5's one prohibition: a changed screenshot is not success.

    The code did the opposite while its own docstring said otherwise, so this was
    a claim the module made about itself and did not keep. There is no test that
    would have failed: the changed-screen branch was unexercised.
    """
    action = DesktopAction(action_type="click", x=1, y=1)
    result = Verifier().check_step(
        expected_result=None,
        before=_frame("o1", ("start",)),
        after=_frame("o2", ("something else entirely",)),
        action_result=ActionResult(action=action, success=True, dry_run=False),
        action_type="click",
    )

    assert result.outcome == "inconclusive"
    assert "not evidence" in result.detail
    # The frame did change, and that stays in the record - it just is not a pass.
    assert result.evidence["changed"] is True


def test_a_move_or_wait_is_verified_by_its_own_completion() -> None:
    """11.1.6: these do not change the screen, so a screen check has nothing to say.

    Verified by the action having completed, and reported as exactly that rather
    than as an observation of a result.
    """
    for action_type in ("move", "wait"):
        action = DesktopAction(action_type=action_type, x=1, y=1, duration=0.1)
        result = Verifier().check_step(
            expected_result=None,
            before=_frame("o1", ("start",)),
            after=_frame("o1", ("start",)),
            action_result=ActionResult(action=action, success=True, dry_run=False),
            action_type=action_type,
        )
        assert result.outcome == "passed", action_type
        assert "own completion" in result.detail
        assert "not an observation of a result" in result.detail


def test_a_move_that_failed_is_still_a_failure() -> None:
    """Its own completion is the check, so a failed one cannot pass."""
    action = DesktopAction(action_type="move", x=1, y=1)
    result = Verifier().check_step(
        expected_result=None,
        before=_frame("o1", ("start",)),
        after=_frame("o2", ("moved",)),
        action_result=ActionResult(
            action=action, success=False, dry_run=False, error="pointer refused"
        ),
        action_type="move",
    )

    assert result.outcome == "failed"
    assert "pointer refused" in result.detail


_TEST_MARKER = "WEEK4_MESSAGE_CHECK_20261006_000001"
_TEST_CONVERSATION = "文件传输助手"


def _message_task() -> TaskSpec:
    return TaskSpec(
        case_id="T04", instruction=f"send {_TEST_MARKER}",
        expect_text=[_TEST_MARKER], success_rules=["sent in the correct conversation"],
        message_conversation=_TEST_CONVERSATION, risk="high",
    )


def _message_frame(tmp_path: Path) -> ObservationSnapshot:
    image = tmp_path / "synthetic-message.png"
    Image.new("RGB", (800, 600), "white").save(image)
    return ObservationSnapshot(
        observation_id="message-frame", captured_at=datetime.now(UTC), image_path=str(image),
        screen_info=ScreenInfo(
            screenshot_width=800, screenshot_height=600, control_width=800, control_height=600,
        ),
        window_id="synthetic-window", foreground_stable=True,
        window_bounds=BoundingBox(left=200, top=20, right=780, bottom=550),
    )


def _sent_assessment(**updates) -> dict:
    return {
        "status": "sent", "conversation": _TEST_CONVERSATION, "marker": _TEST_MARKER,
        "header_box": [300, 50, 450, 75], "message_box": [400, 160, 680, 190],
        "composer_box": [300, 350, 700, 500], "composer_empty": True,
        **updates,
    }


class _MessageClient:
    """Synthetic visual assessments; this client never reads or drives a desktop."""

    def __init__(self, payload: dict | str, *, error: str | None = None) -> None:
        self.content = payload if isinstance(payload, str) else json.dumps(payload)
        self.error = error
        self.calls: list[dict] = []

    def generate_multimodal(self, instruction, **kwargs) -> ModelResponse:
        self.calls.append({"instruction": instruction, **kwargs})
        return ModelResponse(
            content=self.content, model_name="synthetic-vision", provider="test",
            error=self.error,
        )


def test_sent_message_requires_visual_transcript_evidence_without_ocr(tmp_path: Path) -> None:
    client = _MessageClient(_sent_assessment())
    frame = _message_frame(tmp_path)
    verdict = Verifier(message_client=client).check_task(_message_task(), frame)
    assert verdict.passed
    assert frame.elements == [], "the saved real T04 frame's OCR missed header and marker"
    assert verdict.evidence["assessment_method"] == "vision_model"
    assert verdict.evidence["vision_assessment"]["message_box"] == [400, 160, 680, 190]
    assert verdict.evidence["image_path"] == frame.image_path
    assert verdict.evidence["model_name"] == "synthetic-vision"
    prompt = json.dumps(client.calls[0], ensure_ascii=False)
    assert _TEST_MARKER not in prompt and _TEST_CONVERSATION not in prompt


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({"status": "draft", "composer_empty": False}, "failed"),
        ({"status": "not_found", "marker": "", "message_box": None}, "failed"),
        ({"status": "uncertain"}, "inconclusive"),
        ({"conversation": "Wrong conversation"}, "failed"),
        ({"marker": _TEST_MARKER.lower()}, "failed"),
        ({"marker": "WEEK4_MESSAGE_CHECK_OLD"}, "failed"),
        ({"composer_empty": False}, "failed"),
        ({"message_box": [400, 370, 680, 390]}, "inconclusive"),
        ({"message_box": [50, 160, 180, 190]}, "inconclusive"),
        ({"header_box": [210, 50, 250, 75]}, "inconclusive"),
        ({"header_box": [300, 170, 450, 195]}, "inconclusive"),
        ({"composer_box": [300, 350, 900, 500]}, "inconclusive"),
    ],
)
def test_draft_wrong_recipient_and_invalid_regions_never_credit_sent_message(
    tmp_path: Path, updates: dict, expected: str
) -> None:
    frame = _message_frame(tmp_path)
    box = BoundingBox(left=350, top=400, right=650, bottom=420)
    frame.elements = [ElementRef(
        element_id="draft-marker", text=_TEST_MARKER, bounding_box=box, center=box.center,
    )]
    verdict = Verifier(message_client=_MessageClient(_sent_assessment(**updates))).check_task(
        _message_task(), frame
    )
    assert verdict.outcome == expected
    assert not verdict.passed, "OCR marker-anywhere is insufficient even if the marker is exact"


@pytest.mark.parametrize(
    "updates",
    [
        {"composer_empty": "true"}, {"composer_empty": 1},
        {"header_box": [True, 50, 450, 75]}, {"header_box": [300.0, 50, 450, 75]},
        {"header_box": [300, 50, 450]}, {"header_box": [450, 50, 300, 75]},
        {"status": "success"}, {"conversation": None}, {"marker": [_TEST_MARKER]},
        {"status": "draft", "composer_empty": True},
        {"extra_evidence": "trust me"},
    ],
)
def test_visual_assessment_is_strict_json_without_coercion(tmp_path: Path, updates: dict) -> None:
    client = _MessageClient(_sent_assessment(**updates))
    verdict = Verifier(message_client=client).check_task(_message_task(), _message_frame(tmp_path))
    assert verdict.outcome == "inconclusive"
    assert "unusable" in verdict.detail


@pytest.mark.parametrize("kind", ["invalid", "fenced", "missing", "duplicate", "transport"])
def test_malformed_or_failed_visual_response_is_inconclusive(tmp_path: Path, kind: str) -> None:
    content = json.dumps(_sent_assessment())
    error = None
    if kind == "invalid":
        content = "the model said it was sent"
    elif kind == "fenced":
        content = f"```json\n{content}\n```"
    elif kind == "missing":
        payload = _sent_assessment()
        del payload["composer_empty"]
        content = json.dumps(payload)
    elif kind == "duplicate":
        content = content[:-1] + ',"composer_empty":false}'
    else:
        error = "connection unavailable"
    verdict = Verifier(message_client=_MessageClient(content, error=error)).check_task(
        _message_task(), _message_frame(tmp_path)
    )
    assert verdict.outcome == "inconclusive"


@pytest.mark.parametrize("problem", ["image", "missing_file", "stable", "identity", "bounds", "errors", "size"])
def test_missing_visual_evidence_refuses_without_calling_model(tmp_path: Path, problem: str) -> None:
    frame = _message_frame(tmp_path)
    if problem == "image":
        frame.image_path = None
    elif problem == "missing_file":
        frame.image_path = str(tmp_path / "missing.png")
    elif problem == "stable":
        frame.foreground_stable = False
    elif problem == "identity":
        frame.window_id = ""
    elif problem == "bounds":
        frame.window_bounds = None
    elif problem == "size":
        frame.screen_info.screenshot_width = 801
    else:
        frame.errors = ["OCR unavailable"]
    client = _MessageClient(_sent_assessment())
    assert Verifier(message_client=client).check_task(_message_task(), frame).outcome == "inconclusive"
    assert not client.calls


def test_missing_visual_client_does_not_fall_back_to_marker_anywhere(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    frame.elements = _frame("draft", (_TEST_MARKER, _TEST_CONVERSATION)).elements
    assert Verifier().check_task(_message_task(), frame).outcome == "inconclusive"


def test_message_context_accepts_empty_composer_without_sent_marker_and_caches(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    client = _MessageClient(_sent_assessment(status="not_found", marker="", message_box=None))
    verifier = Verifier(message_client=client)
    assert verifier.check_message_context(_message_task(), frame, require_empty=True).passed
    verdict = verifier.check_task(_message_task(), frame)
    assert verdict.outcome == "failed"
    assert verdict.evidence["assessment_cached"] is True
    assert len(client.calls) == 1


def test_message_context_requires_empty_composer_only_when_requested(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    verifier = Verifier(message_client=_MessageClient(_sent_assessment(status="draft", composer_empty=False)))
    assert verifier.check_message_context(_message_task(), frame).passed
    assert verifier.check_message_context(_message_task(), frame, require_empty=True).outcome == "failed"


def test_changed_window_or_image_does_not_reuse_cached_assessment(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    client = _MessageClient(_sent_assessment())
    verifier = Verifier(message_client=client)
    assert verifier.check_task(_message_task(), frame).passed
    frame.window_id = "different-window"
    assert verifier.check_task(_message_task(), frame).passed
    second = tmp_path / "second.png"
    Image.new("RGB", (800, 600), "black").save(second)
    frame.image_path = str(second)
    assert verifier.check_task(_message_task(), frame).passed
    assert len(client.calls) == 3
