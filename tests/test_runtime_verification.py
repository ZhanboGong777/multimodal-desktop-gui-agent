"""What counts as done, and - more importantly - what does not."""

from __future__ import annotations

import json
import os
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
        # Responses describe the 580x530 window crop, whose screenshot origin is
        # (200,20). Verifier evidence must map these back to the original frame.
        "header_box": [100, 30, 250, 55], "message_box": [200, 140, 480, 170],
        "composer_box": [100, 330, 500, 480], "composer_empty": True,
        "outgoing": True, "send_state": "sent",
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
    assert verdict.evidence["source_image_path"] == frame.image_path
    crop_path = Path(verdict.evidence["image_path"])
    assert crop_path != Path(frame.image_path)
    assert client.calls[0]["image_path"] == str(crop_path)
    with Image.open(crop_path) as crop:
        assert crop.size == (580, 530)
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
        ({"message_box": [200, 350, 480, 370]}, "inconclusive"),
        ({"message_box": [-150, 140, -20, 170]}, "inconclusive"),
        ({"header_box": [10, 30, 50, 55]}, "inconclusive"),
        ({"header_box": [100, 150, 250, 175]}, "inconclusive"),
        ({"composer_box": [100, 330, 700, 480]}, "inconclusive"),
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
        {"outgoing": "true"}, {"outgoing": False},
        {"send_state": "pending"}, {"send_state": "failed"}, {"send_state": "unavailable"},
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


def test_message_crop_preserves_pixels_and_records_local_to_screen_transform(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    assert frame.image_path is not None and frame.window_bounds is not None
    pixels = Image.new("RGB", (800, 600), "red")
    pixels.paste("blue", (200, 20, 780, 550))
    pixels.paste("green", (210, 40, 220, 50))
    pixels.save(frame.image_path)
    local_payload = _sent_assessment()
    client = _MessageClient(local_payload)
    verdict = Verifier(message_client=client).check_task(_message_task(), frame)
    assert verdict.passed
    evidence = verdict.evidence
    assert evidence["crop_transform"] == {
        "source_bounds": {"left": 200, "top": 20, "right": 780, "bottom": 550},
        "origin": {"x": 200, "y": 20}, "size": {"width": 580, "height": 530},
        "scale_x": 1, "scale_y": 1,
    }
    assert evidence["vision_assessment_local"] == local_payload
    assert evidence["vision_assessment"]["header_box"] == [300, 50, 450, 75]
    assert evidence["vision_assessment"]["composer_box"] == [300, 350, 700, 500]
    assert evidence["assessment_coordinate_space"] == "foreground_crop_pixels"
    assert evidence["vision_assessment_coordinate_space"] == "screenshot_pixels"
    assert json.loads(evidence["assessment_raw_content"]) == local_payload
    assert evidence["assessment_prompt"] == {
        "system": client.calls[0]["system"],
        "instruction": client.calls[0]["instruction"],
        "context": client.calls[0]["context"],
        "response_format": client.calls[0]["response_format"],
        "include_image_path_in_prompt": False,
    }
    assert client.calls[0]["context"]["image_size"] == {"width": 580, "height": 530}
    assert "crop-local" in client.calls[0]["system"]
    assert "580x530" in client.calls[0]["instruction"]
    with Image.open(evidence["image_path"]) as crop:
        assert crop.getpixel((0, 0)) == (0, 0, 255)
        assert crop.getpixel((10, 20)) == (0, 128, 0)


@pytest.mark.parametrize("field", ["header_box", "message_box", "composer_box"])
def test_screen_valid_box_outside_the_crop_is_rejected_before_translation(
    tmp_path: Path, field: str,
) -> None:
    frame = _message_frame(tmp_path)
    # These coordinates fit both the screenshot and the original window. They
    # are still invalid model evidence because the attached crop is only 580px wide.
    client = _MessageClient(_sent_assessment(**{field: [590, 50, 620, 75]}))
    verdict = Verifier(message_client=client).check_task(_message_task(), frame)
    assert verdict.outcome == "inconclusive"
    assert f"{field} lies outside the attached foreground crop" in verdict.detail
    assert "vision_assessment" not in verdict.evidence
    assert verdict.evidence["assessment_raw_content"] == client.content


def test_changed_crop_bounds_invalidates_cache_and_updates_evidence(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    client = _MessageClient(_sent_assessment())
    verifier = Verifier(message_client=client)
    first = verifier.check_task(_message_task(), frame)
    assert first.passed
    frame.window_bounds = BoundingBox(left=180, top=30, right=770, bottom=560)
    second = verifier.check_task(_message_task(), frame)
    assert second.passed
    assert len(client.calls) == 2
    assert first.evidence["image_path"] != second.evidence["image_path"]
    assert second.evidence["vision_assessment"]["header_box"] == [280, 60, 430, 85]
    assert second.evidence["crop_transform"]["origin"] == {"x": 180, "y": 30}
    third = verifier.check_task(_message_task(), frame)
    assert third.evidence["assessment_cached"] is True
    assert third.evidence["image_path"] == second.evidence["image_path"]
    assert len(client.calls) == 2
    assert Path(first.evidence["image_path"]).is_file(), "previous crop evidence stays available"


def test_same_file_metadata_with_changed_pixels_does_not_reuse_crop_assessment(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    image = tmp_path / "fixed-size-frame.bmp"
    Image.new("RGB", (800, 600), "white").save(image)
    frame.image_path = str(image)
    client = _MessageClient(_sent_assessment())
    verifier = Verifier(message_client=client)
    first = verifier.check_task(_message_task(), frame)
    initial = image.stat()
    Image.new("RGB", (800, 600), "black").save(image)
    os.utime(image, ns=(initial.st_atime_ns, initial.st_mtime_ns))
    assert image.stat().st_size == initial.st_size
    assert image.stat().st_mtime_ns == initial.st_mtime_ns
    second = verifier.check_task(_message_task(), frame)
    assert first.passed and second.passed
    assert len(client.calls) == 2
    assert first.evidence["source_pixel_sha256"] != second.evidence["source_pixel_sha256"]
    assert first.evidence["image_path"] != second.evidence["image_path"]


@pytest.mark.parametrize("bounds", [
    {"left": -1, "top": 20, "right": 780, "bottom": 550},
    {"left": 200, "top": 20, "right": 801, "bottom": 550},
    {"left": 200, "top": 550, "right": 780, "bottom": 550},
    {"left": 200.5, "top": 20, "right": 780, "bottom": 550},
])
def test_invalid_foreground_crop_bounds_never_call_the_model(tmp_path: Path, bounds: dict) -> None:
    frame = _message_frame(tmp_path)
    frame.window_bounds = BoundingBox.model_construct(**bounds)
    client = _MessageClient(_sent_assessment())
    verdict = Verifier(message_client=client).check_task(_message_task(), frame)
    assert verdict.outcome == "inconclusive"
    assert not client.calls
    assert not (tmp_path / "message_assessments").exists()


def test_invalid_json_keeps_raw_response_crop_and_prompt_in_cached_evidence(tmp_path: Path) -> None:
    frame = _message_frame(tmp_path)
    client = _MessageClient("```json\nnot a parsed assessment\n```")
    verifier = Verifier(message_client=client)
    first = verifier.check_message_context(_message_task(), frame, require_empty=True)
    assert first.outcome == "inconclusive"
    assert first.evidence["assessment_raw_content"] == client.content
    assert Path(first.evidence["image_path"]).is_file()
    assert first.evidence["source_image_path"] == frame.image_path
    assert "vision_assessment" not in first.evidence
    second = verifier.check_task(_message_task(), frame)
    assert second.outcome == "inconclusive"
    assert second.evidence["assessment_cached"] is True
    assert second.evidence["assessment_raw_content"] == client.content
    assert len(client.calls) == 1


def _regional_message_frame(tmp_path: Path) -> ObservationSnapshot:
    """Observed OCR/title and real-sized contour proposals, never fabricated targets."""
    frame = _message_frame(tmp_path)
    frame.window_title = "Message client"
    frame.window_class = "ClientWindow"

    def element(number: int, coordinates: tuple, *, text: str = "", source: str = "contour"):
        box = BoundingBox(
            left=coordinates[0], top=coordinates[1], right=coordinates[2], bottom=coordinates[3],
        )
        return ElementRef(
            element_id=f"{frame.observation_id}-e{number:03d}", text=text,
            bounding_box=box, center=box.center, confidence=0.9, source=source,
        )

    frame.elements = [
        element(1, (300, 350, 700, 500)),
        element(2, (310, 50, 450, 75), text=_TEST_CONVERSATION, source="ocr"),
        # An earlier sidebar match must never become the title crop.
        element(3, (210, 35, 280, 60), text=_TEST_CONVERSATION, source="ocr"),
        element(4, (400, 160, 680, 190)),
        element(5, (400, 280, 680, 310)),
        element(6, (405, 165, 675, 185), text="WEEK4_MESSAGE_CHECK_OLD", source="ocr"),
    ]
    pixels = Image.new("RGB", (800, 600), "white")
    pixels.paste("blue", (310, 50, 450, 75))
    pixels.paste("green", (300, 350, 700, 500))
    pixels.paste("magenta", (400, 160, 680, 190))
    pixels.paste("magenta", (400, 280, 680, 310))
    pixels.save(frame.image_path)
    return frame


class _RegionalMessageClient:
    """Independent synthetic replies selected by the actual region request role."""

    def __init__(self, **overrides) -> None:
        self.payloads = {
            "header": {"readable": True, "text": _TEST_CONVERSATION},
            "composer": {"is_composer": True, "empty": True, "draft_text": ""},
            "messages": {
                "status": "sent", "marker": _TEST_MARKER,
                "message_candidate_id": "message-frame-e005", "outgoing": True, "send_state": "sent",
            },
            **overrides,
        }
        self.calls: list[dict] = []

    def generate_multimodal(self, instruction, **kwargs) -> ModelResponse:
        self.calls.append({"instruction": instruction, **kwargs})
        role = kwargs["context"]["region_role"]
        payload = self.payloads[role]
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return ModelResponse(content=content, model_name="synthetic-region-vision", provider="test")


def test_initial_regional_assessment_isolates_header_and_composer_from_old_messages(tmp_path: Path) -> None:
    frame = _regional_message_frame(tmp_path)
    before = Path(frame.image_path).read_bytes()
    client = _RegionalMessageClient()
    verdict = Verifier(message_client=client).check_message_context(_message_task(), frame, require_empty=True)
    assert verdict.passed
    assert [call["context"]["region_role"] for call in client.calls] == ["header", "composer"]
    assert verdict.evidence["assessment_scope"] == "recipient_and_composer"
    assert verdict.evidence["vision_assessment"] == {
        "conversation": _TEST_CONVERSATION, "header_box": [306, 46, 454, 79],
        "composer_box": [300, 350, 700, 500], "composer_empty": True,
    }
    prompt = json.dumps(client.calls, ensure_ascii=False)
    assert _TEST_CONVERSATION not in prompt and _TEST_MARKER not in prompt
    assert "WEEK4_MESSAGE_CHECK_OLD" not in prompt and "WEEK4_MESSAGE_CHECK_" not in prompt
    for call in client.calls:
        schema = call["response_format"]["json_schema"]["schema"]
        assert call["response_format"]["type"] == "json_schema"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert call["image_path"] != frame.image_path
        with Image.open(call["image_path"]) as crop:
            assert (255, 0, 255) not in {colour for _, colour in crop.getcolors(crop.width * crop.height)}
    assert Path(frame.image_path).read_bytes() == before
    assert [panel["label"] for panel in verdict.evidence["region_panels"]["panels"]] == ["HEADER", "COMPOSER"]


def test_final_regional_assessment_uses_three_requests_and_maps_observed_message_candidate(tmp_path: Path) -> None:
    frame = _regional_message_frame(tmp_path)
    client = _RegionalMessageClient()
    verdict = Verifier(message_client=client).check_task(_message_task(), frame)
    assert verdict.passed
    assert [call["context"]["region_role"] for call in client.calls] == ["header", "composer", "messages"]
    message_request = client.calls[2]
    assert message_request["context"]["message_candidate_ids"] == ["message-frame-e005", "message-frame-e004"]
    # Body coordinates must refer to the exact source candidate, not to a
    # neighbour visible in the halo of the contact-sheet panel.
    with Image.open(message_request["image_path"]) as attached, Image.open(frame.image_path) as source:
        for candidate in message_request["context"]["message_candidates"]:
            box = frame.element(candidate["candidate_id"]).bounding_box
            local = candidate["candidate_box"]
            panel = candidate["panel_box"]
            assert panel[0] <= local[0] < local[2] <= panel[2]
            assert panel[1] <= local[1] < local[3] <= panel[3]
            assert attached.crop(tuple(local)).tobytes() == source.crop(
                (box.left, box.top, box.right, box.bottom),
            ).tobytes()
    assert message_request["response_format"]["json_schema"]["schema"]["properties"]["message_candidate_id"]["enum"] == [
        "", "message-frame-e005", "message-frame-e004",
    ]
    assert verdict.evidence["vision_assessment"]["message_box"] == [400, 280, 680, 310]
    assert set(verdict.evidence["vision_assessment"]) == {
        "status", "conversation", "marker", "header_box", "message_box", "composer_box", "composer_empty",
        "outgoing", "send_state",
    }
    assert len(verdict.evidence["assessment_responses"]) == 3
    assert verdict.evidence["assessment_coordinate_space"] == "observed_region_ids"
    assert _TEST_MARKER not in json.dumps(client.calls, ensure_ascii=False)
    assert _TEST_CONVERSATION not in json.dumps(client.calls, ensure_ascii=False)
    assert all(Path(item["image_path"]).is_file() for item in verdict.evidence["assessment_requests"])


def test_overwritten_derived_image_cannot_back_cached_assessment(tmp_path: Path) -> None:
    frame = _regional_message_frame(tmp_path)
    client = _RegionalMessageClient()
    verifier = Verifier(message_client=client)
    first = verifier.check_task(_message_task(), frame)
    assert first.passed and len(client.calls) == 3
    changed = Path(first.evidence["assessment_requests"][-1]["image_path"])
    with Image.open(changed) as pixels:
        Image.new("RGB", pixels.size, "red").save(changed)
    second = verifier.check_task(_message_task(), frame)
    assert second.passed and len(client.calls) == 6
    assert not second.evidence.get("assessment_cached")
    import hashlib
    assert second.evidence["assessment_image_sha256"][str(changed)] == hashlib.sha256(changed.read_bytes()).hexdigest()


@pytest.mark.parametrize("update", [
    {"message_candidate_id": "message-frame-e999"},
    {"message_candidate_id": "message-frame-e002"},
    {"message_candidate_id": ""},
    {"send_state": "pending"},
    {"send_state": "failed"},
    {"send_state": "unavailable"},
    {"outgoing": False},
    {"marker": "WEEK4_MESSAGE_CHECK_OLD"},
])
def test_regional_send_requires_current_bubble_id_exact_marker_and_completed_outgoing_state(
    tmp_path: Path, update: dict,
) -> None:
    client = _RegionalMessageClient()
    client.payloads["messages"].update(update)
    verdict = Verifier(message_client=client).check_task(_message_task(), _regional_message_frame(tmp_path))
    assert not verdict.passed
    assert verdict.outcome in {"failed", "inconclusive"}
    assert len(client.calls) == 3
    assert verdict.evidence["assessment_responses"][-1]["content"] == json.dumps(client.payloads["messages"], ensure_ascii=False)


@pytest.mark.parametrize("role,content", [
    ("header", '{"readable":"true","text":"x"}'),
    ("header", '{"readable":true,"text":123}'),
    ("header", '{"readable":false,"text":""}'),
    ("header", '{"readable":true,"text":"x","readable":true}'),
    ("header", '{"readable":true,"text":"x","header_is_conversation_title":true}'),
    ("composer", '{"is_composer":false,"empty":true,"draft_text":""}'),
    ("composer", '{"is_composer":true,"empty":"true","draft_text":""}'),
    ("composer", '{"is_composer":true,"empty":true,"draft_text":"a draft"}'),
    ("composer", '{"is_composer":true,"empty":false,"draft_text":null}'),
    ("messages", '{"status":"success","marker":"","message_candidate_id":"","outgoing":true,"send_state":"sent"}'),
    ("messages", '{"status":"sent","marker":[],"message_candidate_id":"message-frame-e005","outgoing":true,"send_state":"sent"}'),
    ("messages", '{"status":"sent","marker":"x","message_candidate_id":"message-frame-e005","outgoing":1,"send_state":"sent"}'),
    ("messages", '{"status":"sent","marker":"x","message_candidate_id":"message-frame-e005","outgoing":true,"send_state":"success"}'),
    ("messages", '{"status":"draft","marker":"x","message_candidate_id":"","outgoing":false,"send_state":"unavailable"}'),
    ("messages", '{"status":"not_found","marker":"","message_candidate_id":"message-frame-e005","outgoing":false,"send_state":"unavailable"}'),
])
def test_regional_json_roles_types_and_draft_consistency_are_strict(
    tmp_path: Path, role: str, content: str,
) -> None:
    client = _RegionalMessageClient(**{role: content})
    verdict = Verifier(message_client=client).check_task(_message_task(), _regional_message_frame(tmp_path))
    assert verdict.outcome == "inconclusive"
    assert "unusable" in verdict.detail
    assert verdict.evidence["assessment_responses"][-1]["content"] == content


def test_initial_regional_assessment_rejects_draft_without_requesting_messages(tmp_path: Path) -> None:
    client = _RegionalMessageClient(composer={"is_composer": True, "empty": False, "draft_text": "unfinished draft"})
    verdict = Verifier(message_client=client).check_message_context(
        _message_task(), _regional_message_frame(tmp_path), require_empty=True,
    )
    assert verdict.outcome == "failed"
    assert "must be empty" in verdict.detail
    assert [call["context"]["region_role"] for call in client.calls] == ["header", "composer"]


def test_initial_and_final_regional_cache_scopes_do_not_reuse_each_other(tmp_path: Path) -> None:
    frame = _regional_message_frame(tmp_path)
    client = _RegionalMessageClient()
    verifier = Verifier(message_client=client)
    initial = verifier.check_message_context(_message_task(), frame, require_empty=True)
    final = verifier.check_task(_message_task(), frame)
    assert initial.passed and final.passed
    assert len(client.calls) == 5
    assert initial.evidence["image_path"] != final.evidence["image_path"]
    again_initial = verifier.check_message_context(_message_task(), frame, require_empty=True)
    again_final = verifier.check_task(_message_task(), frame)
    assert again_initial.evidence["assessment_cached"] is True
    assert again_final.evidence["assessment_cached"] is True
    assert len(client.calls) == 5
    assert "marker" not in again_initial.evidence["vision_assessment"]
    assert again_final.evidence["vision_assessment"]["marker"] == _TEST_MARKER


@pytest.mark.parametrize("change", ["pixels", "header", "composer", "message"])
def test_regional_pixel_or_roi_changes_trigger_new_assessments_and_distinct_evidence(
    tmp_path: Path, change: str,
) -> None:
    frame = _regional_message_frame(tmp_path)
    client = _RegionalMessageClient()
    verifier = Verifier(message_client=client)
    first = verifier.check_task(_message_task(), frame)
    assert first.passed
    if change == "pixels":
        with Image.open(frame.image_path) as pixels:
            changed = pixels.copy()
        changed.putpixel((410, 290), (1, 2, 3))
        changed.save(frame.image_path)
    else:
        index = {"header": 1, "composer": 0, "message": 4}[change]
        item = frame.elements[index]
        old = item.bounding_box
        box = BoundingBox(left=old.left + 1, top=old.top, right=old.right, bottom=old.bottom)
        frame.elements[index] = item.model_copy(update={"bounding_box": box, "center": box.center})
    second = verifier.check_task(_message_task(), frame)
    assert second.passed
    assert len(client.calls) == 6
    assert second.evidence["image_path"] != first.evidence["image_path"]
    assert Path(first.evidence["image_path"]).is_file()
    assert Path(second.evidence["image_path"]).is_file()
    assert first.evidence["region_panels"] != second.evidence["region_panels"]


@pytest.mark.parametrize("role", ["header", "composer", "messages"])
def test_missing_cached_regional_request_image_is_rebuilt_before_evidence_is_reused(tmp_path: Path, role: str) -> None:
    frame = _regional_message_frame(tmp_path)
    client = _RegionalMessageClient()
    verifier = Verifier(message_client=client)
    first = verifier.check_task(_message_task(), frame)
    assert first.passed
    request = next(item for item in first.evidence["assessment_requests"] if item["role"] == role)
    missing = Path(request["image_path"])
    missing.unlink()
    second = verifier.check_task(_message_task(), frame)
    assert second.passed
    assert missing.is_file()
    assert len(client.calls) == 6
    assert not second.evidence.get("assessment_cached", False)
    assert verifier.check_task(_message_task(), frame).evidence["assessment_cached"] is True
    assert len(client.calls) == 6


@pytest.mark.parametrize("role", ["header", "composer", "messages"])
def test_malformed_regional_reply_keeps_actual_raw_content_and_request_evidence(tmp_path: Path, role: str) -> None:
    frame = _regional_message_frame(tmp_path)
    raw = "```json\n{broken:response}\n```"
    client = _RegionalMessageClient(**{role: raw})
    verifier = Verifier(message_client=client)
    first = verifier.check_task(_message_task(), frame)
    assert first.outcome == "inconclusive"
    responses = first.evidence["assessment_responses"]
    assert responses[-1]["role"] == role and responses[-1]["content"] == raw
    assert raw in first.evidence["assessment_raw_content"]
    assert all(Path(item["image_path"]).is_file() for item in first.evidence["assessment_requests"])
    assert "vision_assessment" not in first.evidence
    count = len(client.calls)
    second = verifier.check_task(_message_task(), frame)
    assert second.outcome == "inconclusive" and second.evidence["assessment_cached"] is True
    assert second.evidence["assessment_responses"] == responses
    assert len(client.calls) == count
