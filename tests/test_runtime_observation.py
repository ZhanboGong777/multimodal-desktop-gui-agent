"""Invariants of the observation service that do not need a screen.

`observe()` itself needs a desktop, so it is exercised through the runner and the
integration tests. What is left here is the part a run depends on before it ever
captures anything: the id a frame is addressed by, and which elements are offered
to the model as targets.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from gui_agent.config import Config, load_config
from gui_agent.perception import capture as capture_module
from gui_agent.perception.capture import ForegroundWindowContext
from gui_agent.perception.ocr import EngineSelection, OcrError, OcrOutput
from gui_agent.recording import RunSession
from gui_agent.runtime import observation
from gui_agent.runtime.observation import (
    ObservationError,
    ObservationService,
    describe_elements,
    foreground_matches,
)
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot
from gui_agent.runtime.visual_grounding import GroundingError, match_candidate, validate_frames
from gui_agent.schemas import BoundingBox, Point, ScreenInfo, UIElement


def _service() -> ObservationService:
    # Constructing the service does not capture anything: the OCR engine is built
    # on the first observe(), which is why this needs no desktop.
    return ObservationService(Config())


def test_each_observation_gets_its_own_id() -> None:
    """7.4.2: two observations must not share an id.

    The id is how a step record points back at the frame its coordinate came from.
    A repeat would not just be untidy - it would make the evidence ambiguous, and
    the whole point of a frame-local id is that it can be checked.
    """
    service = _service()

    ids = [service.next_id() for _ in range(3)]

    assert ids == ["obs-0001", "obs-0002", "obs-0003"]
    assert len(set(ids)) == 3


@pytest.mark.parametrize("blank_text", ["", "   "])
def test_contours_are_offered_with_an_explicit_unlabelled_marker(blank_text: str) -> None:
    """T04 needs contours that OCR cannot label, without inventing their meaning.

    Whitespace is not a label either. Both forms must show the id and geometry
    that the visual model can compare against the screenshot.
    """
    snapshot = ObservationSnapshot(
        observation_id="obs-0001",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=100, screenshot_height=100, control_width=100, control_height=100
        ),
        elements=[
            ElementRef(
                element_id="obs-0001-e000",
                text=blank_text,
                bounding_box=BoundingBox(left=0, top=0, right=20, bottom=20),
                center=Point(x=10, y=10),
                confidence=0.9,
                source="contour",
            ),
            ElementRef(
                element_id="obs-0001-e001",
                text="Search",
                bounding_box=BoundingBox(left=30, top=0, right=80, bottom=20),
                center=Point(x=55, y=10),
                confidence=0.9,
            ),
        ],
    )

    rendered = describe_elements(snapshot)

    assert rendered.splitlines() == [
        "obs-0001-e000  <unlabelled box>  conf=0.90  center=(10,10)  box=(0,0,20,20)",
        "obs-0001-e001  'Search'  conf=0.90  center=(55,10)  box=(30,0,80,20)",
    ]
    assert snapshot.elements[0].text == blank_text, "the marker must not become observed text"


def test_the_element_line_carries_what_a_step_needs_to_aim() -> None:
    """The prompt line has to carry the id, the text and the box, in that order."""
    snapshot = ObservationSnapshot(
        observation_id="obs-0002",
        captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=100, screenshot_height=100, control_width=100, control_height=100
        ),
        elements=[
            ElementRef(
                element_id="obs-0002-e007",
                text="Fetch origin",
                bounding_box=BoundingBox(left=10, top=20, right=110, bottom=40),
                center=Point(x=60, y=30),
                confidence=0.94,
            )
        ],
    )

    line = describe_elements(snapshot)

    assert line.startswith("obs-0002-e007  'Fetch origin'")
    assert "conf=0.94" in line
    assert "center=(60,30)" in line
    assert "box=(10,20,110,40)" in line


# ───────── observe(), against a prepared frame instead of a screen ─────────
SCREEN = ScreenInfo(
    screenshot_width=200, screenshot_height=100, control_width=200, control_height=100
)


class _PreparedCapture:
    """One prepared frame, standing in for the screen."""

    def __init__(self) -> None:
        self.image = Image.new("RGB", (200, 100), "white")
        self.screen_info = SCREEN
        self.captured_at = datetime.now(UTC)
        self.image_path = None


class _StubEngine:
    """An OCR backend with a fixed answer."""

    name = "stub"

    def __init__(self, elements: list[UIElement] | None = None, error: str | None = None) -> None:
        self._elements = elements or []
        self._error = error

    def recognize(self, image: object, **kwargs: object) -> OcrOutput:
        if self._error:
            raise OcrError(self._error)
        return OcrOutput(elements=list(self._elements), engine=self.name, elapsed_ms=1.0)


def _element(text: str, index: int = 0) -> UIElement:
    return UIElement(
        text=text,
        bounding_box=BoundingBox(left=index * 20, top=0, right=index * 20 + 18, bottom=10),
        confidence=0.9,
    )


def _patch(
    monkeypatch: pytest.MonkeyPatch, engine: _StubEngine, *, notices: list[str] | None = None
) -> None:
    monkeypatch.setattr(observation, "capture_monitor", lambda *a, **k: _PreparedCapture())
    monkeypatch.setattr(
        observation,
        "create_ocr_engine",
        lambda config: EngineSelection(engine=engine, notices=list(notices or [])),
    )


def test_observe_builds_one_frame_from_a_prepared_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """16.1 asks for prepared screenshots rather than the operator's desktop.

    The whole of `observe()` had no test at all - it needs a screen, and the suite
    is not allowed to take one. A prepared frame plus a stubbed backend is what the
    section actually asks for, and it runs the id assignment, the geometry and the
    snapshot assembly for real.
    """
    _patch(monkeypatch, _StubEngine([_element("Fetch origin")]))

    snapshot = ObservationService(Config()).observe()

    assert snapshot.observation_id == "obs-0001"
    assert snapshot.ocr_engine == "stub"
    assert snapshot.image_path is None
    assert snapshot.screen_info.screenshot_width == 200
    assert [item.text for item in snapshot.elements if item.text] == ["Fetch origin"]
    assert snapshot.elements[0].element_id == "obs-0001-e000"
    assert snapshot.processing_time_ms >= 0.0


def test_observe_numbers_its_frames_in_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """7.4.2: two observations must not share an id, and the service is what mints
    them. `next_id` was tested; the path that actually uses it was not."""
    _patch(monkeypatch, _StubEngine([_element("Browser")]))
    service = ObservationService(Config())

    ids = [service.observe().observation_id for _ in range(3)]

    assert ids == ["obs-0001", "obs-0002", "obs-0003"]


def test_an_ocr_failure_is_recorded_on_the_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """7.3.8: every OCR error goes into the record.

    A frame with a broken text pass is still a frame - and the alternative, quietly
    reusing the previous frame's elements, would attach today's coordinates to
    yesterday's screen.
    """
    _patch(monkeypatch, _StubEngine(error="tesseract binary missing"))

    snapshot = ObservationService(Config()).observe()

    assert any("tesseract binary missing" in error for error in snapshot.errors), snapshot.errors
    assert snapshot.observation_id == "obs-0001", "the frame is still the current one"


def test_a_capture_failure_stops_the_frame_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    """No screenshot means no observation, and the caller has to hear about it."""

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("monitor_index 1 is out of range")

    monkeypatch.setattr(observation, "capture_monitor", boom)

    with pytest.raises(ObservationError, match="monitor_index 1 is out of range"):
        ObservationService(Config()).observe()


def test_the_element_cap_keeps_the_labelled_elements(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_select` ranks text ahead of unlabelled contours.

    That ordering is what stops a screenful of boxes from pushing the labels out of
    the prompt, even at a deliberately small cap or when contours have higher
    confidence than OCR.
    """
    labels = [
        _element(f"label {index}", index).model_copy(update={"confidence": 0.1})
        for index in range(5)
    ]
    blanks = [
        _element("", index).model_copy(update={"confidence": 1.0}) for index in range(80)
    ]
    _patch(monkeypatch, _StubEngine(labels + blanks))

    snapshot = ObservationService(Config(), max_elements=10).observe()

    assert len(snapshot.elements) == 10
    assert [item.text for item in snapshot.elements if item.text] == [
        f"label {index}" for index in range(5)
    ]
    assert all(item.text for item in snapshot.elements[:5])
    assert all(not item.text for item in snapshot.elements[5:])


@pytest.mark.parametrize("label_count", [66, 100, 301])
def test_week4_element_budget_keeps_labels_before_the_full_contour_pool(
    monkeypatch: pytest.MonkeyPatch, label_count: int
) -> None:
    """300 covers the measured 66 OCR elements plus all 200 configured contours.

    100 fixes the chosen OCR allowance; 301 exercises overflow, where labels
    still outrank even high-confidence contours instead of being displaced.
    The frame and both perception backends are prepared, never live.
    """
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "week4.yaml")
    assert config.execution.max_elements == 300
    assert config.perception.ui_detection.max_candidates == 200
    assert ObservationService(Config()).max_elements == config.execution.max_elements
    assert Config().execution.max_elements == config.execution.max_elements
    labels = [
        _element(f"label {index}", index % 10).model_copy(update={"confidence": 0.1})
        for index in range(label_count)
    ]
    contours = [
        _element("", index % 10).model_copy(update={"confidence": 1.0, "source": "contour"})
        for index in range(config.perception.ui_detection.max_candidates)
    ]
    _patch(monkeypatch, _StubEngine(labels))

    def detect(*args: object, **kwargs: object) -> list[UIElement]:
        assert kwargs["max_candidates"] == 200
        return contours

    monkeypatch.setattr(observation, "detect_ui_candidates", detect)

    snapshot = ObservationService(config, max_elements=config.execution.max_elements).observe()

    retained_labels = min(label_count, config.execution.max_elements)
    assert len(snapshot.elements) == min(label_count + len(contours), config.execution.max_elements)
    assert [item.text for item in snapshot.elements[:retained_labels]] == [
        f"label {index}" for index in range(retained_labels)
    ]
    assert all(not item.text for item in snapshot.elements[retained_labels:])
    assert sum(item.source == "contour" for item in snapshot.elements) == min(
        len(contours), config.execution.max_elements - retained_labels
    )


def test_a_contour_failure_is_recorded_and_the_frame_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contours are optional context; the labelled elements are the part that matters.

    Losing them narrows what the model can aim at, which is worth recording - but it
    is not a reason to throw away a frame that OCR read perfectly well.
    """
    _patch(monkeypatch, _StubEngine([_element("Browser")]))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("contour detection is unavailable")

    monkeypatch.setattr(observation, "detect_ui_candidates", boom)

    snapshot = ObservationService(Config()).observe()

    assert any("ui detection failed" in error for error in snapshot.errors), snapshot.errors
    assert [item.text for item in snapshot.elements if item.text] == ["Browser"]


def test_a_fallback_engine_says_so_in_the_frame_it_produced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """7.1.4 asks for the fallback engine to be recorded, not only its name.

    The engine selection produces a notice when PaddleOCR is unavailable and
    Tesseract takes over. `ObservationService` copied those into a `notices`
    property that nothing in the runtime read, so a run recorded
    `ocr_engine: tesseract` and lost the reason - which is the half that explains
    a slow or poor frame.
    """
    _patch(
        monkeypatch,
        _StubEngine([_element("Fetch origin")]),
        notices=["PaddleOCR unavailable (no module); falling back to Tesseract."],
    )

    snapshot = ObservationService(Config()).observe()

    assert snapshot.notices == [
        "PaddleOCR unavailable (no module); falling back to Tesseract."
    ]
    # A fallback that worked is not a degraded frame. `errors` is what makes the
    # verifier refuse to judge an observation, and this must not land there.
    assert snapshot.errors == []


def test_a_frame_from_an_engine_that_reported_nothing_carries_no_notices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common case stays quiet, so a notice in a record means something."""
    _patch(monkeypatch, _StubEngine([_element("Browser")]))

    assert ObservationService(Config()).observe().notices == []


def _visual_frame(
    tmp_path: Path,
    observation_id: str,
    *,
    shift: tuple[int, int] = (0, 0),
    boxes: tuple[tuple[int, int, str], ...] = ((60, 60, "normal"),),
) -> ObservationSnapshot:
    """Prepared control images, never a screenshot or a desktop action."""
    image = Image.new("RGB", (360, 240), "white")
    draw = ImageDraw.Draw(image)
    refs = []
    for index, (left, top, appearance) in enumerate(boxes):
        left += shift[0]
        top += shift[1]
        box = BoundingBox(left=left, top=top, right=left + 40, bottom=top + 40)
        if appearance != "blank":
            colour = (40, 100, 180) if appearance == "normal" else (190, 70, 40)
            draw.rectangle((left, top, left + 39, top + 39), fill=colour, outline="black", width=2)
            draw.line((left + 8, top + 8, left + 29, top + 29), fill="white", width=4)
            draw.line((left + 8, top + 29, left + 29, top + 8), fill="black", width=3)
        refs.append(
            ElementRef(
                element_id=f"{observation_id}-e{index:03d}", text="", source="contour",
                bounding_box=box, center=box.center, confidence=0.5,
            )
        )
    path = tmp_path / f"{observation_id}.png"
    image.save(path)
    return ObservationSnapshot(
        observation_id=observation_id,
        captured_at=datetime.now(UTC),
        image_path=str(path),
        screen_info=ScreenInfo(
            screenshot_width=360, screenshot_height=240, control_width=360, control_height=240,
        ),
        elements=refs,
        window_id="12:1001", window_title="test conversation", window_class="TestMessenger",
        window_bounds=BoundingBox(
            left=20 + shift[0], top=20 + shift[1], right=320 + shift[0], bottom=210 + shift[1],
        ),
        foreground_stable=True,
    )


@pytest.mark.parametrize("shift", [(0, 0), (19, 11)])
def test_anonymous_identity_uses_current_pixels_and_id_after_translation(
    tmp_path: Path, shift: tuple[int, int],
) -> None:
    """Window motion and changed candidate indices must not reuse an old coordinate."""
    source = _visual_frame(tmp_path, "obs-0001")
    current = _visual_frame(
        tmp_path, "obs-0002", shift=shift,
        boxes=((180, 110, "different"), (60, 60, "normal")),
    )

    result = match_candidate(source, current, "obs-0001-e000")

    assert result.element_id == "obs-0002-e001"
    assert result.center == Point(x=80 + shift[0], y=80 + shift[1])
    assert source.elements[0].element_id == "obs-0001-e000"


@pytest.mark.parametrize(
    "change",
    ["missing", "appearance", "duplicate", "source", "size", "center", "confidence", "outside"],
)
def test_anonymous_identity_refuses_missing_changed_or_ambiguous_candidates(
    tmp_path: Path, change: str,
) -> None:
    """An enabled send control requires current model selection if its pixels changed."""
    source = _visual_frame(tmp_path, "obs-0001")
    boxes = ((60, 60, "normal"),)
    if change == "appearance":
        boxes = ((60, 60, "different"),)
    if change == "duplicate":
        boxes = ((60, 60, "normal"), (180, 110, "normal"))
    current = _visual_frame(tmp_path, "obs-0002", boxes=boxes)
    candidate = current.elements[0]
    updates: dict = {}
    if change == "missing":
        current.elements = []
    elif change == "source":
        updates["source"] = "ocr"
    elif change == "size":
        updates["bounding_box"] = BoundingBox(left=60, top=60, right=120, bottom=100)
    elif change == "center":
        updates["center"] = Point(x=65, y=65)
    elif change == "confidence":
        updates["confidence"] = 0.1
    elif change == "outside":
        updates["bounding_box"] = BoundingBox(left=0, top=0, right=40, bottom=40)
        updates["center"] = Point(x=20, y=20)
    if updates:
        current.elements = [candidate.model_copy(update=updates)]

    with pytest.raises(GroundingError):
        match_candidate(source, current, "obs-0001-e000")


def test_a_blank_crop_cannot_manufacture_a_perfect_visual_match(tmp_path: Path) -> None:
    source = _visual_frame(tmp_path, "obs-0001", boxes=((60, 60, "blank"),))
    current = _visual_frame(tmp_path, "obs-0002", boxes=((60, 60, "blank"),))

    with pytest.raises(GroundingError, match="texture"):
        match_candidate(source, current, "obs-0001-e000")


def test_a_disappearing_control_cannot_be_replaced_by_its_surviving_visual_clone(
    tmp_path: Path,
) -> None:
    source = _visual_frame(
        tmp_path, "obs-0001", boxes=((60, 60, "normal"), (180, 110, "normal")),
    )
    current = _visual_frame(tmp_path, "obs-0002", boxes=((180, 110, "normal"),))

    with pytest.raises(GroundingError, match="source candidate is visually ambiguous"):
        match_candidate(source, current, "obs-0001-e000")


@pytest.mark.parametrize(
    "change",
    [
        "errors", "stable", "identity", "unknown_identity", "title", "class", "bounds", "resize",
        "scale", "duplicate_id", "foreign_id",
    ],
)
def test_visual_identity_refuses_invalid_frame_pairs(tmp_path: Path, change: str) -> None:
    source = _visual_frame(tmp_path, "obs-0001")
    current = _visual_frame(tmp_path, "obs-0002")
    updates = {
        "errors": {"errors": ["ui detection failed"]},
        "stable": {"foreground_stable": False},
        "identity": {"window_id": "12:1002"},
        "unknown_identity": {"window_id": ""},
        "title": {"window_title": "another conversation"},
        "class": {"window_class": "FileDialog"},
        "bounds": {"window_bounds": None},
        "resize": {"window_bounds": BoundingBox(left=20, top=20, right=330, bottom=210)},
        "scale": {"screen_info": current.screen_info.model_copy(update={"scale_x": 0.5})},
        "duplicate_id": {"elements": [current.elements[0], current.elements[0]]},
        "foreign_id": {
            "elements": [current.elements[0].model_copy(update={"element_id": "obs-0099-e000"})],
        },
    }

    with pytest.raises(GroundingError):
        validate_frames(source, current.model_copy(update=updates[change]))


@pytest.mark.parametrize("change", ["missing", "unreadable", "wrong_dimensions"])
def test_visual_identity_requires_readable_image_evidence(tmp_path: Path, change: str) -> None:
    source = _visual_frame(tmp_path, "obs-0001")
    current = _visual_frame(tmp_path, "obs-0002")
    if change == "missing":
        current.image_path = None
    elif change == "unreadable":
        Path(current.image_path).write_bytes(b"not an image")
    else:
        Image.new("RGB", (30, 20), "white").save(current.image_path)

    with pytest.raises(GroundingError, match="image"):
        validate_frames(source, current)


def test_a_candidate_cannot_be_rebound_from_an_unrelated_source_id(tmp_path: Path) -> None:
    source = _visual_frame(tmp_path, "obs-0001")
    current = _visual_frame(tmp_path, "obs-0002")

    with pytest.raises(GroundingError, match="absent"):
        match_candidate(source, current, "obs-0099-e000")


def test_observation_forwards_consistent_foreground_and_prioritises_its_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _StubEngine([_element("Conversation")]))
    capture = _PreparedCapture()
    capture.window_id = "22:1200"
    capture.window_title = "Conversation"
    capture.window_class = "TestMessenger"
    capture.window_bounds = BoundingBox(left=20, top=10, right=180, bottom=90)
    capture.foreground_stable = True
    monkeypatch.setattr(observation, "capture_monitor", lambda *a, **k: capture)

    def detect(*args: object, **kwargs: object) -> list[UIElement]:
        assert kwargs["priority_region"] == capture.window_bounds
        return []

    monkeypatch.setattr(observation, "detect_ui_candidates", detect)
    snapshot = ObservationService(Config()).observe()

    assert snapshot.window_id == capture.window_id
    assert snapshot.window_bounds == capture.window_bounds
    assert snapshot.foreground_stable


@pytest.mark.parametrize("change", ["same", "identity", "title", "class", "motion", "unknown"])
def test_dispatch_foreground_check_refuses_post_capture_switch_or_motion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str,
) -> None:
    snapshot = _visual_frame(tmp_path, "obs-0001")
    context = ForegroundWindowContext(
        window_id=snapshot.window_id, window_title=snapshot.window_title,
        window_class=snapshot.window_class, window_bounds=snapshot.window_bounds,
    )
    replacements = {
        "same": context,
        "identity": ForegroundWindowContext(
            "12:1002", context.window_title, context.window_class, context.window_bounds,
        ),
        "title": ForegroundWindowContext(
            context.window_id, "another conversation", context.window_class, context.window_bounds,
        ),
        "class": ForegroundWindowContext(
            context.window_id, context.window_title, "FileDialog", context.window_bounds,
        ),
        "motion": ForegroundWindowContext(
            context.window_id, context.window_title, context.window_class,
            BoundingBox(left=30, top=20, right=330, bottom=210),
        ),
        "unknown": ForegroundWindowContext(),
    }
    monkeypatch.setattr(observation, "foreground_window_context", lambda: replacements[change])

    assert foreground_matches(snapshot) is (change == "same")
    assert ObservationService(Config()).foreground_matches(snapshot) is (change == "same")


@pytest.mark.parametrize("change", ["same", "identity", "title", "class", "motion"])
def test_capture_brackets_pixels_with_one_consistent_foreground_context(
    monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    """Fake MSS and fake Win32 metadata keep this capture regression fully offline."""
    before = ForegroundWindowContext(
        "12:1001", "test", "TestMessenger", BoundingBox(left=10, top=10, right=90, bottom=70),
    )
    after = {
        "same": before,
        "identity": ForegroundWindowContext("12:1002", "test", "TestMessenger", before.window_bounds),
        "title": ForegroundWindowContext("12:1001", "changed", "TestMessenger", before.window_bounds),
        "class": ForegroundWindowContext("12:1001", "test", "FileDialog", before.window_bounds),
        "motion": ForegroundWindowContext(
            "12:1001", "test", "TestMessenger",
            BoundingBox(left=20, top=10, right=100, bottom=70),
        ),
    }[change]
    contexts = iter((before, after))

    class Session:
        def __enter__(self):
            self.monitors = [{}, {"left": 0, "top": 0, "width": 200, "height": 100}]
            return self

        def __exit__(self, *args):
            return False

        def grab(self, monitor):
            return type("Shot", (), {"size": (200, 100), "rgb": bytes(200 * 100 * 3)})()

    monkeypatch.setattr(capture_module.mss, "mss", Session)
    monkeypatch.setattr(capture_module, "foreground_window_context", lambda: next(contexts))
    frame = capture_module.capture_monitor(control_size=(100, 50))

    assert frame.window_id == after.window_id
    assert frame.foreground_stable is (change == "same")
    assert frame.window_bounds == BoundingBox(
        left=after.window_bounds.left * 2, top=after.window_bounds.top * 2,
        right=after.window_bounds.right * 2, bottom=after.window_bounds.bottom * 2,
    )


def test_recorded_observation_keeps_window_identity_used_to_authorise_a_target(tmp_path: Path) -> None:
    snapshot = _visual_frame(tmp_path, "obs-0001")
    recorder = TaskRecorder(RunSession.create(tmp_path / "run", session_id="visual-test"))

    path = recorder.save_observation(snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["window_id"] == snapshot.window_id
    assert payload["window_bounds"] == snapshot.window_bounds.model_dump()
    assert payload["foreground_stable"] is True
