"""Invariants of the observation service that do not need a screen.

`observe()` itself needs a desktop, so it is exercised through the runner and the
integration tests. What is left here is the part a run depends on before it ever
captures anything: the id a frame is addressed by, and which elements are offered
to the model as targets.
"""

from __future__ import annotations

from datetime import UTC, datetime

from gui_agent.config import Config
from gui_agent.runtime.observation import ObservationService, describe_elements
from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot
from gui_agent.schemas import BoundingBox, Point, ScreenInfo


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


def test_contours_are_not_offered_as_targets() -> None:
    """8.1.4: the element list is what the model aims with, so it holds real text.

    A bare contour box has no label. Listing it would invite the model to describe
    it as "the search box", which is an invention the adapter would then refuse -
    or worse, resolve against the wrong element.
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
                text="",
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

    assert "obs-0001-e001" in rendered
    assert "obs-0001-e000" not in rendered, "an unlabelled box is not a target"
    assert rendered.count("\n") == 0, "one line per offered element"


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
