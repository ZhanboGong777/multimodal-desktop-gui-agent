"""Region geometry isolates observed pixels without granting new action targets."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from PIL import Image

from gui_agent.runtime.message_regions import (
    MessageRegionError,
    render_message_regions,
    select_message_regions,
)
from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot
from gui_agent.schemas import BoundingBox, Point, ScreenInfo


def _element(
    number: int,
    coordinates: tuple[int, int, int, int],
    *,
    source: str = "contour",
    text: str = "",
    confidence: float = 0.5,
) -> ElementRef:
    box = BoundingBox(
        left=coordinates[0], top=coordinates[1], right=coordinates[2], bottom=coordinates[3],
    )
    return ElementRef(
        element_id=f"obs-0001-e{number:03d}", text=text, bounding_box=box,
        center=box.center, confidence=confidence, source=source,
    )


def _snapshot() -> ObservationSnapshot:
    return ObservationSnapshot(
        observation_id="obs-0001", captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=800, screenshot_height=600, control_width=800, control_height=600,
        ),
        window_id="123:456", window_title="Message client", window_class="ClientWindow",
        window_bounds=BoundingBox(left=80, top=40, right=720, bottom=540),
        foreground_stable=True,
        elements=[
            _element(1, (300, 400, 700, 520)),
            _element(2, (310, 90, 430, 112), source="ocr", text="PRIVATE_HEADER_TEXT"),
            # This sidebar row precedes the actual header in reading order.
            _element(3, (100, 65, 230, 87), source="ocr", text="PRIVATE_HEADER_TEXT"),
            _element(4, (400, 240, 680, 280)),
            _element(5, (400, 320, 680, 360)),
            _element(6, (500, 135, 550, 154), source="ocr", text="12:34"),
        ],
    )


def _with_pixels(snapshot: ObservationSnapshot, tmp_path) -> np.ndarray:
    """Use position-dependent RGB values so even one-pixel mapping errors are visible."""
    yy, xx = np.indices((600, 800))
    pixels = np.stack((xx % 256, yy % 256, (xx * 3 + yy * 5) % 256), axis=-1).astype(np.uint8)
    destination = tmp_path / "source.png"
    Image.fromarray(pixels).save(destination)
    snapshot.image_path = str(destination)
    return pixels


def test_selects_observed_composer_header_and_latest_first_messages() -> None:
    snapshot = _snapshot()
    regions = select_message_regions(snapshot)
    assert regions.observation_id == snapshot.observation_id
    assert regions.composer is snapshot.elements[0]
    assert regions.header_element_ids == ("obs-0001-e002",)
    assert regions.header_box == BoundingBox(left=306, top=86, right=434, bottom=116)
    assert [item.element_id for item in regions.messages] == ["obs-0001-e005", "obs-0001-e004"]


@pytest.mark.parametrize("dx,dy", [(30, 50), (-20, -25)])
def test_region_selection_translates_with_observed_window(dx: int, dy: int) -> None:
    snapshot = _snapshot()
    original = select_message_regions(snapshot)

    def translate(box: BoundingBox) -> BoundingBox:
        return BoundingBox(
            left=box.left + dx, top=box.top + dy, right=box.right + dx, bottom=box.bottom + dy,
        )

    snapshot.window_bounds = translate(snapshot.window_bounds)
    snapshot.elements = [
        item.model_copy(update={
            "bounding_box": translate(item.bounding_box),
            "center": translate(item.bounding_box).center,
        })
        for item in snapshot.elements
    ]
    moved = select_message_regions(snapshot)
    assert moved.header_box == translate(original.header_box)
    assert moved.header_element_ids == original.header_element_ids
    assert moved.composer.bounding_box == translate(original.composer.bounding_box)
    assert [item.bounding_box for item in moved.messages] == [
        translate(item.bounding_box) for item in original.messages
    ]


def test_header_merges_same_row_fragments_and_excludes_date_and_sidebar() -> None:
    snapshot = _snapshot()
    snapshot.elements[1] = _element(2, (310, 90, 365, 110), source="ocr", text="First")
    snapshot.elements.append(_element(7, (369, 92, 460, 114), source="ocr", text="Second"))
    regions = select_message_regions(snapshot)
    assert regions.header_element_ids == ("obs-0001-e002", "obs-0001-e007")
    assert regions.header_box == BoundingBox(left=306, top=86, right=464, bottom=118)


def test_header_padding_stays_in_observed_chat_span_and_foreground() -> None:
    snapshot = _snapshot()
    snapshot.elements[1] = _element(2, (300, 41, 700, 63), source="ocr", text="Header")
    regions = select_message_regions(snapshot)
    assert regions.header_box == BoundingBox(left=300, top=40, right=700, bottom=67)


@pytest.mark.parametrize("second", [
    (320, 430, 700, 530),
    (300, 400, 700, 520),
])
def test_ambiguous_composer_candidates_are_refused_even_with_duplicate_geometry(second) -> None:
    snapshot = _snapshot()
    snapshot.elements.append(_element(7, second))
    with pytest.raises(MessageRegionError, match="exactly one"):
        select_message_regions(snapshot)


def test_missing_composer_is_refused_without_coordinate_fallback() -> None:
    snapshot = _snapshot()
    snapshot.elements.pop(0)
    with pytest.raises(MessageRegionError, match="exactly one"):
        select_message_regions(snapshot)


def test_missing_header_is_refused_when_only_sidebar_ocr_remains() -> None:
    snapshot = _snapshot()
    snapshot.elements = [item for item in snapshot.elements if item.source != "ocr" or
                         item.element_id == "obs-0001-e003"]
    with pytest.raises(MessageRegionError, match="upper OCR row"):
        select_message_regions(snapshot)


@pytest.mark.parametrize("coordinates", [
    (310, 166, 430, 188),  # below the foreground's upper quarter
    (290, 90, 430, 112),  # extends into the sidebar
    (600, 90, 710, 112),  # extends outside the composer horizontal span
])
def test_header_requires_one_complete_ocr_box_in_the_bounded_upper_chat_span(coordinates) -> None:
    snapshot = _snapshot()
    snapshot.elements = [snapshot.elements[0], _element(2, coordinates, source="ocr", text="H")]
    with pytest.raises(MessageRegionError, match="upper OCR row"):
        select_message_regions(snapshot)


@pytest.mark.parametrize("coordinates", [
    (299, 240, 680, 280),  # one pixel outside the chat span
    (400, 240, 701, 280),
    (400, 115, 680, 155),  # overlaps the header crop
    (400, 370, 680, 401),  # overlaps the composer
    (400, 240, 479, 280),  # narrower than 20 percent of composer
    (400, 240, 680, 257),  # shorter than 18 native pixels
    (300, 116, 700, 400),  # enclosing transcript contour
    (50, 240, 350, 280),   # extends outside the foreground
])
def test_invalid_or_enclosing_message_regions_are_excluded(coordinates) -> None:
    snapshot = _snapshot()
    snapshot.elements = snapshot.elements[:3] + [_element(7, coordinates)]
    regions = select_message_regions(snapshot)
    assert not regions.messages


def test_message_candidates_keep_real_bounds_and_deduplicate_near_contour_borders() -> None:
    snapshot = _snapshot()
    snapshot.elements.append(_element(7, (400, 320, 680, 360)))
    snapshot.elements.append(_element(8, (401, 321, 679, 359)))
    regions = select_message_regions(snapshot)
    assert [item.element_id for item in regions.messages] == ["obs-0001-e005", "obs-0001-e004"]
    assert regions.messages[0].bounding_box == snapshot.elements[4].bounding_box


def test_message_limit_keeps_ten_latest_observed_candidates() -> None:
    snapshot = _snapshot()
    snapshot.elements = snapshot.elements[:3] + [
        _element(100 + index, (400, 130 + index * 20, 680, 148 + index * 20))
        for index in range(12)
    ]
    regions = select_message_regions(snapshot)
    assert len(regions.messages) == 10
    assert [item.element_id for item in regions.messages] == [
        f"obs-0001-e{index:03d}" for index in range(111, 101, -1)
    ]


def test_low_confidence_bad_centers_and_non_contours_cannot_supply_message_evidence() -> None:
    snapshot = _snapshot()
    snapshot.elements[3].confidence = 0.34
    snapshot.elements[4].center = Point(x=1, y=1)
    snapshot.elements.append(_element(7, (400, 200, 680, 225), source="ocr", text="Message"))
    snapshot.elements.append(_element(8, (400, 160, 680, 185), source="manual"))
    assert not select_message_regions(snapshot).messages


@pytest.mark.parametrize("field,value", [
    ("foreground_stable", False), ("window_id", ""), ("window_title", ""),
    ("window_class", ""), ("window_bounds", None), ("errors", ["capture degraded"]),
])
def test_unknown_or_degraded_foreground_is_refused(field, value) -> None:
    snapshot = _snapshot()
    setattr(snapshot, field, value)
    with pytest.raises(MessageRegionError, match="stable, readable"):
        select_message_regions(snapshot)


def test_foreground_outside_screenshot_is_refused() -> None:
    snapshot = _snapshot()
    snapshot.window_bounds = BoundingBox(left=-1, top=40, right=720, bottom=540)
    with pytest.raises(MessageRegionError, match="outside"):
        select_message_regions(snapshot)


@pytest.mark.parametrize("duplicate", [True, False])
def test_duplicate_or_stale_candidate_ids_are_refused(duplicate: bool) -> None:
    snapshot = _snapshot()
    snapshot.elements[3].element_id = snapshot.elements[4].element_id if duplicate else "obs-old-e004"
    with pytest.raises(MessageRegionError, match="element ids"):
        select_message_regions(snapshot)


def test_rendered_crops_preserve_native_pixels_and_reversible_layout(tmp_path) -> None:
    snapshot = _snapshot()
    original = _with_pixels(snapshot, tmp_path)
    regions = select_message_regions(snapshot)
    rendered = render_message_regions(snapshot, regions, tmp_path / "regions.png")
    with Image.open(rendered.image_path) as image:
        sheet = np.asarray(image)
        assert image.size == rendered.image_size
    assert [panel.label for panel in rendered.panels] == [
        "HEADER", "COMPOSER", "MESSAGE:obs-0001-e005", "MESSAGE:obs-0001-e004",
    ]
    for panel in rendered.panels:
        old, new = panel.source_box, panel.panel_box
        assert (old.width, old.height) == (new.width, new.height)
        np.testing.assert_array_equal(
            original[old.top:old.bottom, old.left:old.right],
            sheet[new.top:new.bottom, new.left:new.right],
        )
        transform = panel.as_dict()["source_to_panel"]
        assert transform["scale_x"] == transform["scale_y"] == 1.0
        assert old.left + transform["translate_x"] == new.left
        assert old.top + transform["translate_y"] == new.top
    # The renderer adds a plain matte alongside narrow crops, never guessed content.
    header = rendered.panels[0].panel_box
    assert np.all(sheet[header.top:header.bottom, header.right:] == 245)
    metadata = rendered.as_dict()
    assert metadata["native_pixels"] is True
    assert "PRIVATE_HEADER_TEXT" not in json.dumps(metadata)


def test_initial_context_sheet_excludes_all_message_panels(tmp_path) -> None:
    snapshot = _snapshot()
    _with_pixels(snapshot, tmp_path)
    regions = select_message_regions(snapshot)
    rendered = render_message_regions(
        snapshot, regions, tmp_path / "context.png", include_messages=False,
    )
    assert [panel.label for panel in rendered.panels] == ["HEADER", "COMPOSER"]
    assert rendered.image_size == (
        regions.composer.bounding_box.width,
        regions.header_box.height + regions.composer.bounding_box.height + 48,
    )


def test_message_halo_is_clamped_but_candidate_box_remains_observed(tmp_path) -> None:
    snapshot = _snapshot()
    snapshot.elements = snapshot.elements[:3] + [_element(7, (300, 116, 700, 140))]
    _with_pixels(snapshot, tmp_path)
    regions = select_message_regions(snapshot)
    rendered = render_message_regions(snapshot, regions, tmp_path / "halo.png")
    panel = rendered.panels[-1]
    assert panel.source_box == BoundingBox(left=300, top=116, right=700, bottom=172)
    assert panel.candidate_boxes == {"obs-0001-e007": snapshot.elements[-1].bounding_box}
    assert panel.as_dict()["candidate_boxes"]["obs-0001-e007"] == {
        "left": 300, "top": 116, "right": 700, "bottom": 140,
    }


def test_renderer_refuses_regions_from_a_different_frame(tmp_path) -> None:
    snapshot = _snapshot()
    _with_pixels(snapshot, tmp_path)
    regions = select_message_regions(snapshot)
    altered = replace(regions, observation_id="obs-0002")
    with pytest.raises(MessageRegionError, match="exact observation"):
        render_message_regions(snapshot, altered, tmp_path / "wrong.png")


def test_renderer_refuses_arbitrary_replacement_crop_geometry(tmp_path) -> None:
    snapshot = _snapshot()
    _with_pixels(snapshot, tmp_path)
    regions = select_message_regions(snapshot)
    altered = replace(regions, header_box=BoundingBox(left=90, top=50, right=300, bottom=150))
    with pytest.raises(MessageRegionError, match="exact observation"):
        render_message_regions(snapshot, altered, tmp_path / "wrong.png")


@pytest.mark.parametrize("problem", ["missing_path", "missing_file", "wrong_dimensions"])
def test_renderer_requires_an_actual_matching_source_image(tmp_path, problem) -> None:
    snapshot = _snapshot()
    if problem == "missing_file":
        snapshot.image_path = str(tmp_path / "absent.png")
    elif problem == "wrong_dimensions":
        path = tmp_path / "small.png"
        Image.new("RGB", (400, 300)).save(path)
        snapshot.image_path = str(path)
    with pytest.raises(MessageRegionError):
        render_message_regions(snapshot, select_message_regions(snapshot), tmp_path / "out.png")
    assert not (tmp_path / "out.png").exists()


@pytest.mark.parametrize("destination", ["source.png", "lossy.jpg"])
def test_renderer_cannot_overwrite_source_or_use_lossy_output(tmp_path, destination) -> None:
    snapshot = _snapshot()
    _with_pixels(snapshot, tmp_path)
    original = (tmp_path / "source.png").read_bytes()
    with pytest.raises(MessageRegionError):
        render_message_regions(snapshot, select_message_regions(snapshot), tmp_path / destination)
    assert (tmp_path / "source.png").read_bytes() == original
