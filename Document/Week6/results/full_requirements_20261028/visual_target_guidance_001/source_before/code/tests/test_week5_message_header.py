"""Measured title layout must isolate current pixels without recipient matching."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from PIL import Image, ImageDraw

from gui_agent.runtime.message_regions import (
    MessageRegionError,
    render_message_regions,
    select_message_regions,
)
from gui_agent.runtime.schemas import ElementRef, ObservationSnapshot
from gui_agent.schemas import BoundingBox, ScreenInfo


def element(number, coordinates, *, text="", source="contour", confidence=0.5):
    box = BoundingBox(left=coordinates[0], top=coordinates[1], right=coordinates[2], bottom=coordinates[3])
    return ElementRef(
        element_id=f"obs-0001-e{number:03d}", text=text, bounding_box=box,
        center=box.center, source=source, confidence=confidence,
    )


def measured_frame():
    """Portable recorded geometry from T04_20261009_083042_910048_75cebb04."""
    return ObservationSnapshot(
        observation_id="obs-0001", captured_at=datetime.now(UTC),
        screen_info=ScreenInfo(
            screenshot_width=2560, screenshot_height=1600,
            control_width=2560, control_height=1600,
        ),
        window_id="observed:foreground", window_title="Observed client", window_class="ObservedWindow",
        window_bounds=BoundingBox(left=813, top=76, right=2155, bottom=1047),
        foreground_stable=True,
        elements=[
            element(127, (1284, 820, 2127, 1019)),
            element(46, (1295, 145, 1425, 169), text="ARBITRARY_TITLE", source="ocr"),
            element(100, (2100, 90, 2120, 108), text="X", source="ocr"),
            element(5, (1000, 411, 1150, 434), text="ARBITRARY_TITLE", source="ocr"),
            element(6, (2050, 147, 2080, 170), text="...", source="ocr"),
            element(7, (2210, 131, 2280, 159), text="OTHER_WINDOW", source="ocr"),
            element(8, (1652, 228, 1760, 249), text="12:34", source="ocr"),
            element(131, (1437, 281, 2040, 337)),
        ],
    )


def test_real_geometry_chooses_content_title_instead_of_close_sidebar_or_other_window():
    frame = measured_frame()
    selected = select_message_regions(frame)
    assert selected.header_element_ids == ("obs-0001-e046",)
    assert selected.header_box == BoundingBox(left=1291, top=141, right=1429, bottom=173)
    assert selected.composer is frame.elements[0]
    assert selected.messages == (frame.elements[-1],)


@pytest.mark.parametrize("title,chrome,sidebar", [
    ("X", "ARBITRARY_TITLE", "X"),
    ("Another conversation", "Another conversation", "Another conversation"),
    ("文件传输助手", "文件传输助手", "文件传输助手"),
    ("a", "Close", "Other"),
])
def test_changing_ocr_strings_does_not_change_layout_selection(title, chrome, sidebar):
    frame = measured_frame()
    frame.elements[1].text, frame.elements[2].text, frame.elements[3].text = title, chrome, sidebar
    assert select_message_regions(frame).header_element_ids == ("obs-0001-e046",)


@pytest.mark.parametrize("factor,dx,dy,monitor_left,monitor_top", [
    (1, 0, 0, 0, 0), (2, 100, 60, -2560, 120), (1, -300, 100, 2560, -500),
])
def test_header_layout_uses_native_foreground_geometry_with_scale_and_monitor_offsets(
    factor, dx, dy, monitor_left, monitor_top,
):
    frame = measured_frame()

    def transform(box):
        return BoundingBox(
            left=box.left * factor + dx, top=box.top * factor + dy,
            right=box.right * factor + dx, bottom=box.bottom * factor + dy,
        )

    frame.window_bounds = transform(frame.window_bounds)
    frame.screen_info = ScreenInfo(
        screenshot_width=2560 * factor + max(dx, 0), screenshot_height=1600 * factor + max(dy, 0),
        control_width=2560, control_height=1600,
        monitor_left=monitor_left, monitor_top=monitor_top, scale_x=1 / factor, scale_y=1 / factor,
    )
    frame.elements = [
        item.model_copy(update={"bounding_box": transform(item.bounding_box),
                                "center": transform(item.bounding_box).center})
        for item in frame.elements
    ]
    selected = select_message_regions(frame)
    title_box = frame.elements[1].bounding_box
    assert selected.header_element_ids == ("obs-0001-e046",)
    assert selected.header_box == BoundingBox(
        left=title_box.left - 4, top=title_box.top - 4,
        right=title_box.right + 4, bottom=title_box.bottom + 4,
    )


@pytest.mark.parametrize("remaining", ["chrome", "left_message", "timestamp", "all"])
def test_missing_title_never_falls_back_to_chrome_or_first_transcript_row(remaining):
    frame = measured_frame()
    options = {
        "chrome": [element(100, (2100, 90, 2120, 108), text="ARBITRARY_TITLE", source="ocr")],
        "left_message": [element(10, (1295, 228, 1425, 249), text="ARBITRARY_TITLE", source="ocr")],
        "timestamp": [element(11, (1652, 228, 1760, 249), text="ARBITRARY_TITLE", source="ocr")],
    }
    frame.elements = frame.elements[:1] + (
        [item for values in options.values() for item in values] if remaining == "all" else options[remaining]
    )
    with pytest.raises(MessageRegionError, match="no upper OCR row"):
        select_message_regions(frame)


@pytest.mark.parametrize("coordinates", [(1295, 126, 1425, 143), (1295, 176, 1425, 193)])
def test_independent_competing_content_title_rows_are_refused(coordinates):
    frame = measured_frame()
    frame.elements.append(element(12, coordinates, text="ARBITRARY_TITLE", source="ocr"))
    with pytest.raises(MessageRegionError, match="competing rows"):
        select_message_regions(frame)


def test_disconnected_same_row_title_anchors_are_ambiguous_instead_of_concatenated():
    frame = measured_frame()
    frame.elements[1] = element(46, (1295, 145, 1325, 169), text="First", source="ocr")
    frame.elements.append(element(12, (1410, 145, 1440, 169), text="Second", source="ocr"))
    with pytest.raises(MessageRegionError, match="contiguous"):
        select_message_regions(frame)


def test_adjacent_title_fragments_are_merged_but_detached_right_toolbar_text_is_excluded():
    frame = measured_frame()
    frame.elements[1] = element(46, (1295, 145, 1355, 169), text="First", source="ocr")
    frame.elements.append(element(12, (1360, 147, 1425, 170), text="Second", source="ocr"))
    selected = select_message_regions(frame)
    assert selected.header_element_ids == ("obs-0001-e046", "obs-0001-e012")
    assert selected.header_box.right == 1429


@pytest.mark.parametrize("problem", ["empty", "low_confidence", "bad_center", "manual", "outside"])
def test_unusable_title_evidence_has_no_layout_fallback(problem):
    frame = measured_frame()
    title = frame.elements[1]
    if problem == "empty":
        title.text = ""
    elif problem == "low_confidence":
        title.confidence = 0.34
    elif problem == "bad_center":
        title.center = title.center.model_copy(update={"x": 0})
    elif problem == "manual":
        title.source = "manual"
    else:
        frame.elements[1] = element(46, (1270, 145, 1425, 169), text="Title", source="ocr")
    with pytest.raises(MessageRegionError, match="no upper OCR row"):
        select_message_regions(frame)


@pytest.mark.parametrize("border_width,factor", [(190, 1.0), (150, 1.0), (150, 0.25)])
def test_missing_title_text_inside_observed_compact_message_border_is_not_a_header(border_width, factor):
    frame = measured_frame()
    frame.elements = frame.elements[:1] + [
        element(20, (1295, 145, 1425, 169), text="ARBITRARY_TITLE", source="ocr"),
        element(21, (1290, 140, 1290 + border_width, 180)),
    ]
    if factor != 1.0:
        def scale(box):
            return BoundingBox(**{
                edge: round(getattr(box, edge) * factor) for edge in ("left", "top", "right", "bottom")
            })
        frame.window_bounds = scale(frame.window_bounds)
        frame.screen_info = ScreenInfo(
            screenshot_width=round(2560 * factor), screenshot_height=round(1600 * factor),
            control_width=2560, control_height=1600, scale_x=1 / factor, scale_y=1 / factor,
        )
        frame.elements = [
            item.model_copy(update={"bounding_box": scale(item.bounding_box),
                                    "center": scale(item.bounding_box).center})
            for item in frame.elements
        ]
    with pytest.raises(MessageRegionError, match="no upper OCR row"):
        select_message_regions(frame)


def test_observed_full_width_pane_border_does_not_masquerade_as_a_compact_message():
    frame = measured_frame()
    frame.elements.append(element(21, (1284, 130, 2127, 190)))
    assert select_message_regions(frame).header_element_ids == ("obs-0001-e046",)


def test_new_header_crop_preserves_actual_source_pixels_and_never_rewrites_failed_frame(tmp_path):
    frame = measured_frame()
    source = tmp_path / "source.png"
    pixels = Image.new("RGB", (2560, 1600), (30, 40, 50))
    draw = ImageDraw.Draw(pixels)
    draw.rectangle((1291, 141, 1428, 172), fill=(20, 90, 170))
    draw.rectangle((2096, 86, 2123, 111), fill=(190, 30, 20))
    pixels.save(source)
    frame.image_path = str(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    selected = select_message_regions(frame)
    sheet = render_message_regions(frame, selected, tmp_path / "header_regions.png", include_messages=False)
    panel = sheet.panels[0]
    with Image.open(sheet.image_path) as rendered:
        box = panel.panel_box
        actual = rendered.crop((box.left, box.top, box.right, box.bottom)).convert("RGB")
    original = pixels.crop((1291, 141, 1429, 173))
    assert actual.tobytes() == original.tobytes()
    assert panel.candidate_ids == ("obs-0001-e046",)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
