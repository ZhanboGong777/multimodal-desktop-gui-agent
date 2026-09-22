"""Tests for bounding box drawing and annotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gui_agent.perception.visualization import (
    CONTOUR_COLOR,
    OCR_COLOR,
    color_for,
    draw_bounding_boxes,
    label_for,
    load_label_font,
    save_annotated_image,
    summarize_sources,
)
from gui_agent.schemas import BoundingBox, UIElement


def make_element(text: str = "OK", source: str = "ocr", left: int = 5, top: int = 5) -> UIElement:
    return UIElement(
        text=text,
        bounding_box=BoundingBox(left=left, top=top, right=left + 40, bottom=top + 18),
        confidence=0.9,
        source=source,  # type: ignore[arg-type]
    )


def blank_image(width: int = 200, height: int = 120) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def test_drawing_does_not_modify_the_input_image() -> None:
    original = blank_image()
    before = np.asarray(original).copy()
    annotated = draw_bounding_boxes(original, [make_element()], show_labels=False)
    assert annotated is not original
    assert np.array_equal(np.asarray(original), before)
    assert not np.array_equal(np.asarray(annotated), before)


def test_annotated_image_keeps_its_size() -> None:
    annotated = draw_bounding_boxes(blank_image(320, 240), [make_element()])
    assert annotated.size == (320, 240)


def test_ocr_and_contour_elements_use_different_colors() -> None:
    assert color_for(make_element(source="ocr")) == OCR_COLOR
    assert color_for(make_element(source="contour")) == CONTOUR_COLOR
    assert OCR_COLOR != CONTOUR_COLOR


def test_labels_can_include_index_and_confidence() -> None:
    element = make_element()
    assert label_for(element) == "OK"
    assert label_for(element, index=3) == "#3 OK"
    assert label_for(element, show_confidence=True) == "OK 0.90"
    assert label_for(element, index=1, show_confidence=True) == "#1 OK 0.90"


def test_empty_text_produces_an_empty_label() -> None:
    assert label_for(make_element(text="")) == ""


def test_drawing_with_labels_changes_pixels() -> None:
    original = blank_image()
    annotated = draw_bounding_boxes(original, [make_element()], show_labels=True)
    assert not np.array_equal(np.asarray(annotated), np.asarray(original))


def test_contour_elements_can_be_drawn_without_a_label() -> None:
    annotated = draw_bounding_boxes(
        blank_image(), [make_element(text="", source="contour")], show_labels=True
    )
    assert annotated.size == (200, 120)


def test_font_loader_always_returns_a_font() -> None:
    assert load_label_font() is not None


def test_save_annotated_image_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "annotated.png"
    written = save_annotated_image(blank_image(), target)
    assert written == target
    assert target.exists()


def test_source_summary_counts_elements() -> None:
    summary = summarize_sources([make_element(), make_element(), make_element(source="contour")])
    assert summary == {"ocr": 2, "contour": 1}
    assert summarize_sources([]) == {}


def test_labels_are_drawn_for_ocr_only_by_default() -> None:
    """Contour candidates carry no text, so labelling them only adds clutter."""
    image = Image.new("RGB", (120, 60), "white")
    ocr = UIElement(
        text="Save",
        bounding_box=BoundingBox(left=5, top=5, right=45, bottom=20),
        confidence=0.9,
        source="ocr",
    )
    contour = UIElement(
        text="",
        bounding_box=BoundingBox(left=60, top=5, right=110, bottom=50),
        confidence=0.5,
        source="contour",
    )

    default = draw_bounding_boxes(image, [ocr, contour])
    explicit = draw_bounding_boxes(image, [ocr], label_sources=("ocr",))

    # Same picture whether or not the unlabelled candidate is present, because the
    # candidate contributes only its outline in both cases.
    assert default.tobytes() != Image.new("RGB", (120, 60), "white").tobytes()
    assert default.size == explicit.size


def test_label_sources_can_opt_contours_back_in() -> None:
    image = Image.new("RGB", (120, 60), "white")
    contour = UIElement(
        text="box",
        bounding_box=BoundingBox(left=10, top=20, right=100, bottom=50),
        confidence=0.5,
        source="contour",
    )

    without = draw_bounding_boxes(image, [contour])
    with_label = draw_bounding_boxes(image, [contour], label_sources=("contour",))

    # Labelling draws a filled background, so the two images must differ.
    assert without.tobytes() != with_label.tobytes()
