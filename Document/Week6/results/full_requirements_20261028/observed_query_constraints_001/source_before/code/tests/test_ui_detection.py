"""Tests for contour based UI candidate detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from gui_agent.perception.ui_detection import _containment, _deduplicate, detect_ui_candidates
from gui_agent.schemas import BoundingBox


def synthetic_image(width: int = 400, height: int = 300) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[40:140, 50:250] = 255
    array[180:220, 300:360] = 255
    return Image.fromarray(array, mode="RGB")


def test_detects_a_simple_rectangle() -> None:
    candidates = detect_ui_candidates(synthetic_image(), min_area=100)
    assert candidates
    assert all(element.source == "contour" for element in candidates)
    centers = [(element.center.x, element.center.y) for element in candidates if element.center]
    assert any(abs(x - 150) <= 6 and abs(y - 90) <= 6 for x, y in centers)


def test_candidates_carry_a_confidence_value() -> None:
    candidates = detect_ui_candidates(synthetic_image(), min_area=100)
    assert all(0.0 <= element.confidence <= 1.0 for element in candidates)


def test_min_area_filters_everything_out() -> None:
    assert detect_ui_candidates(synthetic_image(), min_area=100_000) == []


def test_max_area_ratio_filters_large_regions() -> None:
    assert detect_ui_candidates(synthetic_image(), min_area=100, max_area_ratio=0.001) == []


def test_aspect_ratio_filter_rejects_extreme_shapes() -> None:
    candidates = detect_ui_candidates(
        synthetic_image(), min_area=100, min_aspect=0.95, max_aspect=1.05
    )
    assert candidates == []


def test_empty_image_returns_no_candidates() -> None:
    assert detect_ui_candidates(Image.new("RGB", (50, 50), "black")) == []


def test_deduplicate_keeps_the_larger_box() -> None:
    outer = BoundingBox(left=0, top=0, right=100, bottom=100)
    inner = BoundingBox(left=10, top=10, right=90, bottom=90)
    assert _deduplicate([inner, outer]) == [outer]


def test_deduplicate_keeps_disjoint_boxes() -> None:
    first = BoundingBox(left=0, top=0, right=10, bottom=10)
    second = BoundingBox(left=50, top=50, right=60, bottom=60)
    assert len(_deduplicate([first, second])) == 2


def test_containment_of_disjoint_boxes_is_zero() -> None:
    a = BoundingBox(left=0, top=0, right=10, bottom=10)
    b = BoundingBox(left=50, top=50, right=60, bottom=60)
    assert _containment(a, b) == 0.0


def test_max_candidates_caps_the_result() -> None:
    candidates = detect_ui_candidates(synthetic_image(), min_area=100, max_candidates=1)
    assert len(candidates) <= 1


def test_candidates_inside_excluded_regions_are_dropped() -> None:
    """Contour candidates must not re-frame text OCR has already located."""
    image = synthetic_image()  # existing helper in this module

    unfiltered = detect_ui_candidates(image, min_rectangularity=0.0)
    assert unfiltered, "the synthetic image should produce candidates at all"

    first = unfiltered[0].bounding_box
    filtered = detect_ui_candidates(
        image, min_rectangularity=0.0, exclude=[first], exclusion_threshold=0.5
    )
    assert len(filtered) < len(unfiltered)


def test_rectangularity_filter_can_be_disabled() -> None:
    """The default is off, because the useful value depends on the screen theme."""
    image = synthetic_image()
    assert len(detect_ui_candidates(image, min_rectangularity=0.0)) >= len(
        detect_ui_candidates(image, min_rectangularity=0.9)
    )


def _thin_dark_border() -> Image.Image:
    """A native one-pixel border with the real failed frame's dark-theme contrast."""
    image = Image.new("RGB", (600, 300), (29, 29, 30))
    ImageDraw.Draw(image).rounded_rectangle(
        (60, 70, 480, 190), radius=12, outline=(45, 45, 46), width=1,
    )
    return image


def test_native_thin_rectangle_survives_when_blurred_edges_have_no_candidate() -> None:
    image = _thin_dark_border()
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    blurred_edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 20, 60)
    assert not np.count_nonzero(blurred_edges), "this frame must reproduce the blur failure"
    candidates = detect_ui_candidates(image)
    assert len(candidates) == 1
    box = candidates[0].bounding_box
    assert abs(box.left - 60) <= 2 and abs(box.top - 70) <= 2
    assert abs(box.right - 480) <= 3 and abs(box.bottom - 190) <= 3
    assert candidates[0].source == "contour" and not candidates[0].text


def test_thin_border_pass_does_not_promote_open_l_shape_to_editor_box() -> None:
    image = Image.new("RGB", (600, 300), (29, 29, 30))
    ImageDraw.Draw(image).line([(60, 70), (60, 190), (480, 190)], fill=(45, 45, 46), width=1)
    assert detect_ui_candidates(image) == []


def test_isolated_noise_does_not_become_a_native_rectangle_candidate() -> None:
    image = Image.new("RGB", (600, 300), (29, 29, 30))
    for x in range(20, 580, 20):
        for y in range(20, 280, 20):
            image.putpixel((x, y), (45, 45, 46))
    assert detect_ui_candidates(image) == []


@pytest.mark.parametrize("filters", [
    {"min_area": 100_000}, {"max_area_ratio": 0.10},
    {"min_aspect": 0.95, "max_aspect": 1.05}, {"min_rectangularity": 0.999},
])
def test_native_pass_obeys_the_same_area_aspect_and_rectangularity_filters(filters) -> None:
    assert detect_ui_candidates(_thin_dark_border(), **filters) == []


def test_native_pass_obeys_ocr_exclusion() -> None:
    assert detect_ui_candidates(
        _thin_dark_border(), exclude=[BoundingBox(left=50, top=60, right=490, bottom=200)],
    ) == []


def test_native_and_blurred_duplicate_borders_share_one_candidate() -> None:
    image = _thin_dark_border()
    ImageDraw.Draw(image).rounded_rectangle(
        (60, 70, 480, 190), radius=12, outline="white", width=3,
    )
    assert len(detect_ui_candidates(image)) == 1


def test_native_pass_keeps_foreground_candidate_with_a_one_candidate_budget() -> None:
    image = _thin_dark_border()
    ImageDraw.Draw(image).rectangle((500, 10, 590, 50), fill="white")
    candidates = detect_ui_candidates(
        image, max_candidates=1, priority_region=BoundingBox(left=50, top=60, right=490, bottom=200),
    )
    assert len(candidates) == 1
    assert candidates[0].bounding_box.width > 400


def test_two_distinct_thin_rectangles_remain_two_candidates() -> None:
    image = Image.new("RGB", (600, 400), (29, 29, 30))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((60, 70, 480, 190), radius=12, outline=(45, 45, 46), width=1)
    draw.rounded_rectangle((60, 250, 480, 370), radius=12, outline=(45, 45, 46), width=1)
    candidates = detect_ui_candidates(image)
    assert len(candidates) == 2
    assert abs(candidates[0].bounding_box.center.y - candidates[1].bounding_box.center.y) > 170
