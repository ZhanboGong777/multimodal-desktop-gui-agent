"""Tests for contour based UI candidate detection."""

from __future__ import annotations

import numpy as np
from PIL import Image

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
