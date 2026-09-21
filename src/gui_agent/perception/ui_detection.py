"""Simple non-text UI candidate detection based on OpenCV contours.

OCR only finds text. Buttons, inputs and icons without a label still produce a
visible rectangle, so Week 2 uses thresholding plus contour analysis to propose
candidate regions. This is deliberately not a semantic detector.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image

from ..schemas import BoundingBox, UIElement
from .preprocessing import to_numpy

DEFAULT_MIN_AREA = 100
DEFAULT_MAX_AREA_RATIO = 0.5
DEFAULT_MIN_ASPECT = 0.08
DEFAULT_MAX_ASPECT = 12.0
# Off by default: the useful value depends on the theme behind the screen.
DEFAULT_MIN_RECTANGULARITY = 0.0
DEFAULT_MAX_CANDIDATES = 200
DEFAULT_EXCLUSION_THRESHOLD = 0.5
CONTOUR_CONFIDENCE = 0.5


def _containment(inner: BoundingBox, outer: BoundingBox) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    left = max(inner.left, outer.left)
    top = max(inner.top, outer.top)
    right = min(inner.right, outer.right)
    bottom = min(inner.bottom, outer.bottom)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    return overlap / float(inner.width * inner.height)


def _overlap_over_smaller(first: BoundingBox, second: BoundingBox) -> float:
    """Intersection area as a fraction of the smaller of the two boxes."""
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    smaller = min(first.width * first.height, second.width * second.height)
    return overlap / float(smaller) if smaller else 0.0


def _deduplicate(boxes: list[BoundingBox], containment_threshold: float = 0.8) -> list[BoundingBox]:
    """Drop boxes that are largely contained in a bigger, already kept box."""
    ordered = sorted(boxes, key=lambda box: box.width * box.height, reverse=True)
    kept: list[BoundingBox] = []
    for candidate in ordered:
        if any(_containment(candidate, existing) >= containment_threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def detect_ui_candidates(
    image: Image.Image | np.ndarray,
    *,
    min_area: int = DEFAULT_MIN_AREA,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
    min_aspect: float = DEFAULT_MIN_ASPECT,
    max_aspect: float = DEFAULT_MAX_ASPECT,
    min_rectangularity: float = DEFAULT_MIN_RECTANGULARITY,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    exclude: Sequence[BoundingBox] = (),
    exclusion_threshold: float = DEFAULT_EXCLUSION_THRESHOLD,
) -> list[UIElement]:
    """Return simple rectangular candidates marked with ``source="contour"``.

    ``exclude`` holds boxes this detector should not duplicate - in practice the
    OCR text regions. Without it the contour pass re-frames every line of text it
    can find, which is both redundant and what makes the annotated image
    unreadable. A candidate is dropped when it overlaps an excluded box by at
    least ``exclusion_threshold`` of whichever of the two is smaller.
    """
    array = to_numpy(image)
    height, width = array.shape[:2]
    if height == 0 or width == 0:
        return []

    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(width * height)
    boxes: list[BoundingBox] = []

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < min_area or area > image_area * max_area_ratio:
            continue
        aspect = box_width / float(box_height) if box_height else 0.0
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if min_rectangularity > 0.0:
            # How much of the bounding box the contour actually fills. A button or
            # an input is nearly solid, a stray cluster of edge pixels is not. Note
            # that the useful threshold is theme dependent, so this is opt-in.
            filled = cv2.contourArea(contour) / float(area) if area else 0.0
            if filled < min_rectangularity:
                continue
        box = BoundingBox(left=x, top=y, right=x + box_width, bottom=y + box_height)
        if (
            exclude
            and max((_overlap_over_smaller(box, region) for region in exclude), default=0.0)
            >= exclusion_threshold
        ):
            continue
        boxes.append(box)

    kept = _deduplicate(boxes)[:max_candidates]
    return [
        UIElement(text="", bounding_box=box, confidence=CONTOUR_CONFIDENCE, source="contour")
        for box in kept
    ]
