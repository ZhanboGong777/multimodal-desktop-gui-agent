"""Simple non-text UI candidate detection based on OpenCV contours.

OCR only finds text. Buttons, inputs and icons without a label still produce a
visible rectangle, so Week 2 uses thresholding plus contour analysis to propose
candidate regions. This is deliberately not a semantic detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

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
_THIN_BORDER_RECTANGULARITY = 0.90


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
    """Drop duplicate borders while retaining controls inside a larger container.

    Offline replay of T04_20261005_192417 found the send control inside the input
    container: containment alone discarded it. Similar-size overlapping boxes are
    duplicate borders; a much smaller inner box can be a separate control.
    """
    # On dense screens containment used most of the contour pass's Python time.
    # Compute immutable geometry once per call, and reject the size condition
    # before doing intersection arithmetic. The sort remains stable and the two
    # original thresholds are unchanged; smaller controls inside containers stay.
    ordered = sorted(
        ((box, box.width * box.height) for box in boxes),
        key=lambda item: item[1], reverse=True,
    )
    kept: list[BoundingBox] = []
    geometry: list[tuple[int, int, int, int, int]] = []
    for candidate, area in ordered:
        left, top, right, bottom = (
            candidate.left, candidate.top, candidate.right, candidate.bottom,
        )
        duplicate = False
        for old_left, old_top, old_right, old_bottom, old_area in geometry:
            if area < old_area * 0.5:
                continue
            overlap_width = min(right, old_right) - max(left, old_left)
            overlap_height = min(bottom, old_bottom) - max(top, old_top)
            overlap = (overlap_width * overlap_height
                       if overlap_width > 0 and overlap_height > 0 else 0)
            if overlap / float(area) >= containment_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        geometry.append((left, top, right, bottom, area))
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
    priority_region: BoundingBox | None = None,
) -> list[UIElement]:
    """Return simple rectangular candidates marked with ``source="contour"``.

    ``exclude`` holds boxes this detector should not duplicate - in practice the
    OCR text regions. Without it the contour pass re-frames every line of text it
    can find, which is both redundant and what makes the annotated image
    unreadable. A candidate is dropped when it overlaps an excluded box by at
    least ``exclusion_threshold`` of the candidate's own area. A large input
    containing a small OCR label must remain targetable. ``priority_region`` is
    the freshly captured foreground window, whose controls precede desktop clutter
    when the bounded candidate budget is applied.
    """
    array = to_numpy(image)
    height, width = array.shape[:2]
    if height == 0 or width == 0:
        return []

    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Saved T04 pixels have a dark input border that 40/120 missed; offline replay
    # with 20/60 recovered it without inventing a box or changing action safety.
    edges = cv2.Canny(blurred, 20, 60)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # A one-pixel dark-theme border can disappear in the blur even though its
    # native pixels have adequate contrast. T04's 2026-10-07 frame had no editor
    # candidate for this reason. Preserve the normal noise-resistant pass, and
    # supplement it with native-pixel edges only when they enclose a nearly full
    # rectangle. Open strokes, text fragments and noisy clusters must not gain
    # broad action boxes merely because the blur was removed.
    native_edges = cv2.dilate(
        cv2.Canny(gray, 20, 60), np.ones((2, 2), np.uint8), iterations=1,
    )
    thin_contours, _ = cv2.findContours(
        native_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
    )
    image_area = float(width * height)
    boxes: list[BoundingBox] = []
    seen: set[tuple[int, int, int, int]] = set()

    for contour, thin_border in chain(
        ((contour, False) for contour in contours),
        ((contour, True) for contour in thin_contours),
    ):
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < min_area or area > image_area * max_area_ratio:
            continue
        aspect = box_width / float(box_height) if box_height else 0.0
        if aspect < min_aspect or aspect > max_aspect:
            continue
        required_rectangularity = max(
            min_rectangularity, _THIN_BORDER_RECTANGULARITY if thin_border else 0.0,
        )
        if required_rectangularity > 0.0:
            # How much of the bounding box the contour actually fills. A button or
            # an input is nearly solid, a stray cluster of edge pixels is not. Note
            # that the useful threshold is theme dependent, so this is opt-in.
            filled = cv2.contourArea(contour) / float(area) if area else 0.0
            if filled < required_rectangularity:
                continue
        # Both edge passes often return precisely the same border. Only discard
        # it after that pass's area/aspect/rectangularity checks have succeeded.
        # This is frame-local geometry, never a cached target from another frame.
        coordinates = (x, y, x + box_width, y + box_height)
        if coordinates in seen:
            continue
        box = BoundingBox(left=x, top=y, right=x + box_width, bottom=y + box_height)
        if (
            exclude
            and max((_containment(box, region) for region in exclude), default=0.0)
            >= exclusion_threshold
        ):
            continue
        boxes.append(box)
        seen.add(coordinates)

    kept = _deduplicate(boxes)
    if priority_region is not None:
        kept.sort(key=lambda box: _containment(box, priority_region) == 1.0, reverse=True)
    kept = kept[:max_candidates]
    return [
        UIElement(text="", bounding_box=box, confidence=CONTOUR_CONFIDENCE, source="contour")
        for box in kept
    ]
