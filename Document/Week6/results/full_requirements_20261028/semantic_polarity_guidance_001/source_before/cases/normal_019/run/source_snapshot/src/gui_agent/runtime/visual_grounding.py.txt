"""Re-ground a frame-local candidate only when current pixels establish its identity.

T04 exposed anonymous contours to the model, but the runner correctly takes a
new screenshot before acting, invalidating every planned ID. A saved coordinate,
candidate index or nearest box cannot repair that. This module compares actual
target pixels and surrounding context, then returns a real current-frame element
for the unchanged adapter and executor to validate.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..schemas import BoundingBox
from .schemas import ElementRef, ObservationSnapshot


class GroundingError(RuntimeError):
    """Current evidence cannot safely identify the planned candidate."""


# Conservative acceptance bounds, fixed by prepared-image regressions rather
# than presented as measurements of a live desktop. A changed button must go
# back to the visual model; a permissive brightness-invariant score alone would
# treat its disabled and enabled colours as the same executable target.
_PADDING = 16
_BOX_TOLERANCE = 2
_MIN_CORRELATION = 0.98
_MIN_MARGIN = 0.04
_MAX_MEAN_ERROR = 3.0
_MAX_CHANGED_FRACTION = 0.02
_MIN_TEXTURE_STD = 8.0
_MIN_EDGE_PIXELS = 16


def _inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    return (
        outer.left <= inner.left < inner.right <= outer.right
        and outer.top <= inner.top < inner.bottom <= outer.bottom
    )


def _load_image(snapshot: ObservationSnapshot) -> np.ndarray:
    if not snapshot.image_path:
        raise GroundingError(f"{snapshot.observation_id}: missing image evidence")
    try:
        with Image.open(Path(snapshot.image_path)) as image:
            if image.size != (
                snapshot.screen_info.screenshot_width,
                snapshot.screen_info.screenshot_height,
            ):
                raise GroundingError(f"{snapshot.observation_id}: image dimensions disagree")
            return np.asarray(image.convert("RGB")).copy()
    except (OSError, ValueError) as exc:
        raise GroundingError(f"{snapshot.observation_id}: unreadable image evidence: {exc}") from exc


def _validate_snapshot(snapshot: ObservationSnapshot) -> None:
    if snapshot.errors:
        raise GroundingError(f"{snapshot.observation_id}: degraded observation")
    if not (
        snapshot.foreground_stable
        and snapshot.window_id
        and snapshot.window_title
        and snapshot.window_class
        and snapshot.window_bounds is not None
    ):
        raise GroundingError(f"{snapshot.observation_id}: unknown or unstable foreground")
    ids = [item.element_id for item in snapshot.elements]
    if len(set(ids)) != len(ids) or any(
        not element_id.startswith(f"{snapshot.observation_id}-e") for element_id in ids
    ):
        # Otherwise a verified second element could resolve to a different first
        # element sharing its ID when the unchanged adapter performs its lookup.
        raise GroundingError(f"{snapshot.observation_id}: invalid candidate ids")
    screen = snapshot.screen_info
    for scale, expected in (
        (screen.scale_x, screen.control_width / screen.screenshot_width),
        (screen.scale_y, screen.control_height / screen.screenshot_height),
    ):
        if scale is None or not math.isfinite(scale) or not math.isclose(scale, expected):
            raise GroundingError(f"{snapshot.observation_id}: inconsistent screen scale")
    bounds = snapshot.window_bounds
    if (
        bounds.right <= 0
        or bounds.bottom <= 0
        or bounds.left >= screen.screenshot_width
        or bounds.top >= screen.screenshot_height
    ):
        raise GroundingError(f"{snapshot.observation_id}: foreground outside captured monitor")


def validate_frames(source: ObservationSnapshot, current: ObservationSnapshot) -> None:
    """Require known, readable frames of one stable foreground window.

    Window translation is allowed, resizing and screen geometry changes are not.
    These checks establish a frame pair, never a candidate identity by position.
    """
    _validate_snapshot(source)
    _validate_snapshot(current)
    if source.screen_info.model_dump() != current.screen_info.model_dump():
        raise GroundingError("screen geometry changed between observations")
    if (source.window_id, source.window_title, source.window_class) != (
        current.window_id, current.window_title, current.window_class
    ):
        raise GroundingError("foreground window changed between observations")
    assert source.window_bounds is not None and current.window_bounds is not None
    if (source.window_bounds.width, source.window_bounds.height) != (
        current.window_bounds.width, current.window_bounds.height
    ):
        raise GroundingError("foreground window resized between observations")
    if current.captured_at < source.captured_at:
        raise GroundingError("current observation predates the source frame")
    _load_image(source)
    _load_image(current)


def _candidate_is_valid(element: ElementRef, snapshot: ObservationSnapshot) -> bool:
    bounds = snapshot.window_bounds
    if bounds is None or element.confidence < 0.35:
        return False
    box = element.bounding_box
    return bool(
        element.source in {"ocr", "contour"}
        and _inside(box, bounds)
        and 0 <= box.left < box.right <= snapshot.screen_info.screenshot_width
        and 0 <= box.top < box.bottom <= snapshot.screen_info.screenshot_height
        and box.width * box.height <= bounds.width * bounds.height * 0.5
        and element.center == box.center
    )


def _score_candidates(
    element: ElementRef,
    crop: BoundingBox,
    template: np.ndarray,
    image: np.ndarray,
    snapshot: ObservationSnapshot,
) -> list[tuple[float, bool, ElementRef]]:
    box = element.bounding_box
    assert snapshot.window_bounds is not None
    matches: list[tuple[float, bool, ElementRef]] = []
    for candidate in snapshot.elements:
        candidate_box = candidate.bounding_box
        if (
            candidate.source != element.source
            or candidate.text.strip() != element.text.strip()
            or abs(candidate_box.width - box.width) > _BOX_TOLERANCE
            or abs(candidate_box.height - box.height) > _BOX_TOLERANCE
            or not _candidate_is_valid(candidate, snapshot)
        ):
            continue
        best_score = -1.0
        best_safe = False
        for offset_y in range(-_BOX_TOLERANCE, _BOX_TOLERANCE + 1):
            for offset_x in range(-_BOX_TOLERANCE, _BOX_TOLERANCE + 1):
                left = candidate_box.left - (box.left - crop.left) + offset_x
                top = candidate_box.top - (box.top - crop.top) + offset_y
                right, bottom = left + crop.width, top + crop.height
                if not (
                    0 <= left < right <= image.shape[1]
                    and 0 <= top < bottom <= image.shape[0]
                    and snapshot.window_bounds.left <= left
                    and right <= snapshot.window_bounds.right
                    and snapshot.window_bounds.top <= top
                    and bottom <= snapshot.window_bounds.bottom
                ):
                    continue
                patch = image[top:bottom, left:right]
                score = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                if not math.isfinite(score):
                    continue
                difference = np.abs(patch.astype(np.float32) - template.astype(np.float32))
                target_difference = difference[
                    box.top - crop.top:box.bottom - crop.top,
                    box.left - crop.left:box.right - crop.left,
                ]
                safe = bool(
                    float(difference.mean()) <= _MAX_MEAN_ERROR
                    and float(target_difference.mean()) <= _MAX_MEAN_ERROR
                    and float(np.mean(np.max(target_difference, axis=2) > 20))
                    <= _MAX_CHANGED_FRACTION
                )
                if score > best_score or (score == best_score and safe):
                    best_score, best_safe = score, safe
        if best_score >= 0:
            matches.append((best_score, best_safe, candidate))
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches


def match_candidate(
    source: ObservationSnapshot, current: ObservationSnapshot, element_id: str
) -> ElementRef:
    """Return a unique compatible current candidate, or refuse without coordinates.

    The small offset search accommodates a contour boundary shifting by up to two
    pixels. It searches only actual current candidates, and both target pixels and
    surrounding context must agree; no old location is accepted as a fallback.
    """
    validate_frames(source, current)
    element = source.element(element_id)
    if element is None:
        raise GroundingError(f"candidate {element_id!r} is absent from its source frame")
    if not _candidate_is_valid(element, source):
        raise GroundingError("source candidate is outside foreground or has invalid geometry")
    old_image = _load_image(source)
    new_image = _load_image(current)
    box = element.bounding_box
    assert source.window_bounds is not None
    crop = BoundingBox(
        left=max(0, source.window_bounds.left, box.left - _PADDING),
        top=max(0, source.window_bounds.top, box.top - _PADDING),
        right=min(old_image.shape[1], source.window_bounds.right, box.right + _PADDING),
        bottom=min(old_image.shape[0], source.window_bounds.bottom, box.bottom + _PADDING),
    )
    template = old_image[crop.top:crop.bottom, crop.left:crop.right]
    target = old_image[box.top:box.bottom, box.left:box.right]
    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < _MIN_TEXTURE_STD or np.count_nonzero(
        cv2.Canny(gray, 40, 120)
    ) < _MIN_EDGE_PIXELS:
        # Constant patches can yield perfect normalized correlation everywhere.
        # A border and context alone must not manufacture identity for blank data.
        raise GroundingError("source candidate has insufficient visual texture")
    source_matches = _score_candidates(element, crop, template, old_image, source)
    if len(source_matches) > 1 and source_matches[0][0] - source_matches[1][0] < _MIN_MARGIN:
        # If one of two identical old controls disappears, the surviving clone
        # must not inherit the selected one's identity merely because it is alone.
        raise GroundingError("source candidate is visually ambiguous")
    matches = _score_candidates(element, crop, template, new_image, current)
    if not matches or matches[0][0] < _MIN_CORRELATION or not matches[0][1]:
        raise GroundingError("source candidate disappeared or its appearance changed")
    if len(matches) > 1 and matches[0][0] - matches[1][0] < _MIN_MARGIN:
        raise GroundingError("multiple current candidates are visually ambiguous")
    return matches[0][2]
