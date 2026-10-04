"""Locate a text target among perceived UI elements.

When several elements match, the first one is *not* silently chosen: every
candidate is returned ordered by confidence so a caller can show the list, pick
an index, or record why a specific element was used.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from ..coordinates import screenshot_to_control
from ..schemas import BoundingBox, Point, ScreenInfo, UIElement

MatchMode = Literal["exact", "contains", "case_insensitive"]
DEFAULT_MATCH_MODE: MatchMode = "contains"
DEFAULT_MIN_CONFIDENCE = 0.5


@dataclass
class GroundingMatch:
    """One element that matched the query."""

    element: UIElement
    matched_text: str
    confidence: float
    bounding_box: BoundingBox
    screenshot_center: Point
    control_center: Point | None = None


@dataclass
class GroundingResult:
    """Every candidate for a query, ordered by confidence (highest first)."""

    query: str
    match_mode: str
    candidates: list[GroundingMatch] = field(default_factory=list)
    selected_index: int | None = None
    message: str = ""

    @property
    def found(self) -> bool:
        return bool(self.candidates)

    @property
    def selected(self) -> GroundingMatch | None:
        """The chosen candidate, or the best one when no index was given."""
        if not self.candidates:
            return None
        if self.selected_index is None:
            return self.candidates[0]
        if 0 <= self.selected_index < len(self.candidates):
            return self.candidates[self.selected_index]
        return None


def _without_whitespace(value: str, *, fold_case: bool) -> str:
    """The text with every whitespace character removed.

    Used as a last resort in the mode that already ignores case, because OCR does not
    reproduce spacing reliably and the characters it drops are not meaningful in the
    languages this project meets. The measurement that put this here: T02 failed with

        no element matches '在 Google 中搜索，或输入网址' in obs-0002

    while the frame held '在Google 中搜索，或输入网址' - the same characters with the
    space after 在 missing, because the OCR engine did not emit it. Every space-sensitive
    comparison fails on that pair, and the model had read the text correctly.

    Restricted to the case-insensitive mode on purpose. `exact` means the whole string and
    `contains` is case-sensitive, so a space-blind version of either would match
    'OK' against 'OK Cancel' and 'read' against 'README.md' - both of which the existing
    tests assert must not match, and both of which the first version of this function did.
    """
    collapsed = "".join(value.split())
    return collapsed.casefold() if fold_case else collapsed


def text_matches(query: str, text: str, match_mode: MatchMode = DEFAULT_MATCH_MODE) -> bool:
    """Compare a query against element text using the requested mode.

    Except in the case-insensitive mode, which falls back to ignoring whitespace when the
    spacing is the only difference - see :func:`_without_whitespace` for the frame that
    made that necessary.
    """
    if not query or not text:
        return False
    if match_mode == "exact":
        return text.strip() == query.strip()
    if match_mode == "case_insensitive":
        if query.strip().casefold() in text.strip().casefold():
            return True
        return _without_whitespace(query, fold_case=True) in _without_whitespace(
            text, fold_case=True
        )
    return query.strip() in text


def find_text(
    elements: Iterable[UIElement],
    query: str,
    *,
    match_mode: MatchMode = DEFAULT_MATCH_MODE,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    candidate_index: int | None = None,
    screen_info: ScreenInfo | None = None,
) -> GroundingResult:
    """Find elements whose text matches ``query``.

    Returns a :class:`GroundingResult` describing every candidate. A missing
    target is reported through ``message`` instead of raising.
    """
    if not query or not query.strip():
        return GroundingResult(query=query, match_mode=match_mode, message="empty query")

    candidates: list[GroundingMatch] = []
    for element in elements:
        if element.confidence < min_confidence:
            continue
        if not text_matches(query, element.text, match_mode):
            continue
        center = element.center or element.bounding_box.center
        control_center = (
            screenshot_to_control(center, screen_info) if screen_info is not None else None
        )
        candidates.append(
            GroundingMatch(
                element=element,
                matched_text=element.text,
                confidence=element.confidence,
                bounding_box=element.bounding_box,
                screenshot_center=center,
                control_center=control_center,
            )
        )

    candidates.sort(
        key=lambda match: (-match.confidence, match.bounding_box.top, match.bounding_box.left)
    )

    if not candidates:
        return GroundingResult(
            query=query,
            match_mode=match_mode,
            message=f"no element matched {query!r} (mode={match_mode}, min_confidence={min_confidence})",
        )

    result = GroundingResult(query=query, match_mode=match_mode, candidates=candidates)
    if candidate_index is not None:
        if 0 <= candidate_index < len(candidates):
            result.selected_index = candidate_index
        else:
            result.message = (
                f"candidate_index {candidate_index} is out of range (0..{len(candidates) - 1})"
            )
    return result


def format_candidates(result: GroundingResult) -> list[str]:
    """Human readable candidate list for CLI output and logs."""
    lines: list[str] = []
    for index, match in enumerate(result.candidates):
        box = match.bounding_box
        marker = "*" if result.selected_index == index else " "
        lines.append(
            f"{marker}[{index}] {match.matched_text!r} conf={match.confidence:.2f} "
            f"box=({box.left},{box.top},{box.right},{box.bottom}) "
            f"screenshot_center=({match.screenshot_center.x},{match.screenshot_center.y})"
            + (
                f" control_center=({match.control_center.x},{match.control_center.y})"
                if match.control_center
                else ""
            )
        )
    return lines
