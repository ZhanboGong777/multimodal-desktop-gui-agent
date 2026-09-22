"""Tests for text target grounding."""

from __future__ import annotations

import pytest

from gui_agent.perception.grounding import find_text, format_candidates, text_matches
from gui_agent.schemas import BoundingBox, Point, ScreenInfo, UIElement


def make_element(
    text: str,
    confidence: float = 0.9,
    left: int = 0,
    top: int = 0,
    width: int = 40,
    height: int = 20,
) -> UIElement:
    return UIElement(
        text=text,
        bounding_box=BoundingBox(left=left, top=top, right=left + width, bottom=top + height),
        confidence=confidence,
    )


def test_exact_matching_requires_the_whole_string() -> None:
    assert text_matches("OK", "OK", "exact")
    assert not text_matches("OK", "OK Cancel", "exact")


def test_contains_matching_is_case_sensitive() -> None:
    assert text_matches("READ", "README.md", "contains")
    assert not text_matches("read", "README.md", "contains")


def test_case_insensitive_matching_ignores_case() -> None:
    assert text_matches("read", "README.md", "case_insensitive")
    assert text_matches("READ", "README.md", "case_insensitive")


def test_blank_inputs_never_match() -> None:
    assert not text_matches("", "README", "contains")
    assert not text_matches("read", "", "contains")


def test_multiple_candidates_are_ordered_by_confidence() -> None:
    elements = [make_element("File", 0.6), make_element("File", 0.95), make_element("File", 0.8)]
    result = find_text(elements, "File", match_mode="exact", min_confidence=0.5)
    assert len(result.candidates) == 3
    assert [round(match.confidence, 2) for match in result.candidates] == [0.95, 0.8, 0.6]
    assert result.selected_index is None
    assert result.selected is not None
    assert result.selected.confidence == pytest.approx(0.95)


def test_candidate_index_selects_a_specific_match() -> None:
    elements = [make_element("File", 0.6), make_element("File", 0.95)]
    result = find_text(elements, "File", match_mode="exact", candidate_index=1)
    assert result.selected is not None
    assert result.selected.confidence == pytest.approx(0.6)


def test_out_of_range_candidate_index_is_reported_without_crashing() -> None:
    result = find_text([make_element("File")], "File", match_mode="exact", candidate_index=7)
    assert result.found
    assert result.selected is not None
    assert "out of range" in result.message


def test_low_confidence_elements_are_filtered_out() -> None:
    result = find_text([make_element("File", 0.2)], "File", match_mode="exact", min_confidence=0.5)
    assert not result.found
    assert "no element matched" in result.message


def test_missing_target_is_reported_rather_than_raised() -> None:
    result = find_text([make_element("Other")], "Nope")
    assert not result.found
    assert result.selected is None
    assert result.candidates == []


def test_empty_query_is_rejected() -> None:
    result = find_text([make_element("anything")], "   ")
    assert not result.found
    assert result.message


def test_control_center_is_filled_when_screen_info_is_supplied() -> None:
    screen = ScreenInfo(
        screenshot_width=200, screenshot_height=100, control_width=100, control_height=100
    )
    result = find_text(
        [make_element("OK", left=100, top=40)], "OK", match_mode="exact", screen_info=screen
    )
    match = result.selected
    assert match is not None
    assert match.screenshot_center == Point(x=120, y=50)
    assert match.control_center == Point(x=60, y=50)


def test_control_center_is_absent_without_screen_info() -> None:
    result = find_text([make_element("OK")], "OK", match_mode="exact")
    assert result.selected is not None
    assert result.selected.control_center is None


def test_format_candidates_marks_the_selected_entry() -> None:
    elements = [make_element("A", 0.9), make_element("A", 0.7)]
    result = find_text(elements, "A", match_mode="exact", candidate_index=1)
    lines = format_candidates(result)
    assert lines[1].startswith("*")
    assert lines[0].startswith(" ")
    assert len(lines) == 2
