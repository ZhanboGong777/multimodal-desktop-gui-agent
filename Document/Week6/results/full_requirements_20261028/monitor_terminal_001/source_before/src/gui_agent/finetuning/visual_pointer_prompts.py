"""Versioned input-only visual instructions; raster geometry stays independent."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .candidate_grounding import center
from .evaluation import canonical_json
from .pointer_protocol import TEXT_CHARACTERS, pointer_user_prompt, select_candidates

DEDUP_PROMPT_VERSION = "week5.visual-pointer-prompt.dedup.v2"
COMPACT_PROMPT_VERSION = "week5.visual-pointer-prompt.compact.v3"
GALLERY_PROMPT_VERSION = "week5.visual-pointer-prompt.gallery.v4"
GALLERY_SPATIAL_PROMPT_VERSION = "week5.visual-pointer-prompt.gallery-native-spatial.v1"
GALLERY_PROMPT_VERSIONS = (GALLERY_PROMPT_VERSION, GALLERY_SPATIAL_PROMPT_VERSION)
VISUAL_PROMPT_VERSIONS = (DEDUP_PROMPT_VERSION, COMPACT_PROMPT_VERSION, *GALLERY_PROMPT_VERSIONS)
GALLERY_SPATIAL_HEADER = (
    "Current subtask only. Return only complete ordered JSON action array; exact requested text. "
    "action_type fields: "
    'click double_click right_click move: point="native_head"; '
    'scroll: point="native_head",scroll_amount(nonzero integer); '
    'drag: start_point="native_head",end_point="native_head"; '
    "type_text:text; key_press:key; hotkey:keys; wait:duration. "
    "Only listed fields. Every mouse endpoint uses the literal native_head; "
    "a separate head predicts its native screenshot point. "
    "UI above; shuffled candidate crop proposals below. Header IDs match table. "
    "Act on native UI, never gallery. Inspect UI to identify the intended control. "
    "Table ID X Y OCR<=24chars\n"
)
COMPACT_HEADER = (
    "Current subtask only. Return only complete ordered JSON action array; exact requested text. "
    "action_type fields: "
    "click double_click right_click move: candidate dx dy; "
    "scroll: candidate dx dy scroll_amount(nonzero integer); "
    "drag: start_candidate start_dx start_dy end_candidate end_dx end_dy; "
    "type_text:text; key_press:key; hotkey:keys; wait:duration. "
    "IDs/offsets integers; only listed fields, no x/y. "
    "Native pixel mouse/both drag endpoints=center+(dx,dy); intended offsets in bounds. "
    "Black badges/white IDs and cyan 1px leaders are annotations, not UI controls. "
    "Badge ID=table ID; leader tip=native candidate center; never click badges. "
    "Candidate proposals; inspect screenshot; no region fits: []. "
    "Table ID X Y OCR<=24chars\n"
)
GALLERY_INSTRUCTION = (
    "UI above; shuffled candidate crop proposals below. Header IDs match table. "
    "Act on native UI, never gallery. No fit: []. "
)
GALLERY_HEADER = COMPACT_HEADER.replace(
    "Black badges/white IDs and cyan 1px leaders are annotations, not UI controls. "
    "Badge ID=table ID; leader tip=native candidate center; never click badges. "
    "Candidate proposals; inspect screenshot; no region fits: []. ",
    GALLERY_INSTRUCTION,
)
LEGACY_VISUAL_PREFIX = (
    "Black badges with white numbers and cyan one-pixel leader lines are annotations, "
    "not UI controls. Each badge number is exactly the ID in the candidate table. "
    "A leader ends at that candidate's original native center; do not click the badge. "
    "Use the original native center plus dx/dy, including both drag endpoints.\n"
)


def checked_prompt_version(value: Any) -> str | None:
    """None selects the byte-identical legacy API; other versions are explicit."""
    if value is not None and (not isinstance(value, str) or value not in VISUAL_PROMPT_VERSIONS):
        raise ValueError("unknown visual_prompt_version")
    return value


def declared_prompt_version(value: Mapping[str, Any]) -> str | None:
    """A missing field means legacy; an explicit null/empty field is invalid."""
    if "visual_prompt_version" not in value:
        return None
    version = value["visual_prompt_version"]
    if version is None:
        raise ValueError("explicit visual_prompt_version cannot be null")
    return checked_prompt_version(version)


def build_visual_user_prompt(
    instruction: str, actor_step: str, snapshot: Mapping[str, Any], *,
    visual_prompt_version: str | None = None,
) -> str:
    """Preserve all context/table evidence; neither targets nor reviews are inputs."""
    version = checked_prompt_version(visual_prompt_version)
    if version is None:
        return LEGACY_VISUAL_PREFIX + pointer_user_prompt(instruction, actor_step, snapshot)
    if any(not isinstance(value, str) or not value.strip() for value in (instruction, actor_step)):
        raise ValueError("visual acting requires explicit goal and current subtask")
    table = "\n".join(
        f"{i} {center(candidate['bbox'])[0]} {center(candidate['bbox'])[1]} "
        f"{canonical_json(candidate['text'][:TEXT_CHARACTERS])}"
        for i, candidate in enumerate(select_candidates(snapshot, instruction=instruction, actor_step=actor_step), 1)
    )
    if version in (COMPACT_PROMPT_VERSION, *GALLERY_PROMPT_VERSIONS):
        header = (GALLERY_SPATIAL_HEADER if version == GALLERY_SPATIAL_PROMPT_VERSION else
                  GALLERY_HEADER if version == GALLERY_PROMPT_VERSION else COMPACT_HEADER)
        return (header + f"Native size: {snapshot['image_width']}x{snapshot['image_height']}\n{table}\n"
                f"Overall goal: {instruction}\nCurrent subtask: {actor_step}")
    return (
        "Using the screenshot, execute only Current subtask. Return only its complete ordered JSON action array; "
        "copy requested text exactly; no later subtasks. "
        "Each object: action_type plus fields: "
        "click/double_click/right_click/move: candidate,dx,dy; "
        "scroll: candidate,dx,dy,scroll_amount (nonzero integer); "
        "drag: start_candidate,start_dx,start_dy,end_candidate,end_dx,end_dy; "
        "type_text: text; key_press: key; hotkey: keys; wait: duration. "
        "No other fields/x/y. IDs/offsets are integers. "
        "Mouse point (both drag endpoints)=native candidate center+(dx,dy); offset to the intended pixel within native bounds. "
        "Black badges with white numbers and cyan 1-pixel leaders are annotations, not UI controls. "
        "Badge number=Table ID; leader ends at original native center; never click badges. "
        "Regions are proposals, not control evidence; inspect the screenshot. "
        "If no region fits, return []. Table: ID X Y OCR-text (at most24characters).\n"
        f"Native size: {snapshot['image_width']}x{snapshot['image_height']}\n{table}\n"
        f"Overall goal: {instruction}\nCurrent subtask: {actor_step}"
    )
