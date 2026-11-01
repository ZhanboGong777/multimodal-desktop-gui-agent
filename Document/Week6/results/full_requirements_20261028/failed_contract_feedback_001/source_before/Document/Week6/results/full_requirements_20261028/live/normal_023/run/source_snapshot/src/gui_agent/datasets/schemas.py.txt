"""Unified schemas for the public GUI datasets.

The three source datasets describe the same idea - a natural-language
instruction plus a sequence of UI actions - with completely different field
names, nesting and action vocabularies. Everything downstream works on the two
structures below, so an adapter only has to translate, never reshape.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from ..schemas import BoundingBox, Point, SchemaModel


class GUIActionStep(SchemaModel):
    """One action inside a task trajectory.

    ``action_type`` is a free string rather than the control layer's literal
    union: the source datasets use vocabularies this project does not implement
    (``SELECT``, ``TYPE``, ``CLICK`` with different meanings). Hovering over an
    unknown verb must not fail the whole sample, so the normalised verb lives
    here and the untouched original is kept in ``raw_action``.
    """

    step_index: int = Field(ge=0)
    action_type: str
    target_text: str | None = None
    target_bbox: BoundingBox | None = None
    coordinates: Point | None = None
    input_text: str | None = None
    raw_action: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_type")
    @classmethod
    def _require_and_fold(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("action_type must not be empty")
        return cleaned.casefold()


class GUITaskSample(SchemaModel):
    """One task from any of the source datasets, in the project's own shape.

    Image bytes are never embedded: only a path is kept, so an exported JSONL
    stays small and the screenshots remain separate artefacts.
    """

    sample_id: str
    dataset_name: str
    instruction: str
    image_path: str | None = None
    observation: str | None = None
    actions: list[GUIActionStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_split: str | None = None

    @field_validator("sample_id", "instruction", "dataset_name")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned

    @property
    def step_count(self) -> int:
        return len(self.actions)

    def action_type_counts(self) -> dict[str, int]:
        """Action vocabulary of this sample, for dataset statistics."""
        counts: dict[str, int] = {}
        for step in self.actions:
            counts[step.action_type] = counts.get(step.action_type, 0) + 1
        return counts
