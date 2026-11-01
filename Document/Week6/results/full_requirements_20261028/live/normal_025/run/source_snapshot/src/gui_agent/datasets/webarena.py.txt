"""WebArena adapter.

WebArena ships task definitions rather than trajectories: an intent, a start URL
and an evaluation specification. There is no action sequence to extract, so a
sample carries the intent and the start URL and leaves ``actions`` empty. That is
still enough for the planner to produce a first plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import DatasetAdapter
from .normalize import clean_text, first_present
from .schemas import GUITaskSample


class WebArenaAdapter(DatasetAdapter):
    name = "webarena"

    def to_sample(self, record: Mapping[str, Any], *, split: str | None = None) -> GUITaskSample:
        payload = dict(record)
        instruction = clean_text(
            first_present(payload, "intent", "instruction", "task", "goal", "query")
        )
        if not instruction:
            raise ValueError("WebArena record has no intent")

        sample_id = (
            clean_text(first_present(payload, "task_id", "id", "sample_id", "uid"))
            or f"webarena-{abs(hash(instruction)) % 10**8}"
        )

        sites = payload.get("sites") or payload.get("site")
        if isinstance(sites, Sequence) and not isinstance(sites, (str, bytes)):
            site_list = [clean_text(s) for s in sites if clean_text(s)]
        else:
            site_list = [clean_text(sites)] if clean_text(sites) else []

        metadata: dict[str, Any] = {
            "start_url": clean_text(first_present(payload, "start_url", "url", "start")) or None,
            "sites": site_list,
            "require_login": payload.get("require_login"),
            # The evaluation spec is what makes WebArena useful as a benchmark.
            "eval": payload.get("eval") or payload.get("reference_answers"),
        }

        return GUITaskSample(
            sample_id=sample_id,
            dataset_name=self.name,
            instruction=instruction,
            image_path=None,  # WebArena tasks are live pages, not stored screenshots.
            actions=[],
            source_split=split,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
