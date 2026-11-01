"""Fixed image-only compact selection, prioritizing control-sized proposals.

These heuristics do not establish control semantics. Labels remain separately
reviewed, and the selector has no instruction, target point or annotation input.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .candidate_grounding import center

POLICY = "control_balanced_v2"
LIMIT = 64
QUOTA = 32
GRID = 4


def _tier(item: dict[str, Any], width: int, height: int) -> int:
    box = item["bbox"]
    w, h = box["right"] - box["left"], box["bottom"] - box["top"]
    if item["source"] == "ocr":
        text = item["text"].strip()
        readable = 3 <= len(text) <= 32 and sum(c.isalnum() for c in text) >= 3
        short_line = readable and 8 <= h <= 48 and w >= h
        if short_line and item.get("evidence", {}).get("kind") == "aligned_ocr_parent":
            return 0
        return 1 if short_line else 2
    if 12 <= h <= 64 and 24 <= w <= width * 0.45 and w >= 1.2 * h:
        return 0
    if 12 <= h <= 64 and 12 <= w <= 64:
        return 1
    if 12 <= h <= 96 and 24 <= w <= width * 0.9 and w * h <= width * height * 0.06:
        return 2
    return 3


def select_control_balanced(
    candidates: list[dict[str, Any]], width: int, height: int,
) -> list[dict[str, Any]]:
    """Global meaningful OCR tiers; contour tiers before spatial round-robin."""
    pools = {}
    for source in ("ocr", "contour"):
        items = [c for c in candidates if c["source"] == source]
        if source == "ocr":
            pools[source] = sorted(items, key=lambda c: (
                _tier(c, width, height), -c["confidence"],
                c["bbox"]["top"], c["bbox"]["left"], c["candidate_id"],
            ))
            continue
        ordered = []
        for tier in range(4):
            buckets = defaultdict(list)
            for item in items:
                if _tier(item, width, height) == tier:
                    x, y = center(item["bbox"])
                    buckets[(min(GRID - 1, y * GRID // height),
                             min(GRID - 1, x * GRID // width))].append(item)
            for bucket in buckets.values():
                # Favor bounded controls over panels, without choosing by GT.
                bucket.sort(key=lambda c: (
                    (c["bbox"]["right"] - c["bbox"]["left"]) *
                    (c["bbox"]["bottom"] - c["bbox"]["top"]),
                    c["bbox"]["top"], c["bbox"]["left"], c["candidate_id"],
                ))
            while any(buckets.values()):
                for key in sorted(buckets):
                    if buckets[key]:
                        ordered.append(buckets[key].pop(0))
        pools[source] = ordered
    selected = pools["ocr"][:QUOTA] + pools["contour"][:QUOTA]
    selected.extend((pools["ocr"][QUOTA:] + pools["contour"][QUOTA:])[:LIMIT-len(selected)])
    return sorted(selected, key=lambda c: c["candidate_id"])
