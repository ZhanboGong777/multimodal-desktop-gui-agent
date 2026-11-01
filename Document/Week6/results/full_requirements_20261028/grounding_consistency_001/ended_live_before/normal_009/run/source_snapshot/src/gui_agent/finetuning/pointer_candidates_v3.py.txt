"""Opt-in query and geometry relations over unchanged raw proposals.

Only the goal/current subtask, OCR text and proposal geometry enter selection.
Relations are proposal evidence, never proof of control identity or hit success.
The 24 spatial + 8 contextual allocation is fixed; no labels or targets enter it.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from typing import Any

POLICY = "query_relations_v3"
LIMIT = 64
QUOTA = 32
SPATIAL = 24
CONTEXTUAL = 8
GRID = 4
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_WORDS = re.compile(r"[a-z]+|\d+")
_QUOTED = re.compile(r'''["“‘「『]([^"”’」』]+)["”’」』]|(?<!\w)'([^']+)'(?!\w)''')


def _normal(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def _tokens(text: str) -> set[str]:
    normalized = _normal(text)
    tokens = set(_WORDS.findall(normalized))
    for run in _CJK.findall(normalized):
        tokens.update(run[i:i+2] for i in range(len(run)-1))
    return tokens


def _query(instruction: str, actor_step: str) -> list[tuple[str, set[str], list[set[str]], int]]:
    return [(_normal(text), _tokens(text),
             [_tokens(a or b) for a, b in _QUOTED.findall(text)], weight)
            for text, weight in ((actor_step, 2), (instruction, 1))]


def _score(text: str, query: list[tuple[str, set[str], list[set[str]], int]]) -> int:
    normalized, tokens = _normal(text), _tokens(text)
    if sum(c.isalnum() for c in normalized) < 2:
        return 0
    # Latin/digit boundaries prevent a short word from matching inside another.
    phrase = re.compile(r"(?<![a-z0-9])" + re.escape(normalized) + r"(?![a-z0-9])")
    return sum(weight * (len(tokens & context_tokens)
                        + 2 * bool(phrase.search(context))
                        + max((len(tokens & quoted) for quoted in quotes), default=0))
               for context, context_tokens, quotes, weight in query)


def _geometry(item: dict[str, Any]) -> tuple[int, int, int, int]:
    box = item["bbox"]
    return box["left"], box["top"], box["right"], box["bottom"]


def _size(item: dict[str, Any]) -> tuple[int, int]:
    left, top, right, bottom = _geometry(item)
    return right-left, bottom-top


def _center(item: dict[str, Any]) -> tuple[int, int]:
    left, top, right, bottom = _geometry(item)
    return (left+right)//2, (top+bottom)//2


def _area(item: dict[str, Any]) -> int:
    width, height = _size(item)
    return width*height


def _validate(candidates: list[dict[str, Any]], width: int, height: int) -> None:
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("positive integer native dimensions are required")
    identifiers = set()
    for item in candidates:
        identifier = item["candidate_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("raw candidate IDs must be unique nonempty strings")
        identifiers.add(identifier)
        if item["source"] not in {"ocr", "contour"}:
            raise ValueError("only raw OCR and contour proposals are supported")
        box = item["bbox"]
        if set(box) != {"left", "top", "right", "bottom"} or any(
            type(v) is not int for v in box.values()
        ):
            raise ValueError("raw candidate bbox requires four integer edges")
        left, top, right, bottom = _geometry(item)
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError("raw candidate bbox is outside the native image")
        confidence = item["confidence"]
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("confidence must be finite in [0,1]")
        if not isinstance(item["text"], str):
            raise TypeError("raw candidate text must be a string")


def _ocr_rank(item: dict[str, Any], query: list) -> tuple:
    text = _normal(item["text"])
    width, height = _size(item)
    alphanumeric = sum(c.isalnum() for c in text)
    meaningful = alphanumeric >= 2
    short_label = 2 <= alphanumeric <= 64 and 8 <= height <= 64 and width >= height
    parent = item.get("evidence", {}).get("kind") == "aligned_ocr_parent"
    return (int(not meaningful), -_score(text, query), int(not short_label),
            -int(parent), -item["confidence"],
            item["bbox"]["top"], item["bbox"]["left"], item["candidate_id"])


def _unique_geometry(candidates: list[dict[str, Any]], query: list) -> list[dict[str, Any]]:
    # A raw parent with the same geometry cannot occupy another source slot.
    result, seen = [], set()
    for item in sorted(candidates, key=lambda c: (
        c["source"], _ocr_rank(c, query) if c["source"] == "ocr" else (c["candidate_id"],)
    )):
        key = item["source"], _geometry(item)
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _spatial(items: list[dict[str, Any]], source: str, width: int, height: int) -> list:
    """Exactly the v1 source-local 4x4 confidence/area round-robin ordering."""
    buckets = defaultdict(list)
    for item in items:
        x, y = _center(item)
        buckets[(min(GRID-1, y*GRID//height), min(GRID-1, x*GRID//width))].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda c: (-c["confidence"] if source == "ocr" else 0,
                                   -_area(c), c["candidate_id"]))
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    return ordered


def _contains(parent: dict[str, Any], child: dict[str, Any]) -> bool:
    left, top, right, bottom = _geometry(parent)
    cl, ct, cr, cb = _geometry(child)
    return (left <= cl and top <= ct and cr <= right and cb <= bottom
            and _geometry(parent) != _geometry(child))


def _small(item: dict[str, Any], width: int, height: int) -> bool:
    w, h = _size(item)
    return 10 <= h <= 64 and 12 <= w <= min(180, width//4) and w*h <= width*height*0.03


def _occupied(item: dict[str, Any], ocr: list[dict[str, Any]]) -> bool:
    left, top, right, bottom = _geometry(item)
    for text in ocr:
        if not any(c.isalnum() for c in text["text"]):
            continue
        tl, tt, tr, tb = _geometry(text)
        overlap = max(0, min(right, tr)-max(left, tl)) * max(0, min(bottom, tb)-max(top, tt))
        if overlap >= min(_area(item), _area(text))/2:
            return True
    return False


def _field(item: dict[str, Any], ocr: list[dict[str, Any]]) -> bool:
    w, h = _size(item)
    return (not _normal(item["text"]) and 10 <= h <= 48 and w >= 3*h
            and not _occupied(item, ocr))


def _field_pair(a: dict[str, Any], b: dict[str, Any]) -> bool:
    aw, ah = _size(a)
    bw, bh = _size(b)
    left, right = sorted((a, b), key=lambda c: c["bbox"]["left"])
    gap = right["bbox"]["left"]-left["bbox"]["right"]
    return (max(aw, bw) <= 1.5*min(aw, bw)
            and max(ah, bh) <= 1.5*min(ah, bh)
            and abs(a["bbox"]["bottom"]-b["bbox"]["bottom"]) <= max(3, 0.25*min(ah, bh))
            and 0 <= gap <= max(48, 2*max(ah, bh)))


def _adjacent(region: dict[str, Any], text: dict[str, Any]) -> bool:
    left, top, right, bottom = _geometry(region)
    tl, tt, tr, tb = _geometry(text)
    h, th = bottom-top, tb-tt
    if left <= (tl+tr)//2 <= right and top <= (tt+tb)//2 <= bottom:
        return True
    overlap_y = min(bottom, tb)-max(top, tt)
    horizontal_gap = max(left-tr, tl-right, 0)
    horizontal = overlap_y >= min(h, th)/2 and horizontal_gap <= max(48, 2*h)
    above_gap = top-tb
    above = min(right, tr) > max(left, tl) and 0 <= above_gap <= max(24, h)
    return horizontal or above


def _relations(contours: list[dict[str, Any]], ocr: list[dict[str, Any]],
               width: int, height: int, query: list) -> list[list[dict[str, Any]]]:
    small = [c for c in contours if _small(c, width, height)]
    coarse = {c["candidate_id"]: sum(_contains(c, child) for child in small) >= 2
              for c in contours}
    ocr_relevance = [(t, _score(t["text"], query)) for t in ocr]
    relevance = {c["candidate_id"]: max(
        [_score(c["text"], query)]
        + [score for text, score in ocr_relevance if score and _adjacent(c, text)]
    ) for c in contours}
    groups = []

    def add(members: list, tier: int) -> None:
        members = sorted(members, key=lambda c: c["candidate_id"])
        rank = (max(coarse[c["candidate_id"]] for c in members),
                -max(relevance[c["candidate_id"]] for c in members), tier,
                sum(_area(c) for c in members), tuple(c["candidate_id"] for c in members))
        groups.append((rank, members))

    fields = [c for c in contours if _field(c, ocr)]
    for index, candidate in enumerate(fields):
        for other in fields[index+1:]:
            if _field_pair(candidate, other):
                add([candidate, other], 0)
    for candidate in contours:
        if candidate.get("evidence", {}).get("kind") == "contour_text_line_parent":
            add([candidate], 3)
        elif _small(candidate, width, height):
            add([candidate], 2)
        elif relevance[candidate["candidate_id"]]:
            add([candidate], 4)
    return [members for _, members in sorted(groups, key=lambda entry: entry[0])]


def select_query_relations(
    candidates: list[dict[str, Any]], width: int, height: int, *,
    instruction: str, actor_step: str,
) -> list[dict[str, Any]]:
    """Select raw objects with fixed spatial and query/relation source quotas.

    A field pair consumes only its not-yet-selected members, and is considered
    only when the remaining relationship slots can accommodate the entire pair.
    Coarse parents containing multiple small frames are ranked after the frames;
    no parent is created or substituted for a child by this selector.
    """
    if any(not isinstance(text, str) or not text.strip() for text in (instruction, actor_step)):
        raise ValueError("nonempty instruction and actor_step are required")
    _validate(candidates, width, height)
    query = _query(instruction, actor_step)
    unique = _unique_geometry(candidates, query)
    pools = {source: _spatial([c for c in unique if c["source"] == source],
                              source, width, height) for source in ("ocr", "contour")}
    context = {"ocr": [[c] for c in sorted(pools["ocr"], key=lambda c: _ocr_rank(c, query))],
               "contour": _relations(pools["contour"], pools["ocr"], width, height, query)}
    selected = []
    identifiers = set()
    for source in ("ocr", "contour"):
        chosen = pools[source][:SPATIAL]
        identifiers.update(c["candidate_id"] for c in chosen)
        used = 0
        for group in context[source]:
            fresh = [c for c in group if c["candidate_id"] not in identifiers]
            if fresh and used+len(fresh) <= CONTEXTUAL:
                chosen.extend(fresh)
                identifiers.update(c["candidate_id"] for c in fresh)
                used += len(fresh)
        for candidate in pools[source]:
            if len(chosen) >= QUOTA:
                break
            if candidate["candidate_id"] not in identifiers:
                chosen.append(candidate)
                identifiers.add(candidate["candidate_id"])
        selected.extend(chosen)
    for candidate in pools["ocr"]+pools["contour"]:
        if len(selected) >= LIMIT:
            break
        if candidate["candidate_id"] not in identifiers:
            selected.append(candidate)
            identifiers.add(candidate["candidate_id"])
    return sorted(selected, key=lambda c: c["candidate_id"])
