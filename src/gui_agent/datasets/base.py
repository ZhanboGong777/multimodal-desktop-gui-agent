"""Adapter interface shared by the three dataset sources.

An adapter has two halves on purpose:

* :meth:`DatasetAdapter.to_sample` is **pure** - it takes one raw record and
  returns a :class:`GUITaskSample`. That is where the semantic mapping lives, and
  it can be unit tested against small hand-written fixtures with no download.
* :func:`read_records` does the I/O - JSON, JSONL or a zip of either - so the
  fixtures and a real download go through exactly the same code path.
"""

from __future__ import annotations

import json
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from .schemas import GUITaskSample


class DatasetError(RuntimeError):
    """Raised when a source cannot be read at all."""


class DatasetAdapter(ABC):
    """Translates one source dataset into :class:`GUITaskSample` objects."""

    #: Value written to ``GUITaskSample.dataset_name``.
    name: ClassVar[str]

    @abstractmethod
    def to_sample(self, record: Mapping[str, Any], *, split: str | None = None) -> GUITaskSample:
        """Translate one raw record. Pure - no file or network access."""

    def summary(self) -> str:
        return f"{self.name} adapter"


def read_records(source: Path | str) -> Iterator[dict[str, Any]]:
    """Yield raw records from a JSON file, a JSONL file, or a zip of them.

    Used for fixtures and for the smaller real sources (ScreenAgent's ``test.zip``
    and WebArena's ``test.raw.json``). HuggingFace-backed sources stream instead.
    """
    path = Path(source)
    if not path.exists():
        raise DatasetError(f"source does not exist: {path}")

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if member.endswith("/") or not member.endswith((".json", ".jsonl")):
                    continue
                with archive.open(member) as handle:
                    yield from _parse_bytes(handle.read(), member)
        return

    yield from _parse_bytes(path.read_bytes(), str(path))


def _parse_bytes(payload: bytes, label: str) -> Iterator[dict[str, Any]]:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return

    # Try the whole document first. A pretty-printed JSON file contains newlines
    # but is NOT JSONL, and a naive "has a newline, so treat it as JSONL" test
    # shreds it: every inner line then carries a trailing comma and is dropped.
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        yield from _parse_jsonl(text)
        return

    if isinstance(document, list):
        for item in document:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(document, dict):
        # WebArena and some exports wrap the list under a key.
        for key in ("tasks", "data", "records", "samples", "items"):
            value = document.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield document


def _parse_jsonl(text: str) -> Iterator[dict[str, Any]]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            yield item
