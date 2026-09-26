"""Validation, statistics and JSONL export for the unified samples.

The export is the Week 3 deliverable, so it has to be re-readable: writing is
immediately followed by a round-trip check in the CLI, and ``read_jsonl`` rebuilds
the models through Pydantic rather than trusting the file.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .schemas import GUITaskSample

#: Fields a sample cannot be useful without.
REQUIRED_FIELDS = ("sample_id", "dataset_name", "instruction")


class DatasetExportError(RuntimeError):
    """Raised when samples cannot be written or read back."""


def validate_sample(sample: GUITaskSample) -> list[str]:
    """Return the problems that make a sample unusable, if any.

    An empty action list is *not* a problem: WebArena ships task definitions with
    no trajectory and those are still valid planning inputs.
    """
    problems: list[str] = []
    for name in REQUIRED_FIELDS:
        value = getattr(sample, name, None)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"missing {name}")
    for index, step in enumerate(sample.actions):
        if not step.action_type.strip():
            problems.append(f"step {index}: empty action_type")
    return problems


@dataclass
class DatasetStats:
    """Counts produced by a dataset preparation run."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    missing_fields: dict[str, int] = field(default_factory=dict)
    action_types: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def skip_ratio(self) -> float:
        return self.invalid / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "missing_fields": dict(sorted(self.missing_fields.items())),
            "action_types": dict(sorted(self.action_types.items(), key=lambda kv: -kv[1])),
            "errors": self.errors[:20],
        }

    def report(self) -> str:
        lines = [
            f"读取样本数: {self.total}",
            f"成功转换数: {self.valid}",
            f"跳过样本数: {self.invalid}",
        ]
        if self.missing_fields:
            lines.append("缺失字段统计:")
            for name, count in sorted(self.missing_fields.items()):
                lines.append(f"  {name}: {count}")
        if self.action_types:
            lines.append("动作类型统计:")
            for name, count in sorted(self.action_types.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name}: {count}")
        if self.errors:
            lines.append("错误样例:")
            for message in self.errors[:5]:
                lines.append(f"  {message}")
        return "\n".join(lines)


def collect_stats(samples: Iterable[GUITaskSample], *, errors: Sequence[str] = ()) -> DatasetStats:
    """Fold a stream of samples into the counts the CLI reports."""
    stats = DatasetStats(errors=list(errors))
    for sample in samples:
        stats.total += 1
        problems = validate_sample(sample)
        if problems:
            stats.invalid += 1
            for problem in problems:
                key = problem.split(":", 1)[0]
                stats.missing_fields[key] = stats.missing_fields.get(key, 0) + 1
            continue
        stats.valid += 1
        for name, count in sample.action_type_counts().items():
            stats.action_types[name] = stats.action_types.get(name, 0) + count
    return stats


def write_jsonl(samples: Iterable[GUITaskSample], path: str | Path) -> int:
    """Write one sample per line. Returns how many were written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json(exclude_none=False))
            handle.write("\n")
            written += 1
    return written


def read_jsonl(path: str | Path) -> list[GUITaskSample]:
    """Read a JSONL export back through the schema, so a bad file fails loudly."""
    target = Path(path)
    if not target.exists():
        raise DatasetExportError(f"no such export: {target}")
    samples: list[GUITaskSample] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(GUITaskSample.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetExportError(f"{target}:{number} is not a valid sample: {exc}") from exc
    return samples


def iter_jsonl(path: str | Path) -> Iterator[GUITaskSample]:
    """Stream a JSONL export without holding it all in memory."""
    yield from read_jsonl(path)
