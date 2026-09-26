"""Convert a public GUI dataset into the project's unified JSONL format.

Examples:
    python scripts/week3_prepare_dataset.py --dataset webarena \
        --input data/raw/webarena/test.raw.json --output data/processed/webarena.jsonl --limit 20
    python scripts/week3_prepare_dataset.py --dataset screenagent \
        --input data/raw/screenagent/test.zip --validate-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.datasets import DatasetError, get_adapter, read_records
from gui_agent.datasets.schemas import GUITaskSample
from gui_agent.datasets.validation import (
    collect_stats,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="screenagent, mind2web or webarena")
    parser.add_argument("--input", required=True, help="raw file, JSONL file or zip archive")
    parser.add_argument("--output", default=None, help="destination JSONL path")
    parser.add_argument("--split", default=None, help="split name recorded on each sample")
    parser.add_argument("--limit", type=int, default=20, help="maximum samples to convert")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed when shuffling")
    parser.add_argument("--shuffle", action="store_true", help="shuffle before applying --limit")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="convert and validate, but write nothing to disk",
    )
    parser.add_argument("--stats-json", default=None, help="also write statistics here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2

    try:
        adapter = get_adapter(args.dataset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        records = list(read_records(args.input))
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.shuffle:
        random.Random(args.seed).shuffle(records)

    samples: list[GUITaskSample] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        if len(samples) >= args.limit:
            break
        try:
            samples.append(adapter.to_sample(record, split=args.split))
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the run
            errors.append(f"record {index}: {type(exc).__name__}: {exc}")

    stats = collect_stats(samples)
    stats.total += len(errors)  # records that never became samples
    stats.invalid += len(errors)
    stats.errors = errors

    print(f"dataset : {adapter.name}")
    print(f"input   : {args.input}")
    print(f"limit   : {args.limit}")
    print()
    print(stats.report())

    if args.stats_json:
        target = Path(args.stats_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2), "utf-8")
        print(f"\nstatistics: {target}")

    if args.validate_only:
        print("\n--validate-only: nothing written")
        return 0 if stats.valid else 1

    if not args.output:
        print("\nerror: --output is required unless --validate-only is given", file=sys.stderr)
        return 2

    path = Path(args.output)
    written = write_jsonl(samples, path)
    print(f"\n输出文件位置: {path}  ({written} 行)")

    # The export only counts as good if it reads back through the schema.
    try:
        reloaded = read_jsonl(path)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        print(f"error: the export does not read back: {exc}", file=sys.stderr)
        return 1
    print(f"回读验证: {len(reloaded)} 个样本通过 Schema 校验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
