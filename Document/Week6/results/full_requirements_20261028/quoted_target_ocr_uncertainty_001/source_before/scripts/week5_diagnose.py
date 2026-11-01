"""Freeze train/validation-only diagnostic samples and audit supervision on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.diagnostics import diagnose_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-train-root")
    parser.add_argument("--source-file", action="append", default=[],
                        help="additional pinned source/README/download evidence to hash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--validation-limit", type=int, default=0,
                        help="0 includes all validation; a positive cap uses fixed label strata")
    parser.add_argument("--frozen-selection", help="verify a previous selection without changing it")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    sources = [repo / path for path in (
        "src/gui_agent/finetuning/diagnostics.py", "src/gui_agent/finetuning/data.py",
        "src/gui_agent/finetuning/prompts.py", "scripts/week5_diagnose.py",
        "tests/test_week5_diagnostics.py",
    )] + [Path(path) for path in args.source_file]
    try:
        manifest = diagnose_dataset(
            args.train, args.validation, args.output_dir, seed=args.seed,
            train_limit=args.train_limit, validation_limit=args.validation_limit,
            raw_train_root=args.raw_train_root, frozen_selection=args.frozen_selection,
            source_files=sources,
        )
    except (ValueError, TypeError, OSError) as exc:
        print(f"development diagnostic refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(Path(args.output_dir).resolve() / "manifest.json"),
                      "selected_counts": manifest["selected_counts"],
                      "selection_sha256": manifest["selection_sha256"],
                      "test_data_consumed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
