"""Prepare local public GUI screenshots/actions; no network, model or GUI calls.

Example: python scripts/week5_prepare_data.py --source screenagent train data/raw/train
         --source screenagent test data/raw/test.zip --output-dir data/processed/week5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.data import DataPreparationError, SourceSpec, prepare_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=3, action="append", required=True,
                        metavar=("DATASET", "SOURCE_SPLIT", "PATH"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--source-url", default=None, help="provenance only; never downloaded")
    parser.add_argument("--source-revision", default=None)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_dataset([
            SourceSpec(dataset, split, path, args.source_url, args.source_revision)
            for dataset, split, path in args.source
        ], args.output_dir, seed=args.seed, validation_fraction=args.validation_fraction)
    except (DataPreparationError, OSError, ValueError) as exc:
        print(f"data preparation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(Path(args.output_dir).resolve() / "manifest.json"),
                      "input_records": manifest["input_records"],
                      "retained_samples": manifest["retained_samples"],
                      "exclusion_counts": manifest["exclusion_counts"],
                      "splits": {key: item["samples"] for key, item in manifest["outputs"].items()},
                      "ready_for_training": manifest["ready_for_training"],
                      "ready_for_heldout_comparison": manifest["ready_for_heldout_comparison"]}, indent=2))
    return 0 if manifest["ready_for_heldout_comparison"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
