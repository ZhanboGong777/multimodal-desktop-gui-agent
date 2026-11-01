"""Prepare verified conditional actor train/dev data; never consume official test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.actor_contract import prepare_actor_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "raw-train-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_actor_contract(args.train, args.validation, args.raw_train_root, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"conditional actor preparation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"outputs": result["outputs"], "reason_counts": result["reason_counts"],
                      "test_data_consumed": False, "tokenized": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
