"""Prepare source-only original-goal development data without teacher step context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.step_contract import prepare_original_goals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "raw-train-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_original_goals(args.train, args.validation, args.raw_train_root, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"original-goal preparation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"outputs": manifest["outputs"], "selected_fields": manifest["selected_fields"],
                      "test_data_consumed": False, "tokenized": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
