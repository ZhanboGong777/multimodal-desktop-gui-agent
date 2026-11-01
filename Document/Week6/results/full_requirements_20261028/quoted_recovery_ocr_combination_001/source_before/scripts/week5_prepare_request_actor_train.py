"""Prepare only the registered original V1 train request-actor source contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.request_actor_contract import prepare_request_actor_train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--logical-split", required=True, choices=("train",))
    args = parser.parse_args(argv)
    try:
        result = prepare_request_actor_train(args.plan, args.plan_sha256, logical_split=args.logical_split)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"request actor training-source preparation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"version": result["version"], "logical_split": "train",
                      "original_rows": result["original_rows"], "outputs": result["outputs"],
                      "reason_counts": result["reason_counts"], "training_admission_performed": False,
                      "development_data_consumed": False, "test_data_consumed": False},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
