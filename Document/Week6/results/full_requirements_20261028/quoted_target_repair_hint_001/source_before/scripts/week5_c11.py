"""Prepare/train a NEW C11 recipe run. Defaults to a metadata-only plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.c11_training import prepare, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--train-jsonl", required=True)
    preparation.add_argument("--validation-jsonl", required=True)
    preparation.add_argument("--base-model", required=True)
    training = commands.add_parser("train")
    training.add_argument("--prepared", required=True)
    for command in (preparation, training):
        command.add_argument("--output", required=True)
        command.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "plan_only", "arguments": vars(args), "model_loads": 0,
                          "processor_calls": 0, "desktop_inputs": 0}))
        return 0
    if args.command == "prepare":
        result = prepare(args.train_jsonl, args.validation_jsonl, args.base_model, args.output)
    else:
        result = train(args.prepared, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
