"""Train a local Qwen-VL language LoRA adapter on prepared screenshot/action JSONL.

No downloads or desktop actions are performed. --smoke performs a real optimizer
update (at most one), and remains explicitly labelled as a smoke run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.prompts import TRAINING_PROMPT_VARIANTS
from gui_agent.finetuning.training import (
    INPUT_IMAGE_PROTOCOLS,
    LOSS_STRATEGIES,
    SAMPLING_POLICIES,
    TARGET_PROTOCOLS,
    TOKEN_LOSS_STRATEGIES,
    TrainingConfig,
    run_training,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="saved TrainingConfig JSON; flags below override its values")
    options = {
        "base-model-path": str, "train-jsonl": str, "validation-jsonl": str,
        "output-dir": str, "processor-path": str, "initial-adapter-path": str, "device": str,
        "epochs": int, "max-steps": int, "max-micro-steps": int, "max-optimizer-attempts": int,
        "gradient-accumulation-steps": int,
        "learning-rate": float, "weight-decay": float, "max-grad-norm": float,
        "lora-rank": int, "lora-alpha": int, "lora-dropout": float,
        "min-pixels": int, "max-pixels": int, "max-seq-length": int, "seed": int,
        "visual-asset-manifest-path": str, "visual-asset-manifest-sha256": str,
        "visual-original-root": str, "visual-asset-root": str,
        "visual-prompt-version": str,
        "numeric-tokenizer-json-path": str, "numeric-tokenizer-json-sha256": str,
        "spatial-tokenizer-json-path": str, "spatial-tokenizer-json-sha256": str,
        "spatial-head-learning-rate": float,
    }
    for name, kind in options.items():
        parser.add_argument("--" + name, type=kind, default=None)
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"))
    parser.add_argument("--prompt-variant", choices=TRAINING_PROMPT_VARIANTS)
    parser.add_argument("--target-protocol", choices=TARGET_PROTOCOLS)
    parser.add_argument("--input-image-protocol", choices=INPUT_IMAGE_PROTOCOLS)
    parser.add_argument("--sampling-policy", choices=SAMPLING_POLICIES)
    parser.add_argument("--loss-strategy", choices=LOSS_STRATEGIES)
    parser.add_argument("--token-loss-strategy", choices=TOKEN_LOSS_STRATEGIES)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    try:
        values = TrainingConfig.from_json(args.config).to_dict() if args.config else {}
        for name, value in vars(args).items():
            if name != "config" and value is not None:
                values[name] = value
        config = TrainingConfig.from_dict(values)
        summary = run_training(config)
    except (OSError, TypeError, ValueError) as exc:
        print(f"training refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({name: summary.get(name) for name in (
        "status", "run_kind", "sampling_policy", "loss_strategy", "token_loss_strategy", "termination_reason", "optimizer_steps", "micro_steps",
        "samples_seen", "epochs_completed", "elapsed_seconds", "artifacts", "error",
    )}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 130 if summary["status"] == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
