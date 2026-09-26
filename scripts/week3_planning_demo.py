"""Turn an instruction into a validated TaskPlan.

    python scripts/week3_planning_demo.py --provider mock \
        --instruction "Open the browser and search for GUI agents"

The plan is printed and saved to outputs/week3/plans/<task_id>.json. Nothing is
executed: Week 3 stops at the plan, and the executor is wired in during Week 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.models import create_model_client
from gui_agent.planning import TaskPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--provider", choices=["mock", "openai_compatible"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--image", default=None, help="screenshot to include as context")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--quiet", action="store_true", help="only write the file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.provider:
        config.model.provider = args.provider
    if args.model:
        config.model.model_name = args.model
    if args.max_steps:
        config.planning.max_steps = args.max_steps

    client = create_model_client(config.model)
    planner = TaskPlanner(
        client,
        max_steps=config.planning.max_steps,
        require_structured_output=config.planning.require_structured_output,
        allow_real_execution=config.planning.allow_real_execution,
    )

    context = {}
    if args.image:
        context["image_path"] = args.image

    result = planner.plan(args.instruction, context=context, image_path=args.image)

    if not args.quiet:
        print(f"provider   : {client.name} ({client.model_name})")
        print(f"instruction: {args.instruction}")
        print(f"attempts   : {result.attempts}")
        print(f"elapsed    : {result.elapsed_ms:.1f} ms")
        print()

    if not result.ok:
        print(f"no valid plan: {result.error}")
        return 1

    plan = result.plan
    assert plan is not None
    if not args.quiet:
        print(f"task_id    : {plan.task_id}")
        print(f"summary    : {plan.summary}")
        print(f"steps      : {plan.step_count} ({len(plan.executable_steps)} executable)")
        print(f"confirm    : {plan.requires_confirmation}")
        if plan.assumptions:
            print("assumptions:")
            for item in plan.assumptions:
                print(f"  - {item}")
        print()
        print(f"  {'step':<8}{'action':<12}{'target':<14}description")
        print(f"  {'-' * 72}")
        for step in plan.steps:
            target = (step.target_text or "-")[:12]
            marker = "  (terminal)" if step.is_terminal else ""
            print(
                f"  {step.step_id:<8}{step.action_type:<12}{target:<14}{step.description[:34]}{marker}"
            )

    # The config points at outputs/week2; Week 3 keeps its artefacts alongside it.
    base = (
        Path(args.output_directory)
        if args.output_directory
        else Path(str(config.output.directory).replace("week2", "week3"))
    )
    directory = base
    plans = directory / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    target = plans / f"{plan.task_id}.json"
    target.write_text(
        json.dumps(
            {
                "plan": plan.as_dict(),
                "model": result.response.as_dict() if result.response else None,
                "elapsed_ms": round(result.elapsed_ms, 3),
                "attempts": result.attempts,
                "allow_real_execution": result.metadata.get("allow_real_execution", False),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"plan saved : {target}")
    print("nothing was executed: Week 3 produces plans, Week 4 executes them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
