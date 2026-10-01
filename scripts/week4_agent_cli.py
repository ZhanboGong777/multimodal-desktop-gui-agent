"""Week 4 entry point: run one task through the closed loop.

    python scripts/week4_agent_cli.py --case T01                 # dry run
    python scripts/week4_agent_cli.py --case T01 --execute       # real actions, after confirmation
    python scripts/week4_agent_cli.py --instruction "Open the browser"

Dry run is the default and cannot be turned on implicitly. ``--execute`` does not
skip the confirmation prompt: there is deliberately no ``--yes`` that would let a
script send a message without a human reading it first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.control.executor import ActionExecutor
from gui_agent.models import CLIENTS, create_model_client
from gui_agent.planning import TaskPlanner
from gui_agent.recording import RunSession
from gui_agent.runtime import (
    ActionAdapter,
    ExecutionOptions,
    ObservationService,
    TaskRecorder,
    TaskRunner,
    TaskSpec,
    Verifier,
)
from gui_agent.runtime.tasks import case_ids, get_case, get_case_for_run

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_TIMEOUT = 3
EXIT_CANCELLED = 130

_STATUS_TO_EXIT = {
    "succeeded": EXIT_OK,
    "dry_run_completed": EXIT_OK,
    "failed": EXIT_FAILED,
    "blocked": EXIT_BLOCKED,
    "timed_out": EXIT_TIMEOUT,
    "cancelled": EXIT_CANCELLED,
}


def _positive_int(value: str) -> int:
    """argparse type for a limit that has to be greater than zero.

    Validated here rather than left to `ExecutionOptions`: the model's own error is
    a pydantic traceback, and an uncaught one exits 1 - which the exit-code table
    reserves for "the run failed", not "you typed the arguments wrong".
    """
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Safety: dry run is the default. --execute requires an interactive "
            "confirmation and never sends a message without a second, separate "
            "confirmation for that task."
        ),
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "week4.yaml"))
    parser.add_argument("--instruction", default=None, help="natural-language task")
    parser.add_argument("--case", default=None, help=f"one of: {', '.join(case_ids())}")
    parser.add_argument("--list-cases", action="store_true", help="print the task cases and exit")
    parser.add_argument("--provider", choices=sorted(CLIENTS), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--execute", action="store_true", help="allow real desktop actions")
    parser.add_argument("--max-actions", type=_positive_int, default=None)
    parser.add_argument("--task-timeout", type=_positive_float, default=None)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _confirm(plan) -> bool:
    """Ask once, and treat anything that is not an explicit yes as no."""
    print()
    print(f"  plan: {plan.summary or '(no summary)'}")
    for step in plan.steps:
        target = f" -> {step.target_text}" if step.target_text else ""
        print(f"    {step.step_id:<8} {step.action_type:<12}{target}")
    print()
    try:
        answer = input("  Execute these actions on your desktop? [y/N] ").strip().casefold()
    except EOFError:
        print("  no interactive terminal available; refusing to assume consent")
        return False
    return answer in {"y", "yes"}


def _confirm_high_risk(task: TaskSpec, plan) -> bool:
    """A second, separate confirmation for anything irreversible or public.

    The operator is shown the text the plan will actually type, not just the
    instruction: agreeing to "send the marker" and agreeing to the marker itself
    are different decisions, and only the second one is being made here.
    """
    print()
    print(f"  This task is marked {task.risk!r} and cannot be undone by looking at the screen.")
    print(f"  Task      : {task.instruction}")
    if task.target_app:
        print(f"  Target    : {task.target_app}")
    if task.expect_text:
        print(f"  Must appear on screen afterwards: {task.expect_text}")
    for step in plan.steps:
        text = (step.arguments or {}).get("text")
        if step.action_type == "type_text" and text:
            print(f"  Will type : {text!r}")
    try:
        answer = input("  Confirm this separately? [y/N] ").strip().casefold()
    except EOFError:
        print("  no interactive terminal available; refusing to assume consent")
        return False
    return answer in {"y", "yes"}


def _countdown(seconds: int) -> None:
    print(f"  executing in {seconds}s — move the pointer to a screen corner to abort")
    for remaining in range(seconds, 0, -1):
        print(f"    {remaining}...", flush=True)
        import time

        time.sleep(1)


def _execute_callbacks(args, task: TaskSpec) -> dict[str, object]:
    """The callbacks an execute run hands to the runner.

    Named and returned as a dict so the wiring itself is testable. The second
    confirmation used to be a local variable nobody passed on, which no test
    noticed because no test asked whether the prompt could ever be reached.
    """
    if not args.execute:
        return {}
    return {
        "confirm": _confirm,
        "countdown": _countdown,
        # Passed unconditionally: the runner decides which risk levels need it, so
        # a caller cannot quietly drop the prompt for the tasks that need it most.
        "high_risk_confirm": lambda plan: _confirm_high_risk(task, plan),
    }


def _load_dotenv(path: Path) -> int:
    """Set variables from a `.env` file, never overriding the environment.

    `override=False` is the whole point: a value exported in the shell is a
    decision made later than the file, and silently replacing it is how "it worked
    yesterday" happens. The format is the small subset a template needs - blank
    lines, `#` comments, an optional `export`, `KEY=VALUE`, optional quotes.
    Returns how many variables were actually set.
    """
    if not path.is_file():
        return 0
    applied = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.replace("_", "").isalnum():
            continue
        if key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
        applied += 1
    return applied


def _apply_environment(config, args) -> None:
    """Resolve model settings: CLI flag, then environment, then YAML, then default.

    13.1.1 asks for that order and names this entry point as the place to enforce
    it. It cannot be left to the client: ``ModelConfig.model_name`` has a
    non-empty default, so the client's own environment fallback is never reached -
    setting ``GUI_AGENT_MODEL`` used to change nothing at all.
    """
    for variable, attribute in (
        ("GUI_AGENT_MODEL", "model_name"),
        ("GUI_AGENT_BASE_URL", "base_url"),
    ):
        value = os.environ.get(variable, "").strip()
        if value:
            setattr(config.model, attribute, value)
    if args.provider:
        config.model.provider = args.provider
    if args.model:
        config.model.model_name = args.model
    if args.base_url:
        config.model.base_url = args.base_url


def main() -> int:
    # Read the file `.env.example` tells the operator to create, before anything
    # looks at the environment. Values already exported in the shell win.
    _load_dotenv(Path(".env"))

    args = parse_args()

    if args.list_cases:
        for case_id in case_ids():
            case = get_case(case_id)
            assert case is not None
            print(f"  {case_id}  risk={case.risk:<7} {case.instruction}")
        return EXIT_OK

    if not args.case and not args.instruction:
        print("error: pass --instruction or --case (see --list-cases)", file=sys.stderr)
        return EXIT_BLOCKED

    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot load config: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    _apply_environment(config, args)
    # `is not None`, not truthiness: the old test silently dropped a value the
    # operator had deliberately chosen, and no value here can be zero anyway.
    if args.max_actions is not None:
        config.execution.max_actions = args.max_actions
    if args.task_timeout is not None:
        config.execution.task_timeout_seconds = args.task_timeout

    # ── the task ───────────────────────────────────────────────────────
    message_marker = ""
    if args.case:
        task, message_marker = get_case_for_run(args.case)
        if task is None:
            print(
                f"error: unknown case {args.case!r}; known: {', '.join(case_ids())}",
                file=sys.stderr,
            )
            return EXIT_BLOCKED
        if args.instruction:
            task.instruction = args.instruction
    else:
        # Free-form instructions carry no success rule, and the runner will refuse
        # to call them complete. Say so up front rather than after the run.
        task = TaskSpec(case_id="adhoc", instruction=args.instruction or "")
        print("  note: --instruction without --case defines no success rule;")
        print("        the run will be reported blocked unless you add one.")

    if args.execute and not sys.stdin.isatty():
        # 12.2.5: with no terminal there is no way to ask, and consent is never
        # assumed. Reported as blocked rather than cancelled - the operator did not
        # decline, the confirmation could not be obtained at all.
        #
        # Checked before the run directory is created: a refusal that leaves an
        # empty session behind is indistinguishable from a run that died early, and
        # the evidence collector would later pick it up as "the latest run".
        print(
            "  --execute needs an interactive terminal: the confirmation cannot be "
            "asked for here, and consent is never assumed.",
            file=sys.stderr,
        )
        return EXIT_BLOCKED

    output_directory = Path(args.output_directory or config.output.directory)

    # ── wire the loop ──────────────────────────────────────────────────
    client = create_model_client(config.model)
    planner = TaskPlanner(
        client,
        max_steps=config.planning.max_steps,
        require_structured_output=config.planning.require_structured_output,
        allow_real_execution=config.execution.require_success_rules and args.execute,
    )
    observer = ObservationService(
        config, max_elements=config.execution.max_elements, output_directory=output_directory
    )
    adapter = ActionAdapter(max_wait_seconds=config.execution.max_wait_seconds)
    executor = ActionExecutor(config.control)
    verifier = Verifier(poll_interval_seconds=config.execution.verification_poll_interval_seconds)
    # RunSession keys its directory by session id; prefixing with the case id
    # keeps the five task runs distinguishable in the output tree.
    stamp = __import__("datetime").datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    session = RunSession.create(output_directory, session_id=f"{task.case_id}_{stamp}")
    recorder = TaskRecorder(session)

    options = ExecutionOptions(
        execute=args.execute,
        max_actions=config.execution.max_actions,
        task_timeout_seconds=config.execution.task_timeout_seconds,
        verification_timeout_seconds=config.execution.verification_timeout_seconds,
        verification_poll_interval_seconds=config.execution.verification_poll_interval_seconds,
        confirm=True,
    )

    if not args.quiet:
        print(f"  platform   : {sys.platform}")
        print(f"  provider   : {client.name} ({client.model_name})")
        print(f"  mode       : {'EXECUTE (real desktop actions)' if args.execute else 'dry run'}")
        print(f"  case       : {task.case_id}  risk={task.risk}")
        if message_marker:
            # 15.4: the marker changes every run, so the operator has to be
            # told which one this run is looking for.
            print(f"  marker     : {message_marker}")
        print(
            f"  limits     : {options.max_actions} actions, {options.task_timeout_seconds:g}s budget"
        )
        print(f"  records    : {session.directory}")

    callbacks = _execute_callbacks(args, task)

    runner = TaskRunner(
        observer=observer,
        planner=planner,
        adapter=adapter,
        executor=executor,
        verifier=verifier,
        recorder=recorder,
    )

    try:
        # Splatted, not passed by name: a callback the runner does not accept is a
        # TypeError rather than a safety prompt that silently never fires.
        result = runner.run(task, options, **callbacks)  # type: ignore[arg-type]
    except KeyboardInterrupt:
        print("\n  interrupted by the user; nothing further will be executed")
        return EXIT_CANCELLED

    # ── report ─────────────────────────────────────────────────────────
    print()
    print(f"  Task status : {result.status}")
    print(f"  actions     : {result.action_count}")
    print(f"  elapsed     : {result.elapsed_ms / 1000:.1f}s")
    if result.verification is not None:
        print(f"  verification: {result.verification.outcome} ({result.verification.method})")
        print(f"                {result.verification.detail}")
    if result.error:
        print(f"  error       : {result.error}")
    for note in result.notes:
        print(f"  note        : {note}")
    print(f"  records     : {session.directory}")
    print(f"  summary     : {Path(session.directory) / 'task_summary.json'}")

    if not args.quiet:
        print()
        print(
            json.dumps(
                {"status": result.status, "actions": result.action_count}, ensure_ascii=False
            )
        )

    return _STATUS_TO_EXIT.get(result.status, EXIT_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
