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
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config, resolve_model_config
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


def _observer_for(config, session, output_directory: Path) -> ObservationService:
    """The observation service, with its frames inside the run they belong to.

    `obs-NNNN.json` names the image each coordinate was measured on, so the image
    has to be where the record says it is - or the traceability the module is built
    around ends at the edge of the session folder. The directory is the run's own
    `frames/`; `output_directory` is still accepted so the fallback stays in one
    place if a session is ever created without one.
    """
    frames = Path(getattr(session, "directory", output_directory)) / "frames"
    return ObservationService(
        config, max_elements=config.execution.max_elements, output_directory=frames
    )


def _show_plan(plan) -> None:
    """Print what a dry run would do: 12.2.2's summary, steps, text and targets.

    The default mode is a dry run, and its whole purpose is to let the operator
    see the plan before deciding to execute it. This display lived only inside the
    execute-mode confirmation, so the mode that exists to show the plan was the
    one mode that never showed it.
    """
    print()
    print(f"  plan       : {plan.summary or '(no summary)'}")
    for step in plan.steps:
        target = f" -> {step.target_text}" if step.target_text else ""
        print(f"    {step.step_id:<8} {step.action_type:<12}{target}")
        typed = step.arguments.get("text") if isinstance(step.arguments, dict) else None
        if typed:
            # The operator is deciding whether to let this run; what it would type
            # is the part that matters most, and it was only visible in execute mode.
            print(f"             would type: {typed}")


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


def load_environment(path: Path = Path(".env")) -> bool:
    """Read `./.env` into the environment, never overriding what is already set.

    `override=False` is the whole point: a value exported in the shell is a decision
    made later than the file, and silently replacing it is how "it worked
    yesterday" happens. The parsing is `python-dotenv`'s, which is already a
    declared dependency - a hand-rolled subset of it lived here first, from before
    anyone checked whether the project already had one.

    Returns whether a file was read. A missing `.env` is normal, not an error.
    """
    from dotenv import load_dotenv

    return bool(load_dotenv(path, override=False))


def _apply_environment(config, args) -> None:
    """Resolve model settings: CLI flag, then environment, then YAML, then default.

    13.1.1 asks for that order and names this entry point as the place to enforce
    it. The rule itself lives in `gui_agent.config`, because the warmup probe has
    to reach the same backend this does, and two copies of a precedence rule is
    how the two drift apart.
    """
    resolve_model_config(
        config,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )


#: The name `week4_warmup.py` writes by default, inside the output directory the
#: runs use. Looked for there rather than relative to the working directory, so a
#: run and its warmup record always come from the same tree.
WARMUP_RECORD_NAME = "warmup.json"


def _attach_warmup(session, record: Path) -> Path | None:
    """Put the warmup record this run follows into the run's own directory.

    16.5.4 asks for the warmup time to be kept beside the run's own timings, and
    13.4.1 asks that it be kept out of them. Both are satisfied by copying the
    record in rather than adding its number to anything: the run directory then
    holds the warmup that preceded it, `week4_collect_evidence.py` carries it into
    the repository with the rest of the evidence, and no task timing changes.

    A copy, not a move, and no timestamp arithmetic: whether a warmup "belongs" to
    a run is the operator's reading of two timestamps, not something this can know.
    """
    source = Path(record)
    if not source.is_file():
        return None
    destination = Path(session.directory) / "warmup.json"
    shutil.copy2(source, destination)
    return destination


def main() -> int:
    # Read the file `.env.example` tells the operator to create, before anything
    # looks at the environment. Values already exported in the shell win.
    dotenv_loaded = load_environment()

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
    # RunSession keys its directory by session id; prefixing with the case id
    # keeps the five task runs distinguishable in the output tree.
    stamp = __import__("datetime").datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    session = RunSession.create(output_directory, session_id=f"{task.case_id}_{stamp}")
    recorder = TaskRecorder(session)
    # The frames belong to the run that took them. They used to be written to the
    # directory the sessions live in, so every run's screenshots piled up beside the
    # session folders and each `obs-NNNN.json` pointed outside its own run.
    observer = _observer_for(config, session, output_directory)
    adapter = ActionAdapter(max_wait_seconds=config.execution.max_wait_seconds)
    executor = ActionExecutor(config.control)
    verifier = Verifier(poll_interval_seconds=config.execution.verification_poll_interval_seconds)
    # 14.1: the effective configuration travels with the run. The summary carries
    # the model and the limits; this is everything else the frames depended on.
    recorder.save_config(config.model_dump(mode="json"))
    warmup_record = _attach_warmup(session, output_directory / WARMUP_RECORD_NAME)

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
        # `load_environment` returns whether it read a file and the answer was
        # thrown away. Settings come from four places; when the endpoint is wrong,
        # "was my .env even read?" is the first question, and `./` means the
        # process's working directory - a run started elsewhere silently skips it.
        print(
            "  env file   : ./.env"
            if dotenv_loaded
            else "  env file   : none in this directory (flags, environment and YAML only)"
        )
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
        if warmup_record is not None:
            print(f"  warmup     : {warmup_record.name} copied from {output_directory}")
        else:
            # Both modes, because both record planning_ms: a cold model inflates a
            # dry run's planning number exactly as much as a real one's, and that
            # number is what the report quotes.
            print(
                f"  warmup     : none in {output_directory} - run scripts/week4_warmup.py "
                "first, or the cold start lands in this run's planning time"
            )

    callbacks = _execute_callbacks(args, task)
    if not args.execute:
        # Not a confirmation: nothing is dispatched, so nothing is being authorised.
        callbacks["on_plan"] = _show_plan

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
    print(f"  records    : {session.directory}")
    print(f"  summary    : {Path(session.directory) / 'task_summary.json'}")

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
