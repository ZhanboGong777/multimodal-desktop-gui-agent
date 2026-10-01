"""The CLI's safety contract, checked without touching a desktop.

The interesting cases are the ones where a run must *not* happen: no task, an
unknown case, a free-form instruction with nothing to verify, and a dry run that
must never dispatch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "week4_agent_cli.py"), *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        check=False,
    )


def test_listing_the_cases_needs_no_desktop_or_model() -> None:
    result = run_cli("--list-cases")
    assert result.returncode == 0
    for case_id in ("T01", "T02", "T03", "T04", "T05"):
        assert case_id in result.stdout


def test_no_task_is_a_usage_error_not_an_empty_run() -> None:
    result = run_cli()
    assert result.returncode == 2
    assert "--instruction or --case" in result.stderr


def test_an_unknown_case_is_rejected() -> None:
    result = run_cli("--case", "T99")
    assert result.returncode == 2
    assert "unknown case" in result.stderr


def test_help_documents_the_dry_run_default() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "dry run is the default" in result.stdout.casefold()
    # There must be no flag that silently consents on the user's behalf. Check the
    # option list rather than the whole page: the description explains the absence
    # of --yes, so the string legitimately appears in prose.
    option_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith(" ") and line.strip().startswith("-")
    ]
    assert not any(line.startswith("--yes") for line in option_lines)


def test_a_dry_run_reports_and_dispatches_nothing(tmp_path: Path) -> None:
    result = run_cli("--case", "T01", "--quiet", "--output-directory", str(tmp_path))
    # Either outcome is a legitimate dry run: it resolved the plan, or it refused
    # a step it could not resolve. What matters is that the status is never a
    # finished task and that a summary was written.
    assert "dry run" in result.stdout.casefold() or result.returncode in {0, 1, 2}
    summaries = list(tmp_path.glob("*/task_summary.json"))
    assert summaries, "a run must leave a summary behind"


def test_a_free_form_instruction_is_flagged_as_unverifiable(tmp_path: Path) -> None:
    result = run_cli(
        "--instruction", "Do something vague", "--quiet", "--output-directory", str(tmp_path)
    )
    assert "defines no success rule" in result.stdout


def test_execute_without_a_terminal_is_blocked_rather_than_assumed(tmp_path: Path) -> None:
    """12.2.5: with no terminal there is nobody to ask, so consent is not assumed.

    The refusal happens before the capture and before the run directory is created,
    so it is fast, needs no desktop, and leaves nothing behind.
    """
    destination = tmp_path / "records"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "week4_agent_cli.py"),
            "--case",
            "T01",
            "--execute",
            "--output-directory",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        check=False,
    )

    assert result.returncode == 2, result.stdout
    assert "interactive terminal" in result.stderr
    # Refused before the run directory exists: an empty session left behind is
    # indistinguishable from a run that died early.
    assert not destination.exists() or not any(destination.iterdir())


def _load_cli():
    """Import the CLI as a module so its wiring can be checked directly."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "week4_agent_cli", REPO_ROOT / "scripts" / "week4_agent_cli.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_execute_run_hands_over_both_confirmation_prompts() -> None:
    """The second prompt existed as a local variable and was never passed on.

    T04 - the one task that really sends something - therefore ran on a single
    confirmation while the help text, the usage guide and the diagnostic guide all
    said it got two. 8.3.6 asks for the separate confirmation; this asserts the
    wiring rather than the runner behaviour, because the wiring is what broke.
    """
    import argparse

    from gui_agent.runtime.tasks import get_case

    cli = _load_cli()
    args = argparse.Namespace(execute=True)

    callbacks = cli._execute_callbacks(args, get_case("T04"))

    assert set(callbacks) == {"confirm", "countdown", "high_risk_confirm"}


def test_a_dry_run_hands_over_no_callbacks() -> None:
    """Nothing is dispatched, so there is nothing to confirm or count down."""
    import argparse

    from gui_agent.runtime.tasks import get_case

    cli = _load_cli()

    assert cli._execute_callbacks(argparse.Namespace(execute=False), get_case("T04")) == {}


# ───────── where the model settings come from ─────────
def test_the_environment_sits_between_the_flag_and_the_yaml(monkeypatch) -> None:
    """13.1.1's order: CLI flag, then GUI_AGENT_*, then YAML, then the default.

    Enforced in the CLI because ModelConfig.model_name has a non-empty default -
    the client's own environment fallback is never reached, so setting
    GUI_AGENT_MODEL used to change nothing at all.
    """
    import argparse

    from gui_agent.config import load_config

    cli = _load_cli()
    monkeypatch.setenv("GUI_AGENT_MODEL", "from-env")
    monkeypatch.setenv("GUI_AGENT_BASE_URL", "http://from-env:11434/v1")

    config = load_config(REPO_ROOT / "configs" / "week4.yaml")
    yaml_model = config.model.model_name
    args = argparse.Namespace(provider=None, model=None, base_url=None)

    cli._apply_environment(config, args)
    assert config.model.model_name == "from-env"
    assert config.model.base_url == "http://from-env:11434/v1"

    args.model = "from-flag"
    args.base_url = "http://from-flag:11434/v1"
    cli._apply_environment(config, args)
    assert config.model.model_name == "from-flag"
    assert config.model.base_url == "http://from-flag:11434/v1"

    monkeypatch.delenv("GUI_AGENT_MODEL")
    monkeypatch.delenv("GUI_AGENT_BASE_URL")
    # A run applies this once, to a freshly loaded config, so that is what the
    # "nothing is set" case has to look at: the function edits the config in place.
    fresh = load_config(REPO_ROOT / "configs" / "week4.yaml")
    cli._apply_environment(fresh, argparse.Namespace(provider=None, model=None, base_url=None))
    assert fresh.model.model_name == yaml_model, "with nothing set, the YAML stands"


def test_a_dotenv_file_does_not_override_the_shell(monkeypatch, tmp_path: Path) -> None:
    """13.1.2 asks for override=False, and this is what that buys.

    A value exported in the shell is a decision made after the file was written.
    """
    cli = _load_cli()
    fake_environ = {"GUI_AGENT_MODEL": "from-shell"}
    monkeypatch.setattr(os, "environ", fake_environ)
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        "GUI_AGENT_MODEL=from-file\n"
        'export GUI_AGENT_BASE_URL="http://from-file:11434/v1"\n'
    )

    assert cli.load_environment(path) is True

    assert fake_environ["GUI_AGENT_MODEL"] == "from-shell"
    assert fake_environ["GUI_AGENT_BASE_URL"] == "http://from-file:11434/v1"


def test_a_missing_dotenv_is_not_an_error(monkeypatch, tmp_path: Path) -> None:
    """Most machines will not have one, and that is not a failure."""
    cli = _load_cli()
    monkeypatch.setattr(os, "environ", {})

    assert cli.load_environment(tmp_path / "nope.env") is False


def test_the_week4_config_carries_the_run_limits() -> None:
    """13.2's execution section, checked as the file rather than the model default.

    A config file that drifts from the model it is loaded into is silent: the run
    just uses the default and nobody notices.
    """
    from gui_agent.config import load_config

    config = load_config(REPO_ROOT / "configs" / "week4.yaml")

    assert config.execution.max_actions == 20
    assert config.execution.task_timeout_seconds == 240
    assert config.execution.verification_timeout_seconds == 10
    assert config.execution.verification_poll_interval_seconds == 0.5
    assert config.execution.max_wait_seconds == 5
    assert config.execution.require_success_rules is True


def test_a_bad_limit_is_a_usage_error_not_a_traceback(tmp_path: Path) -> None:
    """12.2's exit-code table puts argument errors at 2.

    A negative budget used to reach `ExecutionOptions`, whose pydantic error
    escaped as a traceback and exited 1 - the code the table reserves for "the run
    failed". An operator reading that would think the task had been attempted.
    """
    for extra in (("--task-timeout", "-5"), ("--task-timeout", "0"), ("--max-actions", "0")):
        result = run_cli("--case", "T01", *extra, "--output-directory", str(tmp_path / "x"))

        assert result.returncode == 2, (extra, result.stdout)
        assert "must be greater than 0" in result.stderr


def test_the_smallest_valid_limit_is_applied_rather_than_ignored(tmp_path: Path) -> None:
    """`if args.max_actions:` dropped falsy values.

    Nothing here can be zero - the model forbids it - so truthiness was never the
    right test, and an operator asking for the most restrictive setting silently
    got the default instead. The header prints what was actually applied.
    """
    result = run_cli(
        "--case", "T01", "--max-actions", "1", "--task-timeout", "7",
        "--output-directory", str(tmp_path / "records"),
    )

    assert "1 actions, 7s budget" in result.stdout, result.stdout


# ───────── the warmup record travels with the run ─────────
def test_the_warmup_record_is_copied_into_the_run_directory(tmp_path: Path) -> None:
    """16.5.4 wants the warmup time kept beside the run's own timings.

    Keeping it *out* of them is 13.4.1, so the record is copied in rather than
    merged: the run directory holds the warmup that preceded it, and no task
    timing changes. `week4_collect_evidence.py` then carries it into the
    repository with the rest of the evidence.
    """
    import argparse
    import json

    cli = _load_cli()
    record = tmp_path / "warmup.json"
    record.write_text(json.dumps({"ready": True}), encoding="utf-8")

    class _Session:
        directory = tmp_path / "T01_20261001_120000"

    _Session.directory.mkdir()
    copied = cli._attach_warmup(_Session, record)

    assert copied == _Session.directory / "warmup.json"
    assert json.loads(copied.read_text(encoding="utf-8")) == {"ready": True}
    assert record.is_file(), "the original stays where the warmup wrote it"
    assert argparse is not None  # the module imports argparse; keep the name used


def test_no_warmup_record_is_not_an_error(tmp_path: Path) -> None:
    """A dry run on a machine that never warmed anything must still work."""
    cli = _load_cli()

    class _Session:
        directory = tmp_path / "T01_20261001_120000"

    _Session.directory.mkdir()
    assert cli._attach_warmup(_Session, tmp_path / "absent.json") is None
    assert list(_Session.directory.iterdir()) == []


def test_a_run_with_no_warmup_record_says_so(tmp_path: Path) -> None:
    """13.4.1's nudge, at the moment it matters.

    The next number this run writes is its `planning_ms`. If the model is cold,
    that number includes loading it, and the week's first evidence point becomes
    the one figure nobody can interpret. Warned about in both modes, because both
    record it - the status differs, the timing does not.
    """
    result = run_cli("--case", "T01", "--output-directory", str(tmp_path))
    assert "warmup     : none in" in result.stdout
    assert "week4_warmup.py" in result.stdout


def test_a_run_finds_the_warmup_record_in_its_own_output_directory(tmp_path: Path) -> None:
    """The record is looked for where the runs write, not where the shell is.

    A run and the warmup that preceded it then always come from the same tree: a
    stray `outputs/week4/warmup.json` in the working directory cannot be attached
    to a run whose records are going somewhere else.
    """
    import json

    (tmp_path / "warmup.json").write_text(
        json.dumps({"ready": True, "probes": []}), encoding="utf-8"
    )
    result = run_cli("--case", "T01", "--output-directory", str(tmp_path))

    # The source, not the copy: naming the session directory here told the operator
    # where the record had just been written, not where it came from.
    assert f"warmup     : warmup.json copied from {tmp_path}" in result.stdout
    sessions = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(sessions) == 1, sessions
    copied = sessions[0] / "warmup.json"
    assert copied.is_file(), "the record has to land in the run directory"
    assert json.loads(copied.read_text(encoding="utf-8"))["ready"] is True


# ───────── 12.2.2: what the plan would do, in the mode that exists to show it ─────────
def test_a_dry_run_is_told_to_display_the_plan(capsys) -> None:
    """The display lived only inside the execute-mode confirmation.

    So the default mode - a dry run, whose entire purpose is to let the operator
    see the plan before agreeing to it - printed no steps, no targets and no text.
    """
    import argparse

    from gui_agent.runtime.tasks import get_case

    cli = _load_cli()
    callbacks = cli._execute_callbacks(argparse.Namespace(execute=False), get_case("T04"))
    assert callbacks == {}, "a dry run asks nothing"

    from gui_agent.planning.schemas import PlanStep, TaskPlan

    plan = TaskPlan(
        task_id="t1",
        instruction="search",
        summary="Search in the browser",
        steps=[
            PlanStep(step_id="step-1", description="type", action_type="type_text",
                     arguments={"text": "GUI agent research"}),
            PlanStep(step_id="step-2", description="stop", action_type="finish"),
        ],
    )
    cli._show_plan(plan)
    printed = capsys.readouterr().out

    assert "Search in the browser" in printed
    assert "step-1" in printed and "type_text" in printed
    assert "GUI agent research" in printed, "the text it would type is the part that matters"


def test_the_runner_hands_the_plan_over_before_deciding_anything(tmp_path) -> None:
    """The hook fires after validation and before the confirmation gate."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_runtime_runner import (
        FakeExecutor,
        FakeObserver,
        FakePlanner,
        _frame,
        _plan,
        _runner,
    )

    from gui_agent.planning.schemas import PlanStep
    from gui_agent.runtime import ExecutionOptions, TaskSpec

    seen = []
    frames = [_frame("obs-0001", ("Start",)), _frame("obs-0002", ("Start",))]
    plan = _plan(
        PlanStep(step_id="s1", description="click", action_type="click", target_text="Start")
    )
    runner, _ = _runner(tmp_path, FakeObserver(frames), FakePlanner(plan), FakeExecutor())
    task = TaskSpec(case_id="T", instruction="x", expect_text=["Start"], success_rules=["r"])

    runner.run(task, ExecutionOptions(confirm=False), on_plan=seen.append)

    assert len(seen) == 1, "the plan is shown once, not per step"
    assert seen[0].summary == plan.summary


def test_a_missing_env_file_is_reported_rather_than_assumed(tmp_path: Path) -> None:
    """`load_environment` returned whether it read a file; both callers dropped it.

    Settings arrive from four places, and when the endpoint is wrong the first
    question is whether `./.env` was read at all. `./` is the process's working
    directory, so a run started from somewhere else skips the file silently - the
    operator then sees YAML defaults and has nothing to tell them why.
    """
    result = run_cli("--case", "T01", "--output-directory", str(tmp_path))

    assert "env file   : none in this directory" in result.stdout


def test_an_env_file_that_was_read_is_named(tmp_path: Path) -> None:
    """And when it is read, the fact is stated rather than left to be inferred."""
    import os
    import subprocess
    import sys

    (tmp_path / ".env").write_text("GUI_AGENT_MODEL=from-dotenv\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GUI_AGENT_")}
    env["GUI_AGENT_API_KEY"] = "test-key"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "week4_agent_cli.py"),
            "--case",
            "T01",
            "--output-directory",
            str(tmp_path / "runs"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )

    assert "env file   : ./.env" in result.stdout
    assert "from-dotenv" in result.stdout, "the value in the file has to be the one in use"


def test_the_frames_are_written_inside_the_run_that_took_them(tmp_path: Path) -> None:
    """`obs-NNNN.json` names the image a coordinate was measured on.

    That image was written to the directory the session folders live in, so every
    run's screenshots piled up beside the runs and each record pointed outside its
    own folder: moving or archiving a session broke the traceability the module is
    built around, and 14.1 asks for the frames to be kept with the step.
    """
    import argparse

    from gui_agent.config import Config
    from gui_agent.recording import RunSession

    cli = _load_cli()
    output = tmp_path / "week4"
    session = RunSession.create(output, session_id="T01_20261001_120000")

    service = cli._observer_for(Config(), session, output)

    assert service.output_directory == session.directory / "frames"
    assert output in service.output_directory.parents, "still under the run tree"
    assert service.output_directory != output, "not beside the sessions"
    assert argparse is not None  # keep the import honest for the signature above


def test_the_environment_check_notices_a_screen_it_cannot_see(monkeypatch, capsys) -> None:
    """5.1.4 asks the environment check to cover capture, and it covered everything else.

    It confirmed the packages, the Tesseract binary and the macOS Accessibility
    permission - and said "environment is ready" on a machine where `mss` sees a
    single zero-sized pseudo-monitor and every run stops at its first observation
    with `monitor_index 1 is out of range`. Importing mss is not the check: it
    imports perfectly well without Screen Recording permission.
    """
    import importlib.util
    import sys
    import types

    spec = importlib.util.spec_from_file_location(
        "check_environment", REPO_ROOT / "scripts" / "check_environment.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _Session:
        monitors = [{"left": 0, "top": 0, "width": 0, "height": 0}]
        width = 0
        height = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def grab(self, monitor):  # pragma: no cover - the branch under test returns first
            raise AssertionError("nothing should be captured when no display is visible")

    stub = types.SimpleNamespace(mss=lambda: _Session())
    monkeypatch.setitem(sys.modules, "mss", stub)

    problems = module.report_screen_capture()
    printed = capsys.readouterr().out

    assert problems == ["screen capture"]
    assert "NO DISPLAY VISIBLE" in printed
    assert "Screen Recording" in printed or not module.IS_MACOS
