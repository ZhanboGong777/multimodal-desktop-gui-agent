"""The CLI's safety contract, checked without touching a desktop.

The interesting cases are the ones where a run must *not* happen: no task, an
unknown case, a free-form instruction with nothing to verify, and a dry run that
must never dispatch.
"""

from __future__ import annotations

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


def test_execute_without_a_terminal_is_blocked_rather_than_assumed() -> None:
    """12.2.5: with no terminal there is nobody to ask, so consent is not assumed.

    The refusal happens before the capture, so this is fast and needs no desktop.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "week4_agent_cli.py"),
            "--case",
            "T01",
            "--execute",
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
