"""The configuration surface: what it refuses, and what it does by default.

16.2's configuration row asks for unknown keys to be rejected, ranges to be
enforced, and the default mode to be the safe one. All three were true and none
of them were tested, which is the combination that lets a later edit remove a
guarantee without anything going red.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gui_agent.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_an_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    """A misspelled section must fail, not be ignored.

    A config file that is silently half-applied is worse than one that is refused:
    the run uses the defaults and the operator believes it used their settings.
    """
    path = _write(tmp_path, "executon:\n  max_actions: 3\n")

    with pytest.raises(ValidationError, match="executon"):
        load_config(path)


def test_an_unknown_key_inside_a_section_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "execution:\n  max_action: 3\n")

    with pytest.raises(ValidationError, match="max_action"):
        load_config(path)


def test_out_of_range_limits_are_rejected(tmp_path: Path) -> None:
    """Zero actions or a negative budget is not a limit, it is a mistake."""
    for body in (
        "execution:\n  max_actions: 0\n",
        "execution:\n  task_timeout_seconds: 0\n",
        "execution:\n  max_wait_seconds: 0\n",
        "execution:\n  max_elements: 0\n",
    ):
        with pytest.raises(ValidationError):
            load_config(_write(tmp_path, body))


def test_verification_can_be_switched_off_but_not_negated(tmp_path: Path) -> None:
    """`ge=0` on purpose: a deadline of zero means "look once", not "look forever"."""
    config = load_config(_write(tmp_path, "execution:\n  verification_timeout_seconds: 0\n"))

    assert config.execution.verification_timeout_seconds == 0


def test_the_defaults_are_the_safe_mode() -> None:
    """Mock provider, dry run, confirmation on.

    Every one of those has to be chosen explicitly to change, so a config file that
    forgets a key cannot end up driving a real desktop.
    """
    config = Config()

    assert config.model.provider == "mock"
    assert config.control.dry_run is True
    assert config.planning.allow_real_execution is False
    assert config.execution.require_success_rules is True


def test_the_shipped_week4_config_keeps_that_mode() -> None:
    """The file the CLI loads by default must not quietly enable execution."""
    config = load_config(REPO_ROOT / "configs" / "week4.yaml")

    assert config.model.provider == "mock"
    assert config.control.dry_run is True


def test_an_empty_file_is_the_defaults(tmp_path: Path) -> None:
    assert load_config(_write(tmp_path, "")).model.provider == "mock"
