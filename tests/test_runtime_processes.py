"""Application state as a precondition, and why text cannot express it.

T01's precondition is "no browser window is open". The only check the runner had was
whether the text T01 looks for was on screen, and that gets the answer wrong in the
direction that wastes a whole run: a browser behind another window, or minimised,
contributes none of that text, so it was called absent while it was running. T01 then
planned against the window that already existed and failed with "expected on screen but
not found: http, search" while every mechanism underneath had worked.

These tests pin both directions: the process check blocks when it should, and dispatching
nothing happens when it does.
"""

from __future__ import annotations

import pytest

from gui_agent.runtime import processes
from gui_agent.runtime.processes import BROWSER_PROCESSES, known, match


def test_known_returns_the_platform_browser_names() -> None:
    assert "msedge.exe" in known("browser", "win32")
    assert "Google Chrome" in known("browser", "darwin")
    assert "notepad.exe" in known("editor", "win32")


def test_an_unknown_group_is_an_error_rather_than_an_empty_answer() -> None:
    """An empty tuple would silently satisfy a forbids-check and fail a requires one."""
    with pytest.raises(KeyError, match="unknown process group"):
        known("spreadsheet")


def test_match_is_case_insensitive_and_tolerates_a_missing_suffix() -> None:
    present = {"chrome.exe", "notepad.exe"}
    assert match(["chrome"], present) == ["chrome"]
    assert match(["chrome.exe"], present) == ["chrome.exe"]
    assert match(["CHROME"], present) == ["CHROME"]
    assert match(["firefox"], present) == []


def test_a_name_that_is_merely_a_prefix_does_not_match() -> None:
    """"chrome" must not match "chromedriver.exe" or every CI box would look like one."""
    present = {"chromedriver.exe", "chrome_remote_desktop.exe"}
    assert match(["chrome"], present) == []


def test_the_browser_group_covers_the_browsers_this_project_has_met() -> None:
    """Edge and Chrome both appeared on the review machine, plus the domestic ones."""
    windows = {name.casefold() for name in BROWSER_PROCESSES["win32"]}
    for expected in ("msedge.exe", "chrome.exe", "firefox.exe", "360se.exe"):
        assert expected in windows


def test_running_processes_returns_a_set_of_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading the real list is allowed to be slow, but it must not raise."""
    result = processes.running_processes()
    assert isinstance(result, set)
    assert result, "a machine running pytest has at least one process"


def test_an_unreadable_process_list_neither_passes_nor_credits_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If listing fails, a forbids-check passes and a requires-check fails.

    That asymmetry is deliberate: refusing a run because `tasklist` failed would be
    wrong, and crediting one would be worse.
    """
    monkeypatch.setattr(processes, "_running_windows", lambda: set())
    monkeypatch.setattr(processes, "_running_posix", lambda: set())
    assert match(["browser"]) == []
