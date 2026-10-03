"""Which desktop applications are actually running.

Why this exists: the preconditions of T01 and T05 are about application state - "no
browser window is open", "the success rule does not already hold" - and the only check
the runner had was text on screen. Text cannot answer either question. A browser that is
open but behind another window, or minimised, contributes no text, so:

* T01 was told "no browser window is open" while one was running, planned against the
  existing window, and was then failed for not opening anything; and
* T05's marker is read off a screen where the window may simply not be visible, so
  "the marker is gone" and "the window is behind something" are indistinguishable.

Both were reported as model failures on the Windows node. This module is the other half
of the check: ask the operating system which processes exist, and treat "window" as
"process of that name is running", which is what the prose preconditions mean.

No third-party dependency: `tasklist` on Windows and `pgrep` elsewhere are present on a
machine that can run a GUI at all, and `tasklist` is called with the OEM code page
because that is what it writes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable

#: Process names, per platform, that count as "a browser is running".
BROWSER_PROCESSES: dict[str, tuple[str, ...]] = {
    "win32": ("msedge.exe", "chrome.exe", "firefox.exe", "iexplore.exe", "brave.exe",
              "opera.exe", "vivaldi.exe", "360se.exe", "360chrome.exe", "sogouexplorer.exe"),
    "darwin": ("Google Chrome", "Safari", "firefox", "Microsoft Edge", "Brave Browser",
               "Opera", "Vivaldi"),
}

#: Process names, per platform, that count as "a text editor is running".
EDITOR_PROCESSES: dict[str, tuple[str, ...]] = {
    "win32": ("notepad.exe", "notepad++.exe", "Code.exe", "sublime_text.exe", "gedit.exe"),
    "darwin": ("TextEdit", "Notepad++", "Code", "Sublime Text"),
}

_PLATFORM = "darwin" if sys.platform == "darwin" else "win32"


def known(kind: str, platform: str | None = None) -> tuple[str, ...]:
    """The process names a task can ask about by short name."""
    table = {"browser": BROWSER_PROCESSES, "editor": EDITOR_PROCESSES}
    try:
        per_platform = table[kind]
    except KeyError:
        raise KeyError(f"unknown process group {kind!r}; known: {sorted(table)}") from None
    return per_platform[platform or _PLATFORM]


def running_processes() -> set[str]:
    """Every running process name, lower-cased, or an empty set if it cannot be read.

    An empty set is deliberately indistinguishable from "nothing matched" at the call
    site only when the query itself is empty; callers pass names, so an unreadable list
    makes a `forbids_processes` check pass and a `requires_processes` check fail. That
    asymmetry is the safe direction: we never refuse a run because listing failed, and
    we never credit one either.
    """
    if _PLATFORM == "win32":
        return _running_windows()
    return _running_posix()


def _running_windows() -> set[str]:
    try:
        completed = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            capture_output=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    text = completed.stdout.decode("oem", errors="replace")
    names: set[str] = set()
    for line in text.splitlines():
        match = re.match(r'\s*"([^"]+)"', line)
        if match:
            names.add(match.group(1).strip().casefold())
    return names


def _running_posix() -> set[str]:
    try:
        completed = subprocess.run(
            ["ps", "-axco", "comm"], capture_output=True, check=False, timeout=20, text=True
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip().casefold() for line in completed.stdout.splitlines() if line.strip()}


def match(names: Iterable[str], running: set[str] | None = None) -> list[str]:
    """Which of ``names`` are running.

    Matching is by lower-cased full name, and a name without a suffix also matches the
    suffixed one - `chrome` matches `chrome.exe` - because a task definition should not
    have to know the platform's suffix.
    """
    present = running_processes() if running is None else running
    found: list[str] = []
    for name in names:
        wanted = name.casefold()
        if wanted in present:
            found.append(name)
            continue
        if not wanted.endswith((".exe", ".app")) and f"{wanted}.exe" in present:
            found.append(name)
    return found
