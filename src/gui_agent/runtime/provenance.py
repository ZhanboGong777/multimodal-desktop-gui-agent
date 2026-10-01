"""Who ran this, on what, and from which revision.

14.2 asks the run summary to carry the commit, the platform and the Python
version, and the basic task report's environment table asks the operator for the
same four facts. Both were being filled in by hand from memory, which is how a
report ends up quoting a revision the run did not use.

Everything here is best effort and returns a string: a run must not fail because
`git` is not on PATH.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .schemas import ObservationSnapshot

#: Set by the caller when the checkout has no git metadata - a copy unpacked onto
#: the Windows node, for instance. A provenance field that silently reads "" is
#: worse than one the operator can fill in.
COMMIT_ENV = "GUI_AGENT_COMMIT"

_REPO_ROOT = Path(__file__).resolve().parents[3]


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def platform_name() -> str:
    """'darwin' or 'win32' - the same vocabulary the action adapter uses."""
    return "darwin" if sys.platform == "darwin" else "win32"


def os_description() -> str:
    return f"{platform.system()} {platform.release()}"


def git_commit(repo_root: Path | None = None) -> str:
    """Short commit of the checkout, or an empty string when it cannot be read."""
    override = os.environ.get(COMMIT_ENV, "").strip()
    if override:
        return override
    if shutil.which("git") is None:
        return ""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root or _REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def screen_description(snapshot: ObservationSnapshot | None) -> str:
    """'screenshot WxH, control WxH' - the geometry a coordinate was mapped through.

    Recorded because a click that landed wrong is unreadable afterwards without it:
    the same screenshot pixel means different control points under different scales.
    """
    if snapshot is None:
        return ""
    info = snapshot.screen_info
    return (
        f"screenshot {info.screenshot_width}x{info.screenshot_height}, "
        f"control {info.control_width}x{info.control_height}"
    )
