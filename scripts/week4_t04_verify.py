"""Run T04 end to end, and refuse to start unless the desktop can actually be driven.

Why this exists. T04 is the one Week 4 case without a recorded verdict, and every attempt to get
one has failed on something outside the case: the desktop locked mid-run (`BitBlt` from Windows,
`FailSafeException` from pyautogui's corner guard), the client's window moving between the frame
that located a control and the click that used it, or a budget too small to reach the verifier.
Each of those is checkable in advance, and this script checks them.

What it does not do, deliberately:

* it does not disable pyautogui's fail-safe. That guard aborts when the pointer reaches a screen
  corner, and it is the last thing between a mis-computed coordinate and a runaway run;
* it does not skip the CLI's two confirmations. T04 sends a real message, and the project's rule is
  that a human agrees to that. The script answers the prompts the way the operator already agreed
  to - it does not remove them;
* it does not send anything to anyone but the conversation named for the case.

Usage, with the desktop unlocked and the conversation already open::

    python scripts/week4_t04_verify.py                # checks only
    python scripts/week4_t04_verify.py --run          # checks, then the run

Exit codes: 0 the checks passed (and the run succeeded, if asked for), 1 a check failed or the run
did not succeed, 2 the desktop is not in a state where a run is meaningful.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Anything matching this in the foreground window means the session cannot be driven.
_LOCK_MARKERS = ("锁屏", "Lock screen", "Lock Screen")

#: Planning on this node has been measured at 110-120 s per call, and the case allows four
#: attempts before it gives up. `configs/week4.yaml` carries 600 s, which reaches three attempts
#: and stops before the verification that would produce a verdict - measured by comparing two runs
#: that differ only in budget. So the run asks for more than the config default.
DEFAULT_TASK_TIMEOUT = 1800


def _fail(message: str, code: int = 2) -> int:
    print(f"  REFUSING : {message}")
    return code


def check_desktop() -> int:
    """Checks that the session can be driven at all, before anything is planned."""
    from gui_agent.perception.capture import capture_monitor, foreground_window_title

    try:
        title = foreground_window_title()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return _fail(f"the foreground window could not be read: {exc}")

    if not title:
        return _fail(
            "the foreground window is empty, which is what a disconnected or unusable session "
            "reports; a locked one names the lock screen instead"
        )
    if any(marker in title for marker in _LOCK_MARKERS):
        return _fail(f"the desktop is locked (foreground: {title!r})")

    try:
        frame = capture_monitor(
            monitor_index=1, output_directory=REPO / "outputs/week4", save=False
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return _fail(f"screen capture failed, so no frame could be observed: {exc}")

    print(f"  desktop  : drivable - foreground {title[:44]!r}, capture {frame.image.size}")

    try:
        import pyautogui

        point = pyautogui.position()
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return _fail(f"the pointer position could not be read: {exc}")

    if (point.x, point.y) == (0, 0):
        return _fail(
            "the pointer reads as (0, 0), which is also where pyautogui's fail-safe triggers; a "
            "real session does not park the cursor in the corner, so nothing would be dispatched"
        )
    print(f"  pointer  : {point.x},{point.y}")
    return 0


def check_client() -> int:
    """Checks the client the case needs, and reports where it is - the position moves."""
    import ctypes
    from ctypes import wintypes

    from gui_agent.runtime.tasks import MESSAGE_CONVERSATION

    user32 = ctypes.windll.user32
    user32.EnumWindows.argtypes = [ctypes.c_void_p, wintypes.LPARAM]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]

    found: list[tuple[int, int, int, int, int]] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _: int) -> bool:
        name = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, name, 256)
        if name.value == "Qt51514QWindowIcon" and user32.IsWindowVisible(hwnd):
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            found.append((hwnd, rect.left, rect.top, rect.right, rect.bottom))
        return True

    user32.EnumWindows(proc(visit), 0)
    if not found:
        return _fail(
            "the messaging client is not running. Its window class is what this checks; the case "
            "itself needs the conversation already open, which no program can check for you"
        )

    _hwnd, left, top, right, bottom = found[0]
    size = f"{right - left}x{bottom - top}"
    print(f"  client   : at ({left},{top})-({right},{bottom}) [{size}]")
    print(f"  expects  : the conversation {MESSAGE_CONVERSATION!r} open, with no earlier message")
    print("             carrying this run's marker - the CLI prints the marker it will use")
    print(
        "  reminder : do not open a screenshot while the run is going. A displayed frame sits"
        "\n             over the desktop and absorbs the clicks; measured, and it cost this case"
        "\n             several runs before it was noticed."
    )
    return 0


def run_case(task_timeout: int) -> int:
    """Runs the case through the CLI, which is where the confirmations live."""
    command = [
        str(REPO / ".venv" / "Scripts" / "python.exe"),
        str(REPO / "scripts" / "week4_agent_cli.py"),
        "--case",
        "T04",
        "--execute",
        "--task-timeout",
        str(task_timeout),
        "--ocr-engine",
        "paddleocr",
        "--ocr-min-confidence",
        "0.3",
    ]
    print(f"\n  running  : {' '.join(command[1:])}")
    print("             answer the two confirmations in this console; the case sends a real message\n")
    completed = subprocess.run(command, cwd=str(REPO), check=False)
    return completed.returncode


def report_latest() -> int:
    """Prints the newest T04 run's verdict, so the outcome does not depend on reading stdout."""
    runs = sorted(
        (REPO / "outputs" / "week4").glob("T04_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        print("  no T04 run recorded")
        return 1
    summary = runs[0] / "task_summary.json"
    if not summary.exists():
        print(f"  {runs[0].name}: no task_summary.json, so the run did not finish")
        return 1
    data = json.loads(summary.read_text(encoding="utf-8"))
    verdict = (data.get("verification") or {}).get("outcome")
    print(
        f"\n  {runs[0].name}: status={data.get('status')} actions={data.get('action_count')} "
        f"verification={verdict}"
    )
    for note in (data.get("notes") or [])[-4:]:
        print(f"    note: {note}")
    return 0 if data.get("status") == "succeeded" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the case after the checks; without it, only the checks are performed",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=DEFAULT_TASK_TIMEOUT,
        help=f"task budget in seconds (default {DEFAULT_TASK_TIMEOUT}; the config's 600 is too "
        "small to reach the verifier on this node)",
    )
    args = parser.parse_args()

    print("T04 preflight")
    for check in (check_desktop, check_client):
        code = check()
        if code:
            return code

    if not args.run:
        print("\n  checks passed. Re-run with --run to execute the case.")
        return 0

    if run_case(args.task_timeout):
        print("\n  the CLI exited non-zero; the run's own verdict follows")
    return report_latest()


if __name__ == "__main__":
    raise SystemExit(main())
