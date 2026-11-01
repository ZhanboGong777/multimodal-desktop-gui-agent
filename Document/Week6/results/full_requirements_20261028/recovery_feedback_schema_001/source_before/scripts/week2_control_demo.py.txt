"""Validate and optionally perform one desktop action.

A dry run is the default: nothing is moved or typed unless ``--execute`` is
passed explicitly.

Examples:
    python scripts/week2_control_demo.py --action move --x 400 --y 300
    python scripts/week2_control_demo.py --action drag --start-x 100 --start-y 100 \\
        --end-x 400 --end-y 300 --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.control.executor import ActionExecutor
from gui_agent.control.safety import describe_action
from gui_agent.perception.capture import capture_monitor
from gui_agent.recording import RunSession, build_run_summary
from gui_agent.schemas import DesktopAction, Point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--action",
        required=True,
        choices=[
            "move",
            "click",
            "double_click",
            "right_click",
            "drag",
            "scroll",
            "type_text",
            "key_press",
            "hotkey",
            "wait",
        ],
    )
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--start-x", type=int, default=None)
    parser.add_argument("--start-y", type=int, default=None)
    parser.add_argument("--end-x", type=int, default=None)
    parser.add_argument("--end-y", type=int, default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--keys", nargs="+", default=None)
    parser.add_argument("--amount", type=int, default=None, help="scroll amount")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--execute", action="store_true", help="really send mouse/keyboard events")
    parser.add_argument("--session-id", default=None)
    return parser.parse_args()


def build_action(args: argparse.Namespace) -> DesktopAction:
    payload: dict[str, object] = {"action_type": args.action, "duration": args.duration}
    if args.x is not None and args.y is not None:
        payload["x"], payload["y"] = args.x, args.y
    if args.start_x is not None and args.start_y is not None:
        payload["start"] = Point(x=args.start_x, y=args.start_y)
    if args.end_x is not None and args.end_y is not None:
        payload["end"] = Point(x=args.end_x, y=args.end_y)
    if args.text is not None:
        payload["text"] = args.text
    if args.key is not None:
        payload["key"] = args.key
    if args.keys is not None:
        payload["keys"] = args.keys
    if args.amount is not None:
        payload["scroll_amount"] = args.amount
    return DesktopAction(**payload)  # type: ignore[arg-type]


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    started = time.perf_counter()

    try:
        action = build_action(args)
    except Exception as exc:  # noqa: BLE001 - any schema error must be reported to the user
        print(f"invalid action: {exc}")
        return 2

    frame = capture_monitor(config.perception.monitor_index)
    session = RunSession.create(config.perception.output_directory, args.session_id)
    lines: list[str] = []

    def note(message: str) -> None:
        print(message)
        lines.append(message)

    note(f"planned action: {describe_action(action)}")
    note(
        f"monitor: {frame.image.width}x{frame.image.height}, control space "
        f"{frame.screen_info.control_width}x{frame.screen_info.control_height}"
    )

    dry_run = not args.execute
    if args.execute:
        note(
            "--execute given: real control is enabled, move the pointer to a screen corner to abort"
        )

    def countdown(message: str) -> None:
        note(message)

    executor = ActionExecutor(
        config.control, screen=frame.screen_info, notice=note, countdown=countdown
    )
    result = executor.execute(action, dry_run=dry_run)

    note(f"success={result.success} dry_run={result.dry_run} error={result.error}")
    session.save_action(action, result)
    session.save_summary(
        build_run_summary(
            selected_target=action.target_description,
            action=action,
            result=result,
            capture_time_ms=frame.capture_time_ms,
            ocr_time_ms=0.0,
            total_time_ms=(time.perf_counter() - started) * 1000.0,
            error=result.error,
        )
    )
    session.save_log(lines)
    note(f"run record: {session.directory}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
