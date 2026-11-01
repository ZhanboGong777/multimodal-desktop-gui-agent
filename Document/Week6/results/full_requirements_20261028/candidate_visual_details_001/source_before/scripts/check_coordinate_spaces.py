"""Report whether screenshot and control coordinates share the same space.

Run this on every platform before trusting any coordinate. On a Retina Mac, or on
Windows with display scaling, the two spaces differ: the screenshot is larger than
the control space, and OCR coordinates must be mapped through ``ScreenInfo``
before they reach PyAutoGUI.

Example:
    python scripts/check_coordinate_spaces.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.coordinates import screenshot_to_control
from gui_agent.perception.capture import capture_monitor
from gui_agent.schemas import Point

TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--monitor", type=int, default=None, help="monitor index (default from config)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    index = args.monitor if args.monitor is not None else config.perception.monitor_index

    frame = capture_monitor(index)
    info = frame.screen_info

    probe = Point(x=info.screenshot_width // 2, y=info.screenshot_height // 2)
    mapped = screenshot_to_control(probe, info)

    print(f"platform   : {sys.platform}")
    print(f"monitor    : {index}")
    print(f"screenshot : {info.screenshot_width} x {info.screenshot_height}")
    print(f"control    : {info.control_width} x {info.control_height}")
    print(f"scale      : {info.scale_x:.4f} x {info.scale_y:.4f}")
    print(f"offset     : ({info.monitor_left}, {info.monitor_top})")
    print(f"probe      : screenshot ({probe.x},{probe.y}) -> control ({mapped.x},{mapped.y})")
    print()

    identical = (
        abs(info.scale_x - 1.0) < TOLERANCE
        and abs(info.scale_y - 1.0) < TOLERANCE
        and info.monitor_left == 0
        and info.monitor_top == 0
    )
    if identical:
        print("RESULT: the two spaces are identical on this machine.")
        print("        Mapping still runs, but it is a no-op here, so a mapping bug")
        print("        would stay hidden. Check a scaled display too if you can.")
        return 0

    print("RESULT: coordinate mapping IS required on this machine.")
    print("        Screenshot pixels must be scaled and offset before PyAutoGUI")
    print("        sees them. Never pass OCR coordinates straight to the executor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
