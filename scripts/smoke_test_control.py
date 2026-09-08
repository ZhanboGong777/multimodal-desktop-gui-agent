"""Safely check macOS mouse and keyboard automation support."""

from __future__ import annotations

import argparse
import time

import pyautogui
from ApplicationServices import AXIsProcessTrusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run a small reversible mouse movement and press/release Shift.",
    )
    return parser.parse_args()


def nearby_point(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    offset = 40
    target_x = x + offset if x + offset < width else x - offset
    target_y = y + offset if y + offset < height else y - offset
    return max(0, target_x), max(0, target_y)


def main() -> int:
    args = parse_args()
    pyautogui.FAILSAFE = True

    screen_width, screen_height = pyautogui.size()
    original_x, original_y = pyautogui.position()
    trusted = bool(AXIsProcessTrusted())

    print(f"Screen size: {screen_width} x {screen_height}")
    print(f"Current mouse position: ({original_x}, {original_y})")
    print(f"macOS Accessibility permission: {trusted}")

    if not args.execute:
        print("Dry run completed; no mouse or keyboard action was performed.")
        print("Run again with --execute to perform the reversible control test.")
        return 0

    if not trusted:
        print("Enable Accessibility permission for Visual Studio Code, then restart it.")
        return 1

    target_x, target_y = nearby_point(
        original_x,
        original_y,
        screen_width,
        screen_height,
    )

    print("Control test starts in 3 seconds. Move the pointer to a screen corner to abort.")
    time.sleep(3)

    try:
        pyautogui.moveTo(target_x, target_y, duration=0.3)
        pyautogui.moveTo(original_x, original_y, duration=0.3)
        pyautogui.keyDown("shift")
        pyautogui.keyUp("shift")
    finally:
        pyautogui.keyUp("shift")

    print("Mouse movement and keyboard event smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
