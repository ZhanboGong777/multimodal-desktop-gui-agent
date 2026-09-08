"""Capture the desktop and run a basic OCR smoke test."""

from __future__ import annotations

from pathlib import Path

import pytesseract
from mss import MSS
from PIL import Image

OUTPUT_PATH = Path("outputs/week1/screenshot_test.png")


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MSS() as capture:
        primary_monitor = capture.monitors[1]
        screenshot = capture.grab(primary_monitor)
        image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        image.save(OUTPUT_PATH)

    text = pytesseract.image_to_string(image, lang="eng").strip()
    preview = " ".join(text.split())[:200]

    print(f"Screenshot saved to: {OUTPUT_PATH.resolve()}")
    print(f"OCR preview: {preview or '[no English text detected]'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
