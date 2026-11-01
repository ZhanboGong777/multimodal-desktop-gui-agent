"""Run a basic OpenCV image-processing smoke test."""

from __future__ import annotations

from pathlib import Path

import cv2

INPUT_PATH = Path("outputs/week1/screenshot_test.png")
GRAY_PATH = Path("outputs/week1/screenshot_gray.png")
BINARY_PATH = Path("outputs/week1/screenshot_binary.png")


def main() -> int:
    image = cv2.imread(str(INPUT_PATH))
    if image is None:
        print(f"Input image not found or unreadable: {INPUT_PATH.resolve()}")
        print("Run scripts/smoke_test_perception.py first.")
        return 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray_saved = cv2.imwrite(str(GRAY_PATH), gray)
    binary_saved = cv2.imwrite(str(BINARY_PATH), binary)
    if not gray_saved or not binary_saved:
        print("Failed to save one or more processed images.")
        return 1

    height, width = gray.shape
    print(f"Input image: {INPUT_PATH.resolve()}")
    print(f"Image size: {width} x {height}")
    print(f"Grayscale image: {GRAY_PATH.resolve()}")
    print(f"Binary image: {BINARY_PATH.resolve()}")
    print("OpenCV image-processing smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
