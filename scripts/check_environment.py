"""Report whether the Week 1 development environment is ready."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys

MODULES = {
    "mss": "screen capture",
    "PIL": "image processing",
    "cv2": "OpenCV",
    "pytesseract": "OCR Python wrapper",
    "pyautogui": "mouse and keyboard control",
    "pynput": "input event control",
    "yaml": "YAML configuration",
    "dotenv": "environment variables",
    "pytest": "testing",
    "torch": "PyTorch",
}


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")

    print("\nPython packages:")
    missing: list[str] = []
    for module, purpose in MODULES.items():
        available = installed(module)
        print(f"  [{'OK' if available else 'MISSING'}] {module}: {purpose}")
        if not available:
            missing.append(module)

    tesseract = shutil.which("tesseract")
    print(f"\nTesseract CLI: {tesseract or 'MISSING'}")

    if installed("torch"):
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        print(f"MPS available: {torch.backends.mps.is_available()}")

    if missing or tesseract is None:
        print("\nEnvironment check is incomplete. Install the missing items and run again.")
        return 1

    print("\nWeek 1 base environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
