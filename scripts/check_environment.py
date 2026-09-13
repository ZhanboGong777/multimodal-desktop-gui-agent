"""Check the Week 1 development environment on macOS or Windows."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

IS_MACOS = sys.platform == "darwin"
REPO_ROOT = Path(__file__).resolve().parents[1]

MODULES = {
    "mss": "screen capture",
    "PIL": "image processing",
    "cv2": "OpenCV",
    "numpy": "numerical arrays",
    "pytesseract": "OCR Python wrapper",
    "pyautogui": "mouse and keyboard control",
    "pynput": "input event control",
    "yaml": "YAML configuration",
    "dotenv": "environment variables",
    "pydantic": "data validation",
    "httpx": "HTTP client",
    "openai": "OpenAI-compatible model client",
    "anthropic": "Anthropic model client",
    "pytest": "testing",
    "torch": "PyTorch",
}


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def check_torch(require_gpu: bool) -> list[str]:
    """Report PyTorch acceleration and run a real CUDA operation when requested."""
    if not installed("torch"):
        print("\nPyTorch: MISSING")
        return ["torch"]

    import torch

    problems: list[str] = []
    print(f"\nPyTorch version: {torch.__version__}")

    if IS_MACOS:
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()
        print(f"MPS built: {mps_built}")
        print(f"MPS available: {mps_available}")
        if not mps_available:
            print("[WARNING] MPS is unavailable; CPU execution remains usable.")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if require_gpu and not cuda_available:
        problems.append("CUDA is unavailable")
        return problems

    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        print(f"CUDA device: {properties.name}")
        print(f"CUDA memory: {properties.total_memory / 1024**3:.1f} GB")
        print(f"CUDA runtime: {torch.version.cuda}")

        if require_gpu:
            try:
                left = torch.randn((512, 512), device="cuda")
                right = torch.randn((512, 512), device="cuda")
                _ = left @ right
                torch.cuda.synchronize()
                print("CUDA operation: PASSED")
            except Exception as exc:  # noqa: BLE001 - report the runtime failure
                print(f"CUDA operation: FAILED ({type(exc).__name__}: {exc})")
                problems.append("CUDA operation failed")

    return problems


def report_macos_permission() -> None:
    """Report Accessibility permission without importing macOS modules elsewhere."""
    if not IS_MACOS:
        print("macOS Accessibility permission: N/A")
        return
    if not installed("ApplicationServices"):
        print("macOS Accessibility permission: UNKNOWN (pyobjc not installed)")
        return

    from ApplicationServices import AXIsProcessTrusted

    trusted = bool(AXIsProcessTrusted())
    print(f"macOS Accessibility permission: {trusted}")
    if not trusted:
        print("[WARNING] Enable Accessibility permission before running control tests.")


def report_nvidia_smi() -> None:
    """Print NVIDIA driver information when the command is available."""
    executable = shutil.which("nvidia-smi")
    print(f"nvidia-smi: {executable or 'MISSING'}")
    if executable is None:
        print("[WARNING] nvidia-smi was not found; PyTorch CUDA remains the decisive test.")
        return

    result = subprocess.run(
        [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout.strip() or result.stderr.strip()
    print(f"NVIDIA GPU: {output or 'no output'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require working CUDA and run a small GPU operation.",
    )
    args = parser.parse_args()

    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Interpreter: {sys.executable}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")

    print("\nPython packages:")
    problems: list[str] = []
    for module, purpose in MODULES.items():
        available = installed(module)
        print(f"  [{'OK' if available else 'MISSING'}] {module}: {purpose}")
        if not available:
            problems.append(module)

    tesseract = shutil.which("tesseract")
    print(f"\nTesseract CLI: {tesseract or 'MISSING'}")
    if tesseract is None:
        problems.append("Tesseract CLI")

    report_macos_permission()
    problems.extend(check_torch(require_gpu=args.gpu))

    if args.gpu:
        print()
        report_nvidia_smi()

    config_path = REPO_ROOT / "configs" / "default.yaml"
    print(f"\nConfig file: {config_path} (exists: {config_path.exists()})")
    if not config_path.exists():
        problems.append("configs/default.yaml")

    if problems:
        print(f"\nEnvironment check FAILED: {', '.join(problems)}")
        return 1

    print("\nWeek 1 environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
