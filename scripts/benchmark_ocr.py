"""Benchmark the OCR backends on one image.

Built so the same measurement can be repeated on macOS and on the Windows GPU
node, and so it is obvious whether PaddleOCR is really running on the GPU.

Examples:
    python scripts/benchmark_ocr.py
    python scripts/benchmark_ocr.py --image outputs/week2/<session>/before.png --repeat 3
    python scripts/benchmark_ocr.py --engines tesseract paddleocr --language ch_sim
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image

from gui_agent.config import load_config
from gui_agent.perception.ocr import OcrError, create_ocr_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--image",
        default=None,
        help="image to benchmark; defaults to the most recent outputs/week2/*/before.png",
    )
    parser.add_argument("--engines", nargs="+", default=["tesseract", "paddleocr"])
    parser.add_argument(
        "--language", action="append", default=None, help="repeatable language code"
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--repeat", type=int, default=2, help="timed runs per engine")
    return parser.parse_args()


def latest_screenshot() -> Path | None:
    root = REPO_ROOT / "outputs" / "week2"
    if not root.is_dir():
        return None
    candidates = sorted(
        root.glob("*/before.png"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


# Running the probe in a child process is not paranoia. Importing PaddlePaddle
# loads its CUDA and cuDNN libraries into the interpreter, and PaddleOCR pulls
# Torch in transitively (PaddleX -> ModelScope). On a Windows node with both
# GPU builds installed the two cuDNN copies collide and Torch then fails with
# "WinError 127 ... cudnn_cnn64_9.dll". Keeping the probe out of process means
# the benchmark can still import PaddleOCR afterwards.
PADDLE_PROBE = (
    "import paddle; "
    "print('device=%s cuda_build=%s version=%s' % ("
    "paddle.device.get_device(), "
    "paddle.device.is_compiled_with_cuda(), "
    "paddle.__version__))"
)


def describe_paddle_device() -> str:
    """Report the PaddlePaddle build without importing it into this process."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", PADDLE_PROBE],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - the probe must never break the run
        return f"probe failed ({type(exc).__name__}: {exc})"

    output = (completed.stdout or "").strip()
    if output:
        return output
    detail = (completed.stderr or "").strip().splitlines()
    return f"probe failed ({detail[-1] if detail else 'no output'})"


def benchmark_engine(
    name: str,
    image: Image.Image,
    config_path: str,
    languages: list[str] | None,
    min_confidence: float,
    repeat: int,
) -> dict[str, object] | None:
    """Return timing statistics for one backend, or None when it is unavailable."""
    config = load_config(config_path)
    config.perception.ocr.engine = name  # type: ignore[assignment]
    # Benchmark the real backend; a fallback would hide the result.
    config.perception.ocr.fallback_engine = "none"
    if languages:
        config.perception.ocr.languages = languages
    config.perception.ocr.min_confidence = min_confidence

    try:
        engine = create_ocr_engine(config.perception.ocr).engine
    except OcrError as exc:
        print(f"  {name:<10} unavailable: {exc}")
        return None

    times: list[float] = []
    regions = 0
    error: str | None = None
    for _ in range(max(1, repeat)):
        try:
            started = time.perf_counter()
            output = engine.recognize(
                image,
                languages=config.perception.ocr.languages,
                min_confidence=config.perception.ocr.min_confidence,
            )
            times.append((time.perf_counter() - started) * 1000.0)
            regions = len(output.elements)
        except OcrError as exc:
            error = str(exc)
            break

    if error is not None or not times:
        print(f"  {name:<10} failed: {error or 'no result'}")
        return None

    return {
        "engine": name,
        "regions": regions,
        "runs": times,
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
    }


def main() -> int:
    args = parse_args()
    image_path = Path(args.image) if args.image else latest_screenshot()
    if image_path is None or not image_path.exists():
        print("No benchmark image found.")
        print("Run the perception demo first, or pass --image <path>.")
        return 2

    image = Image.open(image_path).convert("RGB")
    print(f"image      : {image_path}")
    print(f"size       : {image.width}x{image.height}")
    print(f"paddle     : {describe_paddle_device()}")
    print()

    results = []
    for name in args.engines:
        result = benchmark_engine(
            name, image, args.config, args.language, args.min_confidence, args.repeat
        )
        if result is not None:
            results.append(result)

    if not results:
        print("\nNo engine produced a result.")
        return 1

    print()
    print(f"  {'engine':<12}{'regions':>8}{'runs (ms)':>28}{'mean ms':>12}")
    print(f"  {'-' * 60}")
    for result in results:
        runs = ", ".join(f"{value:.0f}" for value in result["runs"])  # type: ignore[union-attr]
        print(
            f"  {result['engine']:<12}{result['regions']:>8}{runs:>28}{result['mean_ms']:>12.0f}"  # type: ignore[arg-type]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
