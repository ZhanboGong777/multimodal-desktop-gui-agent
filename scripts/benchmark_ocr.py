"""Benchmark the OCR backends on one image.

Repeated runs per engine, reported as mean, standard deviation, P95, min and max
over the warm runs. The first call in a process pays for model loading and warm-up
caches, so it is timed and shown separately as the cold start.

Examples:
    python scripts/benchmark_ocr.py
    python scripts/benchmark_ocr.py --repeat 20
    python scripts/benchmark_ocr.py --engines tesseract --image outputs/week1/screenshot_test.png
"""

from __future__ import annotations

import argparse
import math
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

# Running the probe in a child process is not paranoia. Importing PaddlePaddle
# loads its CUDA and cuDNN libraries into the interpreter, and PaddleOCR pulls
# Torch in transitively (PaddleX -> ModelScope). On a Windows node with both GPU
# builds installed the two cuDNN copies collide and Torch then fails with
# "WinError 127 ... cudnn_cnn64_9.dll". Keeping the probe out of process means the
# benchmark can still import PaddleOCR afterwards.
PADDLE_PROBE = (
    "import paddle; "
    "print('device=%s cuda_build=%s version=%s' % ("
    "paddle.device.get_device(), "
    "paddle.device.is_compiled_with_cuda(), "
    "paddle.__version__))"
)


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
    parser.add_argument(
        "--repeat", type=int, default=10, help="timed runs per engine; the first is the cold start"
    )
    return parser.parse_args()


def latest_screenshot() -> Path | None:
    root = REPO_ROOT / "outputs" / "week2"
    if not root.is_dir():
        return None
    candidates = sorted(
        root.glob("*/before.png"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


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


def describe_timings(times: list[float]) -> dict[str, float]:
    """Mean, standard deviation, P95, min and max over the warm runs.

    P95 uses the nearest-rank method, so with 10 samples it equals the maximum;
    use more repetitions when the tail matters.
    """
    ordered = sorted(times)
    count = len(ordered)
    return {
        "count": float(count),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if count > 1 else 0.0,
        "p95": ordered[max(0, math.ceil(0.95 * count) - 1)],
    }


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
        selection = create_ocr_engine(config.perception.ocr)
    except OcrError as exc:
        print(f"  {name:<10} unavailable: {exc}")
        return None
    for notice in selection.notices:
        print(f"  {name:<10} notice: {notice}")
    engine = selection.engine

    runs = max(1, repeat)
    cold_ms: float | None = None
    warm: list[float] = []
    regions = 0
    error: str | None = None

    # The first call in a process pays for model loading and warm-up caches, so it
    # is timed and reported separately instead of polluting the steady-state set.
    for index in range(runs):
        try:
            started = time.perf_counter()
            output = engine.recognize(
                image,
                languages=config.perception.ocr.languages,
                min_confidence=config.perception.ocr.min_confidence,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            if index == 0:
                cold_ms = elapsed
            else:
                warm.append(elapsed)
            regions = len(output.elements)
        except OcrError as exc:
            error = str(exc)
            break

    if error is not None or not warm:
        print(f"  {name:<10} failed: {error or 'no warm run completed'}")
        return None

    return {
        "engine": name,
        "regions": regions,
        "cold_ms": cold_ms,
        "warm": warm,
        **describe_timings(warm),
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
    print(f"repeat     : {args.repeat} per engine (1 cold start + {max(0, args.repeat - 1)} warm)")
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
    print("  Warm runs only; each engine's first call is listed separately as cold start.")
    print()
    header = (
        f"  {'engine':<12}{'regions':>8}{'n':>4}{'min':>9}{'mean':>9}"
        f"{'stdev':>8}{'p95':>9}{'max':>9}{'cold':>9}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for result in results:
        print(
            f"  {result['engine']:<12}{result['regions']:>8}{int(result['count']):>4}"
            f"{result['min']:>9.0f}{result['mean']:>9.0f}{result['stdev']:>8.0f}"
            f"{result['p95']:>9.0f}{result['max']:>9.0f}{(result['cold_ms'] or 0.0):>9.0f}"
        )
    print()
    print("  all figures in milliseconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
