"""Benchmark continuous screen capture.

Repeated frames, reported as mean, standard deviation, P95, min and max over the
warm frames. The first capture in a process initialises the MSS session and is
noticeably slower, so it is timed and shown separately as the cold start.

Examples:
    python scripts/benchmark_capture.py
    python scripts/benchmark_capture.py --frames 30
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.perception.capture import (
    CaptureRegion,
    capture_monitor,
    capture_region,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--monitor", type=int, default=None, help="monitor index (default from config)"
    )
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument(
        "--frames", type=int, default=30, help="frames to capture; the first is the cold start"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    index = args.monitor if args.monitor is not None else config.perception.monitor_index
    region = CaptureRegion(*args.region) if args.region else None

    def grab():
        if region is not None:
            return capture_region(region, monitor_index=index)
        return capture_monitor(index)

    print(f"monitor    : {index}")
    print(f"frames     : {args.frames} (1 cold start + {max(0, args.frames - 1)} warm)")
    print()

    cold_ms: float | None = None
    warm: list[float] = []
    sequence_started = time.perf_counter()
    size = None
    for i in range(max(1, args.frames)):
        frame = grab()
        size = frame.image.size
        if i == 0:
            cold_ms = frame.capture_time_ms
        else:
            warm.append(frame.capture_time_ms)
    elapsed = time.perf_counter() - sequence_started

    ordered = sorted(warm)
    n = len(ordered)
    stats = {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if n > 1 else 0.0,
        "p95": ordered[max(0, math.ceil(0.95 * n) - 1)],
        "max": ordered[-1],
    }

    print(f"screen     : {size[0]}x{size[1]}")
    print()
    print("  Warm frames only; the first capture is listed separately as cold start.")
    print()
    header = f"  {'n':>4}{'min':>9}{'mean':>9}{'stdev':>8}{'p95':>9}{'max':>9}{'cold':>9}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    print(
        f"  {n:>4}{stats['min']:>9.1f}{stats['mean']:>9.1f}{stats['stdev']:>8.1f}"
        f"{stats['p95']:>9.1f}{stats['max']:>9.1f}{(cold_ms or 0.0):>9.1f}"
    )
    print()
    print(f"  capture_only_fps (warm mean): {1000.0 / stats['mean']:.1f}")
    print(
        f"  effective_fps over {args.frames} frames: {args.frames / elapsed:.2f}"
        f"  ({elapsed:.2f} s, no interval)"
    )
    print()
    print("  all figures in milliseconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
