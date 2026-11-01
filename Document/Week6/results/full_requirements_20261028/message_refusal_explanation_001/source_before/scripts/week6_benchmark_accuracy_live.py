"""Evaluate a manually frozen real saved screenshot; no GUI/model calls.

The ROI's labels and buttons are manually annotated before this probe. External
desktop UI is unannotated: report ROI precision only, never full-frame precision.
This is perception evidence, not attribution or success evidence for the run
that supplied the saved screenshot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import week6_benchmark_accuracy as benchmark
from PIL import Image

ROOT = benchmark.ROOT
DIRECTORY = ROOT / "Document/Week6/results/perception_accuracy_live_20261024"


def inside_roi(element, roi: list[int]) -> bool:
    left, top, right, bottom = benchmark.box(element)
    return roi[0] <= (left + right) / 2 < roi[2] and roi[1] <= (top + bottom) / 2 < roi[3]


def timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"min_ms": min(values), "median_ms": statistics.median(values),
            "max_ms": max(values), "p95_nearest_rank_ms": ordered[math.ceil(.95 * len(values)) - 1]}


def run(batch: str, tessdata_dir: Path, scratch_dir: Path) -> None:
    gt_path = DIRECTORY / "ground_truth_frozen_001.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    freeze_path = benchmark.DATA / "algorithm_freeze_001.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source = ROOT / gt["source_path"]
    assert benchmark.sha(source) == gt["source_sha256"]
    assert all(benchmark.sha(ROOT / p) == value for p, value in freeze["source_sha256"].items())
    for name in ("eng.traineddata", "chi_sim.traineddata"):
        assert benchmark.sha(tessdata_dir / name) == benchmark.sha(benchmark.RESOURCES / name)
    output = DIRECTORY / batch
    output.mkdir(exist_ok=False)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch_dir)
    previous_tessdata = os.environ.get("TESSDATA_PREFIX")
    os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
    manifest = {"started_utc": datetime.now(UTC).isoformat(), "ground_truth_sha256": benchmark.sha(gt_path),
                "algorithm_freeze_sha256": benchmark.sha(freeze_path),
                "probe_source_sha256": benchmark.sha(Path(__file__)),
                "image_sha256": benchmark.sha(source), "tessdata_dir": str(tessdata_dir),
                "repeats": 3, "retuning": False,
                "source_run_attribution": "Source runtime batch had unexplained external/manual input events and is invalid as agent execution-success evidence. This probe evaluates only its saved screenshot pixels."}
    benchmark.save(output / "run_manifest.json", manifest)
    baseline = benchmark.load_baseline()
    engines = {
        "native_eng": baseline(default_languages=["en"]),
        "native_bilingual": baseline(default_languages=["en", "zh"]),
        "screen_bilingual": benchmark.TesseractOCREngine(
            default_languages=["en", "zh"], profile="screen", scale=2.0, tessdata_dir=tessdata_dir),
    }
    rows = []
    try:
        for repeat in range(3):
            arms = list(engines) if repeat % 2 == 0 else list(reversed(engines))
            for arm in arms:
                started = time.perf_counter()
                image = Image.open(source).convert("RGB")
                decoded = time.perf_counter()
                result = engines[arm].recognize(image)
                ocr_done = time.perf_counter()
                candidates = benchmark.detect_ui_candidates(
                    image, exclude=[element.bounding_box for element in result.elements])
                finished = time.perf_counter()
                roi_ocr = [element for element in result.elements if inside_roi(element, gt["roi"])]
                roi_controls = [element for element in candidates if inside_roi(element, gt["roi"])]
                row = {"arm": arm, "repeat": repeat,
                       "decode_ms": (decoded - started) * 1000,
                       "ocr_ms": (ocr_done - decoded) * 1000,
                       "ui_ms": (finished - ocr_done) * 1000,
                       "full_ms": (finished - started) * 1000,
                       "roi_text_metrics": benchmark.measure(gt["labels"], roi_ocr),
                       "roi_control_metrics": benchmark.control_metrics(gt["controls"], roi_controls),
                       "full_frame_precision": None,
                       "outside_roi_ocr_count": len(result.elements) - len(roi_ocr),
                       "roi_predictions": [element.model_dump(mode="json") for element in roi_ocr],
                       "full_frame_predictions": [element.model_dump(mode="json") for element in result.elements],
                       "full_frame_candidates": [element.model_dump(mode="json") for element in candidates],
                       "errors": result.errors}
                rows.append(row)
                benchmark.save(output / f"{repeat}_{arm}.json", row)
    except Exception:
        benchmark.save(output / "failure.json", {"traceback": traceback.format_exc(),
                                                  "completed_records": len(rows)})
        raise
    finally:
        if previous_tessdata is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = previous_tessdata
    summary = {"schema": "week6.real_screenshot.ocr_accuracy.v1", "finished_utc": datetime.now(UTC).isoformat(),
               "independent_frames": 1, "independent_roi_labels": len(gt["labels"]), "repeats_per_arm": 3,
               "arms": {}, "source_unchanged": all(
                   benchmark.sha(ROOT / path) == value for path, value in freeze["source_sha256"].items()),
               "ground_truth_unchanged": benchmark.sha(gt_path) == manifest["ground_truth_sha256"],
               "probe_source_unchanged": benchmark.sha(Path(__file__)) == manifest["probe_source_sha256"],
               "image_unchanged": benchmark.sha(source) == manifest["image_sha256"],
               "gpu_frameworks_imported": [name for name in ("torch", "paddle") if sys.modules.get(name) is not None],
               "no_desktop_or_model_calls": True, "full_frame_precision": None,
               "latency_scope": "Full 2560x1600 saved PNG decode + CPU OCR + UI contours; ROI is only used for metrics, not cropped OCR input.",
               "limitations": ["One easy English-only local lab screenshot, not independent desktop-distribution accuracy",
                               "No accuracy improvement when both methods already recognize the five labels",
                               "External UI outside ROI is not annotated and is excluded from precision",
                               manifest["source_run_attribution"]]}
    for arm in engines:
        selected = [row for row in rows if row["arm"] == arm]
        summary["arms"][arm] = {
            "unique_frame_first_repetition_text": selected[0]["roi_text_metrics"],
            "unique_frame_first_repetition_controls": selected[0]["roi_control_metrics"],
            "all_three_repetitions_text_counts": {key: sum(row["roi_text_metrics"][key] for row in selected)
                                                   for key in ("tp", "fp", "fn")},
            "all_three_repetitions_same_roi_predictions": len({json.dumps(row["roi_predictions"], sort_keys=True) for row in selected}) == 1,
            "stage_timing": {name: timing([row[name] for row in selected])
                             for name in ("decode_ms", "ocr_ms", "ui_ms", "full_ms")}}
    benchmark.save(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="external_001")
    parser.add_argument("--tessdata-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.batch.isascii() or not args.batch.replace("_", "").isalnum():
        parser.error("batch must be a simple ASCII name")
    run(args.batch, args.tessdata_dir, args.scratch_dir)


if __name__ == "__main__":
    main()
