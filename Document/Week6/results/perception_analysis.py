"""Summarize the retained Week 6 CPU observations without re-running OCR."""

from __future__ import annotations

import cProfile
import hashlib
import importlib.util
import io
import json
import pstats
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import cv2
from PIL import Image

from gui_agent.perception.ui_detection import detect_ui_candidates
from gui_agent.schemas import BoundingBox


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(first: list[int], second: list[int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(right - left, 0) * max(bottom - top, 0)
    return intersection / float(
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1]) - intersection
    )


def main() -> None:
    cv2.setNumThreads(1)
    directory = ROOT / "Document/Week6/results/perception_benchmark"
    baseline = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
    optimized = json.loads((directory / "optimized.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "fixture_manifest.json").read_text(encoding="utf-8"))
    by_fixture = []
    for fixture in manifest["fixtures"]:
        rows = [row for row in optimized["paired_ui_runs"] if row["fixture"] == fixture["id"]]
        medians = {variant: statistics.median(row[variant]["elapsed_ms"] for row in rows)
                   for variant in ("baseline", "optimized")}
        boxes = [[item["bounding_box"][key] for key in ("left", "top", "right", "bottom")]
                 for item in rows[0]["optimized"]["elements"]]
        best_ious = [max((iou(box, found) for found in boxes), default=0)
                     for box in fixture["controls"]]
        observations = [row for row in optimized["observations"]
                        if row["fixture"] == fixture["id"]]
        by_fixture.append({
            "fixture": fixture["id"], "paired_repetitions": len(rows),
            "ui_median_ms": medians,
            "median_reduction_percent": 100 * (1 - medians["optimized"] / medians["baseline"]),
            "all_ordered_candidates_equal": all(row["exact_output_equal"] for row in rows),
            "candidate_count": len(boxes), "controls": fixture["controls"],
            "best_control_iou": best_ious, "control_iou_threshold": .8,
            "controls_covered": sum(value >= .8 for value in best_ious),
            "controls_total": len(best_ious),
            "label_coverage_first_repeat": observations[0]["label_coverage"],
            "warm_stage_median_ms": {
                stage: statistics.median(row["snapshot"]["stage_timings_ms"][stage]
                                         for row in observations if not row["cold_engine"])
                for stage in ("capture", "engine_init", "ocr", "ui_detection", "selection")
            },
        })
    raw_labels = [label for row in by_fixture
                  for label in row["label_coverage_first_repeat"]["labels"]]
    language_coverage = {
        language: {"matched": sum(label["matched"] for label in raw_labels
                                   if label["language"] == language),
                   "total": sum(label["language"] == language for label in raw_labels)}
        for language in ("en", "zh")
    }
    source = directory / "baseline_ui_detection.py"
    spec = importlib.util.spec_from_file_location("gui_agent.perception._week6_profile", source)
    assert spec is not None and spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    # Same image and its same-frame OCR boxes, exactly as in the paired benchmark.
    last = optimized["observations"][-1]
    exclusion = [BoundingBox.model_validate(item["bounding_box"])
                 for item in last["snapshot"]["elements"] if item["source"] == "ocr"]
    image = Image.open(ROOT / manifest["fixtures"][-1]["path"]).convert("RGB")
    for variant, function in (("baseline", legacy.detect_ui_candidates),
                              ("optimized", detect_ui_candidates)):
        profile = cProfile.Profile()
        with profile:
            for _ in range(3):
                function(image, exclude=exclusion)
        output = io.StringIO()
        pstats.Stats(profile, stream=output).sort_stats("cumtime").print_stats(30)
        (directory / f"analysis_{variant}_profile.txt").write_text(
            output.getvalue(), encoding="utf-8")
    old = optimized["summaries"]["ui_baseline"]["median_ms"]
    new = optimized["summaries"]["ui_optimized"]["median_ms"]
    old_warm = baseline["summaries"]["warm_observation"]["median_ms"]
    new_warm = optimized["summaries"]["warm_observation"]["median_ms"]
    report = {
        "schema": "week6.perception.analysis.v1", "fixtures": by_fixture,
        "unique_fixture_label_coverage": language_coverage,
        "controls_covered": sum(row["controls_covered"] for row in by_fixture),
        "controls_total": sum(row["controls_total"] for row in by_fixture),
        "paired_ui": {"pairs": len(optimized["paired_ui_runs"]),
                      "baseline_median_ms": old, "optimized_median_ms": new,
                      "median_reduction_percent": 100 * (1 - new / old),
                      "all_ordered_candidates_equal":
                          optimized["summaries"]["all_ui_outputs_equal"]},
        "serial_warm_observation": {"baseline_median_ms": old_warm,
                                    "optimized_median_ms": new_warm,
                                    "change_percent": 100 * (new_warm / old_warm - 1),
                                    "end_to_end_improvement_proven": False},
        "test_validation": {"command": "python -X utf8 -m pytest -q "
                            "tests/test_week6_perception.py tests/test_ui_detection.py "
                            "tests/test_runtime_observation.py "
                            "--basetemp=.pytest-tmp-week6-perception",
                            "passed": 95, "failed": 0, "elapsed_seconds": 2.75},
        "input_sha256": {path.name: sha256(path) for path in (
            directory / "baseline.json", directory / "optimized.json",
            directory / "fixture_manifest.json", directory / "baseline_ui_detection.py",
            directory / "baseline_observation.py")},
        "limitations": optimized["limits"] + [
            ("Control coverage only measures these 24 known synthetic rectangles; "
             "other candidates are not fully annotated, so precision is not claimed."),
            "Cold calls across serial runs are not a controlled initialization speed comparison.",
            "cProfile overhead is diagnostic; use unprofiled paired values for latency claims.",
        ],
    }
    (ROOT / "Document/Week6/results/perception_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "paired_ui", "serial_warm_observation", "unique_fixture_label_coverage",
        "controls_covered", "controls_total")}, indent=2))


if __name__ == "__main__":
    main()
