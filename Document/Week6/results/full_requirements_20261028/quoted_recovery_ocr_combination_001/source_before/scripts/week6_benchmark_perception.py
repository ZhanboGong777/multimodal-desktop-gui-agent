"""Repeatable CPU perception benchmark; never captures or controls the desktop.

The generated screens are fixtures, not evidence of desktop task success. OCR is
run for real. UI-only comparisons reuse each fixture's measured OCR boxes (not
another frame's targets) so OCR variance cannot hide a contour regression.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.util
import io
import json
import platform
import pstats
import statistics
import subprocess
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import PIL
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from gui_agent.config import Config
from gui_agent.perception import ui_detection
from gui_agent.perception.capture import CaptureResult
from gui_agent.perception.ocr import EngineSelection, create_ocr_engine
from gui_agent.runtime import observation
from gui_agent.schemas import ScreenInfo

BASELINE_REF = "9149e064d90ab6796e7f25ece6cbbd026785b26c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline(directory: Path, ref: str) -> tuple[types.ModuleType, types.ModuleType]:
    modules = []
    for relative, package in (
        ("src/gui_agent/perception/ui_detection.py", "gui_agent.perception"),
        ("src/gui_agent/runtime/observation.py", "gui_agent.runtime"),
    ):
        source = subprocess.check_output(["git", "show", f"{ref}:{relative}"], cwd=ROOT)
        destination = directory / ("baseline_" + Path(relative).name)
        destination.write_bytes(source)
        spec = importlib.util.spec_from_file_location(
            f"{package}._week6_baseline_{Path(relative).stem}", destination,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    modules[1].detect_ui_candidates = modules[0].detect_ui_candidates
    return modules[0], modules[1]


def fixtures(directory: Path) -> list[dict[str, Any]]:
    """Generate fixed labels, controls and dark thin-border adversarial cases."""
    directory.mkdir(parents=True, exist_ok=True)
    latin_path = Path("C:/Windows/Fonts/arial.ttf")
    chinese_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(latin_path), 26)
    chinese_font = ImageFont.truetype(str(chinese_path), 26)
    manifest: list[dict[str, Any]] = []
    for name, dark, dense in (
        ("light_form", False, False), ("dark_form", True, False),
        ("dense_dashboard", False, True), ("thin_border", True, True),
    ):
        background, foreground = ((29, 29, 30), "white") if dark else ("white", "black")
        image = Image.new("RGB", (960, 640), background)
        draw = ImageDraw.Draw(image)
        labels: list[dict[str, Any]] = []
        controls: list[list[int]] = []
        for text, x, y, language in (
            ("Account Settings", 35, 28, "en"), ("Name", 35, 100, "en"),
            ("Search", 35, 230, "en"), ("Save", 70, 362, "en"),
            ("Cancel", 295, 362, "en"), ("用户名", 610, 100, "zh"),
            ("设置", 610, 230, "zh"),
        ):
            selected = chinese_font if language == "zh" else font
            draw.text((x, y), text, fill=foreground, font=selected)
            labels.append({"text": text, "language": language,
                           "box": list(draw.textbbox((x, y), text, font=selected))})
        for box in ((35, 140, 505, 202), (35, 270, 505, 332),
                    (35, 350, 205, 410), (260, 350, 430, 410),
                    (590, 140, 905, 202), (590, 270, 905, 332)):
            color = (45, 45, 46) if name == "thin_border" else foreground
            draw.rectangle(box, outline=color, width=1 if name == "thin_border" else 2)
            controls.append(list(box))
        if dense:
            for row in range(6):
                for column in range(18):
                    x, y = 20 + column * 51, 438 + row * 30
                    draw.rectangle((x, y, x + 39, y + 22), outline=foreground)
                    draw.text((x + 3, y + 1), str((row + column) % 10),
                              font=ImageFont.truetype(str(latin_path), 16), fill=foreground)
        path = directory / f"{name}.png"
        image.save(path)
        manifest.append({"id": name, "path": str(path.relative_to(ROOT)),
                         "sha256": digest(path), "size": list(image.size),
                         "labels": labels, "controls": controls,
                         "selection_reason": "fixed bilingual form, density/theme stress; synthetic"})
    metadata = {"schema": "week6.perception.fixtures.v1", "fixtures": manifest,
                "fonts": [{"path": str(path), "sha256": digest(path)}
                          for path in (latin_path, chinese_path)],
                "desktop_capture": False, "public_screenshots_used": False}
    (directory.parent / "fixture_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"min_ms": min(values), "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values), "max_ms": max(values),
            "p95_ms": ordered[max(0, int(np.ceil(len(values) * .95)) - 1)]}


def signature(elements: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in elements]


def coverage(labels: list[dict[str, Any]], elements: list[Any]) -> dict[str, Any]:
    """Exact normalized label inside its own region; no semantic/fuzzy credit."""
    rows = []
    for label in labels:
        left, top, right, bottom = label["box"]
        observed = [item.text for item in elements
                    if left <= item.bounding_box.center.x <= right
                    and top <= item.bounding_box.center.y <= bottom and item.text]
        expected = "".join(label["text"].casefold().split())
        rows.append({**label, "observed": observed,
                     "matched": any("".join(text.casefold().split()) == expected
                                    for text in observed)})
    return {"labels": rows, "matched": sum(row["matched"] for row in rows),
            "total": len(rows)}


def run(args: argparse.Namespace) -> None:
    destination = (ROOT / args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = fixtures(destination / "fixtures")
    baseline_ui, baseline_observation = load_baseline(destination, args.baseline_ref)
    config = Config()
    config.perception.ocr.engine = "tesseract"
    config.perception.ocr.languages = ["en"]
    config.perception.ocr.min_confidence = 0.5
    # Fix OpenCV's worker count in this process for reproducibility, not production.
    cv2.setNumThreads(1)
    available_languages = pytesseract.get_languages(config="")
    observations = []
    ui_runs = []
    timings: dict[str, float] = {}
    module = baseline_observation if args.phase == "baseline" else observation
    current_fixture: dict[str, Any] = {}

    def offline_capture(*unused: Any, **unused_keywords: Any) -> CaptureResult:
        started = time.perf_counter()
        path = ROOT / current_fixture["path"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        result = CaptureResult(image=image, captured_at=datetime.now(UTC), image_path=path,
                               screen_info=ScreenInfo(screenshot_width=width,
                                                      screenshot_height=height,
                                                      control_width=width, control_height=height))
        timings["capture"] = (time.perf_counter() - started) * 1000
        return result

    def engine_factory(settings: Any) -> EngineSelection:
        started = time.perf_counter()
        selection = create_ocr_engine(settings)
        timings["engine_init"] = (time.perf_counter() - started) * 1000
        original = selection.engine.recognize

        def timed_recognize(*values: Any, **keywords: Any) -> Any:
            tick = time.perf_counter()
            try:
                return original(*values, **keywords)
            finally:
                timings["ocr"] = (time.perf_counter() - tick) * 1000
        selection.engine.recognize = timed_recognize
        return selection

    detector = baseline_ui.detect_ui_candidates if args.phase == "baseline" else (
        ui_detection.detect_ui_candidates)

    def timed_detect(*values: Any, **keywords: Any) -> Any:
        tick = time.perf_counter()
        try:
            return detector(*values, **keywords)
        finally:
            timings["ui_detection"] = (time.perf_counter() - tick) * 1000

    with patch.object(module, "capture_monitor", offline_capture), \
            patch.object(module, "create_ocr_engine", engine_factory), \
            patch.object(module, "detect_ui_candidates", timed_detect):
        service = module.ObservationService(config, save_images=False)
        select = service._select

        def timed_select(elements: list[Any]) -> Any:
            tick = time.perf_counter()
            try:
                return select(elements)
            finally:
                timings["selection"] = (time.perf_counter() - tick) * 1000
        service._select = timed_select
        for fixture in manifest:
            current_fixture = fixture
            for repeat in range(args.repetitions):
                timings.clear()
                snapshot = service.observe()
                record = {"fixture": fixture["id"], "repeat": repeat,
                          "cold_engine": len(observations) == 0,
                          "instrumented_stage_ms": {"engine_init": 0.0, **timings},
                          "snapshot": snapshot.model_dump(mode="json"),
                          "label_coverage": coverage(fixture["labels"], snapshot.elements)}
                observations.append(record)
            # Same-frame OCR exclusions, image loaded once outside UI-only timer.
            ocr_boxes = [item.bounding_box for item in snapshot.elements if item.source == "ocr"]
            with Image.open(ROOT / fixture["path"]) as source:
                image = source.convert("RGB")
            for repeat in range(args.ui_repetitions):
                # Alternate order to avoid always giving one version warm CPU caches.
                results = {}
                order = ["baseline", "optimized"] if repeat % 2 == 0 else ["optimized", "baseline"]
                for variant in order:
                    implementation = (baseline_ui if variant == "baseline" else ui_detection)
                    tick = time.perf_counter()
                    elements = implementation.detect_ui_candidates(image, exclude=ocr_boxes)
                    duration = (time.perf_counter() - tick) * 1000
                    results[variant] = {"elapsed_ms": duration, "elements": signature(elements)}
                ui_runs.append({"fixture": fixture["id"], "repeat": repeat, "order": order,
                                "exact_output_equal": results["baseline"]["elements"] ==
                                results["optimized"]["elements"], **results})
    profile = cProfile.Profile()
    with profile:
        for _ in range(3):
            baseline_ui.detect_ui_candidates(image, exclude=ocr_boxes)
    profile_text = io.StringIO()
    pstats.Stats(profile, stream=profile_text).sort_stats("cumtime").print_stats(30)
    (destination / f"{args.phase}_baseline_profile.txt").write_text(
        profile_text.getvalue(), encoding="utf-8")
    report = {"schema": "week6.perception.cpu_benchmark.v1", "phase": args.phase,
              "created_at_utc": datetime.now(UTC).isoformat(),
              "baseline_ref": args.baseline_ref,
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "opencv": cv2.__version__,
                              "numpy": np.__version__, "pillow": PIL.__version__,
                              "pytesseract": pytesseract.__version__,
                              "tesseract": str(pytesseract.get_tesseract_version()),
                              "tesseract_languages": available_languages,
                              "opencv_threads": cv2.getNumThreads()},
              "configuration": config.perception.model_dump(mode="json"),
              "limits": ["Synthetic fixture results, not real desktop success rates.",
                         "capture is PNG decode, not MSS/OS desktop capture latency.",
                         "Engine init times the factory only; lazy model init belongs to OCR.",
                         "Tesseract English only; Chinese ground truth is retained without credit.",
                         "CPU only. No planner, GUI action, GPU model, or external message.",
                         "UI-only paired timings exclude file load and use same-frame OCR boxes."],
              "observations": observations, "paired_ui_runs": ui_runs,
              "summaries": {"warm_observation": summary([
                  row["snapshot"]["processing_time_ms"] for row in observations
                  if not row["cold_engine"]]),
                  "ui_baseline": summary([row["baseline"]["elapsed_ms"] for row in ui_runs]),
                  "ui_optimized": summary([row["optimized"]["elapsed_ms"] for row in ui_runs]),
                  "all_ui_outputs_equal": all(row["exact_output_equal"] for row in ui_runs)},
              "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in (
                  ROOT / "src/gui_agent/runtime/observation.py",
                  ROOT / "src/gui_agent/perception/ui_detection.py",
                  ROOT / "src/gui_agent/perception/ocr.py", Path(__file__).resolve())}}
    path = destination / f"{args.phase}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "summaries": report["summaries"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--ui-repetitions", type=int, default=25)
    parser.add_argument("--output", default="Document/Week6/results/perception_benchmark")
    args = parser.parse_args()
    if args.repetitions < 1 or args.ui_repetitions < 1:
        parser.error("repetition counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
