"""Frozen bilingual UI OCR comparison. Saved fixtures only; no desktop input.

Prepare freezes both development and holdout pixels/annotations and copies the
baseline implementation before optimization. Run never changes those inputs.
Matching requires exact whitespace-normalized text AND IoU >= .5, one-to-one.
All rendered text is annotated, so unmatched predictions count as false positives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytesseract
from PIL import Image, ImageDraw, ImageFont

from gui_agent.perception.ocr import TesseractOCREngine
from gui_agent.perception.ui_detection import detect_ui_candidates

DATA = ROOT / "Document/Week6/results/perception_accuracy_20261024"
RESOURCES = ROOT / "Document/Week6/resources/tesseract"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=False)
    fonts = {"en": Path("C:/Windows/Fonts/arial.ttf"),
             "zh": Path("C:/Windows/Fonts/msyh.ttc")}
    records = []
    # Six layouts per split; held-out labels, placements and sizes differ. Freeze
    # these before running either OCR implementation, including failed examples.
    sets = {
        "development": [
            ["Account Settings", "User Name", "Open Project", "Apply", "Cancel", "用户设置", "保存修改", "退出"],
            ["Search History", "Folder Name", "Refresh", "Export", "Close", "搜索记录", "新建任务", "确认"],
            ["Workspace", "Status Ready", "Run Analysis", "Reset", "Back", "工作空间", "任务列表", "完成"],
            ["View Options", "File Name", "Import Data", "Preview", "Next", "查看选项", "导入数据", "返回"],
            ["System Status", "Queue Size", "Retry Task", "Pause", "Resume", "系统状态", "重新尝试", "暂停"],
            ["Draft Editor", "Local File", "Save Draft", "Undo", "Finish", "草稿编辑", "保存草稿", "撤销"],
        ],
        "holdout": [
            ["Network Tools", "Server Name", "Test Connection", "Start", "Stop", "网络工具", "连接测试", "停止"],
            ["Report Viewer", "Page Number", "Print Preview", "Zoom", "Return", "报告查看", "打印预览", "缩放"],
            ["Package Manager", "Version List", "Check Updates", "Install", "Remove", "软件管理", "检查更新", "安装"],
            ["Image Gallery", "Image Size", "Rotate Image", "Select", "Clear", "图片浏览", "旋转图片", "选择"],
            ["Calendar Event", "Event Title", "Add Reminder", "Today", "Delete", "日历事件", "添加提醒", "删除"],
            ["Resource Monitor", "Memory Usage", "Show Details", "Record", "Reload", "资源监测", "显示详情", "记录"],
        ],
    }
    for split, layouts in sets.items():
        for index, texts in enumerate(layouts):
            width, height = (820, 520) if split == "development" else (910, 570)
            dark = index % 2 == 1
            bg, fg = ((29, 32, 38), (210, 214, 222)) if dark else ((248, 249, 251), (48, 52, 60))
            if index in (2, 5):
                fg = (116, 122, 130) if not dark else (133, 139, 149)
            image = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(image)
            labels, controls = [], []
            size = [13, 15, 17, 14, 16, 12][index]
            if split == "holdout":
                size += 1
            positions = [(35, 33), (35, 103), (35, 180), (35, 281), (230, 281),
                         (width - 270, 103), (width - 270, 180), (width - 270, 281)]
            if split == "holdout":
                positions = [(x + 13, y + 19 + (index % 3) * 8) for x, y in positions]
            for number, (text, (x, y)) in enumerate(zip(texts, positions)):
                language = "en" if number < 5 else "zh"
                font = ImageFont.truetype(str(fonts[language]), size + (3 if number == 0 else 0))
                draw.text((x, y), text, font=font, fill=fg)
                box = list(draw.textbbox((x, y), text, font=font))
                labels.append({"text": text, "language": language, "box": box})
                if number >= 2:
                    control = [x - 10, y - 10, x + max(box[2] - x + 22, 126), y + 40]
                    draw.rectangle(control, outline=(90, 96, 104), width=1)
                    controls.append(control)
            name = f"{split}_{index + 1:02d}"
            path = DATA / f"{name}.png"
            image.save(path)
            records.append({"id": name, "split": split, "path": path.name, "size": [width, height],
                            "sha256": sha(path), "labels": labels, "controls": controls,
                            "font_size": size, "dark": dark, "low_contrast": index in (2, 5)})
    baseline = DATA / "baseline_ocr.py"
    baseline.write_bytes((ROOT / "src/gui_agent/perception/ocr.py").read_bytes())
    save(DATA / "frozen_manifest.json", {
        "schema": "week6.ocr_accuracy.frozen.v1", "created_utc": datetime.now(UTC).isoformat(),
        "matching": "exact clean_text + one-to-one IoU >= 0.5; every visible label annotated",
        "splits": "Development may guide design; holdout is evaluated once after code/profile frozen.",
        "desktop_capture": False, "baseline_source_sha256": sha(baseline),
        "baseline_git_ref": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "fonts": [{"path": str(path), "sha256": sha(path)} for path in fonts.values()],
        "fixtures": records,
    })
    print(DATA / "frozen_manifest.json")


def load_baseline() -> Any:
    path = DATA / "baseline_ocr.py"
    spec = importlib.util.spec_from_file_location("gui_agent.perception._accuracy_baseline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TesseractOCREngine


def iou(first: list[int], second: list[int]) -> float:
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return overlap / (a + b - overlap) if a + b > overlap else 0.0


def box(element: Any) -> list[int]:
    b = element.bounding_box
    return [b.left, b.top, b.right, b.bottom]


def measure(labels: list[dict[str, Any]], elements: list[Any]) -> dict[str, Any]:
    used: set[int] = set()
    matches = []
    for label in labels:
        candidates = [(iou(label["box"], box(item)), index) for index, item in enumerate(elements)
                      if index not in used and " ".join(item.text.split()) == label["text"]]
        overlap, chosen = max(candidates, default=(0.0, -1))
        passed = overlap >= .5
        if passed:
            used.add(chosen)
        matches.append({"text": label["text"], "language": label["language"],
                        "matched": passed, "iou": overlap})
    tp, fp, fn = len(used), len(elements) - len(used), len(labels) - len(used)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0, "labels": matches}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(row["metrics"][key] for row in rows) for key in ("tp", "fp", "fn")}
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {**counts, "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "ocr_median_ms": statistics.median(row["ocr_ms"] for row in rows),
            "ui_median_ms": statistics.median(row["ui_ms"] for row in rows),
            "decode_ocr_ui_median_ms": statistics.median(row["full_ms"] for row in rows),
            "by_language": {language: {
                "matched": sum(label["matched"] for row in rows for label in row["metrics"]["labels"]
                               if label["language"] == language),
                "total": sum(1 for row in rows for label in row["metrics"]["labels"]
                             if label["language"] == language),
            } for language in ("en", "zh")}}


def control_metrics(controls: list[list[int]], candidates: list[Any]) -> dict[str, Any]:
    """All closed rectangle controls in these generated scenes are annotated."""
    edges = sorted((iou(control, box(item)), gt, pred)
                   for gt, control in enumerate(controls) for pred, item in enumerate(candidates))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for overlap, gt, pred in reversed(edges):
        if overlap >= .8 and gt not in used_gt and pred not in used_pred:
            used_gt.add(gt)
            used_pred.add(pred)
    tp, fp, fn = len(used_gt), len(candidates) - len(used_pred), len(controls) - len(used_gt)
    return {"tp": tp, "fp": fp, "fn": fn, "iou_threshold": .8}


def run(split: str, batch: str, repeats: int, resource_dir: Path, scratch_dir: Path) -> None:
    manifest = json.loads((DATA / "frozen_manifest.json").read_text(encoding="utf-8"))
    assert sha(DATA / "baseline_ocr.py") == manifest["baseline_source_sha256"]
    output = DATA / batch
    output.mkdir(exist_ok=False)
    for name in ("eng.traineddata", "chi_sim.traineddata"):
        assert sha(resource_dir / name) == sha(RESOURCES / name)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(scratch_dir)
    sources = [ROOT / "src/gui_agent/perception/ocr.py", ROOT / "src/gui_agent/config.py",
               ROOT / "src/gui_agent/perception/ui_detection.py", Path(__file__)]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    baseline = load_baseline()
    # The first arm reproduces the installed English-only baseline. The second
    # isolates language availability. Only the third contains screen processing.
    engines = {
        "native_eng": baseline(default_languages=["en"]),
        "native_bilingual": baseline(default_languages=["en", "zh"]),
        "screen_bilingual": TesseractOCREngine(
            default_languages=["en", "zh"], profile="screen", scale=2.0,
            tessdata_dir=resource_dir),
    }
    original_tessdata = os.environ.get("TESSDATA_PREFIX")
    os.environ["TESSDATA_PREFIX"] = str(resource_dir)
    results: list[dict[str, Any]] = []
    try:
        for fixture in manifest["fixtures"]:
            if fixture["split"] != split:
                continue
            path = DATA / fixture["path"]
            assert sha(path) == fixture["sha256"]
            for repeat in range(repeats):
                arms = list(engines)
                if repeat % 2:
                    arms.reverse()
                for arm in arms:
                    started = time.perf_counter()
                    image = Image.open(path).convert("RGB")
                    decoded = time.perf_counter()
                    recognized = engines[arm].recognize(image)
                    ocr_done = time.perf_counter()
                    candidates = detect_ui_candidates(
                        image, exclude=[item.bounding_box for item in recognized.elements])
                    ended = time.perf_counter()
                    metrics = measure(fixture["labels"], recognized.elements)
                    result = {"fixture": fixture["id"], "arm": arm, "repeat": repeat,
                              "decode_ms": (decoded - started) * 1000,
                              "ocr_ms": (ocr_done - decoded) * 1000,
                              "ui_ms": (ended - ocr_done) * 1000,
                              "full_ms": (ended - started) * 1000,
                              "metrics": metrics,
                              "control_metrics": control_metrics(fixture["controls"], candidates),
                              "predictions": [item.model_dump(mode="json") for item in recognized.elements],
                              "ui_candidates": [item.model_dump(mode="json") for item in candidates],
                              "errors": recognized.errors}
                    results.append(result)
                    save(output / f"{fixture['id']}_{repeat}_{arm}.json", result)
    except Exception:
        save(output / "failure.json", {"created_utc": datetime.now(UTC).isoformat(),
                                       "traceback": traceback.format_exc(),
                                       "completed_records": len(results),
                                       "interpretation": "benchmark execution failed; not an accuracy score"})
        raise
    finally:
        if original_tessdata is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = original_tessdata
    summary = {"schema": "week6.ocr_accuracy.result.v1", "split": split,
               "created_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
               "tesseract": str(pytesseract.get_tesseract_version()),
               "repeats": repeats, "fixture_manifest_sha256": sha(DATA / "frozen_manifest.json"),
               "source_sha256": source_hashes,
               "source_unchanged": source_hashes == {str(path.relative_to(ROOT)): sha(path) for path in sources},
               "resource_provenance_sha256": sha(RESOURCES / "provenance.json"),
               "tessdata_directory": str(resource_dir), "scratch_directory": str(scratch_dir),
               "arms": {arm: aggregate([row for row in results if row["arm"] == arm]) for arm in engines},
               "unique_frame_first_repetition": {
                   arm: aggregate([row for row in results if row["arm"] == arm and row["repeat"] == 0])
                   for arm in engines},
               "contour_controls_all_repetitions": {arm: {
                   key: sum(row["control_metrics"][key] for row in results if row["arm"] == arm)
                   for key in ("tp", "fp", "fn")} for arm in engines},
               "runs": len(results), "desktop_capture": False, "gpu_model_inference": False,
               "gpu_frameworks_imported": [name for name in ("torch", "paddle")
                                           if sys.modules.get(name) is not None],
               "latency_scope": "PIL PNG decode + actual CPU Tesseract + contour extraction; no OS capture/LLM/action",
               "all_repetitions_included": True,
               "limitations": ["Synthetic local fixtures, not desktop task success rate",
                               "GT covers all visible text and all drawn rectangular controls in these fixtures only",
                               "Repeated frames are correlated; counts are frame-repetition results, not independent labels"]}
    save(output / "summary.json", summary)
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--batch", default="development_001")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tessdata-dir", type=Path, default=RESOURCES)
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / ".tmp-week6-ocr")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        if not 1 <= args.repeats <= 20 or not args.batch.isascii() or not args.batch.replace("_", "").isalnum():
            parser.error("bounded repeats and a simple ASCII batch name are required")
        run(args.split, args.batch, args.repeats, args.tessdata_dir, args.scratch_dir)


if __name__ == "__main__":
    main()
