"""Fresh, frozen CPU full-frame OCR comparison; never captures the desktop."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Document/Week6/results/full_requirements_20261028/perception_fixture_001"
RESOURCES = ROOT / "Document/Week6/resources/tesseract"
RESOURCE_DEST = Path("D:/week6-full-ocr/tessdata")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"Preserve existing evidence: {path}")
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def prepare() -> None:
    """Freeze new development/holdout images and an unchanged subprocess baseline."""
    DATA.mkdir(parents=True, exist_ok=False)
    fonts = {"en": Path("C:/Windows/Fonts/arial.ttf"), "zh": Path("C:/Windows/Fonts/msyh.ttc")}
    groups = {
        "development": [
            ["Access Panel", "Session Code", "Change Folder", "Accept", "Dismiss", "设备连接", "保存配置", "重启"],
            ["Text Console", "Line Count", "Load Script", "Execute", "Abort", "文本控制", "加载脚本", "终止"],
            ["Storage View", "Disk Capacity", "Scan Storage", "Browse", "Eject", "存储空间", "扫描磁盘", "弹出"],
            ["Color Studio", "Brush Width", "Pick Color", "Fill", "Erase", "颜色面板", "选取颜色", "擦除"],
        ],
        "holdout": [
            ["Audio Settings", "Input Level", "Restore Audio", "Enable", "Disable", "声音设置", "调整音量", "静音"],
            ["Display Panel", "Screen Mode", "Adjust Display", "Extend", "Mirror", "显示面板", "调整亮度", "投影"],
            ["Archive Browser", "Entry Count", "Extract Archive", "Compress", "Verify", "压缩文件", "解压内容", "校验"],
            ["Device Center", "Battery Level", "Pair Device", "Connect", "Disconnect", "设备中心", "配对设备", "断开"],
            ["Backup Wizard", "Target Drive", "Schedule Backup", "Create", "Restore", "备份向导", "恢复文件", "计划"],
            ["Font Inspector", "Style Sample", "Manage Fonts", "Regular", "Italic", "字体检查", "管理字体", "粗体"],
            ["Map Explorer", "Route Length", "Locate Address", "Navigate", "Pin", "地图浏览", "定位地址", "路线"],
            ["Security Center", "Access Policy", "Review Access", "Permit", "Revoke", "安全中心", "访问策略", "授权"],
        ],
    }
    fixtures = []
    for split, layouts in groups.items():
        for index, labels_text in enumerate(layouts):
            width, height = (840, 540) if split == "development" else (960 + index * 12, 600 + index * 6)
            size = ([13, 15, 17, 14] if split == "development" else [14, 16, 18, 13, 15, 17, 19, 14])[index]
            dark, low = index % 2 == 1, index % 4 == 3
            background = (27, 31, 37) if dark else (247, 249, 250)
            foreground = ((134, 142, 151) if low else (213, 219, 225)) if dark else (45, 51, 58)
            image = Image.new("RGB", (width, height), background)
            draw = ImageDraw.Draw(image)
            positions = [(42, 34), (42, 112), (42, 198), (42, 302), (256, 302),
                         (width - 292, 112), (width - 292, 198), (width - 292, 302)]
            labels, controls = [], []
            for number, (text, (x, y)) in enumerate(zip(labels_text, positions)):
                language = "en" if number < 5 else "zh"
                font = ImageFont.truetype(str(fonts[language]), size + (3 if number == 0 else 0))
                draw.text((x, y), text, fill=foreground, font=font)
                box = list(draw.textbbox((x, y), text, font=font))
                labels.append({"text": text, "language": language, "box": box})
                if number >= 2:
                    rect = [x - 11, y - 10, x + max(box[2] - x + 20, 140), y + 43]
                    draw.rectangle(rect, outline=(88, 96, 105), width=1)
                    controls.append(rect)
            name = f"{split}_{index + 1:02d}"
            path = DATA / f"{name}.png"
            image.save(path)
            fixtures.append({"id": name, "split": split, "path": path.name, "sha256": sha(path),
                             "size": [width, height], "font_size": size, "dark": dark,
                             "low_contrast": low, "labels": labels, "controls": controls})
    sources = {}
    for name in ["ocr.py", "ui_detection.py"]:
        source = ROOT / "src/gui_agent/perception" / name
        dest = DATA / f"baseline_{name}"
        shutil.copyfile(source, dest)
        sources[name] = {"path": dest.name, "sha256": sha(dest), "original": str(source.relative_to(ROOT))}
    RESOURCE_DEST.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ["eng.traineddata", "chi_sim.traineddata"]:
        source, dest = RESOURCES / name, RESOURCE_DEST / name
        if dest.exists() and sha(source) != sha(dest):
            raise RuntimeError("Do not overwrite differing existing language resources")
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert sha(source) == sha(dest)
        copied.append({"source": str(source), "destination": str(dest), "sha256": sha(dest), "bytes": dest.stat().st_size})
    save(DATA / "frozen_manifest.json", {
        "schema": "week6.full_perception.frozen.v1", "created_utc": datetime.now(UTC).isoformat(),
        "fixtures": fixtures, "baseline_sources": sources,
        "fonts": [{"path": str(p), "sha256": sha(p)} for p in fonts.values()], "resource_copy": copied,
        "evaluation_rule": "Exact normalized text and one-to-one IoU>=0.5; unmatched outputs count FP. All drawn rectangles evaluated at IoU>=0.8.",
        "development_rule": "Only development frames may be evaluated before source/parameters freeze; new holdout once afterward. Old holdout and results unchanged.",
    })
    save(DATA / "preregistered_plan.json", {
        "schema": "week6.full_perception.plan.v1", "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "Retain screen recognition while reducing full PNG-decode + OCR + contour wall time; no OS capture, model or action claims.",
        "arms": ["native_eng_subprocess", "native_bilingual_subprocess", "screen_bilingual_subprocess", "screen_bilingual_persistent"],
        "fixed_parameters": {"screen_scale": 2.0, "screen_psm": 11, "screen_oem": 1, "confidence": .5},
        "equivalence_gate": "Persistent screen ordered UIElement text, bounds, confidence and all UI candidates equal screen subprocess for every measured frame/repetition; original pixels unchanged.",
        "accuracy_gate": "On new holdout, persistent P/R/F1 equal screen subprocess and F1 greater than native bilingual. Score unique frames once, not correlated repeats as independent labels.",
        "speed_gate": "All warm repetitions included. Full-frame median at least 10% lower than each of the other three arms; per-frame persistent median not over5% slower than screen. p95/max/slow frames retained.",
        "cold_rule": "Three fresh-engine first-frame measurements per arm reported separately, including creation/loading; warmup first call per arm retained separately and never described as warm speed.",
        "warm_rule": "3 repeats of every frame; rotate four arm order by frame+repeat, reverse odd repeats; all timing/errors/warnings preserved.",
        "api_sources": ["https://raw.githubusercontent.com/tesseract-ocr/tesseract/5.5.1/include/tesseract/capi.h", "https://raw.githubusercontent.com/tesseract-ocr/tesseract/5.5.1/include/tesseract/baseapi.h"],
        "no_gui_model_training": True, "no_environment_install": True,
    })
    print(DATA / "preregistered_plan.json")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pins() -> dict[str, str]:
    files = [*sorted((ROOT / "src/gui_agent/perception").glob("*.py")),
             ROOT / "src/gui_agent/config.py", Path(__file__),
             ROOT / "scripts/week6_benchmark_accuracy.py",
             ROOT / "configs/week6_accuracy_persistent.yaml",
             ROOT / "tests/test_week6_persistent_ocr.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in files if p.is_file()}


def runtime_pins() -> dict[str, str]:
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("Installed Tesseract executable was not found")
    files = [RESOURCE_DEST / name for name in ("eng.traineddata", "chi_sim.traineddata")]
    files += [Path(executable), Path(executable).parent / "libtesseract-5.dll"]
    return {str(path): sha(path) for path in files}


def run(split: str, batch: str, repeats: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    import pytesseract

    from gui_agent.perception.ocr import TesseractOCREngine
    from gui_agent.perception.ui_detection import detect_ui_candidates

    manifest = json.loads((DATA / "frozen_manifest.json").read_text(encoding="utf-8"))
    before = pins()
    runtime_before = runtime_pins()
    base = DATA.parent
    output = base / batch
    output.mkdir(exist_ok=False)
    if split == "holdout":
        frozen = json.loads((base / "perception_source_freeze_001.json").read_text(encoding="utf-8"))
        assert frozen["source_pins"] == before
        assert frozen["runtime_pins"] == runtime_before
        assert repeats == 10 and batch in {"perception_holdout_primary_001", "perception_holdout_replication_001"}
        save(output / "evaluation_claim.json", {"created_utc": datetime.now(UTC).isoformat(),
                                                "source_freeze_sha256": sha(base / "perception_source_freeze_001.json")})
    baseline_path = DATA / manifest["baseline_sources"]["ocr.py"]["path"]
    assert sha(baseline_path) == manifest["baseline_sources"]["ocr.py"]["sha256"]
    ui_baseline = manifest["baseline_sources"]["ui_detection.py"]
    assert sha(DATA / ui_baseline["path"]) == ui_baseline["sha256"]
    assert sha(ROOT / ui_baseline["original"]) == ui_baseline["sha256"]
    for resource in manifest["resource_copy"]:
        assert sha(Path(resource["source"])) == resource["sha256"]
        assert sha(Path(resource["destination"])) == resource["sha256"]
    baseline = load_module("gui_agent.perception._full_baseline", baseline_path).TesseractOCREngine
    metrics = load_module("_full_accuracy_metrics", ROOT / "scripts/week6_benchmark_accuracy.py")
    arms = ["native_eng_subprocess", "native_bilingual_subprocess",
            "screen_bilingual_subprocess", "screen_bilingual_persistent"]

    def factory(arm):
        settings = {"default_languages": ["en"] if arm == arms[0] else ["en", "zh"],
                    "tessdata_dir": RESOURCE_DEST}
        if arm.startswith("screen"):
            settings.update(profile="screen", scale=2.0)
        if arm == arms[3]:
            return TesseractOCREngine(**settings, transport="persistent")
        return baseline(**settings)

    fixtures = [f for f in manifest["fixtures"] if f["split"] == split]
    records = []

    def measure(fixture, arm, phase, repetition, engine=None):
        path = DATA / fixture["path"]
        assert sha(path) == fixture["sha256"]
        started = time.perf_counter()
        owned = engine is None
        result = {"fixture": fixture["id"], "arm": arm, "phase": phase, "repeat": repetition,
                  "image_sha256": fixture["sha256"], "started_perf_counter": started}
        try:
            if owned:
                engine = factory(arm)
            created = time.perf_counter()
            with Image.open(path) as source:
                image = source.convert("RGB")
            decoded = time.perf_counter()
            original_pixels = image.tobytes()
            recognized = engine.recognize(image)
            ocr_done = time.perf_counter()
            candidates = detect_ui_candidates(image, exclude=[e.bounding_box for e in recognized.elements])
            ended = time.perf_counter()
            assert original_pixels == image.tobytes()
            result.update(status="completed", ended_perf_counter=ended,
                          engine_creation_ms=(created-started)*1000,
                          decode_ms=(decoded-created)*1000, ocr_ms=(ocr_done-decoded)*1000,
                          ui_ms=(ended-ocr_done)*1000, full_ms=(ended-started)*1000,
                          metrics=metrics.measure(fixture["labels"], recognized.elements),
                          control_metrics=metrics.control_metrics(fixture["controls"], candidates),
                          predictions=[e.model_dump(mode="json") for e in recognized.elements],
                          ui_candidates=[e.model_dump(mode="json") for e in candidates], errors=recognized.errors)
        except Exception:  # noqa: BLE001 -- preserve a failed measurement and continue all arms.
            failed_at = time.perf_counter()
            result.update(status="measurement_error", traceback=traceback.format_exc(),
                          ended_perf_counter=failed_at, observed_elapsed_ms=(failed_at-started)*1000)
        finally:
            if owned and engine is not None and hasattr(engine, "close"):
                engine.close()
        save(output / f"{phase}_{fixture['id']}_{repetition:02d}_{arm}.json", result)
        records.append(result)
        return result

    for rep in range(3):
        order = arms[rep:] + arms[:rep]
        for arm in order:
            measure(fixtures[0], arm, "cold", rep)
    engines = {arm: factory(arm) for arm in arms}
    try:
        for frame_index, fixture in enumerate(fixtures):
            for arm in arms:
                measure(fixture, arm, "warmup", 0, engines[arm])
            for rep in range(repeats):
                rotation = (frame_index + rep) % len(arms)
                order = arms[rotation:] + arms[:rotation]
                if rep % 2:
                    order.reverse()
                for arm in order:
                    measure(fixture, arm, "warm", rep, engines[arm])
            print(f"{batch}: {fixture['id']} all {repeats} paired repetitions saved", flush=True)
    finally:
        for engine in engines.values():
            if hasattr(engine, "close"):
                engine.close()
    successes = [r for r in records if r["status"] == "completed"]
    warm = [r for r in successes if r["phase"] == "warm"]

    def timing(rows):
        values = sorted(r["full_ms"] for r in rows)
        return {"n": len(values), "median_ms": statistics.median(values),
                "p95_nearest_rank_ms": values[math.ceil(.95 * len(values))-1],
                "max_ms": max(values), "all_records_included": True} if values else None

    exact = []
    for row in successes:
        if row["arm"] != arms[3]:
            continue
        matches = [r for r in successes if r["arm"] == arms[2]
                   and all(r[key] == row[key] for key in ("fixture", "phase", "repeat"))]
        exact.append({"fixture": row["fixture"], "phase": row["phase"], "repeat": row["repeat"],
                      "passed": len(matches) == 1 and all(matches[0][key] == row[key]
                                                         for key in ("predictions", "ui_candidates", "metrics", "control_metrics"))})
    totals = {arm: metrics.aggregate([r for r in warm if r["arm"] == arm and r["repeat"] == 0])
              for arm in arms if any(r["arm"] == arm and r["repeat"] == 0 for r in warm)}
    timings = {phase: {arm: timing([r for r in successes if r["phase"] == phase and r["arm"] == arm])
                       for arm in arms} for phase in ("cold", "warmup", "warm")}
    gains = {arm: 1 - timings["warm"][arms[3]]["median_ms"] / timings["warm"][arm]["median_ms"]
             for arm in arms[:3]} if len(totals) == 4 else {}
    per_frame = {f["id"]: {arm: timing([r for r in warm if r["fixture"] == f["id"] and r["arm"] == arm])
                           for arm in arms} for f in fixtures}
    paired = []
    for row in warm:
        if row["arm"] != arms[3]:
            continue
        for arm in arms[:3]:
            matches = [r for r in warm if r["arm"] == arm and r["fixture"] == row["fixture"]
                       and r["repeat"] == row["repeat"]]
            if len(matches) == 1:
                baseline_ms = matches[0]["full_ms"]
                paired.append({"fixture": row["fixture"], "repeat": row["repeat"], "baseline": arm,
                               "baseline_full_ms": baseline_ms, "candidate_full_ms": row["full_ms"],
                               "candidate_minus_baseline_ms": row["full_ms"] - baseline_ms,
                               "relative_reduction": 1 - row["full_ms"] / baseline_ms})
    controls = {arm: {key: sum(r["control_metrics"][key] for r in warm
                               if r["arm"] == arm and r["repeat"] == 0)
                       for key in ("tp", "fp", "fn")} for arm in arms}
    errors = [r for r in records if r["status"] != "completed" or r.get("errors")]
    checks = {"expected_record_count": len(records) == 12 + len(fixtures) * (repeats + 1) * 4,
              "no_measurement_errors": not errors,
              "ordered_screen_predictions_and_ui_exact": bool(exact) and all(r["passed"] for r in exact),
              "accuracy_preserved_and_above_both_native": len(totals) == 4 and totals[arms[3]]["f1"] == totals[arms[2]]["f1"] and all(totals[arms[3]]["f1"] > totals[arm]["f1"] for arm in arms[:2]),
              "rectangle_metrics_exact_to_screen": controls[arms[3]] == controls[arms[2]],
              "warm_full_frame_median_10pct_faster_than_all_three": len(gains) == 3 and all(v >= .10 for v in gains.values()),
              "no_frame_median_over_5pct_slower_than_screen": all(v[arms[3]] and v[arms[2]] and v[arms[3]]["median_ms"] <= v[arms[2]]["median_ms"] * 1.05 for v in per_frame.values()),
              "sources_unchanged": before == pins(),
              "runtime_libraries_resources_unchanged": runtime_before == runtime_pins(),
              "no_gpu_frameworks_imported": not any(sys.modules.get(n) is not None for n in ("torch", "paddle"))}
    save(output / "summary.json", {"schema": "week6.full_perception.result.v1", "split": split,
        "created_utc": datetime.now(UTC).isoformat(), "source_pins_before": before, "source_pins_after": pins(),
        "runtime_pins_before": runtime_before, "runtime_pins_after": runtime_pins(),
        "fixture_manifest_sha256": sha(DATA / "frozen_manifest.json"), "repeats": repeats,
        "checks": checks, "all_criteria_passed": all(checks.values()), "timings": timings,
        "full_frame_median_reduction_vs": gains, "unique_frame_accuracy": totals,
        "per_frame_timings": per_frame, "paired_warm_differences": paired,
        "unique_frame_rectangle_metrics": controls, "equivalence_checks": exact, "all_errors": errors,
        "records": len(records), "python": platform.python_version(), "executable": sys.executable,
        "tesseract_version": str(pytesseract.get_tesseract_version()), "library_sha256": sha(Path("C:/Program Files/Tesseract-OCR/libtesseract-5.dll")),
        "omp_environment": {k: os.environ.get(k) for k in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS")},
        "latency_scope": "Complete PNG decode + input pixel-preservation snapshot + real CPU OCR (including same screen processing) + contour extraction. Cold also includes engine creation/lazy initialization; teardown excluded. No OS capture/model/input.",
        "sampling_limits": "Repeated frames correlated; unique-frame accuracy scored once. No dropped slow samples; stdout/stderr preserves C API diagnostics. Cold means a fresh engine, not a cold OS page cache; application CLI/YAML startup is outside this benchmark."})
    print(json.dumps({"checks": checks, "gains": gains, "timings": timings}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--batch", default="perception_development_matrix_001")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.freeze:
        save(DATA.parent / "perception_source_freeze_001.json", {
            "created_utc": datetime.now(UTC).isoformat(), "source_pins": pins(),
            "runtime_pins": runtime_pins(),
            "plan_v3_sha256": sha(DATA / "preregistered_plan_v3.json"),
            "meaning": "No code/parameters changes allowed during two predetermined new holdout batches."})
    elif args.run:
        if not 1 <= args.repeats <= 10 or not args.batch.isascii() or not args.batch.replace("_", "").isalnum():
            parser.error("bounded repeats and a simple ASCII batch name required")
        run(args.split, args.batch, args.repeats)
    else:
        parser.error("Select prepare, freeze or run")
