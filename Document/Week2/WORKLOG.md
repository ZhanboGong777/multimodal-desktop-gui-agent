# Week 2 Work Log

Week 2 topic: desktop perception and control core modules.
Branch: `week2-perception-control`.

## Summary

| Item | Result |
| --- | --- |
| New source modules | 13 |
| New demo scripts | 3 |
| New test files | 10 |
| Tests | 165 passed |
| Coverage | 88% (`pytest --cov=gui_agent`) |
| Ruff | clean |
| Week 1 regression | `check_environment.py` and all three smoke tests still exit 0 |

## Tasks

### 1. Shared data structures (`src/gui_agent/schemas.py`)

Defined `Point`, `BoundingBox`, `ScreenInfo`, `UIElement`, `PerceptionResult`,
`DesktopAction` and `ActionResult` as strict Pydantic models. Unknown fields are
rejected, geometry is validated, and `DesktopAction` enforces the payload each
action type needs - including `drag`, which requires **both** a start and an end
point instead of a single coordinate.

### 2. Coordinate mapping (`src/gui_agent/coordinates.py`)

Pure functions that scale screenshot pixels into control coordinates and add the
monitor offset, plus clamping and an inside-screen test. Kept in a dedicated
top-level module because both the perception and control layers need it and it
must stay free of side effects.

### 3. Continuous capture (`src/gui_agent/perception/capture.py`)

`capture_monitor`, `capture_fullscreen`, `capture_region` and `capture_frames`.
Every capture returns a `ScreenInfo`, per-frame latency, and an optional saved
path. `capture_frames` supports a frame count and an interval, and
`average_capture_ms` / `capture_only_fps` / `effective_sequence_fps` derive the timing the
hand-off asks for.

### 4. Image preprocessing (`src/gui_agent/perception/preprocessing.py`)

`to_grayscale`, `resize_image`, `gaussian_blur`, `otsu_binarize`,
`sharpen_image`, `crop_region` and `apply_preprocessing`. RGB is the module
contract; no helper writes to disk, mutates its input, or copies an image when
no work is required.

### 5. OCR backends (`src/gui_agent/perception/ocr.py`)

A single `OCREngine` interface with `PaddleOCREngine` (primary) and
`TesseractOCREngine` (fallback). Both return `UIElement` objects with text,
confidence and a bounding box. `create_ocr_engine` degrades to Tesseract and
explains why when PaddleOCR is missing. Tesseract is queried through
`image_to_data`, not `image_to_string`, so word positions are preserved.

**Bug found and fixed during the smoke test:** the configuration uses backend
agnostic language codes (`en`), but Tesseract expects `eng`. The first real run
crashed with `Failed loading language 'en'`. `to_tesseract_languages()` now
translates the codes, and `tests/test_ocr.py` covers the mapping.

**Third OCR bug, found on the Windows GPU node:** importing PaddlePaddle before
PaddleOCR makes Torch (pulled in via PaddleX -> ModelScope) fail to load its cuDNN
DLLs with `WinError 127`. The probe now runs in a child process. See the experiment
report for the full trace and the consequence for Week 5.

**Second OCR bug, found when PaddleOCR was installed:** `requirements-ocr.txt`
declared `paddleocr` but left `paddlepaddle` in a comment, so the install was
incomplete. Because importing `paddleocr` succeeds without its engine, the
failure surfaced only on the first `recognize` call, where the import-time
fallback could not catch it. `requirements-ocr.txt` now declares the engine, and
`FallbackOCREngine` degrades to Tesseract on a runtime failure and records why.

### 6. UI candidate detection and annotation

`ui_detection.detect_ui_candidates` uses Canny edges plus contours to propose
simple non-text rectangles with area, aspect-ratio and containment filtering,
marked `source="contour"`. `visualization.draw_bounding_boxes` and
`save_annotated_image` render OCR and contour boxes in different colours on a
copy of the screenshot.

### 7. Text grounding (`src/gui_agent/perception/grounding.py`)

`find_text` supports `exact`, `contains` and `case_insensitive` matching, orders
every candidate by confidence, allows selecting one by index, and reports a
missing target through a message instead of raising. When a `ScreenInfo` is
supplied each match also carries its `control_center`.

### 8. Desktop control (`src/gui_agent/control/`)

`actions.py` wraps PyAutoGUI behind a `ControlBackend` protocol;
`executor.py` turns a `DesktopAction` into a real event or a dry run; `safety.py`
validates coordinates and redacts every ``type_text`` payload from logs, not just
those containing a credential keyword - a bare password like ``hunter2`` would
otherwise have been written in clear.

Safety behaviour: dry run is the default, real control requires an explicit
flag, both drag endpoints are validated, modifier keys are released in
`finally`, FailSafe errors become a failed `ActionResult`, and sensitive text is
never written to the run record.

### 9. Minimal run recording (`src/gui_agent/recording.py`)

Each run gets `outputs/week2/<session_id>/` with the before/after/annotated
images, `perception.json`, `action.json`, `run_summary.json` and `run.log`. No
full trajectory or monitoring system was built - that belongs to Week 4/6.

### 10. Demo scripts

`week2_perception_demo.py`, `week2_control_demo.py` and
`week2_closed_loop_demo.py`. The control and closed-loop demos default to a dry
run and only act when `--execute` is passed.

## Manual smoke test (macOS, built-in display)

```
$ python scripts/week2_perception_demo.py
captured 1470x956 in 268.3 ms
notice: PaddleOCR unavailable (...); falling back to Tesseract.
OCR backend 'tesseract' found 7 text regions in 350.3 ms
contour detection proposed 200 non-text candidates
annotated image: outputs/week2/.../annotated.png

$ python scripts/week2_closed_loop_demo.py --target-text Edge --move-only
[0] 'Edge' conf=0.96 box=(55,11,86,24) screenshot_center=(70,17) control_center=(70,17)
planned action: move at (70,17)
dry run: no real mouse or keyboard event was sent
success=True dry_run=True error=None
```

One real drag was performed on macOS with `--execute` (`from=(400,400)` to
`(800,600)`, `success=True dry_run=False`). No real click or keystroke was sent:
those share the same validated dispatch path and are covered through the fake
backend, and a click or keypress on a live desktop cannot be guaranteed
side-effect free.

## Known limitations

- PaddleOCR is installed on both machines and benchmarked, but it is only usable
  on the Windows GPU node (about 1 080 ms per frame against Tesseract's 1 420 ms).
  On the macOS CPU build it takes about 43 s per frame, so the Mac stays on
  Tesseract and the fallback notice appears there by design.
- `control/actions.py` is only 55% covered because the real PyAutoGUI calls
  cannot be exercised in automated tests by design.
- Multi-monitor support is implemented through monitor offsets but Week 2 only
  validated a single display.
