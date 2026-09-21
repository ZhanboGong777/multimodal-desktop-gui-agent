# Week 2 Experiment Report: Desktop Perception and Control Modules

## 1. Task objective

Week 2 turns the Week 1 environment checks into reusable modules: capture the
screen continuously, preprocess it, read text with positions, propose non-text
UI candidates, locate a target by text, convert screenshot coordinates into
control coordinates, and perform mouse, keyboard and drag actions safely
through a dry-run-first executor.

No multimodal model, planner or agent loop is part of this week.

## 2. Module design

```
src/gui_agent/
├── schemas.py                  strict data models shared by every layer
├── coordinates.py              screenshot -> control mapping (pure functions)
├── config.py                   typed configuration loading
├── recording.py                per-run session directory and run summary
├── perception/
│   ├── capture.py              monitor / region / frame-sequence capture
│   ├── preprocessing.py        grayscale, resize, blur, Otsu, sharpen, crop
│   ├── ocr.py                  OCREngine + PaddleOCR (primary) + Tesseract (fallback)
│   ├── ui_detection.py         contour based non-text candidate detection
│   ├── visualization.py        bounding box drawing and annotated images
│   └── grounding.py            text target lookup with candidate list
└── control/
    ├── actions.py              PyAutoGUI wrappers behind a backend protocol
    ├── executor.py             DesktopAction -> ActionResult, dry run by default
    └── safety.py               coordinate, drag and log redaction checks
```

`coordinates.py` and `recording.py` are top-level because both the perception and
control layers use them and neither belongs to a single layer.

## 3. Experiment environment

This week used two machines: the MacBook Air for development, perception and
control, and a Lenovo Y9000P as the GPU node, used to validate platform differences
and to run the CUDA build of PaddleOCR.

| Item | Development machine (MacBook Air M2) | GPU node (Lenovo Y9000P) |
| --- | --- | --- |
| CPU / GPU | Apple M2, 24 GB unified memory | NVIDIA RTX 4060 Laptop, 8188 MiB |
| OS | macOS 26.3.1, ARM64 | Windows 11 (build 26200) |
| Display | external monitor, 1920 x 1080 | built-in, 2560 x 1600 at 150% scaling |
| Python | 3.12.10 in `.venv` | 3.12.4 in `.venv` |
| Capture | MSS 10.2.0 | MSS 10.2.0 |
| Image processing | OpenCV, Pillow, NumPy | same |
| Tesseract | 5.5.3 (`eng`), pytesseract | Tesseract-OCR under `C:\Program Files` |
| PaddleOCR | 3.7.0 with PaddlePaddle 3.3.1 (**CPU build**) | 3.7.0 with PaddlePaddle-GPU 3.3.1 |
| CUDA runtime | - | CUDA 12.6, driver 560.94 |
| Deep learning | - | PyTorch 2.14.0+cu126 |
| Control | PyAutoGUI | PyAutoGUI |
| Tests | Pytest 9.1.1, pytest-cov, Ruff 0.16.6 | same |

### Environment verification

Both machines ran the environment check and the smoke tests:

| Check | Mac | Windows |
| --- | --- | --- |
| Package completeness | all OK | all OK |
| Capture and OCR smoke test | passed | passed (2560 x 1600) |
| Image processing smoke test | passed | passed |
| Control dry run | passed | passed |
| GPU acceleration | MPS available | **CUDA available: True**, CUDA op PASSED |
| Unit tests | 165 passed | 148 passed (recorded before the hardening pass) |
| Ruff | clean | clean |

The test count matched exactly on both machines before the hardening pass, which
shows the code carries no platform-dependent branches that change behaviour.

> The Windows figures in this table were recorded before the final hardening pass.
> Re-running the suite there after pulling that commit is still outstanding; the
> Mac column already shows the final state (165 tests, 88% coverage).

## 4. Implementation process

1. Data models first, so every later module had a fixed interface to code against.
2. Coordinate mapping as pure functions with its own tests, before any capture
   or control code could depend on it.
3. Capture, then preprocessing, then OCR, then UI candidates and annotation.
4. Grounding, then the control layer, then the minimal run recorder.
5. Demo scripts last, wired to the modules rather than reimplementing them.

## 5. Continuous capture and performance

`capture_frames(count, interval)` records a per-frame latency and times the whole
burst. Two different numbers come out of it, and they must not be confused:

- **`capture_only_fps`** - throughput implied by the mean capture cost alone
  (`1000 / average_capture_ms`), ignoring the wait between frames.
- **`effective_fps`** - the rate the sequence actually achieved
  (`frame_count / elapsed_seconds`), including that wait.

Measured on an external display (1920 x 1080):

| Run | Frames | Latency min / mean / max | `capture_only_fps` | `effective_fps` |
| --- | --- | --- | --- | --- |
| 10 frames, 0.2 s interval | 10 | 31.8 / 36.9 / 49.7 ms | 27.1 | **4.52** (2.21 s) |
| 5 frames, no interval | 5 | 20.9 / 22.1 / 24.4 ms | 45.2 | **43.56** (0.11 s) |

With a 0.2 s interval a frame costs only 37 ms to capture, but the burst still
waits 200 ms between frames, so it advances at **4.52 FPS** rather than 27.1. The
two figures only converge when frames are captured back to back.

A sequence can be produced straight from the command line (one frame by default,
so a plain run does not pay for OCR on every frame):

```
$ python scripts/week2_perception_demo.py --frames 10 --interval 0.2
captured 10 frames in 2.18 s (interval 0.20 s); working on the last one
capture-only throughput 28.7 FPS, effective sequence rate 4.59 FPS
```

A single capture costs 20 - 50 ms, far below the 1 - 2 s per step an agent loop
needs. The frame rate is set by the interval the caller asks for, not by capture
performance. The first capture in a process initialises the MSS session and costs
240 - 535 ms; later frames settle at 20 - 50 ms.

On this machine the screenshot size equals the control size, so the scale factor
is 1.0 and the mapping is a no-op. `scripts/check_coordinate_spaces.py` reports
that explicitly, and the mapping itself is covered against 2x Retina, independent
axis scaling and non-zero monitor offsets.

> The sample console output quoted later in this report (sections 6, 7 and 9) was
> captured on the built-in display (1470 x 956) before the external monitor was
> reconnected, which is why those sizes differ from the table above. Both sets of
> figures are real measurements, taken in different sessions.

## 6. OCR and UI candidate detection

`python scripts/week2_perception_demo.py`:

```
captured 1470x956 in 268.3 ms
notice: PaddleOCR unavailable (PaddleOCR is not installed ...); falling back to Tesseract.
OCR backend 'tesseract' found 7 text regions in 350.3 ms
contour detection proposed 200 non-text candidates
```

Sample recognised regions:

| Source | Text | Confidence | Box | Centre |
| --- | --- | --- | --- | --- |
| ocr | `Edge` | 0.96 | (55,11,86,24) | (70,17) |
| ocr | `«` | 0.82 | (20,7,34,23) | (27,15) |
| ocr | `TARA` | 0.69 | (376,10,427,23) | (401,16) |
| contour | | 0.50 | (1105,244,1470,755) | (1287,499) |

Note that the confidence threshold and the empty-text filter both behave as
intended, and contour candidates carry no text but do carry a box and centre.

### Measured cost per backend

| Platform | Backend | Boxes | Box unit | Time |
| --- | --- | --- | --- | --- |
| MacBook M2, CPU | Tesseract (fallback) | 153 | word | about 1 244 ms |
| MacBook M2, CPU | PaddleOCR, medium model | 116 | line | about 43 200 ms |
| MacBook M2, CPU | PaddleOCR, mobile model | 165 | line | about 34 300 ms |
| MacBook M2, CPU | PaddleOCR, mobile model, half resolution | 134 | line | about 19 100 ms |
| Windows, RTX 4060 | Tesseract (fallback) | 263 | word | about 1 420 ms |
| **Windows, RTX 4060** | **PaddleOCR, medium model** | 86 | line | **about 1 080 ms** |

Two cautions when reading this table. The box counts are **not comparable**:
Tesseract reports one box per word and PaddleOCR one per text line, so 263 against
86 says nothing about recall. The timings are comparable, and they say PaddleOCR is
about 36x faster on the RTX 4060 than on the Mac's CPU (1 080 ms against 43 200 ms)
and about 25% faster than Tesseract on the same Windows screen. The first timed run
of each engine includes model initialisation (about 3.8 s for PaddleOCR) and should
be ignored; the figures above are the second run.

On macOS the arm64 PaddlePaddle build is CPU-only, so the Mac stays on Tesseract -
the fallback the acceptance criteria allow. Starting PaddleOCR on Windows needed
three separate fixes; see section 11.

## 7. Bounding box annotation

`draw_bounding_boxes` draws OCR elements in red and contour candidates in blue on
a copy of the screenshot; `save_annotated_image` writes the result into the
session directory. This is graded Week 2 output, not throw-away debug code.

**Readability work.** An early version labelled every element, which covered a
full screen in more than 350 overlapping tags. Four changes followed:

1. Labels are now drawn for OCR elements only (`label_sources`, default `("ocr",)`).
   Contour candidates carry no text, so their label had no information in it.
2. `--max-ui-candidates` caps how many candidates are kept. Candidates are sorted
   by area, so lowering the cap drops the small, fragmented boxes first and keeps
   the largest, most meaningful ones.
3. Candidates that overlap a region OCR already found are dropped, so the contour
   pass stops re-framing text.
4. `min_rectangularity` is available for tuning.

**A measured caveat about rectangularity.** The useful threshold is not portable
between themes. On the same machine and the same algorithm:

| Rectangularity | Light UI (VS Code light theme) | Dark UI (terminal + browser) |
| --- | --- | --- |
| 0.00 (off) | 424 candidates | 193 candidates |
| 0.55 | 376 candidates | 27 candidates |
| 0.80 | 60 candidates | **1 candidate** |

At 0.55 nothing is removed on the light screen while 86% of the candidates vanish
on the dark one. The parameter therefore defaults to **off** and is documented as
a per-theme tuning knob rather than a safe default.

**A real limitation.** When the screen shows a photographic wallpaper rather than
an application, Canny edges plus contours frame texture detail and the candidates
carry no meaning. The detector is only useful over interfaces with clear
rectangular controls, which is why Week 2 describes it as simple candidate
detection and not as a semantic detector.

 Coordinate conversion test

Covered by `tests/test_coordinate_mapping.py`:

| Case | Expectation |
| --- | --- |
| Screenshot size equals control size | identity |
| Screenshot is 2x the control size | halves the point |
| Axes scale differently (0.5 and 0.8) | each axis scaled independently |
| Monitor offset `(215, 1080)` | offset added after scaling |
| Points at and past the edges | last valid pixel is inside, next one is not |
| Points far outside | clamped onto the monitor |
| Inverted clamping range | raises `ValueError` |

### Measured on both machines

`scripts/check_coordinate_spaces.py` was run on each platform:

| Platform | Display scaling | Screenshot | Control | Scale | Offset |
| --- | --- | --- | --- | --- | --- |
| macOS, external monitor | - | 1920 x 1080 | 1920 x 1080 | 1.0000 | (0, 0) |
| Windows, built-in panel | **150%** | 2560 x 1600 | 2560 x 1600 | 1.0000 | (0, 0) |

The Windows result deserves a note. At 150% scaling the usual expectation is a
mismatch: the screenshot would report physical pixels (2560 x 1600) while the
control API reports logical pixels (1707 x 1067). They agree here because the
**Python 3.12 process is DPI aware** (CPython has declared DPI awareness in its
manifest since 3.8), so `pyautogui` (`GetSystemMetrics`) and `mss` both work in
physical pixels.

**Conclusion:** the conversion is an identity mapping on both machines, so the
mapping layer is never exercised for real. Keeping it is still justified, because a
DPI-unaware process (an older Python, or certain packaging tools), an `mss` build
that reports physical pixels while the control API uses logical points, or a
multi-monitor setup with mixed scaling would all produce a mismatch. The mapping
itself is covered by the seven cases above.

## 9. Mouse, keyboard and drag test

`python scripts/week2_control_demo.py --action move --x 200 --y 200`:

```
planned action: move at=(200,200)
monitor: 1470x956, control space 1470x956
dry run: no real mouse or keyboard event was sent
success=True dry_run=True error=None
```

Drag is validated on **both** endpoints: a drag from `(1,1)` to `(5000,1)` is
refused, and a drag whose start equals its end is refused.

**Measured drag run.** Executed on the macOS machine on a 1920 x 1080 display:

```
planned action: drag from=(400,400) to=(800,600)
monitor: 1920x1080, control space 1920x1080
--execute given: real control is enabled, move the pointer to a screen corner to abort
executing in 3s
success=True dry_run=False error=None
```

The pointer press, drag and release were performed for real on a harmless surface,
and the session directory was written. Click and keyboard actions share the same
validated dispatch path and are covered through the fake backend in the unit tests;
neither was performed for real, which is deliberate, because a real click or
keystroke on a live desktop cannot be guaranteed to be side-effect free.

## 10. Safe closed loop test

`python scripts/week2_closed_loop_demo.py --target-text Edge --move-only`:

```
OCR backend 'tesseract' found 7 regions in 312.4 ms
 [0] 'Edge' conf=0.96 box=(55,11,86,24) screenshot_center=(70,17) control_center=(70,17)
annotated image: outputs/week2/<session>/annotated.png
selected target 'Edge': screenshot center (70,17) -> control (70,17)
planned action: move at (70,17)
dry run: no real mouse or keyboard event was sent
success=True dry_run=True error=None
after image: outputs/week2/<session>/after.png
```

The session directory contains `before.png`, `annotated.png`, `after.png`,
`perception.json`, `action.json`, `run_summary.json` and `run.log`.

## 11. Problems and handling

**Tesseract language code mismatch (fixed).** The configuration stores backend
agnostic codes (`en`), but Tesseract only knows `eng`. The first real run failed
with `Failed loading language 'en'`. `to_tesseract_languages()` now maps the
codes and the mapping is unit tested.

**Whitespace-only typed text (fixed).** `type_text` with `"   "` passed
validation and would have produced a no-op action reported as successful. The
validator now rejects blank text.

**Coverage gap found by measuring (fixed).** The first coverage run showed
`recording.py` at 0% even though the hand-off requires a JSON serialisation test.
`tests/test_recording.py` and `tests/test_preprocessing.py` were added, lifting
the total from 76% to 88%.

**PaddleOCR installed without its engine (fixed).** `requirements-ocr.txt` listed
`paddleocr` but mentioned `paddlepaddle` only in a comment, so installing the file
produced a broken backend. Worse, importing `paddleocr` succeeds without the
engine, so the failure only appeared on the first `recognize` call and the
import-time fallback never fired - the demo aborted instead of degrading.

Two fixes: `requirements-ocr.txt` now declares `paddlepaddle`, and
`FallbackOCREngine` wraps a primary backend so a **runtime** failure switches to
Tesseract permanently and reports why. Verified by reproducing the exact
condition on a machine where `paddleocr` is installed and `paddlepaddle` is not.

**CUDA/cuDNN DLL collision between PaddlePaddle and Torch (worked around).** On the
Windows GPU node, `import paddle` followed by `from paddleocr import PaddleOCR`
failed with:

```
OSError: [WinError 127] The specified procedure could not be found.
Error loading "...\torch\lib\cudnn_cnn64_9.dll" or one of its dependencies.
```

The chain is `paddleocr -> paddlex -> modelscope -> import torch`. Importing
PaddlePaddle first loads its CUDA and cuDNN libraries into the interpreter, and
Torch then cannot load its own cuDNN build. The same import succeeds when Torch
is loaded first, which is why the standalone check passed and the benchmark
failed.

This was unusually hard to diagnose because the OCR layer reported the failure as
"PaddleOCR is not installed": the original exception was swallowed by a broad
handler. Two things then changed. `PaddleOCREngine` includes the underlying error
in its message, and `_prepare_platform()` puts `None` into `sys.modules["torch"]`
before importing PaddleOCR, so ModelScope's transitive `import torch` fails
harmlessly and PaddlePaddle can load its own cuDNN.

That is a **workaround inside the OCR process, not process isolation**: afterwards
Torch simply cannot be imported in that interpreter. Only
`scripts/benchmark_ocr.py` isolates properly, by probing the PaddlePaddle build in
a real child process.

**Consequence for later weeks:** on this node PaddlePaddle and Torch cannot be
imported into the same process. Week 5 fine-tuning uses Torch, so any pipeline
that needs PaddleOCR *and* a Torch model must either run them in separate
processes or align the CUDA/cuDNN builds. This should be settled before Week 5.

**Run record leaked typed text (fixed).** `recording.py` wrote `action.json` and
`run_summary.json` by calling `model_dump()` directly, so the `redact_action()`
helper in `control/safety.py` was never applied - and `ActionResult` carries its
own nested `DesktopAction`, which was missed as well. A credential typed by the
agent would have landed on disk in clear, contradicting what the work log claimed.

`redact_action()` and a new `redact_result()` are now applied on the way out, and
`redact_payload()` works on the serialised copy so the caller's action object is
never modified.

**A deliberate tightening beyond the review.** The review asked for sensitive
content to be masked and named a test `test_non_sensitive_text_is_preserved`.
Implementing that revealed a real hole: keyword matching only catches text that
contains a telltale word, so a bare password such as `hunter2` would still have
been written in clear. Every `type_text` payload is therefore masked, and that
test was replaced by `test_ordinary_typed_text_is_also_redacted`.

The trade-off is that a run record no longer shows what the agent typed, which
costs some diagnostic value; the before/after screenshots still capture it
visually. If a later week needs a readable trajectory for a demonstration, this
should become a config switch rather than a code change.

## 12. Final result

| Criterion | Result |
| --- | --- |
| Perception and control logic moved into `src/gui_agent` | yes, 13 modules |
| Continuous multi-frame capture with latency/FPS | yes; capture-only and effective rates reported separately |
| OCR returns text, confidence and bounding boxes | yes, both backends; PaddleOCR runs on the Windows GPU at about 1 080 ms |
| Non-text UI candidates | yes, 200 candidates on a real screen |
| Bounding boxes drawn and saved | yes |
| Text target lookup | yes, with candidate list and index selection |
| Coordinate conversion | yes, 100% covered |
| Mouse, keyboard, wait and drag | yes; the drag was executed for real on macOS |
| Dry run by default | yes |
| Closed loop executed safely | yes, including one real pointer move |
| Before/after images and run record saved | yes; typed text is always masked in the record |
| Unit tests | 165 passed |
| Coverage | 88% |
| Ruff | clean |
| Week 1 scripts still run | all four exit 0 |
| Platform differences handled | yes; macOS and Windows both verified, plus a Windows display at 150% scaling |
| Windows node | 148 tests passed before the hardening pass, ruff clean, all demos run, GPU OCR at about 1 080 ms |

## 13. Deliverables

Code: continuous capture, preprocessing, OCR interface with two backends, UI
candidate detection, annotation, text grounding, coordinate conversion, mouse /
keyboard / drag control, safety executor, minimal run recording, three demo
scripts and ten test files.

Documentation: `Document/Week2/WORKLOG.md`,
`Document/Week2/Week2_Experiment_Report.md`,
`Document/Week2/Week2_Unit_Test_Report.md`.

## 14. Next week plan

- Add the model client interface and the first task planning layer on top of the
  perception and control modules (Week 3).
- Exercise the preprocessing module for real. All four switches in
  `configs/default.yaml` default to off, so `apply_preprocessing` is a no-op in
  every run: the module is implemented and covered by unit tests, but its effect on
  OCR accuracy has never been measured. Compare the same screen with and without
  grayscale plus Otsu binarisation and record the result.
- Align the cuDNN build. PaddlePaddle is compiled against cuDNN 9.9 while the
  environment carries Torch's 9.5, so PaddlePaddle warns on every start. Text
  recognition is stable at about 1 080 ms, but this should be settled before any
  week needs heavier operators.
- Decide how PaddlePaddle and Torch are kept apart. They cannot share a process on
  the Windows node, and Week 5 fine-tuning needs Torch.
- Optimise the annotation image, showing labels for OCR elements only.
- Validate multi-monitor behaviour and Linux compatibility if a machine becomes
  available.
