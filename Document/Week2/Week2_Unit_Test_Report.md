# Week 2 Unit Test Report

Command:

```bash
pytest --cov=gui_agent --cov-report=term-missing
```

Result: **165 passed**, **88% coverage** over `src/gui_agent`.

## Coverage per module

| Module | Stmts | Miss | Cover |
| --- | --- | --- | --- |
| `schemas.py` | 114 | 0 | 100% |
| `coordinates.py` | 18 | 0 | 100% |
| `recording.py` | 54 | 0 | 100% |
| `perception/grounding.py` | 72 | 1 | 99% |
| `perception/preprocessing.py` | 76 | 1 | 99% |
| `perception/ui_detection.py` | 72 | 2 | 97% |
| `config.py` | 66 | 3 | 95% |
| `perception/visualization.py` | 60 | 4 | 93% |
| `control/safety.py` | 52 | 5 | 90% |
| `perception/capture.py` | 155 | 20 | 87% |
| `perception/ocr.py` | 207 | 43 | 79% |
| `control/executor.py` | 83 | 22 | 73% |
| `control/actions.py` | 66 | 30 | 55% |
| **Total** | **1096** | **131** | **88%** |

The two lowest modules are the ones that talk to the real world:
`control/actions.py` contains only direct PyAutoGUI calls, and `ocr.py`'s
PaddleOCR backend is only reached when PaddleOCR is installed and a model is
loaded. Both are exercised indirectly through the fake-backend and parsing
tests.

## Required test cases from the hand-off

| # | Required case | Where | Result |
| --- | --- | --- | --- |
| 1 | `BoundingBox` centre and invalid box validation | `test_schemas.py` | pass |
| 2 | Frame count, interval and error handling of continuous capture | `test_capture.py` | pass |
| 3 | Screenshot-to-control conversion and boundary clamping | `test_coordinate_mapping.py` | pass |
| 4 | OCR normalisation, empty results and backend fallback | `test_ocr.py` | pass |
| 5 | Simple rectangular UI candidates on a synthetic image | `test_ui_detection.py` | pass |
| 6 | Image size and pixel change after drawing boxes | `test_visualization.py` | pass |
| 7 | Exact, contains and case-insensitive text matching | `test_grounding.py` | pass |
| 8 | Multiple candidates and missing target handling | `test_grounding.py` | pass |
| 9 | Dry run never calls real PyAutoGUI | `test_control_safety.py` | pass |
| 10 | Invalid actions and out-of-bounds drag are rejected | `test_control_safety.py` | pass |
| 11 | Control exceptions become an `ActionResult` | `test_control_safety.py` | pass |
| 12 | Minimal run record serialises to JSON | `test_recording.py` | pass |

Additional coverage beyond the required list:

- `test_preprocessing.py` - every preprocessing helper, configuration driven
  chaining, and a check that no helper mutates its input.
- `test_visualization.py` - labels default to OCR elements only, and
  `label_sources` can opt contour candidates back in.
- `test_ocr.py` - Tesseract language-code translation (`en` -> `eng`), the
  runtime fallback from a failing PaddleOCR to Tesseract, and parsing NumPy
  arrays as well as lists.
- `test_ui_detection.py` - the rectangularity filter and the exclusion of regions
  OCR already covers.
- `test_coordinate_mapping.py` - non-zero monitor offsets and inverted ranges.
- `test_schemas.py` - per-action payload requirements, including `drag`.

## How the desktop is kept out of the test suite

- `test_capture.py` replaces the MSS session with a fake that returns synthetic
  frames, and patches the control-size probe, so no test reads the real screen.
- `test_control_safety.py` passes a `FakeBackend` that records calls instead of
  sending events; dry-run tests assert that no backend is created at all.
- `test_ocr.py` patches `pytesseract.image_to_data`, so Tesseract does not have
  to be installed or run.
- `test_ui_detection.py` and `test_preprocessing.py` generate their own images.

Real screenshots, OCR and pointer control are covered by the manual smoke tests
described in the work log, not by Pytest.

## Known gaps

- The `PyAutoGUIBackend` methods themselves are not unit tested; they are thin
  wrappers and are verified manually with `--execute` on a safe screen.
- PaddleOCR parsing is tested against both the 2.x and 3.x result shapes, and
  against NumPy arrays as well as lists, but always with synthetic payloads: the
  tests never load a real model.

## Note on the diagnostic scripts

`scripts/benchmark_ocr.py` probes the PaddlePaddle build in a **child process**.
That is deliberate: loading PaddlePaddle's CUDA libraries into the benchmark's own
interpreter makes Torch fail to load its cuDNN build on the Windows node (see the
experiment report). The probe is not unit tested because it only reports the local
environment; its parsing is a single `subprocess.run` call.
