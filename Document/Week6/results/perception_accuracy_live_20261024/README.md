# Independent saved real screenshot check

This probe evaluates the saved 2560 × 1600 PNG in
`live_auto_20261024_normal_003/automatic/initial_observation/`. The supplying live
runtime batch had unexplained external/manual inputs and is invalid as agent
execution-success evidence. That does not alter the saved screenshot pixels:
this probe makes no claim about who completed or controlled the source task.

The image was actually inspected before prediction. A coordinate-grid crop was
used to manually annotate the visible five English labels and two buttons in
`ground_truth_frozen_001.json`, with ROI `[1000, 480, 1600, 1130]`. This ground
truth was saved and timestamped before any of this probe's OCR calls. The three
methods, preprocessing parameters and source hashes were already frozen by the
earlier development-only selection. No parameters changed for this image.

OCR and contour extraction receive the entire original screenshot. Metrics only
include predictions whose box center falls inside the fixed ROI. Each label must
match exactly after ordinary whitespace normalization and IoU at least 0.5;
buttons require IoU at least 0.8. External UI is not annotated, so full-frame
precision is undefined, not 100%.

| Method | Independent ROI labels TP / FP / FN | Independent buttons TP / FP / FN | Full frame processing median |
|---|---:|---:|---:|
| Original English | 5 / 0 / 0 | 2 / 0 / 0 | 428.466 ms |
| Original bilingual | 5 / 0 / 0 | 2 / 0 / 0 | 577.626 ms |
| Screen bilingual | 5 / 0 / 0 | 2 / 0 / 0 | 836.893 ms |

Each arm ran three times in alternating order, giving nine raw result records.
All three repetitions had identical ROI predictions. They are repeated
measurements of **one** easy English frame and five labels, not fifteen independent
labels. There is no accuracy improvement on this frame, while the screen profile
costs more CPU time. This is a boundary on the synthetic held-out result's
generalization, not a reason to remove the unfavourable observation.

Timing includes full PNG decode, actual CPU Tesseract and UI contour extraction.
It excludes OS capture, live GUI operation, LLM planning and task execution. The
probe imported neither Torch nor Paddle. Source image, manual ground truth,
algorithm sources and probe source all passed unchanged-byte checks.

The authoritative values, all stage timings and limitations are in
`external_001/summary.json`; `run_manifest.json` pins the probe, image, algorithm
freeze and ground truth before inference. All raw text/box predictions, including
outside-ROI predictions excluded from precision, are preserved per repetition.
