# Frozen bilingual UI accuracy experiment

This is a saved-image CPU experiment, not a real desktop task success rate. It
adds evidence to the earlier four-image timing experiment; it does not replace or
rewrite that experiment's English 20/20 and Chinese 0/8 observations.

Before optimization, `frozen_manifest.json` fixed twelve synthetic UI frames,
all 96 rendered text labels, all 72 drawn rectangle controls, font hashes, image
hashes, development/holdout split and the original OCR implementation copy. The
six holdout frames use different labels, sizes, placements and viewport dimensions
from the six development frames. The method was selected on development frames
and frozen in `algorithm_freeze_001.json` before the one formal holdout evaluation.
No labels, expected strings or fixture names enter production recognition.

Each arm ran three times per frame, alternating order. Text matching is case
sensitive, exact after ordinary whitespace normalization, and requires IoU at
least 0.5 with one-to-one assignment. Every visible text label is annotated;
unmatched OCR elements count as false positives, including erroneous text on
border strokes. The following counts use each frame once, not 144 independent
labels. All 54 repeated-frame result records are retained for latency/stability.

| Holdout method | Text TP / FP / FN | Precision | Recall | F1 | PNG decode + OCR + UI median |
|---|---:|---:|---:|---:|---:|
| Original native English | 28 / 4 / 20 | 87.50% | 58.33% | 70.00% | 199.343 ms |
| Original native, same English + Chinese resources | 30 / 11 / 18 | 73.17% | 62.50% | 67.42% | 342.302 ms |
| Screen profile, English + Chinese | 46 / 2 / 2 | 95.83% | 95.83% | 95.83% | 375.676 ms |

The screen profile matched English 30/30 and Chinese 16/18 labels on holdout.
Relative to the same bilingual native resources, recall rose 33.33 percentage
points and median frame processing cost rose 9.75%. Relative to the original
English-only setup, recall rose 37.50 points and cost rose 88.46%. This establishes
an accuracy benefit on these frames, not a claim that adding language files alone
is an algorithm improvement or that the full agent became faster.

Closed rectangle controls require one-to-one IoU at least 0.8. All methods found
36/36 holdout controls; their extra unmatched contour candidates were respectively
21, 9 and 2. This precision measure applies only to the fully annotated simple
rectangular controls in this fixture set, not arbitrary desktop icons.

The screen method uses one current-frame OCR pass: dark-theme normalization,
global contrast stretch, morphology to suppress long horizontal/vertical control
borders, 2x Lanczos enlargement, Tesseract sparse text layout (PSM 11/OEM 1), CJK
word-gap joining and outward-rounded coordinates mapped/clamped to the original
image. UI contour extraction continues to receive original pixels. No stale-frame
targets, vocabulary correction or ground-truth labels enter the method.

Development history remains visible. Batches 001 and 002 failed at backend path
handling before completing the comparison; they are not scored as OCR failures.
Batch 003 enlarged text without border suppression: recall improved but precision
and English recognition regressed (screen 33/48, precision 70.21%). Batch 004 added
the generic border suppression selected for freeze: screen 44/48, precision
93.62%, recall 91.67%. The holdout was not used to tune that choice.

The held-out low-contrast dark `holdout_06.png` remains a failure: `资源监测` and
`显示详情` are not exactly recognized. All three repetitions retain these errors;
no OCR output was corrected to its expected label. Every other held-out label
matched under the predefined exact-plus-localization metric.

`holdout_001/summary.json` is authoritative for final values, source/resource/input
hashes, per-stage latency and unchanged-source checks. No Torch or Paddle module
was imported, no GPU model inference or training ran, and no OS capture, LLM
planning or desktop action is included in the measured frame latency. Larger
real-world layouts, languages, scales and themes still require independent tests.

Reproduce a new evaluation batch (existing batches are never overwritten):

```powershell
python scripts/week6_benchmark_accuracy.py --split holdout --batch reproduction_001 --repeats 3 --tessdata-dir D:/week6-ocr/tessdata --scratch-dir D:/week6-ocr/scratch
```

Do not run `--prepare` against the existing fixture directory. The checked-in
fixture/annotation/baseline bytes already define this experiment. To evaluate a
new distribution, register a separate fixture manifest and results batch.
