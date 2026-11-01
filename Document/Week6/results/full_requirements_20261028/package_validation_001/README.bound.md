# Multimodal Desktop GUI Agent

### Week 6 system v2.1.0

The `codex/week6-full-requirements` branch implements model-generated ordered
semantic objectives, bounded recovery, persistent bilingual OCR, and parent/child
execution monitoring. Natural-language CLI requests default to
`week6.semantic_subtasks.v6`; the Python API retains its compatible default and
uses `protocol_mode="semantic_subtasks"` for this path.

The same286-file source freeze passed controlled normal048 and transient027
desktop acceptance with the installed `qwen3-vl:8b-instruct` digest for both model roles. The recovery
case includes the fixture's real first review rejection and a subsequent native
fresh-screen action retry within the second semantic objective before local commit. The report records actual event attribution,
the normal-case auxiliary age-check deviation and model-prediction limitations. Failed objectives
remain failed, even when later work reaches the caller's final state. These two
functional cases do not establish a general desktop success rate.

The unfiltered final regression passed **3803 tests in 664.89s**;
pytest and Ruff actually exited0. All286 source/config/test files matched before,
after and against the freeze. The tested README is retained separately because
this final documentation update follows testing.

Validation summary: `3803 tests, ruff clean`.

On8 fully annotated synthetic held-out frames, persistent bilingual OCR achieved
93.6508% text F1; two batches measured108.291 and109.888ms warm full-frame median
time. This includes decoding, preprocessing, actual OCR and coordinate output,
excluding OS capture, model calls and input. Cold start remained slower than the
English baseline. One separately saved real frame showed no ROI accuracy gain
and no universal speed advantage. No OCR/model training was performed.

Quoted action descriptions are checked against the chosen readable control.
An exact competing label still refuses input. Absent ASCII alphabetic labels
allow bounded OCR spelling uncertainty on the already-selected witness only:
one edit for4..9letters, at most2when both names have at least10letters.
Whitespace-only differences also remain uncertain. This does not replace the
native target, ID or coordinates, prove arbitrary intent, or grant completion.
Numeric and shorter names retain the refusal. All original task checks remain.

Caller-final verification text describes evidence to observe, not content to type.
Local save or commit does not add message, title or description editing. Native
unsupported operations still fail the original validation. The same refusal
now explicitly explains local message-content composition; its matching and
refusal branch remain unchanged.

Repair policyv5 adds a fixed unsupported-message retry only for the actual local
guard exception. Its rejected reply stays archived but is omitted from that
retry conversation; caller intent and bounded errors remain. The independent
auditor reconstructs the exception and exact messages. All other histories,
actions, predicates and budgets remain unchanged.

A nonfinal native objective cannot reuse the exact whole immutable caller-final
conjunction while later objectives remain. The existing duplicate-condition
feedback requests a new native reply; no plan is rewritten locally. Partial
conditions and operation meanings still need original evidence review.

After full006 exposed11 producer/auditor feedback-literal mismatches, only the
independent auditor fixed literal was synchronized. All other285 frozen paths,
including executed runtime, configuration and tests, remain byte-identical to
the original GUI pair. New actual saved-record audits passed for both cases;
the original physical evidence and acceptance remain linked, with no new GUI
experiment or new image-viewing claim. Full006 remains a failed historical run.

The final controlled recovery occurred inside one semantic objective, with no
failed semantic objective or new high-level decomposition round. The normal
case retains one operator deviation: the auxiliary30s age check exited1 at38s,
but input was still dispatched within the original bridge deadline. Original
identity checks, immediate pixels, receipts and application events are retained;
there is no fabricated passing precheck record. The strict and receipt audits
passed. This qualification is included in the report.

Optional visual candidate details attach original anonymous contour crops with
current IDs, not synthetic labels. Readable controls require nonblank native
target text. Generation guidance asks that each objective name the operations
needed for its outcome; missing immutable caller-final text means unfinished
work. Guidance does not rewrite native replies or relax guards. The final
high-level failure prompt was not exercised by the successful GUI path.

The high-level model reads caller intent and bounded logical feedback without
current screen text, coordinates or images; action planning observes fresh
screenshots. Model intermediate predicates are untrusted predictions fixed
before input. Credit requires an initially unsatisfied condition, confirmed
input and fresh verification; the caller final is checked separately. Unknown
dispatch stops without replay. Substring matching cannot prove arbitrary
negation or business causality, so original pixels and independent fixture
events supplement controlled acceptance. Monitoring is event-driven and may
be stale during long model calls. Generation guidance does not guarantee that
the model will avoid adding unsupported requirements; validation remains active.

See [usage](Document/Week6/README.md), [release notes](Document/Week6/Release_Notes.md),
[report](Document/Week6/Week6_Robustness_Report.md),
[source freeze](Document/Week6/results/full_requirements_20261028/auditor_prompt_sync_001/final_source_freeze_001.json),
[full regression](Document/Week6/results/full_requirements_20261028/full_validation_007/root_audit_001.json),
[normal acceptance](Document/Week6/results/full_requirements_20261028/review/normal048_root_acceptance_003.json),
[recovery acceptance](Document/Week6/results/full_requirements_20261028/review/transient027_root_acceptance_003.json),
and [ended development attempts](Document/Week6/results/full_requirements_20261028/live_attempts_index_016.json).

Prior documentation bytes are retained in
[history](Document/Week6/history/pre_final_acceptance_20261101_016/manifest.json).
Earlier v2.0 manifests and pending/version-specific records describe those
historical artifacts. The final external distribution receipt binds the actual
Git commit, publication, source ZIP, wheel and new report bytes. The report uses
the supplied Week5 typography/tables and has no cover date/timezone line.
Week7 comparisons remain outside this delivery.

### Week 5 submission

The completed Week 5 delivery is in [Document/Week5](Document/Week5/README.md),
including the [final report](Document/Week5/Week5_Final_Report_20261017.md),
[reproduction instructions](Document/Week5/Reproduction.md), public result summaries,
and the actual [C11 LoRA](artifacts/week5/c11/). The final model is the pinned
Qwen2.5-VL-3B-Instruct with resized pixel action JSON and structure6/coordinate4/other2 CE.
All original 16 offline gates passed. The original 12 desktop messages were verified
across three prompt versions and five successful execution candidates; this is
not a single fixed prompt scoring 12/12. Historical alternative implementations
remain in the source with their CPU tests. Week 4 results below retain their
historical scope; current test collection is checked separately.

### Project Overview

This project develops and optimizes a desktop GUI agent powered by multimodal large
language models. The agent is expected to understand natural-language instructions,
perceive text and interface elements on the screen, plan task steps, and perform
desktop operations through mouse and keyboard controls.

Work is split across two machines: an Apple-silicon laptop for development and a
Windows laptop with an NVIDIA GPU for model serving. The two communicate only over
HTTP through an OpenAI-compatible endpoint, which keeps the CUDA stacks on the
machine that has the GPU.

### Expected Development Progress

| Week | Planned Work |
|---|---|
| Week 1 | Technical research and development environment setup |
| Week 2 | Desktop perception and control module development |
| Week 3 | GUI dataset processing and multimodal Agent framework setup |
| Week 4 | End-to-end system integration and basic desktop task testing |
| Week 5 | Multimodal model fine-tuning and prompt optimization |
| Week 6 | Advanced features and system robustness optimization |
| Week 7 | System evaluation and performance analysis |
| Week 8 | Code cleanup, technical report, and system demonstration |

### Week 4

The project has completed Week 4. `scripts/week4_agent_cli.py` runs one task
through the closed loop: observe, plan, resolve each step against the *current*
frame, act, observe again, verify.

**Dry run is the default.** Real desktop actions need `--execute` and an
interactive confirmation, and there is deliberately no `--yes`.

```bash
python scripts/week4_agent_cli.py --list-cases      # the five task cases
python scripts/week4_agent_cli.py --case T01        # dry run: dispatches nothing
python scripts/week4_offline_demo.py                # the loop, scripted, no desktop
python scripts/week4_collect_evidence.py --latest T01   # copy a run's records into the repo
```

Three properties the loop is built around:

- **Plan, then step-by-step re-observation, with bounded recovery.** Up to four
  planning attempts share the task deadline and cumulative action budget. An
  anonymous control can require a separate visual mapping request; its approved
  action and text stay fixed. Every action still passes adapter resolution and
  executor boundary checks, and T04 never automatically retries after a send attempt.
- **A screenshot changing is not success.** Only the task's own success rule can
  return `succeeded`; `finish` and a run out of steps cannot.
- **Refusal beats guessing.** A stale anonymous id needs unique correspondence
  with an actual current candidate, confirmed against a further screenshot after
  any visual mapping call. Missing, ambiguous or unsafe evidence stops the pass.

The historical Week 6 v2.0 [full006](Document/Week6/results/release_validation_completed_20261026_006/validation.json)
record has 3027 passing tests and a clean Ruff check; the v2.1.0 full regression is
pending. See `Document/Week4/Week4_Usage.md` for the flags, safety model and record layout.

The Chinese deliverables - `Week4_中文实验报告.(md|docx)`, `Week4_Windows复核手册.md`
and the `sync_report_numbers.py` tool that keeps their numbers in step with this
tree - are delivered alongside the repository rather than inside it, so a citation
to one of those filenames will not resolve here.

Real Windows evidence records four of five basic cases passing, across five
successful runs: T01 twice, and T02, T03 and T05 once each. The latest retained
T04 run, `T04_20261006_115446`, was blocked by its initial visual-context check
and dispatched zero actions. T04 has no retained real success. The attempts and timings are in
`Document/Week4/Week4_Basic_Task_Test_Report.md`.

Detected text-free controls now appear in the model's target list as
`<unlabelled box>` with their frame-local ids and geometry. The 300-element cap
allows 100 OCR labels plus the configured 200 contour candidates, with text
ranked first. Contour detection retains small nested controls and prioritises
the freshly captured foreground window. The runner refreshes stale anonymous
targets using their pixels and context; when appearance changes, a constrained
visual mapping call can select only a detected candidate. A new capture confirms
that selection before dispatch, with stable window identity and focus checks.

T04's `message_regions.py` selects observed foreground regions for the header,
possible composer and transcript candidates. The model assesses header and
composer crops independently through strict JSON schemas; the expected recipient,
full marker and image-filename timestamps are withheld from model text. The final
message assessment can name only an observed candidate id and must identify an
outgoing, sent bubble containing the
exact marker above an empty composer. Crops preserve native pixels and record
source bounds and layout transforms. They are visual evidence, not new action targets.
T04 planning lists only observed controls within the visually authorised composer
and tells retries when this run's marker has already been typed. A saved-frame
real-model probe now produces the correct click-editor, type-marker, click-send plan;
the strict final assessment also rejects the old frame's mismatched message marker.
The runner preserves the authorised header pixels and checks current foreground
identity before each input event; both confirmations, current-candidate resolution,
coordinate bounds and fail-safe remain required.

The complete click-input, type-marker, click-send flow passes against the
`prepared-T04` mock provider and synthetic chat; a historical screenshot replay
also retains both required contours within the 200-candidate cap. Two real-model
requests against the saved `115446` frame now pass its isolated recipient/editor
context check with zero desktop actions. This is offline saved-image validation,
not a real message-send result. A new real run still needs the CLI's two confirmations.

### Week 3

The project has completed Week 3. The Week 3 deliverables - the dataset preparation
script and the base Agent framework - are on `main`.

**Datasets.** Three adapters normalise ScreenAgent, Mind2Web and WebArena into one
`GUITaskSample`, each validated against its real archive rather than against a
fixture:

| Source | Converted |
|---|---|
| ScreenAgent (`test.zip`) | 200 / 200 |
| WebArena (`test.raw.json`) | 20 / 20 |
| Mind2Web (Parquet shard, 36 tasks) | 268 / 268 |

`scripts/week3_prepare_dataset.py` reads JSON, JSONL, Parquet and zip, writes a
validated JSONL export, and exits non-zero when the export is empty.

**Model interface.** One `ModelClient` contract with three interchangeable backends:

| Provider | Needs a key | Use |
|---|---|---|
| `mock` | no | tests, demos, offline runs |
| `openai_compatible` | `GUI_AGENT_API_KEY` | a real model, hosted or local |
| `langchain` | `GUI_AGENT_API_KEY` | the same models, through LangChain |

Only the LangChain adapter imports LangChain. Nothing in `planning/` or `datasets/`
does, so the framework cannot become a dependency of the rest of the code.

**Planning.** `TaskPlanner` turns an instruction plus screen context into a
Pydantic-validated `TaskPlan`. `finish` is a planning verb only and is excluded from
the executable steps, so a terminal marker cannot reach an executor. Nothing is
executed in Week 3; the closed loop is Week 4.

Verified with `qwen2.5vl:7b` served by Ollama on the Windows machine and driven from
the Mac over HTTP. The same instruction and screenshot sent through both remote
backends returned byte-identical plans, which is the point of the abstraction.

```bash
python scripts/week3_prepare_dataset.py --dataset webarena \
    --input data/raw/webarena/test.raw.json --output data/processed/webarena.jsonl --limit 20
python scripts/week3_model_demo.py --provider mock
python scripts/week3_planning_demo.py --provider mock --instruction "Open the browser"
```

The historical Week 6 v2.0 [full006](Document/Week6/results/release_validation_completed_20261026_006/validation.json)
record has 3027 passing tests and a clean Ruff check; this is not the current
v2.1.0 suite result. The model client's retry, timeout and error-classification
paths are covered, along with the four vision-payload
failure modes (missing, empty, oversized, unknown type) and the TaskPlan
schema boundaries. Dataset and model dependencies live in
`requirements-agent.txt`, deliberately separate from the base requirements so that a
failure there cannot break the Week 2 perception and control modules.

Reports and evidence for the week are under `Document/Week3/`, including the work
log, dataset notes and the raw JSON from the real model calls.

### Week 2

The project has completed Week 2. The repository now contains reusable screen
capture, image preprocessing, two OCR backends (PaddleOCR with a Tesseract
fallback), non-text UI candidate detection, bounding-box annotation, text
grounding, screenshot-to-control coordinate mapping, dry-run-first desktop
control with drag support, per-run recording, and 165 unit tests at 88% coverage.

Progress and project deliverables will be updated throughout the development
process.
