# Multimodal Desktop GUI Agent

### Week 6 system v2.1.0 — validation pending

Version `2.1.0` is developed on `codex/week6-full-requirements`. The natural-language
CLI now defaults to `semantic_subtasks`: the model proposes an ordered sequence of
semantic objectives before input, then the runner plans and verifies their actions
under shared request, action and time budgets. Each objective's text checks are
frozen before its inputs; the caller's final checks remain unchanged and are
verified on a separate fresh frame. The status record reports verified objectives
against the current plan, including bounded revisions of unfinished objectives.

The current `week6.semantic_subtasks.v4` protocol separates two planning levels.
Semantic decomposition receives the complete caller request and original
OCR/element/window snapshot as text, with no attached image; screenshot files
and their paths remain preserved as provenance. The lower-level `TaskPlanner`
still receives fresh screenshots to ground each input. Later controls can be
hidden during initial planning; the action layer checks their actual availability
when reached. This does not establish coverage of purely graphical tasks.

The explicit persistent Tesseract configuration reuses the installed C API and
requires its DLL and pinned English/Chinese language resources. Legacy
`rolling_actions`, the Python API's default `observed_departure`, and handwritten
`--workflow` inputs remain available. See [Document/Week6](Document/Week6/README.md)
for configuration, records, compatibility and limits.

**The v2.1.0 real-window acceptance and final full regression are pending.** New
evidence is recorded separately under
[full_requirements_20261028](Document/Week6/results/full_requirements_20261028/).
The prior v2.0 normal/transient results and 3027-test full006 are historical;
they do not validate the new source. Failed runs and saved-image probes remain
separate from real desktop acceptance.
The v4 [235-case targeted CPU check](Document/Week6/results/full_requirements_20261028/semantic_sequential_prompt_001/cpu_validation_001/validation.json)
is neither a full regression result nor real-model or desktop acceptance.

Validation target (pending): `3184 tests, ruff clean`. The number is the current
collected suite size, not a claim that all tests passed; the complete regression
remains unverified.

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
