# Multimodal Desktop GUI Agent

### Week 6 system v2.1.0 — validation pending

Version `2.1.0` is developed on `codex/week6-full-requirements`. The natural-language
CLI defaults to `semantic_subtasks`: the model proposes at least two distinct
semantic objectives before input, then the runner plans and verifies their actions
under shared request, action and time budgets. Each objective's text checks are
fixed before its inputs; the caller's final checks remain unchanged and are
verified on a separate fresh frame. The status record reports verified objectives
against the current plan, including bounded revisions of unfinished objectives.

The current `week6.semantic_subtasks.v6` protocol separates caller intent from
screen grounding. Semantic decomposition receives the complete caller request,
its final predicates, and sanitized logical completion/failure records; it receives
no current screen text, geometry or image. The same bounded output schema is
supplied as prompt text and as structured-output format. Original snapshots and
available pixels are archived separately. The lower-level `TaskPlanner` still
receives fresh screenshots to ground each input.

The model may propose intermediate literal postconditions as **untrusted
predictions**. Unlike v5, v6 does not restrict them to the caller's final-predicate
pools, require two caller-final predicates, or reject a caller-final literal merely
because it exceeds the model-goal limit of 200 characters. This is an explicit
[contract change](Document/Week6/results/full_requirements_20261028/semantic_predicted_postconditions_001/contract_change_manifest_001.json),
not a claim that all v5 source restrictions remain. Native instructions and checks
retain strict type, length and count limits; outer whitespace is rejected rather
than silently stripped. The runner does not invent or locally merge a model plan.

Every whole goal predicate must be unsatisfied on a fresh frame before its inputs.
Credit requires known successful input and trusted, fresh, scoped post-observation
evidence. Duplicate/replay guards, failed records, unknown-input stops and the
independent caller-final check remain. A currently visible button label alone
cannot earn progress credit; a genuinely new control can witness a stage, but its
appearance does not prove that every requested operation occurred. The existing
case-insensitive substring matcher remains: for example, `Saved` can match
`Not saved`. It does not understand negation, control roles or semantic causality.
Independent operation-event audits remain necessary for the real acceptance runs;
these limits are not claims of arbitrary-application success.

The action layer now checks observable contradictions between a supplied text
target and explicit element ID in the original and refreshed frames. It rejects
a conflict before input and replans within the same budget; it does not replace
the native ID. Unique visible OCR labels can use text-only targeting. Consistent
IDs can disambiguate repeated labels; an unlabelled visual target with no local
OCR evidence retains its existing grounding path. This is an evidence-consistency
check, not a general proof of a control's role.

Parent semantic status now receives child action and planning counts before the
child returns. Verified-goal counts still come only from the outer completion
ledger. Status elapsed time uses the monitor's own monotonic clock, separately
from execution-clock fixtures; observer overhead remains part of real wall time.
Updates are event-driven, so a long model wait can still be reported as stale.
Ordinary observer faults do not retry input; interrupts and process exit propagate.

Explicit unresolved forms such as `[not specified]`, `[unspecified]`, `[unknown]`,
`<TBD>`, `[未指定]` and `[待填写]` stop before observation or a model request.
Those reserved forms are also unsupported as intended literal content; this
conservative guard does not detect all natural-language ambiguity. Later controls
can be hidden initially, and the action layer checks their availability when
reached. Purely graphical tasks remain unverified.

The explicit persistent Tesseract configuration reuses the installed C API and
requires its DLL and pinned English/Chinese language resources. Legacy
`rolling_actions`, the Python API's default `observed_departure`, and handwritten
`--workflow` inputs remain available. See [Document/Week6](Document/Week6/README.md)
for configuration, records, compatibility and limits.

**The v6 real-window acceptance and final full regression are pending.** New
evidence is recorded separately under
[full_requirements_20261028](Document/Week6/results/full_requirements_20261028/).
The prior v2.0 normal/transient results and 3027-test full006 are historical;
they do not validate the new source. Failed runs and saved-image probes remain
separate from real desktop acceptance.
The historical v4 [235-case targeted CPU check](Document/Week6/results/full_requirements_20261028/semantic_sequential_prompt_001/cpu_validation_001/validation.json)
and [3184-case full002 regression](Document/Week6/results/full_requirements_20261028/full_validation_002/validation.json)
retain their original source binding. The initial v5
[285-case targeted check](Document/Week6/results/full_requirements_20261028/semantic_caller_predicates_001/cpu_validation_001/validation.json),
[3234-case full003 regression](Document/Week6/results/full_requirements_20261028/full_validation_003/validation.json),
and prompt revision `002`'s
[294-case targeted check](Document/Week6/results/full_requirements_20261028/semantic_goal_grouping_prompt_001/cpu_validation_002/validation.json)
also remain historical. The v5 source and failed normal008 records are archived
under the v6 change directory; they are not relabelled as v6 results.

The earlier v6 [313-case targeted check](Document/Week6/results/full_requirements_20261028/semantic_predicted_postconditions_001/cpu_validation_001/validation.json)
and 3262-case collection bind the pre-grounding-fix source. Normal009 stopped
before input on a bridge timeout; normal010 selected the wrong theme and earned
zero semantic-goal credit. Their raw records and negative audits are retained.

The previous [799-case combined CPU check](Document/Week6/results/full_requirements_20261028/grounding_consistency_001/root_combined_validation_002/validation.json)
passed on the grounding/monitor freeze. The later [352-case semantic CPU check](Document/Week6/results/full_requirements_20261028/semantic_repair_feedback_001/root_semantic_validation_002/validation.json)
passed on the first typed-feedback freeze. Normal011 and normal012 each
stopped before input on duplicate goal postconditions. Normal012 received
the fixed system reminders and still produced no accepted correction.
Their raw replies and negative audits remain unchanged.

The earlier conversation-repair experiment represented bounded correction as explicit chat turns.
It preserves the first caller packet, carries only the immediately previous
bounded native reply as an untrusted assistant turn, and ends with fixed
repair instructions plus validation data in a user packet. An oversized
reply is archived in full and omitted from the assistant turn, never
truncated or replaced with a fabricated reply. Exact message arrays are
saved and independently reconstructed. The caller, output schema, native
verification guards and global budgets are unchanged; no local goal merge
or additional model is used. Normal013 did not achieve live acceptance on that experiment.

The [838-case action-output check](Document/Week6/results/full_requirements_20261028/action_output_contract_001/root_affected_validation_002/validation.json)
passed on the preceding source freeze. Its retained first check had one
legacy interface assertion failure; that assertion was separately migrated
without changing its runtime action or verification checks. Normal014
stopped before input on malformed flat action replies. After the format
change, normal015 produced valid TaskPlan objects but paired the requested
label with a different visible element ID three times; the existing guard
refused all inputs. Neither case completed any semantic objective.

The preceding snapshot-binding candidate guided native text/ID pairs;
it did not grant execution or semantic-goal credit. Its [1026-case check](Document/Week6/results/full_requirements_20261028/target_bindings_001/root_affected_validation_002/validation.json)
passed with 270 unchanged code pins and bound development dependencies.
The retained first check had 1,025 passing cases and one exact-key
context assertion failure. That assertion was migrated separately,
preserving its original complete-observation checks. The earlier
838-case result above remains historical. No counts are pooled.

Normal016 ended with task exit 1 after 438.469 seconds: three inputs,
six planning calls and six HTTP requests, with zero unknown inputs.
The prepare objective completed (one of two goals), and a review input
was independently recorded. No commit input occurred. The saved pixels
show the commit button, while the fresh OCR candidates omitted its
literal label, leaving this text-only target unresolved. The reason
for that omission, including any cursor contribution, has not been
isolated. Partial goal credit is not overall acceptance. See the
[retained negative audit](Document/Week6/results/full_requirements_20261028/review/negative_normal016_audit_001.json).

The preceding fixed-guidance candidate passed its [1063-case affected CPU check](Document/Week6/results/full_requirements_20261028/visual_target_guidance_001/root_affected_validation_001/validation.json).
Normal017 still ended with actual task exit 1 after 456.985 seconds.
It made three native inputs and completed one of two objectives. The fixed
policy and recovery feedback reached the model, but two text-only commit
plans still had no matching OCR target. Neither was dispatched. Its two
ignored second-click events remain in the [independent negative audit](Document/Week6/results/full_requirements_20261028/review/negative_normal017_audit_001.json).
There is no established truncation or cursor-causation explanation.

The current `gui_agent.target_binding_schema.v2` candidate closes the open
text-only generation alternative. It permits observed full labels and a
bounded, snapshot-derived set of canonical word spans and neighbouring word
runs. Query sources, limits and observed truncation are retained per attempt.
This deliberately narrows generation; the unchanged runtime still accepts
its wider direct text/substring/word-run interface. Current ID-only targets,
anonymous visual IDs and non-point actions remain available. No local code
chooses a replacement ID or edits a native action. Both query aliases and
both ID aliases are constrained. The endpoint remains `strict:false`.

The [current affected CPU check](Document/Week6/results/full_requirements_20261028/observed_query_constraints_001/root_affected_validation_001/validation.json)
passed 1123 tests with Ruff clean and 274 unchanged code pins.
Development dependency bytes remain bound separately. This is offline
evidence only. The 3620-case collection is not a full execution.
Normal018 completed both model objectives and the unchanged caller final:
six inputs, seven planning/HTTP calls and zero unknown inputs in 629.359 seconds.
The independent application state passed, but overall acceptance and the actual
owner exit remained 1. Three application events occurred 2.781, 4.019 and 2.424
milliseconds after the recorded click API return. The immediately following
fresh-observation completion time was not sampled and remains missing. Its
[negative audit](Document/Week6/results/full_requirements_20261028/review/negative_normal018_audit_001.json)
retains all original intervals. Two caption/no-effect inputs are retained; the
final commit target had OCR support, so this case does not prove targeting when
OCR omits a control.

For fresh normal019 and transient001, the predeclared
[input-operation protocol](Document/Week6/results/full_requirements_20261028/receipt_envelope_001/protocol_001.md)
records the click return separately and ends each operation at the exact return
of its immediate same-window fresh observation. There is no added timing grace,
input, model request or budget. The original strict event-attribution comparison
is unchanged. The private record checker passed 15 synthetic boundary tests and
both its expected accepting/rejecting CLI checks; these are not desktop results.
All 274 tested code pins are unchanged. Original normal001 through018 records
remain separate and unaccepted; old timings are not extended or pooled.

Normal019 then passed both independent audits and Root's operation/semantic/
mid-run reviews on that 274-file source: five inputs, six planning/HTTP calls,
zero unknown inputs and real task exit0 in 667.782 seconds. Its status-caption
noop and final advisory step-prediction failure are retained. See its
[accepted normal record](Document/Week6/results/full_requirements_20261028/review/normal019_root_acceptance_001.json).

Transient001 stopped before input: three identical native semantic replies put
the same required text in both positive and negative lists. The concrete local
error and previous raw answer were delivered in both repair conversations;
this was not missing feedback. It had three HTTP calls, zero inputs, no completed
goals and real task exit1 in 53.343 seconds, and never reached the application's
predeclared review rejection. Its [negative record](Document/Week6/results/full_requirements_20261028/review/negative_transient001_audit_001.json)
and all earlier outcomes remain separate.

The current candidate only clarifies AFTER-state positive/negative polarity in
two fixed semantic prompt constants. A justified negative condition remains
allowed; no local list rewrite, forced empty negative list, scene answer,
validation-rule change or extra request is introduced. The auditor bytes and
all other original production AST are unchanged. Its
[affected CPU check](Document/Week6/results/full_requirements_20261028/semantic_polarity_guidance_001/root_affected_validation_001/validation.json)
passed 1127 cases in 273.65 pytest seconds,
with Ruff clean, 275 stable code pins and separately bound development
dependencies. These are offline checks, not proof of real-model compliance.

Fresh normal020 and transient002, final full004, report and package QA remain
pending. Normal019 certifies its earlier source only. Both new cases use the
same predeclared operation receipt protocol, model digest, callers and budgets.
Validation target (pending): `3624 tests, ruff clean`.
This is the collected suite size, not a full regression or desktop acceptance.

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
