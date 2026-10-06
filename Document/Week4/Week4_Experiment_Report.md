# Week 4 Experiment Report: End-to-End Task Execution

## 1. Task objective

Wire the Week 2 perception and control modules and the Week 3 planner into one
loop that carries a natural-language instruction all the way to a verified desktop
result, expose it through a command line, and test it on five basic tasks.

The stage deliverable is the end-to-end prototype v1.0 plus the basic task test
report.

## 2. Module design

```
src/gui_agent/runtime/
├── schemas.py         ObservationSnapshot, ResolvedAction, TaskSpec, TaskRunResult
├── observation.py     ObservationService: capture + OCR + contours as one frame
├── action_adapter.py  PlanStep -> DesktopAction, or a refusal
├── message_regions.py observed header/composer/message crops and native-pixel mappings
├── verification.py    Verifier: check_step, check_task
├── runner.py          TaskRunner: the finite loop
├── recorder.py        TaskRecorder: per-step records that never overwrite
└── tasks.py           the five controlled cases and their success rules

scripts/week4_agent_cli.py    one task per invocation
scripts/week4_offline_demo.py the loop against scripted frames
configs/week4.yaml            Week 4 limits; Week 2/3 defaults untouched
```

Nothing in `runtime/` reimplements perception or control. It calls
`capture_monitor`, `create_ocr_engine`, `detect_ui_candidates`, `find_text`,
`screenshot_to_control` and `ActionExecutor.execute` exactly as they are.

### Decisions that shape the loop

**Plan, then step-by-step re-observation, with bounded recovery.** The model is
not called once per click. The initial plan is shown before execution, and each step
gets a fresh observation. If a plan cannot reach the task rule, the runner can
request another plan within the wall-clock budget. Every action still passes
the adapter resolution and executor boundary checks. `ExecutionOptions.max_planning_attempts` defaults
to 4. Recovery was introduced to address T02 failures with incomplete plans;
the retained successful run itself records one planning attempt and one model
request. Four attempts are a ceiling, not a promise that a slow model fits four
calls into the run budget.

**Every step is resolved against the current frame.** A plan made from one
screenshot is not a licence to click through a page that has since changed. Before
each action the target is looked up in the newest observation; an element id from
an earlier frame is refused rather than rebound.

**A stale element id had to become re-bindable.** A plan is written from one
frame and executed after another, so every element id in it is already stale by
the time the runner acts. The first version of the adapter refused any id that was
not in the current frame, which made element targeting unusable: the integration
test failed with `names no text target`. The spec allows a re-bind when the step
also names its target in words, so the adapter now re-locates by text and records
`re-bound from obs-0001-e001`. A bare stale id is still refused - there is nothing
to re-locate against, and guessing is worse than stopping.

**The prompt tells the model how to aim.** Week 3's system prompt named no
arguments and no element ids, so a plan could be valid JSON and still be
unresolvable against the frame. The prompt now shows the `{"element_id": ...}`
form, lists the argument each action needs, and asks for this platform's key
names. It was 1 199 characters when the 7B model truncated its own replies and 792
when it stopped; the current one is **979**, and a test pins it under 1 000.

**The model is called synchronously.** 10.3.7 warns that stopping a thread
which is waiting on a model does not cancel the HTTP request underneath it,
and that a reply arriving afterwards must be discarded. There is no such
thread here: `TaskPlanner.plan` runs inside the run, so a cancelled or timed
out run has nothing left in flight to be revived by. The per-request limit is
`model.timeout_seconds`; the run's own budget is separate.

**Screenshots changing is not success.** `ActionResult.success` only means
PyAutoGUI dispatched an event. Only `Verifier.check_task` can return `succeeded`,
and only when the task's own success rule matches the screen. `finish`, a run out
of steps and a changed screen all fail to satisfy it on their own.

## 3. Experiment environment

Initial integration used the Week 3 arrangement: code and tests on the MacBook
Air M2, with Ollama on the Windows node over HTTP. Later tests and real acceptance
also run locally on Windows 11 with Python 3.12.4 and a 2560x1600 primary display.
The retained summaries identify each run's environment and revision; the latest
T04 preparation separately verified a loaded 32768-token model context.

### OCR backend choice

Measured on the same 1470x956 frame, with the engine instance reused:

| Backend | Elements found | Time |
| --- | --- | --- |
| Tesseract | 4 | **241 ms** |
| PaddleOCR (medium) | 2 | 5 525 ms |

Tesseract is 23x faster on this machine, matching the Week 2 conclusion, so
`configs/week4.yaml` selects it.

The Windows review machine measures a different picture - Tesseract 738 ms against
PaddleOCR's 982 ms on a 2560x1600 frame - so this is a per-machine speed choice,
not a correctness one. It only became a free choice after the fix in section 7:
Tesseract reports one row per *word*, and a word list cannot match the multi-word
label a vision model asks for.

## 4. Implementation process

1. Define the observation, action, verification and result types before any loop.
2. Build the observation service, so the loop never assembles its own frames.
3. Build the action adapter and make it refuse: ambiguity, staleness and missing
   parameters are all errors.
4. Build the verifier, and make "no rule" inconclusive rather than passing.
5. Build the recorder so a long run cannot overwrite its own history.
6. Build the runner last, with every collaborator injected.
7. Wire the CLI, with dry run as the default.
8. Test the loop offline against scripted frames before letting it near a desktop.

## 5. Test results

| Machine | Result |
| --- | --- |
| Local Windows verification | **877 passed**, ruff clean |

Week 3 ended at 300 tests (counted on `d67de1f`). Week 4 adds 577: 512 in the fifteen files below, and 65 spread across the other suites -
`test_control_safety.py` 16, `test_model_mock.py` 14, `test_ocr.py` 7 (five of them
the Windows-only OCR workarounds), `test_config.py` 7 (a new file),
`test_plan_parser.py` 5, `test_model_config.py` 10, `test_documented_counts.py` 4
(a new file) and `test_recording.py` 2.

Those numbers are checked rather than maintained: `test_documented_counts.py`
collects the suite in a subprocess and asserts that the totals here and in the
README, and every row of the table below, still describe the code they sit next
to.

Every number here is a collected count taken on `d67de1f` in a worktree and on the
current commit, per file - not a running total. The previous version of this
sentence had drifted: it still described deltas from several rounds earlier and had
never counted `test_recording.py` at all.

**Half of that sentence was still unchecked, and the other half had already gone
wrong in translation.** The check covered the ten tabulated files and the number
they add up to, but not the suites named only in prose - their total, and each of
the eight deltas beside it, were maintained by hand. Adding five OCR tests moved
the list but not the total, and the Chinese version of this report sat there
saying 52 while the eight numbers next to it added up to 57. Nothing failed,
because nothing was looking. The prose half is now checked the same way the table
is: each named file is compared against its own count minus its Week 3 baseline,
the eight must add up to the total the sentence gives, and that total plus the
tabulated ten must equal the figure the sentence claims overall. The same pass
brought `Week4_Windows复核手册.md` under the sync tool - it restates the suite
total and the tabulated-file total, and a reviewer running its commands against a stale
expectation would read a correct tree as a broken one.

**The standard rule-based mock never types for any of the five cases.** All five task instructions
produce a single `click` step, because the intent rules match substrings and
"research" contains "search" - so T02's plan is a click, not a click-and-type. The
mock can produce `type_text` (an instruction like "Enter the query" does), and
nothing pinned that either; it does now. Nothing is wrong with the plan T02 gets:
the smoke double's job is a valid, resolvable plan, and the typing path is covered
by the scripted offline demo and by the executor's own tests. But someone reading
the five dry runs should not conclude that typing was exercised, and that is worth
one line here rather than an inference.

The same pass covered what happens when a local endpoint answers with something
unusable - a 200 with no choices, and a 200 with a blank message. Both raise a
`ModelError` that names the problem rather than letting an `IndexError` surface from
inside the SDK, and neither had ever been reached.

**The hand-back manual's own command did not produce its own number.** It listed
nine files and expected `168 passed`; running it verbatim gives **159**, because
`test_runtime_observation.py` was missing from the list. The reviewer would have run
the documented command, seen a number that did not match its stated expectation, and
had to work out which of the two was wrong. The command is corrected, and the
expectation now itemises those files so a mismatch points at the file rather than
at the total.

Every number that had drifted this round was a hand-maintained count. Three were
found: the two per-file counts in the table below - `test_runtime_runner` said 47
for 48 and `test_runtime_recording` said 10 for 11, both stale since the rounds that
added to them - and the troubleshooting guide's header, which said twenty-eight
while its tables held **forty-two**. That last one was bumped only in the rounds
where someone remembered, and drifted by fourteen without anything noticing.

It is checked by a test now rather than by attention: the test counts the guide's
table rows and asserts the header agrees. A count a reader is invited to trust
should be verified by something other than the person editing it.

Correcting those numbers introduced one of my own, which the clean-clone check
caught: the test added to guard the troubleshooting header also added a case to
`test_runtime_recording.py`, so the tabulated total moved from 168 to 169 while the
sentence still said 168. The two files are also easy to confuse - `test_recording.py`
is a Week 2 suite that gained 2 cases, `test_runtime_recording.py` is a Week 4 file
that gained 12 - and the first correction had them backwards. Both are now measured
rather than reasoned about.

**A failure with two causes, one of which the guide's fix could not touch.** Running
the manual's commands in a clean clone - the check that treats the document as
instructions rather than prose - stopped the CLI smoke and the T01 dry run with
`monitor_index 1 is out of range (available 1..0)`. The diagnostic guide attributed
that message to a sleeping or locked display. It appears just as often when the
process cannot see any screen at all, which on macOS means the terminal lacks
Screen Recording permission, and there the prescribed fix does nothing: an operator
would wake a display that was already awake and be no further along.

The tell is in the message and is measurable - only the aggregate pseudo-monitor at
index 0 exists and it reports `0x0`, so nothing is visible to the process - and the
row now names both causes, which fix belongs to which, and that a remote session
reports the same thing because there is no local display to capture. The run's own
behaviour needed no change: blocked, nothing dispatched, exit 2, no traceback.

| Test file | Covers |
| --- | --- |
| `test_action_adapter.py` | 77 cases: unique, ambiguous, missing and stale targets, split labels, current-frame unlabelled targets, safe pixel correspondence and constrained visual mapping to a detected candidate, missing/changed/duplicate/invalid candidates, parameter errors, coordinate boundaries, key whitelist, platform hotkeys, scaling and `finish` refusal |
| `test_runtime_runner.py` | 103 cases: the offline closed loop, dry runs, recording, confirmation gates and preconditions; the complete prepared T04 send flow; scoped candidate lists and exact typing arguments, translation and current composer checks, typed-state retry context, changed recipients/focus, missing/duplicate controls, drafts, ineffective sends, request deadlines, cumulative action budgets and refusal of repeated typing or another input after a send attempt |
| `test_runtime_verification.py` | 110 cases: generic rule matching and polling; isolated header/editor/message JSON schemas, strict outgoing/send state and exact marker/recipient attribution, source/crop geometry and candidate ids, draft rejection, hidden filename context, malformed types/fields, unavailable/uncertain evidence, derived-image cache integrity and window-crop fallback checks |
| `test_message_regions.py` | 42 cases: observed foreground region selection, translation, sidebar exclusion and OCR-row merging; ambiguous/missing composer or header refusal, bounded latest-first message candidates, duplicate borders, native-pixel crops and transforms, initial message exclusion, indicator halos, source-image validation and no source overwrite |
| `test_runtime_observation.py` | 55 cases: capture/observer foreground identity, bounds and stability, fail-closed unavailable metadata, current focus checks, frame geometry, OCR-failure records, labelled/unlabelled rendering, text-first ranking, the 100-label/200-contour cap and label overflow |
| `test_runtime_recording.py` | 15 cases: redaction, append-only steps, per-frame files, summary, and where the provenance comes from |
| `test_week4_cli.py` | 27 cases: argument errors, no `--yes`, dry-run default, summary always written, `--execute` refused without a terminal, the callback set an execute run hands over, `.env` loading, the flag/environment/YAML precedence, the numeric limits, and the warmup record the run copies in and warns about when it is missing |
| `test_week4_integration.py` | 14 cases: the loop against a real OpenAI-compatible test server, prompt/image transport, missing or mislabelled images; low-contrast input detection, nested send controls, OCR exclusion by candidate area and foreground prioritisation before the candidate cap |
| `test_week4_prompts.py` | 15 cases: the prompt fits its budget, describes element targeting and every action's arguments, and the user turn is the JSON envelope the planner actually sends |
| `test_week4_cases.py` | 15 cases: the invariants the five case definitions must hold - a machine-checkable rule, a declared precondition, a distinctive marker, a copy handed back by `get_case`, and the fresh marker a send-message run is given |
| `test_week4_demo.py` | 3 cases: the runnable demonstration, run - it completes with four dispatched actions and `verification: passed`, leaves the records a finished run leaves, and fails on purpose when asked |
| `test_week4_warmup.py` | 6 cases: the warmup 13.4 asks for and nothing provided - it probes text and then a real screenshot through the project's client, records cold/warm state, memory, per-request time, the classifier's verdict and the configured retries, and writes the record the operator keeps |
| `test_week4_evidence.py` | 15 cases: what the evidence collector copies, what it refuses to copy, which run `--latest` picks, and its error paths |
| `test_week4_preflight.py` | 8 cases: the checks that refuse a doomed run before it starts - a server context window below the prompt size, one that is big enough, no model loaded, the terminal wordings that were actually seen in a polluted frame, the element-cap mechanism the screen check has to explain, and the script's own flags |
| `test_runtime_processes.py` | 7 cases: application state as a precondition, which screen text cannot express - the per-platform browser and editor name groups, an unknown group being an error rather than an empty answer, case-insensitive matching with and without the platform suffix, a prefix that must not match, and the asymmetry when the process list cannot be read at all |

### The offline closed loop

`test_runtime_runner.py::test_the_full_loop_runs_offline_and_verifies` is the gate
that had to pass before any real action was allowed. It drives a scripted observer,
planner and executor through observe → plan → resolve → act → re-observe → verify
and asserts that the action was dispatched with `dry_run=False`, that the resolved
coordinate came from the second frame rather than the first, and that the run is
only `succeeded` because the task rule matched.

`test_a_changed_screen_that_does_not_match_the_rule_fails` covers the dangerous
near miss: the screen did change, the action did succeed, and the task still fails.

### What the suite refuses to let happen

A passing suite says the code works; it does not say the tests would notice if it
stopped. The difference was measured by deleting one load-bearing behaviour at a
time and running the whole suite against the deletion. Each row below was applied
on its own to a clean tree at `80f9835` and reverted; the third column names a test
that fails because of it, and the last row is why this pass exists.

| Behaviour removed | Where | What notices |
| --- | --- | --- |
| the task verifier returns `succeeded` with no rule to check | `verification.py` | `test_a_task_with_no_rule_is_inconclusive_never_passed` |
| the runner acts on the plan without re-observing | `runner.py` | `test_the_coordinate_comes_from_the_current_frame_not_the_plan`, and eight more |
| the step log opens for writing instead of appending | `recorder.py` | `test_steps_accumulate_instead_of_overwriting` |
| a bare element id from an earlier frame is accepted | `action_adapter.py` | `test_a_stale_id_without_a_text_target_is_still_refused` |
| a step after `finish` is allowed to run | `planning/schemas.py` | `test_a_step_after_finish_is_rejected` |
| duplicate step ids are accepted | `planning/schemas.py` | `test_a_duplicate_step_id_is_rejected` |
| the precondition guard is switched off | `runner.py` | `test_a_real_run_is_blocked_when_the_goal_already_holds` |
| a risky task skips its second confirmation | `runner.py` | `test_a_risky_task_is_confirmed_a_second_time_before_acting` |
| a dry run dispatches for real | `runner.py` | `test_a_dry_run_dispatches_nothing_and_is_not_a_success` |
| a task is called `succeeded` without its rule passing | `runner.py` | `test_a_changed_screen_that_does_not_match_the_rule_fails` |
| **the screenshot is dropped from the request to the model** | `models/openai_compatible.py` | **nothing, until this pass** - see §7 |

Every one of the three decisions in §2 is in that table, which is the point: they
are the properties worth breaking on purpose, and ten of the eleven were already
guarded by a test that fails for the right reason. The eleventh is the subject of
the finding in §7 - the week's defining behaviour, unverified at the only layer
where it means anything.

### A runnable demonstration

```bash
python scripts/week4_offline_demo.py
```

```
executor: click at (280,78)   [DISPATCHED]
executor: click at (280,122)   [DISPATCHED]
executor: type_text text='GUI agent research'   [DISPATCHED]
executor: key_press key='enter'   [DISPATCHED]

status      : succeeded
actions     : 4 dispatched
verification: passed - all success rules matched against the current screen
```

`--fail-at 2` stops after the second action with status `failed`, which is the
behaviour the runner is supposed to have.

This runs against scripted frames. It demonstrates the loop's mechanics; it is not
evidence that any real task succeeds.

## 6. Basic task results

**Four of the five cases have passed on the real Windows desktop, across five
successful runs.** Each credited run records both `status=succeeded` and
`execute=true`; T01 passed twice, and T02, T03 and T05 once each. T04 has no
successful run in the retained evidence.

| Case | Attempts | Successful runs | Recorded statuses | Successful evidence run ids |
| --- | --- | --- | --- | --- |
| T01 - open the browser | 6 | 2 | 2 succeeded, 3 failed, 1 blocked | `T01_20261004_140000`, `T01_20261004_213859` |
| T02 - search the web | 17 | 1 | 1 succeeded, 16 failed | `T02_20261004_210409` |
| T03 - open a specified file | 2 | 1 | 1 succeeded, 1 failed | `T03_20261004_171941` |
| T04 - send a message | 4 | 0 | 2 failed, 1 timed_out, 1 blocked | none; latest `T04_20261006_115446` was blocked with zero actions |
| T05 - close the application | 9 | 1 | 1 succeeded, 5 failed, 2 blocked, 1 timed_out | `T05_20261004_180657` |
| Total | 38 | 5 | 5 succeeded, 27 failed, 4 blocked, 2 timed_out | 4 of 5 cases |

All 39 retained `task_summary.json` files were read for this inventory: 38 have
`execute=true`, including the four blocked runs; one is a mock dry run. The
recorded-run success rate is **5 / 38 = 13.2 %**. Mean `execution_ms` is
**35 522.6 ms** over the 38 real attempts and **8 306.6 ms** over the five
successes; mean initial `planning_ms` is **128 785.9 ms** over all real attempts.
There are 36 dispatched action records, 34 with passed step checks. A step pass
does not establish task success. These aggregates retain failed/blocked attempts
and exclude uncollected historical references, offline probes and the dry run.

A dry run against the rule-based mock stops at the first step with
`no element matches 'desktop'`, which is correct: the mock's plan names elements
that are not on the screen, and the adapter refuses to guess. Mock plans are not
evidence of real capability and are not counted as such here.

`Week4_Basic_Task_Test_Report.md` records the retained attempts, timings and
verification rules. The complete prepared T04 test establishes the repaired
flow with a simulated chat and mock model. The real `115446` acceptance attempt
stopped before planning because its visual header/composer geometry was invalid;
its summary records one model request, zero actions and `status=blocked`.
Subsequent saved-image model checks do not establish a real T04 success.

## 7. Problems and handling

**Unlabelled controls were detected but omitted from the model's target list.**
The supplied Windows review of `T04_20261005_192417` reports four failed planning
attempts, no dispatched actions, and a frame with 60 selected elements but only
six text labels inside the WeChat window. That diagnostic run is external review
material, not a retained run directory in this checkout. OCR and contour detection
were already connected; `describe_elements()` discarded every text-free element
after selection, so the model was given no identity for an unlabelled control.

The renderer now exposes those detected candidates as `<unlabelled box>`, with
their frame-local `element_id`, confidence, centre and bounding box. It supplies
geometry without inventing names such as "input box" or "send button". A current
frame's id can resolve a text-free candidate through the existing coordinate
mapping and boundary checks. The placeholder is not a text label, and no freely
chosen model coordinate or stale bare id is accepted.

`execution.max_elements` is raised from 60 to 300, matching the Python defaults:
200 configured contour candidates plus 100 OCR slots. The supplied Windows OCR
measurement was 66 labels, so the OCR allowance provides 34 slots beyond that
frame. Text remains first, ordered by confidence, and even a frame with more
than 300 labels cannot have those labels displaced by contours. Fixtures test
the 100-label/200-contour capacity and label overflow; this allocation is a
bounded design allowance, not a new measurement of prompt size or live accuracy.
The larger prompt still needs a sufficient server context window.

**The follow-up completes anonymous target refresh without reusing coordinates.**
The initial visibility repair left planned anonymous ids stale after the required
pre-action capture. The runner now validates the captured foreground identity,
class, title, bounds, screen geometry and image evidence, then compares the old
target's pixels and context with actual current candidates. If appearance changed
(such as a send button becoming enabled), one constrained visual mapping request
can return only a unique current candidate id. It cannot change the approved
action, typed marker or recipient. The runner captures again after that request
and requires a unique pixel match before the unchanged adapter and executor map
and check the current centre. Unknown, absent or ambiguous evidence is refused;
no nearest box, old coordinate or freely chosen model coordinate is a fallback.

**The detector must retain the controls before refresh can work.** Offline replay
of the locally saved `T04_20261005_192417/obs-0005` screenshot found that the former
Canny thresholds missed the dark input border and the send contour was beyond the
200-candidate cap. Canny now uses 20/60, similar-size duplicate borders are removed
while small nested controls survive, and OCR exclusions use the candidate's own
area so a small label cannot erase a complete input. Fresh foreground-contained
contours precede desktop clutter before truncation. Replaying those saved pixels
with their recorded OCR exclusions retains the input at index 0 and send control
at index 50 within 200 candidates. This is a historical-image replay, not a new
capture or a measurement of the live model's accuracy.

**The real initial assessment confused sidebar text and old bubbles.**
`T04_20261006_115446` sent a full 2560x1600 screenshot to the visual assessor.
The returned title box lay in the sidebar area and its proposed composer lay in
the transcript; it also called the empty editor nonempty. Strict geometry
validation blocked the run before planning or input. A window-only image still
confused the same regions in a later saved-image probe, so merely cropping to the
window was insufficient.

`message_regions.py` now proposes a unique bottom-wide foreground contour, a
padded upper OCR-row header within its x span, and observed transcript contours.
The model receives independent header and composer source crops through strict
JSON schemas and must confirm readability and editor semantics. Initial context
does not include old message crops. The final message request receives native-pixel
candidate crops with nearby pixels for pending/failed-send indicators, returns
one observed candidate id, and transcribes the whole visible marker. Neither the
expected recipient nor the full expected marker is supplied, and image filenames
are excluded from model text so their timestamps cannot supply missing digits.
Actual candidate bounds inside message crops keep marker reading separate from
the nearby indicator halo. Source bounds,
unaltered candidate boxes, native-pixel transforms, source pixel hash, prompts and
raw/schema responses are recorded. No proposed region supplies a free click point.
Source and derived-image hashes validate cached assessments before reuse.

**T04 still requires strict sent-message evidence.** The exact fresh marker must
be in an outgoing bubble marked sent, below the authorised header and above an
empty composer. Drafts, wrong conversations, pending/failed/unavailable send states,
malformed JSON and uncertain evidence cannot pass. The window-crop fallback also
retains strict schema and original-screen geometry checks. Before planning, T04
requires the correct header and empty composer; before every input event, those
header pixels and current foreground identity must still match. Keyboard sending,
unapproved text and repeated typing are refused. Both confirmations, candidate
resolution, coordinate bounds and fail-safe remain required. The initial assessment
or refusal is preserved in `message_context.json`.

**The complete mock flow passes; the latest real attempt remains blocked.**
`test_t04_complete_send_uses_fresh_candidates_and_strict_message_evidence` runs
the real planner, runner, adapter, verifier and recorder with the `prepared-T04`
mock provider, synthetic moving-window screenshots and a recording action backend.
It confirms both risk prompts, click-input -> type-marker -> click-send, current
candidate centres, the changed send button's visual refresh and final sent evidence.
Related cases refuse wrong or changed recipients, changed focus, missing/duplicate
controls, a draft-only plan, an ineffective send and expired budgets. Action limits
are cumulative across planning passes. Once a send may have been dispatched, T04
stops further inputs in that plan, verifies the result and does not automatically
retry; repeated typing is also refused. These prepared checks do not change any
historical verdict. The latest inventory is 39 directories, 38 real attempts,
five successes and four passing cases, including the zero-action `115446` refusal.

The local record
`outputs/week4_t04_crop_validation_20261006_132746/verifier_context_isolated.json`
uses the saved `115446` source frame and the real `qwen2.5vl:7b` endpoint. Two
independent JSON-schema requests transcribed `文件传输助手` and identified an
empty composer, producing a passed recipient/editor context check with zero
desktop actions. This offline saved-image result is not a new real run, a sent
message or a promotion of the blocked summary. Planning, input, sending and final
verification still require their own real acceptance evidence.

**Planning needs the authorised editor's candidates, not every desktop target.**
The first saved-frame real-model plan consumed 21 806 prompt tokens, omitted
`arguments.text`, and named a sidebar contour as its send target. The production
T04 context now lists only observed current candidates inside the independently
authorised composer, supplies the exact typing arguments, and states whether the
marker has already been typed so a retry can plan only the remaining send.
The unchanged screenshot with that context produced click-editor, type the
complete 35-character marker, click the actual `发送` candidate in 31 904 ms,
using 5 841 prompt tokens. Adapter resolution confirmed the real candidate centres
without dispatch. The local record is `planner_scoped_response.json` in the same
probe directory. This is one saved-frame planning result, not a live input test.

The readable-title old-frame verifier probe, `verifier_sent_explicit_readability.json`,
made three real-model requests in 12 487 ms. It read the actual header, an empty
editor and the old outgoing bubble, then returned `failed` because the observed
marker did not exactly match the fresh run marker. The earlier refusal that
reported unreadable title evidence is retained separately. All these probes
record zero desktop actions; the strict evidence requirements were preserved.

The latest production verifier, including actual candidate geometry, hidden
image filenames and derived-image hash validation, was also tested on that saved
frame. `verifier_sent_final_production.json` records three requests in 74 585 ms
and the same failed exact-marker comparison, with zero desktop actions. Earlier
probe timings remain attributed to their own recorded versions and are not
replaced by this later measurement.

**The plan names elements that are not on screen.** Running a dry run against the
mock produced `no element matches 'desktop' in obs-0002`. This is the designed
behaviour - the adapter refuses rather than clicking a default position - but it
means a rule-based mock cannot demonstrate the loop. Handled by adding
`scripts/week4_offline_demo.py` with scripted frames, which exercises the whole
path without a model.

**The integration test proved the prompt carries the observation.** The first
version targeted an element id the model could not have known, and the test failed
in a way that turned out to be a design bug rather than a test bug: element ids
from the planning frame are always stale by the time the action runs. That is now
covered by two adapter tests and asserted in the integration test, which checks
that the step acted against `obs-0002` while the plan was written from `obs-0001`.

**The screen went to sleep mid-run.** Two cases returned
`capture failed: CaptureError: monitor_index 1 is out of range (available 1..0)`
once the display slept. The run reported `blocked` with that reason and dispatched
nothing, which is what should happen: a run without a screen is not a run.

**`RunSession` keys its directory by session id, not by prefix.** The first CLI
version passed `prefix=`, which does not exist. Sessions are now named
`<case>_<timestamp>` so the five cases stay distinguishable in the output tree.

**Recording skipped observations with no image.** `save_observation` returned early
when `image_path` was None, which discarded the element list - the part that makes
a coordinate traceable. It now always writes the JSON; only the image is optional.

**A blank coordinator field was not tested.** Adding the Week 4 tests surfaced that
`task_id`, `instruction`, `step_id` and `description` are validated for blank
values but had no test. Four cases were added in Week 3's follow-up.

**Word-level OCR made every real target unresolvable.** The Windows review found
that `TesseractOCREngine` emitted one element per word, so a step asking for
`Summary (required)` - a label plainly visible on the screen - stopped the run with
`no element matches`. The mock planner had hidden this: it only ever picks an
element that is already in the list, so it never asks for a phrase, and the Mac dry
runs passed while the real model could not resolve a single target. Fixed in
`merge_words_into_lines`: `image_to_data` already carries `block_num`, `par_num`
and `line_num` for every word, so the words are merged back into the line a reader
sees. A gap wider than 1.2 line heights splits them again, so two controls sharing
one text row ("File  Edit  View") do not become a single target whose centre is a
patch of empty pixels. Verified against the real Tesseract binary on rendered
labels, not only against mocked payloads.

**A phrase can still span two elements.** A plan may name two adjacent labels as
one target, which no single element's text matches. `ActionAdapter._locate` now
falls back to matching the query's tokens against a run of neighbouring elements -
neighbours in *reading* order, which is not the order the observation stores them
in, since `_select` ranks by confidence - and still refuses when more than one run
matches.

**The prompt did not fit the server's context window.** A 2560x1600 screenshot plus
the element list measured 7 517 tokens against Ollama's 4096 default, so planning
failed with HTTP 400 before any action was dispatched. The fix is a server setting
(`OLLAMA_CONTEXT_LENGTH=16384`), documented in `configs/week4.yaml`. The runner now
recognises that failure and appends the setting to the message instead of leaving
the raw provider error, and `configs/week4.yaml` states the requirement up front.

**Two of the five rules could be satisfied by doing nothing.** Found while checking
why T05's dry run reported that its rule *would read passed* on a screen where the
test application was not open. T05 only requires the marker text to be gone, so a
window that was never opened - or was minimised - satisfies it; T01 looks for
`http` and `search`, which any browser that was already running already carries.
Both are named in the manual's own table as results that must **not** count. A real
run now evaluates its rule against the untouched first frame and is `blocked` when
it already holds, with the reason recorded. `ExecutionOptions.require_preconditions`
- a field that until now was declared and never read - is what switches this on,
and only a task that declares preconditions is checked, since declaring them is how
a task says it assumes a starting state. Dry runs are exempt: they dispatch nothing
and their verdict is forced to inconclusive, so the guard would only stop the
pipeline check they exist to perform.

**The report's evidence would not have been in the repository.** `outputs/` is
untracked, so `outputs/week4/<case>_<timestamp>/task_summary.json` - the path the
basic task report points at - exists only on the machine that ran the task. A run
id that resolves nowhere is not traceability, and the hand-off asks for the
opposite. `scripts/week4_collect_evidence.py` now copies a finished run's summary
and step log into `Document/Week4/evidence/<run id>/`. It copies named text records
rather than the session directory on purpose: the screenshots and the per-frame
element lists are a picture of the whole desktop, and publishing the whole desktop
is not what "attach the evidence" should mean.

**The declared preconditions contradicted the guard.** T01's said "the browser may
already be running; that is recorded, not hidden", and T05's said only that the
application was open. Both were written before the guard existed, and both now
describe a setup that would be refused: a browser already running satisfies T01's
rule, and an application open *without* the sample file does not put T05's marker
on screen, so T05's rule is already true and the run would be void. The two cases
now declare the state their rules actually need. `test_week4_cases.py` holds the
invariants a case must satisfy to be runnable at all - a machine-checkable rule and
a declared precondition - because a case missing either is not "not yet measured",
it is unrunnable, and its row in the report would mean nothing.

**Two prompt rules were wrong, and one "fix" went to a function nobody called.**
The system prompt said "keep every string under 60 characters", which 8.2.3 rules
out - it invites the model to abbreviate the very marker the task is verified
against. It limits `description` and `summary` now and says to copy text exactly.
8.1.7 asks for the execution limits as well as the step cap, so the planner's
context carries "at most N actions and T s for the whole task" rather than letting
the model discover the budget by having its plan refused. Both of those are live.

The third finding was not. `build_user_prompt` cut the serialized context at 4 000
characters, which would have been a 8.2.10 violation - and it was fixed, and
described here, before anyone checked whether the function was on the path. It was
not: the planner goes through `ModelRequest.to_messages`, which sends a compact
JSON envelope, and nothing in `src/` or `scripts/` ever called `build_user_prompt`.
The live path never truncated anything; the list is capped structurally by
`execution.max_elements`. The function and its twelve tests are gone, its helper
with them, and the tests now pin the envelope the planner actually sends - through
the planner's own call, so the two cannot drift apart again. A stale comment in the
mock that named the dead function is corrected too.

The lesson is the one this report keeps meeting from a new angle: the fix was
tested and the tests passed, which felt like evidence. What was missing was the
question the tests could not ask - whether the code being tested is ever reached.

**A wrong-typed argument escaped as a traceback.** `int(scroll_amount)` and
`float(duration)` raised bare `ValueError`. The runner catches
`ActionResolutionError`, so a malformed plan would have left the run as an
unhandled exception instead of a recorded failure - and 8.3.4 asks for parameters
that are complete *and* correctly typed. Both conversions now raise the module's
own error. `test_action_adapter.py` also gained the two cases 9.4 lists and the
first pass had missed: an out-of-range coordinate and a wrong-typed argument.

**The hand-off's own step vocabulary was rejected.** 10.5.3 names
`pending_runtime_resolution` for a step whose page is not open yet. This runner
resolves every step against a fresh frame, so it never writes that value - but the
schema's `Literal` did not contain it, which means a model that followed the
hand-off would have had its entire plan rejected as invalid. The value is accepted
now; an invented one still is not.

**A screen with no readable text failed silently.** The desktop went to sleep
mid-round: capture still returned a 1920x1080 frame and contour detection still
found 60 boxes, but OCR returned no text at all, so every text target became
unresolvable and the run stopped with a bare `no element matches '@'`. The run now
records "the first frame had no readable text" as a note. Diagnosing that by hand
took longer than writing the note did. In the same area, the mock no longer aims
at OCR fragments like `@` or `HO`: they are gone by the next frame, so a dry run
that dies on one says nothing about the pipeline.

**The second confirmation for a risky task was never wired up.** The CLI built
`high_risk_fn` and never passed it to the runner, and `TaskRunner.run` had no
parameter to receive it - so T04, the one case that really sends a message, ran on a
single confirmation while the CLI's own `--help`, the usage guide and the
troubleshooting guide all stated it got a second, separate one. 8.3.6 and 12.2.6
ask for it by name. This is the defect pattern worth naming: the claim was in three
places and the code was in none, and nothing could catch it because no test asked
whether the prompt was ever reached - the CLI tests covered argument errors and the
dry-run default, and none of them ran with `--execute`.

`run()` now takes `high_risk_confirm` and calls it for `medium` and `high` tasks,
after the first confirmation and before the countdown; the risk policy sits in the
runner so a caller cannot forget it. The CLI hands its callbacks over as a splatted
dict, so a callback the runner does not accept is a `TypeError` instead of a safety
prompt that silently never fires, and `_execute_callbacks` is a named function so
the wiring itself has a test. The prompt also prints the text the plan will
actually type, which is what 12.2.6 asks to be shown before a message goes out.
The same pass implemented 12.2.5: `--execute` without a terminal returns `blocked`
(exit 2) rather than reading the missing answer as consent.

**The run summary did not carry enough to read it elsewhere.** 14.2 lists what a
summary must contain - commit, platform, Python version, screenshot and control
geometry, start and finish times, planning attempts, model requests, the evidence
directory, the failing step - and ours had 13 fields, most of that list absent. The
basic task report's environment table asks the operator for the same facts, so both
were being filled in from memory, which is how a report ends up quoting a revision
the run did not use. They are recorded now. `planning_attempts` and
`model_requests` are deliberately two counters: 14.2.1 asks for that, and the
distinction is real - one plan can cost several transport attempts, and the request
count is taken in `ModelClient._with_retries`, the single place a request leaves
the process, so retries are included. Geometry is worth its line for a different
reason: a click that landed wrong cannot be re-read afterwards without knowing
which scale it was mapped through.

**A refused run left an empty session behind.** The non-interactive refusal added
for 12.2.5 was checked *after* the run directory had been created, so every refused
`--execute` left an orphan `outputs/week4/<case>_<timestamp>/` with no summary in
it. Nothing in the repository could notice - `outputs/` is not tracked - but the
evidence collector's `--latest` sorts by name, so a later orphan shadows the last
run that actually produced evidence, and the operator is told "the run did not
finish" about a run they never started. It surfaced by cloning the repository and
running the whole suite inside the clone: a clean checkout should have no `outputs/`
at all, and it had one. The check now runs before the directory is created, and
`latest_session` prefers a session that has a summary over one that merely sorts
later.

**`GUI_AGENT_MODEL` changed nothing at all.** 13.1.1 asks for the order "explicit
flag, then `GUI_AGENT_*`, then YAML, then default", and the environment leg was
dead: `ModelConfig.model_name` always has a value, `create_model_client` always
passes it, and the client's own `model_name or env.get(GUI_AGENT_MODEL)` fallback
therefore never ran. Following `.env.example` produced a client on the wrong model
with no error. The CLI resolves the order itself now, and `_apply_environment` is a
named function so the order has a test rather than a comment.

The same file told the operator to copy it to `.env`, and nothing read `.env`
either. 13.1.2 asks the CLI to load it with `override=False`, so the CLI does -
shell first, file second, because a value exported in the shell is a decision made
later than the file. The Week 2 and Week 3 entry points still do not read it, and
`.env.example` now says which is which instead of implying that both work.

**The run limits were configured under a name the hand-off does not use.** 13.2
proposes an `execution:` section holding the run-layer limits; this had them under
`agent:`. Renamed, since a reader with the hand-off open should not have to work
out that the two are the same thing. `max_wait_seconds` was added with it and
actually wired into the action adapter - the wait bound had been the literal `10`
in `_resolve_wait`, so the configured value could not have had any effect.
Re-planning is now controlled by `max_planning_attempts`, default 4, and the
runner checks it and the wall-clock budget on each recovery pass. Every action
still passes the adapter resolution and executor boundary checks.

**Three of the plan-validation rules were not enforced.** 8.3 lists what a plan
has to satisfy before it may run; three of them were missing, and two of those
mattered. A step *after* `finish` was executed, because `is_executable` looks only
at the verb - so a plan that said "stop" and then added one more click had that
click dispatched. A plan with a non-empty `errors` list ran anyway, which reads the
model's "I could not work this out" as "go ahead"; the field existed and nothing
read it. Duplicate `step_id`s were accepted, which leaves two steps sharing one
address in the record. All three are refused at parse time now, except `errors`,
which is a decision for the runner rather than a malformed plan.

**Two runs inside one second shared a directory.** 14.1 warns about a name made
only of a seconds-precision timestamp, and `RunSession.create` called `mkdir(...,
exist_ok=True)`: the second run silently reused the first's directory, appending to
its step log and overwriting its summary. That is not a theoretical race - a run
that is blocked before it captures anything finishes in well under a second, and
the five-case loop is exactly the kind of thing that produces them back to back.
`create` now opens a fresh directory, suffixing `_2`, `_3` and keeping the session
id equal to the directory name so the recorder's run id cannot disagree with it.

**T04 could only ever be run once.** 15.4 asks for a fresh message identifier on
every run, and the case was defined with a fixed one. That makes the case
single-use in two ways at once: after one attempt the previous message is still in
the conversation, so the rule - "the marker is on screen" - is already satisfied,
and the precondition that requires no earlier message carrying that marker refuses
the retry. The marker is minted per run now and printed as `marker` so the operator
knows what this run is looking for. The preconditions were left alone because they
say "this marker" rather than naming the literal; an earlier draft rewrote them and
that code was a no-op, which is the kind of thing that reads as if it works.

**The run summary had one timing number, and it was wrong for blocked runs.**
16.5.4 asks for the phases separately: the confirmation prompt is a person reading
the plan, and counting it as system time makes a slow operator look like a slow
model. `execution_ms` is now the primary measure - from the gate to the final
verdict - with `planning_ms`, `confirmation_ms` and the whole-run `elapsed_ms`
beside it, and the three phases add up to the whole by construction.

Splitting them exposed a worse problem underneath. `_blocked` passed
`self.clock()` as the run's start time, so **every** blocked run recorded
`elapsed_ms` of 0.0 - including one that spent twenty seconds inside a model call
before giving up. That is the case the Windows review actually hit: its T01
reported a 400 after a long wait, and the summary it left behind said the run took
no time at all.

**Driving a real terminal found two more wrong numbers.** The unit tests pass a
`confirm=` callback straight to the runner, so nothing had ever exercised the CLI's
own interaction - and the first time one did, through a pty, the cancelled path
turned out to report `planning_attempts: 0` although the notes said a plan had been
made, and to put the operator's reading time into `execution_ms`. The second is the
misattribution 16.5.4 exists to prevent: a hesitant person looking like a slow
model. Both are fixed. The same run now records one planning attempt, the wait in
`confirmation_ms`, and `execution_ms` at zero - which is right, because a refused
plan costs the system nothing.

That exercise also checked the second confirmation. A T04 driven through a pty
shows two prompts, and declining the second cancels with nothing dispatched.
Before that check the path had not been exercised end to end, which is how the
missing wiring survived.

**A failing endpoint was attempted three times, and the record said one.** The
OpenAI SDK retries twice on its own, and `max_retries` was never passed to it - so
`model.max_retries: 0`, which 10.3.4 asks for on the first real loop, did not mean
"one attempt". Measured against a deliberately hanging endpoint with
`timeout_seconds: 3`: the call took twelve seconds, and the summary recorded one
model request. That is the multi-layer stacking the spec warns about, and it made
`model_requests` an undercount rather than a measurement - the one thing a
provenance field must not be. The SDK is now given our retry count, the same call
takes five seconds, and the test suite itself got eight seconds faster because the
tests that exercise failing endpoints stopped retrying three times each.

**Planning that failed was reported as execution.** The `planned` stamp was set
only after a plan succeeded, so a model call that failed after two seconds reported
`planning_ms: 0` and put those seconds into `execution_ms` - a blocked run looking
like one that had been busy acting. This is the shape the Windows review hit
exactly: its 400 arrived after a long wait. Both paths were verified end to end
through the CLI against a stand-in endpoint that reproduces the review's error
verbatim: the 400 now reports `planning_ms: 2087.5` with `execution_ms: 0.0`, and a
hanging endpoint reports `APITimeoutError` as its own class rather than as a
generic failure.

**A screen that went away during the final verdict ended the run as a traceback.**
Measuring coverage rather than guessing at what was untested showed that the whole
of `ObservationService.observe()` had never been executed by the suite - it needs a
screen, and the suite is deliberately not allowed to take one. 16.1 asks for
*prepared* screenshots instead, and with a prepared frame plus a stubbed backend
the method runs for real: ids, geometry, the OCR-failure record, the element cap.
`observe()` went from 54% covered to 99% and the runtime total from 90% to 98%,
with `runner.py`, `verification.py`, `recorder.py` and `tasks.py` at 100%.

Writing those tests is what found the crash. `check_task_with_polling` is the one
observer call the runner does not wrap, so a display that slept between the last
action and the verdict let the exception out of `run()` entirely - and the CLI
catches only `KeyboardInterrupt`, so the operator got a traceback instead of a
result. Sleeping displays are the single most common environmental event in this
project; this was the one place it was still fatal. The polling loop now reports
`inconclusive` with the reason, which is the honest verdict for a screen nobody can
look at.

The same pass covered a whole action type that had no success path under test
(`drag` was only ever exercised by its error branches) and the provenance
override that lets a checkout without git metadata record a commit.

**A constant that knew where the sample file goes, and told nobody.**
`SAMPLE_DIRECTORY` held the per-platform path for T03's test file and was read by
nothing - not the code, not the tests, not a document. T03's precondition said "the
week4 test folder", which is a description rather than an instruction: the operator
still had to work out where to put the file. The precondition names the actual
folder now, which is what the constant was for. A sweep for the same pattern across
the runtime, planning and models packages found no others; the four schema fields
that are written but never read are records - they go into the JSON evidence for a
person to read, which is what a record is for.

That sweep looked at constants. Running the same question over every *function* -
is it referenced anywhere in `src/` or `scripts/`? - turned up
`provenance.os_description`, written in this week's provenance pass and never
wired, because the field it was for did not exist. It is wired now: the summary
carries `os_version`, and it reports `platform.platform()` rather than the OS family
alone, since `platform.release()` answers "10" on Windows 11 and the build number is
what actually identifies the machine. The basic task report's environment table now
says which summary field each of its rows comes from, so the operator checks rather
than remembers.

The same sweep lists a handful of Week 2 and Week 3 helpers that no production path
calls either - `capture_fullscreen`, `crop_region`, `sharpen_image`,
`elements_to_payload`, `update_screen`, `planner.system_prompt`. They are that
week's API surface, not this week's, so they are recorded here rather than pruned:
removing another week's public functions during Week 4 is the kind of tidy-up that
looks harmless and is not.

**A hand-rolled `.env` parser, next to a declared library that does the same job.**
`python-dotenv` has been in `requirements.txt` all along and nothing imported it;
when `.env` loading was added to the CLI in this week, it was written from scratch
instead. The CLI uses `load_dotenv(path, override=False)` now - one call, the same
guarantee, and one fewer thing to maintain. The behaviour it was written for is
unchanged and still checked end to end: a value exported in the shell beats the
file.

Asking the same reachability question of the dependency list turned up four more
declarations nothing imports - `anthropic`, `httpx` and `pynput` in the base
requirements, and `datasets` and `langchain` in the Week 3 set (`langchain-openai`
is imported; `langchain` itself is not). They are other weeks' dependency
decisions and are recorded here rather than pruned, for the same reason as the
functions above.

One message was outright wrong and is fixed: the OpenAI backend's "install it with
`pip install -r requirements-agent.txt`" named a file that does not list `openai`.
It arrives there only transitively, through `langchain-openai`, so the instruction
happened to work. It names `requirements.txt` now. `RunnerError`, a class defined in
the runner, never raised, never caught and not exported, went with it.

**Half the executor's vocabulary had never been dispatched.** The runtime tests use
a fake executor and the control tests stopped at `click` and `drag`, so the real
`ActionExecutor` had never dispatched `type_text`, `key_press`, `hotkey`, `scroll`
or `wait` - and T02 is a type-then-submit task. All five are covered now, along
with `move`, `double_click` and `right_click`, the countdown ordering and the
action delay.

The same pass reached the layer underneath, which had no test at all:
`PyAutoGUIBackend`, the wrapper that actually drives the machine. It is testable
without moving a mouse - the constructor imports pyautogui, so a stub module in
`sys.modules` takes its place and every primitive can be checked for the call it
forwards to. That matters because a wrapper typo is invisible everywhere except on
the real desktop: a `double_click` dispatched as a single click looks like a
sluggish application, not a bug. `control/actions.py` went from 55% to 100% and the
control package from 81% to 95%.

One thing the wrapper's tests document rather than fix: `_release_modifiers` tries
nine key names after every action, and three of them - `control`, `cmd`, `super` -
are not in PyAutoGUI's key list on any platform, so those releases always raise and
are always swallowed. That is deliberate and correct (the remaining six still cover
a genuinely stuck key, and the release also runs when the action itself failed),
but an operator reading a log full of caught exceptions should know it is
expected.

**All five exit codes are now verified end to end.** The table in 12.2 had only
ever been read, not exercised. Driving the CLI against stand-in endpoints covered
the rest: a successful answer exits 0; a plan naming a target that is not on screen
exits 1 with `failed`; the context overflow and an exhausted budget exit 2; a
budget that expires between steps exits 3 with `timed_out`; and answering `n` at
the confirmation exits 130. The stand-in reproduces the review's own 400 verbatim,
so the path its T01 took is now covered by something repeatable.

**Two numeric flags did not behave as their help text reads.** `--task-timeout -5`
reached `ExecutionOptions`, whose pydantic error escaped as a traceback and exited
1 - the code the table reserves for "the run failed", so an operator would read a
typing mistake as a task that had been attempted. And `--max-actions 0` was
silently dropped, because the assignment tested truthiness rather than `is not
None`; an operator asking for the most restrictive setting got the default twenty
instead. Both are validated at parse time now, which puts them where the table says
they belong: exit 2, with the offending value in the message. The header also
printed a sub-second budget as `0s`, which read as "no budget at all".

**A table was allowed to contradict itself.** The count sentence in §5 had been
guarded for two rounds - a test collects the suite and checks it - but only for the
half it tabulates. The other half names eight suites in prose, states a total for
them and a delta for each, and every one of those numbers was still maintained by
hand. Adding five OCR tests moved the list and not the total: the Chinese version
of that table read 52 in its own total column while the eight numbers beside it
added up to 57, and the suite stayed green because it was only looking at the
other half of the sentence. The check now derives all nine numbers and asserts both
sums, and `Week4_Windows复核手册.md` is under the same tool - a reviewer running its
commands against a stale `expected N passed` reads a correct tree as a broken one.
This is the same failure as the dead `build_user_prompt`, one layer out: the check
existed and was pointed at less than the claim.

**The image was assumed to be arriving.** §8.1 warns in as many words that a
screenshot path in the prompt is not the same as the model seeing the picture, and
says acceptance should look at the request. Nothing did. The encoder has thorough
tests - format, size limit, missing file, empty file, five MIME types - but they
call `to_vision_messages` themselves; the runner test asserts the planner received
a *path*; the prompt test asserts the planner passed a *path* on. Every one of them
would keep passing if the helper were never called in production.

That is not a hypothetical, so it was measured rather than argued: the image was
removed from every outgoing request and the suite ran. **All of it passed.** The
single defining property of the week - the model is shown the screen - was
unverified at the only layer where it means anything, and the tests that looked
like they covered it were covering the two hops of Python around it.

Four tests now read the body that a real server received over a real socket. The
screenshot arrives as base64 image data and decodes to the bytes of the file the
observation captured, not merely to something image-shaped; it is the *first*
frame's image, matching the element ids listed in that same request, so image and
element list describe one screen; a frame whose capture produced no file sends a
text-only request rather than a malformed block; and a missing or mislabelled file
blocks the run before any request leaves the machine, with the filename in the
note - because a model asked to plan from nothing returns a confident plan about a
screen it never saw. Re-running the same deletion against the new tests fails four
of them.

**A P0 requirement with no implementation, no test and no record.** 13.4 asks for
four things before the week's real tasks: warm the model up, keep that time out of
a warm-start task's timing, prove the vision path rather than only the text path,
and record cold/warm state, memory, the request time, a failure classification and
the retry count. There was no warmup of any kind. The manual told the operator to
run `ollama run qwen2.5vl:7b "ok"` - the text-only warmup that the same section
says is not enough - and nothing recorded what it cost.

Nothing failed, and nothing could have: this is the one class of gap a test suite
does not see, because a requirement that was never implemented has no code to be
uncovered and no assertion to be wrong. It was found by reading the acceptance
list against the tree rather than by running anything, which is also how the
`require_preconditions` wiring and the missing `test_runtime_observation.py` in the
manual's own command had been found. Three of the week's P0 items were missing
pieces that only a line-by-line check against the hand-back notes surfaced.

The cost of not having it was concrete rather than theoretical: the first T01 run
on the Windows box would have paid for loading the weights and warming the vision
encoder, and recorded it as the plan's own latency - on a machine where that has
been measured at 8-10 s cold and about 72 s with memory tight. The first data point
of the week's real evidence would have been wrong, and wrong in the direction that
looks like a model problem.

`scripts/week4_warmup.py` sends a text probe and then a real screenshot through the
project's own client, because 13.4.6 is explicit that a long-timeout probe
succeeding says nothing about the client the tasks use; labels the session's first
request as the cold one; and writes a record with every field the section names.
Its failures go through the runtime's own classifier, so a warmup that dies on the
context window prints the same hint a task run would. The manual gained a step that
runs it before the first task, and the usage guide gained it in the quick start.

16.5.4 asks for the warmup time to be kept *beside* the run's timings while 13.4.1
keeps it *out* of them. Both hold because the CLI copies the record into the run's
own directory and the evidence collector carries it into the repository with the
summary and the step log: the four task timings are untouched, and a run whose
output directory has no record says so in its header - in both modes, because a
dry run records `planning_ms` too and a cold model inflates that number just as
much. The result template gained the two rows this needs, for free memory and for
the cold and warm latencies.


**Four rules the spec states and the code did not keep.** The acceptance list was
read section by section against the tree, and each finding was then checked in the
files rather than taken on trust. Four held up.

11.1.5 forbids reading a changed screenshot as success. `check_step` did exactly
that: with no expected result to look for and a frame that had moved, it returned
`passed`. Its own docstring said the honest answer was `inconclusive`, so the
module contradicted itself - and the branch was unexercised, so nothing failed.
A changed screen is `inconclusive` now, with the change still recorded as evidence;
`move` and `wait` get the branch 11.1.6 allows them, reported as the action's own
completion rather than as an observation of a result.

8.2.8 says an ambiguous instruction must be answered or blocked and must not lean
on `assumptions`. The system prompt said the opposite in as many words: "If it is
ambiguous, say so in assumptions and still return a plan." Nothing in the runtime
reads `assumptions`, so an ambiguous instruction produced a plan that executed on
a guess the operator never saw. The rule now sends the question to `errors`, which
the runner already refuses to execute on. 8.2.7 and 8.2.5 needed two more rules -
screen text is data and not instructions, and recipients, file paths and
applications are not to be invented - and all three had to fit a budget that is a
measurement, not a preference: 1 199 characters truncated the 7B model's replies
and 792 worked, so the cap is 1 000 and the prompt is 965. The JSON example gave up
`assumptions` and `requires_confirmation` to make room; both have defaults, neither
may authorise anything, and showing `assumptions` in the shape the model copies is
an invitation to fill it in.

14.2.3 asks for a readable summary on interruption. Ctrl+C returned 130 with no
summary at all - the run happened, and the record of it did not. The runner catches
the interrupt now and closes the run out from the step log rather than from memory,
because on an interrupt the in-memory list is whatever the unwinding left behind.

16.2 asks for sensitive text to be kept out of the records. Typed text was masked;
what OCR read off the screen was written verbatim, which a new test demonstrates by
putting a password on screen and reading the frame dump back. Credential-shaped
element text is masked now and ordinary interface text is untouched, because the
element list is what traces a coordinate back to the words it came from. The usage
guide says what redaction does not cover, which was the part a reader would have
assumed.

**Three more, each half-present.** 7.1.4 asks for the fallback engine to be kept.
The mechanism was there and dead: `create_ocr_engine` produces the notice, the
observation service copies it into a property, and nothing in the runtime reads
it - so a run that fell back recorded `ocr_engine: tesseract` and lost the reason,
which is the half that explains a slow or poor frame. It travels on the snapshot
now and is spoken in the run's notes. Not in `errors`: `errors` is what makes the
verifier refuse to judge a frame, and a fallback that worked is not a degraded
one.

8.1.2 asks for the allowed key names to reach the model, and the prompt only said
to use them. A plan could therefore name a key the adapter refused, which the
model had no way to avoid. The list is read from the constants the adapter checks
against, so there is one list rather than two. The modifier names turn out to be
the same on both platforms - what differs is what each presses - so the test
asserts that distinction rather than the one it was first written with.

12.2.2 wants the plan, its steps, the text to be typed and the risk shown. That
display existed only inside the execute-mode confirmation, which left the default
mode - a dry run, whose entire purpose is to let the operator see the plan before
agreeing to it - as the one mode that showed none of it. A dry run now shows the
plan through an `on_plan` hook, which is not a confirmation: nothing is
dispatched, so nothing is being authorised.

**A mismatch was computed and never read.** 10.1.12 says to carry on when a step's
expectation holds and to stop when the result does not match it. An action *error*
stopped the run; a step verification of `failed` did not, so a step that plainly had
not done what it was for was recorded and the plan carried on with the next one -
typing the rest of a query into a window that never took focus. The value was
assigned to the step record and nothing ever looked at it again.

A real run stops on that now, and a dry run deliberately does not: nothing is
dispatched in a dry run, so an unchanged screen is the expected outcome rather than
a mismatch, and stopping there would truncate the one mode whose purpose is to walk
the whole plan and show it. The mismatch is still recorded either way.

**The configuration that produced a run was not in the run.** 14.1 names
`run_config.json` and nothing wrote it. The summary carried the model name and the
run's own limits, so a reader could not tell which `max_elements`, OCR engine,
verification timeouts or planning limits produced the frames in front of them - and
the test report's environment table asks for `timeout_seconds` "in the config the
run used", which is that file. It is written with every run now, credential-shaped
values masked by key name rather than trusted to be absent, because the record is
copied into the repository by the evidence collector.

**The five tasks would have failed on a fresh Windows machine for a reason no
document mentioned.** `configs/week4.yaml` selects `ocr.engine: tesseract`; step 1
of the hand-back manual installs `requirements-agent.txt` and
`requirements-dev.txt` and nothing else. `pytesseract` arrives with those - it is
the wrapper. The `tesseract` executable does not, and the manual never mentioned
it. Every observation on a machine without it reads zero labels, so every text
target fails to resolve and all five cases fail with a message about
localisation.

The failure was reported, which is why this is a documentation defect rather than
a silent one: the engine wraps it into an `OcrError`, the frame records
`ocr unavailable: Tesseract failed: tesseract is not installed or it's not in your
PATH`, and the run says the first frame had no readable text. But that last
sentence is the one the manual explains, and the manual explains it as a locked or
dark screen - so the operator would have woken a display that was already awake.
The run now repeats the observation's own errors above that line and says outright
that a missing binary reads the same way as a locked screen, step 1 installs the
binary, and the diagnostic row carries the distinction.

This is the second time a correct-looking failure message sent the operator to the
wrong cause; the first was the capture permission in §7. Both were found by
running the manual's own instructions rather than reading them.

**A test guard that was never real, and a returned value nobody read.** Two rounds
ago the warmup tests were given a subprocess environment containing
`GUI_AGENT_DISABLE_DOTENV=1`, on the assumption that it stopped the child from
reading a `.env`. It stops nothing: the variable appears in no file under `src/` or
`scripts/`. What actually keeps a real `.env` out is the working directory, since
both entry points load `./.env` relative to the process's cwd - and with the
children running in the repository root, a reviewer who had created `.env` as the
usage guide tells them to would have had it loaded. The tests passed anyway,
because every one of them passes `--base-url` explicitly and the flag wins, so the
guard was decorative and nothing failed.

The environment handed to those children was also built from nothing rather than
copied, which is not portable: Windows needs `SystemRoot` for socket setup, and
`PATH=/usr/bin:/bin` means nothing there. They now run in a temporary directory
with a copy of the ambient environment and the `GUI_AGENT_*` variables removed,
and both properties are demonstrated rather than asserted - a `.env` placed in the
child's directory is shown being read, and shown losing to the flag.

`load_environment` returns whether it read a file, and both callers discarded the
answer. Settings arrive from four places; when the endpoint turns out to be wrong,
"was my `.env` read at all?" is the first question, and the answer depends on where
the process was started. Both entry points now print which it was.

**Every run's record said it had no verification.** `_finish` is what writes
`task_summary.json`, and both callers attached the verification to the returned
object after it returned:

    result = self._finish(..., status)
    result.verification = verification

So the object the CLI printed from carried `passed`, and the file on disk - the one
the reviewer reads, the one the evidence collector copies into the repository, the
one the results table in `Week4_Basic_Task_Test_Report.md` has a column for - said
`"verification": null`, for every run this project has produced. The dry-run path
lost its verdict the same way. `_finish` takes the verification now, so it is
written before the summary is serialised.

It was found by running the hand-back manual's step 3 rather than reading it, and
by writing a test that reads the file instead of the returned object. That
distinction is the whole episode: a test asserting on `result.verification` would
have passed throughout.

**The demonstration had quietly stopped working.** Step 3 promises four dispatched
actions, `succeeded` and `verification: passed`. The demo was stopping at its own
second step: its scripted screens did not contain the words its scripted plan
expected, and real runs began stopping on an unobserved expectation two rounds ago.
Nothing ran the demo - the suite covers the loop with its own frames - so nothing
failed. Its frames agree with its plan now, and `test_week4_demo.py` runs the
script and asserts what the operator is promised, including that `--fail-at` still
fails.

**The step log stored the text it said it never stored.** A step record names its
action twice: `resolved` is what the adapter decided, `action_result` is what the
executor was handed. Only the second went through the redactor, so `steps.jsonl` -
and `task_summary.json`, which embeds the same records - carried the raw typed text
while the usage guide stated that only its length is kept. Two rounds of redaction
work had covered the argument map and the copied action, and neither covered the
original.

Found by reading the files a real run produced instead of the object the code
returned: every assertion in the suite checked the in-memory result or the
recorder's own helper, and the artefact was the one thing nobody looked at. Both
copies are masked now, in the recorder rather than at the call site, so the step log
and the summary cannot diverge; the test asserts the string is absent from both
files. The guide also now says what is *not* masked - on-screen OCR text and the
instruction itself - because those two are deliberate and a reader would otherwise
assume they were covered.

**The frames were not where the record said they were.** `obs-NNNN.json` names the
image each coordinate was measured on, and those images were written to the
directory the session folders live in rather than into the run. Every run's
screenshots therefore piled up beside the runs, `outputs/week4/` filled with PNGs
that belonged to no session, and each run's record pointed outside its own folder -
so moving, archiving or shipping a session broke the traceability the module is
built around. 14.1 asks for the frames to be kept with the step for exactly this
reason.

They live in `<session>/frames/` now, and the warmup probe's own screenshot - which
belongs to no run at all - in `outputs/week4/warmup/`, so the output tree reads as
runs and nothing else. The evidence collector still copies text only: a screenshot
is a picture of the whole desktop, and that is what the frames directory is for.

**The template the reviewer fills in described an evidence bundle from two rounds
ago.** It said the collector copies "the two text records - `task_summary.json` and
`steps.jsonl`", while the collector had been copying four for two rounds. The two
additions - the configuration the frames were produced with, and the warmup that
preceded the run - arrived in the repository unannounced, so a reviewer reading the
bundle would have found two files the instructions did not account for and no
reason to open either. It also lacked the warmup command, which 13.4.1 wants run
before any task, so the first row's timing would have included a cold start.

The list is asserted against the collector's own now, so a fifth record cannot be
added quietly either. This is the third document to fall out of step with the
artefacts in as many rounds - the usage guide's run-directory listing and its
redaction paragraph were the others - and all three were found by reading the
document against the files a run actually produces rather than against the code
that writes them.

**The environment check did not check the thing that was stopping every run.**
`scripts/check_environment.py` confirmed the packages, the Tesseract binary and the
macOS Accessibility permission, and then printed "Week 1 environment is ready" on a
machine where `mss` sees one zero-sized pseudo-monitor, every observation fails and
every run stops before it plans. Importing `mss` is not the check - it imports
perfectly well without Screen Recording permission.

It now attempts an actual capture and says which permission is missing and where to
grant it, which is the difference between an operator following the diagnostic
guide to "wake the screen" and an operator fixing the thing that is wrong. The
hand-back manual gained the check as a step in its own right, because on Windows it
also confirms the capture path end to end before five tasks depend on it.

This is the third failure message in the week that named a cause the operator could
not act on - after the capture error attributed only to a sleeping display, and the
missing Tesseract binary that read as a locked screen.

**The first real runs are in, and the report says what they are.** Ten attempts on
the Windows node, five cases, zero successes - recorded as measured, not as
unmeasured, because this project has spent its whole length insisting those are
different things. Every attempt is kept, including the three that were blocked
before anything was dispatched and the one that sent nothing: T04's record shows a
single click and then a stop on an empty-text step.

The failures are one kind: the plan could not be executed as the model wrote it.
Two `type_text` steps with no text, two targets that were the model's own
descriptions rather than anything on screen ('browser', 'week4 test application'),
one text OCR read twice so the adapter refused to choose between the candidates,
three precondition blocks, one exhausted budget, one click that missed a close
button. Nothing crashed and no safety property failed.

The finding worth most is that the model aimed at the terminal's own text: the one
click T04 dispatched targeted `'Open the test conversation first.'`, a line the CLI
had printed. The mock never does this - it only chooses among elements matching the
instruction - so five dry runs and every scripted test were blind to it. The remedy
is operational, and both manuals now say it.

Two prompt rules follow from the two commonest failures: `target_text` must be a
list entry copied word for word or left out, and every argument is required, with
`type_text` never empty. They fit the same measured budget as before (983
characters of 1 000). The precondition guard's note also carries its evidence now -
how many elements were read, how many observation errors, and the first text seen -
because "the rule already holds" is equally true of a closed application and one
that is merely behind another window, and the Windows round had to parse
`obs-NNNN.json` by hand to tell those apart.

## 8. Deliverables

- `src/gui_agent/runtime/` - the run layer, including pixel correspondence and constrained visual target mapping.
- `scripts/week4_agent_cli.py` - the command-line entry point.
- `scripts/week4_warmup.py` - the warmup 13.4 asks for: a text probe and a real
  screenshot through the project's own client, with the cold/warm record the operator keeps.
- `scripts/week4_offline_demo.py` - the loop against scripted frames.
- `scripts/week4_collect_evidence.py` - copies a finished run's text records into
  `Document/Week4/evidence/`, so a run id in the test report resolves inside the
  repository rather than only on the machine that produced it.
- `configs/week4.yaml` - Week 4 limits, with `ExecutionConfig` added to `config.py`.
- 577 new tests.
- `Document/Week4/Week4_Usage.md` - flags, the safety model, the record layout.
- `Document/Week4/Week4_Troubleshooting.md` - seventy-three symptoms with what to check
  and what the code actually does about each.
- `Document/Week4/Week4_Basic_Task_Test_Report.md` - the five results, one row per
  case, with the precondition, the attempt and success counts, the verification
  method and the evidence path each row needs.
- This report.

**W4-13 diagnostic guide.** The hand-off asks for a table of fifteen situations -
connection, model, perception, resolution, execution, verification and recording -
each with what to check and how it is handled. That is
`Document/Week4/Week4_Troubleshooting.md`, grown to seventy-three as the Windows review
added situations the first pass had not met. Each row states the behaviour this
code has, not the behaviour it ought to have; a diagnostic guide describing
behaviour the implementation does not have would be worse than none.

**Where the Chinese deliverables live.** Three files this report refers to are
delivered alongside the repository rather than inside it, because they are written
for the person doing the Windows review and are maintained in the same folder as
the Chinese report they belong to: `Week4_中文实验报告.md` and its `.docx`,
`Week4_Windows复核手册.md`, and `sync_report_numbers.py`, the tool that keeps the
numbers in both of them in step with this repository. They sit in the week 4
document folder of the internship deliverables, with the hand-back notes they were
written against. A reader who follows a citation to one of those filenames and
does not find it in the tree has not hit a missing document.

## 9. Limits

- **Four of five basic tasks have passed in five real successful runs.** T01
  passed twice; T02, T03 and T05 once each. T04's complete repaired flow has
  passed only with the prepared mock and synthetic chat. The latest real attempt
  was blocked before input; the new isolated-region context passes on a saved
  image with the real model and establishes no real send.
- Only the primary monitor is supported.
- `type_text` uses `pyautogui.typewrite`, so ASCII only; Unicode input is not
  claimed.
- **Re-planning is bounded.** `ExecutionOptions.max_planning_attempts` defaults to
  4; the wall-clock budget and cumulative action cap can stop recovery sooner.
  T04 never retries after a send attempt. Every action still passes
  the adapter resolution and executor boundary checks. Recovery was implemented after T02 failures with incomplete
  plans; its retained successful run records one planning attempt. This does
  not guarantee recovery for every task or failure.
- **Foreground checks reduce the focus gap but cannot make input atomic.**
  Captures record title, class, identity, bounds and whether the foreground stayed
  stable during capture. Anonymous actions and T04 require current identity/bounds
  checks immediately before dispatch; T04 also preserves the authorised conversation
  header pixels. Missing metadata is refused. A foreground change after the final
  check but before the OS consumes input remains possible. Both risk confirmations,
  coordinate boundaries and PyAutoGUI fail-safe remain enabled.
- **Merging OCR words into lines trades a little precision for resolvability.** A
  target that names one word inside a longer mixed line - "main" in "Current branch
  main" - now resolves to the centre of the whole line, because the element no
  longer carries the individual word boxes. Before the merge that target resolved
  to nothing at all, so this is the better half of the trade, but a click meant for
  a control that shares a text row with unrelated text can land on the neighbour.
  The gap rule exists to keep that rare.
- **The success rules for T01 and T05 are weaker than the written standard.** They
  can only look for text on screen, so they cannot tell "I closed it" from "it was
  never open". The precondition check covers that half for a real run, and from the
  command line it cannot be switched off: `require_preconditions` is an `ExecutionOptions`
  field with no CLI flag and no config key. A run started from Python with it set to false
  proves nothing about either case.
