# Week 4: basic task test report

Five controlled tasks, run through `scripts/week4_agent_cli.py`. Every row has to
be filled in from a real run. An unfilled row means "not measured" - it never
means "passed", and a dry run is never a success however clean it looks.

## How to fill a row

```bash
export GUI_AGENT_API_KEY=ollama
export GUI_AGENT_BASE_URL=http://<windows-host>:11434/v1
export OLLAMA_CONTEXT_LENGTH=32768        # before the server starts; verify /api/ps

# Warm the model first, and separately. The first request a cold server sees costs
# seconds to a minute, and a task that pays it records a model load as its own
# planning time - the first number in the table would be the one nobody can read.
python scripts/week4_warmup.py \
    --provider openai_compatible --model qwen2.5vl:7b \
    --json outputs/week4/warmup.json --repeat 2

python scripts/week4_agent_cli.py --case T01 \
    --provider openai_compatible --model qwen2.5vl:7b --execute
```

Then put the evidence where the report can point at it:

```bash
python scripts/week4_collect_evidence.py --latest T01
```

`outputs/` is not tracked, so a run id on its own resolves to nothing on any other
machine. The collector copies the text records into
`Document/Week4/evidence/<run id>/`: `task_summary.json` (the verdict and timings),
`steps.jsonl` (one line per step), `run_config.json` (the settings the frames were
produced with) and `warmup.json` when the run has one. It deliberately leaves the
screenshots and the per-frame observation files behind - they are a picture of the
whole desktop, and on-screen text is in them verbatim.

`status=succeeded` **and** `execute=true` are both required before a task counts as
a real success. They are not sufficient when the model and action backend are
simulated: prepared test records are explicitly offline and excluded from this
real-desktop inventory.

## Results

This report was reconciled against every checked-in `task_summary.json` under
`Document/Week4/evidence/`: **39 run directories**, of which **38 record
`execute=true`** and one is a dry run. **Four of the five cases have recorded
successes: T01, T02, T03 and T05. There are five successful runs**, because T01
succeeded twice. T04 has no recorded success; changing the code does not change
these historical verdicts. The latest T04 acceptance attempt was blocked by its
initial visual-context assessment and dispatched zero desktop actions.

Here, an **attempt** is one checked-in run with `execute=true`, including a run
blocked by a precondition or initial context check. The four blocked runs are shown separately in the
status breakdown; they dispatched no actions. A planning retry within a run is
not another attempt. Uncollected historical runs are excluded from every total.

| Case | Task and preconditions | Attempts | Successes | Recorded statuses | Successful evidence |
| --- | --- | --- | --- | --- | --- |
| T01 | open the browser; desktop visible, launch entry uncovered, no browser running | **6** | **2** | 2 succeeded, 3 failed, 1 blocked | `T01_20261004_140000`, `T01_20261004_213859` |
| T02 | search the web; browser open and focused, English input method | **17** | **1** | 1 succeeded, 16 failed | `T02_20261004_210409` |
| T03 | open the specified `week4_sample.txt` in the week4 test folder; no file of that name already open | **2** | **1** | 1 succeeded, 1 failed | `T03_20261004_171941` |
| T04 | send a fresh per-run marker to the open test conversation; operator agrees to a real send | **4** | **0** | 2 failed, 1 timed_out, 1 blocked | none; all four available runs are listed below |
| T05 | close the test application; `WEEK4-OPEN-FILE-OK` visible in the focused application first | **9** | **1** | 1 succeeded, 5 failed, 2 blocked, 1 timed_out | `T05_20261004_180657` |
| Total | five controlled cases | **38** | **5** | 5 succeeded, 27 failed, 4 blocked, 2 timed_out | **4 / 5 cases** |

Every successful summary records `execute=true`, final `verification=passed` and
the following dispatched actions. Milliseconds are rounded to the nearest whole
number here; aggregate means below use the original floating-point values.

| Successful run | execute | Final verification | action_count | Dispatched actions | execution_ms |
| --- | --- | --- | --- | --- | --- |
| `T01_20261004_140000` | true | passed | 1 | double_click | 6 938 |
| `T01_20261004_213859` | true | passed | 1 | double_click | 7 563 |
| `T02_20261004_210409` | true | passed | 2 | type_text, key_press Enter | **11 969** |
| `T03_20261004_171941` | true | passed | 1 | double_click | 5 313 |
| `T05_20261004_180657` | true | passed | 1 | hotkey Alt+F4 | 9 750 |

T01's first success and its later repeat both belong to the 2026-10-04 follow-up
runs. T02's successful final rule requires the query and `google.com`, so merely
typing the query in an address bar is insufficient. T03's final rule requires
`WEEK4-OPEN-FILE-OK`; T05 requires that marker to disappear. The recorded automatic
verdicts must still be read with the limitations in §"Per-case success rules".

### Why the preconditions are not optional

T01 and T05 can appear satisfied without doing anything: a browser may already be
running, and T05's marker is absent while its window is shut or hidden. Such runs
are `blocked`, cannot be credited with reaching the goal, and dispatch no actions.
T01's recorded blocked run used the browser-process guard; T05's two recorded
blocked runs found the goal already satisfied on the untouched screen.

T04's marker is minted per run (`WEEK4_MESSAGE_CHECK_<timestamp>`) and printed as
`marker`. A previous run's message cannot satisfy this run's rule. A draft, a sent
message in another conversation and a message sent by an external probe all fail
the attribution required for this controlled task.

## Every recorded real attempt

These tables include **all 38 checked-in `execute=true` summaries**, grouped by
case and ordered by run id. Each row resolves to
`Document/Week4/evidence/<run id>/task_summary.json`; no unavailable run is mixed
into this inventory. `actions` is `action_count`, not planned steps or planning
attempts. The stop descriptions come from `stop_reason`, step errors, final
verification or recorded notes. The summary's `commit` identifies the code used
by each run; the runs span multiple commits, rather than all using one revision.

### T01 - open the browser

| run id | status | actions | execution_ms | planning_attempts | Recorded result or stop |
| --- | --- | --- | --- | --- | --- |
| `T01_20261003_172642` | failed | 0 | 1 454 | 1 | ambiguous `msedge`: two matching elements |
| `T01_20261003_180109` | failed | 3 | 9 375 | 1 | step 3: expected result `Task completed` not observed |
| `T01_20261004_000434` | failed | 1 | 18 281 | 1 | final rule missing `http`, `search` |
| `T01_20261004_004112` | blocked | 0 | 1 828 | 0 | precondition: `msedge.exe`, `chrome.exe` already running |
| `T01_20261004_140000` | succeeded | 1 | 6 938 | 1 | final verification passed |
| `T01_20261004_213859` | succeeded | 1 | 7 563 | 1 | final verification passed |

### T02 - search the web

| run id | status | actions | execution_ms | planning_attempts | Recorded result or stop |
| --- | --- | --- | --- | --- | --- |
| `T02_20261003_173909` | failed | 0 | 2 047 | 1 | step 1: no element matches 'browser' |
| `T02_20261004_001545` | failed | 0 | 1 890 | 1 | step 1: no element matches 'browser' |
| `T02_20261004_005017` | failed | 0 | 2 469 | 1 | step 1: no element matches 'browser' |
| `T02_20261004_005225` | failed | 1 | 12 718 | 1 | step 2: no element matches 'search bar' |
| `T02_20261004_145300` | failed | 0 | 1 172 | 1 | step 1: empty `arguments.text` |
| `T02_20261004_150104` | failed | 0 | 1 922 | 1 | step 1: no element matches 'browser' |
| `T02_20261004_152743` | failed | 0 | 1 875 | 1 | step 1: no element matches 'search bar' |
| `T02_20261004_153537` | failed | 1 | 16 579 | 1 | expected on screen but not found: GUI agent research |
| `T02_20261004_161652` | failed | 2 | 22 156 | 1 | expected on screen but not found: GUI agent research |
| `T02_20261004_162722` | failed | 1 | 15 672 | 1 | expected on screen but not found: GUI agent research |
| `T02_20261004_163241` | failed | 0 | 1 703 | 1 | ambiguous `about:blank`: two matching elements |
| `T02_20261004_181529` | failed | 1 | 17 782 | 1 | expected on screen but not found: GUI agent research |
| `T02_20261004_185721` | failed | 2 | 21 266 | 1 | expected on screen but not found: google.com/search |
| `T02_20261004_191217` | failed | 1 | 16 141 | 1 | expected on screen but not found: google.com |
| `T02_20261004_193222` | failed | 3 | 131 234 | 2 | expected on screen but not found: GUI agent research, google.com |
| `T02_20261004_203337` | failed | 2 | 71 734 | 1 | expected on screen but not found: google.com |
| `T02_20261004_210409` | succeeded | 2 | 11 969 | 1 | final verification passed |

### T03 - open a specified file

| run id | status | actions | execution_ms | planning_attempts | Recorded result or stop |
| --- | --- | --- | --- | --- | --- |
| `T03_20261003_174110` | failed | 0 | 1 922 | 1 | step 1: empty `arguments.text` |
| `T03_20261004_171941` | succeeded | 1 | 5 313 | 1 | final verification passed |

### T04 - send a message

| run id | status | actions | execution_ms | planning_attempts | Recorded result or stop |
| --- | --- | --- | --- | --- | --- |
| `T04_20261003_174709` | failed | 1 | 6 062 | 1 | step 2: empty `arguments.text`; only the preceding click dispatched |
| `T04_20261005_003044` | timed_out | 4 | 499 125 | 3 | marker checks failed; timed out after 3 planning attempts, no final verdict |
| `T04_20261005_012521` | failed | 3 | 255 093 | 2 | marker check failed; second pass's typing expectation failed |
| `T04_20261006_115446` | blocked | 0 | 108 484 | 0 | initial visual context: header/composer regions are not valid active-chat evidence |

### T05 - close the application

| run id | status | actions | execution_ms | planning_attempts | Recorded result or stop |
| --- | --- | --- | --- | --- | --- |
| `T05_20261003_174334` | failed | 0 | 1 718 | 1 | ambiguous `x`: five matching elements |
| `T05_20261003_180511` | blocked | 0 | 1 109 | 0 | precondition: goal already holds on untouched screen |
| `T05_20261003_181018` | timed_out | 0 | 0 | 1 | no actions; 633 453 ms in confirmation, no final verdict |
| `T05_20261003_182259` | blocked | 0 | 2 391 | 0 | precondition: goal already holds on untouched screen |
| `T05_20261003_182640` | failed | 1 | 12 906 | 1 | still on screen but should be gone: WEEK4-OPEN-FILE-OK |
| `T05_20261004_173604` | failed | 1 | 16 125 | 1 | still on screen but should be gone: WEEK4-OPEN-FILE-OK |
| `T05_20261004_174538` | failed | 1 | 17 000 | 1 | still on screen but should be gone: WEEK4-OPEN-FILE-OK |
| `T05_20261004_175356` | failed | 1 | 17 094 | 1 | still on screen but should be gone: WEEK4-OPEN-FILE-OK |
| `T05_20261004_180657` | succeeded | 1 | 9 750 | 1 | final verification passed |

## Dry runs

`T01_20261001_215642` is the sole remaining dry-run evidence directory. It records
`execute=false`, `status=dry_run_completed`, `verification=inconclusive`, provider
`mock` and model `mock-vision-model` on macOS. Its one action record is dry-run
output, not a dispatched desktop action. It is excluded from the 38 attempts,
five successes, action totals and timing means.

## Historical rounds and unavailable evidence

The historical round names describe dated subsets of the inventory, not separate
counts to add to it:

- **First round, 2026-10-03:** 10 checked-in real runs, with no `succeeded`
  verdicts. The contemporary account also mentioned an uncollected T05 run;
  it is listed as unavailable below and does not raise the recorded total to 11.
- **Second round, 2026-10-04 follow-up runs:** 25 checked-in real runs, including
  **all five successes**. Early failures in this round do not describe its final
  outcome. T02 has 16 recorded runs in this round, plus its first-round failure,
  for 17 overall.
- **Later T04 follow-ups, 2026-10-05:** two checked-in runs, one `timed_out` and one
  `failed`. Together with its first-round failure, T04 has three recorded attempts
  and no success by that date.
- **Real T04 acceptance, 2026-10-06:** `T04_20261006_115446` adds one blocked
  attempt with zero actions. T04 now has four retained attempts and no success.

These earlier references cannot be verified from checked-in run directories and
are excluded from the aggregate statistics:

| Historical run reference | Evidence availability and reported context |
| --- | --- |
| `T01_20261004_000322` | **Unavailable:** no directory here. The earlier narrative reported a failed single click that selected the browser shortcut. That report is not a checked-in summary. |
| `T05_20261003_180229` | **Unavailable:** evidence was not collected; the earlier narrative reported `blocked` because the goal already held. |
| `T02_20261004_182646` | **Unavailable:** previously described as an uncollected false `succeeded` verdict in `outputs/week4`; the query was reportedly still in the address bar. It is not one of the five recorded successes. |
| `T04_20261005_192417` | **Unavailable in this evidence snapshot:** the external background document reports `failed`, 0 actions, about 895 s and four target-resolution failures. It is not one of the 39 directories audited here, and is not added to the four recorded T04 attempts. |

Some earlier diagnoses also required screenshots or foreground sampling retained
outside the checked-in evidence. The following accounts remain historical
operator observations, rather than new measurements or automatic passes:

- The first T01 success followed changing shortcut clicks to land on the icon;
  T03 followed tolerating OCR's underscore-to-space reading of the filename.
- Early T02 runs were affected by foreground windows, the input method and
  verification that could not distinguish address-bar text from a submitted
  search. `T02_20261004_185721` records a **failed** final check for
  `google.com/search`, even though the operator reported seeing a results page.
  Its recorded status stays failed. Later window-title signals and bounded
  re-planning are available in the harness; planning retries in an existing run
  do not increase the attempt count.
- The original T05 account described a missed close click and a minimised window;
  minimising is excluded by §"Per-case success rules". Later runs encountered
  Notepad tabs or a save dialog. `T05_20261004_180657` records a successful Alt+F4
  action; the earlier blanket claim that this hotkey never works on Windows is
  not supported by that successful summary.
- The operator found that minimising or foregrounding the driver's console,
  subprocess windows and screenshot viewers could disturb the captured desktop.
  These findings explain the need to prepare the foreground and keep observation
  viewers closed during execution. They do not imply that the later successful
  cases were impossible on this single-monitor machine.

## Metrics

Every aggregate below is recomputable from the 39 checked-in summaries, with the
single `execute=false` dry run excluded. **Blocked runs remain in the 38-attempt
inventory and denominator**, with their status disclosed. This uses one consistent
recorded-run definition rather than mixing uncollected output, model calls and
real run directories.

```text
recorded-run success rate = count(execute=true and status=succeeded) / count(execute=true)
case coverage = cases with at least one recorded success / five defined cases
mean successful execution time = sum(execution_ms of all successes) / five successes
mean time over all recorded real attempts = sum(field of execute=true runs) / 38
```

| Metric | Value |
| --- | --- |
| evidence run directories | **39** |
| formal recorded real attempts (`execute=true`) | **38** (T01 6, T02 17, T03 2, T04 4, T05 9) |
| successes (`succeeded` and `execute=true`) | **5**, in the successful-run table above |
| cases with a recorded success | **4 / 5 = 80 %** |
| recorded-run success rate | **5 / 38 = 13.2 %** |
| recorded real-run statuses | 5 succeeded, 27 failed, 4 blocked, 2 timed_out |
| mean `execution_ms` over all five successful runs | **8 306.6 ms** = (6 938 + 7 563 + 11 969 + 5 313 + 9 750) / 5 |
| mean `execution_ms` of the first success per successful case only | **8 492.5 ms** = (6 938 + 11 969 + 5 313 + 9 750) / 4; excludes T01's later repeat |
| mean `execution_ms` over all 38 recorded real attempts | **35 522.6 ms** |
| mean `planning_ms` over all 38 recorded real attempts | **128 785.9 ms** |
| total actions actually dispatched | **36** (`action_count`, excluding dry-run output) |
| dispatched actions with a passed step-level check | **34 / 36**; a step pass is not a task success |

T02 accounts for **17 of the 38 attempts**. T03 has two and T04 has four.
These counts describe the collected evidence, not an estimate of all work performed
on the operator's machine.

`execution_ms` is the harness phase from confirmation to the final verdict or stop;
for a run with no confirmation timestamp, its fallback phase boundary is used.
`planning_ms` measures the initial planning phase, `confirmation_ms` keeps the
operator gate separate, and `elapsed_ms` is the whole run. Re-planning inside the
execution loop contributes to `execution_ms`, so it is not pure desktop input time.
The mean includes blocked runs and runs that dispatched nothing. Warmup is recorded
separately and excluded from these phases.

Historical latency probes found about **200 s** per planning call through the
foreground driver, versus **34 s cold / 4 s warm** without it; the later T04 background
quotes **110-120 s** from observation timestamps inside a run. These are different
measurement conditions and sources. The summaries preserve what each run cost;
the pooled mean does not isolate model latency, memory pressure or driver overhead.

These are controlled tasks on one Windows node plus a separate macOS mock dry run,
not a benchmark or a success-rate estimate for general desktop use.

## Per-case success rules

| Case | Counts as success | Does **not** count |
| --- | --- | --- |
| T01 | a browser window with an address bar or tab strip is in the foreground | a launcher shortcut pressed, the name typed into a search box, or another browser already in front |
| T02 | a results page is loaded and the query text is visible | text sitting in the input box without a submitted search |
| T03 | the file is open in an application | the file merely selected, or a same-named file opened from elsewhere |
| T04 | the unique marker appears as a sent message in the correct conversation | a draft, an old message with the same text, a send to the wrong conversation, or an external probe's send |
| T05 | the target application's window is gone and other applications are untouched | the window minimised, or another window closed |

The right-hand column is the standard a human applies. T01/T02/T03/T05 automatic
rules use observed text, including recorded foreground title and class where
available, so they can be weaker than that standard. The repaired T04 instead uses
strict visual evidence of the active conversation, exact sent marker and empty
composer, as described below. Its latest live acceptance was blocked; saved-image
model validation does not establish its live success rate. For T05 a missing marker does not prove
that a hidden or minimised window was closed; for T01 a visible browser does not
prove this run launched it. The precondition checks reduce that gap. Foreground
metadata in the retained historical runs describes observation time. The current
T04/anonymous-target path also checks foreground identity and bounds immediately
before input, while a change after that check can still race with dispatch.

If a run has to be verified by eye, mark its verification method `manual` in the
notes. An automatic check that did not run is not an automatic pass. A previously
recorded failed or timed-out run is not promoted because a later repair exists.

## Environment

| Item | Recorded Windows real runs | Evidence basis |
| --- | --- | --- |
| Platform and OS | `win32`, `Windows-11-10.0.26200-SP0` | all 38 summaries |
| Display | screenshot 2560x1600, control 2560x1600 in 35 runs; absent in 3 older blocked summaries | `screen` |
| Python | 3.12.4 | `python_version` |
| Model and provider | `qwen2.5vl:7b`, `openai_compatible` | `model_name`, `provider` |
| Commit | varies by run; `T02_20261004_153537` has an empty commit field | `commit`; do not infer an absent revision |
| Context window | historical operator setting: `OLLAMA_CONTEXT_LENGTH=16384`; latest T04 acceptance separately verified actual loaded context 32768 | context is not a summary field; latest `/api/ps` proof is in the external acceptance bundle |
| Endpoint and per-request timeout | inspect each run's `run_config.json`, rather than applying the current configuration retrospectively | `model.base_url`, `model.timeout_seconds` |
| Warmup and memory | read the run's `warmup.json` when recorded; absent fields are unavailable, not zero | cold/warm probe latency and available memory fields |

The macOS directory is the mock dry run described in §"Dry runs". It records
`darwin`, macOS 26.3.1, Python 3.12.10 and a 1920x1080 screenshot/control space.
It establishes no real-task result on macOS.

## Failure notes

**T04 has no recorded success.** Its four available summaries establish different
stopping points:

- `T04_20261003_174709`: one real click passed its step check; the next `type_text`
  had no `arguments.text`, so this run never typed its marker.
- `T04_20261005_003044`: four real actions (`type_text`, `click`, `type_text`,
  `click`) dispatched and passed their step checks across two executed passes.
  Marker verification failed after both passes; a third planning attempt began,
  and the run ended `timed_out` with `verification=null`. Its 703 813 ms elapsed
  time and 499 125 ms execution phase do not establish that the required message
  was sent.
- `T04_20261005_012521`: three real actions dispatched across two planning attempts.
  The first pass's typing and send click passed step checks, but the task-marker
  check failed. The second pass stopped after typing because its expectation
  `Text is typed into the message box` was not observed. Final status is `failed`,
  with `verification=null` and 460 328 ms elapsed.
- `T04_20261006_115446`: the 2560x1600 assessment placed the header in the
  sidebar area and the composer in the transcript, and incorrectly reported
  the empty composer as nonempty. The strict geometry check refused this evidence
  before planning or confirmation: `blocked`, zero actions, zero planning attempts,
  one model request and 108 484 ms elapsed. `message_context.json` retains the
  refusal and raw structured assessment; no new marker was sent.

The historical operator notes report a sent bubble in 文件传输助手 containing
concatenated marker fragments during the `003044` investigation. The background
attributes that send to an **external probe click**, not to a completed controlled
run. It proves neither that the exact per-run marker arrived through the plan nor
that no message was ever sent on the machine. The global claim "no message was
ever sent" is therefore unsupported; the supported result is **zero successful
T04 runs in the checked-in evidence**.

A dispatched click or a passed step-level check only reports that the input path
ran and its local expectation matched. T04's goal requires an exact fresh marker
as a **sent** message in the correct conversation. Draft and sent text must be
distinguished by their position in the conversation, not by finding the same
string somewhere on screen.

The external `T04_20261005_192417` diagnosis identifies a later target-resolution
blocker: unlabelled input and send controls were omitted from the model's target
list. The first repair exposed detected boxes but could not refresh their ids
after the required pre-action capture. The follow-up below implements that refresh
and stricter message attribution. Neither code change changes an old verdict.

## T04 follow-up validation, 2026-10-06

The collected suite now contains **880 tests**, including **577 additions** to
the 300-test Week 3 baseline: 512 cases in the fifteen Week 4 table files and
65 additions in the other suites. These code checks do not add real task attempts.

`test_runtime_runner.py::test_t04_complete_send_uses_fresh_candidates_and_strict_message_evidence`
passes the complete prepared flow: click the input, type the fresh marker, click
send and verify the resulting bubble. It uses the real planner, runtime, adapter,
verifier and recorder, with the **`prepared-T04` mock provider**, synthetic moving
chat screenshots and a recording action backend. Both confirmation callbacks are
checked, current-frame target centres are used, and the enabled send button requires
one constrained visual mapping followed by another capture. The resulting mock
`succeeded` record is simulated; it is excluded from the retained real-run totals.

The refreshed action path requires stable foreground identity, title, class, bounds and
geometry, then matches actual target pixels and their context to current detected
candidates. A changed control can be mapped by the visual model to one candidate
id only. It cannot change the operator-approved action, marker or recipient, and
a further screenshot must confirm that candidate before input. Before T04 starts,
visual context must show the correct conversation header and an empty composer;
the same header pixels and foreground identity are checked before each input event.
The final assessment must transcribe the exact fresh marker as an outgoing sent
bubble below that header and above a non-overlapping empty composer. Pending,
failed, unavailable or uncertain send evidence cannot pass. The initial recipient
assessment or refusal is recorded in `message_context.json`.

The real `115446` refusal exposed a different problem: asking the model to locate
header, composer and message boxes in a whole desktop image allowed sidebar regions
and older bubbles to contaminate the assessment. `message_regions.py` now selects
a unique observed bottom-wide contour as a possible composer, derives a padded
header crop from the upper OCR row within its horizontal span, and bounds
transcript candidates above it. These are geometric proposals; the model must
confirm that the header is readable and the composer region is an editor.
Independent header and composer images use strict JSON schemas and initial
context excludes older message images. Final assessment inspects native-pixel
message crops, with nearby pixels retained to reveal pending/failed-send indicators,
and returns only a real candidate id. The original candidate box, crop source
bounds, layout transform and source-image pixel hash remain recorded. No crop
supplies a new click point, and no expected recipient, full marker or image-path
timestamp is supplied to the assessor's text. Source and derived-image pixel
hashes validate cached evidence; message-crop geometry distinguishes the original
bubble from its indicator halo. Strict schema/type checks, exact marker comparison,
foreground guards, header-pixel preservation, confirmations, action bounds and
fail-safe remain.

The saved-frame probe
`outputs/week4_t04_crop_validation_20261006_132746/verifier_context_isolated.json`
records two real `qwen2.5vl:7b` requests: the header crop transcribed
`文件传输助手`, and the composer crop reported an editor with no draft. The
recipient/editor context verdict is `passed`, with **zero desktop actions**.
This locally saved model evidence is outside the checked-in real-run inventory;
it neither changes the `115446` blocked verdict nor establishes a real sent message.

The same saved frame exposed planning noise: the original 300-element desktop
list cost 21 806 prompt tokens and produced a send click on a sidebar contour.
The production T04 planning context now lists only current observed candidates
inside the independently authorised composer, requires the exact marker in
`arguments.text`, and tells retries when text was already typed. The revised
real-model probe used 5 841 prompt tokens and produced click-editor, type the
complete 35-character marker, then click the actual `发送` OCR candidate. Target
centres still come from the current observed elements; no free coordinate is used.

| Saved-frame probe | Recorded result | Model requests | Desktop actions | Local record |
| --- | --- | --- | --- | --- |
| Isolated header/editor context | passed; actual header and empty editor | 2 | 0 | `verifier_context_isolated.json` |
| Scoped production planner | correct three-action plan, 31 904 ms; resolution without dispatch | 1 | 0 | `planner_scoped_response.json` |
| Sent-message verifier, readable-title probe | failed; observed old marker does not exactly match the fresh run marker, 12 487 ms | 3 | 0 | `verifier_sent_explicit_readability.json` |
| Sent-message verifier, current production geometry/cache/hidden-filename checks | failed; same exact old-marker mismatch, 74 585 ms | 3 | 0 | `verifier_sent_final_production.json` |

These records are in the local
`outputs/week4_t04_crop_validation_20261006_132746/` directory. The earlier
unscoped plan and the first final-assessment refusal are retained there too.
The final old-frame probe demonstrates strict rejection, not task success;
none of these requests dispatched input or contributes a real-run attempt.

The prepared regressions also cover a wrong or changed recipient, lost focus,
missing or duplicate controls, a plan that only leaves a draft, an ineffective send,
expired mapping/verification requests and cumulative action limits. Once a
post-typing click might have sent the message, further inputs in that plan stop
and the runner checks the goal without automatically retrying the send. Repeated
typing is refused, including when the approved plan contains extra send steps.

Separately, the locally available screenshot for historical
`T04_20261005_192417/obs-0005` was replayed through the detector with its recorded
OCR exclusions. Lower-contrast contour detection, small nested-control retention,
candidate-area OCR exclusions and foreground prioritisation retain the input at
candidate index **0** and send at **50**, both within 200 candidates. This is
offline processing of saved pixels, not a new desktop capture, new real run or
live-model accuracy measurement. That run remains outside the checked-in
39-directory evidence snapshot.

**Real T04 was newly attempted and blocked before input.** The inventory now has
39 directories, 38 real attempts, five successful runs and four of five cases
passing. T04 still has zero retained successes. The region fix and saved-frame
model probes add no desktop actions or messages.

For the owner to collect a new real result, prepare the unlocked desktop with
the correct conversation open and empty input, use the English input method,
close screenshot viewers and select the configured real model endpoint. Then run
the existing guarded preflight and explicitly confirmed case:

```powershell
Set-Location D:\Developer\multimodal-desktop-gui-agent
$env:GUI_AGENT_API_KEY = 'ollama'
$env:GUI_AGENT_BASE_URL = 'http://127.0.0.1:11434/v1'
& .\.venv\Scripts\python.exe scripts\week4_t04_verify.py
& .\.venv\Scripts\python.exe scripts\week4_agent_cli.py --case T04 `
  --provider openai_compatible --model qwen2.5vl:7b `
  --base-url http://127.0.0.1:11434/v1 --execute --task-timeout 1800 `
  --ocr-engine paddleocr --ocr-min-confidence 0.3
```

Verify the actual loaded model has a sufficient context window (32768 in the
latest acceptance preparation), then warm it before preparing the desktop. The
explicit provider/model avoid the YAML's mock default. The CLI keeps the existing
1800-second task budget and both confirmations; the second command can send a real
message. The owner should inspect
the new summary and its visual evidence before collecting that run. Keep earlier
evidence directories unchanged. If a send was attempted but verification was
uncertain, inspect the conversation before starting another run; the runtime will
not retry it automatically.
