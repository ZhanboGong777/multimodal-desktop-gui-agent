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

### Three decisions that shape the loop

**One plan, then step-by-step re-observation.** The model is not called once per
click. A plan that is re-derived every step cannot be shown to the operator before
it runs, and re-observing is orders of magnitude cheaper than re-planning. This
also keeps a 20-action task inside a 240 s budget: one planning call at 12–14 s
plus roughly 2 s of observation per step.

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

Unchanged from Week 3: code and tests on the MacBook Air M2, model served by
Ollama on the Windows node over HTTP. Week 4 adds a screen to drive, so the
perception and control paths are exercised for real for the first time.

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
| MacBook Air M2 | **469 passed**, ruff clean |

Week 3 ended at 300 tests (counted on `d67de1f`). Week 4 adds 169: 142 in the ten
files below, 10 in `test_model_mock.py`, 7 in `test_config.py` for the
configuration surface, 7 in `test_plan_parser.py` for the step-status vocabulary
and the plan-ordering rules, 2 in `test_ocr.py` for the label-merging fix, and 1
in `test_model_config.py` for the retry policy.

| Test file | Covers |
| --- | --- |
| `test_action_adapter.py` | 27 cases: unique, ambiguous, missing and stale targets, a target split across word-level elements, parameter errors including wrong types, out-of-range coordinates, key whitelist, platform hotkeys, coordinate scaling, `finish` refusal |
| `test_runtime_runner.py` | 42 cases: the offline closed loop, coordinate provenance, dry-run semantics, budget refusal, cancellation, failed actions, wrong-screen failure, recording, the context-overflow hint, the guard that refuses to start a real run whose goal already holds, the second confirmation a risky task must get, the provenance the summary carries, and the note a frame with no readable text leaves |
| `test_runtime_verification.py` | 10 cases: rule matching, forbidden text, unverifiable tasks, degraded observations, and the two case rules that have to tell a real result from a lookalike |
| `test_config.py` | 7 cases: an unknown key is refused at the top level and inside a section, out-of-range limits are refused, and the defaults are the safe mode |
| `test_runtime_observation.py` | 3 cases: frame-local ids do not repeat, unlabelled contours are not offered as targets, and a prompt line carries what a step needs to aim |
| `test_runtime_recording.py` | 6 cases: redaction, append-only steps, per-frame files, summary |
| `test_week4_cli.py` | 15 cases: argument errors, no `--yes`, dry-run default, summary always written, `--execute` refused without a terminal, the callback set an execute run hands over, `.env` loading, the flag/environment/YAML precedence, and the numeric limits |
| `test_week4_integration.py` | 5 cases: the loop against a real OpenAI-compatible server over a real socket, which reads the element ids out of the prompt it receives |
| `test_week4_prompts.py` | 12 cases: the prompt fits its budget, describes element targeting and every action's arguments, carries the platform, trims the element list by whole lines, and keeps markers verbatim |
| `test_week4_cases.py` | 10 cases: the invariants the five case definitions must hold - a machine-checkable rule, a declared precondition, a distinctive marker, a copy handed back by `get_case`, and the fresh marker a send-message run is given |
| `test_week4_evidence.py` | 12 cases: what the evidence collector copies, what it refuses to copy, which run `--latest` picks, and its error paths |

### The offline closed loop

`test_runtime_runner.py::test_the_full_loop_runs_offline_and_verifies` is the gate
that had to pass before any real action was allowed. It drives a scripted observer,
planner and executor through observe → plan → resolve → act → re-observe → verify
and asserts that the action was dispatched with `dry_run=False`, that the resolved
coordinate came from the second frame rather than the first, and that the run is
only `succeeded` because the task rule matched.

`test_a_changed_screen_that_does_not_match_the_rule_fails` covers the dangerous
near miss: the screen did change, the action did succeed, and the task still fails.

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

**Not yet measured.** The five cases are defined with their success rules in
`src/gui_agent/runtime/tasks.py` and can be listed with `--list-cases`, but a real
result requires a real desktop and a real model.

A dry run against the rule-based mock stops at the first step with
`no element matches 'desktop'`, which is correct: the mock's plan names elements
that are not on the screen, and the adapter refuses to guess. Mock plans are not
evidence of real capability and are not counted as such here.

The results table is filled in by `Week4_Basic_Task_Test_Report.md` once the runs
have been performed.

## 7. Problems and handling

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

**The prompt truncated the observation mid-field.** `build_user_prompt` rendered
the whole context as JSON and cut it at 4 000 characters, which is what 8.2.10
forbids: the cut lands inside the element list, leaving an element id with half its
text. The list is now trimmed one whole element at a time and the prompt says how
many were left out; the short bounded fields beside it are serialized in full. The
same pass fixed the system prompt's blanket "keep every string under 60
characters", which 8.2.3 rules out - it invites the model to abbreviate the very
marker the task is verified against. It now limits `description` and `summary`
only, and says to copy text exactly. 8.1.7 asks for the execution limits as well
as the step cap, so the planner's context now carries "at most N actions and T s
for the whole task" rather than letting the model discover the budget by having
its plan refused.

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
in `_resolve_wait`, so the configured value could not have had any effect. There is
still no `max_replans`: nothing implements re-planning yet, and adding a knob no
code reads would repeat the mistake that `require_preconditions` was.

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

That exercise is also the only time the second confirmation has actually run. A T04
driven through a pty shows two prompts, and declining the second cancels with
nothing dispatched. The path had been wired but never executed end to end, which is
how it came to be left unwired in the first place.

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

## 8. Deliverables

- `src/gui_agent/runtime/` - the run layer (8 modules).
- `scripts/week4_agent_cli.py` - the command-line entry point.
- `scripts/week4_offline_demo.py` - the loop against scripted frames.
- `scripts/week4_collect_evidence.py` - copies a finished run's text records into
  `Document/Week4/evidence/`, so a run id in the test report resolves inside the
  repository rather than only on the machine that produced it.
- `configs/week4.yaml` - Week 4 limits, with `ExecutionConfig` added to `config.py`.
- 102 new tests.
- `Document/Week4/Week4_Usage.md` - flags, the safety model, the record layout.
- `Document/Week4/Week4_Troubleshooting.md` - twenty-seven symptoms with what to check
  and what the code actually does about each.
- `Document/Week4/Week4_Basic_Task_Test_Report.md` - the five results, one row per
  case, with the precondition, the attempt and success counts, the verification
  method and the evidence path each row needs.
- This report.

**W4-13 diagnostic guide.** The hand-off asks for a table of fifteen situations -
connection, model, perception, resolution, execution, verification and recording -
each with what to check and how it is handled. That is
`Document/Week4/Week4_Troubleshooting.md`, grown to twenty-seven as the Windows review
added situations the first pass had not met. Each row states the behaviour this
code has, not the behaviour it ought to have; a diagnostic guide describing
behaviour the implementation does not have would be worse than none.

## 9. Limits

- **The five basic tasks have not been run for real.** Everything above is offline
  or dry-run evidence. This is the main open item.
- Only the primary monitor is supported.
- `type_text` uses `pyautogui.typewrite`, so ASCII only; Unicode input is not
  claimed.
- There is no re-planning. A failed step stops the run; recovery is Week 6.
- **There is no automatic focus check.** Targets are resolved against a fresh frame
  and the adapter refuses an absent or ambiguous one, but nothing reads the
  foreground window to confirm that the intended application has keyboard focus
  before a typing or send step. 11.3 asks for that to be written down when
  cross-platform foreground detection is not implemented, and it is not. The
  mitigation is procedural rather than automatic: a `medium` or `high` risk task is
  confirmed a second time with the text the plan will type, which is why a T04 row
  cannot be credited unless the operator saw that prompt. Nothing here prevents
  every mis-typed character, and no claim to the contrary is made.
- **Merging OCR words into lines trades a little precision for resolvability.** A
  target that names one word inside a longer mixed line - "main" in "Current branch
  main" - now resolves to the centre of the whole line, because the element no
  longer carries the individual word boxes. Before the merge that target resolved
  to nothing at all, so this is the better half of the trade, but a click meant for
  a control that shares a text row with unrelated text can land on the neighbour.
  The gap rule exists to keep that rare.
- **The success rules for T01 and T05 are weaker than the written standard.** They
  can only look for text on screen, so they cannot tell "I closed it" from "it was
  never open". The precondition check covers that half for a real run, but a run
  started with `require_preconditions=false` proves nothing about either case.
