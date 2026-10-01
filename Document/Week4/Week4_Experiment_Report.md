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
when it stopped; the new one is **959**, and a test pins it under 1 000.

**Screenshots changing is not success.** `ActionResult.success` only means
PyAutoGUI dispatched an event. Only `Verifier.check_task` can return `succeeded`,
and only when the task's own success rule matches the screen. `finish`, a run out
of steps and a changed screen all fail to satisfy it on their own.

## 3. Experiment environment

Unchanged from Week 3: code and tests on the MacBook Air M2, model served by
Ollama on the Windows node over HTTP. Week 4 adds a screen to drive, so the
perception and control paths are exercised for real for the first time.

### OCR backend choice

Measured on the same 1470x956 frame:

| Backend | Elements found | Time |
| --- | --- | --- |
| Tesseract | 4 | **241 ms** |
| PaddleOCR (medium) | 2 | 5 525 ms |

Tesseract is 23x faster on this machine, matching the Week 2 conclusion.
`configs/week4.yaml` selects it, and the GPU node keeps PaddleOCR.

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
| MacBook Air M2 | **368 passed**, ruff clean |

Week 3 ended at 300 tests. Week 4 adds 68:

| Test file | Covers |
| --- | --- |
| `test_action_adapter.py` | 16 cases: unique, ambiguous, missing and stale targets, parameter errors, key whitelist, platform hotkeys, coordinate scaling, `finish` refusal |
| `test_runtime_runner.py` | 12 cases: the offline closed loop, coordinate provenance, dry-run semantics, budget refusal, cancellation, failed actions, wrong-screen failure, recording |
| `test_runtime_verification.py` | 8 cases: rule matching, forbidden text, unverifiable tasks, degraded observations |
| `test_runtime_recording.py` | 6 cases: redaction, append-only steps, per-frame files, summary |
| `test_week4_cli.py` | 6 cases: argument errors, no `--yes`, dry-run default, summary always written |
| `test_week4_integration.py` | 5 cases: the loop against a real OpenAI-compatible server over a real socket, which reads the element ids out of the prompt it receives |
| `test_week4_prompts.py` | 8 cases: the prompt fits its budget, describes element targeting and every action's arguments, and the user turn carries the platform |

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

## 8. Deliverables

- `src/gui_agent/runtime/` - the run layer (8 modules).
- `scripts/week4_agent_cli.py` - the command-line entry point.
- `scripts/week4_offline_demo.py` - the loop against scripted frames.
- `configs/week4.yaml` - Week 4 limits, with `AgentConfig` added to `config.py`.
- 68 new tests across 7 files.
- `Document/Week4/Week4_Usage.md` - flags, the safety model, the record layout.
- `Document/Week4/Week4_Troubleshooting.md` - fifteen symptoms with what to check
  and what the code actually does about each.
- This report and the basic task test report.

**W4-13 diagnostic guide.** The hand-off asks for a table of fifteen situations -
connection, model, perception, resolution, execution, verification and recording -
each with what to check and how it is handled. That is
`Document/Week4/Week4_Troubleshooting.md`. Each row states the behaviour this code
has, not the behaviour it ought to have; a diagnostic guide describing behaviour
the implementation does not have would be worse than none.

## 9. Limits

- **The five basic tasks have not been run for real.** Everything above is offline
  or dry-run evidence. This is the main open item.
- Only the primary monitor is supported.
- `type_text` uses `pyautogui.typewrite`, so ASCII only; Unicode input is not
  claimed.
- There is no re-planning. A failed step stops the run; recovery is Week 6.
