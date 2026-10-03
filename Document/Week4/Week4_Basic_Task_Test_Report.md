# Week 4: basic task test report

Five controlled tasks, run through `scripts/week4_agent_cli.py`. Every row has to
be filled in from a real run. An unfilled row means "not measured" - it never
means "passed", and a dry run is never a success however clean it looks.

## How to fill a row

```bash
export GUI_AGENT_API_KEY=ollama
export GUI_AGENT_BASE_URL=http://<windows-host>:11434/v1
export OLLAMA_CONTEXT_LENGTH=16384        # before the server starts

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
a real success.

## Results

Attempts and successes are counted per case: a run that had to be repeated after a
failure is two attempts, and only the runs whose summary reads `succeeded` with
`execute=true` are successes.

| Case | Task | Preconditions before the run | Attempts | Successes | Status | Verification method | Timing basis | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | open the browser | desktop visible and launch entry uncovered; **no browser window open** | 2 | **0** | failed | automatic: `http` and `search` both on screen | from the confirmation gate to the final verdict (`execution_ms`); the summary also keeps `planning_ms`, `confirmation_ms` and the whole-run `elapsed_ms`, and the run directory keeps `warmup.json` for the cold and warm numbers, which are deliberately not part of any of them | `T01_20261003_172642`, `T01_20261003_180109` |
| T02 | search the web | a browser window is open and focused | 1 | **0** | failed | automatic: the query text is on screen | as above | `T02_20261003_173909` |
| T03 | open a specified file | `week4_sample.txt` exists in the week4 test folder (`~/Desktop/week4_test`, or `%USERPROFILE%\Desktop\week4_test`); no file of that name is open | 1 | **0** | failed | automatic: `WEEK4-OPEN-FILE-OK` on screen | as above | `T03_20261003_174110` |
| T04 | send a message | the test conversation is open and holds no earlier message with **this run's** marker; the operator agreed a real message may be sent | 1 | **0** | failed | automatic: this run's marker on screen (the CLI prints it as `marker`) | as above | `T04_20261003_174709` |
| T05 | close the application | `week4_sample.txt` is open in the test application, so the marker is on screen; window focused | 6 | **0** | failed (1 `timed_out`, 3 `blocked`, 2 `failed`) | automatic: `WEEK4-OPEN-FILE-OK` **gone** | as above | `T05_20261003_174334`, `T05_20261003_180511`, `T05_20261003_181018`, `T05_20261003_182259`, `T05_20261003_182640`; the sixth attempt, `T05_20261003_180229`, has no directory here - its evidence was not collected and the run no longer exists on the machine that produced it, so the attempt is recorded without it rather than cited to nothing |

**These are the first real runs of the five cases on any machine.** `succeeded` with
`execute=true` was never reached, so **no case counts as a success**. Every attempt is
listed below with the outcome that was actually written; nothing was deleted, and the
dry-run passes in §"Dry runs" are not counted as attempts or successes.

### Why the preconditions are not optional

A real run checks its own starting state before it plans anything. T01 and T05 are
both satisfiable without doing anything - a browser that was already running
carries the text T01 looks for, and T05 only asks that the marker be gone, which is
already true while the window is shut. If either precondition is unmet the run is
reported `blocked` with `the success rule already holds on the untouched screen`,
and no model call is made and no click is dispatched.

That is a correct outcome, not a failure: it means the screen could not have shown
whether this run did the work. Set the precondition up and run it again.

T04's marker is minted per run (`WEEK4_MESSAGE_CHECK_<timestamp>`) and printed as
`marker`, because 15.4 asks for a fresh identifier every time. A fixed one would
make the case single-use: the previous run's message is still in the conversation,
so the rule would already be satisfied and the run would be refused over a message
it did not send.

## Every attempt, with the outcome it actually recorded

Windows node, commit `2587642`, `qwen2.5vl:7b` at `context_length = 16384`, warm start
before the runs. Times are milliseconds from each `task_summary.json`.

### T01 - open the browser

| # | run id | status | actions | execution_ms | what stopped it |
| --- | --- | --- | --- | --- | --- |
| 1 | `T01_20261003_172642` | failed | 0 | 1 454 | `step-1: 2 elements match 'msedge' (obs-0002-e033:'msedge', obs-0002-e048:'msedge'); refusing to pick one arbitrarily` |
| 2 | `T01_20261003_180109` | failed | **3** | 9 375 | `step-3: expected result not observed: 'Task completed'` |

Attempt 2 is the informative one: three actions were dispatched and step 1 verified -
`expected result observed: browser` - so a browser did come to the foreground. It then
clicked `Close PowerShell` and finally `Finish`, whose expected result `Task completed`
is a phrase the screen never carries. The run stopped there instead of continuing from
a step whose result was not observed.

### T02 - search the web

| # | run id | status | actions | execution_ms | what stopped it |
| --- | --- | --- | --- | --- | --- |
| 1 | `T02_20261003_173909` | failed | 0 | 2 047 | `step-1: no element matches 'browser' in obs-0002` |

The plan asked to click `browser`; no element on screen carried that text.

### T03 - open a specified file

| # | run id | status | actions | execution_ms | what stopped it |
| --- | --- | --- | --- | --- | --- |
| 1 | `T03_20261003_174110` | failed | 0 | 1 922 | `step-1: type_text requires a non-empty arguments.text` |

The plan chose `type_text` for `week4_sample.txt` but supplied no text, so nothing could
be dispatched.

### T04 - send a message

| # | run id | status | actions | execution_ms | what stopped it |
| --- | --- | --- | --- | --- | --- |
| 1 | `T04_20261003_174709` | failed | **1** | 6 062 | `step-2: type_text requires a non-empty arguments.text` |

Step 1 was a real, dispatched click (`dry_run: false`) that the step-level check passed -
`expected result observed: conversation`. Step 2 wanted to type and supplied no text, so
the marker was never entered. **No message was sent.**

### T05 - close the application

| # | run id | status | actions | execution_ms | what stopped it |
| --- | --- | --- | --- | --- | --- |
| 1 | `T05_20261003_174334` | failed | 0 | 1 718 | `step-1: 5 elements match 'x' (...); refusing to pick one arbitrarily` |
| 2 | `T05_20261003_180229` | blocked | 0 | 2 078 | `the success rule already holds on the untouched screen` |
| 3 | `T05_20261003_180511` | blocked | 0 | 1 109 | same |
| 4 | `T05_20261003_181018` | **timed_out** | 0 | 0 | the confirmation prompt was never answered |
| 5 | `T05_20261003_182259` | blocked | 0 | 2 391 | `the success rule already holds on the untouched screen` |
| 6 | `T05_20261003_182640` | failed | **1** | 12 906 | `task verification: still on screen but should be gone: WEEK4-OPEN-FILE-OK` |

Attempts 2, 3 and 5 were blocked because the frame did not contain the marker: the
test application was open but **not foregrounded**, so it was not in the capture. A
controlled check on this machine showed the difference directly:

```text
notepad open but behind other windows  -> frame contains WEEK4-OPEN-FILE-OK : NO
after bringing notepad to the foreground -> frame contains WEEK4-OPEN-FILE-OK : YES
```

Attempt 4 hid the operator's console window so it could not appear in the frame;
hiding it also hid the confirmation prompt, which then could not be answered, and the
run ended `timed_out` after its budget. Attempt 6 moved the console **off the virtual
screen** instead (still a real console, still answerable), foregrounded Notepad first,
and verified the marker was in frame before spending a model call. That attempt ran the
whole loop and dispatched one click at (2089, 25) on a 2560-wide screen. The close
control sits near x = 2540, so the click missed it, and the final observation still
carried Notepad's status-bar text (`Unix (LF)`, `100%`, `18:27`). The window was
minimised rather than closed - which the pass criteria in §"How success is judged"
explicitly exclude.

### What was not attempted

* T04 was never run with the operator agreeing to a real send beyond the one run above,
  which failed before typing anything. No message was ever sent on this machine.
* No case was re-run after its last failure to try to turn it into a success.

## Metrics

The definitions from the hand-off, so a reader does not have to guess what the
counts above mean:

```text
real-task success rate = runs whose goal rule passed / formal real attempts

mean execution time of successful tasks = sum(execution_ms of successes) / successes

mean time over all attempts = sum(execution_ms of all formal attempts) / attempts
```

### Computed from the runs above

| Metric | Value |
| --- | --- |
| formal real attempts | **11** (2 + 1 + 1 + 1 + 6) |
| successes (`succeeded` and `execute=true`) | **0** |
| real-task success rate | **0 / 11 = 0 %** |
| mean execution time of successful tasks | **undefined** - there were no successes |
| mean `execution_ms` over all attempts | **3 731 ms** |
| mean `planning_ms` over all attempts | **44 351 ms** |
| total actions actually dispatched | **5** (T01: 3, T04: 1, T05: 1) |
| actions that passed their step-level check | **4** of those 5 |
| runs that reached the goal rule at all | 2 (T01 attempt 2, T05 attempt 6) |

Two caveats on the timing figures, because they are easy to misread:

* `mean planning_ms` is inflated by memory pressure, not by the model. This machine had
  1.4-2.5 GB of free physical memory during the runs, so the 6 GB vision model was
  partly paged out; single planning calls took 88-111 s instead of the ~13 s the model
  achieves when memory is free. See `Document/Week3`'s timeout experiment for the same
  effect measured in isolation.
* `mean execution_ms` averages over runs that dispatched nothing. It is a measure of
  the harness, not of desktop control.

1. Dry runs and scripted runs are counted separately and never enter these
   numbers. `task_summary.json` records `execute`, so the split is checkable
   rather than remembered.
2. A case stopped by its own precondition is listed separately. It did not run, so
   it is not an attempt - and it is not quietly dropped either.
3. A run that started and then failed, timed out or was cancelled **is** a formal
   attempt, and is classified by its `status`.
4. `execution_ms` is the primary measure: from the confirmation gate to the final
   verdict. The operator's reading time is kept apart in `confirmation_ms`,
   planning in `planning_ms`, and the whole run in `elapsed_ms`, because counting a
   slow operator as a slow model is the easiest way to publish a meaningless
   average.
5. An action success rate, if quoted, has the number of actions actually dispatched
   as its denominator - not the number of tasks.
6. These are five controlled tasks on two machines, not a benchmark. Scale
   evaluation is Week 7.

## Per-case success rules

| Case | Counts as success | Does **not** count |
| --- | --- | --- |
| T01 | a browser window with an address bar or tab strip is in the foreground | a launcher shortcut pressed, the name typed into a search box, or another browser already in front |
| T02 | a results page is loaded and the query text is visible | text sitting in the input box without a submitted search |
| T03 | the file is open in an application | the file merely selected, or a same-named file opened from elsewhere |
| T04 | the unique marker appears as a sent message in the correct conversation | a draft, an old message with the same text, or a send to the wrong conversation |
| T05 | the target application's window is gone and other applications are untouched | the window minimised, or another window closed |

The right-hand column is the standard a human applies. The automatic rule is
weaker than it: it can only look for text on screen, so for T05 it cannot tell a
closed window from one that was never opened, and for T01 it cannot tell a browser
this run launched from one that was already there. The precondition check covers
that gap, which is why a row is only meaningful when its preconditions held.

If a run has to be verified by eye, mark the verification method `manual` in the
notes. An automatic check that did not run is not an automatic pass.

## Environment

Most of this is read off `task_summary.json` rather than remembered - the second
column says which field, so a row can be checked instead of trusted.

| Item | Value | Where it comes from |
| --- | --- | --- |
| Machine | | the summary's `platform` (`darwin` / `win32`) |
| OS and display | | `os_version` and `screen` |
| Commit | | `commit` |
| Model and endpoint | | `provider` and `model_name` |
| Context window (`OLLAMA_CONTEXT_LENGTH`) | | set on the server; not recorded, so write it down here |
| `timeout_seconds` | | `model.timeout_seconds` in the config the run used |
| Free memory before the run | | `memory_available_mb_before` in the run's `warmup.json` |
| Warmup: cold probe and warm repeat | | the `latency_ms` of the `cold-or-idle` and `warm` probes in the same file; 16.5.4 keeps it beside the run's timings and 13.4.1 keeps it out of them |
| Python | | `python_version` |

## Failure notes

Record each failure with its classification - focus wrong, target ambiguous, input
incomplete, not submitted, load timeout, verification insufficient - and the run
id, so the evidence can be found again.
