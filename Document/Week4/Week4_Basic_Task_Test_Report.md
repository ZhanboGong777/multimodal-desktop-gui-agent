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
| T01 | open the browser | desktop visible and launch entry uncovered; **no browser window open** | 2 + 5 | **0 + 1** | **succeeded** | automatic: the marks a browser frame carries in either locale (`http`, `搜索`) | from the confirmation gate to the final verdict (`execution_ms`); the summary also keeps `planning_ms`, `confirmation_ms` and the whole-run `elapsed_ms`, and the run directory keeps `warmup.json` for the cold and warm numbers, which are deliberately not part of any of them | first round: `T01_20261003_172642`, `T01_20261003_180109`; second round, the success: **`T01_20261004_140000`** |
| T02 | search the web | a browser window is open and focused | 1 + 38 | **1** | **succeeded** | automatic: the query text **and** a loaded results page (`google.com`) | as above | first round: `T02_20261003_173909`; the second round's runs are listed under §"Second round" and eleven of them are in `Document/Week4/evidence` |
| T03 | open a specified file | `week4_sample.txt` exists in the week4 test folder (`~/Desktop/week4_test`, or `%USERPROFILE%\Desktop\week4_test`); no file of that name is open | 1 + 7 | **1** | **succeeded** | automatic: `WEEK4-OPEN-FILE-OK` on screen | as above | first round: `T03_20261003_174110`; the success: **`T03_20261004_171941`** |
| T04 | send a message | the test conversation is open and holds no earlier message with **this run's** marker; the operator agreed a real message may be sent | 2 | **0** | timed_out, with the plan's actions all dispatched and no verdict recorded | automatic: this run's marker on screen (the CLI prints it as `marker`) | as above | `T04_20261003_174709` (failed, before any of this branch's fixes); **`T04_20261005_003044`** dispatched all four actions and recorded no verdict - see the note under this table |
| T05 | close the application | `week4_sample.txt` is open in the test application, so the marker is on screen; window focused | 6 | **0** | failed (1 `timed_out`, 3 `blocked`, 2 `failed`) | automatic: `WEEK4-OPEN-FILE-OK` **gone** | as above | `T05_20261003_174334`, `T05_20261003_180511`, `T05_20261003_181018`, `T05_20261003_182259`, `T05_20261003_182640`; the sixth attempt, `T05_20261003_180229`, has no directory here - its evidence was not collected and the run no longer exists on the machine that produced it, so the attempt is recorded without it rather than cited to nothing |

**These are the first real runs of the five cases on any machine.** Every attempt is
listed below with the outcome that was actually written; nothing was deleted, and the
dry-run passes in §"Dry runs" are not counted as attempts or successes.

**T01 succeeded on the second round.** On this machine, with a real click:

```text
run_id       = T01_20261004_140000
status       = succeeded
execute      = True
verification = passed
action_count = 1
step-1 double_click: resolved=True verif=passed err=None
```

Its records are in `Document/Week4/evidence/T01_20261004_140000`.

**T03 succeeded on the third round.** Also with a real click, and also on the first step:

```text
run_id       = T03_20261004_171941
status       = succeeded
execute      = True
verification = passed  (all success rules matched against the current screen)
action_count = 1
step-1 double_click: resolved=True element=obs-0002-e042 click=(401,272) verif=passed
```

Notepad was found running with `*week4_sample.txt` in its title afterwards, which is the
success rule stated as an observation rather than as a verdict. Its records are in
`Document/Week4/evidence/T03_20261004_171941`.

**T02 succeeded on the last round**, with two actions where every earlier attempt had managed
at most one:

```text
run_id       = T02_20261004_210409
status       = succeeded
execute      = True
verification = passed  (all success rules matched against the current screen)
action_count = 2
step-1 type_text  resolved=True verif=passed
step-2 key_press  resolved=True verif=passed
```

Its frames show the page change as well as the verdict, which matters because this case once
produced a false success: the window title read `'新标签页 - Google Chrome'` for `obs-0001`
to `obs-0004`, the query alone appearing from `obs-0003`, and then **`'GUI agent research -
Google 搜索 - Google Chrome'`** for `obs-0005` and `obs-0006` - the results page, confirmed by
a signal that does not go through OCR at all. Its records are in
`Document/Week4/evidence/T02_20261004_210409`.

**T05 succeeded** after three earlier attempts, its close action correct throughout and a
save dialog absorbing it until the document was clean:

```text
run_id       = T05_20261004_180657
status       = succeeded
verification = passed  (all success rules matched against the current screen)
step-1 hotkey keys=['alt','f4'] note='alt+f4'
```

**Four of the five cases have reached `succeeded`** - T01, T02, T03 and T05. The fifth, T04,
is not described as passing anywhere in this report, and its single attempt is the one the
operator agreed to run: it failed before typing anything and **no message was ever sent**.

What the four have in common is worth stating, because three of them are the same defect seen
three times: **the harness refused something a person would call correct.** T01's shortcut was
clicked on its label instead of its icon, and the click landed on the label's text rather than
the icon above it. T03's file was refused because the OCR engine read `week4_sample.txt` as
`week4 sample.txt` - an underscore rendered as a space, which is what happens to a one-pixel
glyph sitting on the baseline. And T02's plan ran out after one action because the prompt had
never said a plan may hold more than one step, while its own `expected_result` described the
finished search.

T05 is the fourth and a different shape: `pyautogui.hotkey('alt', 'f4')` does nothing at all
on Windows, silently, while the same keys pressed by hand in three separate calls close the
window. In every one of the four, **the model had read the screen correctly and the machinery,
not the model, was what failed.**

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

### Second round - the same five cases, after the fixes

Branch `fix/week4-windows-rerun`. Same machine, same model. Eleven real runs
(`execute=true`), and the result is unchanged in the only column that counts: **no run
reached `succeeded`**. What changed is how far each one got and what stopped it.

| run id | status | actions | what stopped it | what it establishes |
| --- | --- | --- | --- | --- |
| `T01_20261004_000322` | failed | 1 | `task verification: expected on screen but not found: http, search` | the plan used `click`; the icon was selected, not launched |
| `T01_20261004_000434` | failed | 1 | same | the plan used **`double_click`** and resolved the shortcut - the verb fix worked, the window still did not open |
| `T02_20261004_001545` | failed | 0 | `step-1: no element matches 'Microsoft Edge' in obs-0002` | the two OCR elements were joined (`resolved=True`) and the failure moved one step later |
| `T02_20261004_005017` | failed | 1 | `step-2: no element matches 'search bar' in obs-0004` | **two** steps dispatched, step 1 verified |
| `T02_20261004_005225` | failed | 1 | `task verification: expected on screen but not found: GUI agent research` | `type_text resolved=True verif=passed` - the text was typed into the frame the run was given |
| `T01_20261004_004112` | **blocked** | 0 | `the run assumes no browser is running, but these are: msedge.exe, chrome.exe` | the new process precondition fired before the capture: `planning_ms = 0.0`, no `frames/`, no `obs-NNNN.json` |
| `T02_20261004_161652` | failed | **2** | `task verification: expected on screen but not found: GUI agent research` | **both** steps resolved and verified - a `type_text` and a `key_press Enter` - and the frame showed the text in the address bar with its autocomplete open, so Enter selected a suggestion instead of submitting |
| `T02_20261004_185721` | failed | 2 | `task verification: expected on screen but not found: google.com/search` | **the goal was reached and the verdict could not see it.** The frame is a real Google results page - `google.com/search?q=GUI+agent+research&oq=...&gs_lcrp=...`, a tab titled `GUI agent research - Google`, an AI overview and arXiv results - and the element list for that frame holds neither the URL nor the title. The search was submitted (steps 1 and 2 both verified) and landed (obs-0005 carries both the query and the results page). What failed is the reading, not the doing |

Reading that last row, and the distinction is not a consolation: **the agent did the task and
the rule could not confirm it.** Three candidate markers were measured against three frames,
and every one of them goes through the same OCR pass:

| frame | query text | tab title | url host |
| --- | --- | --- | --- |
| the real results page (`obs-0005`) | not read | **read** | **read** |
| the next frame of that same run (`obs-0010`) | **read** | not read | not read |
| the frame that had falsely passed | **read** | not read | not read |

The only signal present in every frame is the one that cannot tell a submitted search from
text waiting in an address bar. So on this machine T02 has a demonstrated goal and no
reliable verdict, and `Week4_Troubleshooting.md` names the three changes that would alter
that: a different screen size, a different OCR engine, or the window title read through the
accessibility API rather than through OCR.

The table above is the round's turning points, not every run. T02 produced 30 attempts in
the second round; eleven of them are in `Document/Week4/evidence` and the rest stayed in
`outputs/week4`, which is where the metric block's 61 comes from.

The last two rows are the informative pair. The blocked run is the only verdict in the
round that is *correct by construction* rather than a failure that happened to be useful,
and it costs 1 828 ms instead of the 30 531 ms a doomed run previously spent on planning.
The two-step run is the furthest T02 has reached: the prompt's missing "a plan may hold
more than one step" rule was the reason every earlier attempt stopped after one action.

Two mechanisms were found by instrumenting the driver rather than by reading the code,
and both are written up in `Week4_Troubleshooting.md`:

1. **Minimising a window activates it in z-order terms, and the next window down is the
   desktop.** A driver that minimises its own console to keep it out of the frame
   therefore pushes the case's target behind the desktop, and every capture returns
   icons. `IsIconic` on the target's handle is the diagnostic; a window at
   `(-21333, -21333)` with `IsIconic=True` was never going to be in the frame, and
   `SetForegroundWindow` alone does not bring it forward - `ShowWindow(SW_RESTORE)` has
   to come first.
2. **Focusing the console cancels the minimise.** The launcher reported its console
   handle and then called `SetForegroundWindow` on it, undoing the driver's minimise a
   fraction of a second later. That is why the sequencing read correctly and the frames
   kept showing the desktop.

With both fixed the observation finally contained a desktop - and then contained this
agent's own conversation instead:

```text
obs-0001-e000 -> 'Chat'
obs-0001-e008 -> '1 background job running'
obs-0001-e005 -> 'Trajectory'
step-1 type_text resolved=True verif=passed
```

The run typed into the chat window it was being observed from. That is not a defect in
the case definitions, the prompt or the resolver: **the agent cannot act on a screen that
does not contain itself, and it cannot remove itself from that screen from inside the
run.** On this machine there is one monitor (`\\.\DISPLAY1`, 1707x1067), so no local
window arrangement avoids it. The five cases need a session whose own window is not on
the captured display - another device observing the run, or a headless client.

Every attempt above is kept, including the ones that failed for reasons already known.
None was deleted or rewritten, and no run is described as a success it did not record.

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
| formal real attempts | **82** (T01 18, T02 38, T03 8, T04 1, T05 16) |
| successes (`succeeded` and `execute=true`) | **4** (`T01_20261004_140000`, `T02_20261004_210409`, `T03_20261004_171941`, `T05_20261004_180657`) |
| real-task success rate | **4 / 82 = 4.9 %** |
| mean execution time of successful tasks | **7 254 ms** (6 938 T01, 7 016 T02, 5 313 T03, 9 750 T05) |
| mean `execution_ms` over all attempts | **11 014 ms** |
| mean `planning_ms` over all attempts | **119 304 ms** |
| total actions actually dispatched | **54** |
| actions that passed their step-level check | **48** of those 54 |

One verdict is **excluded** from the counts above and named here rather than deleted:
`T02_20261004_182646` recorded `succeeded` and was wrong. Its own frames show the query
sitting in the address bar with the autocomplete dropdown open and no search submitted, and
the rule it satisfied then asked only for the query to be somewhere on screen. The rule was
changed because of it, that run is in `outputs/week4` but never entered the evidence
directory, and counting it would inflate the rate by a fifth.

**`planning_ms` describes the harness, not the node.** Every figure above was produced through
a driver that re-asserted the foreground window and sampled it every 0.5-0.8 s, and those two
loops starve the HTTP read waiting on the model: the same payload plans in **~200 s** through
that driver and in **34 s cold / 4 s warm** without it. The numbers are left as they happened
- they are what those runs cost - but they should not be read as a property of this machine.
| runs that reached the goal rule at all | 2 (T01, T03) |

Every number above is a count over the directories under `outputs/week4`, restricted to
summaries that record `execute: true`, so anyone with the repository can reproduce it with
`scripts/week4_collect_evidence.py`'s own directory. The first round's table recorded 11 of
the attempts that existed at the time; the rest were repeats of the same failure, kept in
the directory rather than tabulated. The raw directory is the larger, checkable number and
is the one quoted here.

The distribution matters more than the rate: **T02 accounts for 31 of the 61 attempts**,
because it was the case being worked on, and **T03 and T04 have one attempt each** - from
the very first round, before any of this branch's fixes existed. T04 in particular is a
single attempt that failed before typing anything, and no message was ever sent.

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

**T04 is the one case with no recorded verdict, and the evidence for what happened is mixed.**
The run `T04_20261005_003044` is recorded `timed_out`, and every one of its four actions was
dispatched and reported success:

    type_text  resolved=True success=True note='type 35 chars (text taken from target_text...)'
    click      resolved=True success=True note="text '发送' -> obs-0004-e035"
    type_text  resolved=True success=True
    click      resolved=True success=True

So the plan reached its send step, twice, which no earlier T04 run managed. What it did **not** do
is satisfy the task rule: four passes ran `task verification: expected on screen but not found:
WEEK4_MESSAGE_CHECK_20261005_003044`, and the run timed out before the checker ever passed.

**Whether the message was sent is therefore not established**, and the table says `timed_out`
rather than `succeeded` for that reason. What is visible in the frames is a sent message in
文件传输助手 whose text reads `WEEK_MESSAGE_CHECK_20261005_003044WEEK4_MESSAGE_CHECK_20261005_0`,
but the OCR reads that field unreliably - thirty-one observations of one run with the client in
front returned zero reads of a marker sitting in the message box - so a frame that appears to show
it sent is not proof that this run's message was the one that arrived, and the strict reading is
the only one the report can carry.

**The lesson this case paid for, and it is a measurement one.** T04's message box is invisible to
the OCR while its conversation is readable, so a marker in the element list cannot be attributed to
one or the other, and every check in this branch assumed the box. That led to a **sent** message
being read as a draft, and to "no message has been sent" being reported while a message had been.
The two are told apart by **position** - the conversation is right-aligned and higher, the message
box is the full-width field at the bottom - never by the string, which is identical in both.
