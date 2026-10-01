# Week 4: basic task test report

Five controlled tasks, run through `scripts/week4_agent_cli.py`. Every row has to
be filled in from a real run. An unfilled row means "not measured" - it never
means "passed", and a dry run is never a success however clean it looks.

## How to fill a row

```bash
export GUI_AGENT_API_KEY=ollama
export GUI_AGENT_BASE_URL=http://<windows-host>:11434/v1
export OLLAMA_CONTEXT_LENGTH=16384        # before the server starts

python scripts/week4_agent_cli.py --case T01 \
    --provider openai_compatible --model qwen2.5vl:7b --execute
```

Then put the evidence where the report can point at it:

```bash
python scripts/week4_collect_evidence.py --latest T01
```

`outputs/` is not tracked, so a run id on its own resolves to nothing on any other
machine. The collector copies the two text records - `task_summary.json` and
`steps.jsonl` - into `Document/Week4/evidence/<run id>/`, and deliberately leaves
the screenshots and per-frame observations behind: they are a picture of the whole
desktop.

`status=succeeded` **and** `execute=true` are both required before a task counts as
a real success.

## Results

Attempts and successes are counted per case: a run that had to be repeated after a
failure is two attempts, and only the runs whose summary reads `succeeded` with
`execute=true` are successes.

| Case | Task | Preconditions before the run | Attempts | Successes | Status | Verification method | Timing basis | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | open the browser | desktop visible and launch entry uncovered; **no browser window open** | 0 | 0 | not run | automatic: `http` and `search` both on screen | whole run, first observation to final verdict, including the confirmation prompt (`elapsed_ms`) | — |
| T02 | search the web | a browser window is open and focused | 0 | 0 | not run | automatic: the query text is on screen | as above | — |
| T03 | open a specified file | `week4_sample.txt` exists in the week4 test folder; no file of that name is open | 0 | 0 | not run | automatic: `WEEK4-OPEN-FILE-OK` on screen | as above | — |
| T04 | send a message | the test conversation is open and holds no earlier message with the marker; the operator agreed a real message may be sent | 0 | 0 | not run | automatic: `WEEK4_MESSAGE_CHECK_001` on screen | as above | — |
| T05 | close the application | `week4_sample.txt` is open in the test application, so the marker is on screen; window focused | 0 | 0 | not run | automatic: `WEEK4-OPEN-FILE-OK` **gone** | as above | — |

### Why the preconditions are not optional

A real run checks its own starting state before it plans anything. T01 and T05 are
both satisfiable without doing anything - a browser that was already running
carries the text T01 looks for, and T05 only asks that the marker be gone, which is
already true while the window is shut. If either precondition is unmet the run is
reported `blocked` with `the success rule already holds on the untouched screen`,
and no model call is made and no click is dispatched.

That is a correct outcome, not a failure: it means the screen could not have shown
whether this run did the work. Set the precondition up and run it again.

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

| Item | Value |
| --- | --- |
| Machine | |
| OS and display | |
| Commit | |
| Model and endpoint | |
| Context window (`OLLAMA_CONTEXT_LENGTH`) | |
| `timeout_seconds` | |
| Free memory before the run | |

## Failure notes

Record each failure with its classification - focus wrong, target ambiguous, input
incomplete, not submitted, load timeout, verification insufficient - and the run
id, so the evidence can be found again.
