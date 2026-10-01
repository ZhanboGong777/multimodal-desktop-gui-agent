# Week 4: basic task test report

Five controlled tasks, run through `scripts/week4_agent_cli.py`. Each row must be
filled in from a real run; an unfilled row means "not measured", not "passed".

## How to fill a row

```bash
export GUI_AGENT_API_KEY=ollama
export GUI_AGENT_BASE_URL=http://<windows-host>:11434/v1

python scripts/week4_agent_cli.py --case T01 \
    --provider openai_compatible --model qwen2.5vl:7b --execute
```

Record the `Task status`, the verification outcome and the run directory. The
summary at `outputs/week4/<case>_<timestamp>/task_summary.json` has all three.

`status=succeeded` **and** `execute=true` are both required before a task counts as
a real success. A dry run is never a success, however clean it looks.

## Results

| Case | Task | Mode | Model | Status | Verification | Actions | Time | Run id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | open the browser | — | — | not run | — | — | — | — |
| T02 | search the web | — | — | not run | — | — | — | — |
| T03 | open a specified file | — | — | not run | — | — | — | — |
| T04 | send a message | — | — | not run | — | — | — | — |
| T05 | close the application | — | — | not run | — | — | — | — |

## Per-case success rules

| Case | Counts as success | Does **not** count |
| --- | --- | --- |
| T01 | a browser window with an address bar or tab strip is in the foreground | a launcher shortcut pressed, the name typed into a search box, or another browser already in front |
| T02 | a results page is loaded and the query text is visible | text sitting in the input box without a submitted search |
| T03 | the file is open in an application | the file merely selected, or a same-named file opened from elsewhere |
| T04 | the unique marker appears as a sent message in the correct conversation | a draft, an old message with the same text, or a send to the wrong conversation |
| T05 | the target application's window is gone and other applications are untouched | the window minimised, or another window closed |

If a run has to be verified by eye, mark the verification method `manual` in the
notes. An automatic check that did not run is not an automatic pass.

## Environment

| Item | Value |
| --- | --- |
| Machine | |
| OS and display | |
| Commit | |
| Model and endpoint | |
| `timeout_seconds` | |
| Free memory before the run | |

## Failure notes

Record each failure with its classification — focus wrong, target ambiguous, input
incomplete, not submitted, load timeout, verification insufficient — and the run id,
so the evidence can be found again.
