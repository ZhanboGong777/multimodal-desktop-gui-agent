# Week 4 evidence

Each directory here is named after a run id, and holds the text records
`scripts/week4_collect_evidence.py` copied for that run: `task_summary.json`,
`steps.jsonl`, `run_config.json`, and `warmup.json` when the run had one.
Screenshots and per-frame observation files are deliberately **not** copied - they
are a picture of the whole desktop. `outputs/` is not tracked, so this directory is
what makes a run id quoted in a report resolve on another machine.

## Not every directory here is task evidence

Read the summary before treating one as a result:

| In `task_summary.json` | What it is |
| --- | --- |
| `"execute": true` **and** `"provider"` is a real backend | a real attempt, and the only kind that can count towards the five-task results |
| `"execute": false` | a dry run: actions were resolved and validated, nothing was dispatched |
| `"provider": "mock"` | the model's judgement was simulated; this proves the pipeline, not the agent |

A run can be both - a dry run against a mock model, which is what
`T01_20261001_215642` is. It is kept because it is the first record produced
against a **real screen**: Tesseract read 60 elements from the live desktop, the
plan chose `Edge` from the prompt, and the adapter resolved it against a *later*
frame with the note `re-bound from obs-0001-e000 in an earlier frame`. That is the
"every step is re-located on the current frame" rule being exercised against real
capture and real OCR rather than scripted frames.

The five task rows in `Week4_Basic_Task_Test_Report.md` stay `not run` until runs
with `"execute": true` exist. A directory appearing here is not a success.
