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

The checked-in evidence now contains **38 run directories**: one mock dry run
and **37 real attempts** (`execute=true` with `provider=openai_compatible`).
Five real runs have `status=succeeded` and `verification.outcome=passed`:
`T01_20261004_140000`, `T01_20261004_213859`, `T02_20261004_210409`,
`T03_20261004_171941`, and `T05_20261004_180657`. That is **4/5 task cases**
(T01/T02/T03/T05), with two successful T01 runs; T04 has no successful run.
The 37 attempts split as T01 6, T02 17, T03 2, T04 3, T05 9. A directory
appearing here is not a success; read the summary's execution mode and verdict.
See `../Week4_Basic_Task_Test_Report.md` for the reconciled results and metrics.

`live_observation.json` preserves the earlier **darwin observation baseline**,
including its historical task definitions. It is not a current five-task result;
its scope and current-results metadata distinguish it from the Windows runs.
Existing run-directory records remain unchanged by the target-rendering repair.
The supplied review's `T04_20261005_192417` diagnostic is not checked in here and
is excluded from these counts. No new desktop run was performed for the repair.
