# Week 4 usage: running the closed loop

The Week 4 entry point is `scripts/week4_agent_cli.py`. It performs one task:
observe the screen, plan, resolve each step against the current frame, act,
observe again and verify.

**Dry run is the default.** `--execute` is required for real mouse and keyboard
events, and it does not skip the confirmation prompt. There is deliberately no
`--yes`: sending a message without a human reading it first is not a feature.

## Quick start

```bash
cd /path/to/multimodal-desktop-gui-agent
source .venv/bin/activate

# what can be run
python scripts/week4_agent_cli.py --list-cases

# a dry run: resolves and validates, dispatches nothing
python scripts/week4_agent_cli.py --case T01

# a demonstration of the whole loop with no desktop and no model
python scripts/week4_offline_demo.py
```

Real execution, once the desktop is in a known state:

```bash
export GUI_AGENT_API_KEY=ollama
export GUI_AGENT_BASE_URL=http://<windows-host>:11434/v1

python scripts/week4_agent_cli.py --case T01 \
    --provider openai_compatible --model qwen2.5vl:7b \
    --execute
```

## Command line

| Flag | Meaning | Default |
| --- | --- | --- |
| `--config` | configuration file | `configs/week4.yaml` |
| `--instruction` | free-form task | prompts nothing; needs `--case` for a rule |
| `--case` | one of `T01`–`T05` | none |
| `--list-cases` | print the cases and exit | |
| `--provider` | `mock`, `openai_compatible`, `langchain` | from config |
| `--model` | model name | from config |
| `--base-url` | endpoint | from config or `GUI_AGENT_BASE_URL` |
| `--execute` | allow real desktop actions | off |
| `--max-actions` | action cap for this run | 20 |
| `--task-timeout` | wall-clock budget, seconds | 240 |
| `--output-directory` | where records are written | `outputs/week4` |

Exit codes: `0` completed (check the task status), `1` failed, `2` blocked or bad
arguments, `3` timed out, `130` cancelled.

## The five task cases

| Case | Task | Risk |
| --- | --- | --- |
| T01 | open the browser | low |
| T02 | search the web for a given query | low |
| T03 | open a specified file | low |
| T04 | send a message to a test conversation | high |
| T05 | close the test application | medium |

Each case carries its own success rule. A task whose rule cannot be checked is
reported `blocked` rather than treated as finished, and `--instruction` without
`--case` defines no rule at all.

`T04` and `T05` ask for a second, separate confirmation because they are not
reversible by looking at the screen.

## What a run leaves behind

```
outputs/week4/<case>_<timestamp>/
├── obs-0001.json        # elements, geometry and timing for one frame
├── obs-0002.json        # ... one file per observation, never overwritten
├── steps.jsonl          # one line per step, appended
└── task_summary.json    # status, verification, action count, timings
```

Each observation file records the frame-local element ids, so a coordinate in
`steps.jsonl` can be traced back to the frame it was resolved from.

Typed text is redacted in the step log: the record notes that something was
typed and how long it was, never what it said.

## Safety

- Nothing is dispatched unless `--execute` is passed, and the confirmation prompt
  is answered.
- Every action is validated twice: the adapter checks the target exists in the
  current frame and maps the coordinate, and the executor checks the resulting
  point is inside the captured monitor.
- A stale element id from an earlier frame is refused, not silently rebound.
- Keyboard input is restricted to a fixed key set; the model cannot drive a shell
  through `type_text`.
- The run stops after `max_actions`, at `task_timeout`, on any action error, and
  on `Ctrl+C`.
- A changed screenshot is not success. Only the task's own success rule can mark
  a run `succeeded`.
