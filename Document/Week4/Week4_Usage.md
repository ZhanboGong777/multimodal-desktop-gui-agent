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
export OLLAMA_CONTEXT_LENGTH=16384     # set before the server starts

python scripts/week4_agent_cli.py --case T01 \
    --provider openai_compatible --model qwen2.5vl:7b \
    --execute
```

Instead of exporting them, copy `.env.example` to `.env` and fill it in: the CLI
reads `./.env` itself. A variable already exported in the shell wins over the file.
The order is explicit flag, then the environment (including `.env`), then the YAML,
then the built-in default - so `GUI_AGENT_MODEL` does override `model_name` in the
config. The Week 2 and Week 3 scripts do not read `.env`; for those, export the
variables or source the file.

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
reversible by looking at the screen. It prints the text the plan will actually
type, so the decision is about the message and not about the instruction. The risk
level that triggers it lives in the runner, not in the CLI.

`--execute` needs a terminal. With nobody to ask, the run is `blocked` (exit code
2) before anything is captured: a missing answer is never read as consent.

## What a run leaves behind

```
outputs/week4/<case>_<timestamp>/
├── obs-0001.json        # elements, geometry and timing for one frame
├── obs-0002.json        # ... one file per observation, never overwritten
├── steps.jsonl          # one line per step, appended
└── task_summary.json    # status, verification, action count, timings, provenance
```

`task_summary.json` also carries where and when the run happened - commit,
platform, Python version, screenshot and control geometry, start and finish times,
planning attempts, transport requests and the failing step - so a row in the test
report can be read without asking the machine it came from. `planning_attempts` and
`model_requests` are separate counters: one plan can cost several requests.

Each observation file records the frame-local element ids, so a coordinate in
`steps.jsonl` can be traced back to the frame it was resolved from.

Nothing under `outputs/` is tracked, so a run id quoted in a report resolves to
nothing anywhere but this machine. To make one travel with the repository:

```bash
python scripts/week4_collect_evidence.py --latest T01
# -> Document/Week4/evidence/T01_<timestamp>/{task_summary.json,steps.jsonl}
```

The collector copies those two text records and nothing else. The screenshots and
the observation files are a picture of the whole desktop; `--no-steps` narrows it
to the summary alone.

Typed text is redacted in the step log: the record notes that something was
typed and how long it was, never what it said.

## When something goes wrong

`Document/Week4/Week4_Troubleshooting.md` lists twenty-six symptoms - from a refused
connection to a save-changes dialog - with what to check first and what the code
does about each. The two most common while setting up:

- `monitor_index 1 is out of range (available 1..0)` - the display is asleep. The
  run reports `blocked` and dispatches nothing; wake the screen and re-run.
- `400 ... exceeds the available context size` - the screenshot plus the element
  list did not fit the server's window. This is a server setting, not a code path:
  raise it (Ollama: `OLLAMA_CONTEXT_LENGTH=16384`) before starting the server. The
  failure message carries the same instruction.
- `the success rule already holds on the untouched screen` - the task's goal is
  true before anything ran, so the run is `blocked` without a model call. Set the
  task's precondition up first; see the safety note below.
- `no element matches '...'` - either the text really is not on the screen, or it
  arrived split across elements. The step fails and the run stops rather than
  clicking a default position.

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
- A real run refuses to start when its success rule already holds on the untouched
  first frame, so a task cannot be credited with a state it did not create. This
  matters for T01 and T05, whose rules are satisfiable without doing anything.
  Only tasks that declare preconditions are checked, and dry runs are exempt
  because their verdict is inconclusive by construction.
