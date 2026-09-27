# Week 3 Work Log

Week 3 topic: public GUI dataset processing and the base Agent framework.
Branch: `week3-dataset-agent`.

## Summary

| Item | Result |
| --- | --- |
| New source modules | 18 (`datasets/` 8, `models/` 5, `planning/` 5) |
| New scripts | 3 |
| New test files | 10 |
| Tests | 281 passed |
| Ruff | clean |
| Week 1 + Week 2 regression | all previous tests still pass |

## Tasks

### 1. Dataset schemas (`src/gui_agent/datasets/schemas.py`)

`GUITaskSample` and `GUIActionStep` are the project's own shape. The three source
datasets have completely different field names, nesting and action vocabularies;
everything downstream - validation, export, the planner - works on these two
structures, so an adapter only translates and never reshapes.

Two deliberate choices:

- **`action_type` is a free string, not the control layer's literal union.** The
  sources use verbs this project does not implement (`SELECT`, `TYPE` with a
  different meaning). An unknown verb is normalised and kept, with the untouched
  original in `raw_action`, so one odd record cannot fail a whole sample.
- **Images are paths, never bytes.** An exported JSONL stays small and the
  screenshots remain separate artefacts.

### 2. Adapters (`datasets/base.py`, `screenagent.py`, `mind2web.py`, `webarena.py`)

An adapter has two halves on purpose:

- `to_sample()` is **pure** - one raw record in, one `GUITaskSample` out. That is
  where the semantic mapping lives, and it is unit tested against hand-written
  fixtures with no download.
- `read_records()` does the I/O - JSON, JSONL or a zip of either - so a fixture
  and a real download travel the same code path.

Field lookups are defensive (`first_present(record, "instruction", "task", "goal", ...)`)
because the exports differ between splits and versions. WebArena carries a task
definition rather than a trajectory, so its samples legitimately have no actions.

### 3. Validation and export (`datasets/validation.py`)

`validate_sample` reports what makes a sample unusable. An empty action list is
*not* a problem - WebArena tasks have none and are still valid planning input.
`write_jsonl` is followed immediately by `read_jsonl`, which rebuilds the models
through Pydantic, so a corrupt export fails loudly instead of quietly.

`collect_stats` produces the counts the CLI prints: read, converted, skipped,
missing-field histogram and action vocabulary.

### 4. Preparation CLI (`scripts/week3_prepare_dataset.py`)

Supports `--dataset --input --output --split --limit --seed --validate-only`, plus
`--shuffle` and `--stats-json`. `--limit` defaults to 20: Week 3 validates the
pipeline on small samples rather than downloading whole datasets.

### 5. Model interface (`src/gui_agent/models/`)

`ModelClient` is the only surface business code sees. `ModelResponse` carries
content, provider, model name, latency, usage and an error, and its `as_dict()`
deliberately omits the raw provider payload so credentials cannot reach a log.

Two backends:

- `MockModelClient` - rule-based, deterministic, offline, free. It is the default
  and the test oracle.
- `OpenAICompatibleClient` - any OpenAI-compatible endpoint. Credentials come from
  `GUI_AGENT_API_KEY` / `GUI_AGENT_BASE_URL` / `GUI_AGENT_MODEL` and never from the
  config file.

  A screenshot is sent as a real vision block: the file is read, base64-encoded and
  attached to the last user turn as `{"type": "image_url", "image_url": {"url":
  "data:image/png;base64,..."}}`. The first version put the *path* in the text
  payload instead, which meant the model could not see the image at all - a
  multimodal backend that silently answers about nothing is worse than one that
  fails. Unknown suffixes, empty files and images above 20 MB are refused with a
  clear error rather than dropped from the request.

Retries are bounded, and a failed call returns a `ModelResponse` with `error` set
rather than raising into the caller.

### 6. Planning (`src/gui_agent/planning/`)

`TaskPlan` and `PlanStep` with Pydantic validation. The action vocabulary is the
control layer's plus exactly one verb: `finish`, which marks the end of a plan and
is excluded from `executable_steps`, so it never reaches the executor.

`parse_plan` recovers JSON from markdown fences or surrounding prose and then lets
Pydantic decide; anything that still fails is reported, never silently repaired.
`TaskPlanner` allows one controlled format retry and stops there. It executes
nothing.

### 7. Demonstrations

- `scripts/week3_prepare_dataset.py` - dataset to JSONL with statistics.
- `scripts/week3_model_demo.py` - model call with a log-safe response view.
- `scripts/week3_planning_demo.py` - instruction to validated plan, saved under
  `outputs/week3/plans/`.

Both demos take `--provider` from `sorted(CLIENTS)` rather than a hard-coded
list, so registering a backend is enough to make it selectable. The list was
hard-coded and went stale the moment a third backend appeared.

### 8. LangChain backend (`src/gui_agent/models/langchain_adapter.py`)

The outline asks for the framework to be built on LangChain/LlamaIndex. The first
cut of this week satisfied the architectural half of that - a provider-independent
`ModelClient` - but not the literal one: `langchain` sat in
`requirements-agent.txt` and was never imported. This adapter closes the gap.

- Registered as provider `langchain`; `create_model_client` now calls
  `factory(**common, base_url=...)` instead of branching on two hard-coded types.
- Wraps `langchain_openai.ChatOpenAI`. Credentials, environment variables and the
  vision payload are shared with the direct backend, not re-implemented: the
  adapter calls `OpenAICompatibleClient.to_vision_messages()` and only converts
  the resulting dicts into `SystemMessage` / `HumanMessage` / `AIMessage`.
- `max_retries=0` is set on `ChatOpenAI` so `ModelClient` keeps ownership of the
  retry policy and the two remote backends stay comparable.
- Every LangChain exception is re-raised as `ModelError`; no LangChain type leaks
  past this module, and nothing in `planning/` or `datasets/` imports it.

## Bugs found and fixed while building

**The Mind2Web adapter read the verb from the wrong end of the string.** A real
`action_reprs` entry is `"[textbox]  US City,State or Zip Code -> TYPE: 08817"`:
tag, element text, then the operation. Splitting on the first space made the HTML tag
the verb, so every `action_type` in the shard came out as `[button]`, `[link]`,
`[input]`. Validation still reported 268/268 valid, because `normalize_action` passes
unknown verbs through by design - a green result over data with no usable verbs.

**The Mind2Web adapter treated one row as one task.** `action_reprs` is the whole
task and `target_action_index` selects the step the row actually is. Expanding the
list multiplied every task by its own action count: the shard's 268 rows cover 36
tasks, and the adapter emitted whole trajectories per row.

**`screenshot` is a struct, not a string.** `struct<bytes: binary, path: string>`, so
`image_path` came out `None` and 302.7 MB of embedded JPEGs in one shard were dropped.

All three were found by running the adapter over a real parquet shard, after the
fixture had been passing for a week. The fixture could not have found them: it used
the invented shape `"CLICK [Submit]"`, which is what the adapter assumed. **A fixture
written by the same person from the same belief cannot falsify that belief.**

**The record reader shredded pretty-printed JSON.** The first version switched to
line-by-line parsing whenever the text contained a newline. A formatted JSON file
is not JSONL, so every inner line carried a trailing comma and was dropped: a
six-record fixture produced one record. The reader now parses the whole document
first and only falls back to JSONL. Two regression tests cover it.

**The mock violated the plan schema.** It emitted an `image_path` key that
`TaskPlan` does not declare. Strict validation rejected it, which is the schema
working as intended - the fix was to remove the field, not to relax the model.

**The planner conflated rules with the task.** It passed the entire rendered
prompt as the instruction, so a backend that echoes the instruction back returned
the template instead of the task. Instructions and rules are now separate: rules
travel in the system turn.

**A test reached the real network.** The failing-backend test used the default
base URL and waited for a 60 s timeout. It now points at a closed local port and
uses a 0.5 s timeout.

**The ScreenAgent adapter was written against guessed field names.** The fixtures
used `instruction` / `image`; the real archive uses `task_prompt` /
`saved_image_name`. The first run over `test.zip` read 898 records and converted
none. It now converts 200/200, with the four real `action_type` families mapped and
`EvaluateSubTaskAction` excluded from the trajectory. WebArena converted 20/20 on
the first attempt. Details are in the dataset notes.

**The `.gitignore` was hiding an entire package.** A bare `models/` in the
template ignore file matches a directory of that name at *any* depth, so it also
matched `src/gui_agent/models/` - and Git never reported it, because ignored files
do not appear in `git status`. Every test passed on the machine that created the
files and every clean clone failed at import with
`ModuleNotFoundError: No module named 'gui_agent.models'`. The generated-output
patterns are now anchored to the repository root with a leading slash.

**pytest could not create its temporary directory on Windows.** The default base
directory is `%TEMP%/pytest-of-<user>`. On the Windows machine, whose account name
is not ASCII, that folder twice came back as `WinError 5 拒绝访问` before a single
test ran: a locked or half-removed directory pytest could not rotate. First seen
in Week 2 with 12 errors, then again here with 27. `pyproject.toml` now sets
`--basetemp=.pytest-tmp`, a gitignored directory inside the repository.

Both were found by running the suite on the second machine, not by reading the
code. The first would have shipped a repository that could not be cloned and used.

## Known limitations

- All three adapters have now been exercised against their real archives. Mind2Web
  was the last to be closed and the most informative: see the bugs below.
- The LangChain backend has now been exercised against the real `qwen2.5vl:7b` on
  the Windows node, and the two remote backends returned byte-identical plans for
  the same instruction and screenshot. What remains unmeasured is not correctness
  but comfort: `qwen2.5vl:7b` does not fit in 8 GB of VRAM, so about 26% of its
  layers run on the CPU and one planning call takes 12-14 s once loaded (74 s
  cold). Whether that is fast enough for Week 4's interactive loop is an open
  question, not a defect.
- The coverage figure depends on `pytest-cov`, which lives in
  `requirements-dev.txt` and not in `requirements-agent.txt`. A machine that
  follows the agent requirements alone cannot reproduce `--cov`; the Windows node
  hit exactly this and had to install it separately.
- The real-model path needs `GUI_AGENT_API_KEY`; it cannot be exercised in
  automated tests, so the mock backend is the one the test suite proves.
