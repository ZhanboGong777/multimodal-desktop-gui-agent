# Week 3 Experiment Report: Public GUI Datasets and the Base Agent Framework

## 1. Task objective

Week 3 turns the Week 2 perception and control modules into the base of an agent:
public GUI datasets are read and normalised, a provider-independent model
interface is added, and an instruction can be decomposed into a validated plan.

No model is trained this week. LoRA fine-tuning is Week 5; Week 3 builds the data
and the plumbing that Week 5 will consume. The end-to-end "instruction ->
perception -> plan -> execute -> feedback" loop is Week 4: Week 3 stops at the
plan.

## 2. Module design

```text
src/gui_agent/
├── datasets/
│   ├── schemas.py        GUITaskSample, GUIActionStep
│   ├── base.py           DatasetAdapter, read_records
│   ├── normalize.py      action vocabulary, text and number helpers
│   ├── screenagent.py    desktop trajectories
│   ├── mind2web.py       web trajectories
│   ├── webarena.py       task definitions
│   └── validation.py     validate, summarise, export, re-read
├── models/
│   ├── base.py           ModelClient, ModelRequest, ModelResponse
│   ├── mock.py           deterministic offline backend
│   ├── openai_compatible.py  any OpenAI-compatible endpoint
│   └── langchain_adapter.py  the same endpoint, through LangChain
└── planning/
    ├── schemas.py        TaskPlan, PlanStep, PLAN_ACTION_TYPES
    ├── prompts.py        system prompt and user turn
    ├── parser.py         JSON recovery plus validation
    └── planner.py        TaskPlanner, PlanResult

scripts/
├── week3_prepare_dataset.py
├── week3_model_demo.py
└── week3_planning_demo.py
```

## 3. Experiment environment

| Item | Development machine | GPU node |
| --- | --- | --- |
| Device | MacBook Air M2, 24 GB | Lenovo Y9000P, RTX 4060 Laptop 8 GB |
| OS | macOS 26.3.1 ARM64 | Windows 11 |
| Python | 3.12.10 | 3.12.4 |
| New dependencies | `datasets`, `langchain`, `langchain-openai` (`requirements-agent.txt`) | same |
| Week 2 modules | unchanged, 165 tests still pass | same |

## 4. Implementation process

1. Define the unified schema before any adapter, so the adapters have a target.
2. Split each adapter into a pure `to_sample()` and a shared `read_records()`.
3. Add validation and the JSONL round trip before writing the CLI.
4. Add the model interface with the mock backend first, so everything downstream
   is testable offline.
5. Build the planner on top, with strict validation and no execution path.
6. Finish with the three demonstration scripts.

## 5. Dataset processing results

The pipeline was exercised end to end on a fixture that mirrors WebArena's shape:

```text
$ python scripts/week3_prepare_dataset.py --dataset webarena \
    --input tests/fixtures/week3/webarena_sample.json \
    --output data/processed/webarena_sample.jsonl --limit 5 --split test

dataset : webarena
读取样本数: 5
成功转换数: 5
跳过样本数: 0
输出文件位置: data/processed/webarena_sample.jsonl  (5 行)
回读验证: 5 个样本通过 Schema 校验
```

The exported record carries the unified shape; unrepresentable source fields are
preserved rather than dropped:

```json
{
  "sample_id": "1",
  "dataset_name": "webarena",
  "instruction": "What is the top rated product?",
  "image_path": null,
  "actions": [],
  "metadata": {
    "start_url": "http://shop.test",
    "sites": ["shopping"],
    "eval": {"reference_answers": {"exact_match": "Widget"}}
  }
}
```

A record with no instruction is rejected and counted, not silently accepted: the
fixture contains one such record and `--validate-only` reports it.

### Real archives

The adapters were then run over the actual downloads.

**WebArena** (`test.raw.json`, 812 tasks):

```text
读取样本数: 20   成功转换数: 20   跳过样本数: 0
回读验证: 20 个样本通过 Schema 校验
```

**ScreenAgent** (`test.zip`, 898 records, 48 MB):

```text
读取样本数: 200   成功转换数: 200   跳过样本数: 0
动作类型统计:
  planaction: 64    click: 51      key_press: 9    type_text: 8
  drag: 5           scroll: 3      move: 3         wait: 2
  double_click: 2
回读验证: 200 个样本通过 Schema 校验
```

The ScreenAgent result is the more informative one, because the **first** run
converted nothing at all: the fixtures had guessed the field names, and the real
archive uses others. See section 9.

## 6. Model interface results

| Backend | Network | API key | Deterministic | Used by |
| --- | --- | --- | --- | --- |
| `MockModelClient` | no | no | yes | tests, both demos, offline runs |
| `OpenAICompatibleClient` | yes | `GUI_AGENT_API_KEY` | no | a real model, hosted or local |
| `LangChainClient` | yes | `GUI_AGENT_API_KEY` | no | the same models, called through LangChain |

**Screenshots are sent as real image data.** The payload uses the OpenAI vision
shape - a `text` block plus an `image_url` block holding a base64 `data:` URL -
attached to the last user turn. An earlier version embedded the image *path* in
the text payload, so the model received a filename and no picture. That failure
mode is the dangerous kind: the request succeeds and the model answers about
nothing. The client now raises instead when an image is missing, empty, of an
unknown type, or larger than 20 MB.

This also makes a split setup practical, which suits the available hardware: the
model runs on the Windows GPU node behind an OpenAI-compatible endpoint and the
Mac, which has no discrete GPU, only issues HTTP requests. The hand-off asks for
exactly this ("call it through a separate service process") and it removes the
PaddlePaddle/PyTorch conflict entirely, since the two stacks no longer share a
process - or a machine.

Measured on the mock backend, which performs no inference:

| Measurement | Value |
| --- | --- |
| Call latency | 0.7 ms |
| Health check | ok, offline |
| Steps in the returned plan | 2 |

The honest reading: the mock's latency measures the plumbing, not a model. The
real backend is exercised by a single controlled call; a full measurement would
need a paid endpoint or a locally served vision model, which the hand-off does not
require this week.

### The LangChain backend

The outline asks for the framework to be built "on LangChain/LlamaIndex". The
first version of this week satisfied the architectural half of that requirement -
a provider-independent `ModelClient` - but not the literal one: `langchain` was
listed in `requirements-agent.txt` and never imported. `LangChainClient` closes
that gap without weakening the abstraction.

The design rule is that LangChain stays a *backend*, not a dependency of the
system. Only `langchain_adapter.py` imports it; `planning/`, `datasets/` and the
CLI continue to see `ModelClient` and nothing else. A LangChain upgrade, or its
removal, cannot reach the planner, which is exactly why the interface exists.

| Aspect | Choice |
| --- | --- |
| Provider name | `langchain` (`--provider langchain`) |
| Underlying wrapper | `langchain_openai.ChatOpenAI` |
| Credentials | identical to the direct backend: `GUI_AGENT_API_KEY`, `GUI_AGENT_BASE_URL` |
| Vision payload | built by the shared `to_vision_messages()`, then wrapped in `HumanMessage` |
| Retries | `max_retries=0` on `ChatOpenAI`; `ModelClient` owns the retry policy so both backends behave identically |
| Errors | any LangChain exception becomes a `ModelError`, never a raw traceback |

Two details are worth recording. First, the vision payload is not re-implemented:
the adapter reuses the OpenAI client's encoder and only translates the resulting
messages into LangChain objects, so the two backends cannot drift apart on how an
image is attached. Second, `ChatOpenAI` is constructed lazily, on the first call,
which is what lets `--provider langchain` fail with "`GUI_AGENT_API_KEY` is not
set" instead of a connection error.

**How it was verified.** The Windows GPU node was offline when this was written,
so the backend was not checked against `qwen2.5vl:7b`. Instead the whole path was
exercised against a real OpenAI-compatible server running on localhost, with the
genuine `langchain-openai` client on the other side of a real socket:

```text
tests/test_langchain_integration.py::test_langchain_sends_a_real_vision_request
tests/test_langchain_integration.py::test_usage_is_reported_back
tests/test_langchain_integration.py::test_the_planner_runs_unchanged_on_the_langchain_backend
3 passed
```

The assertions are about bytes that crossed the socket, not about a stub's
arguments: the request carries `model: qwen2.5vl:7b`, a `system` turn, and an
`image_url` block holding a `data:image/png;base64,` URL, and the raw file path
does not appear anywhere in the payload. The third test runs `TaskPlanner`
unchanged on the LangChain backend and gets a validated plan back, which is the
claim that matters - the planner does not know which backend it is talking to.

The CLI was checked the same way, against the same kind of local server:

```text
$ python scripts/week3_planning_demo.py --provider langchain --model qwen2.5vl:7b \
    --instruction "Open the browser and search for GUI agents"

provider   : langchain (qwen2.5vl:7b)
attempts   : 1
steps      : 5 (4 executable)
confirm    : True

  step    action      target        description
  step-1  click       browser icon  Click the browser icon on the desk
  step-2  click       address bar   Click the address bar
  step-3  type_text   address bar   Type the search query
  step-4  key_press   address bar   Press Enter
  step-5  finish      -             Done  (terminal)
nothing was executed: Week 3 produces plans, Week 4 executes them
```

`langchain_adapter.py` is at 100% statement coverage on both machines.

**The real-model check was then run on the Windows node**, and it produced the
result that actually matters for this design. One instruction and one real
screenshot (VS Code, 1707x1067) went through both backends:

| Backend | attempts | steps | Plan | Latency |
| --- | --- | --- | --- | --- |
| `langchain` | 1 | 3 (3 executable) | `click 浏览器` / `type_text GUI agents` / `click First search` | 74 110 ms (cold) |
| `openai_compatible` | 1 | 3 (3 executable) | byte-for-byte identical | 11 868 ms |

The plans are identical. That is the claim worth testing here - not "LangChain
works", which was never in doubt, but "the backends are interchangeable and the
planner cannot tell them apart".

The 74 s is not wrapper overhead: it is the first call loading a 6 GB model into
VRAM. Re-running with the model resident gave 14 082 ms, the same order as the
direct backend's 11 868 ms.

One hardware characteristic is worth recording, because it bounds what this setup
can do. `qwen2.5vl:7b` (6 GB) does **not** fit entirely in the RTX 4060's 8 GB:
`ollama ps` reports `5.9 GB  26%/74% CPU/GPU`, so roughly a quarter of the layers
run on the CPU. Setting `num_ctx` anywhere between 1024 and 8192 did not change
the allocation - Ollama still reported `CONTEXT 4096` - which makes this a hard
constraint of model size against VRAM rather than a tuning mistake. Throughput
settles at about 30 tok/s, putting one planning call at 12-14 s. No thermal or
power throttling was active (50 C, 7.78 W, `SW Power Cap: Not Active`).

## 7. Planning results

```text
$ python scripts/week3_planning_demo.py --provider mock \
    --instruction "Open the browser and search for GUI agents"

provider   : mock (mock-vision-model)
attempts   : 1
steps      : 2 (1 executable)
confirm    : True

  step    action      target        description
  step-1  click       agents        Open the browser and search for GUI agents
  step-2  finish      -             report the result and stop  (terminal)

plan saved : outputs/week3/plans/mock-task.json
nothing was executed: Week 3 produces plans, Week 4 executes them
```

Two properties matter more than the output above:

- **`finish` is a planning verb only.** `PLAN_ACTION_TYPES` is the control layer's
  vocabulary plus `finish`; `executable_steps` excludes it, so a terminal marker
  can never reach the executor.
- **Nothing is executed.** `allow_real_execution` is recorded as `false` on the
  result and no module in `planning/` imports the control layer.

Rejection paths were tested as carefully as the happy path: unknown action verbs,
a plan above `max_steps`, a reply with no JSON, an empty reply, and a backend that
is down. Each produces a clear error and no plan; the format retry is bounded at
one.

## 8. Test results

Both machines, same suite:

| Machine | Result |
| --- | --- |
| MacBook Air M2 | **285 passed**, 89% coverage, ruff clean |
| Lenovo Y9000P (Windows) | **281 passed** in 47.75 s, ruff clean (at `73e65dd`) |

The two machines agreed exactly at `73e65dd`: 281 each. The Mac then closed the
Mind2Web gap and added four regression tests, moving to 285. Those four have not
been re-run on Windows, and that is stated rather than left for a reader to spot.

Getting to agreement is itself part of the record. The Windows run stood at 254 for
a while, because it predated the vision payload and the LangChain backend. That was
a point on the same line, not a disagreement - but it is exactly the kind of gap
that gets quietly smoothed over in a report, so it is stated instead.

Agreement is the point of running both, because two defects only ever appeared on
the second machine - and neither would have been found by reading the code.

```text
pytest      : 285 passed
coverage    : 89% over src/gui_agent
ruff        : All checks passed!
```

| Test file | Covers |
| --- | --- |
| `test_dataset_schemas.py` | required fields, unknown fields, action normalisation, geometry |
| `test_dataset_adapters.py` | all three adapters, tolerant field lookup, reader formats |
| `test_dataset_validation.py` | validation, statistics, JSONL round trip |
| `test_model_mock.py` | deterministic plans, retry bounds, log-safe response view |
| `test_model_config.py` | provider selection, credentials from the environment |
| `test_plan_parser.py` | JSON recovery, unknown verbs, step limits |
| `test_planner.py` | end-to-end planning, determinism, failure paths |
| `test_prepare_dataset_cli.py` | the preparation CLI, its flags and its output files |
| `test_langchain_adapter.py` | message conversion, credentials, error mapping |
| `test_langchain_integration.py` | the LangChain backend over a real HTTP socket |

Automated tests never touch the network, the desktop or a real API key.

### Mind2Web

| Attempt | Result |
| --- | --- |
| `osunlp/Mind2Web`, split `test` | rejected: available splits are `['train']` |
| `osunlp/Multimodal-Mind2Web`, split `test` | rejected: available splits are `train`, `test_domain`, `test_task`, `test_website` |
| `osunlp/Multimodal-Mind2Web`, split `test_task`, streaming | no record within five minutes |
| `osunlp/Multimodal-Mind2Web`, split `test_task`, **one parquet shard fetched directly** | **268/268 rows converted, 0 validation issues** |

Streaming was the wrong tool. The archive is stored as parquet shards, so a single
shard can be fetched on its own (296 MB) and read locally in seconds - no streaming,
no 13.6 GB download. This closed the last gap in the dataset work, and it did more
than that: running the adapter on real rows found two defects that the fixture could
not, and a third piece of missing data.

**Defect 1: the verb is the last field, not the first.** A real repr reads
`"[textbox]  US City,State or Zip Code -> TYPE: 08817"` - tag, element text, then
the operation. Splitting on the first space made the HTML tag the verb and the rest
the target:

| | `action_type` values across the shard |
| --- | --- |
| Before | `[button]` 600, `[link]` 427, `[input]` 262, `[span]` 247 ... |
| After | `click` 218, `type_text` 44, `move` 4, `enter` 2 |

Every action type was an HTML tag name. The schema accepted them, because
`normalize_action` deliberately passes unknown verbs through rather than rejecting
them - so validation reported 268/268 valid on data where not one action had a
usable verb.

**Defect 2: one row is one step, not one task.** `action_reprs` holds the entire
task; `target_action_index` says which of those steps the row is. Treating the list
as the trajectory multiplied every task by the length of its own action list. The
shard is 268 rows covering 36 tasks, so the adapter was emitting whole trajectories
per row - about 14 fabricated steps per sample.

**Defect 3: the screenshot struct was dropped.** `screenshot` is
`struct<bytes: binary, path: string>`, not a string, so `image_path` came out
`None`. The shard alone carries 302.7 MB of embedded JPEGs that were being discarded.

The lesson is sharper than "run the adapter on real data". A hand-written fixture
encodes the author's *belief* about the schema, so it cannot find the case where
that belief is wrong: the Mind2Web fixture used the invented shape `"CLICK
[Submit]"` - verb first - which is precisely what the adapter assumed. The two
agreed, and both were wrong. Only the archive could settle it. Five regression tests
now pin the real format, including one that reads the shard's actual reprs.

**The `.gitignore` silently excluded the whole model layer.** A bare `models/`
matches at any depth, so `src/gui_agent/models/` was never committed. Ignored
files do not appear in `git status`, so the Mac reported a clean tree while every
clone failed with `ModuleNotFoundError: No module named 'gui_agent.models'`. Caught
only when the Windows machine pulled the branch. Fixed by anchoring the
generated-output patterns to the repository root.

**pytest could not create its temporary directory on Windows.** `WinError 5` on
`%TEMP%/pytest-of-<user>` stopped 27 tests before any of them ran. The account name
is not ASCII and the folder had been left locked or half-removed; the same thing
happened in Week 2 with 12 errors. `--basetemp` now points inside the repository.

Neither defect was visible from the code, and neither would have been found on a
single machine.

### The real model, end to end

The model runs on the Windows GPU node and the Mac only issues HTTP requests. The
final configuration is `qwen2.5vl:7b` behind Ollama on the RTX 4060, reached at
`http://172.16.114.104:11434/v1`.

A real instruction against a real screenshot produced a valid, sensible plan:

```text
instruction: Open the browser and search for multimodal GUI agents
model      : qwen2.5vl:7b   (100% GPU, 7887 of 8188 MiB)

  step-1  click       browser        Open the browser
  step-2  type_text   search bar     Type 'multimodal GUI agents' in the search bar
  step-3  key_press   search bar     Press Enter to search
  step-4  click       first result   Click on the first search result
  step-5  finish      browser        Finish the task          (terminal)

assumptions            : ["Ambiguous search term"]
requires_confirmation  : True
executed               : nothing
```

The plan is semantically correct, ends with the terminal verb, records its own
assumption and is saved to `outputs/week3/plans/<task_id>.json`. The raw artefact
is kept in `Document/Week3/evidence/real_model_plan_qwen25vl_7b.json`.

**A prompt problem, found only against a real model.** The first attempts failed
with "no JSON object found in the response" even though the reply visibly began
with `{`. The cause was the prompt, not the parser: the system prompt was long
enough that a 7B model kept talking past the point where the JSON ended, and the
response was truncated mid-object. Cutting `SYSTEM_PROMPT` from 1199 to 792
characters - keeping the field list and the JSON example, dropping the prose around
them - fixed it immediately, and the first attempt then succeeded.

The Mock backend could never have surfaced this: it returns a fixed dictionary, so
prompt length has no effect on it.

### The real model, split across two machines

The model runs on the Windows GPU node and the Mac only issues HTTP requests, which
suits the available hardware and removes the PaddlePaddle/PyTorch conflict entirely.
The Mac reached the endpoint on the first attempt, on the same subnet:

```text
Mac 172.16.114.87/26  ->  Windows 172.16.114.104/26
GET /v1/models        ->  {"data": [{"id": "qwen2.5vl:3b", ...}]}
```

A real multimodal call then succeeded through the project's own client:

| Measurement | Value |
| --- | --- |
| Image | 520 x 160 PNG, synthetic, text `SAVE-BUTTON-913` |
| Model answer | `SAVE-BUTTON-913` |
| Latency | 2.9 s, including base64 encoding |
| Result | correct |

**The 3B model cannot read a real desktop screenshot.** The same endpoint fails
with `500 prediction aborted, token repeat limit reached` on a 1920 x 1080 desktop
capture. The failure was isolated deliberately:

| Variation | Result |
| --- | --- |
| Text-only planning, simple instruction | valid JSON, 5 steps |
| Text-only planning, compound instruction | valid JSON, 5 steps |
| Synthetic 520 x 160 image | correct reading |
| Real 1920 x 1080 screenshot | 500 |
| Same screenshot downscaled to 1024 / 768 / 512 | 500 |
| Same screenshot with `num_ctx` 8192 / 16384 | 500 |

Image size and context window are therefore not the cause: the model is simply not
capable enough for a cluttered full-screen capture.

The obvious next step - a larger vision model - was taken on the Windows machine,
and the result is recorded here because it was not the expected one. `llava:7b`
(4.7 GB, the manual's first choice) reads text off a screenshot (PASS, 3.1 s) and
describes it accurately (PASS, 4.4 s), yet it does not emit structured output at
all. Shown a prompt containing a JSON example, it treated that example as
something *in the picture* and narrated it - "The interface is displaying a JSON
object with a task ID, instruction, summary, steps..." - and it kept doing so when
told to output only a JSON object with no prose. The reply's first character was
not `{` and `json.loads()` failed outright. Larger is not the same as more
obedient: `qwen2.5vl:7b` was chosen because it follows the format instruction, not
because it is the biggest model available.

This is a result, not a blocker, and it is worth recording plainly: **a small
vision model can pass a synthetic smoke test and still be unusable on real
screens.** The synthetic verification the Windows machine ran was necessary but
not sufficient.

## 9. Problems and handling

**The record reader shredded pretty-printed JSON.** It switched to line-by-line
parsing whenever the text contained a newline. A formatted JSON file is not JSONL,
so every inner line carried a trailing comma and was dropped: a six-record fixture
yielded one record. The reader now parses the whole document first. Two regression
tests cover it.

**The mock emitted a field the schema forbids.** Strict validation rejected it,
which is the schema doing its job. The fix was to remove the field rather than to
loosen `TaskPlan`.

**The planner conflated rules with the task.** Passing the whole rendered prompt
as the instruction made an echoing backend return the template. Rules now travel
in the system turn and the instruction stays a task.

**A test waited on a real network timeout.** The failing-backend test used the
default base URL; it now targets a closed local port with a 0.5 s timeout.

**The ScreenAgent adapter was written against guessed field names and converted
nothing.** The fixtures assumed `instruction` / `image` / `task_id`. Running the
adapter over the real `test.zip` read 898 records and converted **zero**: the
archive uses `task_prompt`, `saved_image_name`, and no task id at all. Worse, the
shape of `actions[]` was wrong too - each entry is tagged by `action_type` and the
four families present (`MouseAction`, `KeyboardAction`, `PlanAction`,
`EvaluateSubTaskAction`) carry completely different payloads, only the first two of
which are desktop actions.

The adapter was rewritten against the archive: 200/200 records now convert, the
mouse and keyboard families map onto real verbs with coordinates and bounding
boxes, and evaluations are excluded from `actions` and kept in
`metadata.evaluations` so they cannot inflate a trajectory.

The lesson is worth recording: **a hand-written fixture encodes what the author
believes the schema to be.** It cannot discover that the belief is wrong. One run
over the real archive found in minutes what the fixtures would never have found.

## 10. Final result

| Criterion | Result |
| --- | --- |
| Three adapters validated against their real archives | yes, 268/268 for Mind2Web |
| One validated JSONL export | yes, round-trips through the schema |
| `--limit` supported, no full download by default | yes, default 20 |
| Raw data and images kept out of Git | yes, `data/` ignored |
| Mock backend runs offline | yes |
| One real multimodal backend | implemented; a live call needs `GUI_AGENT_API_KEY` |
| Instruction becomes a structured TaskPlan | yes |
| Plan validated and never executed | yes |
| Framework built on LangChain, per the outline | yes, `LangChainClient`, same `ModelClient` |
| New and existing tests pass, Ruff clean | 285 passed, ruff clean |
| README, WORKLOG and report updated | yes |

## 11. Deliverables

Code: the dataset adapters and schema, the preparation CLI, the model interface
with three interchangeable backends, the planning module, three demonstration
scripts, ten test files and updated configuration.

Documents: this report, `Document/Week3/WORKLOG.md` and
`Document/Week3/Week3_Dataset_Notes.md`.

## 12. Next week plan

1. Wire the planner output into the Week 2 executor behind an explicit
   confirmation flag, completing the Week 4 loop.
2. Run the adapters over the official ScreenAgent and WebArena archives and
   correct any field variants the fixtures did not cover.
3. Decide how to store the Mind2Web screenshots: one shard alone carries 302.7 MB
   of embedded JPEGs, and Week 5 will need them on disk.
4. Decide whether the 26% CPU spill of `qwen2.5vl:7b` is acceptable for Week 4's
   interactive loop, or whether to trial `minicpm-v` (5.5 GB) as a smaller fit.
5. Keep the dataset sample limit small until Week 5 needs training splits.
