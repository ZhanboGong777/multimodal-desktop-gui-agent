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
│   └── openai_compatible.py  any OpenAI-compatible endpoint
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

```text
pytest      : 245 passed
coverage    : 88% over src/gui_agent
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

Automated tests never touch the network, the desktop or a real API key.

### Mind2Web

| Attempt | Result |
| --- | --- |
| `osunlp/Mind2Web`, split `test` | rejected: available splits are `['train']` |
| `osunlp/Multimodal-Mind2Web`, split `test` | rejected: available splits are `train`, `test_domain`, `test_task`, `test_website` |
| `osunlp/Multimodal-Mind2Web`, split `test_task`, streaming | no record within five minutes |

The Mind2Web adapter is therefore covered by fixtures only. The other two sources
were validated against their real archives; this one was not, and the report says
so rather than implying otherwise.

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
| Three adapters with minimal sample tests | yes |
| One validated JSONL export | yes, round-trips through the schema |
| `--limit` supported, no full download by default | yes, default 20 |
| Raw data and images kept out of Git | yes, `data/` ignored |
| Mock backend runs offline | yes |
| One real multimodal backend | implemented; a live call needs `GUI_AGENT_API_KEY` |
| Instruction becomes a structured TaskPlan | yes |
| Plan validated and never executed | yes |
| New and existing tests pass, Ruff clean | 245 passed, ruff clean |
| README, WORKLOG and report updated | yes |

## 11. Deliverables

Code: the dataset adapters and schema, the preparation CLI, the model interface
with both backends, the planning module, three demonstration scripts, six test
files and updated configuration.

Documents: this report, `Document/Week3/WORKLOG.md` and
`Document/Week3/Week3_Dataset_Notes.md`.

## 12. Next week plan

1. Wire the planner output into the Week 2 executor behind an explicit
   confirmation flag, completing the Week 4 loop.
2. Run the adapters over the official ScreenAgent and WebArena archives and
   correct any field variants the fixtures did not cover.
3. Complete the single real-model call and record its latency and output.
4. Add the LangChain adapter as one more `ModelClient`; the planner needs no change.
5. Keep the dataset sample limit small until Week 5 needs training splits.
