# Week 4 troubleshooting

Forty-two situations the closed loop can run into, in the order they tend to appear.
Each row says what to check first, and what this implementation actually does —
the second column matters, because a diagnostic guide that describes behaviour the
code does not have is worse than none.

Symptoms are grouped by where they surface: connection, model, perception,
resolution, execution, verification, recording.

## Connection

| Symptom | Check first | What this code does |
| --- | --- | --- |
| `Connection refused` / no response | IP, port, that the service is listening, that both machines are on the same subnet, firewall rule | `ModelError` with the endpoint in the message. The run is `blocked`, never `failed` — nothing was attempted |
| Service reachable, `--health-check` passes, but the model never sees the screen | Send a request **with** an image; check the model name and that it is a vision model | A text-only success is not evidence of vision. `ObservationService` always passes `image_path` from the captured frame |
| A long pause before the first reply | Whether the model is loaded (`ollama ps`), free system memory, how many other things are competing for it | First call loads the model. Warm the endpoint before a task; see the Windows manual step 5 |
| Requests are slow, or a timeout is suspected | Cold or warm state, free memory, concurrency, and how many retry layers are stacked | `model.timeout_seconds` is a **per-request** limit, not a task budget. Week 3 measured 8-10 s cold with memory free and about 72 s with it exhausted, and the slow part was the vision encoder paging, not model loading. A 72 s success obtained with a long probe timeout does **not** prove the default client would have timed out either - measure it through the client before drawing a conclusion |

The endpoint is read from `GUI_AGENT_BASE_URL` or `--base-url`. A missing API key is
reported before any network call, so a connection error is never a credential error
in disguise.

## Model

| Symptom | Check first | What this code does |
| --- | --- | --- |
| HTTP 500 | Service log, model and memory state, the request body | Reported as `ModelError` with the underlying exception name. **Kept separate from a timeout and from a truncated reply** — the three have different causes |
| A failing request takes about three times its `timeout_seconds` | The SDK's own retry policy as well as `model.max_retries` | The SDK is given `model.max_retries` now, so `0` means one attempt rather than one attempt plus two the SDK added on its own. `model_requests` counts what actually left the process |
| `400 ... request (N tokens) exceeds the available context size (M tokens)` | The server's context window against the prompt size. A 2560x1600 screenshot plus the element list measured 7 517 tokens; Ollama serves 4096 by default | `blocked` before any action. The message has the fix appended: raise `OLLAMA_CONTEXT_LENGTH` (16384 worked on the review machine) or lower `execution.max_elements`. This is a server setting, not a code path — nothing in the run can change it |
| `... N further elements omitted to fit the prompt` | How much text is on screen, and `execution.max_elements` | Not an error. The element list is trimmed one **whole** element at a time — an id with half its text would be unusable — and the prompt says how many were dropped. Raise `execution.max_elements` or the server's context window to keep more |
| Reply truncated mid-JSON | Finish reason, response length, **prompt length**, number of steps | `PlanParseError`; the planner retries the format once, then reports `blocked`. A partial plan is never executed |
| Plan is valid JSON but the steps do not fit the screen | Whether the prompt carried the element list | The adapter refuses at resolution time; nothing is clicked |
| `duplicate step_id` / `... is finish but N step(s) follow it` | The plan's step ids, and where `finish` sits | The plan is rejected at parse time and never executed. A step after the terminal step would otherwise be dispatched, because executability looks only at the verb |
| `the plan reports errors and will not be executed` | What the model put in the plan's `errors` list | `blocked` before any action. The model uses that field to say it could not work the task out, and running it would read "I am not sure" as "go ahead" |
| `no element matches '...'` | Take a fresh screenshot; check the OCR language and threshold; is the page still loading? | The step fails and the run stops. It does **not** click a default position, and it does not retry with a guessed target |
| `... is not from observation ... and the step names no text target` | Whether the model gave a `target_text` alongside an element id | A bare stale id cannot be re-located, so the run stops. A plan that gives both is re-bound automatically |

## Perception

| Symptom | Check first | What this code does |
| --- | --- | --- |
| `monitor_index 1 is out of range (available 1..0)` | Whether the display is asleep or locked | The run is `blocked` with that message and dispatches nothing. Wake the screen and re-run |
| Clicks land slightly off | Screenshot-to-control scale, any preprocessing resize, monitor offset | Coordinates go through `screenshot_to_control` and then an in-bounds check. **No per-task pixel offset is ever added** |
| OCR finds nothing on a clearly readable screen | Engine (`tesseract` vs `paddleocr`), language code, `min_confidence` | Measured on this machine: Tesseract 241 ms vs PaddleOCR 5 525 ms on the same frame. `configs/week4.yaml` selects Tesseract |
| OCR returns a fallback notice | Whether the primary engine started | The notice is recorded on the observation, and `ocr_engine` says which one actually ran |
| `the first frame had no readable text` | Whether the screen is locked, asleep or showing something with no text in it | Recorded as a note, not a failure: the capture worked and contour detection still found boxes, but OCR returned no labels, so no text target can resolve against that frame. Wake the screen and re-run |

## Resolution

| Symptom | Check first | What this code does |
| --- | --- | --- |
| `N elements match '...'` | Element ids, whether a region or context could disambiguate | The run stops and lists the candidates. **It never takes the first match** |
| `no element matches '...'` although the label is plainly on screen | Whether the backend returned words instead of lines — check the element texts in the observation JSON | Tesseract rows are merged back into lines, so `Summary (required)` is one element. If a phrase still spans two elements, the adapter matches the query's tokens against a run of neighbouring elements and refuses when more than one run matches |
| `N element runs match '...'` | How many places on screen carry that phrase | The run stops and lists the runs, same policy as an ambiguous single element |
| `scroll_amount must be a number` / `duration must be a number` | What the plan put in the argument | `ActionResolutionError`, so the run records a failed step. A bare `ValueError` here would have escaped the runner entirely and ended the run as a traceback |
| A dry run fails with `no element matches '...'` while you are using the computer | Whether the desktop changed between the two observations | Correct, and not a defect: the plan is written from one frame and every step is re-resolved against the next, so a target that scrolled away, closed or was covered is refused. Run the dry runs on a desktop you are not touching |
| The screen sleeps between the last action and the verdict | Whether the display went off | `inconclusive`, with `could not look at the screen to verify` as the reason. Nothing is claimed either way — a screen nobody can look at is not a passed task *or* a failed one |
| A step resolves to a point outside the monitor | The screenshot size against the control size | `ActionResolutionError`; the point is refused, not clamped to the edge |
| `key '...' is not in the allowed key set` | The key name the plan used | Only a fixed key list is accepted. This is deliberate: the model must not be able to drive a shell through `type_text` or `key_press` |

## Execution

| Symptom | Check first | What this code does |
| --- | --- | --- |
| Text lands in the wrong field | Which element the step targeted, and the focus after the preceding click | The action fails and the run stops. It does not continue typing, and it does not press Enter |
| Typing does not submit | Whether the plan contains the submit step, and what it defines as the trigger | A draft is not success. T02 requires the results page, not text in the box |
| Chinese or other non-ASCII input fails | `pyautogui.typewrite` and the current input method | Not supported and not claimed. `type_text` is ASCII-only; a clipboard-based path would need its own dependency and tests |
| The model presses something destructive | The plan, and the task's risk level | `medium` and `high` tasks ask for a second, separate confirmation, and it prints the text the plan will actually type. The risk policy is in the runner, so a caller cannot drop the prompt. There is no `--yes` |
| `--execute needs an interactive terminal` | Whether stdin is a terminal | `blocked`, exit code 2, before anything is captured. With nobody to ask, consent is not assumed. Run it from a terminal, or leave `--execute` off for a dry run |

## Verification

| Symptom | Check first | What this code does |
| --- | --- | --- |
| The model says it finished but the task did not | The verifier's evidence, the case's success rule | `finish` cannot mark a task complete. Only `Verifier.check_task` can, and only against the rule |
| A dry run reports `inconclusive` | Whether `--execute` was passed | Correct: nothing was dispatched. The rule's own verdict is kept in the evidence |
| A task with no success rule | Whether `--case` was used | `blocked` before any action. A free-form `--instruction` defines no rule |
| A real run stops with `the success rule already holds on the untouched screen` | Whether the screen is already in the goal state | `blocked` before planning, so no model call and no click. T05 only needs its marker gone, and T01 only needs `http` and `search` visible, so both are satisfiable without doing anything. Set the task's precondition up first. `require_preconditions=False` accepts the risk and is recorded in the run |
| A browser captcha or consent wall appears | The new screenshot | The run stops and records the obstacle. **No evasion step is ever added** |
| A "save changes?" dialog appears on close | Whether the test window holds unsaved content | T05's rule requires the window gone and other applications untouched. The run stops rather than discarding content |

## Recording

| Symptom | Check first | What this code does |
| --- | --- | --- |
| Two runs overwrite each other | The session id | Directories are `<case>_<timestamp>`. Observations are `obs-NNNN.json`; steps append to `steps.jsonl` |
| Statistics mix a dry run with a real run | Each summary's `execute` field | `status=succeeded` **and** `execute=true` are both required. A dry run is never counted |
| Typed content in the logs | `recorder.redact` | Text arguments are stored as `<redacted>(N chars)`. The record says something was typed, not what |
| A step has no evidence | Whether the observation was saved | Every observation writes its element list even when no image was saved, so a coordinate can always be traced to its frame |

## When recording a real problem

Record the reproduce command, the commit, the config, the steps, expected against
actual, the evidence path, the change and the regression result. "Improved the model
behaviour" is not a record.
