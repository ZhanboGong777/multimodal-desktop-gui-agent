# Week 3 Dataset Notes

Sources, download entry points, field mappings and the limits found while writing
the adapters. Raw data is never committed: `data/raw/` and `data/processed/` are in
`.gitignore`, and only hand-made fixtures live under `tests/fixtures/week3/`.

## Download entry points

| Dataset | Entry point | Size | Week 3 use |
| --- | --- | --- | --- |
| ScreenAgent | `https://github.com/niuzaisheng/ScreenAgent/raw/refs/heads/main/data/ScreenAgent/test.zip` | small | primary sample source |
| ScreenAgent (repo) | `https://github.com/niuzaisheng/ScreenAgent` | - | reference |
| Multimodal-Mind2Web | `https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web` | ~13.6 GB | stream a few samples |
| Mind2Web | `https://huggingface.co/datasets/osunlp/Mind2Web` | ~6.74 GB | metadata-only alternative |
| Mind2Web (code) | `https://github.com/OSU-NLP-Group/Mind2Web` | - | reference |
| WebArena tasks | `https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json` | small | task definitions |
| WebArena (repo) | `https://github.com/web-arena-x/webarena` | - | reference |

All URLs above were checked and answered HTTP 200/206 at the time of writing.

## Recommended order

1. ScreenAgent `test.zip` - desktop trajectories, closest to this project.
2. WebArena `test.raw.json` - task definitions, single small file.
3. Stream a handful of Mind2Web samples.
4. Only if needed: training splits or the human-trajectory archive.

## Field mapping

### ScreenAgent

**Verified against the real `test.zip`** (898 records). Each file is one *step* of
a session, not a whole task. Every record carries `task_prompt`, `send_prompt`,
`status`, `video_width/height`, `saved_image_name`, `current_task` and `actions`.

| Source | Unified |
| --- | --- |
| `task_prompt` | `instruction` |
| `saved_image_name` | `image_path` |
| `current_task` | `observation` and `metadata.current_task` |
| `video_width` / `video_height` | `metadata` |
| `status` | `metadata.status` |
| `actions[]` | `GUIActionStep[]`, except evaluations (below) |
| record basename | `sample_id` |

`actions[]` entries are tagged by `action_type`, and the four values seen in the
archive are not the same kind of thing:

| `action_type` | Payload | Mapped to |
| --- | --- | --- |
| `MouseAction` | `mouse_action_type`, `mouse_button`, `mouse_position`, `clickable_area` | `click` / `double_click` / `right_click` / `move` / `drag` / `scroll` + `coordinates` + `target_bbox` |
| `KeyboardAction` | `keyboard_action_type`, `keyboard_text`, `keyboard_key` | `type_text` / `key_press` / `hotkey` + `input_text` |
| `PlanAction` | `element` | verb kept verbatim, target in `target_text` |
| `EvaluateSubTaskAction` | `situation`, `advice` | **dropped from `actions`**, kept in `metadata.evaluations` |

An evaluation is not a step: counting it would inflate every trajectory. Across
400 sampled records the archive contains 240 evaluations, 128 plan actions, 102
mouse actions and 74 keyboard actions.

### Mind2Web

| Source | Unified |
| --- | --- |
| `confirmed_task` / `instruction` | `instruction` |
| `actions[]` (dicts) or `action_reprs[]` (strings) | `GUIActionStep[]` |
| `operation` / `action` | `action_type` (normalised) |
| `element.text` / `element` | `target_text` |
| `value` | `input_text` |
| `website`, `domain`, `subdomain` | `metadata` |

One row is one **step**, not one task: `action_reprs` holds the whole task and
`target_action_index` selects which step the row is. An entry reads
`[textbox]  US City,State or Zip Code -> TYPE: 08817` - an element tag, the element
text, then the operation at the *end*. Splitting on the first space makes the tag the
verb, which is exactly what the first version of this adapter did.

### WebArena

| Source | Unified |
| --- | --- |
| `intent` / `instruction` | `instruction` |
| `start_url` / `url` | `metadata.start_url` |
| `sites` | `metadata.sites` |
| `eval` / `reference_answers` | `metadata.eval` |
| - | `actions` stays empty: WebArena ships no trajectory |

## Limits found

1. **Mind2Web is too large to download casually.** 13.6 GB with screenshots, and
   streaming is slower than the size suggests (see 6). One Parquet shard fetched
   directly - 296 MB, 268 rows over 36 tasks - is enough to validate the adapter, and
   is what Week 3 used. The preparation script takes `--limit` for the same reason.
2. **WebArena's value is its evaluation format, not its screenshots.** It defines
   tasks, start pages and expected outcomes; running it needs a full Dockerised
   website stack, which Week 3 explicitly does not deploy.
3. **Field names vary between splits and versions.** Every lookup therefore tries
   several spellings before giving up, and the untouched record is kept so nothing
   is lost when a guess is wrong.
4. **Action vocabularies do not line up.** ScreenAgent and Mind2Web both use
   `CLICK`/`TYPE`, but with different payloads, and WebArena uses verbs this
   project does not implement. Unknown verbs are normalised, kept verbatim in
   `raw_action`, and never cause a sample to be dropped.
5. **The fixtures were wrong about ScreenAgent, and the real archive proved it.**
   The hand-written fixtures used `instruction` / `image` / `task_id`. The real
   records use `task_prompt` / `saved_image_name` and have no task id at all: the
   first run over `test.zip` read 898 records and converted **zero**. The adapter
   was rewritten against the archive and now converts 200/200 with a full action
   vocabulary (`planaction` 64, `click` 51, `key_press` 9, `type_text` 8, `drag` 5,
   `scroll` 3, `move` 3, `wait` 2, `double_click` 2). Hand-written fixtures cannot
   substitute for one run over the real data.
6. **Mind2Web streaming is slower than the file size suggests.** With `datasets`
   installed, both `osunlp/Mind2Web` and `osunlp/Multimodal-Mind2Web` accept
   `streaming=True`, and their split names differ from the obvious guess:

   | Repository | Streaming splits |
   | --- | --- |
   | `osunlp/Mind2Web` | `train` only |
   | `osunlp/Multimodal-Mind2Web` | `train`, `test_domain`, `test_task`, `test_website` |

   Streaming `test_task` did not deliver its first record within five minutes: the
   archive is sharded, and a shard has to arrive before the iterator yields anything.
   Downloading one shard directly sidesteps this entirely, and that is what closed the
   gap: **268/268 rows converted, 0 validation issues**. Running it also exposed three
   adapter defects the fixture had been agreeing with - the verb read from the wrong
   end, one row treated as a whole task, and the `screenshot` struct dropped - all
   fixed, with five regression tests pinning the real format.

7. **Parquet is binary, and the reader treated every input as text.** Validating the
   adapter directly gave 268/268. Handing the same file to
   `scripts/week3_prepare_dataset.py` read six lines of mojibake, converted nothing,
   wrote an empty file **and still exited 0** - so the documented command looked like
   it worked. `read_records()` now reads Parquet in batches, and an export that
   produced no samples exits 1. Adapter-level and command-level results now agree,
   which is the only reason the command can be handed to someone else.

## Local layout

Raw downloads and exports live inside the repository under `data/`, which Git
ignores (see the anchored `/data/raw/` and `/data/processed/` rules):

```text
<repo>/data/raw/{screenagent,mind2web,webarena}/
<repo>/data/processed/*.jsonl
```

An earlier version of this note proposed a shared directory outside the
repository (`~/Datasets/gui-agent-week3/` and `D:\Datasets\gui-agent-week3\`).
That layout was never used: every byte of Week 3 data was written under `data/`,
and neither external directory exists. It is mentioned here only so the mistake
is not repeated - a second machine reading the old note would have downloaded
48 MB into a directory nothing else looks at.
