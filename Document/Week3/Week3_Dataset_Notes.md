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

`action_reprs` entries look like `CLICK [Submit]`; the verb is split off and the
bracketed text becomes the target.

### WebArena

| Source | Unified |
| --- | --- |
| `intent` / `instruction` | `instruction` |
| `start_url` / `url` | `metadata.start_url` |
| `sites` | `metadata.sites` |
| `eval` / `reference_answers` | `metadata.eval` |
| - | `actions` stays empty: WebArena ships no trajectory |

## Limits found

1. **Mind2Web is too large to download casually.** 13.6 GB with screenshots. Week 3
   streams a handful of samples instead; the preparation script takes `--limit` for
   exactly this reason.
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
   archive is sharded, and a shard has to arrive before the iterator yields
   anything. The adapter therefore remains **fixture-tested only**, and this is a
   real gap rather than an oversight. The hand-off's own warning - do not download
   the whole dataset first - is confirmed, and the practical route is a short
   one-off download of a single shard, or `Mind2Web` (6.74 GB) rather than
   `Multimodal-Mind2Web` (13.6 GB).

## Local layout

Raw downloads and exports live outside the repository:

```text
Mac:     /Users/caleb/Datasets/gui-agent-week3/{screenagent,mind2web,webarena}/
Windows: D:\Datasets\gui-agent-week3\{screenagent,mind2web,webarena}\
```

If a project-local layout is preferred, `data/raw/` and `data/processed/` are
already ignored by Git.
