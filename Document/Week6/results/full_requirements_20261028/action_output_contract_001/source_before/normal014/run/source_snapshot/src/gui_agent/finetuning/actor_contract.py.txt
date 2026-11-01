"""Verified conditional actor inputs from the published ScreenAgent request export.

This is a different profile from instruction-only evaluation: a pre-action
planner or operator must supply the subtask at deployment. Export consistency
does not establish an immutable capture time or an independent gold planner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .data import _actions
from .diagnostics import ensure_development_isolation, load_development_rows
from .evaluation import file_sha256, strict_json_loads, value_sha256
from .step_contract import original_goal_input

VERSION = "week5.screenagent.conditional-actor.v1"
POLICY = "two concordant original actor request declarations; original source goal priority"
SOURCE_REVISION = "2312c852908dda5fbc43d903fad5929a9dec649d"
INTRO = "您对 Windows 操作系统非常熟悉。您的任务是操作计算机屏幕，完成指定的任务。"
LINUX_INTRO = "您对 Linux 操作系统非常熟悉。您的任务是操作计算机屏幕，完成指定的任务。"
TEMPLATES = {INTRO: "actor_agent_win_zh", LINUX_INTRO: "actor_agent_linux_zh"}
PLAN_LINE = "我们为这个总体目标制定了一个实施计划："
OPERATIONS = "您可以使用鼠标和键盘，可选择的操作有："
ACTIVE_PREFIX = '现在的子任务是 "'
ACTIVE_END = '"。'
FINAL_END = '"，请根据现有屏幕图像的状态给出详细的下一步操作。'
GOAL_LINE = re.compile(
    r'^您现在的目标是操作计算机屏幕，视频宽度：(\d+)，视频高度：(\d+)，总体目标是 "(.*)"。$'
)
FORBIDDEN_PARTS = {"test", "official_test", "official-test", "test.jsonl", "official_test.jsonl",
                   "official-test.jsonl", "test.zip"}
EXCLUDED_INPUT_FIELDS = ("send_prompt", "send_prompt_en", "send_prompt_zh", "sub_task_list",
                         "advice", "actions", "action_annotations", "LLM_response",
                         "LLM_response_editer", "LLM_response_editer_en", "LLM_response_editer_zh")


def verify_actor_request(raw: Mapping[str, Any], width: int, height: int) -> dict[str, Any]:
    """Parse fixed template positions before comparing the raw current_task field.

    No target text, target length, response or plan element chooses an input.
    Raw action family only rules out Plan/Evaluate exports and mixed phases.
    Unknown templates and contradictory activity declarations fail closed.
    """
    def refuse(reason: str) -> dict[str, Any]:
        return {"eligible": False, "reason": reason}

    current = raw.get("current_task")
    if not isinstance(current, str) or not current.strip():
        return refuse("missing_or_empty_current_task")
    if any(character in current for character in "\r\n\x00"):
        return refuse("unsupported_multiline_current_task")
    prompt = raw.get("send_prompt")
    if not isinstance(prompt, str) or not prompt:
        return refuse("missing_original_send_prompt")
    actions = raw.get("actions")
    if not isinstance(actions, list) or not actions or any(
        not isinstance(action, dict) or action.get("action_type") not in
        {"MouseAction", "KeyboardAction", "WaitAction"} for action in actions
    ):
        return refuse("not_exclusive_actor_phase")
    try:
        goal, field = original_goal_input(raw.get("task_prompt"), raw.get("task_prompt_en"),
                                          raw.get("instruction"))
    except (TypeError, ValueError):
        return refuse("invalid_source_goal")
    lines = prompt.splitlines()
    if len(lines) < 6 or lines[0] not in TEMPLATES:
        return refuse("unsupported_actor_template")
    goal_match = GOAL_LINE.fullmatch(lines[1])
    if goal_match is None:
        return refuse("missing_authoritative_goal_line")
    if goal_match.group(3) != goal:
        return refuse("request_goal_differs_from_source_goal")
    if any(type(value) is not int or value <= 0 for value in
           (width, height, raw.get("video_width"), raw.get("video_height"))):
        return refuse("invalid_native_dimensions")
    if (int(goal_match.group(1)), int(goal_match.group(2))) != (width, height) or (
        raw["video_width"], raw["video_height"]
    ) != (width, height):
        return refuse("request_dimensions_differ_from_native_image")
    active_indices = [index for index, line in enumerate(lines) if "现在的子任务是" in line]
    operations_indices = [index for index, line in enumerate(lines) if line == OPERATIONS]
    if len(active_indices) != 2 or len(operations_indices) != 1:
        return refuse("missing_or_ambiguous_activity_declarations")
    first, last = active_indices
    last_nonempty = max(index for index, line in enumerate(lines) if line)
    if not (1 < first < operations_indices[0] < last == last_nonempty):
        return refuse("activity_declarations_outside_actor_positions")
    if PLAN_LINE not in lines[2:first]:
        return refuse("unsupported_actor_template")
    front, tail = lines[first], lines[last]
    if not (front.startswith(ACTIVE_PREFIX) and front.endswith(ACTIVE_END)
            and tail.startswith(ACTIVE_PREFIX) and tail.endswith(FINAL_END)):
        return refuse("unsupported_activity_declaration_syntax")
    front_task = front[len(ACTIVE_PREFIX):-len(ACTIVE_END)]
    tail_task = tail[len(ACTIVE_PREFIX):-len(FINAL_END)]
    if front_task != tail_task:
        return refuse("conflicting_request_activity_declarations")
    if front_task != current:
        return refuse("request_activity_differs_from_current_task")
    return {"eligible": True, "reason": "verified_original_actor_request",
            "instruction": goal, "goal_field": field, "actor_step": front_task,
            "request_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "declaration_line_indices": [first, last], "request_template": TEMPLATES[lines[0]],
            "source_revision": SOURCE_REVISION}


def _forbid_test_path(path: Path) -> None:
    if any(part.casefold() in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError("official test path forbidden before reading data")


def read_actor_source(root: Path, row: Mapping[str, Any]) -> tuple[Path, dict[str, Any], str]:
    """Bind a safe source record to its original prepared session and instruction."""
    relative = row.get("record_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("record_path is required")
    safe = PurePosixPath(relative.replace("\\", "/"))
    if safe.is_absolute() or ".." in safe.parts or ":" in relative or safe.suffix != ".json":
        raise ValueError("unsafe source record path")
    _forbid_test_path(Path(str(safe)))
    path = (root / str(safe)).resolve(strict=True)
    _forbid_test_path(path)
    if not path.is_relative_to(root):
        raise ValueError("source record escapes training root")
    content = path.read_bytes()
    raw = strict_json_loads(content.decode("utf-8-sig"))
    if not isinstance(raw, dict):
        raise TypeError("raw source must be an object")
    if row["source"] != "screenagent" or row["group_id"] != f"screenagent:{raw.get('session_id')}":
        raise ValueError("raw source session/group mismatch")
    previous = raw.get("task_prompt_en") or raw.get("task_prompt") or raw.get("instruction")
    if not isinstance(previous, str) or previous.strip() != row["instruction"]:
        raise ValueError("input must match original V1 task instruction")
    return path, raw, hashlib.sha256(content).hexdigest()


def _source_image(root: Path, record: Path, raw: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    name = raw.get("saved_image_name")
    if not isinstance(name, str) or not name or Path(name).name != name or ":" in name:
        raise ValueError("unsafe source image name")
    path = (record.parent / "images" / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("source image escapes training root")
    _forbid_test_path(path)
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1 or image.size != (row["image_width"], row["image_height"]):
            raise ValueError("source image native dimensions differ")
        rgb = image.convert("RGB")
        digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
    if digest != row["image_sha256"] or file_sha256(path) != row["image_file_sha256"]:
        raise ValueError("source screenshot differs from prepared native screenshot")
    return path


def _conversion_sources() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    paths = [folder / name for name in ("actor_contract.py", "step_contract.py", "data.py",
                                       "diagnostics.py", "evaluation.py", "prompts.py")]
    paths += [folder.parent / "schemas.py", folder.parent / "datasets/schemas.py"]
    cli = folder.parents[2] / "scripts/week5_prepare_actor_contract.py"
    if cli.is_file():
        paths.append(cli)
    return {str(path): file_sha256(path) for path in paths}


def prepare_actor_contract(
    train_path: str | Path, validation_path: str | Path, raw_train_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Apply one source-only actor policy to both splits, retaining all rejection reasons."""
    paths = {"train": Path(train_path).resolve(), "validation": Path(validation_path).resolve()}
    root = Path(raw_train_root).resolve()
    for path in [*paths.values(), root]:
        _forbid_test_path(path)
    root = root.resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("output already exists; use a new directory")
    dependencies = _conversion_sources()
    before = {split: file_sha256(path) for split, path in paths.items()}
    original = {split: [strict_json_loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                        if line.strip()] for split, path in paths.items()}
    # Reject declared test screenshots before the image verifier can open them.
    for split, rows in original.items():
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("image_path"), str):
                raise TypeError("prepared image_path is required")
            image = Path(row["image_path"])
            image = image if image.is_absolute() else paths[split].parent / image
            _forbid_test_path(image.resolve())
    checked = {split: load_development_rows(path, split=split) for split, path in paths.items()}
    ensure_development_isolation(checked["train"], checked["validation"])
    derived: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    records, reasons = [], Counter()
    for split, rows in original.items():
        verified = {row["sample_id"]: row for row in checked[split]}
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            current = verified[row["sample_id"]]
            if value_sha256(row) != current["source_sample_sha256"]:
                raise ValueError("prepared row changed after validation")
            if "instruction_source" in row or "actor_step" in row or "actor_step_source" in row:
                raise ValueError("use original V1 prepared data without derived context")
            path, raw, digest = read_actor_source(root, row)
            decision = verify_actor_request(raw, row["image_width"], row["image_height"])
            record = {"split": split, "sample_id": row["sample_id"], "group_id": row["group_id"],
                      "raw_record_path": str(path), "raw_record_sha256": digest,
                      "eligible": decision["eligible"], "reason": decision["reason"]}
            if decision["eligible"]:
                actions, annotations = _actions(raw, "screenagent", row["image_width"], row["image_height"])
                if actions != row["actions"] or annotations != row.get("action_annotations", []):
                    raise ValueError("prepared executable labels differ from raw actor source")
                image = _source_image(root, path, raw, current)
                # Bind the already verified expected bytes, never resample a changed
                # source image as a new baseline between validation and final checks.
                record.update(source_image_path=str(image), source_image_sha256=current["image_file_sha256"])
                new = copy.deepcopy(row)
                new["instruction"] = decision["instruction"]
                new["instruction_source"] = {"version": VERSION, "selected_field": decision["goal_field"],
                    "policy": "first nonempty task_prompt, task_prompt_en, instruction; never target-dependent",
                    "raw_record_sha256": digest, "raw_record_path": row["record_path"]}
                new["actor_step"] = decision["actor_step"]
                new["actor_step_source"] = {"version": VERSION, "policy": POLICY,
                    "raw_record_path": row["record_path"], "raw_record_sha256": digest,
                    **{name: decision[name] for name in ("request_prompt_sha256", "declaration_line_indices",
                                                        "request_template", "source_revision")},
                    "chronology_claim": "Published actor request/export consistency; not a signed pre-response timestamp"}
                derived[split].append(new)
            records.append(record)
            reasons[f"{split}:{decision['reason']}"] += 1
    if any(not rows for rows in derived.values()):
        raise ValueError("both train and validation need at least one verified actor request")

    def verify_sources() -> None:
        if {split: file_sha256(path) for split, path in paths.items()} != before:
            raise ValueError("prepared inputs changed during conversion")
        if any(file_sha256(record["raw_record_path"]) != record["raw_record_sha256"] for record in records):
            raise ValueError("raw source changed during conversion")
        if any(file_sha256(row["image_path"]) != row["image_file_sha256"]
               for rows in checked.values() for row in rows):
            raise ValueError("prepared screenshot changed during conversion")
        if any(file_sha256(record["source_image_path"]) != record["source_image_sha256"]
               for record in records if record["eligible"]):
            raise ValueError("source screenshot changed during conversion")
        if _conversion_sources() != dependencies:
            raise ValueError("conversion source changed during conversion")

    verify_sources()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    try:
        outputs = {}
        for split, rows in derived.items():
            path = output / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                                    + "\n" for row in rows), encoding="utf-8")
            outputs[split] = {"path": str(path), "sha256": file_sha256(path), "samples": len(rows),
                              "groups": len({row["group_id"] for row in rows}),
                              "excluded_samples": len(original[split]) - len(rows)}
        manifest = {"version": VERSION, "policy": POLICY, "source_revision": SOURCE_REVISION,
            "input_contract": "original goal + source-verified pre-actor subtask + native screenshot",
            "scope": "conditional actor development profile; not instruction-only or independent test",
            "excluded_model_input_fields": list(EXCLUDED_INPUT_FIELDS), "test_data_consumed": False,
            "model_or_processor_loaded": False, "token_length_preflight_performed": False,
            "input_data": {split: {"path": str(path), "sha256": before[split]} for split, path in paths.items()},
            "raw_training_root": str(root), "records": records, "reason_counts": dict(reasons),
            "outputs": outputs, "conversion_dependencies_sha256": dependencies,
            "labels_and_annotations_preserved_and_raw_verified": True,
            "identities_groups_images_dimensions_preserved": True, "split_isolation_verified": True,
            "inputs_unchanged_during_conversion": True,
            "chronology_limit": "Export is editable and lacks signed request timestamps. Two matching original "
                                "request declarations support pre-actor context, not independent planner quality.",
            "deployment_requirement": "A pre-action planner or operator must provide actor_step; never infer "
                                      "it from the current response or gold action chunk."}
        verify_sources()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
                                 + "\n", encoding="utf-8")
        verify_sources()
        if any(file_sha256(item["path"]) != item["sha256"] for item in outputs.values()):
            raise ValueError("output data changed during conversion")
    except (Exception, KeyboardInterrupt) as error:
        manifest_path.unlink(missing_ok=True)
        failure = {"status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error": f"{type(error).__name__}: {error}"}
        (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    return manifest
