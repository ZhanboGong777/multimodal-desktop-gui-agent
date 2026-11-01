"""Source-only task inputs for screenshot-conditioned current-step training.

The original user goal is authoritative. Teacher plans, current actions and
responses never choose the input. Executable targets and split identities stay
unchanged; this conversion does not claim to resolve all chunk ambiguity.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from .data import _actions
from .diagnostics import ensure_development_isolation, load_development_rows
from .evaluation import file_sha256, strict_json_loads, value_sha256

VERSION = "week5.original-goal.current-step.v1"
POLICY = "first nonempty task_prompt, task_prompt_en, instruction; never target-dependent"
EXCLUDED = ("current_task", "send_prompt", "send_prompt_en", "send_prompt_zh",
            "actions", "action_annotations", "LLM_response", "LLM_response_editer")


def original_goal_input(
    original: str | None, english: str | None, fallback: str | None = None,
) -> tuple[str, str]:
    """Choose a field by fixed priority, retaining the selected string verbatim."""
    candidates = (("task_prompt", original), ("task_prompt_en", english),
                  ("instruction", fallback))
    for name, value in candidates:
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a string or null")
    for name, value in candidates:
        if isinstance(value, str) and value.strip():
            return value, name
    raise ValueError("source has no nonempty task goal")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [strict_json_loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def _conversion_source_hashes() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    sources = [folder / name for name in
               ("step_contract.py", "data.py", "diagnostics.py", "evaluation.py", "prompts.py")]
    sources += [folder.parent / "schemas.py", folder.parent / "datasets/schemas.py"]
    cli = folder.parents[2] / "scripts/week5_prepare_original_goals.py"
    if cli.is_file():
        sources.append(cli)
    return {str(path): file_sha256(path) for path in sources}


def _source(root: Path, row: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    relative = row.get("record_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("record_path is required")
    safe = PurePosixPath(relative.replace("\\", "/"))
    if safe.is_absolute() or ".." in safe.parts or ":" in relative or safe.suffix != ".json":
        raise ValueError("unsafe source record path")
    path = (root / str(safe)).resolve(strict=True)
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
        raise ValueError("input must match the original V1 task instruction")
    # These labels are audited only. They never determine the new instruction.
    actions, annotations = _actions(raw, "screenagent", row["image_width"], row["image_height"])
    if actions != row["actions"] or annotations != row.get("action_annotations", []):
        raise ValueError("prepared executable labels differ from raw source")
    return path, raw, hashlib.sha256(content).hexdigest()


def prepare_original_goals(
    train_path: str | Path, validation_path: str | Path, raw_train_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify and copy V1 development rows, changing only goals and their provenance."""
    paths = {"train": Path(train_path).resolve(), "validation": Path(validation_path).resolve()}
    if any(p.name.casefold() in {"test.jsonl", "official_test.jsonl", "official-test.jsonl"}
           for p in paths.values()):
        raise ValueError("official test paths forbidden before reading data")
    root = Path(raw_train_root).resolve(strict=True)
    if root.name.casefold() in {"test", "official_test", "official-test"}:
        raise ValueError("official test source forbidden")
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("output already exists; use a new directory")
    sources_before = _conversion_source_hashes()
    before = {split: file_sha256(path) for split, path in paths.items()}
    checked = {split: load_development_rows(path, split=split) for split, path in paths.items()}
    ensure_development_isolation(checked["train"], checked["validation"])
    originals = {split: _read_rows(path) for split, path in paths.items()}
    derived: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    records, fields, changed = [], Counter(), Counter()
    for split, rows in originals.items():
        verified = {r["sample_id"]: r for r in checked[split]}
        for row in sorted(rows, key=lambda r: r["sample_id"]):
            if value_sha256(row) != verified[row["sample_id"]]["source_sample_sha256"]:
                raise ValueError("prepared row changed after validation")
            if "instruction_source" in row:
                raise ValueError("use original V1 data without instruction_source")
            path, raw, raw_hash = _source(root, row)
            goal, field = original_goal_input(raw.get("task_prompt"), raw.get("task_prompt_en"),
                                              raw.get("instruction"))
            new = copy.deepcopy(row)
            new["instruction"] = goal
            new["instruction_source"] = {
                "version": VERSION, "policy": POLICY, "selected_field": field,
                "raw_record_path": row["record_path"], "raw_record_sha256": raw_hash,
                "previous_instruction_sha256": value_sha256(row["instruction"]),
            }
            derived[split].append(new)
            fields[f"{split}:{field}"] += 1
            changed[split] += goal != row["instruction"]
            records.append({"sample_id": row["sample_id"], "path": str(path), "sha256": raw_hash})
    def verify_sources() -> None:
        if any(file_sha256(paths[s]) != before[s] for s in paths):
            raise ValueError("prepared inputs changed during conversion")
        if any(file_sha256(r["path"]) != r["sha256"] for r in records):
            raise ValueError("raw source changed during conversion")
        if any(file_sha256(r["image_path"]) != r["image_file_sha256"]
               for rows in checked.values() for r in rows):
            raise ValueError("screenshot changed during conversion")
        if _conversion_source_hashes() != sources_before:
            raise ValueError("conversion source code changed during conversion")

    verify_sources()
    output.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for split, rows in derived.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)
                                + "\n" for r in rows), encoding="utf-8")
        outputs[split] = {"path": str(path), "sha256": file_sha256(path), "samples": len(rows),
                          "groups": len({r["group_id"] for r in rows}),
                          "instruction_changed_samples": changed[split]}
    manifest = {
        "version": VERSION, "policy": POLICY, "selected_fields": dict(fields),
        "test_data_consumed": False, "model_or_processor_loaded": False,
        "input_fields_excluded": list(EXCLUDED), "input_contract": "original goal + native screenshot",
        "labels_preserved_and_raw_verified": True, "identities_and_images_preserved": True,
        "group_and_image_isolation_verified": True, "inputs_unchanged_during_conversion": True,
        "input_data": {s: {"path": str(paths[s]), "sha256": before[s]} for s in paths},
        "raw_training_root": str(root), "raw_records": records, "outputs": outputs,
        "conversion_source_sha256": file_sha256(__file__),
        "conversion_dependencies_sha256": sources_before,
        "token_length_preflight_performed": False,
        "limitation": "Preserves original action chunks, including possible scope ambiguity; "
                      "does not infer teacher subtask context or prove model improvement.",
    }
    manifest_path = output / "manifest.json"
    try:
        verify_sources()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                 + "\n", encoding="utf-8")
        verify_sources()
        if any(file_sha256(item["path"]) != item["sha256"] for item in outputs.values()):
            raise ValueError("output data changed during conversion")
    except (OSError, ValueError, KeyboardInterrupt) as error:
        # Retain incomplete outputs for inspection, without a success manifest.
        manifest_path.unlink(missing_ok=True)
        failure = {"status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                   "error": f"{type(error).__name__}: {error}"}
        (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    return manifest
