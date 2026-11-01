"""Prepare independent bilingual-goal train/validation data on CPU, without test.

Only raw task_prompt and task_prompt_en determine the new instruction. Current
actions, responses, action descriptions and send_prompt never choose its content.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui_agent.finetuning.diagnostics import (
    bilingual_goal_input,
    ensure_development_isolation,
    file_sha256,
    load_development_rows,
    value_sha256,
)

VERSION = "week5.bilingual-goals.v1"
_FORBIDDEN_FILES = {"test.jsonl", "official_test.jsonl", "official-test.jsonl"}


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text: str) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON: {value}")
    return json.loads(text, object_pairs_hook=_object_pairs, parse_constant=invalid)


def _source_record(root: Path, row: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    relative = row.get("record_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("record_path is required for original task lookup")
    safe = PurePosixPath(relative.replace("\\", "/"))
    if safe.is_absolute() or ".." in safe.parts or ":" in relative or safe.suffix != ".json":
        raise ValueError("unsafe source record lookup")
    source = (root / str(safe)).resolve(strict=True)
    if not source.is_relative_to(root):
        raise ValueError("source record escapes official training root")
    content = source.read_bytes()
    raw = _json(content.decode("utf-8-sig"))
    if not isinstance(raw, dict):
        raise TypeError("raw source must be an object")
    if row["source"] != "screenagent" or row["group_id"] != f"screenagent:{raw.get('session_id')}":
        raise ValueError("source session/group provenance mismatch")
    original, english = raw.get("task_prompt"), raw.get("task_prompt_en")
    previous = english or original
    if not isinstance(previous, str) or previous.strip() != row["instruction"]:
        raise ValueError("source English/fallback instruction provenance mismatch")
    # Do not inspect raw actions, current_task, send_prompt or any response when
    # constructing this field. The renderer cannot receive a target argument.
    instruction = bilingual_goal_input(original, english)
    return source, {"instruction": instruction, "original_goal": original,
                    "english_goal": english}, hashlib.sha256(content).hexdigest()


def _literal_stats(rows: list[dict[str, Any]], goals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Post-construction lexical audit; it never affects row content or inclusion."""
    counts = Counter({key: 0 for key in (
        "type_text_actions", "previous_english_or_fallback_literal_present",
        "original_goal_literal_present", "english_goal_literal_present",
        "either_goal_literal_present", "new_bilingual_instruction_literal_present",
    )})
    for row in rows:
        source = goals[row["sample_id"]]
        for action in row["actions"]:
            if action["action_type"] != "type_text":
                continue
            text = action["text"]
            original = source["original_goal"]
            english = source["english_goal"]
            original_has = isinstance(original, str) and text in original
            english_has = isinstance(english, str) and text in english
            counts.update({
                "type_text_actions": 1,
                "previous_english_or_fallback_literal_present": int(text in row["instruction"]),
                "original_goal_literal_present": int(original_has),
                "english_goal_literal_present": int(english_has),
                "either_goal_literal_present": int(original_has or english_has),
                "new_bilingual_instruction_literal_present": int(text in source["instruction"]),
            })
    return {**dict(counts), "interpretation": "Exact lexical evidence only; no model performance "
            "or learnability claim. Short substrings can be incidental matches."}


def prepare_bilingual(
    train_path: str | Path, validation_path: str | Path, raw_train_root: str | Path,
    output_dir: str | Path, *, source_audit: str | Path | None = None,
    max_seq_length: int = 1536, source_files: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Copy every development row unchanged except instruction and explicit goal provenance."""
    paths = {"train": Path(train_path).resolve(), "validation": Path(validation_path).resolve()}
    if any(path.name.casefold() in _FORBIDDEN_FILES for path in paths.values()):
        raise ValueError("official test path is forbidden before reading any data")
    raw_root = Path(raw_train_root).resolve(strict=True)
    if raw_root.name.casefold() in {"test", "official_test", "official-test"}:
        raise ValueError("official test raw source is forbidden")
    if type(max_seq_length) is not int or max_seq_length <= 0:
        raise ValueError("max_seq_length must be a positive integer")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty; preserve existing data and use a new path")
    input_hashes = {split: file_sha256(path) for split, path in paths.items()}
    source_hashes = [{"path": str(Path(path).resolve(strict=True)), "sha256": file_sha256(path)}
                     for path in source_files or []]
    checked = {split: load_development_rows(path, split=split) for split, path in paths.items()}
    ensure_development_isolation(checked["train"], checked["validation"])
    originals = {split: [_json(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                        if line.strip()] for split, path in paths.items()}
    for split in paths:
        by_id = {row["sample_id"]: row for row in checked[split]}
        if any(value_sha256(row) != by_id[row["sample_id"]]["source_sample_sha256"]
               for row in originals[split]):
            raise ValueError("prepared row changed between validation and copy")
    audit_hash = None
    audited = None
    if source_audit is not None:
        audit_file = Path(source_audit).resolve(strict=True)
        audit_hash = file_sha256(audit_file)
        document = _json(audit_file.read_text(encoding="utf-8-sig"))
        audited = {}
        for record in document["records"]:
            if record["sample_id"] in audited:
                raise ValueError("duplicate source-audit sample_id")
            audited[record["sample_id"]] = record
        ids = {row["sample_id"] for rows in originals.values() for row in rows}
        if set(audited) != ids:
            raise ValueError("source audit does not cover exactly these train/validation records")
    result = {"train": [], "validation": []}
    raw_evidence, goals = [], {}
    changed = Counter()
    for split, rows in originals.items():
        for row in sorted(rows, key=lambda value: value["sample_id"]):
            if "instruction_source" in row:
                raise ValueError("input already has instruction_source; use the original V1 dataset")
            source, goal, raw_hash = _source_record(raw_root, row)
            if audited is not None:
                previous = audited[row["sample_id"]]
                if (previous["sha256"] != raw_hash or
                        previous["record_path"].replace("\\", "/") != row["record_path"].replace("\\", "/")):
                    raise ValueError("raw training record differs from the frozen source audit")
            goals[row["sample_id"]] = goal
            new = copy.deepcopy(row)
            new["instruction"] = goal["instruction"]
            new["instruction_source"] = {
                "version": VERSION, "fields": ["task_prompt", "task_prompt_en"],
                "task_prompt": goal["original_goal"], "task_prompt_en": goal["english_goal"],
                "policy": "retain both distinct nonempty task goals; no target-dependent selection",
                "raw_record_path": row["record_path"], "raw_record_sha256": raw_hash,
                "previous_instruction_sha256": value_sha256(row["instruction"]),
            }
            assert {key: value for key, value in new.items()
                    if key != "instruction_source"} == {**row, "instruction": goal["instruction"]}
            changed[split] += new["instruction"] != row["instruction"]
            result[split].append(new)
            raw_evidence.append({"sample_id": row["sample_id"], "path": str(source), "sha256": raw_hash})
    for split, path in paths.items():
        if file_sha256(path) != input_hashes[split]:
            raise ValueError("original prepared dataset changed during conversion")
    if any(file_sha256(row["image_path"]) != row["image_file_sha256"]
           for rows in checked.values() for row in rows):
        raise ValueError("original screenshot changed during conversion")
    if any(file_sha256(record["path"]) != record["sha256"] for record in raw_evidence):
        raise ValueError("raw task source changed during conversion")
    if any(file_sha256(record["path"]) != record["sha256"] for record in source_hashes):
        raise ValueError("conversion source code changed during conversion")
    if source_audit is not None and file_sha256(source_audit) != audit_hash:
        raise ValueError("frozen source audit changed during conversion")
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, rows in result.items():
        destination = output / f"{split}.jsonl"
        destination.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                               allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        outputs[split] = {"path": str(destination), "samples": len(rows),
                          "groups": len({row["group_id"] for row in rows}),
                          "instruction_changed_samples": changed[split],
                          "sha256": file_sha256(destination)}
    manifest = {
        "version": VERSION, "test_data_consumed": False, "model_or_processor_loaded": False,
        "input_contract": "instruction from raw task_prompt + task_prompt_en only; native screenshot",
        "input_fields_excluded": ["current_task", "send_prompt", "send_prompt_en", "send_prompt_zh",
                                  "LLM_response*", "actions", "action descriptions", "target_bbox"],
        "preserved_fields": "all original fields except instruction; instruction_source is new provenance",
        "images_copied_or_modified": False, "group_and_image_isolation_verified": True,
        "input_data": {split: {"path": str(path), "sha256": input_hashes[split]}
                       for split, path in paths.items()},
        "raw_training_root": str(raw_root), "raw_records": raw_evidence,
        "source_audit": {"path": str(Path(source_audit).resolve()), "sha256": audit_hash}
                        if source_audit is not None else None,
        "source_files": source_hashes, "outputs": outputs,
        "literal_lexical_evidence": {split: _literal_stats(rows, goals) for split, rows in originals.items()},
        "token_length_preflight": {
            "performed": False, "max_seq_length": max_seq_length,
            "required_before_training": True,
            "recommendation": "Use the exact pinned processor, use_fast setting, chat template, "
                              "native images and assistant target on all train/validation rows; "
                              "reject any full image-expanded sequence above this limit without truncation.",
        }, "inputs_unchanged_during_conversion": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                    allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--raw-train-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-audit", help="verify against frozen raw_field_review.json")
    parser.add_argument("--max-seq-length", type=int, default=1536, help="preflight recommendation only")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    sources = [repo / path for path in (
        "scripts/week5_prepare_bilingual.py", "tests/test_week5_bilingual.py",
        "src/gui_agent/finetuning/diagnostics.py", "src/gui_agent/finetuning/data.py",
        "src/gui_agent/finetuning/prompts.py",
    )]
    try:
        manifest = prepare_bilingual(args.train, args.validation, args.raw_train_root, args.output_dir,
                                     source_audit=args.source_audit, max_seq_length=args.max_seq_length,
                                     source_files=sources)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"bilingual preparation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(Path(args.output_dir).resolve() / "manifest.json"),
                      "outputs": manifest["outputs"],
                      "literal_lexical_evidence": manifest["literal_lexical_evidence"],
                      "test_data_consumed": False, "tokenized": False}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
