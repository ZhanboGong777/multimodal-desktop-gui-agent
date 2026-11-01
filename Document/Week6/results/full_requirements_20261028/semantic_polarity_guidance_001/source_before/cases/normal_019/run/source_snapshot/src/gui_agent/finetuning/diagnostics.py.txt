"""CPU-only development-data audits and result-independent diagnostic selections.

This module never loads a model or an official test set. Its labels and raw-field
reviews belong on the scoring/audit side, never in a generation prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .data import _actions, validate_actions
from .prompts import action_target_json

DIAGNOSTIC_VERSION = "week5.development-data-audit.v1"
POINT_ACTIONS = frozenset({"click", "double_click", "right_click", "move", "scroll"})
CLICK_ACTIONS = frozenset({"click", "double_click", "right_click"})
FIELD_POLICIES = (
    {"fields": ["task_prompt", "task_prompt_en", "task_prompt_zh"],
     "decision": "allow_overall_goal_only",
     "reason": "Documented overall user goal; current contract uses task_prompt_en with fallback.",
     "current_model_input": True},
    {"fields": ["video_width", "video_height", "saved_image_name"],
     "decision": "allow_pixels_and_verified_native_dimensions",
     "reason": "Load actual screenshot pixels; filenames are provenance, never prompt text.",
     "current_model_input": False},
    {"fields": ["send_prompt", "send_prompt_en", "send_prompt_zh"],
     "decision": "exclude_whole_field",
     "reason": "Mixed instructions/history/action JSON examples; no verified safe span parser.",
     "current_model_input": False},
    {"fields": ["LLM_response", "LLM_response_editer", "LLM_response_editer_en",
                "LLM_response_editer_zh", "actions"],
     "decision": "labels_only_never_model_input",
     "reason": "Responses and parsed corrected actions are current-step supervision.",
     "current_model_input": False},
    {"fields": ["current_task"], "decision": "requires_temporal_and_annotation_review",
     "reason": "May already describe the current labeled action. README does not establish "
               "pre-inference origin; do not copy action descriptions into a new input.",
     "current_model_input": False},
    {"fields": ["session_id"], "decision": "grouping_only",
     "reason": "Prevent trajectory leakage; session/sample IDs never enter prompt text.",
     "current_model_input": False},
)


def value_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(text: str) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def _annotations(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    actions = row["actions"]
    annotations = row.get("action_annotations", [])
    if not isinstance(annotations, list) or (annotations and len(annotations) != len(actions)):
        raise ValueError("action_annotations must be an aligned array")
    result = []
    for index, action in enumerate(actions):
        annotation = annotations[index] if annotations else {}
        if not isinstance(annotation, dict) or set(annotation) - {"target_bbox"}:
            raise ValueError("unknown annotation fields")
        box = annotation.get("target_bbox")
        if box is not None:
            if not isinstance(box, dict) or set(box) != {"left", "top", "right", "bottom"}:
                raise ValueError("invalid native target bbox")
            if any(type(value) is not int for value in box.values()):
                raise ValueError("native target bbox needs integer coordinates")
            if not (0 <= box["left"] < box["right"] <= row["image_width"]
                    and 0 <= box["top"] < box["bottom"] <= row["image_height"]):
                raise ValueError("native target bbox outside image or empty")
            if action["action_type"] not in POINT_ACTIONS:
                raise ValueError("bbox annotation requires a point action")
        result.append(dict(annotation))
    return result


def load_development_rows(path: str | Path, *, split: str) -> list[dict[str, Any]]:
    """Read verified train/validation only; keep bytes and decoded-pixel hashes distinct."""
    if split not in {"train", "validation"}:
        raise ValueError("diagnostics accepts train/validation only")
    dataset = Path(path).resolve()
    if dataset.name.casefold() in {"test.jsonl", "official_test.jsonl", "official-test.jsonl"}:
        raise ValueError("official test file is forbidden for development diagnostics")
    rows, seen = [], set()
    for number, line in enumerate(dataset.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = _strict_json(line)
        if not isinstance(row, dict):
            raise TypeError(f"line {number}: sample must be an object")
        for key in ("sample_id", "group_id", "image_path", "instruction", "source", "source_split"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"line {number}: {key} is required")
        if row["source_split"] not in {"train", "validation"}:
            raise ValueError("official test/unknown source split is forbidden")
        if split == "train" and row["source_split"] != "train":
            raise ValueError("training diagnostics requires a training-source sample")
        if row["sample_id"] in seen:
            raise ValueError("duplicate sample_id")
        seen.add(row["sample_id"])
        if not isinstance(row.get("actions"), list):
            raise TypeError("actions must be an array")
        actions = validate_actions(row["actions"], row.get("image_width"), row.get("image_height"))
        # The diagnostic must not silently normalize an already prepared target.
        if actions != row["actions"]:
            raise ValueError("prepared target is not canonical")
        annotations = _annotations(row)
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = dataset.parent / image_path
        image_path = image_path.resolve(strict=True)
        with Image.open(image_path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated screenshot is forbidden")
            if image.size != (row["image_width"], row["image_height"]):
                raise ValueError("image dimensions differ from native metadata")
            rgb = image.convert("RGB")
            decoded = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
        if row.get("image_sha256") and decoded != row["image_sha256"]:
            raise ValueError("decoded RGB hash differs from prepared metadata")
        rows.append({**row, "actions": actions, "action_annotations": annotations,
                     "image_path": str(image_path), "image_sha256": decoded,
                     "image_file_sha256": file_sha256(image_path),
                     "source_sample_sha256": value_sha256(row), "diagnostic_split": split})
    if not rows:
        raise ValueError("development dataset is empty")
    return sorted(rows, key=lambda row: row["sample_id"])


def ensure_development_isolation(train: Sequence[Mapping], validation: Sequence[Mapping]) -> None:
    for field in ("sample_id", "group_id", "image_sha256"):
        if {row[field] for row in train} & {row[field] for row in validation}:
            raise ValueError(f"training/validation leakage: {field}")


def bilingual_goal_input(original_goal: str | None, english_goal: str | None) -> str:
    """Proposed V2 text, derived only from the two goals, never target-dependent."""
    values = []
    for label, value in (("Original task", original_goal), ("English task", english_goal)):
        if value is not None and not isinstance(value, str):
            raise TypeError("task goals must be strings or absent")
        if value is not None and value.strip():
            values.append((label, value.strip()))
    if not values:
        raise ValueError("at least one task goal is required")
    if len(values) == 1 or values[0][1] == values[1][1]:
        return values[0][1]
    return "\n".join(f"{label}: {value}" for label, value in values)


def _stratum(row: Mapping[str, Any]) -> str:
    length = len(row["actions"])
    bucket = "1" if length == 1 else "2" if length == 2 else "3+"
    kinds = "+".join(sorted({action["action_type"] for action in row["actions"]}))
    return f"length={bucket};types={kinds}"


def select_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 42, limit: int = 0,
) -> list[Mapping[str, Any]]:
    """Hash-rank within label strata and round-robin; independent of any model result."""
    if type(seed) is not int or type(limit) is not int or limit < 0:
        raise ValueError("seed/limit must be integers and limit nonnegative")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selection sample_id")
    if limit == 0 or limit >= len(rows):
        return sorted(rows, key=lambda row: row["sample_id"])
    strata = defaultdict(list)
    for row in rows:
        strata[_stratum(row)].append(row)
    for values in strata.values():
        values.sort(key=lambda row: (value_sha256([seed, row["sample_id"]]), row["sample_id"]))
    keys = sorted(strata, key=lambda key: (value_sha256([seed, key]), key))
    chosen, offset = [], 0
    while len(chosen) < limit:
        for key in keys:
            if offset < len(strata[key]):
                chosen.append(strata[key][offset])
                if len(chosen) == limit:
                    break
        offset += 1
    return sorted(chosen, key=lambda row: row["sample_id"])


def audit_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Lexical absence/annotation disagreement are evidence, never unlearnability verdicts."""
    kinds, lengths, dimensions, strata = Counter(), Counter(), Counter(), Counter()
    text_total = text_present = clicks = annotated_clicks = point_outside_box = 0
    per_sample = []
    input_targets = defaultdict(set)
    input_samples = defaultdict(list)
    for row in rows:
        length = len(row["actions"])
        lengths[str(length)] += 1
        dimensions[f"{row['image_width']}x{row['image_height']}"] += 1
        strata[_stratum(row)] += 1
        literals, boxes = [], []
        target_hash = value_sha256(row["actions"])
        input_targets[(row["image_sha256"], row["instruction"])].add(target_hash)
        input_samples[(row["image_sha256"], row["instruction"])].append({
            "sample_id": row["sample_id"], "group_id": row["group_id"], "target_sha256": target_hash})
        for index, action in enumerate(row["actions"]):
            kind = action["action_type"]
            kinds[kind] += 1
            if kind == "type_text":
                present = action["text"] in row["instruction"]
                text_total += 1
                text_present += present
                literals.append({"action_index": index, "literal_sha256": value_sha256(action["text"]),
                                 "literal_characters": len(action["text"]),
                                 "exact_literal_in_instruction": present})
            if kind in CLICK_ACTIONS:
                clicks += 1
                box = row["action_annotations"][index].get("target_bbox")
                hit = None
                if box is not None:
                    annotated_clicks += 1
                    hit = box["left"] <= action["x"] < box["right"] \
                        and box["top"] <= action["y"] < box["bottom"]
                    point_outside_box += not hit
                boxes.append({"action_index": index, "native_bbox_present": box is not None,
                              "target_point_inside_annotated_bbox": hit})
        per_sample.append({"sample_id": row["sample_id"], "sequence_length": length,
                           "stratum": _stratum(row), "type_text_lexical_evidence": literals,
                           "click_annotation_evidence": boxes})
    ambiguous = sum(len(labels) > 1 for labels in input_targets.values())
    conflicts = [{"image_sha256": image_hash, "instruction_sha256": value_sha256(instruction),
                  "samples": sorted(input_samples[(image_hash, instruction)],
                                    key=lambda sample: sample["sample_id"])}
                 for (image_hash, instruction), labels in sorted(input_targets.items()) if len(labels) > 1]
    return {"samples": len(rows), "groups": len({row["group_id"] for row in rows}),
            "action_count": sum(kinds.values()), "action_type_counts": dict(sorted(kinds.items())),
            "sequence_length_counts": dict(sorted(lengths.items(), key=lambda item: int(item[0]))),
            "multi_action_samples": sum(count for length, count in lengths.items() if int(length) > 1),
            "native_image_dimensions": dict(sorted(dimensions.items())),
            "stratum_counts": dict(sorted(strata.items())),
            "type_text": {"actions": text_total, "literal_present_in_instruction": text_present,
                          "literal_absent_from_instruction": text_total - text_present,
                          "interpretation": "Lexical evidence only: absence does not establish "
                                            "unlearnability; screenshot/history may provide text."},
            "clicks": {"actions": clicks, "native_bbox_present": annotated_clicks,
                       "native_bbox_absent": clicks - annotated_clicks,
                       "annotated_target_point_outside_bbox": point_outside_box,
                       "bbox_convention": "native-pixel half-open [left,right) x [top,bottom)",
                       "interpretation": "Real annotations only; no bbox synthesized from target point."},
            "identical_image_and_instruction_with_different_targets": ambiguous,
            "input_label_conflicts_for_manual_review": conflicts,
            "per_sample": sorted(per_sample, key=lambda row: row["sample_id"])}


def review_raw_fields(rows: Sequence[Mapping], raw_train_root: str | Path) -> dict[str, Any]:
    """Inspect only retained development records; raw strings are not exported as inputs."""
    root = Path(raw_train_root).resolve(strict=True)
    if root.name.casefold() in {"test", "official_test", "official-test"}:
        raise ValueError("official test raw source is forbidden")
    counts, schema_markers, literal_mentions = Counter(), Counter(), Counter()
    current_task_equal_element = 0
    records = []
    goal_evidence = defaultdict(Counter)
    goal_examples = defaultdict(list)
    for row in sorted(rows, key=lambda value: value["sample_id"]):
        if row["source"] != "screenagent":
            raise ValueError("raw ScreenAgent audit requires ScreenAgent records")
        relative = row.get("record_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("raw audit requires record_path")
        source = (root / relative.replace("\\", "/")).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("raw record path escapes training source")
        raw_bytes = source.read_bytes()
        raw = _strict_json(raw_bytes.decode("utf-8-sig"))
        if not isinstance(raw, dict) or raw.get("session_id") != row["group_id"].split(":", 1)[-1]:
            raise ValueError("raw session provenance mismatch")
        original_goal = raw.get("task_prompt_en") or raw.get("task_prompt") or raw.get("instruction")
        if not isinstance(original_goal, str) or original_goal.strip() != row["instruction"]:
            raise ValueError("raw task instruction provenance mismatch")
        raw_actions, raw_annotations = _actions(raw, "screenagent", row["image_width"], row["image_height"])
        if raw_actions != row["actions"] or raw_annotations != row.get("action_annotations", []):
            raise ValueError("raw current label provenance mismatch")
        for index, action in enumerate(row["actions"]):
            if action["action_type"] != "type_text":
                continue
            literal = action["text"]
            original = raw.get("task_prompt")
            english = raw.get("task_prompt_en")
            original_has = isinstance(original, str) and literal in original
            english_has = isinstance(english, str) and literal in english
            chosen_has = literal in row["instruction"]
            split = row.get("diagnostic_split", "train")
            stats = goal_evidence[split]
            stats.update({"type_text_actions": 1, "selected_instruction_literal_present": int(chosen_has),
                          "english_goal_literal_present": int(english_has),
                          "original_goal_literal_present": int(original_has),
                          "either_goal_literal_present": int(original_has or english_has),
                          "original_only_literal_present": int(original_has and not english_has),
                          "english_only_literal_present": int(english_has and not original_has)})
            category = "original_only" if original_has and not english_has else "english_only" \
                if english_has and not original_has else "both" if english_has and original_has else "neither"
            goal_examples[category].append({
                "sample_id": row["sample_id"], "diagnostic_split": split,
                "action_index": index, "literal": literal, "original_goal": original,
                "english_goal": english, "record_path": relative,
                "image_path": row["image_path"], "image_sha256": row.get("image_sha256"),
                "original_literal_present": original_has, "english_literal_present": english_has,
                "category": category,
            })
        counts.update(raw.keys())
        elements = {action.get("element") for action in raw.get("actions", [])
                    if isinstance(action, dict) and isinstance(action.get("element"), str)}
        current_task_equal_element += raw.get("current_task") in elements
        summaries = {}
        for field, value in raw.items():
            if not isinstance(value, str):
                continue
            has_schema = bool(re.search(r'action_?type|mouse_position|keyboard_text|clickable_area',
                                        value.replace("\\", ""), re.IGNORECASE))
            schema_markers[field] += has_schema
            literals = [action["text"] for action in row["actions"]
                        if action["action_type"] == "type_text"]
            mentions = sum(text in value for text in literals)
            literal_mentions[field] += mentions
            summaries[field] = {"characters": len(value), "sha256": value_sha256(value),
                                "contains_action_schema_marker": has_schema,
                                "exact_current_type_text_literal_mentions": mentions}
        records.append({"sample_id": row["sample_id"], "record_path": relative,
                        "sha256": hashlib.sha256(raw_bytes).hexdigest(), "fields": sorted(raw),
                        "string_summaries": summaries})
    known = {field for policy in FIELD_POLICIES for field in policy["fields"]}
    # These examples illustrate source translation, never choose a model input or
    # a diagnostic sample. Include each evidence category before adding examples.
    examples = []
    categories = sorted(goal_examples)
    for values in goal_examples.values():
        values.sort(key=lambda example: (value_sha256([42, example["sample_id"], example["action_index"]]),
                                         example["sample_id"], example["action_index"]))
    offset = 0
    while len(examples) < min(10, sum(len(values) for values in goal_examples.values())):
        for category in categories:
            if offset < len(goal_examples[category]):
                examples.append(goal_examples[category][offset])
                if len(examples) == 10:
                    break
        offset += 1
    return {"scope": "only retained train/validation records from official training source",
            "records": records, "field_present_counts": dict(sorted(counts.items())),
            "field_action_schema_marker_counts": dict(sorted(schema_markers.items())),
            "field_current_type_text_literal_mentions": dict(sorted(literal_mentions.items())),
            "current_task_exactly_matches_action_element_records": current_task_equal_element,
            "field_policies": list(FIELD_POLICIES), "unknown_fields_require_review": sorted(set(counts)-known),
            "goal_translation_lexical_evidence_by_split": {
                split: dict(sorted(stats.items())) for split, stats in sorted(goal_evidence.items())},
            "goal_translation_examples": examples,
            "goal_translation_examples_scope": "scoring-side source examples only; selection and model "
                                               "input never depend on these target-literal categories",
            "goal_translation_interpretation": "Exact lexical evidence only, including short strings; "
                                               "absence is not a semantic translation or learnability verdict.",
            "proposed_v2_goal_policy": "On every row retain original and English task goals if distinct; "
                                       "deduplicate only equal/absent goals. Decide using the two goal "
                                       "fields alone, never actions/literal-presence/response/history.",
            "standalone_prior_history_verified": False,
            "history_admission_requirements": [
                "Prove the field existed before current-step label/response generation.",
                "Parse an explicit allowlist of past observations/actions, excluding current/future labels.",
                "Preserve trajectory ordering and image timing; do not reconstruct history from file order alone.",
                "Manually review leaked action descriptions/examples and version the new input contract.",
                "Validate any context change on frozen validation; keep official test unopened for selection.",
            ], "model_inputs_exported": False,
            "scoring_side_examples_contain_labels_not_safe_prompt_payload": True}


def _selection_entry(row: Mapping) -> dict[str, Any]:
    return {**{key: row[key] for key in (
        "sample_id", "group_id", "source", "source_split", "diagnostic_split", "image_path",
        "image_width", "image_height", "image_sha256", "image_file_sha256", "source_sample_sha256",
    )}, "instruction_sha256": value_sha256(row["instruction"]),
        "target_sha256": value_sha256(row["actions"]),
        "target_json_sha256": hashlib.sha256(action_target_json(row["actions"]).encode()).hexdigest(),
        "annotations_sha256": value_sha256(row["action_annotations"]), "stratum": _stratum(row)}


def diagnose_dataset(
    train_path: str | Path, validation_path: str | Path, output_dir: str | Path, *,
    seed: int = 42, train_limit: int = 64, validation_limit: int = 0,
    raw_train_root: str | Path | None = None, source_files: Sequence[str | Path] = (),
    frozen_selection: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze all identities before future generation; refuse output/evidence overwrite."""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("diagnostic output directory is not empty; use a new destination")
    train_file, validation_file = Path(train_path).resolve(), Path(validation_path).resolve()
    before = {"train": file_sha256(train_file), "validation": file_sha256(validation_file)}
    sources = [{"path": str(Path(path).resolve(strict=True)), "sha256": file_sha256(path)}
               for path in source_files]
    train = load_development_rows(train_file, split="train")
    validation = load_development_rows(validation_file, split="validation")
    ensure_development_isolation(train, validation)
    selected = {"train": select_diagnostic_rows(train, seed=seed, limit=train_limit),
                "validation": select_diagnostic_rows(validation, seed=seed, limit=validation_limit)}
    selection = {"version": DIAGNOSTIC_VERSION, "seed": seed,
                 "input_data_sha256": before, "test_data_consumed": False,
                 "method": "all if limit=0; otherwise hash-ranked type/sequence strata round-robin",
                 "limits": {"train": train_limit, "validation": validation_limit},
                 "purposes": {"train": "training-fit diagnostic, never heldout generalization",
                              "validation": "development validation, never official test result"},
                 "samples": [_selection_entry(row) for split in ("train", "validation")
                             for row in selected[split]]}
    if frozen_selection is not None:
        saved = _strict_json(Path(frozen_selection).read_text(encoding="utf-8-sig"))
        if saved != selection:
            raise ValueError("frozen selection differs from data/image/label/seed policy")
    raw_review = review_raw_fields(train + validation, raw_train_root) if raw_train_root else {
        "scope": "raw source not supplied", "field_policies": list(FIELD_POLICIES),
        "raw_source_verified": False, "model_inputs_exported": False}
    audit = {"version": DIAGNOSTIC_VERSION, "test_data_consumed": False,
             "desktop_task_e2e": False, "selection_frozen_before_generation": True,
             "train": audit_rows(train), "validation": audit_rows(validation),
             "selected_counts": {key: len(value) for key, value in selected.items()},
             "limitations": ["No model result or learnability verdict is produced by this CPU audit.",
                             "Validation is for development; official test is forbidden for selection.",
                             "Input text remains overall task only until a reviewed context contract exists."]}
    after = {"train": file_sha256(train_file), "validation": file_sha256(validation_file)}
    if after != before:
        raise ValueError("development data changed during audit")
    if any(file_sha256(source["path"]) != source["sha256"] for source in sources):
        raise ValueError("source code/evidence changed during audit")
    if any(file_sha256(row["image_path"]) != row["image_file_sha256"] for row in train + validation):
        raise ValueError("screenshot bytes changed during audit")
    if raw_train_root and any(
        file_sha256(Path(raw_train_root) / record["record_path"].replace("\\", "/")) != record["sha256"]
        for record in raw_review["records"]
    ):
        raise ValueError("raw training record changed during audit")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("selection.json", selection), ("audit.json", audit),
                        ("raw_field_review.json", raw_review)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                             allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"version": DIAGNOSTIC_VERSION, "test_data_consumed": False,
                "input_data": {key: {"path": str(path), "sha256": before[key]}
                               for key, path in (("train", train_file), ("validation", validation_file))},
                "source_files": sources, "selection_sha256": value_sha256(selection),
                "raw_training_root": str(Path(raw_train_root).resolve()) if raw_train_root else None,
                "files": [{"path": name, "bytes": (output / name).stat().st_size,
                           "sha256": file_sha256(output / name)}
                          for name in ("selection.json", "audit.json", "raw_field_review.json")],
                "selected_counts": audit["selected_counts"], "inputs_unchanged_during_audit": True}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                         + "\n", encoding="utf-8")
    return manifest
