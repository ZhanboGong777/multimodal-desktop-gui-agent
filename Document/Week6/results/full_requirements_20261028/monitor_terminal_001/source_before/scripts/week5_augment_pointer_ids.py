"""Prepare registered train-only four-view candidate-numbering data, without encoding."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gui_agent.finetuning.evaluation import (
    canonical_json,
    file_sha256,
    strict_json_loads,
    value_sha256,
)
from gui_agent.finetuning.pointer_numbering import (
    SEED,
    VERSION,
    VIEWS,
    numbered_view,
    remap_target,
    verify_numbered_view,
)
from gui_agent.finetuning.pointer_protocol import (
    MOUSE_TYPES,
    map_pointer_actions,
    selection_binding,
    validate_pointer_sample,
)
from gui_agent.finetuning.training import load_prepared_records


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def within(path: str | Path, workspace_root: Path) -> Path:
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(workspace_root), f"path escapes workspace: {resolved}")
    return resolved


def reject_nontrain_path(path: str | Path, workspace_root: Path) -> None:
    value = Path(path)
    parts = value.resolve().relative_to(workspace_root).parts if value.is_absolute() else value.parts
    require(not any(re.search(r"(?:^|[_.-])(?:validation|dev|development|test|official[_-]test)(?:[_.-]|$)", part.casefold()) for part in parts), f"explicit validation/dev/test path rejected: {path}")


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_bound(path: Path, expected_sha256: str | None = None) -> tuple[bytes, str]:
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    require(expected_sha256 is None or digest == expected_sha256, f"expected input SHA differs: {path}")
    require(file_sha256(path) == digest, f"input changed during initial read: {path}")
    return contents, digest


def source_slots(row: dict) -> list[tuple[int, str]]:
    return [(index, endpoint) for index, action in enumerate(row["candidate_target"])
            if action["action_type"] in MOUSE_TYPES
            for endpoint in (("start_", "end_") if action["action_type"] == "drag" else ("",))]


def prepare_dataset(train_path: str | Path, output_dir: str | Path, *, preparation_plan: str | Path,
                    workspace_root: str | Path = ROOT,
                    views: int = VIEWS, seed: int = SEED) -> dict[str, Any]:
    """Create only new files; all admitted-source rows and all four views required."""
    workspace = Path(workspace_root).resolve(strict=True)
    source = within(train_path, workspace)
    output = within(output_dir, workspace)
    reject_nontrain_path(source, workspace)
    reject_nontrain_path(output, workspace)
    require(type(views) is int and views == VIEWS and type(seed) is int and seed == SEED, "registered4views/seed42 required")
    require(source.is_file(), "train source file missing")
    require(not output.exists(), "output already exists; refusing collision/overwrite")
    plan_path = within(preparation_plan, workspace)
    plan_bytes, plan_sha = read_bound(plan_path)
    plan = strict_json_loads(plan_bytes.decode("utf8"))
    require(plan["schema"] == "week5.train_only_numbering_preparation_plan.v1" and plan["status"] == "registered_before_actual_augmentation", "registered train-only preparation plan required")
    bound = {Path(name).resolve(): digest for name, digest in plan["bound_sha256"].items()}
    require(source in bound, "train source path is not the registered train entry")
    budget = plan["actual_preparation_budget"]
    require(budget["views_per_parent"] == views and budget["seed"] == seed, "registered view/seed budget differs")
    source_bytes, source_initial_sha = read_bound(source, bound[source])
    raw_rows = [strict_json_loads(line) for line in source_bytes.decode("utf8").splitlines() if line.strip()]
    require(bool(raw_rows), "training source cannot be empty")
    require(len(raw_rows) == budget["source_train_rows"] and len(raw_rows) * views == budget["total_derived_rows"], "registered train membership/row budget differs")
    for row in raw_rows:
        require(isinstance(row, dict) and row.get("source_split") == "train", "only source_split=train rows allowed; dev/test rejected")
        require(all(row.get(field, "train") == "train" for field in ("split", "partition", "dataset_split", "admission_split")), "explicit validation/dev/test row partition rejected")
        require(row["candidate_snapshot"].get("compact_policy") == "query_relations_v3", "only original v3 training parents allowed")
        require(Path(row["image_path"]).is_absolute(), "frozen parent image_path must be absolute")
        within(row["image_path"], workspace)
        reject_nontrain_path(row["image_path"], workspace)
        parent_review = within(row["candidate_label_evidence"]["review_path"], workspace)
        reject_nontrain_path(parent_review, workspace)
        if "record_path" in row:
            reject_nontrain_path(row["record_path"], workspace)
        require(parent_review in bound and bound[parent_review] == row["candidate_label_evidence"]["review_sha256"], "parent review is not bound in the registered plan")
    require(len({row["sample_id"] for row in raw_rows}) == len(raw_rows), "duplicate training parent IDs")
    reviews = {}
    initially_bound = {str(source): source_initial_sha, str(plan_path): plan_sha}
    input_paths = {source, plan_path, Path(__file__).resolve(), ROOT / "src/gui_agent/finetuning/pointer_numbering.py"}
    for row in raw_rows:
        path = within(row["candidate_label_evidence"]["review_path"], workspace)
        input_paths.add(path)
        input_paths.add(Path(row["image_path"]).resolve())
        if path not in reviews:
            review_bytes, review_sha = read_bound(path, row["candidate_label_evidence"]["review_sha256"])
            initially_bound[str(path)] = review_sha
            document = strict_json_loads(review_bytes.decode("utf8"))
            require(document["schema"] == "week5.pointer_semantic_review.v1" and document["status"] == "closed", "original semantic review must be closed")
            reviews[path] = document
        require(initially_bound[str(path)] == row["candidate_label_evidence"]["review_sha256"], "parents disagree about shared review SHA")
        for index, endpoint in source_slots(row):
            matches = [r for r in reviews[path]["reviews"] if r["sample_id"] == row["sample_id"] and r["action_index"] == index and r.get("endpoint", "") == endpoint]
            require(len(matches) == 1 and matches[0]["decision"] == "accepted", "parent requires unique accepted train review slot")
            require(matches[0].get("split", "train") == "train", "shared review slot explicitly belongs to validation/test")
            snapshot_path = within(matches[0]["snapshot_path"], workspace)
            reject_nontrain_path(snapshot_path, workspace)
            input_paths.add(snapshot_path)
            _, snapshot_sha = read_bound(snapshot_path, matches[0]["snapshot_file_sha256"])
            initially_bound[str(snapshot_path)] = snapshot_sha
    source_modules = [ROOT / "src/gui_agent/finetuning" / name for name in
                      ("pointer_protocol.py", "candidate_grounding.py", "pointer_candidates_v3.py", "training.py", "evaluation.py", "data.py")]
    input_paths.update(source_modules)
    before = {str(path): file_sha256(path) for path in sorted(input_paths)}
    require(all(before[name] == digest for name, digest in initially_bound.items()), "initially bound source/review/snapshot changed while reading")
    before.update(initially_bound)
    loaded = load_prepared_records(source, target_protocol="candidate_compact")
    require(len(loaded) == len(raw_rows), "source loader silently changed row count")
    require(len({r["group_id"] for r in raw_rows}) == budget["source_train_groups"], "registered train group count differs")
    require(len({r["image_sha256"] for r in raw_rows}) == budget["unique_native_images"], "registered train image count differs")
    # Use original rows for immutable parent hashes/fields; loader supplies independent validation only.
    for original, normalized in zip(raw_rows, loaded, strict=True):
        expected_row = copy.deepcopy(original)
        expected_row["image_path"] = str(Path(original["image_path"]).resolve())
        require(all(normalized.get(key) == value for key, value in expected_row.items()), "source row fields changed during loading; no implicit rewrite allowed")
        require(set(normalized) <= set(expected_row) | {"image_sha256", "image_file_sha256"}, "loader injected unexpected source fields")
        require(original["image_file_sha256"] == normalized["image_file_sha256"], "source image file hash differs")
    plans = []
    for row in raw_rows:
        for view_index in range(views):
            view = numbered_view(row, view_index=view_index, seed=seed)
            verify_numbered_view(row, view)
            target = remap_target(row["candidate_target"], view["old_to_new"])
            require(remap_target(target, view["new_to_old"]) == row["candidate_target"], "whole pointer target inverse is not exact")
            derived_id = f"{row['sample_id']}:numbering-v1:view{view_index}"
            plans.append((row, view, target, derived_id))
    require(len({plan[3] for plan in plans}) == len(raw_rows) * views, "derived sample ID collision")
    require(not ({plan[3] for plan in plans} & {row["sample_id"] for row in raw_rows}), "derived IDs overlap parents")
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    prepared, inherited = [], []
    review_path = output / "derived_semantic_review.json"
    for parent, view, target, identifier in plans:
        row = copy.deepcopy(parent)
        snapshot = view["candidate_snapshot"]
        snapshot_path = snapshots / f"{value_sha256({'sample_id': identifier})}.json"
        write_json(snapshot_path, snapshot)
        binding = selection_binding(snapshot, instruction=parent["instruction"], actor_step=parent["actor_step"])
        row.update(sample_id=identifier, parent_sample_id=parent["sample_id"], parent_row_sha256=value_sha256(parent),
                   candidate_snapshot=snapshot, candidate_target=target, candidate_selection_evidence=binding,
                   candidate_label_evidence={"review_path": str(review_path), "review_sha256": None},
                   training_admission=False, tokens_not_measured=True, quality_not_evaluated=True)
        row["numbering_label_evidence"] = {"version": VERSION, "view_index": view["view_index"], "seed": seed,
                                          "parent_selection_evidence": view["parent_selection_evidence"],
                                          "parent_candidate_label_evidence": copy.deepcopy(parent["candidate_label_evidence"]),
                                          "parent_row_sha256": value_sha256(parent),
                                          "old_to_new": view["old_to_new"], "new_to_old": view["new_to_old"],
                                          "native_GT_is_label_side_only": True,
                                          "derived_snapshot_is_preselected_remap_not_new_detection": True}
        parent_review_path = Path(parent["candidate_label_evidence"]["review_path"]).resolve()
        for index, endpoint in source_slots(parent):
            original = next(r for r in reviews[parent_review_path]["reviews"] if r["sample_id"] == parent["sample_id"] and r["action_index"] == index and r.get("endpoint", "") == endpoint)
            review = copy.deepcopy(original)
            selected_index = target[index][endpoint + "candidate"]
            review.update(sample_id=identifier, preferred_candidate_id=snapshot["candidates"][selected_index - 1]["candidate_id"],
                          snapshot_path=str(snapshot_path), snapshot_file_sha256=file_sha256(snapshot_path),
                          annotation_origin="inherited_remapped_original_AI_review", fresh_image_reviewed=False,
                          parent_sample_id=parent["sample_id"], parent_review_record_sha256=value_sha256(original),
                          parent_review_path=str(parent_review_path), parent_review_file_sha256=file_sha256(parent_review_path))
            inherited.append(review)
        require(map_pointer_actions(canonical_json(target), snapshot, row["image_path"], instruction=row["instruction"], actor_step=row["actor_step"], expected_selection=binding) == parent["actions"], "derived target differs from complete original native actions")
        prepared.append(row)
    parent_ids = {row["sample_id"] for row in raw_rows}
    shared_other_annotations = any(record["sample_id"] not in parent_ids for doc in reviews.values() for record in doc["reviews"])
    declared_validation = any(doc.get("shared_review_contains_validation_annotations") is True or any(
        r.get("split") == "validation" for r in doc["reviews"]) or any(
        item.get("partition") == "validation" for item in doc.get("partition_receipts", [])) for doc in reviews.values())
    write_json(review_path, {"schema": "week5.pointer_semantic_review.v1", "status": "closed",
                             "annotation_origin": "inherited_remapped_original_AI_review",
                             "fresh_image_reviewed": False, "new_gold_created": False,
                             "shared_review_contains_validation_annotations": declared_validation,
                             "shared_review_contains_other_parent_annotations": shared_other_annotations,
                             "shared_review_validation_presence_declaration": "explicit original document/partition receipt or review-record split; missing metadata is not inferred from unrelated train parents",
                             "validation_annotations_used": 0, "reviews": inherited})
    review_sha = file_sha256(review_path)
    for row in prepared:
        row["candidate_label_evidence"]["review_sha256"] = review_sha
        validate_pointer_sample(row)
    data_path = output / "train.jsonl"
    with data_path.open("x", encoding="utf8") as handle:
        for row in prepared:
            handle.write(canonical_json(row) + "\n")
    require(len(load_prepared_records(data_path, target_protocol="candidate_compact")) == len(prepared), "derived actual loader row count changed")
    after = {name: file_sha256(name) for name in before}
    require(after == before, "input/source hashes changed during preparation")
    summary = {"schema": VERSION, "status": "completed_CPU_numbering_preparation_only",
               "created_at": datetime.now(UTC).isoformat(), "source_path": str(source), "output_path": str(output),
               "registered_plan_path": str(plan_path), "registered_plan_sha256": plan_sha,
               "parent_rows": len(raw_rows), "derived_rows": len(prepared), "views": views, "seed": seed,
               "groups": len({r["group_id"] for r in prepared}), "source_split": "train",
               "unique_images": len({r["image_sha256"] for r in prepared}),
               "training_admission": False, "tokens_not_measured": True, "quality_not_evaluated": True,
               "model_processor_detector_GUI_calls": 0, "validation_dataset_or_predictions_read": 0,
               "shared_review_contains_validation_annotations": declared_validation,
               "shared_review_contains_other_parent_annotations": shared_other_annotations,
               "shared_review_validation_presence_declaration": "explicit original document/partition receipt or review-record split; absent metadata is not proof of no validation annotations",
               "validation_annotations_used": 0,
               "new_sources_or_gold_created": False, "native_whole_roundtrip_verified": len(prepared),
               "actual_pointer_loader_verified": len(prepared), "variant0_original_table_order": True,
               "input_source_sha256_before": before, "input_source_sha256_after": after,
               "output_sha256": {str(path): file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
               "current_model_campaign_not_reopened": True, "new_campaign_unseen_source_evaluation_required": True}
    write_json(output / "preparation_receipt.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--views", type=int, choices=[VIEWS], default=VIEWS)
    parser.add_argument("--seed", type=int, choices=[SEED], default=SEED)
    args = parser.parse_args()
    summary = prepare_dataset(args.train, args.output, preparation_plan=args.plan, views=args.views, seed=args.seed)
    print(json.dumps({"status": summary["status"], "parent_rows": summary["parent_rows"],
                      "derived_rows": summary["derived_rows"], "output": summary["output_path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
