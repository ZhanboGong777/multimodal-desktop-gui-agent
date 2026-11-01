"""Register audited visual assets as explicit inputs, without training admission.

The caller pins a plan SHA before execution. This script copies metadata only;
it never regenerates an image, loads a processor, or changes the old campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gui_agent.finetuning.candidate_grounding import snapshot_hash
from gui_agent.finetuning.evaluation import canonical_json, strict_json_loads, value_sha256
from gui_agent.finetuning.pointer_protocol import map_pointer_actions
from gui_agent.finetuning.visual_pointer_input import (
    MANIFEST_SCHEMA,
    PROTOCOL,
    VisualAssetRegistry,
    validate_visual_input,
    visual_input_binding,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_read(path: Path, digest: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"registered bytes differ: {path}")
    return content


def safe_path(value: str, root: Path, *, existing: bool = True) -> Path:
    lexical = Path(value)
    resolved = lexical.resolve(strict=existing)
    if not lexical.is_absolute() or lexical != resolved or not resolved.is_relative_to(root):
        raise ValueError("path alias, relative path or escaped registered root")
    if any(part.casefold() in {"dev", "test", "validation", "dev.jsonl", "test.jsonl"}
           for part in resolved.parts):
        raise ValueError("non-training data path refused")
    return resolved


def merge_binding(bound: dict[str, str], path: Path, digest: str) -> None:
    name = str(path)
    if name in bound and bound[name] != digest:
        raise ValueError("conflicting registered or first-read SHA256 baseline")
    bound[name] = digest


def write_new(path: Path, value: object, *, owned_files: set[Path] | None = None) -> None:
    safe_path(str(path), ROOT / "outputs", existing=False)
    content = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        if owned_files is not None:
            owned_files.add(path)
        handle.write(content)


def run(plan_path: Path, expected_sha: str) -> int:
    start = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    bound: dict[str, str] = {str(plan_path): expected_sha}
    output = None
    result = None
    error = None
    cancelled = False
    owned_files: set[Path] = set()
    try:
        plan = strict_json_loads(bound_read(plan_path, expected_sha).decode("utf-8"))
        if (plan["schema"], plan["status"]) != (
            "week5.visual_input_preparation_plan.v1", "registered_before_execution",
        ):
            raise ValueError("registered visual input plan required")
        if plan["budgets"] != {"source_views": 404, "parent_rows": 101,
                              "rendered_views": 338, "refused_views": 66,
                              "model_calls": 0, "processor_calls": 0, "gui_calls": 0}:
            raise ValueError("fixed input preparation budget differs")
        output = safe_path(plan["output_dir"], ROOT / "outputs", existing=False)
        if output.exists() or output.parent != plan_path.parent:
            raise ValueError("output must be a fresh sibling of the registered plan")
        for name, digest in plan["bound_sha256"].items():
            path = safe_path(name, ROOT)
            bound_read(path, digest)
            merge_binding(bound, path, digest)
        if bound.get(str(Path(__file__).resolve())) != sha(Path(__file__)):
            raise ValueError("executed helper not pinned in the plan")
        audit_info = plan["asset_audit"]
        audit_path = safe_path(audit_info["path"], ROOT / "outputs")
        if bound.get(str(audit_path)) != audit_info["sha256"]:
            raise ValueError("independent asset audit is not pinned")
        audit = strict_json_loads(bound_read(audit_path, audit_info["sha256"]).decode("utf-8"))
        if (audit["status"] != "passed_AI_internal_asset_correctness_and_complete_denominator_audit"
                or not audit["checks"] or not all(check["passed"] is True for check in audit["checks"])
                or audit["reconstructed_assets"] != 338
                or audit["native_whole_roundtrip_independently_verified"] != 404):
            raise ValueError("independent full asset correctness audit required")

        def read_source(key: str):
            info = plan["sources"][key]
            path = safe_path(info["path"], ROOT)
            if bound.get(str(path)) != info["sha256"]:
                raise ValueError("source is not in the caller-pinned ledger")
            if audit["bound_sha256"].get(str(path)) != info["sha256"]:
                raise ValueError("source bytes were not covered by independent asset audit")
            content = bound_read(path, info["sha256"]).decode("utf-8")
            return [strict_json_loads(line) for line in content.splitlines() if line.strip()]

        rows = read_source("numbered_train")
        parents_list = read_source("original_train_parents")
        decisions = read_source("render_index")
        if (len(rows), len(parents_list), len(decisions)) != (404, 101, 404):
            raise ValueError("complete registered denominators differ")
        parents = {row["sample_id"]: row for row in parents_list}
        if len(parents) != 101 or len({row["sample_id"] for row in rows}) != 404:
            raise ValueError("source identities duplicate")
        if Counter(row["parent_sample_id"] for row in rows) != Counter({key: 4 for key in parents}):
            raise ValueError("four-view parent lineage differs")
        if len({row["group_id"] for row in rows}) != 31:
            raise ValueError("registered original group count differs")
        if any(row["source_split"] != "train" for row in rows + parents_list):
            raise ValueError("training-only scope violated")
        counts = Counter(row["status"] for row in decisions)
        if counts != Counter(rendered_layout_only=338, refused_layout=66):
            raise ValueError("audited layout denominator differs")
        render_path = safe_path(plan["render_manifest"]["path"], ROOT / "outputs")
        if (bound.get(str(render_path)) != plan["render_manifest"]["sha256"]
                or audit["bound_sha256"].get(str(render_path)) != plan["render_manifest"]["sha256"]):
            raise ValueError("renderer manifest is not in registered plan ledger")
        render_manifest = strict_json_loads(bound_read(
            render_path, plan["render_manifest"]["sha256"],
        ).decode("utf-8"))
        if (render_manifest["status"] != "completed_CPU_layout_feasibility_only"
                or render_manifest["complete_decision_denominator"] != 404
                or render_manifest["rendered_views"] != 338 or render_manifest["refused_views"] != 66):
            raise ValueError("renderer result differs from independent audit")
        for name, digest in render_manifest["output_sha256"].items():
            path = safe_path(name, ROOT / "outputs")
            if audit["bound_sha256"].get(str(path)) != digest:
                raise ValueError("renderer asset bytes were not covered by independent audit")
            bound_read(path, digest)
            merge_binding(bound, path, digest)
        output.mkdir(exist_ok=False)
        (output / "raw_snapshots").mkdir()
        (output / "visual_metadata").mkdir()
        raw_paths = {}
        for parent in parents_list:
            raw = parent["candidate_snapshot"]
            path = output / "raw_snapshots" / (snapshot_hash(raw) + ".json")
            if not path.exists():
                write_new(path, raw)
            raw_paths[parent["sample_id"]] = (path, sha(path))
        prepared, registry_entries, all_decisions = [], [], []
        for index, (row, decision) in enumerate(zip(rows, decisions, strict=True)):
            parent = parents[row["parent_sample_id"]]
            native_path = safe_path(row["image_path"], ROOT / "data")
            if audit["bound_sha256"].get(str(native_path)) != row["image_file_sha256"]:
                raise ValueError("original native file was not covered by independent audit")
            bound_read(native_path, row["image_file_sha256"])
            merge_binding(bound, native_path, row["image_file_sha256"])
            if decision["source_index"] != index or decision["sample_id"] != row["sample_id"]:
                raise ValueError("layout order or sample identity differs")
            if decision["source_row_sha256"] != value_sha256(row) or decision["parent_row_sha256"] != value_sha256(parent):
                raise ValueError("layout original rows differ")
            if row["image_sha256"] != parent["image_sha256"] or row["actions"] != parent["actions"]:
                raise ValueError("augmentation changed original RGB or whole action array")
            mapped = map_pointer_actions(canonical_json(row["candidate_target"]), row["candidate_snapshot"],
                                         row["image_path"], instruction=row["instruction"],
                                         actor_step=row["actor_step"],
                                         expected_selection=row["candidate_selection_evidence"])
            if mapped != row["actions"]:
                raise ValueError("complete original native action mapping differs")
            tracked = {"source_index": index, "sample_id": row["sample_id"],
                       "parent_sample_id": row["parent_sample_id"], "group_id": row["group_id"],
                       "source_row_sha256": value_sha256(row), "layout_status": decision["status"],
                       "native_whole_roundtrip_verified": True, "training_admission": False}
            if decision["status"] == "refused_layout":
                tracked.update(status="refused_layout", reason=decision["reason"])
                all_decisions.append(tracked)
                continue
            metadata = decision["renderer_metadata"]
            png = safe_path(decision["model_image_path"], ROOT / "outputs")
            if render_manifest["output_sha256"].get(str(png)) != metadata["model_png_sha256"]:
                raise ValueError("PNG not in the audited renderer ledger")
            path = output / "visual_metadata" / (value_sha256({"sample_id": row["sample_id"]}) + ".json")
            write_new(path, metadata)
            raw_path, raw_sha = raw_paths[row["parent_sample_id"]]
            visual_row = {**row, "input_image_protocol": PROTOCOL,
                          "model_image_path": str(png), "model_png_sha256": metadata["model_png_sha256"],
                          "model_native_rgb_sha256": metadata["model_native_rgb_sha256"],
                          "visual_metadata_path": str(path), "visual_metadata_file_sha256": sha(path),
                          "raw_snapshot_path": str(raw_path), "raw_snapshot_file_sha256": raw_sha,
                          "training_admission": False, "tokens_not_measured": True,
                          "quality_not_evaluated": True}
            registry_entries.append({**visual_input_binding(visual_row),
                                     "font_sha256": metadata["font_sha256"],
                                     "config_sha256": metadata["config_sha256"]})
            prepared.append(visual_row)
            tracked.update(status="registered_asset_pending_processor", input_binding_sha256=registry_entries[-1]["input_binding_sha256"])
            all_decisions.append(tracked)
        registry_path = output / "asset_manifest.json"
        write_new(registry_path, {"schema": MANIFEST_SCHEMA, "status": "approved",
                                  "input_image_protocol": PROTOCOL, "entries": registry_entries},
                  owned_files=owned_files)
        registry_sha = sha(registry_path)
        registry = VisualAssetRegistry(str(registry_path), registry_sha,
                                       str(ROOT / "data"), str(ROOT / "outputs"))
        for index, row in enumerate(prepared):
            descriptor = validate_visual_input(row, registry=registry)
            descriptor.assert_current()
            if descriptor.model_native_rgb_sha256 == descriptor.original_native_rgb_sha256:
                raise ValueError("model processor input did not change raster")
            if (index + 1) % 64 == 0 or index + 1 == len(prepared):
                print(f"INPUT VERIFIED {index + 1}/{len(prepared)}; full denominator=404", flush=True)
        write_new(output / "visual_rows.jsonl", "".join(canonical_json(row) + "\n" for row in prepared))
        write_new(output / "decisions.jsonl", "".join(canonical_json(row) + "\n" for row in all_decisions))
        write_new(output / "registry_config.json", {
            "input_image_protocol": PROTOCOL, "visual_asset_manifest_path": str(registry_path),
            "visual_asset_manifest_sha256": registry_sha, "visual_original_root": str(ROOT / "data"),
            "visual_asset_root": str(ROOT / "outputs"),
        })
        if any(sha(Path(path)) != digest for path, digest in bound.items()):
            raise ValueError("bound source or audited asset changed during preparation")
        ledger = {str(path): sha(path) for path in sorted(output.rglob("*")) if path.is_file()}
        result = {"schema": "week5.visual_input_preparation_result.v1",
                  "status": "completed_input_asset_registration_only", "source_views": 404,
                  "parent_rows": 101, "groups": 31, "registered_views": 338, "refused_views": 66,
                  "native_whole_roundtrips": 404, "registered_inputs_validated": 338,
                  "output_sha256": ledger, "bound_sha256": bound,
                  "asset_manifest_sha256": registry_sha, "training_admission": False,
                  "processor_calls": 0, "model_calls": 0, "gui_calls": 0,
                  "tokens_measured": False, "quality_evaluated": False,
                  "W5_complete": False, "T04_closed": False}
        write_new(output / "manifest.json", result, owned_files=owned_files)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - keep failure evidence
        error = {"type": type(exc).__name__, "message": str(exc)}
        cancelled = isinstance(exc, KeyboardInterrupt)
    try:
        stable = all(Path(path).exists() and sha(Path(path)) == digest for path, digest in bound.items())
        if result is not None:
            stable = stable and all(sha(Path(path)) == digest for path, digest in result["output_sha256"].items())
    except OSError:
        stable = False
    code = 0 if error is None and stable and result is not None else 1
    if code != 0 and output is not None:
        for path in owned_files:
            # Only this run's two publication files; no recursive cleanup.
            if safe_path(str(path), ROOT / "outputs", existing=False).parent == output:
                path.unlink(missing_ok=True)
    receipt = {"schema": "week5.visual_input_preparation_execution_receipt.v1",
               "command": sys.orig_argv, "cwd": str(Path.cwd()), "started_at_utc": started,
               "finished_at_utc": datetime.now(UTC).isoformat(), "exit_code": code,
               "elapsed_seconds": time.perf_counter() - start, "cancelled": cancelled,
               "error": error, "bound_inputs_stable": stable, "bound_sha256": bound,
               "result_valid": code == 0, "output_dir": str(output) if output else None,
               "counts": None if result is None else {key: result[key] for key in
                    ("source_views", "registered_views", "refused_views", "native_whole_roundtrips")},
               "training_admission": False, "processor_calls": 0, "model_calls": 0, "gui_calls": 0}
    write_new(plan_path.parent / "preparation_execution_receipt_001.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("exit_code", "elapsed_seconds", "error", "counts")}), flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    return run(safe_path(str(args.plan), ROOT / "outputs"), args.plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
