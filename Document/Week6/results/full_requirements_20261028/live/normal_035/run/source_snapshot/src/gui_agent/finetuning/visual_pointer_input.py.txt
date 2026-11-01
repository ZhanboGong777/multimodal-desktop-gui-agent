"""Explicit, caller-registered visual raster contract; no label-side access.

This validates frozen engineering assets. It neither admits training samples nor
proves that a processor can read the annotation. Native coordinates and grouping
continue to identify the original screenshot, never the annotated raster.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw

from .candidate_grounding import center, snapshot_hash, validate_snapshot
from .evaluation import value_sha256
from .pointer_protocol import (
    CONTEXT_POLICY,
    select_candidates,
    selection_binding,
)
from .visual_pointer import (
    _config,
    _leader,
    _native_hash,
    _pixels,
    _positions,
    _raw_protection,
    _rect,
    _sha,
)
from .visual_pointer_prompts import (
    GALLERY_PROMPT_VERSIONS,
    build_visual_user_prompt,
    declared_prompt_version,
)
from .visual_pointer_v2 import LAYOUT_POLICY, VERSION

PROTOCOL = "visual_pointer_v2"
MANIFEST_SCHEMA = "week5.visual_pointer_asset_manifest.v1"
GALLERY_PROTOCOL = "visual_pointer_gallery_v3"
GALLERY_MANIFEST_SCHEMA = "week5.visual_pointer_gallery_asset_manifest.v1"
GALLERY_PROMPT_VERSION = "week5.visual-pointer-prompt.gallery.v4"
VISUAL_PROTOCOLS = (PROTOCOL, GALLERY_PROTOCOL)
FORBIDDEN_SNAPSHOT_KEYS = frozenset({
    "actions", "action_annotations", "candidate_target", "candidate_label_evidence", "review",
    "targets", "request", "response", "plan", "task_record", "sample_id",
})


def _path(value: Any) -> str:
    """Lexical absolute identity only: this helper does not access the filesystem."""
    if not isinstance(value, (str, Path)):
        raise TypeError("visual asset path must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("visual asset path must be absolute without parent traversal")
    return str(path)


def _visible(sample: Mapping[str, Any]) -> dict[str, Any]:
    protocol = sample["input_image_protocol"]
    if protocol not in VISUAL_PROTOCOLS:
        raise ValueError("explicit registered visual input protocol required")
    version = declared_prompt_version(sample)
    if (protocol == GALLERY_PROTOCOL) != (version in GALLERY_PROMPT_VERSIONS):
        raise ValueError("gallery protocol and explicit approved gallery prompt must agree")
    width, height = sample["image_width"], sample["image_height"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("native dimensions must be positive integers")
    instruction, step = sample["instruction"], sample["actor_step"]
    if any(not isinstance(v, str) or not v.strip() for v in (instruction, step)):
        raise ValueError("explicit instruction and actor_step required")
    snapshot = sample["candidate_snapshot"]
    if not isinstance(snapshot, Mapping):
        raise TypeError("candidate snapshot must be an object")
    if FORBIDDEN_SNAPSHOT_KEYS.intersection(snapshot):
        raise ValueError("candidate snapshot must contain input-only evidence")
    snapshot = copy.deepcopy(dict(snapshot))
    if (snapshot["image_width"], snapshot["image_height"]) != (width, height):
        raise ValueError("snapshot native dimensions differ")
    binding = selection_binding(snapshot, instruction=instruction, actor_step=step)
    expected = sample["candidate_selection_evidence"]
    if not isinstance(expected, Mapping) or dict(expected) != binding:
        raise ValueError("all-policy candidate selection binding differs")
    selected = select_candidates(snapshot, instruction=instruction, actor_step=step)
    if not 0 < len(selected) <= 64:
        raise ValueError("complete visual table must contain 1..64 selected candidates")
    result = {"image_path": _path(sample["image_path"]), "image_width": width, "image_height": height,
            "instruction": instruction, "actor_step": step, "candidate_snapshot": snapshot,
            "candidate_selection_evidence": binding, "selected": selected}
    version = declared_prompt_version(sample)
    if version is not None:
        result["visual_prompt_version"] = version
    return result


def visual_input_binding(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Pure ledger lookup payload, excluding manifest identity and all supervision.

    The returned digest is a lookup assertion, not an approval. Only the manifest
    hash supplied separately by the trusted caller establishes the baseline.
    """
    inputs = _visible(sample)
    snapshot = inputs["candidate_snapshot"]
    numbered = "numbering_version" in snapshot or snapshot.get("preselection_policy") == CONTEXT_POLICY
    raw_path = sample.get("raw_snapshot_path")
    raw_file_sha = sample.get("raw_snapshot_file_sha256")
    if (raw_path is None) != (raw_file_sha is None) or (numbered and raw_path is None):
        raise ValueError("numbered input requires a registered raw snapshot sidecar")
    version = declared_prompt_version(inputs)
    prompt = build_visual_user_prompt(inputs["instruction"], inputs["actor_step"], snapshot, visual_prompt_version=version)
    protocol = sample["input_image_protocol"]
    payload = {
        "input_image_protocol": protocol, "original_image_path": inputs["image_path"],
        "native_width": inputs["image_width"], "native_height": inputs["image_height"],
        "original_image_file_sha256": _sha(sample["image_file_sha256"], "original file"),
        "original_native_rgb_sha256": _sha(sample["image_sha256"], "original RGB"),
        "candidate_snapshot_sha256": snapshot_hash(snapshot),
        "candidate_selection_evidence": inputs["candidate_selection_evidence"],
        "context_sha256": inputs["candidate_selection_evidence"]["context_sha256"],
        "raw_protection_snapshot_sha256": _sha(snapshot["parent_raw_snapshot_sha256"] if numbered else snapshot_hash(snapshot), "raw snapshot"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model_image_path": _path(sample["model_image_path"]),
        "model_png_sha256": _sha(sample["model_png_sha256"], "model PNG"),
        "model_native_rgb_sha256": _sha(sample["model_native_rgb_sha256"], "model RGB"),
        "visual_metadata_path": _path(sample["visual_metadata_path"]),
        "visual_metadata_file_sha256": _sha(sample["visual_metadata_file_sha256"], "metadata file"),
        "raw_snapshot_path": _path(raw_path) if raw_path is not None else None,
        "raw_snapshot_file_sha256": _sha(raw_file_sha, "raw sidecar") if raw_file_sha is not None else None,
    }
    if version is not None:
        payload["visual_prompt_version"] = version
    if protocol == GALLERY_PROTOCOL:
        from .candidate_gallery import CONFIG, resolve_gallery_geometry

        geometry = resolve_gallery_geometry(inputs["image_width"])
        payload["canvas_width"] = inputs["image_width"]
        payload["canvas_height"] = inputs["image_height"] + (
            (len(inputs["selected"]) + CONFIG["columns"] - 1) // CONFIG["columns"] * geometry["tile_height"]
        )
    return {**payload, "input_binding_sha256": value_sha256(payload)}


@dataclass(frozen=True)
class VisualAssetRegistry:
    """Approval baseline supplied by the caller, never constructed from a row."""
    manifest_path: str | Path
    manifest_sha256: str
    original_root: str | Path
    asset_root: str | Path


@dataclass(frozen=True)
class VisualPointerInput:
    width: int
    height: int
    rgb_bytes: bytes
    prompt: str
    original_image_path: str
    original_image_file_sha256: str
    original_native_rgb_sha256: str
    model_image_path: str
    model_png_sha256: str
    model_native_rgb_sha256: str
    input_binding_sha256: str
    evidence_json: str
    _recheck: Callable[[], None] = field(repr=False, compare=False)
    visual_prompt_version: str | None = None
    input_image_protocol: str = PROTOCOL
    model_width: int | None = None
    model_height: int | None = None

    @property
    def model_size(self) -> tuple[int, int]:
        if self.model_width is None and self.model_height is None:
            return self.width, self.height
        if any(type(value) is not int or value <= 0 for value in (self.model_width, self.model_height)):
            raise ValueError("visual model dimensions must be a complete positive pair")
        return self.model_width, self.model_height

    def assert_current(self) -> None:
        self._recheck()

    def to_pil(self) -> Image.Image:
        """A fresh RGB copy; consumers can use the same immutable raster twice."""
        self.assert_current()
        return Image.frombytes("RGB", self.model_size, self.rgb_bytes)


def _json(data: bytes, label: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}")
            result[key] = value
        return result
    result = json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))
    if not isinstance(result, dict):
        raise TypeError(f"{label} must be a JSON object")
    return result


def _canonical(path: str | Path, root: Path | None = None) -> Path:
    lexical = Path(_path(path))
    resolved = lexical.resolve(strict=True)
    if lexical != resolved:
        raise ValueError("visual asset path alias or replacement refused")
    if root is not None and (not resolved.is_relative_to(root) or resolved == root):
        raise ValueError("visual asset escaped its caller-registered root")
    return resolved


def _bound_bytes(path: Path, sha: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("registered visual asset SHA256 differs")
    return data


def _image(data: bytes, dimensions: tuple[int, int], *, model: bool) -> tuple[Image.Image, str]:
    with Image.open(io.BytesIO(data)) as image:
        if image.size != dimensions or getattr(image, "n_frames", 1) != 1 or (model and image.format != "PNG"):
            raise ValueError("visual raster native dimensions, frame count or PNG format differ")
        return image.convert("RGB"), _native_hash(image.convert("RGBA"))


def _box(value: Any, dimensions: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, Mapping) or set(value) != {"left", "top", "right", "bottom"}:
        raise ValueError("visual table bounding box differs")
    box = tuple(value[k] for k in ("left", "top", "right", "bottom"))
    if any(type(v) is not int for v in box) or not (0 <= box[0] < box[2] <= dimensions[0] and 0 <= box[1] < box[3] <= dimensions[1]):
        raise ValueError("visual table bounding box outside native canvas")
    return box


def _paint(metadata: dict, inputs: dict, raw: dict, original: Image.Image, model: Image.Image) -> None:
    cfg = _config(metadata["config"])
    table = metadata["id_table"]
    selected = inputs["selected"]
    if not isinstance(table, list) or len(table) != len(selected) or type(metadata["selected_candidate_count"]) is not int or metadata["selected_candidate_count"] != len(selected):
        raise ValueError("visual table does not preserve every selected ID")
    masks = {k: Image.new("L", original.size, 0) for k in ("badge", "leader", "ocr", "centre")}
    boxes = []
    for identifier, (row, candidate) in enumerate(zip(table, selected, strict=True), 1):
        point = center(candidate["bbox"])
        native_center = row["native_center"]
        _box(row["bbox"], original.size)
        if (type(row["id"]) is not int or row["id"] != identifier or row["candidate_id"] != candidate["candidate_id"]
                or row["raw_candidate_id"] != candidate.get("original_id", candidate["candidate_id"])
                or row["bbox"] != candidate["bbox"] or not isinstance(native_center, list)
                or any(type(v) is not int for v in native_center) or native_center != list(point)):
            raise ValueError("visual table ID/raw geometry/native center differs")
        box = _box(row["badge_bbox"], original.size)
        line = row["leader"]
        if (not isinstance(line, list) or len(line) != 2 or any(not isinstance(p, list) or len(p) != 2
                or any(type(v) is not int for v in p) or not (0 <= p[0] < original.width and 0 <= p[1] < original.height) for p in line)
                or line != [list(p) for p in _leader(box, point)]):
            raise ValueError("visual leader does not end at its original native center")
        glyph = row["glyph_bbox"]
        if not isinstance(glyph, list) or len(glyph) != 4 or any(type(v) is not int for v in glyph) or glyph[2] <= glyph[0] or glyph[3] <= glyph[1] or (
                box[2] - box[0], box[3] - box[1]) != (glyph[2] - glyph[0] + 2 * cfg.badge_padding, glyph[3] - glyph[1] + 2 * cfg.badge_padding):
            raise ValueError("visual numeric glyph/badge geometry differs")
        index = row["static_position_index"]
        if type(index) is not int or not 0 <= index < 264 or type(row["static_position_rank"]) is not int or row["static_position_rank"] != index + 1:
            raise ValueError("visual static position index differs")
        positions = list(_positions(*point, box[2] - box[0], box[3] - box[1], cfg))
        if index >= len(positions) or positions[index] != (box[0], box[1], row["search"]):
            raise ValueError("visual badge does not match its fixed static search position")
        for a in boxes:
            if a[0] - cfg.label_gap < box[2] and box[0] < a[2] + cfg.label_gap and a[1] - cfg.label_gap < box[3] and box[1] < a[3] + cfg.label_gap:
                raise ValueError("visual badge gap violated")
        boxes.append(box)
        _rect(ImageDraw.Draw(masks["badge"]), box, 255)
        ImageDraw.Draw(masks["leader"]).line([tuple(p) for p in line], fill=255, width=1)
        masks["centre"].putpixel(point, 255)
    ocr = [c for c in raw["candidates"] if c["source"] == "ocr"]
    for candidate in ocr:
        _rect(ImageDraw.Draw(masks["ocr"]), _box(candidate["bbox"], original.size), 255)
    paint = ImageChops.lighter(masks["badge"], masks["leader"])
    diff = ImageChops.difference(original, model)
    changed = ImageChops.lighter(ImageChops.lighter(*diff.split()[:2]), diff.split()[2]).point([0] + [255] * 255)
    if ImageChops.subtract(changed, paint).getbbox() is not None or changed.getbbox() is None:
        raise ValueError("model raster changed outside annotations or equals original")
    counts = {"canvas_pixels": original.width * original.height, "painted_pixels": _pixels(paint),
              "changed_pixels": _pixels(changed), "badge_pixels": _pixels(masks["badge"]),
              "leader_pixels": _pixels(masks["leader"]), "raw_ocr_candidate_count": len(ocr),
              "raw_ocr_union_pixels": _pixels(masks["ocr"]),
              "raw_ocr_painted_pixels": _pixels(ImageChops.multiply(paint, masks["ocr"])),
              "raw_ocr_changed_pixels": _pixels(ImageChops.multiply(changed, masks["ocr"])),
              "badge_raw_ocr_overlap_pixels": _pixels(ImageChops.multiply(masks["badge"], masks["ocr"])),
              "badge_leader_overlap_pixels": _pixels(ImageChops.multiply(masks["badge"], masks["leader"])),
              "badge_candidate_center_overlap_pixels": _pixels(ImageChops.multiply(masks["badge"], masks["centre"]))}
    if any(counts[k] for k in ("badge_raw_ocr_overlap_pixels", "badge_leader_overlap_pixels", "badge_candidate_center_overlap_pixels")):
        raise ValueError("visual badges overlap protected OCR/leader/center")
    if counts["painted_pixels"] > counts["canvas_pixels"] * cfg.max_paint_fraction or counts["raw_ocr_painted_pixels"] > counts["raw_ocr_union_pixels"] * cfg.max_raw_ocr_paint_fraction:
        raise ValueError("visual raster paint budget violated")
    counts["paint_fraction"] = counts["painted_pixels"] / counts["canvas_pixels"]
    counts["raw_ocr_paint_fraction"] = counts["raw_ocr_painted_pixels"] / counts["raw_ocr_union_pixels"] if counts["raw_ocr_union_pixels"] else 0.0
    recorded = metadata["paint"]
    if not isinstance(recorded, Mapping) or set(recorded) != set(counts) or any(
            type(recorded[k]) is not type(v) or recorded[k] != v for k, v in counts.items()):
        raise ValueError("visual raster differs from recorded full-mask counts")


def validate_visual_input(sample: Mapping[str, Any], *, registry: VisualAssetRegistry,
                          raw_snapshot: Mapping[str, Any] | None = None) -> VisualPointerInput:
    """Resolve a registered input raster or fail; never silently use image_path."""
    if sample.get("input_image_protocol") == GALLERY_PROTOCOL:
        return _validate_gallery_input(sample, registry=registry, raw_snapshot=raw_snapshot)
    if not isinstance(registry, VisualAssetRegistry):
        raise TypeError("trusted caller VisualAssetRegistry required")
    original_root = _canonical(registry.original_root)
    asset_root = _canonical(registry.asset_root)
    if not original_root.is_dir() or not asset_root.is_dir():
        raise ValueError("caller-registered visual roots must be directories")
    manifest_path = _canonical(registry.manifest_path, asset_root)
    manifest_sha = _sha(registry.manifest_sha256, "caller manifest")
    manifest = _json(_bound_bytes(manifest_path, manifest_sha), "approved asset manifest")
    version = declared_prompt_version(manifest)
    expected_manifest_keys = {"schema", "status", "input_image_protocol", "entries"}
    if version is not None:
        expected_manifest_keys.add("visual_prompt_version")
    if set(manifest) != expected_manifest_keys or (
            manifest["schema"], manifest["status"], manifest["input_image_protocol"]) != (MANIFEST_SCHEMA, "approved", PROTOCOL):
        raise ValueError("approved visual asset manifest contract differs")
    if declared_prompt_version(sample) != version:
        raise ValueError("row and caller-approved manifest visual_prompt_version differ")
    binding = visual_input_binding(sample)
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise TypeError("asset manifest entries must be a list")
    ledger = {}
    expected_keys = set(binding) | {"font_sha256", "config_sha256"}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ValueError("asset manifest entry contract differs")
        if declared_prompt_version(entry) != version:
            raise ValueError("asset manifest contains mixed visual prompt versions")
        key = _sha(entry["input_binding_sha256"], "entry binding")
        payload = {k: v for k, v in entry.items() if k not in {"input_binding_sha256", "font_sha256", "config_sha256"}}
        if key != value_sha256(payload) or key in ledger:
            raise ValueError("asset manifest binding is inconsistent or duplicated")
        _sha(entry["font_sha256"], "registered font")
        _sha(entry["config_sha256"], "registered config")
        ledger[key] = entry
    entry = ledger.get(binding["input_binding_sha256"])
    if entry is None or any(entry[k] != v for k, v in binding.items()):
        raise ValueError("visible input is not in the caller-approved asset ledger")
    inputs = _visible(sample)
    paths = {"original": _canonical(entry["original_image_path"], original_root),
             "model": _canonical(entry["model_image_path"], asset_root),
             "metadata": _canonical(entry["visual_metadata_path"], asset_root)}
    digests = {"original": entry["original_image_file_sha256"], "model": entry["model_png_sha256"],
               "metadata": entry["visual_metadata_file_sha256"]}
    if paths["original"] == paths["model"] or paths["original"].samefile(paths["model"]):
        raise ValueError("annotated model PNG must be distinct from the original native file")
    if entry["raw_snapshot_path"] is not None:
        paths["raw"] = _canonical(entry["raw_snapshot_path"], asset_root)
        digests["raw"] = entry["raw_snapshot_file_sha256"]
    if len(set(paths.values())) != len(paths):
        raise ValueError("visual asset roles must use distinct paths")
    data = {k: _bound_bytes(path, digests[k]) for k, path in paths.items()}
    metadata = _json(data["metadata"], "visual metadata")
    if declared_prompt_version(metadata) != version:
        raise ValueError("metadata and caller-approved visual_prompt_version differ")
    if version is not None:
        if metadata.get("prompt_derivation") != "metadata_only_rebind":
            raise ValueError("new visual prompt requires explicit metadata-only lineage")
        parent_path = _canonical(metadata["prompt_parent_metadata_path"], asset_root)
        if parent_path in paths.values():
            raise ValueError("prompt parent metadata must be a distinct immutable asset")
        parent_sha = _sha(metadata["prompt_parent_metadata_file_sha256"], "parent visual metadata")
        parent_data = _bound_bytes(parent_path, parent_sha)
        parent_metadata = _json(parent_data, "legacy parent visual metadata")
        additions = {"visual_prompt_version", "prompt_derivation", "prompt_parent_metadata_path", "prompt_parent_metadata_file_sha256"}
        if declared_prompt_version(parent_metadata) is not None or set(metadata) != set(parent_metadata) | additions or any(
                metadata[key] != value for key, value in parent_metadata.items() if key != "prompt_sha256"):
            raise ValueError("prompt-only metadata changed parent raster/geometry/evidence")
        paths["prompt_parent"] = parent_path
        digests["prompt_parent"] = parent_sha
    loaded_raw = _json(data["raw"], "input-only raw snapshot") if "raw" in data else None
    if (loaded_raw is not None and FORBIDDEN_SNAPSHOT_KEYS.intersection(loaded_raw)) or (
            isinstance(raw_snapshot, Mapping) and FORBIDDEN_SNAPSHOT_KEYS.intersection(raw_snapshot)):
        raise ValueError("raw sidecar must contain input-only snapshot evidence")
    if loaded_raw is not None and snapshot_hash(loaded_raw) != entry["raw_protection_snapshot_sha256"]:
        raise ValueError("registered raw snapshot hash differs")
    if raw_snapshot is not None and (not isinstance(raw_snapshot, Mapping) or snapshot_hash(dict(raw_snapshot)) != entry["raw_protection_snapshot_sha256"]):
        raise ValueError("explicit raw snapshot differs from approved ledger")
    raw = _raw_protection(inputs, raw_snapshot if raw_snapshot is not None else loaded_raw)
    if snapshot_hash(raw) != entry["raw_protection_snapshot_sha256"]:
        raise ValueError("raw protection differs from approved ledger")
    validate_snapshot(inputs["candidate_snapshot"], paths["original"])
    dimensions = (inputs["image_width"], inputs["image_height"])
    original, rgba_sha = _image(data["original"], dimensions, model=False)
    model, _ = _image(data["model"], dimensions, model=True)
    if _native_hash(original) != entry["original_native_rgb_sha256"] or _native_hash(model) != entry["model_native_rgb_sha256"]:
        raise ValueError("native original/model RGB identity differs")
    if entry["original_image_file_sha256"] == entry["model_png_sha256"] or entry["original_native_rgb_sha256"] == entry["model_native_rgb_sha256"]:
        raise ValueError("model PNG cannot replace its annotation with the original raster")
    cfg = _config(metadata["config"])
    checks = {"version": VERSION, "layout_policy": LAYOUT_POLICY, "layout_status": "all_ids_rendered",
              "native_width": dimensions[0], "native_height": dimensions[1],
              "original_image_path": str(paths["original"]), "original_native_rgba_sha256": rgba_sha,
              "model_png_bytes": len(data["model"]), "config_sha256": value_sha256(asdict(cfg)),
              "all_selected_ids_rendered": True, "native_coordinate_mapping_preserved": True}
    for k in ("original_image_file_sha256", "original_native_rgb_sha256", "model_png_sha256", "model_native_rgb_sha256",
              "candidate_snapshot_sha256", "candidate_selection_evidence", "context_sha256", "prompt_sha256",
              "raw_protection_snapshot_sha256", "font_sha256", "config_sha256"):
        checks[k] = entry[k]
    if value_sha256(asdict(cfg)) != entry["config_sha256"] or any(
            type(metadata.get(k)) is not type(v) or metadata.get(k) != v for k, v in checks.items()):
        raise ValueError("visual metadata disagrees with approved input identities")
    if metadata["style"] != {"badge": "black", "digits": "white", "leader": "cyan", "leader_width": 1}:
        raise ValueError("visual annotation style differs")
    _paint(metadata, inputs, raw, original, model)
    prompt = build_visual_user_prompt(inputs["instruction"], inputs["actor_step"], inputs["candidate_snapshot"],
                                      visual_prompt_version=version)

    def recheck() -> None:
        if (_canonical(registry.original_root) != original_root or _canonical(registry.asset_root) != asset_root
                or _canonical(registry.manifest_path, asset_root) != manifest_path):
            raise ValueError("caller-registered paths changed")
        _bound_bytes(manifest_path, manifest_sha)
        if visual_input_binding(sample) != binding:
            raise ValueError("visible input context/selection changed")
        if raw_snapshot is not None:
            if FORBIDDEN_SNAPSHOT_KEYS.intersection(raw_snapshot):
                raise ValueError("explicit raw snapshot must remain input-only")
            if snapshot_hash(dict(raw_snapshot)) != entry["raw_protection_snapshot_sha256"]:
                raise ValueError("explicit raw snapshot changed")
        for name, path in paths.items():
            root = original_root if name == "original" else asset_root
            if _canonical(path, root) != path:
                raise ValueError("registered visual asset path changed")
            _bound_bytes(path, digests[name])

    recheck()
    evidence = {**binding, "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                "font_sha256": entry["font_sha256"], "config_sha256": entry["config_sha256"]}
    return VisualPointerInput(*dimensions, model.tobytes(), prompt, str(paths["original"]),
                              entry["original_image_file_sha256"], entry["original_native_rgb_sha256"],
                              str(paths["model"]), entry["model_png_sha256"], entry["model_native_rgb_sha256"],
                              binding["input_binding_sha256"], json.dumps(evidence, sort_keys=True, ensure_ascii=False), recheck,
                              visual_prompt_version=version)


def _validate_gallery_input(sample: Mapping[str, Any], *, registry: VisualAssetRegistry,
                            raw_snapshot: Mapping[str, Any] | None) -> VisualPointerInput:
    """An external canvas is never admitted by the same-dimensions v2 branch."""
    from .candidate_gallery import CONFIG, compose_gallery_rgb
    from .visual_pointer_inference import _input_only

    if not isinstance(registry, VisualAssetRegistry):
        raise TypeError("trusted caller VisualAssetRegistry required")
    original_root, asset_root = _canonical(registry.original_root), _canonical(registry.asset_root)
    if not original_root.is_dir() or not asset_root.is_dir():
        raise ValueError("caller-registered visual roots must be directories")
    manifest_path = _canonical(registry.manifest_path, asset_root)
    manifest_sha = _sha(registry.manifest_sha256, "caller manifest")
    manifest = _json(_bound_bytes(manifest_path, manifest_sha), "gallery asset manifest")
    version = declared_prompt_version(manifest)
    if set(manifest) != {"schema", "status", "input_image_protocol", "visual_prompt_version", "entries"} or (
            manifest["schema"], manifest["status"], manifest["input_image_protocol"]) != (
            GALLERY_MANIFEST_SCHEMA, "approved", GALLERY_PROTOCOL) or version not in GALLERY_PROMPT_VERSIONS:
        raise ValueError("approved gallery manifest contract differs")
    if declared_prompt_version(sample) != version:
        raise ValueError("gallery row and caller-approved manifest prompt version differ")
    inputs = _visible(sample)
    _input_only(inputs["candidate_snapshot"])
    binding = visual_input_binding(sample)
    ledger = {}
    if not isinstance(manifest["entries"], list):
        raise TypeError("gallery asset entries must be a list")
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or set(entry) != set(binding) | {"font_sha256", "config_sha256"}:
            raise ValueError("gallery asset entry contract differs")
        payload = {k: v for k, v in entry.items() if k not in {"input_binding_sha256", "font_sha256", "config_sha256"}}
        key = _sha(entry["input_binding_sha256"], "gallery entry")
        if key != value_sha256(payload) or key in ledger or entry["input_image_protocol"] != GALLERY_PROTOCOL or declared_prompt_version(entry) != version:
            raise ValueError("gallery asset binding is inconsistent, mixed or duplicated")
        _sha(entry["font_sha256"], "gallery font")
        if entry["config_sha256"] != value_sha256(CONFIG):
            raise ValueError("gallery layout config differs")
        ledger[key] = entry
    entry = ledger.get(binding["input_binding_sha256"])
    if entry is None or any(entry[k] != v for k, v in binding.items()):
        raise ValueError("gallery input not in caller-approved ledger")
    paths = {"original": _canonical(entry["original_image_path"], original_root),
             "model": _canonical(entry["model_image_path"], asset_root),
             "metadata": _canonical(entry["visual_metadata_path"], asset_root)}
    digests = {"original": entry["original_image_file_sha256"], "model": entry["model_png_sha256"],
               "metadata": entry["visual_metadata_file_sha256"]}
    if entry["raw_snapshot_path"] is not None:
        paths["raw"] = _canonical(entry["raw_snapshot_path"], asset_root)
        digests["raw"] = entry["raw_snapshot_file_sha256"]
    if len(set(paths.values())) != len(paths) or paths["original"].samefile(paths["model"]):
        raise ValueError("gallery asset roles must be distinct")
    data = {key: _bound_bytes(path, digests[key]) for key, path in paths.items()}
    metadata = _json(data["metadata"], "gallery metadata")
    raw = _json(data["raw"], "gallery raw snapshot") if "raw" in data else raw_snapshot
    if raw is not None:
        _input_only(raw)
        if snapshot_hash(raw) != entry["raw_protection_snapshot_sha256"]:
            raise ValueError("gallery raw sidecar differs")
    if raw_snapshot is not None:
        _input_only(raw_snapshot)
        if snapshot_hash(raw_snapshot) != entry["raw_protection_snapshot_sha256"]:
            raise ValueError("explicit gallery raw snapshot differs")
    raw = _raw_protection(inputs, raw)
    if snapshot_hash(raw) != entry["raw_protection_snapshot_sha256"]:
        raise ValueError("gallery parent raw protection differs")
    validate_snapshot(inputs["candidate_snapshot"], paths["original"])
    native_size = (inputs["image_width"], inputs["image_height"])
    canvas_size = (binding["canvas_width"], binding["canvas_height"])
    original, _ = _image(data["original"], native_size, model=False)
    model, _ = _image(data["model"], canvas_size, model=True)
    if (_native_hash(original), _native_hash(model)) != (entry["original_native_rgb_sha256"], entry["model_native_rgb_sha256"]):
        raise ValueError("gallery original/canvas RGB identity differs")
    font_path = _canonical(metadata["font_path"])
    font_bytes = _bound_bytes(font_path, entry["font_sha256"])
    if font_path in paths.values():
        raise ValueError("gallery font must have a distinct asset role")
    composed = compose_gallery_rgb(original.tobytes(), native_size, inputs["selected"],
                                   font_bytes=font_bytes, font_sha256=entry["font_sha256"])
    prompt = build_visual_user_prompt(inputs["instruction"], inputs["actor_step"], inputs["candidate_snapshot"],
                                      visual_prompt_version=version)
    expected = {**composed["metadata"], "status": "rendered_not_training_admission",
                "original_image_path": str(paths["original"]), "original_image_file_sha256": entry["original_image_file_sha256"],
                "original_native_rgb_sha256": entry["original_native_rgb_sha256"], "font_path": str(font_path),
                "gallery_png_sha256": entry["model_png_sha256"], "gallery_png_bytes": len(data["model"]),
                "context_sha256": entry["context_sha256"], "candidate_snapshot_sha256": entry["candidate_snapshot_sha256"],
                "raw_snapshot_sha256": entry["raw_protection_snapshot_sha256"],
                "candidate_selection_evidence": entry["candidate_selection_evidence"],
                "input_image_protocol": GALLERY_PROTOCOL, "visual_prompt_version": version,
                "prompt_sha256": entry["prompt_sha256"], "model_png_sha256": entry["model_png_sha256"],
                "model_native_rgb_sha256": entry["model_native_rgb_sha256"], "model_png_bytes": len(data["model"])}
    if (tuple(composed["size"]) != canvas_size or composed["rgb_bytes"] != model.tobytes()
            or value_sha256(metadata) != value_sha256(expected)
            or hashlib.sha256(prompt.encode("utf8")).hexdigest() != entry["prompt_sha256"]):
        raise ValueError("gallery pixels/overview/tiles/metadata/prompt differ from deterministic input-only codec")
    paths["font"], digests["font"] = font_path, entry["font_sha256"]
    def recheck() -> None:
        if (_canonical(registry.original_root) != original_root or _canonical(registry.asset_root) != asset_root
                or _canonical(registry.manifest_path, asset_root) != manifest_path):
            raise ValueError("gallery caller paths changed")
        _bound_bytes(manifest_path, manifest_sha)
        if visual_input_binding(sample) != binding:
            raise ValueError("gallery context/selection changed")
        if raw_snapshot is not None:
            _input_only(raw_snapshot)
            if snapshot_hash(raw_snapshot) != entry["raw_protection_snapshot_sha256"]:
                raise ValueError("explicit gallery raw changed")
        for name, path in paths.items():
            root = original_root if name == "original" else None if name == "font" else asset_root
            if _canonical(path, root) != path:
                raise ValueError("gallery asset path changed")
            _bound_bytes(path, digests[name])
    recheck()
    evidence = {**binding, "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                "font_sha256": entry["font_sha256"], "config_sha256": entry["config_sha256"]}
    return VisualPointerInput(*native_size, model.tobytes(), prompt, str(paths["original"]),
                              entry["original_image_file_sha256"], entry["original_native_rgb_sha256"],
                              str(paths["model"]), entry["model_png_sha256"], entry["model_native_rgb_sha256"],
                              binding["input_binding_sha256"], json.dumps(evidence, sort_keys=True, ensure_ascii=False), recheck,
                              visual_prompt_version=version, input_image_protocol=GALLERY_PROTOCOL,
                              model_width=canvas_size[0], model_height=canvas_size[1])
