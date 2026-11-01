"""Synthetic registered raster guards; no actual rows, model or processor use."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import test_week5_visual_pointer as fixtures
from PIL import Image

from gui_agent.finetuning import visual_pointer_input as vi
from gui_agent.finetuning import visual_pointer_v2 as v2
from gui_agent.finetuning.candidate_grounding import snapshot_hash
from gui_agent.finetuning.evaluation import file_sha256, value_sha256
from gui_agent.finetuning.pointer_numbering import numbered_view
from gui_agent.finetuning.visual_pointer_prompts import (
    DEDUP_PROMPT_VERSION,
    build_visual_user_prompt,
    declared_prompt_version,
)

font_asset = fixtures.font_asset


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


class Asset:
    def __init__(self, tmp_path, font_asset, *, policy="query_relations_v3", numbered=False):
        self.original_root = tmp_path / "native"
        self.asset_root = tmp_path / "assets"
        self.original_root.mkdir()
        self.asset_root.mkdir()
        self.row = fixtures.make_sample(self.original_root, policy=policy, candidates=[
            fixtures.candidate("native-A", 100, 120), fixtures.candidate("native-B", 380, 280)])
        self.raw = copy.deepcopy(self.row["candidate_snapshot"])
        if numbered:
            self.row["candidate_snapshot"] = numbered_view(self.row, view_index=1)["candidate_snapshot"]
            fixtures.bind(self.row)
        result = v2.render_visual_pointer_v2(self.row, **font_asset[0], raw_snapshot=self.raw if numbered else None)
        self.metadata = result["metadata"]
        self.prompt = result["prompt"]
        self.model = self.asset_root / "model.png"
        self.sidecar = self.asset_root / "metadata.json"
        self.manifest = self.asset_root / "manifest.json"
        self.model.write_bytes(result["png_bytes"])
        self.row.update(input_image_protocol=vi.PROTOCOL, model_image_path=str(self.model),
                        visual_metadata_path=str(self.sidecar))
        if numbered:
            self.raw_path = self.asset_root / "raw_snapshot.json"
            write_json(self.raw_path, self.raw)
            self.row.update(raw_snapshot_path=str(self.raw_path), raw_snapshot_file_sha256=file_sha256(self.raw_path))
        self.register()

    def register(self):
        """Only synthetic caller re-approval, to exercise checks beyond SHA guards."""
        write_json(self.sidecar, self.metadata)
        self.row.update(model_png_sha256=file_sha256(self.model),
                        model_native_rgb_sha256=self.metadata["model_native_rgb_sha256"],
                        visual_metadata_file_sha256=file_sha256(self.sidecar))
        binding = vi.visual_input_binding(self.row)
        self.entry = {**binding, "font_sha256": self.metadata["font_sha256"],
                      "config_sha256": self.metadata["config_sha256"]}
        manifest = {"schema": vi.MANIFEST_SCHEMA, "status": "approved",
                    "input_image_protocol": vi.PROTOCOL, "entries": [self.entry]}
        version = declared_prompt_version(self.row)
        if version is not None:
            manifest["visual_prompt_version"] = version
        write_json(self.manifest, manifest)
        self.registry = vi.VisualAssetRegistry(self.manifest, file_sha256(self.manifest),
                                               self.original_root, self.asset_root)

    def validate(self, **kwargs):
        return vi.validate_visual_input(self.row, registry=self.registry, **kwargs)


def dedup_asset(asset):
    """Rebind synthetic metadata while retaining every byte of its existing PNG."""
    parent_path = asset.sidecar
    parent_sha = file_sha256(parent_path)
    asset.metadata = {**copy.deepcopy(asset.metadata), "visual_prompt_version": DEDUP_PROMPT_VERSION,
                      "prompt_derivation": "metadata_only_rebind", "prompt_parent_metadata_path": str(parent_path),
                      "prompt_parent_metadata_file_sha256": parent_sha}
    asset.prompt = build_visual_user_prompt(asset.row["instruction"], asset.row["actor_step"], asset.row["candidate_snapshot"],
                                           visual_prompt_version=DEDUP_PROMPT_VERSION)
    asset.metadata["prompt_sha256"] = hashlib.sha256(asset.prompt.encode("utf8")).hexdigest()
    asset.sidecar = asset.asset_root / "dedup_metadata.json"
    asset.manifest = asset.asset_root / "dedup_manifest.json"
    asset.row.update(visual_prompt_version=DEDUP_PROMPT_VERSION, visual_metadata_path=str(asset.sidecar))
    asset.register()
    return asset


def test_dedup_rebind_reuses_png_and_whole_input_evidence(asset):
    old_row, old_png = copy.deepcopy(asset.row), asset.model.read_bytes()
    legacy = asset.validate()
    dedup_asset(asset)
    result = asset.validate()
    assert result.visual_prompt_version == DEDUP_PROMPT_VERSION
    assert result.prompt == asset.prompt and result.prompt != legacy.prompt
    assert result.rgb_bytes == legacy.rgb_bytes and asset.model.read_bytes() == old_png
    for name in ("instruction", "actor_step", "candidate_snapshot", "candidate_selection_evidence"):
        assert asset.row[name] == old_row[name]
    assert json.loads(result.evidence_json)["visual_prompt_version"] == DEDUP_PROMPT_VERSION
    result.assert_current()


@pytest.mark.parametrize("scope,value", [("row", None), ("row", ""), ("row", "unknown"), ("row", "missing"),
                                         ("manifest", None), ("manifest", "unknown"), ("manifest", "missing"),
                                         ("entry", "unknown"), ("entry", "missing"),
                                         ("metadata", None), ("metadata", "unknown"), ("metadata", "missing")])
def test_approved_rebind_rejects_unknown_missing_or_mixed_version(asset, scope, value):
    dedup_asset(asset)
    if scope == "row":
        changed = asset.row
    elif scope == "metadata":
        changed = asset.metadata
    else:
        manifest = json.loads(asset.manifest.read_text(encoding="utf8"))
        changed = manifest if scope == "manifest" else manifest["entries"][0]
    if value == "missing":
        changed.pop("visual_prompt_version")
    else:
        changed["visual_prompt_version"] = value
    if scope in {"manifest", "entry"}:
        write_json(asset.manifest, manifest)
        asset.registry = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), asset.original_root, asset.asset_root)
    elif scope == "metadata":
        asset.register()  # caller approves altered bytes; contract still rejects
    with pytest.raises((ValueError, KeyError)):
        asset.validate()


@pytest.mark.parametrize("change", ["geometry", "font", "paint", "missing_lineage", "parent_changed"])
def test_new_prompt_cannot_rewrite_parent_raster_provenance(asset, change):
    dedup_asset(asset)
    if change == "geometry":
        asset.metadata["id_table"][0]["native_center"][0] += 1
    elif change == "font":
        asset.metadata["font_sha256"] = "0" * 64
    elif change == "paint":
        asset.metadata["paint"]["changed_pixels"] += 1
    elif change == "missing_lineage":
        asset.metadata.pop("prompt_parent_metadata_path")
    else:
        Path(asset.metadata["prompt_parent_metadata_path"]).write_bytes(b"changed approved parent")
    asset.register()
    with pytest.raises((ValueError, KeyError)):
        asset.validate()


def test_descriptor_rechecks_prompt_parent_after_request_validation(asset):
    dedup_asset(asset)
    result = asset.validate()
    Path(asset.metadata["prompt_parent_metadata_path"]).write_bytes(b"parent changed during worker request")
    with pytest.raises(ValueError):
        result.assert_current()


@pytest.mark.parametrize("value", [None, "unknown"])
def test_unused_approved_manifest_entry_cannot_mix_prompt_versions(asset, value):
    dedup_asset(asset)
    manifest = json.loads(asset.manifest.read_text(encoding="utf8"))
    other = copy.deepcopy(manifest["entries"][0])
    other["visual_prompt_version"] = value
    payload = {key: item for key, item in other.items() if key not in {"input_binding_sha256", "font_sha256", "config_sha256"}}
    other["input_binding_sha256"] = value_sha256(payload)
    manifest["entries"].append(other)
    write_json(asset.manifest, manifest)
    asset.registry = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), asset.original_root, asset.asset_root)
    with pytest.raises(ValueError):
        asset.validate()


@pytest.fixture
def asset(tmp_path, font_asset):
    return Asset(tmp_path, font_asset)


def test_registered_model_raster_is_distinct_native_size_immutable_and_copied(asset):
    native = Path(asset.row["image_path"]).read_bytes()
    result = asset.validate()
    assert result.width == 1024 and result.height == 768
    assert type(result.rgb_bytes) is bytes and len(result.rgb_bytes) == 1024 * 768 * 3
    assert result.prompt == asset.prompt
    assert result.original_image_file_sha256 == asset.row["image_file_sha256"]
    assert result.model_png_sha256 != result.original_image_file_sha256
    assert result.model_native_rgb_sha256 != result.original_native_rgb_sha256
    with Image.open(asset.model) as actual:
        assert result.rgb_bytes == actual.convert("RGB").tobytes()
    first, second = result.to_pil(), result.to_pil()
    first.putpixel((0, 0), (255, 0, 0))
    assert second.tobytes() == result.rgb_bytes and first.tobytes() != result.rgb_bytes
    result.assert_current()
    assert Path(asset.row["image_path"]).read_bytes() == native
    assert json.loads(result.evidence_json)["manifest_sha256"] == asset.registry.manifest_sha256


def test_visible_only_binding_validation_and_no_manifest_hash_cycle(asset, monkeypatch):
    row = fixtures.VisibleOnly(copy.deepcopy(asset.row))
    row.update(sample_id=object(), actions=object(), targets=object(), review=object(),
               candidate_target=object(), candidate_label_evidence=object(),
               visual_asset_manifest_sha256=object(), visual_asset_manifest_path=object())
    binding = vi.visual_input_binding(row)
    assert binding == vi.visual_input_binding(asset.row)
    assert not any("manifest" in key for key in binding)
    result = vi.validate_visual_input(row, registry=asset.registry)
    assert result.rgb_bytes == asset.validate().rgb_bytes
    # Pure factory lookup has no filesystem dependency, even if all paths vanish.
    monkeypatch.setattr(Path, "resolve", lambda *_args, **_kwargs: pytest.fail("pure binding accessed filesystem"))
    assert vi.visual_input_binding(row) == binding


def test_row_cannot_supply_its_own_trusted_registry(asset):
    with pytest.raises(TypeError, match="trusted caller"):
        vi.validate_visual_input(asset.row, registry={"manifest_sha256": file_sha256(asset.manifest)})
    asset.manifest.write_text("{}", encoding="utf-8")
    asset.row["visual_asset_manifest_sha256"] = file_sha256(asset.manifest)
    with pytest.raises(ValueError, match="SHA256"):
        asset.validate()


@pytest.mark.parametrize("field,value", [
    ("instruction", "A different visible goal"), ("actor_step", "Another subtask"),
    ("image_sha256", "0" * 64), ("model_png_sha256", "0" * 64),
    ("model_native_rgb_sha256", "0" * 64), ("visual_metadata_file_sha256", "0" * 64),
    ("input_image_protocol", "native"), ("image_width", True),
])
def test_unapproved_visible_changes_fail_closed(asset, field, value):
    asset.row[field] = value
    with pytest.raises((ValueError, KeyError)):
        asset.validate()


@pytest.mark.parametrize("role", ["model", "sidecar", "manifest", "original"])
def test_bound_file_mutation_rejected_initially_and_after_descriptor(asset, role):
    descriptor = asset.validate()
    path = Path(asset.row["image_path"]) if role == "original" else getattr(asset, role)
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="SHA256"):
        descriptor.assert_current()
    with pytest.raises(ValueError, match="SHA256"):
        descriptor.to_pil()
    with pytest.raises(ValueError, match="SHA256"):
        asset.validate()


def test_descriptor_catches_context_and_snapshot_changes(asset):
    descriptor = asset.validate()
    asset.row["actor_step"] = "changed after validation"
    with pytest.raises(ValueError):
        descriptor.assert_current()
    asset.row["actor_step"] = "Click Alpha."
    asset.row["candidate_snapshot"]["candidates"][0]["bbox"]["left"] += 1
    fixtures.bind(asset.row)
    with pytest.raises(ValueError, match="context/selection changed"):
        descriptor.assert_current()


@pytest.mark.parametrize("policy", ["spatial_v1", "query_relations_v3"])
def test_every_selection_policy_requires_exact_frozen_binding(tmp_path, font_asset, policy):
    asset = Asset(tmp_path, font_asset, policy=policy)
    asset.validate()
    asset.row["candidate_selection_evidence"] = {}
    with pytest.raises(ValueError, match="all-policy"):
        asset.validate()


def test_numbered_parent_loaded_from_registered_input_only_sidecar(tmp_path, font_asset):
    asset = Asset(tmp_path, font_asset, numbered=True)
    result = asset.validate()
    assert result.rgb_bytes == asset.validate(raw_snapshot=asset.raw).rgb_bytes
    assert asset.entry["raw_protection_snapshot_sha256"] == snapshot_hash(asset.raw)
    descriptor = asset.validate(raw_snapshot=asset.raw)
    asset.raw["candidates"][0]["text"] = "mutated input OCR"
    with pytest.raises(ValueError, match="explicit raw snapshot"):
        asset.validate(raw_snapshot=asset.raw)
    with pytest.raises(ValueError, match="explicit raw snapshot changed"):
        descriptor.assert_current()


def test_numbered_missing_unbound_and_modified_raw_sidecar_refused(tmp_path, font_asset):
    asset = Asset(tmp_path, font_asset, numbered=True)
    row = copy.deepcopy(asset.row)
    row.pop("raw_snapshot_path")
    row.pop("raw_snapshot_file_sha256")
    with pytest.raises(ValueError, match="registered raw snapshot"):
        vi.validate_visual_input(row, registry=asset.registry, raw_snapshot=asset.raw)
    asset.raw_path.write_bytes(asset.raw_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        asset.validate()


def test_numbered_wrong_parent_hash_and_label_record_sidecar_refused(tmp_path, font_asset):
    asset = Asset(tmp_path, font_asset, numbered=True)
    asset.raw["actions"] = [{"secret": "not input geometry"}]
    write_json(asset.raw_path, asset.raw)
    asset.row["raw_snapshot_file_sha256"] = file_sha256(asset.raw_path)
    asset.row["candidate_snapshot"]["parent_raw_snapshot_sha256"] = snapshot_hash(asset.raw)
    fixtures.bind(asset.row)
    asset.register()
    with pytest.raises(ValueError, match="input-only"):
        asset.validate()


def test_raw_ocr_protection_includes_regions_absent_from_selected_table(asset):
    inputs = vi._visible(asset.row)
    raw = copy.deepcopy(asset.raw)
    badge = asset.metadata["id_table"][0]["badge_bbox"]
    raw["candidates"].append(fixtures.candidate("unselected-raw-OCR", badge["left"], badge["top"], source="ocr",
                                               width=badge["right"] - badge["left"], height=badge["bottom"] - badge["top"]))
    assert "unselected-raw-OCR" not in inputs["candidate_selection_evidence"]["selected_candidate_ids"]
    with Image.open(asset.row["image_path"]) as source, Image.open(asset.model) as model, pytest.raises(ValueError, match="protected OCR"):
        vi._paint(asset.metadata, inputs, raw, source.convert("RGB"), model.convert("RGB"))


def test_reapproved_stale_snapshot_is_refused_even_with_matching_ledger(asset):
    asset.row["candidate_snapshot"]["image_file_sha256"] = "0" * 64
    fixtures.bind(asset.row)
    asset.register()
    with pytest.raises(ValueError, match="identity differs|stale"):
        asset.validate()


@pytest.mark.parametrize("change,pattern", [
    (lambda m: m.update(version="week5.visual-pointer.v1"), "identities"),
    (lambda m: m.update(layout_status="pending"), "identities"),
    (lambda m: m.update(all_selected_ids_rendered=1), "identities"),
    (lambda m: m.update(prompt_sha256="0" * 64), "identities"),
    (lambda m: m.update(context_sha256="0" * 64), "identities"),
    (lambda m: m.update(original_native_rgba_sha256="0" * 64), "identities"),
    (lambda m: m["id_table"].pop(), "every selected ID"),
    (lambda m: m["id_table"][0].update(id=True), "ID/raw geometry"),
    (lambda m: m["id_table"][0].update(raw_candidate_id="wrong"), "ID/raw geometry"),
    (lambda m: m["id_table"][0].update(native_center=[0, 0]), "ID/raw geometry"),
    (lambda m: m["id_table"][0].update(leader=[[0, 0], [0, 0]]), "leader"),
    (lambda m: m["id_table"][0].update(static_position_rank=999), "static position"),
    (lambda m: m["id_table"][0].update(static_position_index=1, static_position_rank=2), "fixed static"),
    (lambda m: m["paint"].update(painted_pixels=m["paint"]["painted_pixels"] + 1), "full-mask"),
    (lambda m: m["paint"].update(raw_ocr_union_pixels=999), "full-mask"),
    (lambda m: m["config"].update(max_paint_fraction=.09), "cannot exceed"),
])
def test_reapproved_metadata_still_requires_structural_geometry_integrity(asset, change, pattern):
    change(asset.metadata)
    asset.register()
    with pytest.raises(ValueError, match=pattern):
        asset.validate()


@pytest.mark.parametrize("kind", ["dimensions", "frames", "format", "original", "outside_mask"])
def test_reapproved_model_asset_still_requires_native_annotated_png(asset, kind):
    with Image.open(asset.model) as existing:
        model = existing.convert("RGB")
    if kind == "dimensions":
        model = model.resize((64, 64))
        model.save(asset.model)
    elif kind == "frames":
        other = model.copy()
        other.putpixel((0, 0), (1, 2, 3))
        model.save(asset.model, save_all=True, append_images=[other], duration=1, loop=0)
    elif kind == "format":
        model.save(asset.model, format="BMP")
    elif kind == "original":
        asset.model.write_bytes(Path(asset.row["image_path"]).read_bytes())
        with Image.open(asset.model) as original:
            model = original.convert("RGB")
    else:
        model.putpixel((0, 0), (250, 250, 250))
        model.save(asset.model)
    asset.metadata.update(model_png_sha256=file_sha256(asset.model),
                          model_native_rgb_sha256=hashlib.sha256(f"{model.width}x{model.height}:".encode() + model.tobytes()).hexdigest(),
                          model_png_bytes=asset.model.stat().st_size)
    asset.register()
    with pytest.raises(ValueError, match="dimensions|frame count|PNG format|original raster|outside annotations"):
        asset.validate()


def test_absolute_roots_parent_traversal_alias_and_cross_role_rejected(asset, tmp_path):
    row = copy.deepcopy(asset.row)
    row["model_image_path"] = str(asset.asset_root / ".." / "assets" / "model.png")
    with pytest.raises(ValueError, match="parent traversal"):
        vi.visual_input_binding(row)
    wrong = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), tmp_path / "outside", asset.asset_root)
    (tmp_path / "outside").mkdir()
    with pytest.raises(ValueError, match="escaped"):
        vi.validate_visual_input(asset.row, registry=wrong)
    asset.row["model_image_path"] = asset.row["image_path"]
    asset.register()
    overlapping = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), asset.original_root, tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        vi.validate_visual_input(asset.row, registry=overlapping)


def test_hardlinked_model_cannot_alias_original_even_under_shared_root(asset, tmp_path):
    asset.model.unlink()
    os.link(asset.row["image_path"], asset.model)
    asset.register()
    registry = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), tmp_path, tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        vi.validate_visual_input(asset.row, registry=registry)


def test_path_resolution_replacement_caught_without_reading_new_asset(asset, tmp_path, monkeypatch):
    descriptor = asset.validate()
    other = tmp_path / "other.png"
    other.write_bytes(asset.model.read_bytes())
    resolve = Path.resolve
    # Platform-independent simulation of resolve changing after a link swap.
    # Hardlink identity is separately exercised with the actual filesystem.
    monkeypatch.setattr(Path, "resolve", lambda self, *args, **kwargs: other if self == asset.model else resolve(self, *args, **kwargs))
    with pytest.raises(ValueError, match="alias|replacement"):
        descriptor.assert_current()


@pytest.mark.parametrize("kind", ["pending", "duplicate", "incorrect_key", "extra_label", "font", "config"])
def test_caller_bound_manifest_still_requires_closed_ledger_schema(asset, kind):
    manifest = json.loads(asset.manifest.read_text(encoding="utf-8"))
    if kind == "pending":
        manifest["status"] = "pending"
    elif kind == "duplicate":
        manifest["entries"].append(copy.deepcopy(asset.entry))
    elif kind == "incorrect_key":
        manifest["entries"][0]["input_binding_sha256"] = "0" * 64
    elif kind == "extra_label":
        manifest["entries"][0]["actions"] = []
    else:
        manifest["entries"][0][f"{kind}_sha256"] = "0" * 64
    write_json(asset.manifest, manifest)
    registry = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), asset.original_root, asset.asset_root)
    with pytest.raises(ValueError):
        vi.validate_visual_input(asset.row, registry=registry)


def test_mutation_during_validation_is_detected_at_final_recheck(asset, monkeypatch):
    original = vi._paint
    def mutate(*args):
        original(*args)
        asset.model.write_bytes(asset.model.read_bytes() + b"changed during validation")
    monkeypatch.setattr(vi, "_paint", mutate)
    with pytest.raises(ValueError, match="SHA256"):
        asset.validate()


def test_binding_has_exact_payload_hash_and_excludes_labels_manifest(asset):
    binding = vi.visual_input_binding(asset.row)
    digest = binding.pop("input_binding_sha256")
    assert value_sha256(binding) == digest
    assert binding["candidate_snapshot_sha256"] == snapshot_hash(asset.row["candidate_snapshot"])
    assert set(binding).isdisjoint({"sample_id", "actions", "candidate_target", "review", "manifest_sha256", "manifest_path"})


class UnreadableSecretSnapshot(Mapping):
    def __init__(self, snapshot, key):
        self.snapshot, self.key = snapshot, key

    def __iter__(self):
        yield from self.snapshot
        yield self.key

    def __len__(self):
        return len(self.snapshot) + 1

    def __getitem__(self, key):
        if key == self.key:
            pytest.fail("input contract read an embedded supervision value")
        return self.snapshot[key]


@pytest.mark.parametrize("key", ["actions", "candidate_target", "review", "sample_id", "request", "plan"])
def test_embedded_forbidden_snapshot_keys_refused_before_secret_values_read(asset, key):
    row = copy.deepcopy(asset.row)
    row["candidate_snapshot"] = UnreadableSecretSnapshot(row["candidate_snapshot"], key)
    with pytest.raises(ValueError, match="input-only"):
        vi.visual_input_binding(row)
    with pytest.raises(ValueError, match="input-only"):
        vi.validate_visual_input(row, registry=asset.registry)
    raw = UnreadableSecretSnapshot(asset.raw, key)
    with pytest.raises(ValueError, match="input-only"):
        asset.validate(raw_snapshot=raw)


def test_registered_raw_sidecar_rejects_embedded_sample_id(tmp_path, font_asset):
    asset = Asset(tmp_path, font_asset, numbered=True)
    asset.raw["sample_id"] = "forbidden raw identity label"
    write_json(asset.raw_path, asset.raw)
    asset.row["raw_snapshot_file_sha256"] = file_sha256(asset.raw_path)
    asset.row["candidate_snapshot"]["parent_raw_snapshot_sha256"] = snapshot_hash(asset.raw)
    fixtures.bind(asset.row)
    asset.register()
    with pytest.raises(ValueError, match="input-only"):
        asset.validate()


def test_descriptor_recheck_refuses_new_raw_secret_before_serialization(asset):
    raw = copy.deepcopy(asset.raw)
    descriptor = asset.validate(raw_snapshot=raw)
    raw["actions"] = object()  # Serialization would raise TypeError if read.
    with pytest.raises(ValueError, match="remain input-only"):
        descriptor.assert_current()
