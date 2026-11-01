"""Synthetic gallery registry/encoder/worker transport, with no actual model."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_week5_visual_pointer as fixtures
import test_week5_visual_pointer_inference as worker_fixtures
import test_week5_visual_pointer_input as assets
import test_week5_visual_pointer_training as train_fixtures
from PIL import Image, ImageFont

from gui_agent.finetuning import candidate_gallery as gallery
from gui_agent.finetuning import evaluation as ev
from gui_agent.finetuning import training
from gui_agent.finetuning import visual_pointer_inference as inference
from gui_agent.finetuning import visual_pointer_input as vi
from gui_agent.finetuning.candidate_grounding import center
from gui_agent.finetuning.evaluation import file_sha256
from gui_agent.finetuning.visual_pointer_prompts import GALLERY_PROMPT_VERSION

visual_asset = train_fixtures.visual_asset


@pytest.fixture
def font_asset(tmp_path, monkeypatch):
    """Pinned fake bytes with two injected test fonts, never a production fallback."""
    fonts = {size: ImageFont.load_default(size=size) for size in (27, 72, 144)}
    path = tmp_path / "synthetic_two_size_font.ttf"
    path.write_bytes(b"SYNTHETIC_PINNED_TEST_FONT_ONLY")
    def load(data, *, size, layout_engine):
        assert data.getvalue() == path.read_bytes() and size in fonts and layout_engine == ImageFont.Layout.BASIC
        return fonts[size]
    monkeypatch.setattr(ImageFont, "truetype", load)
    return {"font_path": path, "font_sha256": file_sha256(path)}, []


def approve(asset):
    assets.write_json(asset.sidecar, asset.metadata)
    asset.row.update(model_png_sha256=file_sha256(asset.model),
                     model_native_rgb_sha256=asset.metadata["model_native_rgb_sha256"],
                     visual_metadata_file_sha256=file_sha256(asset.sidecar))
    entry = {**vi.visual_input_binding(asset.row), "font_sha256": asset.metadata["font_sha256"],
             "config_sha256": asset.metadata["config_sha256"]}
    assets.write_json(asset.manifest, {"schema": vi.GALLERY_MANIFEST_SCHEMA, "status": "approved",
        "input_image_protocol": gallery.PROTOCOL, "visual_prompt_version": GALLERY_PROMPT_VERSION, "entries": [entry]})
    asset.registry = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), asset.original_root, asset.asset_root)


@pytest.fixture
def gallery_asset(visual_asset, font_asset):
    asset = visual_asset
    source = {key: asset.row[key] for key in ("image_path", "image_width", "image_height", "image_file_sha256",
        "image_sha256", "instruction", "actor_step", "candidate_snapshot", "candidate_selection_evidence")}
    result = gallery.render_candidate_gallery(source, **font_asset[0])
    asset.metadata, asset.prompt = result["metadata"], result["prompt"]
    asset.model = asset.asset_root / "gallery.png"
    asset.sidecar = asset.asset_root / "gallery_metadata.json"
    asset.manifest = asset.asset_root / "gallery_manifest.json"
    asset.model.write_bytes(result["png_bytes"])
    asset.row.update(input_image_protocol=gallery.PROTOCOL, visual_prompt_version=GALLERY_PROMPT_VERSION,
                     model_image_path=str(asset.model), visual_metadata_path=str(asset.sidecar))
    approve(asset)
    return asset


def config_for(asset, **changes):
    data = {"input_image_protocol": gallery.PROTOCOL, "target_protocol": "candidate_compact", "prompt_variant": "conditional_actor",
        "visual_prompt_version": GALLERY_PROMPT_VERSION, "visual_asset_manifest_path": str(asset.manifest),
        "visual_asset_manifest_sha256": file_sha256(asset.manifest), "visual_original_root": str(asset.original_root),
        "visual_asset_root": str(asset.asset_root)}
    data.update(changes)
    return training.TrainingConfig(**data)


def hf_for(asset):
    cfg = config_for(asset)
    fields = {key: getattr(cfg, key) for key in ("input_image_protocol", "visual_prompt_version", *training.VISUAL_REGISTRY_FIELDS)}
    return ev.HFConfig(cfg.to_dict(), **fields)


def gallery_worker(asset, **kwargs):
    worker = worker_fixtures.SpyWorker(asset, **kwargs)
    worker.ready.update(input_image_protocol=gallery.PROTOCOL, visual_prompt_version=GALLERY_PROMPT_VERSION)
    return worker


def test_gallery_dimensions_are_distinct_and_native_overview_is_exact(gallery_asset):
    asset = gallery_asset
    descriptor = vi.validate_visual_input(asset.row, registry=asset.registry)
    native_size = (asset.row["image_width"], asset.row["image_height"])
    assert (descriptor.width, descriptor.height) == native_size
    assert descriptor.model_size == tuple(asset.metadata["canvas_size"])
    assert descriptor.model_size[1] > native_size[1]
    canvas = descriptor.to_pil()
    with Image.open(asset.row["image_path"]) as image:
        assert canvas.crop((0, 0, *native_size)).tobytes() == image.convert("RGB").tobytes()
    assert descriptor.input_image_protocol == gallery.PROTOCOL
    assert json.loads(descriptor.evidence_json)["canvas_height"] == descriptor.model_size[1]


@pytest.mark.parametrize("mutation", ["overview", "crop", "table", "seed", "dimensions", "numeric_type"])
def test_even_caller_reapproved_modified_gallery_is_rejected(gallery_asset, mutation):
    asset = gallery_asset
    if mutation in {"overview", "crop", "dimensions"}:
        with Image.open(asset.model) as source:
            image = source.convert("RGB")
        if mutation == "dimensions":
            image = image.crop((0, 0, image.width, image.height - 1))
        else:
            point = (0, 0) if mutation == "overview" else tuple(asset.metadata["id_table"][0]["thumbnail_bbox"][:2])
            image.putpixel(point, (19, 23, 27))
        image.save(asset.model)
        rgb_sha = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
        asset.metadata.update(model_native_rgb_sha256=rgb_sha, gallery_native_rgb_sha256=rgb_sha,
            model_png_sha256=file_sha256(asset.model), gallery_png_sha256=file_sha256(asset.model),
            model_png_bytes=asset.model.stat().st_size, gallery_png_bytes=asset.model.stat().st_size)
    elif mutation == "table":
        asset.metadata["id_table"][0]["native_center"][0] += 1
    elif mutation == "seed":
        asset.metadata["shuffle_seed_derivation"]["effective_seed_hex"] = "0" * 64
    else:
        asset.metadata["selected_candidate_count"] = float(asset.metadata["selected_candidate_count"])
    approve(asset)
    with pytest.raises(ValueError):
        vi.validate_visual_input(asset.row, registry=asset.registry)


@pytest.mark.parametrize("scope", ["row_protocol", "row_version", "manifest_protocol", "manifest_schema", "entry_protocol"])
def test_gallery_cannot_be_downgraded_or_mixed_with_v2(gallery_asset, scope):
    asset = gallery_asset
    if scope == "row_protocol":
        asset.row["input_image_protocol"] = vi.PROTOCOL
    elif scope == "row_version":
        asset.row.pop("visual_prompt_version")
    else:
        manifest = json.loads(asset.manifest.read_text("utf8"))
        if scope == "manifest_protocol":
            manifest["input_image_protocol"] = vi.PROTOCOL
        elif scope == "manifest_schema":
            manifest["schema"] = vi.MANIFEST_SCHEMA
        else:
            manifest["entries"][0]["input_image_protocol"] = vi.PROTOCOL
        assets.write_json(asset.manifest, manifest)
        asset.registry = vi.VisualAssetRegistry(asset.manifest, file_sha256(asset.manifest), asset.original_root, asset.asset_root)
    with pytest.raises(ValueError):
        vi.validate_visual_input(asset.row, registry=asset.registry)


@pytest.mark.parametrize("field,value", [("max_seq_length", 1537), ("max_pixels", 802816), ("min_pixels", 100353),
                                       ("processor_use_fast", False), ("visual_prompt_version", None)])
def test_gallery_config_cannot_widen_profile_or_omit_version(gallery_asset, field, value):
    with pytest.raises(ValueError):
        config_for(gallery_asset, **{field: value})


def test_gallery_training_uses_same_canvas_twice_and_whole_original_supervision(gallery_asset):
    asset = gallery_asset
    original = copy.deepcopy(asset.row)
    processor = train_fixtures.SpyProcessor()
    cfg = config_for(asset)
    assert training.TrainingConfig.from_dict(cfg.to_dict()) == cfg
    encoded = training.encode_training_sample(processor, asset.row, cfg)
    assert [call["size"] for call in processor.calls] == [tuple(asset.metadata["canvas_size"])] * 2
    assert processor.calls[0]["rgb"] == processor.calls[1]["rgb"]
    assert processor.messages[0][0] == processor.messages[1][0]
    assert processor.messages[0][0]["content"][1]["text"] == asset.prompt
    assert json.loads(processor.messages[1][1]["content"]) == original["candidate_target"]
    assert encoded["labels"].tolist() == [[-100] * 6 + [8, 9, 99]]
    assert asset.row == original and "SECRET_BBOX_LABEL" not in json.dumps(processor.messages)


@pytest.mark.parametrize("fault", ["image_mutation", "pixels", "prefix_ids", "eos", "full_length"])
def test_gallery_encoder_keeps_existing_raster_tensor_prefix_eos_length_guards(gallery_asset, fault):
    processor = train_fixtures.SpyProcessor(fault=fault)
    with pytest.raises(ValueError):
        training.encode_training_sample(processor, gallery_asset.row, config_for(gallery_asset))


def test_gallery_loader_preserves_all_native_gt_fields(gallery_asset, tmp_path):
    asset = gallery_asset
    asset.row["training_admission"] = True
    source = tmp_path / "synthetic_gallery_train.jsonl"
    source.write_text(json.dumps(asset.row), "utf8")
    loaded = training.load_prepared_records(source, input_image_protocol=gallery.PROTOCOL,
        target_protocol="candidate_compact", visual_registry=asset.registry, visual_prompt_version=GALLERY_PROMPT_VERSION)[0]
    for field in ("actions", "candidate_target", "action_annotations", "group_id", "sample_id", "image_width", "image_height", "image_path"):
        assert loaded[field] == asset.row[field]


def test_gallery_ack_and_bridge_map_to_original_native_pixels(gallery_asset):
    asset = gallery_asset
    generation = json.dumps(asset.row["candidate_target"])
    result = inference.request_visual_pointer_actions(gallery_worker(asset, generation=generation), asset.row,
        registry=asset.registry, timeout_seconds=1)
    assert result["mapped_actions"] == asset.row["actions"] and result["input_ack_valid"]
    assert result["input_ack"]["schema"] == inference.ACK_SCHEMAS[gallery.PROTOCOL]
    assert result["input_ack"]["input_evidence"]["native_height"] == asset.row["image_height"]
    assert result["input_ack"]["input_evidence"]["canvas_height"] > asset.row["image_height"]


def test_gallery_child_opens_canvas_and_cannot_accept_v2_or_native_requests(gallery_asset, monkeypatch):
    asset = gallery_asset
    config = hf_for(asset)
    state = worker_fixtures.install_child_spies(monkeypatch)
    descriptor, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    mixed = copy.deepcopy(envelope)
    mixed["input_image_protocol"] = vi.PROTOCOL
    messages = worker_fixtures.child_responses(config, [envelope, mixed,
        {"image_path": asset.row["image_path"], "prompt": "native bypass"}, None])
    assert state.processor_calls == state.generate_calls == 1
    assert state.raw_images[0].size == descriptor.model_size
    assert messages[0]["input_image_protocol"] == gallery.PROTOCOL
    assert messages[1]["input_ack"]["schema"] == inference.ACK_SCHEMAS[gallery.PROTOCOL]
    assert all(message["status"] == "error" and "input_ack" not in message for message in messages[2:])


@pytest.mark.parametrize("mutation", ["ack", "worker_version", "context"])
def test_gallery_bridge_fail_closed_without_mapping_preserves_raw_reply(gallery_asset, mutation):
    asset = gallery_asset
    worker = gallery_worker(asset)
    if mutation == "worker_version":
        worker.ready["input_image_protocol"] = vi.PROTOCOL
        with pytest.raises(ValueError):
            inference.request_visual_pointer_actions(worker, asset.row, registry=asset.registry, timeout_seconds=1)
        assert not worker.requests
        return
    if mutation == "ack":
        worker.ack_mutation = lambda ack: ack.update(input_image_protocol=vi.PROTOCOL)
    else:
        worker.mutation = lambda: asset.row.update(actor_step="changed legal context")
    result = inference.request_visual_pointer_actions(worker, asset.row, registry=asset.registry, timeout_seconds=1)
    assert result["raw_reply"]["generation"] and result["mapped_actions"] is None and result["mapping_error"]


def test_v2_and_native_configs_reject_gallery_prompt(gallery_asset):
    with pytest.raises(ValueError):
        config_for(gallery_asset, input_image_protocol=vi.PROTOCOL)
    with pytest.raises(ValueError):
        training.TrainingConfig(visual_prompt_version=GALLERY_PROMPT_VERSION)
    cfg = hf_for(gallery_asset)
    with pytest.raises(ValueError):
        replace(cfg, input_image_protocol=vi.PROTOCOL)


def test_native_coordinate_bounds_do_not_use_taller_canvas(gallery_asset):
    asset = gallery_asset
    y = center(asset.row["candidate_snapshot"]["candidates"][0]["bbox"])[1]
    worker = gallery_worker(asset, generation=json.dumps([{"action_type": "click", "candidate": 1,
        "dx": 0, "dy": asset.row["image_height"] - y + 2}]))
    result = inference.request_visual_pointer_actions(worker, asset.row, registry=asset.registry, timeout_seconds=1)
    assert result["mapped_actions"] is None and result["mapping_error"]


def test_gallery_child_processor_mutation_does_not_generate_or_ack(gallery_asset, monkeypatch):
    asset = gallery_asset
    config = hf_for(asset)
    state = worker_fixtures.install_child_spies(monkeypatch, processor_mutation=lambda image: image.putpixel((0, 0), (1, 2, 3)))
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    messages = worker_fixtures.child_responses(config, [envelope, None])
    assert state.processor_calls == 1 and state.generate_calls == 0
    assert messages[1]["status"] == "error" and "input_ack" not in messages[1]


def test_worker_parent_rejects_native_and_other_visual_protocol_before_send(gallery_asset):
    _, envelope = inference.prepare_visual_request(gallery_asset.row, registry=gallery_asset.registry)
    worker = ev.HFWorker.__new__(ev.HFWorker)
    worker.input_image_protocol = vi.PROTOCOL
    worker.request_count = 0
    worker._request_payload = lambda *_args, **_kwargs: pytest.fail("mixed worker envelope reached send")
    with pytest.raises(ValueError):
        worker.request_visual(envelope, timeout_seconds=1)
    worker.input_image_protocol = gallery.PROTOCOL
    with pytest.raises(ValueError):
        worker.request("synthetic-native.png", "native bypass", timeout_seconds=1)
    assert worker.request_count == 0


def test_gallery_child_expanded_prefix_over1536_never_generates(gallery_asset, monkeypatch):
    asset = gallery_asset
    config = hf_for(asset)
    state = worker_fixtures.install_child_spies(monkeypatch)
    fake_training = sys.modules["gui_agent.finetuning.training"]
    original_loader = fake_training.load_model_and_processor
    class LongPrefixProcessor:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
        def __call__(self, **kwargs):
            result = self.wrapped(**kwargs)
            result["input_ids"] = SimpleNamespace(shape=(1, 1537))
            return result
    def loader(*args, **kwargs):
        model, processor = original_loader(*args, **kwargs)
        return model, LongPrefixProcessor(processor)
    monkeypatch.setattr(fake_training, "load_model_and_processor", loader)
    _, envelope = inference.prepare_visual_request(asset.row, registry=asset.registry)
    messages = worker_fixtures.child_responses(config, [envelope, None])
    assert state.processor_calls == 1 and state.generate_calls == 0
    assert messages[1]["status"] == "error" and "input_ack" not in messages[1]
    assert "exceeds1536" in messages[1]["error"]


def test_responsive_wide_canvas_binding_and_validator_use_shared_geometry(tmp_path, font_asset):
    original_root, asset_root = tmp_path / "wide_native", tmp_path / "wide_assets"
    original_root.mkdir()
    asset_root.mkdir()
    row = fixtures.make_sample(original_root, size=(2048, 768), candidates=[
        fixtures.candidate("wide-A", 120, 160), fixtures.candidate("wide-B", 640, 280)])
    result = gallery.render_candidate_gallery(row, **font_asset[0])
    asset = SimpleNamespace(row=row, metadata=result["metadata"], original_root=original_root, asset_root=asset_root,
        model=asset_root / "wide.png", sidecar=asset_root / "metadata.json", manifest=asset_root / "manifest.json")
    asset.model.write_bytes(result["png_bytes"])
    row.update(input_image_protocol=gallery.PROTOCOL, visual_prompt_version=GALLERY_PROMPT_VERSION,
               model_image_path=str(asset.model), visual_metadata_path=str(asset.sidecar))
    approve(asset)
    descriptor = vi.validate_visual_input(row, registry=asset.registry)
    assert (descriptor.width, descriptor.height) == (2048, 768)
    assert descriptor.model_size == (2048, 768 + gallery.resolve_gallery_geometry(2048)["tile_height"])
    assert descriptor.model_size == tuple(result["metadata"]["canvas_size"])
    assert json.loads(descriptor.evidence_json)["canvas_height"] == descriptor.model_size[1]
