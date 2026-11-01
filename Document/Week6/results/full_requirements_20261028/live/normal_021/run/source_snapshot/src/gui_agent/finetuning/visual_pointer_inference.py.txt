"""Explicit visual raster transport and native pointer mapping, without labels.

The registry is a caller configuration, never an assertion taken from a row or
request. Heavy inference remains in evaluation's child; this module only binds
the visible input, raster validation, acknowledgement, and native coordinates.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .visual_pointer_prompts import checked_prompt_version, declared_prompt_version

if TYPE_CHECKING:
    from .visual_pointer_input import VisualAssetRegistry, VisualPointerInput

PROTOCOL = "visual_pointer_v2"
GALLERY_PROTOCOL = "visual_pointer_gallery_v3"
GALLERY_PROMPT_VERSION = "week5.visual-pointer-prompt.gallery.v4"
GALLERY_SPATIAL_PROMPT_VERSION = "week5.visual-pointer-prompt.gallery-native-spatial.v1"
GALLERY_PROMPT_VERSIONS = (GALLERY_PROMPT_VERSION, GALLERY_SPATIAL_PROMPT_VERSION)
VISUAL_PROTOCOLS = (PROTOCOL, GALLERY_PROTOCOL)

REQUEST_SCHEMA = "week5.visual_pointer_request.v1"
ACK_SCHEMA = "week5.visual_pointer_input_ack.v1"
REQUEST_SCHEMAS = {PROTOCOL: REQUEST_SCHEMA, GALLERY_PROTOCOL: "week5.visual_pointer_gallery_request.v1"}
ACK_SCHEMAS = {PROTOCOL: ACK_SCHEMA, GALLERY_PROTOCOL: "week5.visual_pointer_gallery_input_ack.v1"}
REGISTRY_FIELDS = (
    "visual_asset_manifest_path", "visual_asset_manifest_sha256",
    "visual_original_root", "visual_asset_root",
)
VISIBLE_FIELDS = frozenset({
    "input_image_protocol", "image_path", "image_width", "image_height",
    "image_file_sha256", "image_sha256", "instruction", "actor_step",
    "candidate_snapshot", "candidate_selection_evidence", "model_image_path",
    "model_png_sha256", "model_native_rgb_sha256", "visual_metadata_path",
    "visual_metadata_file_sha256",
})
OPTIONAL_VISIBLE_FIELDS = frozenset({"raw_snapshot_path", "raw_snapshot_file_sha256", "visual_prompt_version"})
_ENVELOPE_FIELDS = frozenset({
    "schema", "input_image_protocol", "visible_sample", "prompt", "prompt_sha256",
})


def _absolute(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trusted visual registry paths must be nonempty strings")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("trusted visual registry paths must be absolute without traversal")
    return str(path)


def registry_identity(registry: VisualAssetRegistry) -> dict[str, str]:
    """Serializable authority identity; construction alone approves no asset."""
    from .visual_pointer_input import VisualAssetRegistry

    if not isinstance(registry, VisualAssetRegistry):
        raise TypeError("trusted caller VisualAssetRegistry required")
    digest = registry.manifest_sha256
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("trusted visual registry SHA256 must be lowercase hexadecimal")
    return {
        "manifest_path": _absolute(str(registry.manifest_path)),
        "manifest_sha256": digest,
        "original_root": _absolute(str(registry.original_root)),
        "asset_root": _absolute(str(registry.asset_root)),
    }


def registry_from_config(config: Any) -> VisualAssetRegistry:
    from .visual_pointer_input import VisualAssetRegistry

    registry = VisualAssetRegistry(
        config.visual_asset_manifest_path, config.visual_asset_manifest_sha256,
        config.visual_original_root, config.visual_asset_root,
    )
    registry_identity(registry)
    return registry


def validate_worker_config(config: Any) -> None:
    """Pure configuration checks; no model, raster, or manifest is opened."""
    protocol = config.input_image_protocol
    prompt_version = checked_prompt_version(config.visual_prompt_version)
    if checked_prompt_version(config.training_config.get("visual_prompt_version")) != prompt_version:
        raise ValueError("HF and training visual_prompt_version differ")
    if protocol not in {"native", *VISUAL_PROTOCOLS}:
        raise ValueError("unknown input_image_protocol")
    values = [getattr(config, field) for field in REGISTRY_FIELDS]
    if config.training_config.get("input_image_protocol", "native") != protocol:
        raise ValueError("HF and training input_image_protocol differ")
    if protocol == "native":
        if prompt_version is not None or any(value is not None for value in values) or any(
            config.training_config.get(field) is not None for field in REGISTRY_FIELDS
        ):
            raise ValueError("native inference cannot carry a visual registry")
        return
    spatial = config.training_config.get("target_protocol") in {"native_spatial_head", "native_spatial_grid_head"}
    if (protocol == GALLERY_PROTOCOL) != (prompt_version in GALLERY_PROMPT_VERSIONS):
        raise ValueError("gallery worker protocol and v4 prompt must agree")
    if protocol == GALLERY_PROTOCOL and tuple(config.training_config.get(key) for key in (
            "min_pixels", "max_pixels", "max_seq_length", "processor_use_fast")) != (100352, 301056, 1536, True):
        raise ValueError("gallery worker requires fixed1536/100352/301056/fast profile")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("visual inference requires all four trusted registry fields")
    registry_from_config(config)
    if spatial:
        if config.training_config["target_protocol"] == "native_spatial_grid_head":
            from .native_spatial_grid_inference import validate_spatial_config
        else:
            from .native_spatial_inference import validate_spatial_config

        validate_spatial_config(config)
    elif prompt_version == GALLERY_SPATIAL_PROMPT_VERSION:
        raise ValueError("native spatial prompt requires its explicit target protocol")
    elif config.training_config.get("target_protocol") != "candidate_compact" or config.training_config.get("prompt_variant") != "conditional_actor":
        raise ValueError("visual inference requires candidate_compact and conditional_actor")
    if config.output_constraint != "none":
        raise ValueError("visual candidate inference requires no output_constraint")
    for field in REGISTRY_FIELDS:
        if field in config.training_config and config.training_config[field] != getattr(config, field):
            raise ValueError("HF and training trusted visual registry differ")


def _input_only(value: Any) -> None:
    """Reject supervision keys before accessing their values, also when nested."""
    from .visual_pointer_input import FORBIDDEN_SNAPSHOT_KEYS

    if isinstance(value, Mapping):
        if FORBIDDEN_SNAPSHOT_KEYS.intersection(value):
            raise ValueError("visual request must contain input-only evidence")
        for key in value:
            if not isinstance(key, str):
                raise TypeError("visual request keys must be strings")
            _input_only(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _input_only(item)


def _json_copy(value: Any) -> Any:
    _input_only(value)
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def visible_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Read only fixed input fields even when the parent has unreadable labels."""
    result = {field: sample[field] for field in VISIBLE_FIELDS}
    for field in OPTIONAL_VISIBLE_FIELDS:
        if field == "visual_prompt_version":
            version = declared_prompt_version(sample)
            if version is not None:
                result[field] = version
            continue
        value = sample.get(field)
        if value is not None:
            result[field] = value
    return _json_copy(result)


def prepare_visual_request(
    sample: Mapping[str, Any], *, registry: VisualAssetRegistry,
) -> tuple[VisualPointerInput, dict[str, Any]]:
    from .visual_pointer_input import validate_visual_input, visual_input_binding

    descriptor = validate_visual_input(sample, registry=registry)
    visible = visible_sample(sample)
    # The same immutable descriptor must still belong to the copied envelope.
    evidence = json.loads(descriptor.evidence_json)
    if visual_input_binding(visible)["input_binding_sha256"] != descriptor.input_binding_sha256:
        raise ValueError("visual input changed while building the request")
    descriptor.assert_current()
    envelope = {
        "schema": REQUEST_SCHEMAS[descriptor.input_image_protocol], "input_image_protocol": descriptor.input_image_protocol,
        "visible_sample": visible, "prompt": descriptor.prompt,
        "prompt_sha256": evidence["prompt_sha256"],
    }
    if descriptor.visual_prompt_version is not None:
        envelope["visual_prompt_version"] = descriptor.visual_prompt_version
    return descriptor, envelope


def checked_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Transport shape validation, independent of any parent asset approval."""
    if not isinstance(envelope, Mapping):
        raise TypeError("visual request must be a mapping")
    version = declared_prompt_version(envelope)
    expected_fields = set(_ENVELOPE_FIELDS) | ({"visual_prompt_version"} if version is not None else set())
    if set(envelope) != expected_fields:
        raise ValueError("visual request has unexpected or missing envelope fields")
    protocol = envelope["input_image_protocol"]
    if protocol not in VISUAL_PROTOCOLS or envelope["schema"] != REQUEST_SCHEMAS[protocol]:
        raise ValueError("explicit visual request protocol required")
    if (protocol == GALLERY_PROTOCOL) != (version in GALLERY_PROMPT_VERSIONS):
        raise ValueError("gallery envelope and v4 prompt must agree")
    sample = envelope["visible_sample"]
    if not isinstance(sample, Mapping) or not VISIBLE_FIELDS <= set(sample) or set(sample) - VISIBLE_FIELDS - OPTIONAL_VISIBLE_FIELDS:
        raise ValueError("visual request has unexpected or missing visible fields")
    if declared_prompt_version(sample) != version:
        raise ValueError("visual request and visible sample prompt versions differ")
    result = _json_copy(envelope)
    prompt = result["prompt"]
    if not isinstance(prompt, str) or not prompt or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != result["prompt_sha256"]:
        raise ValueError("visual request prompt SHA256 differs")
    if result["visible_sample"]["input_image_protocol"] != protocol:
        raise ValueError("visual request cannot carry a native sample")
    return result


def validate_visual_request(
    envelope: Mapping[str, Any], *, registry: VisualAssetRegistry,
) -> VisualPointerInput:
    from .visual_pointer_input import validate_visual_input

    request = checked_envelope(envelope)
    descriptor = validate_visual_input(request["visible_sample"], registry=registry)
    if request["prompt"] != descriptor.prompt or declared_prompt_version(request) != descriptor.visual_prompt_version:
        raise ValueError("visual request prompt differs from the registered input")
    descriptor.assert_current()
    return descriptor


def input_ack(descriptor: VisualPointerInput, registry: VisualAssetRegistry) -> dict[str, Any]:
    descriptor.assert_current()
    acknowledgement = {
        "schema": ACK_SCHEMAS[descriptor.input_image_protocol], "input_image_protocol": descriptor.input_image_protocol,
        "visual_asset_registry": registry_identity(registry),
        "input_evidence": json.loads(descriptor.evidence_json),
    }
    if descriptor.visual_prompt_version is not None:
        acknowledgement["visual_prompt_version"] = descriptor.visual_prompt_version
    return acknowledgement


def assert_visual_raster(descriptor: VisualPointerInput, image: Any) -> None:
    """Files staying fixed does not excuse mutation of the consumed PIL raster."""
    if image.mode != "RGB" or image.size != descriptor.model_size or image.tobytes() != descriptor.rgb_bytes:
        raise ValueError("processor raster changed from the registered immutable RGB")


def assert_visual_ready(ready: Mapping[str, Any], registry: VisualAssetRegistry, *, visual_prompt_version: str | None = None,
                        input_image_protocol: str = PROTOCOL) -> None:
    if input_image_protocol not in VISUAL_PROTOCOLS or (input_image_protocol == GALLERY_PROTOCOL) != (visual_prompt_version in GALLERY_PROMPT_VERSIONS):
        raise ValueError("visual ready caller protocol/version differs")
    if ready.get("kind") != "ready" or ready.get("input_image_protocol") != input_image_protocol or ready.get("visual_asset_registry") != registry_identity(registry):
        raise ValueError("visual worker ready protocol or trusted registry differs")
    if declared_prompt_version(ready) != checked_prompt_version(visual_prompt_version):
        raise ValueError("visual worker ready prompt version differs")
    if ready.get("output_constraint") is not None or ready.get("constraint_hook"):
        raise ValueError("visual candidate inference requires an unconstrained worker")


def request_visual_pointer_actions(
    worker: Any, sample: Mapping[str, Any], *, registry: VisualAssetRegistry,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Map an acknowledged visual reply using the complete original native input."""
    from .pointer_protocol import map_pointer_actions

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("request timeout must be finite and positive")
    version = declared_prompt_version(sample)
    if version == GALLERY_SPATIAL_PROMPT_VERSION:
        raise ValueError("native spatial output requires its raw joint gate, never pointer mapping")
    protocol = sample["input_image_protocol"]
    assert_visual_ready(getattr(worker, "ready", {}), registry, visual_prompt_version=version, input_image_protocol=protocol)
    descriptor, envelope = prepare_visual_request(sample, registry=registry)
    expected_ack = input_ack(descriptor, registry)
    binding = envelope["visible_sample"]["candidate_selection_evidence"]
    reply = worker.request_visual(envelope, timeout_seconds=timeout_seconds)
    result = {
        "raw_reply": reply, "prompt": descriptor.prompt,
        "snapshot_sha256": binding["snapshot_sha256"],
        "candidate_selection_evidence": binding,
        "visual_input_evidence": json.loads(descriptor.evidence_json),
        "input_ack": reply.get("input_ack"), "input_ack_valid": False,
        "mapped_actions": None, "mapping_error": None,
    }
    if reply.get("status") == "ok":
        try:
            assert_visual_ready(getattr(worker, "ready", {}), registry, visual_prompt_version=descriptor.visual_prompt_version,
                                input_image_protocol=descriptor.input_image_protocol)
            if reply.get("input_ack") != expected_ack:
                raise ValueError("visual worker input acknowledgement differs")
            descriptor.assert_current()
            current = visible_sample(sample)
            if current != envelope["visible_sample"]:
                raise ValueError("visual input changed during the worker request")
            result["input_ack_valid"] = True
        except (ValueError, TypeError, KeyError, OSError) as exc:
            result["mapping_error"] = f"{type(exc).__name__}: {exc}"
            return result
        try:
            result["mapped_actions"] = map_pointer_actions(
                reply["generation"], current["candidate_snapshot"], descriptor.original_image_path,
                instruction=current["instruction"], actor_step=current["actor_step"],
                expected_selection=binding,
            )
        except (ValueError, TypeError, KeyError, OSError) as exc:
            result["mapped_actions"] = None
            result["mapping_error"] = f"{type(exc).__name__}: {exc}"
        try:
            descriptor.assert_current()
        except (ValueError, TypeError, KeyError, OSError) as exc:
            result["input_ack_valid"] = False
            result["mapped_actions"] = None
            result["mapping_error"] = f"{type(exc).__name__}: {exc}"
    return result
