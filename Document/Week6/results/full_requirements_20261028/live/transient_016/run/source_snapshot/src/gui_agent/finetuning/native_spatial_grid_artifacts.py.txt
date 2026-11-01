"""Exact external spatial-head artifacts; imports never construct model weights."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .native_spatial_grid_head import STATE_SHAPES, head_configuration, make_spatial_head
from .native_spatial_head import strict_json

SCHEMA = "week5.native-spatial-grid-artifacts.v1"
CONFIG_NAME = "spatial_head_config.json"
MANIFEST_NAME = "spatial_head_manifest.json"
ROLES = ("initial", "final", "partial")
BINDING_KEYS = frozenset({"base_model_id", "base_model_revision", "base_model_config_sha256",
                          "prompt_version", "prompt_sha256", "eos_token_id",
                          "tokenizer_json_sha256", "core_source_sha256", "training_source_sha256",
                          "geometry_source_sha256", "marker_source_sha256"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bindings(value: Mapping) -> dict:
    if not isinstance(value, Mapping) or set(value) != BINDING_KEYS:
        raise ValueError("explicit complete spatial artifact bindings required")
    result = dict(value)
    if type(result["eos_token_id"]) is not int or result["eos_token_id"] < 0:
        raise ValueError("spatial artifact EOS ID must be a nonnegative integer")
    if any(not isinstance(item, str) or not item.strip() for key, item in result.items() if key != "eos_token_id"):
        raise ValueError("spatial artifact bindings must be nonempty strings")
    for key in BINDING_KEYS - {"base_model_id", "prompt_version", "eos_token_id"}:
        expected = 40 if key == "base_model_revision" else 64
        if len(result[key]) != expected or any(char not in "0123456789abcdef" for char in result[key]):
            raise ValueError(f"invalid spatial artifact binding hash: {key}")
    return result


def _state(value: Mapping) -> dict:
    import torch

    if not isinstance(value, Mapping) or set(value) != set(STATE_SHAPES):
        raise ValueError("spatial state fields differ from fixed architecture")
    result = {}
    for name, shape in STATE_SHAPES.items():
        tensor = value[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError("spatial state shape or FP32 dtype differs")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("spatial state contains nonfinite parameters")
        result[name] = tensor.detach().cpu().contiguous()
    return result


def state_sha256(value: Mapping) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(_state(value).items()):
        digest.update(name.encode("utf8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _documents(directory: Path, bindings: Mapping) -> tuple[dict, bytes, dict]:
    wanted = {"schema": SCHEMA, "head_configuration": head_configuration(), "bindings": _bindings(bindings)}
    config_bytes = _canonical(wanted) + b"\n"
    config_path, manifest_path = directory / CONFIG_NAME, directory / MANIFEST_NAME
    if config_path.exists():
        if config_path.read_bytes() != config_bytes:
            raise ValueError("saved spatial head config or bindings differ")
        manifest = strict_json(manifest_path.read_bytes())
        expected_keys = {"schema", "config_file", "config_sha256", "bindings_sha256", "artifacts"}
        if (set(manifest) != expected_keys or manifest["schema"] != SCHEMA
                or manifest["config_file"] != CONFIG_NAME or manifest["config_sha256"] != _sha(config_bytes)
                or manifest["bindings_sha256"] != _sha(_canonical(wanted["bindings"]))
                or not isinstance(manifest["artifacts"], dict) or set(manifest["artifacts"]) - set(ROLES)):
            raise ValueError("saved spatial manifest contract differs")
    else:
        if manifest_path.exists():
            raise ValueError("spatial manifest exists without exact config")
        manifest = {"schema": SCHEMA, "config_file": CONFIG_NAME, "config_sha256": _sha(config_bytes),
                    "bindings_sha256": _sha(_canonical(wanted["bindings"])), "artifacts": {}}
    return wanted, config_bytes, manifest


def save_spatial_head_artifact(head, output_dir: str | Path, *, role: str, bindings: Mapping) -> dict:
    """Persist actual FP32 parameters once; PEFT's save never owns this head."""
    import torch

    if role not in ROLES:
        raise ValueError("unknown spatial artifact role")
    directory = Path(output_dir).resolve(strict=True)
    _, config_bytes, manifest = _documents(directory, bindings)
    filename = f"spatial_head_{role}.pt"
    path = directory / filename
    if path.exists() or role in manifest["artifacts"]:
        raise ValueError("refusing to overwrite saved spatial head")
    state = _state(head.state_dict())
    temporary = directory / (filename + ".tmp")
    if temporary.exists():
        raise ValueError("spatial temporary artifact already exists")
    with temporary.open("xb") as handle:
        torch.save(state, handle)
    temporary.replace(path)
    record = {"filename": filename, "file_sha256": _sha(path.read_bytes()),
              "parameter_sha256": state_sha256(state)}
    record["artifact_tree_sha256"] = _sha(_canonical({CONFIG_NAME: manifest["config_sha256"], filename: record["file_sha256"]}))
    manifest["artifacts"][role] = record
    config_path = directory / CONFIG_NAME
    if not config_path.exists():
        with config_path.open("xb") as handle:
            handle.write(config_bytes)
    temporary_manifest = directory / (MANIFEST_NAME + ".tmp")
    with temporary_manifest.open("wb") as handle:
        handle.write(_canonical(manifest) + b"\n")
    temporary_manifest.replace(directory / MANIFEST_NAME)
    return {"role": role, "output_dir": str(directory), "config_sha256": manifest["config_sha256"],
            "bindings_sha256": manifest["bindings_sha256"], **record}


def inspect_spatial_head_artifact(output_dir: str | Path, *, role: str, expected_bindings: Mapping,
                                  expected_manifest_sha256: str) -> dict:
    """Verify every bound file without Torch/HF/model construction."""
    if role not in {"initial", "final"}:
        raise ValueError("inference requires initial or final spatial artifact")
    directory = Path(output_dir).resolve(strict=True)
    manifest_bytes = (directory / MANIFEST_NAME).read_bytes()
    if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
            or _sha(manifest_bytes) != expected_manifest_sha256):
        raise ValueError("spatial artifact manifest SHA differs")
    _, _, manifest = _documents(directory, expected_bindings)
    if not (directory / CONFIG_NAME).is_file() or role not in manifest["artifacts"]:
        raise ValueError("required saved spatial head/config missing")
    # Validate the entire published role ledger, not just the selected weight.
    for saved_role, record in manifest["artifacts"].items():
        if (not isinstance(record, dict) or set(record) != {"filename", "file_sha256", "parameter_sha256", "artifact_tree_sha256"}
                or record["filename"] != f"spatial_head_{saved_role}.pt"):
            raise ValueError("spatial role ledger contract differs")
        payload = (directory / record["filename"]).read_bytes()
        tree = _sha(_canonical({CONFIG_NAME: manifest["config_sha256"], record["filename"]: _sha(payload)}))
        if _sha(payload) != record["file_sha256"] or tree != record["artifact_tree_sha256"]:
            raise ValueError("spatial weight file/tree SHA differs")
    record = manifest["artifacts"][role]
    file_ledger = {CONFIG_NAME: manifest["config_sha256"], MANIFEST_NAME: expected_manifest_sha256,
                   **{item["filename"]: item["file_sha256"] for item in manifest["artifacts"].values()}}
    return {"role": role, "output_dir": str(directory), "config_sha256": manifest["config_sha256"],
            "bindings_sha256": manifest["bindings_sha256"], "manifest_sha256": expected_manifest_sha256,
            "file_ledger": file_ledger, "full_artifact_tree_sha256": _sha(_canonical(file_ledger)),
            **record}


def load_spatial_head_artifact(output_dir: str | Path, *, role: str, expected_bindings: Mapping,
                               expected_manifest_sha256: str, device: str = "cpu") -> tuple[Any, dict]:
    """Load exact saved initial/final state; refuse every missing or stale file."""
    import torch

    identity = inspect_spatial_head_artifact(output_dir, role=role, expected_bindings=expected_bindings,
                                             expected_manifest_sha256=expected_manifest_sha256)
    directory = Path(identity["output_dir"])
    record = identity
    state = _state(torch.load(directory / record["filename"], map_location="cpu", weights_only=True))
    if state_sha256(state) != record["parameter_sha256"]:
        raise ValueError("spatial parameter SHA differs")
    head = make_spatial_head(seed=0)
    head.load_state_dict(state, strict=True)
    head.to(device=device, dtype=torch.float32)
    head.eval()
    if inspect_spatial_head_artifact(directory, role=role, expected_bindings=expected_bindings,
                                     expected_manifest_sha256=expected_manifest_sha256) != identity:
        raise ValueError("spatial artifact changed during load")
    return head, identity
