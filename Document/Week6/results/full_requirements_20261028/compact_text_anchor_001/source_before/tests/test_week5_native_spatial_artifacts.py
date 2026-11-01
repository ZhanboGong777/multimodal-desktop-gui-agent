"""Small CPU head artifact tests; no HF, CUDA, actual images or dataset reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gui_agent.finetuning import native_spatial_artifacts as artifacts
from gui_agent.finetuning.native_spatial_head import make_spatial_head


def bindings():
    return {"base_model_id": "synthetic-Qwen", "base_model_revision": "a" * 40,
            "base_model_config_sha256": "b" * 64, "prompt_version": "synthetic-semantic-v1",
            "prompt_sha256": "c" * 64, "tokenizer_json_sha256": "d" * 64,
            "core_source_sha256": "e" * 64, "training_source_sha256": "f" * 64,
            "eos_token_id": 99}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_initial_and_final_are_exact_saved_parameters(tmp_path):
    import torch

    head = make_spatial_head(seed=42)
    first = artifacts.save_spatial_head_artifact(head, tmp_path, role="initial", bindings=bindings())
    with torch.no_grad():
        head[0].weight.add_(0.125)
    last = artifacts.save_spatial_head_artifact(head, tmp_path, role="final", bindings=bindings())
    manifest_sha = digest(tmp_path / artifacts.MANIFEST_NAME)
    initial, initial_identity = artifacts.load_spatial_head_artifact(
        tmp_path, role="initial", expected_bindings=bindings(), expected_manifest_sha256=manifest_sha)
    final, final_identity = artifacts.load_spatial_head_artifact(
        tmp_path, role="final", expected_bindings=bindings(), expected_manifest_sha256=manifest_sha)
    assert initial_identity["parameter_sha256"] == first["parameter_sha256"]
    assert final_identity["parameter_sha256"] == last["parameter_sha256"]
    assert torch.equal(final[0].weight, head[0].weight)
    assert not torch.equal(initial[0].weight, final[0].weight)
    assert initial_identity["file_ledger"] == final_identity["file_ledger"]
    assert set(initial_identity["file_ledger"]) == {
        "spatial_head_config.json", "spatial_head_manifest.json", "spatial_head_initial.pt", "spatial_head_final.pt"}
    assert all(parameter.dtype == torch.float32 for parameter in initial.parameters())
    with pytest.raises(ValueError, match="overwrite"):
        artifacts.save_spatial_head_artifact(head, tmp_path, role="initial", bindings=bindings())


@pytest.mark.parametrize("mutation", ["manifest_sha", "config", "weight", "other_role_weight", "binding", "missing"])
def test_pure_inspector_refuses_stale_or_missing_artifacts(tmp_path, monkeypatch, mutation):
    head = make_spatial_head(seed=42)
    artifacts.save_spatial_head_artifact(head, tmp_path, role="initial", bindings=bindings())
    artifacts.save_spatial_head_artifact(head, tmp_path, role="final", bindings=bindings())
    manifest_sha, expected = digest(tmp_path / artifacts.MANIFEST_NAME), bindings()
    if mutation == "manifest_sha":
        manifest_sha = "0" * 64
    elif mutation == "binding":
        expected["eos_token_id"] += 1
    else:
        filename = {"config": artifacts.CONFIG_NAME, "weight": "spatial_head_initial.pt",
                    "other_role_weight": "spatial_head_final.pt", "missing": "spatial_head_initial.pt"}[mutation]
        path = tmp_path / filename
        if mutation == "missing":
            path.unlink()
        else:
            path.write_bytes(path.read_bytes() + b"tampered")
    monkeypatch.setattr(artifacts, "make_spatial_head", lambda **_: pytest.fail("head construction before binding validation"))
    with pytest.raises((ValueError, FileNotFoundError)):
        artifacts.inspect_spatial_head_artifact(tmp_path, role="initial", expected_bindings=expected,
                                                expected_manifest_sha256=manifest_sha)


def test_parameter_hash_mismatch_fails_even_with_rebound_manifest(tmp_path):
    head = make_spatial_head(seed=42)
    artifacts.save_spatial_head_artifact(head, tmp_path, role="initial", bindings=bindings())
    manifest_path = tmp_path / artifacts.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text("utf8"))
    manifest["artifacts"]["initial"]["parameter_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="parameter SHA"):
        artifacts.load_spatial_head_artifact(tmp_path, role="initial", expected_bindings=bindings(),
                                            expected_manifest_sha256=digest(manifest_path))
