"""Public grid artifact contracts using synthetic CPU heads and temporary files."""
from __future__ import annotations

import hashlib
import json

import pytest

from gui_agent.finetuning import native_spatial_grid_artifacts as artifacts
from gui_agent.finetuning.native_spatial_grid_head import (
    STATE_SHAPES,
    head_configuration,
    make_spatial_head,
)


def bindings():
    return {
        "base_model_id": "synthetic-Qwen",
        "base_model_revision": "a" * 40,
        "base_model_config_sha256": "b" * 64,
        "prompt_version": "synthetic-grid-prompt",
        "prompt_sha256": "c" * 64,
        "eos_token_id": 99,
        "tokenizer_json_sha256": "d" * 64,
        "core_source_sha256": "e" * 64,
        "training_source_sha256": "f" * 64,
        "geometry_source_sha256": "1" * 64,
        "marker_source_sha256": "2" * 64,
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf8")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    import torch

    assert not torch.cuda.is_initialized()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(original_threads)


@pytest.fixture(autouse=True)
def cpu_rng_guard(monkeypatch):
    import torch

    assert not torch.cuda.is_initialized()
    original_rng = torch.random.get_rng_state().clone()

    def reject_cuda_seed(*args, **kwargs):
        pytest.fail("CPU artifact operations must not seed CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", reject_cuda_seed)
    yield
    assert not torch.cuda.is_initialized()
    assert torch.equal(original_rng, torch.random.get_rng_state())


def test_initial_final_exact_ten_fp32_tensors_and_changed_parameter_hash(tmp_path):
    import torch

    head = make_spatial_head(seed=42)
    initial_state = {name: value.clone() for name, value in head.state_dict().items()}
    assert len(initial_state) == 10 and set(initial_state) == set(STATE_SHAPES)
    initial_saved = artifacts.save_spatial_head_artifact(
        head, tmp_path, role="initial", bindings=bindings()
    )
    with torch.no_grad():
        head.query_projection.weight.add_(0.125)
    final_saved = artifacts.save_spatial_head_artifact(
        head, tmp_path, role="final", bindings=bindings()
    )
    assert initial_saved["parameter_sha256"] != final_saved["parameter_sha256"]
    config = json.loads((tmp_path / artifacts.CONFIG_NAME).read_text("utf8"))
    assert config["head_configuration"] == head_configuration()
    assert config["bindings"] == bindings()
    identities = []
    for role, wanted, saved in (
        ("initial", initial_state, initial_saved),
        ("final", head.state_dict(), final_saved),
    ):
        loaded, identity = artifacts.load_spatial_head_artifact(
            tmp_path, role=role, expected_bindings=bindings(),
            expected_manifest_sha256=digest(tmp_path / artifacts.MANIFEST_NAME),
        )
        assert identity["parameter_sha256"] == saved["parameter_sha256"]
        assert artifacts.state_sha256(loaded.state_dict()) == saved["parameter_sha256"]
        for name, tensor in loaded.state_dict().items():
            assert tensor.dtype == torch.float32 and tensor.device.type == "cpu"
            assert tuple(tensor.shape) == STATE_SHAPES[name]
            assert torch.equal(tensor, wanted[name])
        identities.append(identity)
    assert identities[0]["file_ledger"] == identities[1]["file_ledger"]
    assert set(identities[0]["file_ledger"]) == {
        artifacts.CONFIG_NAME, artifacts.MANIFEST_NAME,
        "spatial_head_initial.pt", "spatial_head_final.pt",
    }


def test_exact_eleven_bindings_reject_missing_legacy_nine_and_extra(tmp_path, monkeypatch):
    head = make_spatial_head(seed=42)
    complete = bindings()
    assert len(complete) == 11 and set(complete) == set(artifacts.BINDING_KEYS)
    artifacts.save_spatial_head_artifact(head, tmp_path, role="initial", bindings=complete)
    manifest_sha = digest(tmp_path / artifacts.MANIFEST_NAME)
    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()
    invalid = [{key: value for key, value in complete.items() if key != missing}
               for missing in complete]
    invalid.append({key: value for key, value in complete.items()
                    if key not in {"geometry_source_sha256", "marker_source_sha256"}})
    invalid.append({**complete, "unexpected_source_sha256": "3" * 64})
    monkeypatch.setattr(
        artifacts, "make_spatial_head", lambda **_: pytest.fail("invalid bindings built a head")
    )
    for incomplete in invalid:
        with pytest.raises(ValueError, match="complete spatial artifact bindings"):
            artifacts.save_spatial_head_artifact(
                head, rejected_dir, role="initial", bindings=incomplete
            )
        assert not list(rejected_dir.iterdir())
        with pytest.raises(ValueError, match="complete spatial artifact bindings"):
            artifacts.load_spatial_head_artifact(
                tmp_path, role="initial", expected_bindings=incomplete,
                expected_manifest_sha256=manifest_sha,
            )


def test_inspector_rejects_tampered_other_role_without_loading_weights(tmp_path, monkeypatch):
    import torch

    head = make_spatial_head(seed=42)
    for role in ("initial", "final"):
        artifacts.save_spatial_head_artifact(head, tmp_path, role=role, bindings=bindings())
    manifest_sha = digest(tmp_path / artifacts.MANIFEST_NAME)
    other_role = tmp_path / "spatial_head_final.pt"
    other_role.write_bytes(other_role.read_bytes() + b"synthetic-tamper")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("inspector loaded weights"))
    monkeypatch.setattr(
        artifacts, "make_spatial_head", lambda **_: pytest.fail("inspector constructed a head")
    )
    with pytest.raises(ValueError, match="weight file/tree SHA"):
        artifacts.inspect_spatial_head_artifact(
            tmp_path, role="initial", expected_bindings=bindings(),
            expected_manifest_sha256=manifest_sha,
        )


def test_rebound_manifest_still_rejects_wrong_parameter_hash_before_head_build(tmp_path, monkeypatch):
    head = make_spatial_head(seed=42)
    artifacts.save_spatial_head_artifact(head, tmp_path, role="initial", bindings=bindings())
    manifest_path = tmp_path / artifacts.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text("utf8"))
    manifest["artifacts"]["initial"]["parameter_sha256"] = "0" * 64
    manifest_path.write_bytes(canonical(manifest) + b"\n")
    manifest_sha = digest(manifest_path)
    inspected = artifacts.inspect_spatial_head_artifact(
        tmp_path, role="initial", expected_bindings=bindings(),
        expected_manifest_sha256=manifest_sha,
    )
    assert inspected["parameter_sha256"] == "0" * 64
    monkeypatch.setattr(
        artifacts, "make_spatial_head", lambda **_: pytest.fail("bad parameter hash built a head")
    )
    with pytest.raises(ValueError, match="parameter SHA"):
        artifacts.load_spatial_head_artifact(
            tmp_path, role="initial", expected_bindings=bindings(),
            expected_manifest_sha256=manifest_sha,
        )


def test_save_and_bound_load_reject_dtype_shape_and_nonfinite_states(tmp_path, monkeypatch):
    import torch

    head = make_spatial_head(seed=42)

    class StateOnly:
        def __init__(self, state):
            self.state = state

        def state_dict(self):
            return self.state

    monkeypatch.setattr(
        artifacts, "make_spatial_head", lambda **_: pytest.fail("invalid state built a head")
    )
    for mutation in ("dtype", "shape", "nan", "inf"):
        directory = tmp_path / mutation
        directory.mkdir()
        state = {name: value.clone() for name, value in head.state_dict().items()}
        name = "refinement_output.bias"
        if mutation == "dtype":
            state[name] = state[name].to(dtype=torch.float16)
        elif mutation == "shape":
            state[name] = state[name][:1]
        else:
            state[name][0] = float(mutation)
        error = "shape or FP32 dtype" if mutation in {"dtype", "shape"} else "nonfinite"
        with pytest.raises(ValueError, match=error):
            artifacts.save_spatial_head_artifact(
                StateOnly(state), directory, role="initial", bindings=bindings()
            )
        assert not list(directory.iterdir())
        artifacts.save_spatial_head_artifact(head, directory, role="initial", bindings=bindings())
        weight_path = directory / "spatial_head_initial.pt"
        torch.save(state, weight_path)
        manifest_path = directory / artifacts.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text("utf8"))
        record = manifest["artifacts"]["initial"]
        record["file_sha256"] = digest(weight_path)
        record["artifact_tree_sha256"] = hashlib.sha256(canonical({
            artifacts.CONFIG_NAME: manifest["config_sha256"],
            record["filename"]: record["file_sha256"],
        })).hexdigest()
        manifest_path.write_bytes(canonical(manifest) + b"\n")
        manifest_sha = digest(manifest_path)
        artifacts.inspect_spatial_head_artifact(
            directory, role="initial", expected_bindings=bindings(),
            expected_manifest_sha256=manifest_sha,
        )
        with pytest.raises(ValueError, match=error):
            artifacts.load_spatial_head_artifact(
                directory, role="initial", expected_bindings=bindings(),
                expected_manifest_sha256=manifest_sha,
            )
