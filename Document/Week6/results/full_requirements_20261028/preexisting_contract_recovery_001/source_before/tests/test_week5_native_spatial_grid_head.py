"""Tiny synthetic math checks; no actual images, models, processors or quality claim."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gui_agent.finetuning import native_spatial_grid_head as grid


@pytest.fixture
def torch_cpu(monkeypatch):
    import torch

    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    yield torch
    assert not torch.cuda.is_initialized()


def recipe(width=100, height=80):
    centres = ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0))
    spans = tuple(0.0 if size == 1 else 0.2 for size in (width, height))
    valid_indices = (0, 1, 3)
    tokens = [{"token_index": i, "valid": i in valid_indices} for i in range(4)]
    for index, centre in zip(valid_indices, centres):
        tokens[index].update(native_unit_center=[0.0 if size == 1 else c for c, size in zip(centre, (width, height))], native_unit_span=list(spans))
    return {"schema": grid.GEOMETRY_SCHEMA, "native_size": [width, height], "token_order": grid.TOKEN_ORDER,
            "token_count": 4, "tokens": tokens, "geometry_input_sha256": "a" * 64}


def features(width=100, height=80):
    return grid.GridFeatures.from_recipe(recipe(width, height))


def test_lazy_import_contract_and_input_only_geometry():
    command = ("import importlib,sys;sys.path.insert(0,sys.argv[1]);"
               "importlib.import_module('gui_agent.finetuning.native_spatial_grid_head');"
               "assert not ({'torch','transformers','PIL'} & sys.modules.keys())")
    result = subprocess.run([sys.executable, "-c", command, str(Path(grid.__file__).resolve().parents[2])],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    class Poisoned(dict):
        def __getitem__(self, key):
            assert key not in {"actions", "target_unit", "source_annotation"}
            return super().__getitem__(key)

    value = Poisoned(**recipe(), actions=object(), target_unit=object(), source_annotation=object())
    clean = grid.GridFeatures.from_recipe(value)
    assert clean == grid.GridFeatures.from_dict(clean.to_dict())
    assert clean.features_sha256 == grid.GridFeatures.from_dict(json.loads(json.dumps(clean.to_dict()))).features_sha256
    with pytest.raises(ValueError, match="input-only"):
        grid.GridFeatures.from_dict({**clean.to_dict(), "target_unit": [0.1, 0.2]})
    value["tokens"] = [{"token_index": i, "valid": False} for i in range(4)]
    with pytest.raises(ValueError, match="nonempty"):
        grid.GridFeatures.from_recipe(value)
    config = grid.head_configuration()
    assert config["target_protocol"] == "native_spatial_grid_head"
    assert config["target_version"] == "week5.native-spatial-target.v1"
    assert config["grid_auxiliary_weight"] == 0.5 and config["grid_auxiliary_sigma_pixels"] == 20


def test_seed_shape_cpu_only_and_cuda_seed_spy(torch_cpu, monkeypatch):
    torch = torch_cpu
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args, **kwargs: pytest.fail("CUDA seed forbidden"))
    torch.random.default_generator.manual_seed(123)
    before = torch.get_rng_state().clone()
    a, b = grid.make_head(seed=42), grid.make_spatial_head(seed=42)
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert {key: tuple(value.shape) for key, value in a.state_dict().items()} == grid.STATE_SHAPES
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in a.parameters())
    assert torch.equal(a.refinement_output.bias, torch.zeros(2))


def test_live_query_key_local_refinement_and_masked_weights(torch_cpu):
    torch = torch_cpu
    head = grid.make_head(seed=42)
    image = torch.randn(4, 2048, generator=torch.Generator().manual_seed(20)).requires_grad_()
    with torch.no_grad():
        image[2] = float("nan")  # Masked region is never used as a key or cell.
    query = torch.randn(2, 2048, generator=torch.Generator().manual_seed(21)).requires_grad_()
    geometry = features()
    cache = grid.prepare_key_cache(head, image, geometry)
    result = grid.predict_with_cache(head, query, cache)
    assert result["cell_logits"].shape == (2, 3) and result["all_cell_unit_xy"].shape == (2, 3, 2)
    assert (result["all_token_attention_weights"][:, 2] == 0).all()
    torch.testing.assert_close(result["attention_weights"].sum(-1), torch.ones(2))
    labels = torch.tensor([[0.7, 0.2], [0.15, 0.9]], dtype=torch.float32)
    before = {key: result[key].detach().clone() for key in ("prediction_unit", "cell_logits", "all_cell_unit_xy")}
    auxiliary = grid.grid_auxiliary_loss(result, labels, geometry)
    point = grid.point_loss(result["prediction_unit"], labels, native_width=100, native_height=80)
    (point + auxiliary * 0.5).backward()
    assert float(query.grad.norm()) > 0 and float(image.grad[[0, 1, 3]].norm()) > 0
    assert (image.grad[2] == 0).all()
    for layer in (head.query_projection, head.key_projection, head.position_projection, head.refinement_hidden, head.refinement_output):
        assert float(layer.weight.grad.norm()) > 0
    for key, value in before.items():
        torch.testing.assert_close(result[key].detach(), value, rtol=0, atol=0)


@pytest.mark.parametrize("size", [(100, 80), (1, 80), (100, 1), (1, 1)])
def test_native_border_degenerate_axis_and_stdlib_reconstruction(torch_cpu, size):
    torch = torch_cpu
    head = grid.make_head(seed=7)
    image = torch.randn(4, 2048, generator=torch.Generator().manual_seed(12))
    query = torch.randn(2, 2048, generator=torch.Generator().manual_seed(13))
    geometry = features(*size)
    result = grid.predict_with_cache(head, query, grid.prepare_key_cache(head, image, geometry))
    reference = grid.reconstruct_prediction(result["cell_logits"].detach().tolist(), result["all_cell_refinement_logits"].detach().tolist(), geometry)
    torch.testing.assert_close(result["prediction_unit"], torch.tensor(reference["prediction_unit"]), rtol=0, atol=1e-6)
    torch.testing.assert_close(result["all_cell_unit_xy"], torch.tensor(reference["all_cell_unit_xy"]), rtol=0, atol=1e-6)
    assert ((result["prediction_unit"] >= 0) & (result["prediction_unit"] <= 1)).all()
    for axis, bound in enumerate(size):
        if bound == 1:
            assert (result["prediction_unit"][:, axis] == 0).all()
    extreme = grid.reconstruct_prediction([[1e4, -1e4, 0]], [[[1000, -1000], [0, 0], [-1000, 1000]]], geometry)
    assert all(0 <= v <= 1 for v in extreme["prediction_unit"][0])
    labels = torch.tensor([[0.0 if size[0] == 1 else 0.4, 0.0 if size[1] == 1 else 0.6]] * 2)
    assert torch.isfinite(grid.grid_auxiliary_loss(result, labels, geometry))


def test_cached_batch_parity_and_old_parameter_cache_refusal(torch_cpu):
    torch = torch_cpu
    head = grid.make_head(seed=42)
    image = torch.randn(4, 2048, generator=torch.Generator().manual_seed(25))
    query = torch.randn(3, 2048, generator=torch.Generator().manual_seed(26))
    cache = grid.prepare_key_cache(head, image, features())
    batch = grid.predict_with_cache(head, query, cache)
    separate = [grid.predict_with_cache(head, query[i:i + 1], cache)["prediction_unit"] for i in range(3)]
    torch.testing.assert_close(batch["prediction_unit"], torch.cat(separate), rtol=0, atol=1e-6)
    with torch.no_grad():
        head.query_projection.weight.add_(1e-3)
    with pytest.raises(ValueError, match="unmodified"):
        grid.predict_with_cache(head, query, cache)
