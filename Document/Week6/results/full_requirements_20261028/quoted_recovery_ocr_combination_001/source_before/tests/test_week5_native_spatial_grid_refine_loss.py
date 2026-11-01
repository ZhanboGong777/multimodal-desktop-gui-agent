"""Tensor-only local loss contracts; no head/model/processor construction."""
from __future__ import annotations

import math

import pytest

from gui_agent.finetuning import native_spatial_grid_head as grid
from gui_agent.finetuning.native_spatial_grid_head import grid_refine_loss


@pytest.fixture(autouse=True)
def tensor_only_cpu(monkeypatch):
    import torch

    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("no head or model construction/forward in tensor-only loss QA")

    for name in ("make_spatial_head", "make_head", "prepare_key_cache", "predict_with_cache"):
        monkeypatch.setattr(grid, name, forbidden)
    yield
    assert not torch.cuda.is_initialized()


def geometry(width=100, height=80, *, all_overlap=False, zero_spans=False):
    centres = ((0.3, 0.4), (0.5, 0.6), (0.7, 0.4)) if all_overlap else ((0., 0.), (.5, .5), (1., 1.))
    span = .4 if all_overlap else .2
    rows = tuple(tuple(0. if size == 1 else c for c, size in zip(centre, (width, height), strict=True)) for centre in centres)
    spans = tuple(tuple(0. if size == 1 or zero_spans else span for size in (width, height)) for _ in centres)
    bounds = tuple(tuple(max(0., c - 1.5*s) for c, s in zip(centre, spacing, strict=True))
                   + tuple(min(1., c + 1.5*s) for c, s in zip(centre, spacing, strict=True))
                   for centre, spacing in zip(rows, spans, strict=True))
    return grid.GridFeatures(width, height, 4, (0, 1, 3), rows, spans, bounds, "a" * 64)


def residuals(markers=1):
    import torch

    return torch.zeros((markers, 3, 2), dtype=torch.float32, requires_grad=True)


def labels(rows):
    import torch

    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 2)


def test_geometry_target_poison_cannot_change_input_features():
    clean = geometry().to_dict()
    before = grid.GridFeatures.from_dict(clean)
    with pytest.raises(ValueError, match="input-only"):
        grid.GridFeatures.from_dict({**clean, "target_unit": [0., 1.]})
    assert grid.GridFeatures.from_dict(clean) == before


def test_loss_matches_independent_gaussian_smooth_l1_oracle():
    raw = residuals()
    features = geometry(all_overlap=True)
    target = labels([[.45, .55]])
    result = grid_refine_loss({"all_cell_refinement_logits": raw}, target, features)
    weights, cell_losses = [], []
    for centre, bounds in zip(features.native_unit_centers, features.refinement_bounds, strict=True):
        distance = sum(((c-t)*s)**2 for c, t, s in zip(centre, target[0].tolist(), (99, 79), strict=True))
        weights.append(math.exp(-distance/(2*20**2)))
        fractions = [(float(t)-lo)/(hi-lo) for t, lo, hi in zip(target[0], bounds[:2], bounds[2:], strict=True)]
        errors = [abs(.5-t) for t in fractions]
        cell_losses.append(sum(e*e/(2*.1) if e < .1 else e-.05 for e in errors)/2)
    expected = sum(w*value for w, value in zip(weights, cell_losses, strict=True))/sum(weights)
    assert float(result["loss"].detach()) == pytest.approx(expected, abs=1e-7)
    assert result["diagnostics"]["eligible_cells_per_endpoint"] == [3]
    assert result["diagnostics"]["additional_head_forward_calls"] == 0


def test_gradient_reaches_toy_query_key_and_refinement_branches():
    import torch

    q = torch.tensor([[.1, -.2]], requires_grad=True)
    keys = torch.tensor([[.3, -.2], [-.1, .4], [.2, .1]], requires_grad=True)
    refine_weight = torch.tensor([[.2, -.3], [.4, .1]], requires_grad=True)
    all_cells = (q[:, None] + keys[None]) @ refine_weight
    original = all_cells.detach().clone()
    result = grid_refine_loss({"all_cell_refinement_logits": all_cells}, labels([[.45, .55]]), geometry(all_overlap=True))
    result["loss"].backward()
    for value in (q, keys, refine_weight):
        assert value.grad is not None and torch.isfinite(value.grad).all() and (value.grad != 0).all()
    torch.testing.assert_close(all_cells.detach(), original, rtol=0, atol=0)


def test_gt_changes_only_loss_and_never_mutates_existing_predictions():
    import torch

    raw = residuals()
    prediction = {"all_cell_refinement_logits": raw, "prediction_unit": torch.tensor([[.52, .49]]),
                  "cell_logits": torch.tensor([[1., 2., 3.]])}
    before = {key: value.detach().clone() for key, value in prediction.items()}
    first = grid_refine_loss(prediction, labels([[.1, .1]]), geometry())
    second = grid_refine_loss(prediction, labels([[.9, .9]]), geometry())
    assert first["diagnostics"]["eligible_cells_per_endpoint"] == second["diagnostics"]["eligible_cells_per_endpoint"] == [1]
    for key, value in before.items():
        torch.testing.assert_close(prediction[key], value, rtol=0, atol=0)
    # Asymmetric fixed residual produces different label loss, never a new prediction.
    changed_prediction = {**prediction, "all_cell_refinement_logits": raw.clone()}
    changed_prediction["all_cell_refinement_logits"][0, 0, 0] = .7
    changed = grid_refine_loss(changed_prediction, labels([[.1, .1]]), geometry())
    assert float(changed["loss"].detach()) != float(second["loss"].detach())
    for key in ("prediction_unit", "cell_logits"):
        torch.testing.assert_close(prediction[key], before[key], rtol=0, atol=0)
    assert first["diagnostics"]["label_scope"].startswith("GT after all-cell predictions")


@pytest.mark.parametrize("value,sign,index", [(0., 1, 0), (1., -1, 2)])
def test_native_border_eligibility_and_gradient_direction(value, sign, index):
    raw = residuals()
    result = grid_refine_loss({"all_cell_refinement_logits": raw}, labels([[value, value]]), geometry())
    result["loss"].backward()
    assert (raw.grad[0, index] * sign > 0).all()
    assert (raw.grad[0, [i for i in range(3) if i != index]] == 0).all()
    assert result["diagnostics"]["target_fraction_range"] == [value, value]


@pytest.mark.parametrize("size,target,inactive", [((1, 80), [0., .1], 0), ((100, 1), [.1, 0.], 1)])
def test_native_size_one_axis_is_excluded_without_division_by_zero(size, target, inactive):
    import torch

    raw = residuals()
    result = grid_refine_loss({"all_cell_refinement_logits": raw}, labels([target]), geometry(*size))
    result["loss"].backward()
    assert torch.isfinite(raw.grad).all() and (raw.grad[..., inactive] == 0).all()
    assert raw.grad[..., 1-inactive].abs().sum() > 0


def test_all_axes_degenerate_differentiable_zero():
    raw = residuals(2)
    result = grid_refine_loss({"all_cell_refinement_logits": raw}, labels([[0., 0.], [0., 0.]]), geometry(1, 1))
    assert result["loss"].requires_grad and result["loss"].item() == 0
    result["loss"].backward()
    assert (raw.grad == 0).all()
    assert result["diagnostics"]["all_axes_degenerate"] is True
    assert result["diagnostics"]["eligible_cells_per_endpoint"] == [3, 3]
    assert result["diagnostics"]["target_fraction_range"] is None


def test_zero_markers_differentiable_zero():
    raw = residuals(0)
    result = grid_refine_loss({"all_cell_refinement_logits": raw}, labels([]), geometry())
    result["loss"].backward()
    assert raw.grad.shape == raw.shape and result["loss"].item() == 0
    assert result["diagnostics"]["zero_markers"] is True
    assert result["diagnostics"]["eligible_cells_per_endpoint"] == []


def test_input_zero_span_cell_axes_are_excluded():
    raw = residuals()
    result = grid_refine_loss({"all_cell_refinement_logits": raw}, labels([[.5, .5]]), geometry(zero_spans=True))
    result["loss"].backward()
    assert result["loss"].item() == 0 and (raw.grad == 0).all()
    assert result["diagnostics"]["eligible_cells_per_endpoint"] == [1]


def test_no_eligible_cell_fails_closed():
    with pytest.raises(ValueError, match="eligible static cell"):
        grid_refine_loss({"all_cell_refinement_logits": residuals()}, labels([[0., 1.]]), geometry())


@pytest.mark.parametrize("kind", ["nan", "inf", "float64", "shape"])
def test_invalid_residual_fails_closed(kind):
    import torch

    raw = residuals()
    if kind in ("nan", "inf"):
        raw = torch.full((1, 3, 2), float(kind))
    elif kind == "float64":
        raw = raw.double()
    else:
        raw = raw[:, :2]
    with pytest.raises(ValueError, match="all-cell residual"):
        grid_refine_loss({"all_cell_refinement_logits": raw}, labels([[.1, .1]]), geometry())


@pytest.mark.parametrize("kind", ["nan", "out_of_range", "float64", "shape", "requires_grad", "degenerate"])
def test_invalid_label_fails_closed(kind):
    target, features = labels([[.1, .1]]), geometry()
    if kind == "nan":
        target[0, 0] = float("nan")
    elif kind == "out_of_range":
        target[0, 1] = 1.001
    elif kind == "float64":
        target = target.double()
    elif kind == "shape":
        target = target.repeat(2, 1)
    elif kind == "requires_grad":
        target.requires_grad_()
    else:
        features = geometry(1, 80)
    with pytest.raises(ValueError, match="labels|label must be zero"):
        grid_refine_loss({"all_cell_refinement_logits": residuals()}, target, features)
