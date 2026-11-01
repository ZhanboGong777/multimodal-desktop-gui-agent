"""Private grid-head hypothesis: lazy Torch, no model/processor/image/UI access.

All geometry and caches are input-only. Training labels enter auxiliary loss
after all-cell predictions. CPU contracts are not learned capacity evidence.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import native_spatial_head as marker_core
from .native_spatial_grid_geometry import build_recipe
from .native_spatial_head import (
    canonical,
    generate_with_cache_contract,
    generation_policy,
    generation_policy_binding,
    marker_token_positions,
    materialize_semantic_actions,
    point_loss,
    require,
    resolve_final_norm,
    semantic_marker_spans,
    strict_json,
)

TARGET_PROTOCOL = "native_spatial_grid_head"
TARGET_VERSION = "week5.native-spatial-target.v1"
HEAD_VERSION = "week5.native-spatial-grid-head.qk128-cellrefine.v2"
HIDDEN_SIZE = 2048
PROJECTION_SIZE = 128
FEATURE_NORM_EPS = 1e-5
REFINEMENT_FACTOR = 1.5
OUTPUT_INIT_STD = 1e-3
GAUSSIAN_SIGMA_PIXELS = 20.0
GRID_REFINE_LOSS_WEIGHT = 0.5
GRID_REFINE_LOSS_BETA = 0.1
GRID_REFINE_TARGET = (
    "GT only after all-cell predictions; eligible valid cells whose input-static bounds contain target; "
    "Gaussian native-anchor distance sigma20px renormalized over eligible cells; "
    "sigmoid residual versus (target-lower)/(upper-lower); mean SmoothL1 beta0.1 over "
    "nonzero-span native axes per cell and endpoints; size1 target0; no eligible cell fails closed"
)
FEATURE_SCHEMA = "week5.native_grid_head.input_features.v1"
GEOMETRY_SCHEMA = "week5.native_grid_geometry.recipe.v1"
TOKEN_ORDER = "time, merged_row, merged_col; raster after Qwen inverse window_index"
REFINEMENT_POLICY = "native_center +/- 1.5 nominal cell spans intersect [0,axis-1]; input-only static domain v1"
STATE_SHAPES = {
    "query_projection.weight": (128, 2048), "query_projection.bias": (128,),
    "key_projection.weight": (128, 2048), "key_projection.bias": (128,),
    "position_projection.weight": (128, 4), "position_projection.bias": (128,),
    "refinement_hidden.weight": (128, 260), "refinement_hidden.bias": (128,),
    "refinement_output.weight": (2, 128), "refinement_output.bias": (2,),
}
__all__ = [
    "HEAD_VERSION", "HIDDEN_SIZE", "STATE_SHAPES", "TARGET_PROTOCOL", "TARGET_VERSION",
    "FrozenFeatures", "GridFeatures", "KeyCache", "canonical", "generate_with_cache_contract",
    "generation_policy", "generation_policy_binding", "grid_auxiliary_loss", "grid_refine_loss", "head_configuration",
    "make_head", "make_spatial_head", "marker_token_positions", "materialize_semantic_actions",
    "point_loss", "predict_with_cache", "prepare_key_cache", "reconstruct_prediction",
    "require", "resolve_final_norm", "semantic_marker_spans", "strict_json",
]


def _finite(values, length, name):
    require(isinstance(values, (list, tuple)) and len(values) == length
            and all(type(v) in (int, float) and math.isfinite(v) for v in values), f"finite {name} required")
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class GridFeatures:
    """Immutable input geometry; never stores targets, query text or selected GT."""
    native_width: int
    native_height: int
    total_token_count: int
    valid_indices: tuple[int, ...]
    native_unit_centers: tuple[tuple[float, float], ...]
    native_unit_spans: tuple[tuple[float, float], ...]
    refinement_bounds: tuple[tuple[float, float, float, float], ...]
    recipe_input_sha256: str
    token_order: str = TOKEN_ORDER

    def __post_init__(self):
        require(all(type(v) is int and v > 0 for v in (self.native_width, self.native_height, self.total_token_count)), "positive native geometry required")
        require(type(self.valid_indices) is tuple and bool(self.valid_indices)
                and all(type(i) is int and 0 <= i < self.total_token_count for i in self.valid_indices)
                and tuple(sorted(set(self.valid_indices))) == self.valid_indices, "nonempty unique ordered valid indices required")
        count = len(self.valid_indices)
        require(all(type(rows) is tuple and len(rows) == count for rows in
                    (self.native_unit_centers, self.native_unit_spans, self.refinement_bounds)), "immutable aligned geometry required")
        for centre, span, bounds in zip(self.native_unit_centers, self.native_unit_spans, self.refinement_bounds, strict=True):
            require(type(centre) is type(span) is type(bounds) is tuple, "immutable geometry rows required")
            c, s, b = _finite(centre, 2, "centre"), _finite(span, 2, "span"), _finite(bounds, 4, "bounds")
            require(all(0 <= v <= 1 for v in c) and all(v >= 0 for v in s), "input centre/span outside native domain")
            for axis, size in enumerate((self.native_width, self.native_height)):
                expected = (0.0, 0.0) if size == 1 else (max(0.0, c[axis] - REFINEMENT_FACTOR * s[axis]), min(1.0, c[axis] + REFINEMENT_FACTOR * s[axis]))
                require((b[axis], b[axis + 2]) == expected, "derived 1.5-cell static bounds differ")
                if size == 1:
                    require(c[axis] == s[axis] == 0, "degenerate geometry axis must be zero")
        require(self.token_order == TOKEN_ORDER and isinstance(self.recipe_input_sha256, str)
                and len(self.recipe_input_sha256) == 64 and all(ch in "0123456789abcdef" for ch in self.recipe_input_sha256), "explicit recipe/order authority required")

    @classmethod
    def from_recipe(cls, recipe: Mapping):
        require(recipe["schema"] == GEOMETRY_SCHEMA and recipe["token_order"] == TOKEN_ORDER, "approved input recipe contract required")
        width, height = recipe["native_size"]
        count, tokens = recipe["token_count"], recipe["tokens"]
        require(type(count) is int and isinstance(tokens, list) and len(tokens) == count, "complete raster token ledger required")
        indices, centres, spans, bounds = [], [], [], []
        for index, token in enumerate(tokens):
            require(type(token["token_index"]) is int and token["token_index"] == index and type(token["valid"]) is bool, "exact token order/mask required")
            if not token["valid"]:
                continue
            centre = _finite(token["native_unit_center"], 2, "centre")
            span = _finite(token["native_unit_span"], 2, "span")
            lower = tuple(0.0 if size == 1 else max(0.0, c - REFINEMENT_FACTOR * s) for c, s, size in zip(centre, span, (width, height), strict=True))
            upper = tuple(0.0 if size == 1 else min(1.0, c + REFINEMENT_FACTOR * s) for c, s, size in zip(centre, span, (width, height), strict=True))
            indices.append(index)
            centres.append(centre)
            spans.append(span)
            bounds.append((*lower, *upper))
        return cls(width, height, count, tuple(indices), tuple(centres), tuple(spans), tuple(bounds), recipe["geometry_input_sha256"])

    def to_dict(self):
        return {"schema": FEATURE_SCHEMA, "native_width": self.native_width, "native_height": self.native_height,
                "total_token_count": self.total_token_count, "valid_indices": list(self.valid_indices),
                "native_unit_centers": [list(v) for v in self.native_unit_centers], "native_unit_spans": [list(v) for v in self.native_unit_spans],
                "refinement_bounds": [list(v) for v in self.refinement_bounds], "recipe_input_sha256": self.recipe_input_sha256,
                "token_order": self.token_order, "refinement_factor": REFINEMENT_FACTOR, "refinement_policy": REFINEMENT_POLICY}

    @classmethod
    def from_dict(cls, value):
        expected = {"schema", "native_width", "native_height", "total_token_count", "valid_indices", "native_unit_centers",
                    "native_unit_spans", "refinement_bounds", "recipe_input_sha256", "token_order", "refinement_factor", "refinement_policy"}
        require(isinstance(value, dict) and set(value) == expected and value["schema"] == FEATURE_SCHEMA
                and value["refinement_factor"] == REFINEMENT_FACTOR and value["refinement_policy"] == REFINEMENT_POLICY, "exact input-only feature schema required")
        return cls(value["native_width"], value["native_height"], value["total_token_count"], tuple(value["valid_indices"]),
                   tuple(tuple(v) for v in value["native_unit_centers"]), tuple(tuple(v) for v in value["native_unit_spans"]),
                   tuple(tuple(v) for v in value["refinement_bounds"]), value["recipe_input_sha256"], value["token_order"])

    @property
    def features_sha256(self):
        return hashlib.sha256(canonical(self.to_dict()).encode()).hexdigest()

    @property
    def token_count(self):
        return self.total_token_count

    @property
    def native_centers(self):
        return tuple((x * (self.native_width - 1), y * (self.native_height - 1)) for x, y in self.native_unit_centers)


FrozenFeatures = GridFeatures


@dataclass(frozen=True)
class KeyCache:
    features: GridFeatures
    projected_keys: Any
    position_features: Any
    head_identity: int
    parameter_versions: tuple[int, ...]


def head_configuration():
    return {"target_protocol": TARGET_PROTOCOL, "target_version": TARGET_VERSION, "head_version": HEAD_VERSION,
            "hidden_size": HIDDEN_SIZE, "projection_size": PROJECTION_SIZE, "parameter_dtype": "float32",
            "feature_normalization": {"kind": "per-token nonaffine LayerNorm", "eps": FEATURE_NORM_EPS, "live_autograd": True},
            "attention": "L2 normalized Q/K; sqrt(128)*cosine, equivalent RMS-normalized scaled dot product; softmax all input-valid cells",
            "position_features": "input-only native unit centre_xy and span_xy; Linear4->128 added to projected image key",
            "refinement": "all valid cells independently concat(query128,key128,pos4)->Linear260x128->GELU->Linear128x2->sigmoid, lower+(upper-lower)*sigmoid",
            "refinement_bounds_policy": REFINEMENT_POLICY, "refinement_factor": REFINEMENT_FACTOR,
            "refinement_output_init": {"weight": "normal", "std": OUTPUT_INIT_STD, "mean": 0.0, "bias": "zero"},
            "initialization_device": "cpu", "initialization_rng": "isolated CPU default_generator; CUDA RNG untouched",
            "semantic_ce_weight": 1.0, "point_loss_weight": 1.0, "grid_auxiliary_weight": 0.5, "class_weights": 1.0,
            "grid_auxiliary": "label-only per endpoint Gaussian soft target on all valid native centres; stable soft-target CE",
            "grid_refine_loss_weight": GRID_REFINE_LOSS_WEIGHT, "grid_refine_loss_beta": GRID_REFINE_LOSS_BETA,
            "grid_refine_target": GRID_REFINE_TARGET,
            "grid_auxiliary_sigma_pixels": GAUSSIAN_SIGMA_PIXELS, "point_loss": "original mean endpoint/axis SmoothL1 normalized native xy; beta20/max(1,axis-1)",
            "degenerate_axis": "all cell bounds and mixture output zero", "head_learning_rate": 1e-3, "lora_learning_rate": 1e-4,
            "geometry_source": "caller-pinned input-only geometry_recipe.build_recipe", "marker_source": "unchanged semantic_marker_spans/marker_token_positions/materialize_semantic_actions",
            "key_state_source": "same forward final language norm image-prefix states in complete inverse-window raster order",
            "marker_state_source": "consumed final norm state for token containing each marker last content byte",
            "cache_call_contract": "train live keys per same Transformer forward; infer projected keys once prefill, one batched marker head prediction after generation, no extra HF/repair forward",
            "numeric_serialization": "raw FP32 logits/residuals; authoritative shared stdlib float64 softmax/sigmoid mixture for native rounding; actual FP32 mixture audited within1e-6",
            "scope": "new grid-head hypothesis; no capacity/quality proof"}


def make_spatial_head(*, seed=42):
    import torch
    from torch import nn

    require(type(seed) is int and seed >= 0, "nonnegative CPU initialization seed required")

    class GridHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_projection = nn.Linear(2048, 128, dtype=torch.float32, device="cpu")
            self.key_projection = nn.Linear(2048, 128, dtype=torch.float32, device="cpu")
            self.position_projection = nn.Linear(4, 128, dtype=torch.float32, device="cpu")
            self.refinement_hidden = nn.Linear(260, 128, dtype=torch.float32, device="cpu")
            self.refinement_output = nn.Linear(128, 2, dtype=torch.float32, device="cpu")
            nn.init.normal_(self.refinement_output.weight, mean=0.0, std=OUTPUT_INIT_STD)
            nn.init.zeros_(self.refinement_output.bias)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = GridHead()
    require({name: tuple(v.shape) for name, v in head.state_dict().items()} == STATE_SHAPES, "grid artifact architecture differs")
    return head


make_head = make_spatial_head


def _checked_states(head, states, count):
    import torch

    require(isinstance(states, torch.Tensor) and states.ndim == 2 and tuple(states.shape) == (count, HIDDEN_SIZE)
            and states.is_floating_point(), "aligned live Nx2048 norm states required")
    require({name: tuple(v.shape) for name, v in head.state_dict().items()} == STATE_SHAPES
            and all(p.dtype == torch.float32 and p.device == states.device for p in head.parameters()), "FP32 same-device registered head required")


def _rms_normalized_projection(values):
    from torch.nn import functional as F

    return F.normalize(values, p=2, dim=-1, eps=FEATURE_NORM_EPS) * math.sqrt(PROJECTION_SIZE)


def prepare_key_cache(head, image_states, input_geometry):
    import torch
    from torch.nn import functional as F

    require(isinstance(input_geometry, GridFeatures), "typed input-only GridFeatures required")
    _checked_states(head, image_states, input_geometry.total_token_count)
    index = torch.tensor(input_geometry.valid_indices, dtype=torch.long, device=image_states.device)
    selected = image_states.index_select(0, index).float()
    require(bool(torch.isfinite(selected).all()), "finite input-valid image states required")
    with torch.autocast(device_type=image_states.device.type, enabled=False):
        features = F.layer_norm(selected, (HIDDEN_SIZE,), weight=None, bias=None, eps=FEATURE_NORM_EPS)
        position = torch.tensor([(*c, *s) for c, s in zip(input_geometry.native_unit_centers, input_geometry.native_unit_spans, strict=True)], dtype=torch.float32, device=image_states.device)
        keys = _rms_normalized_projection(head.key_projection(features) + head.position_projection(position))
    require(bool(torch.isfinite(keys).all()), "finite live projected keys required")
    return KeyCache(input_geometry, keys, position, id(head), tuple(p._version for p in head.parameters()))


def predict_with_cache(head, marker_states, cache):
    import torch
    from torch.nn import functional as F

    require(isinstance(cache, KeyCache) and cache.head_identity == id(head)
            and cache.parameter_versions == tuple(p._version for p in head.parameters()), "same unmodified head/key cache required")
    require(isinstance(marker_states, torch.Tensor) and marker_states.ndim == 2, "marker state tensor required")
    count = marker_states.shape[0]
    _checked_states(head, marker_states, count)
    require(marker_states.device == cache.projected_keys.device and bool(torch.isfinite(marker_states).all()), "finite same-device marker states required")
    cell_count = len(cache.features.valid_indices)
    with torch.autocast(device_type=marker_states.device.type, enabled=False):
        normalized = F.layer_norm(marker_states.float(), (HIDDEN_SIZE,), weight=None, bias=None, eps=FEATURE_NORM_EPS)
        query = _rms_normalized_projection(head.query_projection(normalized))
        logits = query @ cache.projected_keys.T / math.sqrt(PROJECTION_SIZE)
        attention = torch.softmax(logits, dim=-1)
        q = query[:, None, :].expand(count, cell_count, PROJECTION_SIZE)
        k = cache.projected_keys[None, :, :].expand(count, cell_count, PROJECTION_SIZE)
        p = cache.position_features[None, :, :].expand(count, cell_count, 4)
        refinement = head.refinement_output(F.gelu(head.refinement_hidden(torch.cat((q, k, p), dim=-1))))
        bounds = torch.tensor(cache.features.refinement_bounds, dtype=torch.float32, device=marker_states.device)
        cell_outputs = bounds[None, :, :2] + (bounds[None, :, 2:] - bounds[None, :, :2]) * torch.sigmoid(refinement)
        prediction = (attention[..., None] * cell_outputs).sum(dim=1)
        full_weights = torch.zeros(count, cache.features.total_token_count, dtype=torch.float32, device=marker_states.device)
        full_weights = full_weights.index_copy(1, torch.tensor(cache.features.valid_indices, dtype=torch.long, device=marker_states.device), attention)
    require(all(bool(torch.isfinite(t).all()) for t in (prediction, logits, refinement, cell_outputs, attention)), "finite grid joint outputs required")
    diagnostics = {"head_version": HEAD_VERSION, "marker_count": count, "valid_cell_count": cell_count,
                   "raw_query_rms": marker_states.detach().float().square().mean(-1).sqrt().tolist(),
                   "normalized_query_rms": normalized.detach().square().mean(-1).sqrt().tolist(),
                   "cell_logit_min": float(logits.detach().min()) if logits.numel() else None,
                   "cell_logit_max": float(logits.detach().max()) if logits.numel() else None,
                   "refinement_logit_min": float(refinement.detach().min()) if refinement.numel() else None,
                   "refinement_logit_max": float(refinement.detach().max()) if refinement.numel() else None,
                   "prepare_key_cache_calls": 1, "batched_marker_prediction_calls": 1,
                   "diagnostic_head_forward_calls": 0, "diagnostic_backward_calls": 0,
                   "all_cell_predictions_before_loss": True}
    return {"prediction_unit": prediction, "cell_logits": logits, "all_cell_refinement_logits": refinement,
            "all_cell_unit_xy": cell_outputs, "attention_weights": attention,
            "all_token_attention_weights": full_weights, "valid_cell_indices": cache.features.valid_indices, "diagnostics": diagnostics}


def grid_refine_loss(prediction_result, target_unit, features):
    """Labels supervise existing all-cell residuals inside static input-only bounds."""
    import torch
    from torch.nn import functional as F

    require(isinstance(features, GridFeatures), "typed input geometry required for label-only refinement loss")
    residual = prediction_result["all_cell_refinement_logits"]
    require(isinstance(residual, torch.Tensor) and residual.ndim == 3
            and residual.dtype == torch.float32 and residual.shape[1:] == (len(features.valid_indices), 2)
            and bool(torch.isfinite(residual).all()), "finite FP32 all-cell residual logits required")
    require(isinstance(target_unit, torch.Tensor) and target_unit.dtype == torch.float32
            and not target_unit.requires_grad and target_unit.shape == (residual.shape[0], 2)
            and target_unit.device == residual.device and bool(torch.isfinite(target_unit).all())
            and bool(((target_unit >= 0) & (target_unit <= 1)).all()), "fixed FP32 endpoint labels required")
    require(features.native_width != 1 or bool((target_unit[:, 0] == 0).all()), "degenerate x label must be zero")
    require(features.native_height != 1 or bool((target_unit[:, 1] == 0).all()), "degenerate y label must be zero")
    markers, cells = residual.shape[:2]
    native_active = [features.native_width > 1, features.native_height > 1]
    with torch.autocast(device_type=residual.device.type, enabled=False):
        bounds = torch.tensor(features.refinement_bounds, dtype=torch.float32, device=residual.device)
        lower, upper = bounds[:, :2], bounds[:, 2:]
        span = upper - lower
        eligible = ((target_unit[:, None] >= lower[None])
                    & (target_unit[:, None] <= upper[None])).all(-1)
        counts = eligible.sum(-1)
        require(bool((counts > 0).all()), "every endpoint requires an eligible static cell")
        active = (span > 0) & torch.tensor(native_active, dtype=torch.bool, device=residual.device)
        supervised = eligible[..., None] & active[None]
        fractions = torch.sigmoid(residual)
        safe_span = torch.where(active, span, torch.ones_like(span))
        local_target = torch.where(supervised, (target_unit[:, None] - lower[None]) / safe_span[None],
                                   torch.zeros_like(residual))
        require(bool(torch.isfinite(local_target).all())
                and bool(((local_target[supervised] >= 0) & (local_target[supervised] <= 1)).all()),
                "eligible local fractions must stay inside static bounds")
        if markers and bool(active.any()):
            centres = torch.tensor(features.native_unit_centers, dtype=torch.float32, device=residual.device)
            scale = torch.tensor([features.native_width - 1, features.native_height - 1],
                                 dtype=torch.float32, device=residual.device)
            distance = ((centres[None] - target_unit[:, None]) * scale).square().sum(-1)
            scores = (-distance / (2 * GAUSSIAN_SIGMA_PIXELS ** 2)).masked_fill(~eligible, -torch.inf)
            weights = torch.softmax(scores, dim=-1).detach()
            per_axis = F.smooth_l1_loss(fractions, local_target, beta=GRID_REFINE_LOSS_BETA, reduction="none")
            axis_count = active.sum(-1).clamp_min(1)
            per_cell = (per_axis * active[None]).sum(-1) / axis_count[None]
            per_endpoint = (weights * per_cell).sum(-1)
            loss = per_endpoint.mean()
        else:
            loss = residual.sum() * 0
            per_endpoint = residual.sum(dim=(1, 2)) * 0
    require(bool(torch.isfinite(loss)), "finite local refinement loss required")

    def value_range(values):
        values = values.detach()
        return [float(values.min()), float(values.max())] if values.numel() else None

    diagnostics = {
        "schema": "week5.native_spatial_grid.label_only_refine_diagnostics.v1",
        "marker_count": markers, "valid_cell_count": cells,
        "eligible_cells_per_endpoint": counts.detach().cpu().tolist(),
        "eligible_cells_total": int(counts.detach().sum()),
        "active_native_axes": native_active,
        "zero_span_cell_axes": int((~active).sum()),
        "local_loss_per_endpoint": per_endpoint.detach().cpu().tolist(),
        "target_fraction_range": value_range(local_target[supervised]),
        "eligible_prediction_fraction_range": value_range(fractions[supervised]),
        "all_cell_prediction_fraction_range": value_range(fractions),
        "zero_markers": markers == 0, "all_axes_degenerate": not any(native_active),
        "additional_model_forward_calls": 0, "additional_head_forward_calls": 0,
        "additional_backward_calls": 0,
        "label_scope": "GT after all-cell predictions; input-static eligibility; Gaussian sigma20px renormalized over eligible cells",
    }
    return {"loss": loss, "diagnostics": diagnostics}


def grid_auxiliary_loss(prediction_result, target_unit, features):
    import torch
    from torch.nn import functional as F

    require(isinstance(features, GridFeatures), "typed input geometry required for label-only auxiliary loss")
    logits = prediction_result["cell_logits"]
    require(isinstance(target_unit, torch.Tensor) and target_unit.dtype == torch.float32 and not target_unit.requires_grad
            and target_unit.shape == (logits.shape[0], 2) and target_unit.device == logits.device
            and bool(torch.isfinite(target_unit).all()) and bool(((target_unit >= 0) & (target_unit <= 1)).all()), "fixed FP32 endpoint labels required")
    require(logits.ndim == 2 and logits.dtype == torch.float32 and logits.shape[1] == len(features.valid_indices)
            and bool(torch.isfinite(logits).all()), "finite FP32 all-valid-cell logits required")
    scale = torch.tensor([features.native_width - 1, features.native_height - 1], dtype=torch.float32, device=logits.device)
    require(features.native_width != 1 or bool((target_unit[:, 0] == 0).all()), "degenerate x label must be zero")
    require(features.native_height != 1 or bool((target_unit[:, 1] == 0).all()), "degenerate y label must be zero")
    if not logits.shape[0]:
        return logits.sum() * 0
    centres = torch.tensor(features.native_unit_centers, dtype=torch.float32, device=logits.device)
    distance = ((centres[None] - target_unit[:, None]) * scale).square().sum(-1)
    distribution = torch.softmax(-distance / (2 * GAUSSIAN_SIGMA_PIXELS ** 2), dim=-1).detach()
    return -(distribution * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def reconstruct_prediction(cell_logits: Sequence, residual_logits: Sequence, features: GridFeatures):
    """Authoritative raw-logit serialization math, pure stdlib, no labels/clip."""
    require(isinstance(features, GridFeatures) and isinstance(cell_logits, (list, tuple))
            and isinstance(residual_logits, (list, tuple)) and len(cell_logits) == len(residual_logits), "aligned raw marker arrays required")
    predictions, outputs, attentions, all_weights = [], [], [], []
    cells = len(features.valid_indices)
    for raw, residual in zip(cell_logits, residual_logits, strict=True):
        scores = _finite(raw, cells, "cell logits")
        require(isinstance(residual, (list, tuple)) and len(residual) == cells, "all-cell raw residuals required")
        exponentials = [math.exp(score - max(scores)) for score in scores]
        denominator = math.fsum(exponentials)
        weights = [value / denominator for value in exponentials]
        coords = []
        for values, bounds in zip(residual, features.refinement_bounds, strict=True):
            logits = _finite(values, 2, "refinement logits")
            sigmoid = [1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value)) for value in logits]
            coords.append([bounds[axis] + (bounds[axis + 2] - bounds[axis]) * sigmoid[axis] for axis in (0, 1)])
        prediction = [math.fsum(weight * xy[axis] for weight, xy in zip(exponentials, coords, strict=True)) / denominator for axis in (0, 1)]
        require(all(0 <= v <= 1 and math.isfinite(v) for v in prediction), "bounded raw mixture required")
        full = [0.0] * features.total_token_count
        for index, weight in zip(features.valid_indices, weights, strict=True):
            full[index] = weight
        predictions.append(prediction)
        outputs.append(coords)
        attentions.append(weights)
        all_weights.append(full)
    return {"prediction_unit": predictions, "all_cell_unit_xy": outputs, "attention_weights": attentions, "all_token_attention_weights": all_weights,
            "valid_cell_indices": list(features.valid_indices), "features_sha256": features.features_sha256}


dimensions = marker_core.dimensions


def semantic_target(actions, *, native_width, native_height):
    target = marker_core.semantic_target(actions, native_width=native_width, native_height=native_height)
    return {**target, "target_protocol": TARGET_PROTOCOL}


def objective_contract():
    return {**head_configuration(), "semantic_ce": "complete assistant semantic JSON, end token and newline",
            "label_mask": "expanded prefix only", "manual_CE_calls": 1,
            "model_forward_calls": 1, "default_model_CE_requested": False,
            "auxiliary_label_scope": "GT only after complete all-cell predictions; no GT-selected key"}


@dataclass(frozen=True)
class GridSpatialPlan(marker_core.SpatialPlan):
    image_positions: tuple[int, ...]
    features: GridFeatures


def training_plan(*, full_ids, prefix_ids, labels, target: Mapping, decoder,
                  metadata, grid_thw, image_token_id, patch_size=14, merge_size=2):
    require(target["target_protocol"] == TARGET_PROTOCOL and target["target_version"] == TARGET_VERSION,
            "explicit grid spatial target protocol/version required")
    semantic = marker_core.training_plan(
        full_ids=full_ids, prefix_ids=prefix_ids, labels=labels,
        target={**target, "target_protocol": marker_core.TARGET_PROTOCOL}, decoder=decoder,
    )
    require(type(image_token_id) is int and image_token_id >= 0,
            "explicit model-bound image placeholder ID required")
    recipe = build_recipe(metadata, grid_thw, patch_size=patch_size, merge_size=merge_size)
    require(tuple(recipe["native_size"]) == (semantic.native_width, semantic.native_height),
            "input geometry and target dimensions differ")
    positions = tuple(index for index, token in enumerate(semantic.full_ids[:semantic.prefix_length])
                      if token == image_token_id)
    require(len(positions) == recipe["token_count"] and bool(positions)
            and positions == tuple(range(positions[0], positions[0] + len(positions)))
            and all(position < semantic.prefix_length for position in positions)
            and not any(token == image_token_id for token in semantic.full_ids[semantic.prefix_length:]),
            "all image keys must match the single input-prefix raster grid")
    features = GridFeatures.from_recipe(recipe)
    return GridSpatialPlan(
        semantic.full_ids, semantic.labels, semantic.prefix_length, semantic.marker_positions,
        semantic.target_unit, semantic.slot_keys, semantic.native_width, semantic.native_height,
        semantic.supervised_tokens, positions, features,
    )


def forward_objective(model, head, batch: Mapping, plan: GridSpatialPlan):
    """One model forward: CE1 + point1 + cell auxiliary .5 + local refinement .5."""
    import torch
    from torch.nn import functional as F

    require(isinstance(plan, GridSpatialPlan) and {"input_ids", "labels"} <= set(batch)
            and set(batch) <= marker_core.INPUT_KEYS | {"labels"},
            "explicit grid plan and input-only tensor batch required")
    require(all(isinstance(value, torch.Tensor) for value in batch.values()),
            "model batch must contain tensors only")
    ids, labels = batch["input_ids"], batch["labels"]
    require(ids.dtype == labels.dtype == torch.long and ids.shape == labels.shape == (1, len(plan.full_ids))
            and tuple(ids[0].tolist()) == plan.full_ids and tuple(labels[0].tolist()) == plan.labels,
            "model batch differs from exact immutable spatial token plan")
    _, norm = resolve_final_norm(model)
    captures = []

    def capture(module, inputs, output):
        require(not captures and output.shape == (1, len(plan.full_ids), HIDDEN_SIZE),
                "one final language norm full forward required")
        if plan.marker_positions:
            image_index = torch.tensor(plan.image_positions, dtype=torch.long, device=output.device)
            marker_index = torch.tensor(plan.marker_positions, dtype=torch.long, device=output.device)
            image_states = output[0].index_select(0, image_index)
            queries = output[0].index_select(0, marker_index)
            # Both tensors keep autograd to the same LoRA forward; no future keys.
            cache = prepare_key_cache(head, image_states, plan.features)
            captures.append(predict_with_cache(head, queries, cache))
        else:
            captures.append({"prediction_unit": torch.empty((0, 2), dtype=torch.float32, device=output.device),
                             "all_cell_refinement_logits": torch.empty((0, len(plan.features.valid_indices), 2),
                                                                      dtype=torch.float32, device=output.device),
                             "diagnostics": {"point_count": 0, "head_forward_calls": 0,
                                             "diagnostic_model_forward_calls": 0,
                                             "diagnostic_head_forward_calls": 0,
                                             "diagnostic_backward_calls": 0,
                                             "scope": "no structural marker; head not called"}})

    handle = norm.register_forward_hook(capture)
    try:
        output = model(**{key: value for key, value in batch.items() if key != "labels"},
                       use_cache=False, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
    require(len(captures) == 1, "required final norm was not consumed exactly once")
    logits = output.logits
    require(logits.ndim == 3 and logits.shape[:2] == labels.shape and logits.is_floating_point()
            and logits.device == labels.device, "aligned causal logits required")
    active = labels[:, 1:] != -100
    selected = logits[:, :-1, :][active].float()
    selected_labels = labels[:, 1:][active]
    require(int(active.sum().item()) == plan.supervised_tokens and bool(torch.isfinite(selected).all()),
            "supervised token count or finite logits differs")
    ce = F.cross_entropy(selected, selected_labels)
    prediction = captures[0]["prediction_unit"]
    target = torch.tensor(plan.target_unit, dtype=torch.float32, device=logits.device).reshape(-1, 2)
    point = marker_core.point_loss(prediction, target, native_width=plan.native_width,
                                  native_height=plan.native_height)
    auxiliary = (grid_auxiliary_loss(captures[0], target, plan.features)
                 if plan.marker_positions else point * 0)
    refinement = grid_refine_loss(captures[0], target, plan.features)
    refine_loss = refinement["loss"] if plan.marker_positions else ce * 0
    return {"semantic_ce": ce, "point_loss": point, "grid_aux_ce": auxiliary,
            "grid_refine_loss": refine_loss, "grid_refine_diagnostics": refinement["diagnostics"],
            "total_loss": ce + point + 0.5 * auxiliary + GRID_REFINE_LOSS_WEIGHT * refine_loss,
            "prediction_unit": prediction,
            "supervised_tokens": plan.supervised_tokens, "point_count": len(plan.marker_positions),
            "manual_CE_calls": 1, "model_forward_calls": 1, "default_model_CE_requested": False,
            "head_activation_diagnostics": captures[0]["diagnostics"]}
