"""Narrow frozen dense embedding restoration and forward-only CUDA BF16 AMP.

Imports never touch a model, weight, image or CUDA device. Root invokes repair
after one fresh loader. Chunk byte proofs avoid full GPU/CPU matrix clones.
"""
from __future__ import annotations

import hashlib

SCHEMA = "week5.qwen-resized-native.frozen_embedding_memory_audit.v1"
PROFILE = {"name": "frozen_dense_embeddings_original_bf16_and_cuda_autocast",
           "restore_only": ["model.get_input_embeddings().weight", "model.get_output_embeddings().weight"],
           "restore_dtype": "torch.bfloat16", "other_parameters_unchanged": True,
           "trainable_dtype": "torch.float32", "forward_autocast": {"device_type": "cuda", "dtype": "torch.bfloat16", "enabled": True, "cache_enabled": False},
           "backward_outside_autocast": True, "GradScaler_calls": 0, "custom_loss_calls": 0}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _cpu_chunks(parameter, chunk_elements, operations):
    need(type(chunk_elements) is int and chunk_elements > 0 and parameter.is_contiguous(), "positive bounded chunk size and contiguous parameter required")
    flat = parameter.detach().view(-1)
    for start in range(0, flat.numel(), chunk_elements):
        operations["CPU_chunk_materializations"] += 1
        if flat.device.type == "cuda":
            operations["GPU_to_CPU_chunk_transfers"] += 1
        yield flat[start:start + chunk_elements].to(device="cpu").contiguous()


def _bytes(piece, torch):
    return piece.contiguous().view(torch.uint8).numpy().tobytes()


def _raw_sha(parameter, chunk_elements, operations, torch, *, normalize_float32=False):
    digest = hashlib.sha256()
    for piece in _cpu_chunks(parameter, chunk_elements, operations):
        if normalize_float32:
            piece = piece.to(torch.float32)
        digest.update(_bytes(piece, torch))
    return digest.hexdigest()


def _metadata(parameter, chunk_elements, operations, torch):
    return {"parameter_id": id(parameter), "data_pointer": parameter.data_ptr(), "dtype": str(parameter.dtype),
            "shape": list(parameter.shape), "numel": parameter.numel(), "requires_grad": parameter.requires_grad,
            "device": str(parameter.device), "class": type(parameter).__name__,
            "raw_sha256": _raw_sha(parameter, chunk_elements, operations, torch)}


def restore_frozen_embeddings(model, *, chunk_elements=1048576):
    """Only two getter-selected ordinary frozen weights; tied identity dedup."""
    import torch

    operations = {"CPU_chunk_materializations": 0, "GPU_to_CPU_chunk_transfers": 0, "selected_dtype_conversions": 0,
                  "model_forward_calls": 0, "model_backward_calls": 0}
    named = dict(model.named_parameters())
    need(named and all(not parameter.requires_grad or ("lora_" in name and parameter.dtype == torch.float32)
                      for name, parameter in named.items()), "only FP32 language LoRA may be trainable")
    roles = {"input": model.get_input_embeddings().weight, "output": model.get_output_embeddings().weight}
    selected = {id(parameter): parameter for parameter in roles.values()}
    need(all(any(parameter is candidate for candidate in named.values()) and parameter.requires_grad is False
             and parameter.dtype in (torch.float32, torch.bfloat16) and parameter.ndim == 2
             and type(parameter).__name__ != "Params4bit" and not hasattr(parameter, "quant_state")
             and not getattr(parameter, "is_meta", False) for parameter in selected.values()),
             "getters must identify existing frozen dense FP32/originalBF16 ordinary weights; no Params4bit")
    other_before = {name: _metadata(parameter, chunk_elements, operations, torch) for name, parameter in named.items() if id(parameter) not in selected}
    selected_before = {}
    for identity, parameter in selected.items():
        pre = _metadata(parameter, chunk_elements, operations, torch)
        original_fp32, expected_bf16, roundtrip_fp32 = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        for piece in _cpu_chunks(parameter, chunk_elements, operations):
            need(bool(piece.isfinite().all()), "frozen dense embedding contains nonfinite values")
            original = piece.to(torch.float32)
            restored = piece.to(torch.bfloat16)
            roundtrip = restored.to(torch.float32)
            original_fp32.update(_bytes(original, torch))
            expected_bf16.update(_bytes(restored, torch))
            roundtrip_fp32.update(_bytes(roundtrip, torch))
        need(original_fp32.hexdigest() == roundtrip_fp32.hexdigest(), "promoted dense values are not bit-exact original BF16 (including signed zero)")
        pre.update(original_FP32_sha256=original_fp32.hexdigest(), expected_BF16_sha256=expected_bf16.hexdigest(),
                   exact_BF16_roundtrip=True, roles=[role for role, candidate in roles.items() if candidate is parameter])
        selected_before[identity] = pre
    # No mutation happens until every selected matrix has passed finite/exact checks.
    with torch.no_grad():
        for parameter in selected.values():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
                operations["selected_dtype_conversions"] += 1
    selected_stats = []
    for identity, parameter in selected.items():
        pre = selected_before[identity]
        post = _metadata(parameter, chunk_elements, operations, torch)
        post_fp32 = _raw_sha(parameter, chunk_elements, operations, torch, normalize_float32=True)
        need(id(parameter) == identity and parameter.requires_grad is False and parameter.dtype == torch.bfloat16
             and post["raw_sha256"] == pre["expected_BF16_sha256"] and post_fp32 == pre["original_FP32_sha256"], "restored dense data/identity must equal original value bytes")
        saved = parameter.numel() * (4 - 2) if pre["dtype"] == "torch.float32" else 0
        selected_stats.append({"roles": pre["roles"], "before": pre, "after": post, "post_upcast_FP32_sha256": post_fp32,
                               "value_bytes_preserved": True, "actual_unique_numel": parameter.numel(), "actual_static_bytes_saved": saved})
    need(model.get_input_embeddings().weight is roles["input"]
         and model.get_output_embeddings().weight is roles["output"], "getters/Parameter alias identities changed")
    after_named = dict(model.named_parameters())
    need(set(after_named) == set(named) and all(after_named[name] is parameter for name, parameter in named.items()), "parameter registry changed during memory repair")
    other_after = {name: _metadata(parameter, chunk_elements, operations, torch) for name, parameter in after_named.items() if id(parameter) not in selected}
    need(other_before == other_after, "nonselected LoRA/norm/visual/Params4bit data, dtype, object or grad flag changed")
    return {"schema": SCHEMA, "profile": PROFILE, "selected_unique_parameters": len(selected),
            "tied_input_output": roles["input"] is roles["output"], "selected_parameters": selected_stats,
            "actual_unique_numel": sum(item["actual_unique_numel"] for item in selected_stats),
            "actual_static_bytes_saved": sum(item["actual_static_bytes_saved"] for item in selected_stats),
            "all_other_parameter_metadata_before": other_before, "all_other_parameter_metadata_after": other_after,
            "all_other_parameter_data_dtype_identity_gradflag_unchanged": True, "getter_aliases_preserved": True,
            "operations": operations, "chunk_elements": chunk_elements,
            "training_norms_remain_original_FP32": True,
            "inference_profile": "unchanged public for_training=False loader; norm BF16 differs from kbit training norm FP32; no bitwise equivalence claim"}


def validate_audit(audit, *, require_cuda=True):
    need(audit.get("schema") == SCHEMA and audit.get("profile") == PROFILE
         and type(audit["selected_unique_parameters"]) is int and audit["selected_unique_parameters"] in (1, 2)
         and len(audit["selected_parameters"]) == audit["selected_unique_parameters"]
         and type(audit["tied_input_output"]) is bool
         and audit["tied_input_output"] is (audit["selected_unique_parameters"] == 1)
         and audit["getter_aliases_preserved"] is True
         and audit["all_other_parameter_data_dtype_identity_gradflag_unchanged"] is True
         and audit["all_other_parameter_metadata_before"] == audit["all_other_parameter_metadata_after"], "exact narrow memory profile audit required")
    identities = set()
    roles = []
    for item in audit["selected_parameters"]:
        pre, post = item["before"], item["after"]
        need(all(type(value) is int and value > 0 for value in (pre["parameter_id"], post["parameter_id"], pre["numel"], post["numel"], item["actual_unique_numel"]))
             and type(item["actual_static_bytes_saved"]) is int and item["actual_static_bytes_saved"] >= 0
             and len(pre["shape"]) == 2 and all(type(axis) is int and axis > 0 for axis in pre["shape"])
             and pre["numel"] == pre["shape"][0] * pre["shape"][1]
             and pre["parameter_id"] == post["parameter_id"] and pre["parameter_id"] not in identities
             and pre["requires_grad"] is post["requires_grad"] is False
             and pre["dtype"] in {"torch.float32", "torch.bfloat16"} and post["dtype"] == "torch.bfloat16"
             and pre["shape"] == post["shape"] and pre["numel"] == post["numel"] == item["actual_unique_numel"]
             and pre["exact_BF16_roundtrip"] is item["value_bytes_preserved"] is True
             and post["raw_sha256"] == pre["expected_BF16_sha256"]
             and item["post_upcast_FP32_sha256"] == pre["original_FP32_sha256"]
             and item["actual_static_bytes_saved"] == post["numel"] * (2 if pre["dtype"] == "torch.float32" else 0)
             and item["roles"] == pre["roles"] and item["roles"] in (["input"], ["output"], ["input", "output"])
             and all(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)
                     for value in (pre["raw_sha256"], post["raw_sha256"], pre["original_FP32_sha256"], pre["expected_BF16_sha256"], item["post_upcast_FP32_sha256"]))
             and (not require_cuda or pre["device"].startswith("cuda") and post["device"] == pre["device"]), "selected dense byte/dtype/alias proof differs")
        identities.add(pre["parameter_id"])
        roles.extend(item["roles"])
    need(sorted(roles) == ["input", "output"]
         and type(audit["actual_unique_numel"]) is type(audit["actual_static_bytes_saved"]) is int
         and audit["actual_unique_numel"] == sum(item["actual_unique_numel"] for item in audit["selected_parameters"])
         and audit["actual_static_bytes_saved"] == sum(item["actual_static_bytes_saved"] for item in audit["selected_parameters"])
         and all(type(value) is int and value >= 0 for value in audit["operations"].values())
         and audit["operations"]["model_forward_calls"] == audit["operations"]["model_backward_calls"] == 0,
         "memory tensor operations cannot be counted as model forward/backward calls")
    return audit


class AutocastForward:
    """Callable model proxy. Backward and optimizer run after __call__ returns."""
    def __init__(self, model, counters=None, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.model, self.torch = model, torch_module
        self.counters = counters if counters is not None else {}
        self.counters.update(entries=0, active=0, cache_disabled=0, completed=0, GradScaler_calls=0, custom_loss_calls=0)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.counters["entries"] += 1
        with self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16, enabled=True, cache_enabled=False):
            need(self.torch.is_autocast_enabled("cuda") is True and self.torch.is_autocast_cache_enabled() is False,
                 "actual CUDA BF16 forward context/cache=False required")
            self.counters["active"] += 1
            self.counters["cache_disabled"] += 1
            result = self.model(*args, **kwargs)
            self.counters["completed"] += 1
        return result
