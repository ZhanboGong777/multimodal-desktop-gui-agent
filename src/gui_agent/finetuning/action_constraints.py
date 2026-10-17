"""Opt-in native action JSON generation constraints, without sample or label inputs.

The unchanged evaluation parser remains authoritative. This backend controls
format, not action choice, text correctness, coordinate bounds or completion.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

OUTPUT_CONSTRAINTS = ("none", "json_actions")
BACKEND_VERSIONS = {"lm-format-enforcer": "0.11.3", "interegular": "0.3.3"}
# Avoid inherited LMFE defaults (20 array items, 12 spaces) or LMFE_* env settings.
# The existing generation token budget and hard request deadline remain the limits.
PARSER_CONFIG = {"max_consecutive_whitespaces": 2147483647,
                 "max_json_array_length": 2147483647, "force_json_field_order": False}


def native_action_schema() -> dict[str, Any]:
    """Fixed native bare-array grammar; no goal, annotations, coordinates or counts."""
    integer = {"type": "integer"}
    string = {"type": "string", "minLength": 1}
    point = {"type": "object", "properties": {"x": integer, "y": integer},
             "required": ["x", "y"], "additionalProperties": False}
    fields = {**{kind: {"x": integer, "y": integer}
                 for kind in ("click", "double_click", "right_click", "move")},
              "type_text": {"text": string}, "key_press": {"key": string},
              "hotkey": {"keys": {"type": "array", "items": string, "minItems": 1}},
              "scroll": {"x": integer, "y": integer, "scroll_amount": integer},
              "drag": {"start": point, "end": point},
              "wait": {"duration": {"type": "number", "minimum": 0}}}
    branches = [{"type": "object", "properties": {
                    "action_type": {"type": "string", "enum": [kind]}, **properties},
                 "required": ["action_type", *properties], "additionalProperties": False}
                for kind, properties in fields.items()]
    # Disjoint singleton action_type enums make anyOf an unambiguous typed union.
    return {"type": "array", "minItems": 1, "items": {"anyOf": branches}}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _backend_sources() -> dict[str, str]:
    result = {}
    for distribution, package in (("lm-format-enforcer", "lmformatenforcer"),
                                  ("interegular", "interegular")):
        try:
            root = Path(importlib.metadata.distribution(distribution).locate_file(package))
        except importlib.metadata.PackageNotFoundError:
            continue
        for path in sorted(root.rglob("*.py")):
            result[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def constraint_descriptor(mode: str) -> dict[str, Any]:
    """JSON-compatible provenance; importing this module loads no model libraries."""
    if mode not in OUTPUT_CONSTRAINTS:
        raise ValueError("unknown output_constraint")
    if mode == "none":
        return {"mode": mode, "schema_sha256": None, "backend": None}
    schema = native_action_schema()
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"mode": mode, "schema_sha256": digest, "schema": schema,
            "backend": "lm-format-enforcer", "required_versions": dict(BACKEND_VERSIONS),
            "library_versions": {name: _version(name) for name in BACKEND_VERSIONS},
            "dependency_versions": {name: _version(name) for name in ("pydantic", "packaging", "PyYAML")},
            "backend_source_sha256": _backend_sources(), "parser_config": dict(PARSER_CONFIG),
            "alphabet_policy": "actual tokenizer trie alphabet; fixed limits restored after backend setup",
            "backend_limitations": [
                "integer exponent literals can pass backend but fail unchanged Python-int strict validator",
                "numeric minimum is not enforced by backend; unchanged strict validator remains authoritative",
                "some canonical adjacent-escape tokens are rejected; equivalent split-token paths may remain reachable",
                "backend internal parser errors can force EOS; unchanged strict validator still judges every raw output",
            ],
            "scope": "format constraints only; original strict parser/scorer remain authoritative"}


class NativeActionConstraint:
    """One tokenizer trie per resident worker, independent parser per request."""

    def __init__(self, tokenizer: Any, model_eos_token_ids: int | list[int]) -> None:
        self.descriptor = constraint_descriptor("json_actions")
        if self.descriptor["library_versions"] != BACKEND_VERSIONS:
            raise RuntimeError("json_actions requires pinned lm-format-enforcer 0.11.3 and interegular 0.3.3")
        # Lazy imports: ordinary unconstrained inference needs neither dependency.
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn,
        )

        eos_ids = model_eos_token_ids if isinstance(model_eos_token_ids, list) else [model_eos_token_ids]
        if (not eos_ids or any(type(value) is not int for value in eos_ids)
                or type(tokenizer.eos_token_id) is not int or tokenizer.eos_token_id not in eos_ids):
            raise ValueError("constraint tokenizer EOS is not an actual model generation EOS")
        self.eos_descriptor = {"tokenizer_eos_token_id": tokenizer.eos_token_id,
                               "model_eos_token_ids": list(eos_ids),
                               "constrained_eos_token_ids": [tokenizer.eos_token_id]}
        self._tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        self._parser_class = JsonSchemaParser
        self._config_class = CharacterLevelParserConfig
        self._build_prefix = build_transformers_prefix_allowed_tokens_fn

    def new_prefix_function(self) -> Any:
        # Explicit configuration ignores process LMFE_* variables. The parser and
        # its token-sequence cache must never be reused across independent calls.
        config = self._config_class(**PARSER_CONFIG, alphabet=self._tokenizer_data.tokenizer_alphabet)
        parser = self._parser_class(native_action_schema(), config=config)
        prefix = self._build_prefix(self._tokenizer_data, parser)
        # LMFE 0.11.3 TokenEnforcer replaces the supplied config with a fresh
        # environment-derived one. Restore our fixed limits before its first
        # allowed-token query; the initial string context already has the full
        # tokenizer alphabet. Do not mutate global environment or library code.
        parser.config = config
        if any(getattr(parser.config, key) != value for key, value in PARSER_CONFIG.items()):
            raise RuntimeError("backend did not retain explicit parser configuration")
        return prefix
