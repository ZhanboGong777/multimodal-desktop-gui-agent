"""Auditable screenshot/action supervision, without downloads or desktop actions.

ScreenAgent's official train records are individual steps, grouped by session.
Unlike the permissive Week 3 adapters, training never guesses a missing verb,
coordinate, drag start, or keyboard payload. Official test is never repartitioned.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, get_args

from PIL import Image

from ..datasets.schemas import GUIActionStep
from ..schemas import ActionType, BoundingBox, DesktopAction, Point
from .prompts import action_target_json as serialize_action_target

CANONICAL_ACTION_FIELDS = {
    "click": {"action_type", "x", "y"},
    "double_click": {"action_type", "x", "y"},
    "right_click": {"action_type", "x", "y"},
    "move": {"action_type", "x", "y"},
    "scroll": {"action_type", "x", "y", "scroll_amount"},
    "drag": {"action_type", "start", "end"},
    "type_text": {"action_type", "text"},
    "key_press": {"action_type", "key"},
    "hotkey": {"action_type", "keys"},
    "wait": {"action_type", "duration"},
}
_KEY_ALIASES = {
    "return": "enter", "kp_enter": "enter", "escape": "esc",
    "control_l": "ctrl", "control_r": "ctrl", "control": "ctrl",
    "shift_l": "shift", "shift_r": "shift", "alt_l": "alt", "alt_r": "alt",
    "super_l": "win", "super_r": "win", "super": "win",
    "page_up": "pageup", "page_down": "pagedown", " ": "space",
}
_KEYS = frozenset({
    *"abcdefghijklmnopqrstuvwxyz", *"0123456789", *(f"f{i}" for i in range(1, 13)),
    "enter", "tab", "esc", "space", "backspace", "delete", "up", "down", "left",
    "right", "home", "end", "pageup", "pagedown", "ctrl", "alt", "shift", "win",
})
_NEGATIVE = re.compile(r"(?:^|[_-])(?:neg|negative|rejected)(?:[_-]|$)", re.IGNORECASE)


class DataPreparationError(ValueError):
    """A source or label cannot be used as executable action supervision."""


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    source_split: str
    path: Path | str
    origin_url: str | None = None
    revision: str | None = None


@dataclass
class _Record:
    payload: dict[str, Any] | None
    label: str
    parent: Path | PurePosixPath
    group_hint: str | None = None
    archive: zipfile.ZipFile | None = None
    error: str | None = None


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise DataPreparationError(f"invalid_integer:{name}")
    return value


def _point(value: Any, width: int, height: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise DataPreparationError("missing_coordinates")
    x = _integer(value.get("x", value.get("width")), "x")
    y = _integer(value.get("y", value.get("height")), "y")
    if not (0 <= x < width and 0 <= y < height):
        raise DataPreparationError("coordinates_outside_image")
    return {"x": x, "y": y}


def _key(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise DataPreparationError("missing_keyboard_key")
    key = _KEY_ALIASES.get(value.casefold(), value.casefold())
    if key not in _KEYS:
        raise DataPreparationError("unsupported_keyboard_key")
    return key


def validate_actions(
    actions: Sequence[Mapping[str, Any]], image_width: int, image_height: int,
) -> list[dict[str, Any]]:
    """Validate the bare-array target contract without coercion or extra labels."""
    if (type(image_width) is not int or type(image_height) is not int
            or image_width <= 0 or image_height <= 0):
        raise DataPreparationError("invalid_image_dimensions")
    if not actions:
        raise DataPreparationError("no_executable_actions")
    result = []
    for action in actions:
        if not isinstance(action, Mapping):
            raise DataPreparationError("invalid_action_object")
        kind = action.get("action_type")
        if not isinstance(kind, str) or kind not in get_args(ActionType):
            raise DataPreparationError("unsupported_action_type")
        if set(action) != CANONICAL_ACTION_FIELDS[kind]:
            raise DataPreparationError("incomplete_or_extra_action_fields")
        prepared = dict(action)
        if kind in {"click", "double_click", "right_click", "move", "scroll"}:
            prepared.update(_point(action, image_width, image_height))
        if kind == "drag":
            for name in ("start", "end"):
                if not isinstance(action[name], Mapping) or set(action[name]) != {"x", "y"}:
                    raise DataPreparationError("incomplete_or_extra_point_fields")
                prepared[name] = _point(action[name], image_width, image_height)
        if kind == "scroll":
            amount = _integer(action["scroll_amount"], "scroll_amount")
            if not amount:
                raise DataPreparationError("empty_scroll")
        if kind == "type_text" and (not isinstance(action["text"], str) or not action["text"].strip()):
            raise DataPreparationError("empty_keyboard_text")
        # Preserve leading/trailing spaces and newlines in actual entered text.
        if kind == "key_press":
            prepared["key"] = _key(action["key"])
        if kind == "hotkey":
            if not isinstance(action["keys"], list) or len(action["keys"]) < 2:
                raise DataPreparationError("incomplete_hotkey")
            prepared["keys"] = [_key(item) for item in action["keys"]]
        if kind == "wait":
            value = action["duration"]
            if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
                raise DataPreparationError("invalid_wait_duration")
        DesktopAction.model_validate(prepared)
        result.append(prepared)
    return result


def action_target_json(sample: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    """Only the canonical actions, never ids, annotations, filenames or raw replies."""
    actions = sample["actions"] if isinstance(sample, Mapping) else sample
    return serialize_action_target(actions)


def to_gui_action_steps(actions: Sequence[Mapping[str, Any]]) -> list[GUIActionStep]:
    """Bridge executable targets to the existing Week 3 trajectory schema."""
    return [GUIActionStep(
        step_index=index, action_type=action["action_type"],
        coordinates=Point(x=action["x"], y=action["y"]) if "x" in action else None,
        input_text=action.get("text", action.get("key")), raw_action=dict(action),
    ) for index, action in enumerate(actions)]


def _bbox(raw: Any, width: int, height: int) -> dict[str, int] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping) and "upper_left_position" in raw:
        first = _point(raw["upper_left_position"], width, height)
        last = raw.get("lower_right_position")
        if not isinstance(last, Mapping):
            raise DataPreparationError("invalid_bbox_annotation")
        raw = {"left": first["x"], "top": first["y"],
               "right": last.get("width", last.get("x")),
               "bottom": last.get("height", last.get("y"))}
    if isinstance(raw, list) and len(raw) == 4:
        raw = dict(zip(("left", "top", "right", "bottom"), raw, strict=True))
    if not isinstance(raw, Mapping):
        raise DataPreparationError("invalid_bbox_annotation")
    box = {name: _integer(raw.get(name), name) for name in ("left", "top", "right", "bottom")}
    if not (0 <= box["left"] < box["right"] <= width
            and 0 <= box["top"] < box["bottom"] <= height):
        raise DataPreparationError("invalid_bbox_annotation")
    return BoundingBox.model_validate(box).model_dump()


def _screenagent_action(raw: Mapping[str, Any], width: int, height: int) -> dict | None:
    family = raw.get("action_type")
    if family in {"PlanAction", "EvaluateSubTaskAction"}:
        return None
    if family == "MouseAction":
        verb = raw.get("mouse_action_type")
        if verb in {"click", "left_click", "right_click", "double_click", "move"}:
            kind = "click" if verb == "left_click" else verb
            button = raw.get("mouse_button")
            if kind in {"click", "double_click"} and button not in {"left", "right"}:
                raise DataPreparationError("missing_or_unsupported_mouse_button")
            if kind == "click" and button == "right":
                kind = "right_click"
            elif kind == "double_click" and button != "left":
                raise DataPreparationError("unsupported_double_click_button")
            return {"action_type": kind, **_point(raw.get("mouse_position"), width, height)}
        if verb in {"scroll_up", "scroll_down"}:
            repeat = _integer(raw.get("scroll_repeat"), "scroll_repeat")
            if repeat < 1:
                raise DataPreparationError("invalid_scroll_repeat")
            return {"action_type": "scroll", **_point(raw.get("mouse_position"), width, height),
                    "scroll_amount": repeat if verb == "scroll_up" else -repeat}
        # ScreenAgent drag has only an end point; its start depends on prior pointer state.
        raise DataPreparationError("unsupported_or_incomplete_mouse_action")
    if family == "KeyboardAction":
        verb = raw.get("keyboard_action_type")
        if verb in {"text", "input", "type"}:
            return {"action_type": "type_text", "text": raw.get("keyboard_text")}
        if verb == "press":
            key = raw.get("keyboard_key")
            parts = key.split("+") if isinstance(key, str) and "+" in key else key
            if isinstance(parts, list):
                if len(parts) == 1:
                    return {"action_type": "key_press", "key": _key(parts[0])}
                return {"action_type": "hotkey", "keys": [_key(part) for part in parts]}
            return {"action_type": "key_press", "key": _key(key)}
        raise DataPreparationError("unsupported_or_incomplete_keyboard_action")
    if family == "WaitAction":
        return {"action_type": "wait", "duration": raw.get("wait_time")}
    raise DataPreparationError("unsupported_action_family")


def _actions(payload: Mapping[str, Any], dataset: str, width: int, height: int) -> tuple[list, list]:
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise DataPreparationError("no_executable_actions")
    actions, annotations = [], []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise DataPreparationError("invalid_action_object")
        if dataset == "screenagent":
            action = _screenagent_action(raw, width, height)
            box = raw.get("clickable_area") if action is not None else None
        elif "step_index" in raw:  # Existing GUIActionStep, not arbitrary plan prose.
            kind = raw.get("action_type")
            box = raw.get("target_bbox")
            if kind in {"click", "double_click", "right_click", "move"}:
                action = {"action_type": kind, **_point(raw.get("coordinates"), width, height)}
            elif kind == "type_text":
                action = {"action_type": kind, "text": raw.get("input_text")}
            elif kind == "key_press":
                action = {"action_type": kind, "key": raw.get("input_text")}
            else:
                raise DataPreparationError("unsupported_unified_action")
        else:
            action = dict(raw)
            box = action.pop("target_bbox", None)
        if action is not None:
            actions.append(action)
            bbox = _bbox(box, width, height)
            annotations.append({"target_bbox": bbox} if bbox is not None else {})
    return validate_actions(actions, width, height), annotations


def _parse_records(data: bytes, label: str, parent: Path | PurePosixPath, **kwargs) -> Iterator[_Record]:
    try:
        text = data.decode("utf-8-sig")
        document = json.loads(text)
    except (UnicodeError, ValueError):
        if not label.endswith(".jsonl"):
            yield _Record(None, label, parent, error="invalid_json", **kwargs)
            return
        for number, line in enumerate(data.decode("utf-8-sig", errors="replace").splitlines(), 1):
            if line.strip():
                yield from _parse_records(line.encode("utf-8"), f"{label}:{number}", parent, **kwargs)
        return
    if isinstance(document, dict):
        document = next((document[key] for key in ("records", "samples", "data")
                         if isinstance(document.get(key), list)), [document])
    if not isinstance(document, list):
        yield _Record(None, label, parent, error="invalid_record", **kwargs)
        return
    for index, payload in enumerate(document):
        yield _Record(payload if isinstance(payload, dict) else None,
                      label if len(document) == 1 else f"{label}#{index}", parent,
                      error=None if isinstance(payload, dict) else "invalid_record", **kwargs)


def _records(spec: SourceSpec) -> Iterator[_Record]:
    path = Path(spec.path)
    if not path.exists():
        raise DataPreparationError(f"source does not exist: {path}")
    if path.is_dir():
        files = sorted([*path.rglob("*.json"), *path.rglob("*.jsonl")])
        for file in files:
            yield from _parse_records(file.read_bytes(), str(file.relative_to(path)), file.parent,
                                      group_hint=file.parent.name)
    elif path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if member.endswith((".json", ".jsonl")):
                    name = PurePosixPath(member)
                    yield from _parse_records(archive.read(member), member, name.parent,
                                              group_hint=name.parent.name, archive=archive)
    else:
        yield from _parse_records(path.read_bytes(), path.name, path.parent)


def _image_bytes(record: _Record, dataset: str) -> bytes:
    payload = record.payload or {}
    name = payload.get("saved_image_name") if dataset == "screenagent" else payload.get("image_path")
    if not isinstance(name, str) or not name:
        raise DataPreparationError("missing_screenshot")
    if dataset == "screenagent":
        safe = PurePosixPath(name.replace("\\", "/"))
        if safe.is_absolute() or ".." in safe.parts or ":" in name:
            raise DataPreparationError("unsafe_image_reference")
        candidates = [record.parent / "images" / safe.name, record.parent / safe]
    else:
        candidates = [Path(name)] if Path(name).is_absolute() else [record.parent / name]
    for candidate in candidates:
        if record.archive is not None:
            try:
                return record.archive.read(str(candidate))
            except KeyError:
                continue
        if Path(candidate).is_file():
            return Path(candidate).read_bytes()
    raise DataPreparationError("missing_screenshot")


def _normalise(record: _Record, spec: SourceSpec, image_dir: Path) -> dict[str, Any]:
    if record.error or record.payload is None:
        raise DataPreparationError(record.error or "invalid_record")
    payload = record.payload
    dataset = str(payload.get("dataset_name", spec.dataset)) if spec.dataset == "unified" else spec.dataset
    if dataset == "webarena":
        raise DataPreparationError("webarena_has_no_recorded_actions")
    if _NEGATIVE.search(Path(record.label).stem) or payload.get("is_negative") is True:
        raise DataPreparationError("negative_variant")
    instruction = payload.get("task_prompt_en") or payload.get("task_prompt") or payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise DataPreparationError("missing_instruction")
    group = payload.get("session_id") or payload.get("group_id")
    metadata = payload.get("metadata") or {}
    if not group and isinstance(metadata, Mapping):
        group = metadata.get("session_id") or metadata.get("annotation_id") or metadata.get("group_id")
    if not group and dataset == "screenagent":
        group = record.group_hint
    if not isinstance(group, str) or not group.strip():
        raise DataPreparationError("missing_trajectory_id")
    # The input format and its dataset provenance are distinct: a Week 3 unified
    # ScreenAgent row uses image_path/GUIActionStep, not the original raw fields.
    raw_image = _image_bytes(record, spec.dataset)
    try:
        with Image.open(io.BytesIO(raw_image)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise DataPreparationError("animated_image")
            width, height = image.size
            rgb = image.convert("RGB")
            digest = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(image.format)
            if extension is None:
                raise DataPreparationError("unsupported_image_format")
    except (OSError, ValueError) as exc:
        if isinstance(exc, DataPreparationError):
            raise
        raise DataPreparationError("invalid_screenshot") from exc
    for key, actual in (("video_width", width), ("video_height", height),
                        ("image_width", width), ("image_height", height)):
        if payload.get(key) is not None and _integer(payload[key], key) != actual:
            raise DataPreparationError("declared_image_dimensions_mismatch")
    actions, annotations = _actions(payload, spec.dataset, width, height)
    image_path = image_dir / f"{digest}{extension}"
    if not image_path.exists():
        image_path.write_bytes(raw_image)
    group_id = f"{dataset}:{group.strip()}"
    sample_id = f"{group_id}:{hashlib.sha256(record.label.encode()).hexdigest()[:16]}"
    return {
        "sample_id": sample_id, "group_id": group_id, "image_path": str(image_path.resolve()),
        "instruction": instruction.strip(), "actions": actions, "action_annotations": annotations,
        "image_width": width, "image_height": height, "image_sha256": digest,
        "source": dataset, "source_split": spec.source_split, "record_path": record.label,
    }


def _source_manifest(spec: SourceSpec) -> dict[str, Any]:
    path = Path(spec.path).resolve()
    if not path.exists():
        raise DataPreparationError(f"source does not exist: {path}")
    files = sorted(file for file in path.rglob("*") if file.is_file()) if path.is_dir() else [path]
    entries = [{"path": str(file.relative_to(path)) if path.is_dir() else file.name,
                "size": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
               for file in files]
    return {"dataset": spec.dataset, "source_split": spec.source_split, "path": str(path),
            "origin_url": spec.origin_url, "revision": spec.revision, "files": entries,
            "fingerprint": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}


def _split_train(samples: list[dict], seed: int, fraction: float) -> dict[str, str]:
    """Join sessions sharing any pixels before splitting: different labels still cannot leak."""
    groups = sorted({sample["group_id"] for sample in samples})
    parent = {group: group for group in groups}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    seen = {}
    for sample in samples:
        group = sample["group_id"]
        other = seen.setdefault(sample["image_sha256"], group)
        one, two = find(group), find(other)
        parent[max(one, two)] = min(one, two)
    units = sorted({find(group) for group in groups})
    random.Random(seed).shuffle(units)
    count = min(len(units) - 1, max(1, round(len(units) * fraction))) if len(units) > 1 else 0
    validation = set(units[:count])
    return {group: "validation" if find(group) in validation else "train" for group in groups}


def prepare_dataset(
    sources: Sequence[SourceSpec], output_dir: str | Path, *, seed: int = 42,
    validation_fraction: float = 0.15,
) -> dict[str, Any]:
    """Write train/validation/test JSONL and complete exclusion/split provenance.

    Official test (and an explicit validation source) take priority: an overlapping
    training trajectory is excluded entirely. Exact screenshot/action duplicates
    are removed globally; remaining shared screenshots link whole training groups.
    """
    if not sources or not (0 < validation_fraction < 1):
        raise DataPreparationError("sources and a validation fraction strictly between 0 and 1 are required")
    if any(spec.dataset not in {"screenagent", "unified", "webarena"}
           or spec.source_split not in {"train", "validation", "test"} for spec in sources):
        raise DataPreparationError("unsupported dataset or source split")
    provenance = [_source_manifest(spec) for spec in sources]
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise DataPreparationError("output directory is not empty; choose a new destination")
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir()
    samples, excluded = [], []
    total = 0
    for spec in sources:
        for record in _records(spec):
            total += 1
            try:
                sample = _normalise(record, spec, image_dir)
            except (DataPreparationError, ValueError) as exc:
                excluded.append({"source": spec.dataset, "source_split": spec.source_split,
                                 "record_path": record.label, "reason": str(exc)})
                continue
            samples.append(sample)
    test = [sample for sample in samples if sample["source_split"] == "test"]
    test_images = {sample["image_sha256"] for sample in test}
    test_groups = {sample["group_id"] for sample in test}
    # Entire groups, rather than single rows, are removed when held-out content overlaps.
    conflicting = {sample["group_id"] for sample in samples if sample["source_split"] != "test"
                   and (sample["image_sha256"] in test_images or sample["group_id"] in test_groups)}
    validation_samples = [sample for sample in samples if sample["source_split"] == "validation"
                          and sample["group_id"] not in conflicting]
    protected_images = test_images | {sample["image_sha256"] for sample in validation_samples}
    protected_groups = test_groups | {sample["group_id"] for sample in validation_samples}
    conflicting_train = {sample["group_id"] for sample in samples if sample["source_split"] == "train"
                         and (sample["image_sha256"] in protected_images
                              or sample["group_id"] in protected_groups)}
    dedup, retained, seen_ids = set(), [], set()
    for sample in sorted(samples, key=lambda row: ({"test": 0, "validation": 1, "train": 2}[row["source_split"]],
                                                  row["group_id"], row["record_path"])):
        reason = None
        if sample["source_split"] != "test" and sample["group_id"] in conflicting:
            reason = "official_test_overlap_trajectory"
        elif sample["source_split"] == "train" and sample["group_id"] in conflicting_train:
            reason = "heldout_overlap_trajectory"
        # Object key order is not part of action semantics; preserve the array
        # order and exact text while comparing independently serialized labels.
        fingerprint = (sample["image_sha256"], json.dumps(
            sample["actions"], ensure_ascii=False, sort_keys=True, allow_nan=False))
        if reason is None and fingerprint in dedup:
            reason = "duplicate_image_actions"
        if reason is None and sample["sample_id"] in seen_ids:
            reason = "duplicate_sample_id"
        if reason:
            excluded.append({"source": sample["source"], "source_split": sample["source_split"],
                             "record_path": sample["record_path"], "reason": reason})
            continue
        dedup.add(fingerprint)
        seen_ids.add(sample["sample_id"])
        retained.append(sample)
    assignments = _split_train([row for row in retained if row["source_split"] == "train"],
                               seed, validation_fraction)
    splits = {name: [] for name in ("train", "validation", "test")}
    for sample in retained:
        split = assignments[sample["group_id"]] if sample["source_split"] == "train" else sample["source_split"]
        splits[split].append(sample)
    # Verify the output invariant independently of the assignment algorithm.
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("group_id", "image_sha256"):
            if {row[field] for row in splits[first]} & {row[field] for row in splits[second]}:
                raise DataPreparationError(f"split leakage detected: {field}")
    artifacts = {}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in sorted(rows, key=lambda value: value["sample_id"]):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        artifacts[name] = {"path": str(path), "samples": len(rows),
                           "groups": len({row["group_id"] for row in rows}),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        "schema_version": 1, "seed": seed, "validation_fraction": validation_fraction,
        "target_format": "bare JSON array of canonical DesktopAction objects",
        "image_hash": "SHA256(width x height prefix + decoded native RGB pixels)",
        "split_policy": "official test preserved; training session components sharing pixels split together",
        "sources": provenance, "input_records": total, "retained_samples": len(retained),
        "excluded_samples": len(excluded), "exclusion_counts": dict(sorted(Counter(
            item["reason"] for item in excluded).items())), "exclusions": excluded,
        "group_assignments": assignments, "outputs": artifacts,
        "ready_for_training": bool(splits["train"] and splits["validation"]),
        "ready_for_heldout_comparison": bool(splits["train"] and splits["validation"] and splits["test"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
