"""Input-only native image geometry and complete resized-coordinate labels.

No model, tensor, image, dataset or UI runtime is imported. Native annotations
enter only the target transform; the visible prompt never reads them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping

PROTOCOL_VERSION = "week5.qwen-resized-pixel-action-array.v1"
GEOMETRY_VERSION = "week5.qwen-native-image.pixelcentre-geometry.v1"
PROMPT_VERSION = "week5.qwen-native-image.resized-actor-prompt.v1"
PROFILE = {"min_pixels": 100352, "max_pixels": 802816, "max_seq_length": 1536,
           "patch_size": 14, "spatial_merge_size": 2, "processor_use_fast": True}
MAX_NEW_TOKENS = 256
COORDINATE_POLICY = "integer_pixel_centres_floor_((2*x+1)*destination)/(2*source);reject_never_clamp"
REQUIRED = {**dict.fromkeys(("click", "double_click", "right_click", "move"), frozenset(("x", "y"))),
            "scroll": frozenset(("x", "y", "scroll_amount")), "drag": frozenset(("start", "end")),
            "type_text": frozenset(("text",)), "key_press": frozenset(("key",)),
            "hotkey": frozenset(("keys",)), "wait": frozenset(("duration",))}
ACTION_RULES = (
    'Return only a complete JSON array of desktop actions, without markdown or explanation. '
    'Each object has "action_type" and exactly its required fields. '
    'click/double_click/right_click/move use integer "x","y"; '
    'scroll uses integer "x","y","scroll_amount"; drag uses "start","end", '
    'each with integer "x","y"; type_text uses "text"; key_press uses "key"; '
    'hotkey uses "keys", a nonempty array of strings; wait uses nonnegative "duration" seconds. '
    'Coordinates are absolute integer pixels of the processed screenshot size stated below, '
    'with (0,0) at the top left; never use normalized coordinates. '
    'Produce the complete ordered action chunk for the supplied request, preserving requested '
    'text exactly and avoiding actions on unseen controls or later subtasks.'
)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def size(width, height):
    need(type(width) is int and width > 0 and type(height) is int and height > 0,
         "positive strict integer dimensions required")
    return width, height


def coordinate(value, dimension):
    need(type(dimension) is int and dimension > 0 and type(value) is int and 0 <= value < dimension,
         "coordinate must be a strict integer within its declared image")
    return value


def map_axis(value, source_size, destination_size):
    coordinate(value, source_size)
    need(type(destination_size) is int and destination_size > 0, "positive destination size required")
    result = ((2 * value + 1) * destination_size) // (2 * source_size)
    coordinate(result, destination_size)
    return result


def expected_resize(native_width, native_height):
    """Pinned installed Qwen smart_resize formula; actual processor must confirm.

    Python round is intentionally unchanged from the installed implementation.
    This prediction alone never qualifies or acknowledges an actual image.
    """
    width, height = size(native_width, native_height)
    factor = PROFILE["patch_size"] * PROFILE["spatial_merge_size"]
    need(max(height, width) / min(height, width) <= 200, "unsupported image aspect ratio")
    resized_height, resized_width = round(height / factor) * factor, round(width / factor) * factor
    if resized_height * resized_width > PROFILE["max_pixels"]:
        beta = math.sqrt(height * width / PROFILE["max_pixels"])
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < PROFILE["min_pixels"]:
        beta = math.sqrt(PROFILE["min_pixels"] / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return size(resized_width, resized_height)


def geometry_from_grid(native_width, native_height, grid_thw):
    size(native_width, native_height)
    need(isinstance(grid_thw, (list, tuple)) and len(grid_thw) == 3
         and all(type(value) is int and value > 0 for value in grid_thw), "one strict positive image grid required")
    temporal, grid_height, grid_width = grid_thw
    merge = PROFILE["spatial_merge_size"]
    need(temporal == 1 and grid_height % merge == 0 and grid_width % merge == 0,
         "single native image patch14/merge2 grid required")
    resized_width, resized_height = grid_width * PROFILE["patch_size"], grid_height * PROFILE["patch_size"]
    need((resized_width, resized_height) == expected_resize(native_width, native_height),
         "actual processed image grid differs from pinned input-only resize")
    return {"schema": GEOMETRY_VERSION, "native_width": native_width, "native_height": native_height,
            "resized_width": resized_width, "resized_height": resized_height,
            "image_grid_thw": list(grid_thw), "patch_size": 14, "spatial_merge_size": 2,
            "image_tokens": grid_height * grid_width // 4, "coordinate_policy": COORDINATE_POLICY}


def expected_geometry(native_width, native_height):
    width, height = expected_resize(native_width, native_height)
    return geometry_from_grid(native_width, native_height, [1, height // 14, width // 14])


def checked_geometry(value):
    need(isinstance(value, Mapping), "registered geometry object required")
    expected = geometry_from_grid(value["native_width"], value["native_height"], value["image_grid_thw"])
    need(canonical(value) == canonical(expected), "complete exact geometry contract required")
    return expected


def validate_actions(value, width, height):
    """Original DesktopAction fields/types plus declared-coordinate bounds."""
    size(width, height)
    need(isinstance(value, list) and bool(value), "nonempty complete action array required")
    for action in value:
        need(isinstance(action, dict) and isinstance(action.get("action_type"), str)
             and action["action_type"] in REQUIRED, "unknown action object")
        need(set(action) == REQUIRED[action["action_type"]] | {"action_type"}, "action fields differ")
        for axis, dimension in (("x", width), ("y", height)):
            if axis in action:
                coordinate(action[axis], dimension)
        if "scroll_amount" in action:
            need(type(action["scroll_amount"]) is int, "strict integer scroll amount required")
        for field in ("text", "key"):
            if field in action:
                need(isinstance(action[field], str) and bool(action[field]), "nonempty literal required")
        if "keys" in action:
            need(isinstance(action["keys"], list) and bool(action["keys"])
                 and all(isinstance(key, str) and bool(key) for key in action["keys"]), "nonempty key strings required")
        for field in ("start", "end"):
            if field in action:
                point = action[field]
                need(isinstance(point, dict) and set(point) == {"x", "y"}, "exact drag point required")
                coordinate(point["x"], width)
                coordinate(point["y"], height)
        if "duration" in action:
            duration = action["duration"]
            need(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                 "finite nonnegative duration required")
    return value


def transform_actions(actions, geometry, *, to_resized):
    geometry = checked_geometry(geometry)
    source = "native" if to_resized else "resized"
    destination = "resized" if to_resized else "native"
    source_width, source_height = geometry[source + "_width"], geometry[source + "_height"]
    destination_width, destination_height = geometry[destination + "_width"], geometry[destination + "_height"]
    validate_actions(actions, source_width, source_height)
    result = copy.deepcopy(actions)
    for action in result:
        points = [action] if "x" in action else [action[field] for field in ("start", "end") if field in action]
        for point in points:
            point["x"] = map_axis(point["x"], source_width, destination_width)
            point["y"] = map_axis(point["y"], source_height, destination_height)
    validate_actions(result, destination_width, destination_height)
    return result


def to_resized(actions, geometry):
    return transform_actions(actions, geometry, to_resized=True)


def from_resized(actions, geometry):
    return transform_actions(actions, geometry, to_resized=False)


def visible_input(sample):
    """Explicit source-verified request fields; no labels, IDs or source metadata."""
    size(sample["image_width"], sample["image_height"])
    need(isinstance(sample.get("instruction"), str) and bool(sample["instruction"].strip()), "instruction required")
    result = {key: sample[key] for key in ("instruction", "image_width", "image_height")}
    if "actor_step" in sample:
        need(isinstance(sample["actor_step"], str) and bool(sample["actor_step"].strip()), "nonempty supplied actor context required")
        result["actor_step"] = sample["actor_step"]
    return result


def build_prompt(sample, geometry):
    visible = visible_input(sample)
    geometry = checked_geometry(geometry)
    need((visible["image_width"], visible["image_height"]) == (geometry["native_width"], geometry["native_height"]),
         "request and geometry native dimensions differ")
    text = (ACTION_RULES + f'\nProcessed screenshot size: {geometry["resized_width"]} x {geometry["resized_height"]} pixels.'
            + "\nTask: " + visible["instruction"])
    return text + "\nCurrent subtask: " + visible["actor_step"] if "actor_step" in visible else text


def build_messages(sample, geometry, *, include_target=False):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_prompt(sample, geometry)}]}]
    if include_target:
        messages.append({"role": "assistant", "content": canonical(to_resized(sample["actions"], geometry))})
    return messages
