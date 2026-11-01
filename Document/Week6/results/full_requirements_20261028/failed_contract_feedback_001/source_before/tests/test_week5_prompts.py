"""The offline experiment must never turn reference labels into model input."""

import hashlib
import json

import pytest

from gui_agent.finetuning.prompts import (
    ACTION_SCHEMA,
    PROMPT_VARIANTS,
    TRAINING_PROMPT_VARIANTS,
    action_target_json,
    build_user_prompt,
    prompt_input,
    prompt_rules,
    prompt_sha256,
)


def test_ground_truth_and_provenance_cannot_change_the_model_prompt():
    sample = {"instruction": "Open the browser", "image_width": 1280, "image_height": 720,
              "actor_step": "Click the browser launcher"}
    with_labels = sample | {
        "actions": [{"action_type": "click", "x": 987, "y": 654}],
        "image_path": "SECRET_FILENAME.jpg",
        "sample_id": "SECRET_ID",
        "group_id": "SECRET_GROUP",
        "action_annotations": [{"target_bbox": [900, 600, 1000, 700]}],
    }
    for variant in PROMPT_VARIANTS:
        assert build_user_prompt(sample, variant) == build_user_prompt(with_labels, variant)


def test_prompt_optimization_keeps_the_output_schema_fixed():
    for variant in PROMPT_VARIANTS:
        assert ACTION_SCHEMA in prompt_rules(variant)
    assert prompt_sha256("original") != prompt_sha256("improved")


@pytest.mark.parametrize("variant", ["original", "improved"])
def test_existing_prompt_rules_match_the_preserved_original_source(variant):
    # Pin the rule bytes used by the completed experiment, independent of the
    # new variant's wording and without reading local ignored evidence files.
    expected = {
        "original": "2cf399aa8ef6e14199a895113c695a613488a53a966e5543c9d6c7a9cbd4d854",
        "improved": "41632aef43df6750db2e649d8aac3e13d6075227cbf38c85a5be39b69255de02",
    }
    assert hashlib.sha256(prompt_rules(variant).encode("utf-8")).hexdigest() == expected[variant]


def test_targeted_prompt_cannot_derive_action_count_or_values_from_labels():
    visible = {"instruction": 'Enter "A  B\\nC" and press Enter.', "image_width": 1024, "image_height": 768}
    first = visible | {"actions": [{"action_type": "click", "x": 11, "y": 22}],
                       "action_annotations": [{"target_bbox": [1, 2, 3, 4]}],
                       "source": "FIRST_SOURCE", "source_split": "train", "sample_id": "FIRST_ID"}
    second = visible | {"actions": [{"action_type": "type_text", "text": "SECRET_LABEL"},
                                    {"action_type": "key_press", "key": "enter"}],
                        "action_annotations": [], "source": "SECOND_SOURCE",
                        "source_split": "validation", "sample_id": "SECOND_ID",
                        "image_path": "LABEL_IN_FILENAME.png", "target_count": 777}
    prompt = build_user_prompt(visible, "targeted_sequence")
    assert prompt == build_user_prompt(first, "targeted_sequence") == build_user_prompt(second, "targeted_sequence")
    assert visible["instruction"] in prompt and "1024 x 768 pixels" in prompt
    for secret in ("SECRET_LABEL", "FIRST_SOURCE", "SECOND_SOURCE", "FIRST_ID", "SECOND_ID",
                   "LABEL_IN_FILENAME", "777"):
        assert secret not in prompt


def test_targeted_prompt_serialization_preserves_order_and_exact_multiaction_text():
    actions = [
        {"action_type": "click", "x": 17, "y": 19},
        {"action_type": "type_text", "text": " A  B\nC\t "},
        {"action_type": "key_press", "key": "enter"},
    ]
    target = action_target_json(actions)
    assert json.loads(target) == actions
    assert len(json.loads(target)) == 3
    assert ACTION_SCHEMA in prompt_rules("targeted_sequence")
    assert prompt_sha256("targeted_sequence") not in {prompt_sha256("original"), prompt_sha256("improved")}


def test_current_step_registration_preserves_the_completed_targeted_rule_bytes():
    assert prompt_sha256("targeted_sequence") == "b7c45f07c0014e9c2949d6e2149c368e779ddcc852e9beb4081200c35b7ebb67"
    assert PROMPT_VARIANTS[:4] == ("original", "improved", "targeted_sequence", "current_step")
    assert TRAINING_PROMPT_VARIANTS[:3] == ("original", "targeted_sequence", "current_step")
    assert prompt_sha256("current_step") not in {prompt_sha256(name) for name in PROMPT_VARIANTS[:3]}


def test_current_step_input_cannot_depend_on_teacher_step_history_or_plan():
    visible = {"instruction": "Open the browser and look up Angemon", "image_width": 1024, "image_height": 768}
    oracle = visible | {
        "actions": [{"action_type": "type_text", "text": "PRIVATE_ACTION_LITERAL"}],
        "action_annotations": [{"target_bbox": [901, 902, 903, 904]}],
        "current_task": "PRIVATE_TEACHER_STEP",
        "send_prompt": "PRIVATE_TEACHER_PROMPT",
        "history": [{"action_type": "click", "x": 999, "y": 888}],
        "plan": ["PRIVATE_FUTURE_PLAN"],
        "source": "PRIVATE_SOURCE",
        "record_path": "PRIVATE_RECORD.json",
        "sample_id": "PRIVATE_ID",
        "target_count": 777,
    }
    assert build_user_prompt(oracle, "current_step") == build_user_prompt(visible, "current_step")
    changed_oracle = oracle | {"actions": [], "current_task": "DIFFERENT_TEACHER_STEP", "history": [], "plan": []}
    assert build_user_prompt(changed_oracle, "current_step") == build_user_prompt(visible, "current_step")


def test_current_step_keeps_original_unicode_literals_and_whitespace_without_rewriting():
    instruction = '  在输入框逐字输入“历史上的今天  Angemon\nA\tB”，然后搜索。\n '
    sample = {"instruction": instruction, "image_width": 1200, "image_height": 900}
    prompt = build_user_prompt(sample, "current_step")
    assert prompt.endswith(f"Screenshot size: 1200 x 900 pixels.\nTask: {instruction}")
    assert sample["instruction"] == instruction
    assert "today in history" not in prompt and "Angel" not in prompt


@pytest.mark.parametrize("step", [None, "", "  ", 12, [], {}])
def test_conditional_actor_rejects_missing_or_invalid_step_instead_of_using_labels(step):
    sample = {"instruction": "Search", "image_width": 20, "image_height": 10,
              "current_task": "PRIVATE_RAW_STEP", "actions": [{"action_type": "type_text", "text": "SECRET"}]}
    if step is not None:
        sample["actor_step"] = step
    with pytest.raises(ValueError, match="actor_step"):
        build_user_prompt(sample, "conditional_actor")


def test_conditional_actor_requires_explicit_opt_in_and_preserves_step_verbatim():
    sample = {"instruction": "查信息", "image_width": 1200, "image_height": 900,
              "actor_step": '  输入“历史上的今天  Angemon\\nA”\n ',
              "current_task": "PRIVATE_OTHER_STEP", "send_prompt": "PRIVATE_PROMPT",
              "actions": [{"action_type": "type_text", "text": "PRIVATE_LABEL"}],
              "actor_step_source": {"path": "PRIVATE_SOURCE"}}
    old = prompt_input(sample, "current_step")
    assert set(old) == {"instruction", "image_width", "image_height"}
    assert "actor_step" not in old
    conditional = prompt_input(sample, "conditional_actor")
    assert set(conditional) == {*old, "actor_step"}
    prompt = build_user_prompt(conditional, "conditional_actor")
    assert prompt.endswith("Current subtask: " + sample["actor_step"])
    assert "PRIVATE" not in prompt
    assert sample["actor_step"] not in build_user_prompt(sample, "current_step")
    changed = sample | {"actions": [], "current_task": "CHANGED", "send_prompt": "CHANGED",
                        "actor_step_source": {"label": "CHANGED"}}
    assert build_user_prompt(sample, "conditional_actor") == build_user_prompt(changed, "conditional_actor")
