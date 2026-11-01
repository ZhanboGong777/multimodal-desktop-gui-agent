"""Predicted intermediate observations are fixed contracts, not model proof.

All observations, receipts and replies are offline doubles. Native goal parsing
and real runtime execution are exercised without a model, capture or OS input.
"""

from __future__ import annotations

import json

import pytest
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal


def test_newly_appearing_control_may_witness_a_stage_without_banning_all_captions(tmp_path):
    native = json.loads(proposal())
    native["subgoals"][0].update(instruction="Reach the next configuration stage.", expect_text=["Continue"])
    runner, request, state, _, _ = lab(tmp_path, responses=[json.dumps(native), plan(), plan()],
                                      extra_text_at_stage={1: ["Continue"]})
    request.expect_text, request.forbid_text = ["Final accepted"], []
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 2
    assert result.completed_subgoals[0]["expect_text"] == ["Continue"]
    directory = tmp_path / "semantic"
    before = json.loads((directory / "goal_contract_001_001.json").read_text("utf8"))["before"]
    assert "Continue" not in [element["text"] for element in before["elements"]]
    assert result.final_verification.passed


def test_long_single_caller_final_is_preserved_while_short_model_contracts_execute(tmp_path):
    final_text = ("Complete preserved output: " + "long literal final condition " * 9).rstrip()
    assert len(final_text) > 200
    runner, request, state, client, _ = lab(tmp_path, extra_text_at_stage={2: [final_text]})
    request.expect_text, request.forbid_text = [final_text], []
    original = request.model_dump(mode="json")
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 2 and len(result.completed_subgoals) == 2
    assert result.final_verification.passed and request.model_dump(mode="json") == original
    context = json.loads(client.calls[0]["messages"][-1]["content"])["context"]
    assert context["original_caller_request"] == original
    compiled = json.loads((tmp_path / "semantic/semantic_plan_round_001/compiled_workflow.json").read_text("utf8"))
    assert compiled["workflow"]["final_check"]["expect_text"] == [final_text]
    assert all(len(text) <= 200 for goal in compiled["workflow"]["subgoals"] for text in goal["expect_text"])


@pytest.mark.parametrize("instruction", [" Set the requested preference", "Set the requested preference "])
def test_raw_goal_instruction_is_not_silently_stripped_before_compilation(tmp_path, instruction):
    native = json.loads(proposal())
    native["subgoals"][0]["instruction"] = instruction
    content = json.dumps(native)
    runner, request, state, _, _ = lab(tmp_path, responses=[content])
    result = runner.run(request, options())
    assert result.status == "failed" and "without stripping" in result.error
    assert state.children == [] and state.attempts == 0 and result.model_requests == 1
    directory = tmp_path / "semantic/semantic_plan_round_001"
    assert json.loads((directory / "attempt_01_response.json").read_text("utf8"))["content"] == content
    assert not (directory / "compiled_workflow.json").exists()


def test_empty_predicted_contract_cannot_be_credited_or_replaced_with_caller_final(tmp_path):
    native = json.loads(proposal())
    native["subgoals"][0].update(expect_text=[], forbid_text=[])
    runner, request, state, _, _ = lab(tmp_path, responses=[json.dumps(native)])
    result = runner.run(request, options())
    assert result.status == "failed" and state.children == [] and state.attempts == 0
    assert result.model_requests == 1 and not result.completed_subgoals and result.final_verification is None
