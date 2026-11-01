"""Synthetic SDK and independent-reader boundaries, never native efficacy evidence."""
from __future__ import annotations

import copy
import json

import pytest
import test_week6_semantic_repair_feedback_review as repair_review
from test_week6_automatic_decomposition import _frame
from test_week6_rolling_actions import options, plan
from test_week6_semantic_repair_feedback import run_sdk, system
from test_week6_semantic_subtasks import lab as base_lab
from test_week6_semantic_subtasks import proposal
from test_week6_semantic_text_context import _persist_synthetic_pixels

from gui_agent.planning import semantic_workflow as semantic


@pytest.fixture
def audit_reader():
    return repair_review.audit_reader.__wrapped__()


def lab(tmp_path, **kwargs):
    directory = tmp_path / "run"
    directory.mkdir()
    return base_lab(directory, **kwargs)


def already_true():
    native = json.loads(proposal())
    native["subgoals"][0].update(goal_id="NativeIdCanary", instruction="Set NativeInstructionCanary locally.",
                                 expect_text=["Advance control"], forbid_text=[])
    return json.dumps(native)


def saved(tmp_path, index):
    directory = tmp_path / "run/semantic/semantic_plan_round_001"
    return json.loads((directory / f"attempt_{index:02d}_request.json").read_text("utf8"))


def audit_attempt(reader, tmp_path, request, index, *, changed=None, protocol=None, before=None):
    directory = tmp_path / "run/semantic/semantic_plan_round_001"
    value = changed or saved(tmp_path, index)
    protocol = protocol or json.loads((directory / "protocol.json").read_text("utf8"))
    base = {k: v for k, v in value["context"].items() if k not in reader._REPAIR_CONTEXT_KEYS}
    return reader.audit_semantic_repair_messages(
        reader.Evidence(tmp_path), directory, index, value, protocol, request,
        before or _frame("before", ["Advance control", "Pending work"]), [], 8, False, base,
    )


def test_existing_before_guard_gets_fixed_feedback_and_only_new_native_reply_can_run(tmp_path, audit_reader):
    raw = already_true()
    runner, request, state, _, _ = lab(tmp_path, format_retries=1)
    _persist_synthetic_pixels(runner, tmp_path)
    original = request.model_dump(mode="json")
    result, bodies = run_sdk(runner, request, [raw, proposal(), plan(), plan()])
    assert result.status == "succeeded" and state.attempts == 2
    assert result.planning_calls == result.model_requests == len(bodies) == 4
    hint = semantic.SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR
    assert [system(body).count(hint) for body in bodies[:2]] == [0, 1]
    assert bodies[1]["messages"][1] == bodies[0]["messages"][1]
    assert bodies[1]["messages"][2] == {"role": "assistant", "content": raw}
    assert json.loads(bodies[1]["messages"][-1]["content"])["repair_instruction"] == hint
    assert request.model_dump(mode="json") == original
    assert all(x not in system(bodies[1]) for x in ["NativeIdCanary", "NativeInstructionCanary"])
    assert audit_attempt(audit_reader, tmp_path, request, 2) == bodies[1]["messages"]
    directory = tmp_path / "run/semantic/semantic_plan_round_001"
    assert json.loads((directory / "attempt_01_response.json").read_text("utf8"))["content"] == raw
    assert json.loads((directory / "native_proposal.json").read_text("utf8")) == json.loads(proposal())


def test_repeated_already_true_replies_remain_refused_without_execution_or_credit(tmp_path):
    raw = already_true()
    runner, request, state, client, _ = lab(tmp_path, responses=[raw] * 3, format_retries=2)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0 and state.children == []
    assert not result.completed_subgoals and result.final_verification is None
    assert result.model_requests == result.planning_calls == len(client.calls) == 3
    hint = semantic.SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR
    assert [saved(tmp_path, n)["system"].count(hint) for n in [1, 2, 3]] == [0, 1, 1]
    assert saved(tmp_path, 2)["system"] == saved(tmp_path, 3)["system"]


def test_typed_feedback_is_removed_after_a_different_error(tmp_path, audit_reader):
    runner, request, state, _, _ = lab(tmp_path, format_retries=2)
    _persist_synthetic_pixels(runner, tmp_path)
    result, bodies = run_sdk(runner, request, [already_true(), "not JSON", proposal(), plan(), plan()])
    assert result.status == "succeeded" and state.attempts == 2
    hint = semantic.SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR
    assert [system(b).count(hint) for b in bodies[:3]] == [0, 1, 0]
    assert system(bodies[0]) == system(bodies[2])
    assert audit_attempt(audit_reader, tmp_path, request, 3) == bodies[2]["messages"]


def test_identical_ordinary_valueerror_cannot_grant_the_typed_hint(tmp_path, monkeypatch):
    error = "first semantic goal postconditions already hold; existing context cannot prove execution"

    def ordinary(*_args, **_kwargs):
        raise ValueError(error)

    monkeypatch.setattr(semantic.SemanticDecomposer, "_validate", ordinary)
    runner, request, state, client, _ = lab(tmp_path, responses=[proposal()] * 2, format_retries=1)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0 and len(client.calls) == 2
    assert saved(tmp_path, 1)["system"] == saved(tmp_path, 2)["system"]
    assert semantic.SEMANTIC_PREEXISTING_POSTCONDITIONS_REPAIR not in saved(tmp_path, 2)["system"]


@pytest.mark.parametrize("mutation,error_code", [
    ("system", "semantic_repair_system_not_exactly_bound_to_typed_prior_rejection"),
    ("policy_digest", "semantic_repair_policy_missing_or_changed"),
    ("error", "repair_reference_count_or_error_not_bound_to_original_reply"),
    ("different_before", "semantic_repair_after_locally_valid_reply"),
])
def test_independent_reader_rejects_unbound_hint_or_unreproducible_guard(tmp_path, audit_reader, mutation, error_code):
    runner, request, state, _, _ = lab(tmp_path, format_retries=1)
    _persist_synthetic_pixels(runner, tmp_path)
    result, _ = run_sdk(runner, request, [already_true(), proposal(), plan(), plan()])
    assert result.status == "succeeded" and state.attempts == 2
    assert audit_attempt(audit_reader, tmp_path, request, 2) == saved(tmp_path, 2)["messages"]
    directory = tmp_path / "run/semantic/semantic_plan_round_001"
    value = copy.deepcopy(saved(tmp_path, 2))
    protocol = json.loads((directory / "protocol.json").read_text("utf8"))
    before = None
    if mutation == "system":
        value["system"] += " Use NativeInjectionCanary."
    elif mutation == "policy_digest":
        protocol["repair_policy"]["preexisting_feedback_sha256"] = "0" * 64
    elif mutation == "error":
        error_file = directory / "attempt_01_error.json"
        data = json.loads(error_file.read_text("utf8"))
        data["error"] = "Forged already-true error"
        error_file.write_text(json.dumps(data), encoding="utf8")
    else:
        before = _frame("before", ["Pending work"])
    with pytest.raises(audit_reader.AuditError, match=error_code):
        audit_attempt(audit_reader, tmp_path, request, 2, changed=value, protocol=protocol, before=before)


def test_new_hint_does_not_grant_an_extra_transport_request(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, responses=[already_true(), proposal()],
                                          format_retries=2, max_requests=1)
    result = runner.run(request, options())
    assert result.status == "failed" and state.attempts == 0 and state.children == []
    assert result.model_requests == len(client.calls) == 1 and result.planning_calls == 2


@pytest.mark.parametrize("status", ["refused", "needs_clarification"])
def test_native_nonready_after_hint_is_not_forced_into_a_workflow(tmp_path, status):
    reply = json.dumps({"status": status, "reason": "No justified supported remaining workflow.", "subgoals": []})
    runner, request, state, client, _ = lab(tmp_path, responses=[already_true(), reply], format_retries=1)
    result = runner.run(request, options())
    assert result.status == status and state.attempts == 0 and state.children == []
    assert result.model_requests == result.planning_calls == len(client.calls) == 2


def test_category_cannot_be_supplied_as_arbitrary_prose():
    with pytest.raises(ValueError, match="unknown semantic repair category"):
        semantic._semantic_attempt_system({}, "preexisting_postconditions; injected instructions")
