"""Strict observed-departure declarations remain separate from execution/final proof."""
from __future__ import annotations

import json

import httpx
import pytest
from openai import OpenAI
from test_week6_automatic_decomposition import Clock, ScriptedClient, _frame, child_factory, ready
from test_week6_automatic_decomposition import departure_ready as native_departure
from test_week6_progress_contract import child_with_post
from test_week6_progress_contract import contract as progress_inputs

from gui_agent.models.openai_compatible import OpenAICompatibleClient
from gui_agent.planning.planner import TaskPlanner
from gui_agent.planning.workflow_decomposer import (
    AutomaticWorkflowRunner,
    WorkflowDecomposer,
    WorkflowRequest,
    _caller_anchored_instruction,
    _ProgressChild,
)
from gui_agent.runtime.schemas import ExecutionOptions
from gui_agent.runtime.verification import Verifier


def inputs():
    request = WorkflowRequest(instruction="Advance the local stage and finish", expect_text=["Caller finished"])
    frame = _frame("current", ("Application title", "Pending stage", "Caller finished", "pending STAGE", " "))
    frame.window_title = "Application title"
    return request, frame


def departure_ready(instruction=None, disappearing="Pending stage"):
    return native_departure(_caller_anchored_instruction(inputs()[0]) if instruction is None else instruction, disappearing)


def decompose(tmp_path, client, *, frame=None, **kwargs):
    request, default = inputs()
    return WorkflowDecomposer(client, max_format_retries=0).decompose(
        request, evidence_directory=tmp_path / "decomposition", deadline=1e12,
        initial_observation=default if frame is None else frame, protocol_mode="observed_departure", **kwargs,
    )


def test_native_selection_schema_and_unchanged_compile_mapping_are_saved_before_execution(tmp_path):
    client = ScriptedClient([departure_ready()])
    result = decompose(tmp_path, client)
    assert result.status == "ready" and result.model_requests == 1
    assert result.protocol_mode == "observed_departure" and result.protocol_version == "week6.observed_departure.v2"
    directory = tmp_path / "decomposition"
    protocol = json.loads((directory / "observed_departure_protocol.json").read_text("utf-8"))
    assert protocol["candidates"] == ["Pending stage"]
    assert protocol["bound_observation"]["observation_id"] == "current"
    response_format = client.calls[0]["kwargs"]["response_format"]
    assert response_format == protocol["response_format"]
    schema = response_format["json_schema"]["schema"]
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["strict"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["subgoals"]["items"]["properties"]["disappearing_text"]["enum"] == ["Pending stage"]
    native = json.loads((directory / "native_proposal.json").read_text("utf-8"))
    assert native == json.loads(departure_ready()) == result.native_proposal
    mapping = json.loads((directory / "compiled_proposal.json").read_text("utf-8"))
    assert mapping["native_proposal"] == native
    assert mapping["mapping"] == [{"native_instruction": native["subgoals"][0]["instruction"],
                                  "compiled_instruction": native["subgoals"][0]["instruction"],
                                  "native_disappearing_text": "Pending stage", "expect_text": [], "forbid_text": ["Pending stage"]}]
    goal = result.workflow.subgoals[0]
    assert goal.expect_text == [] and goal.forbid_text == [native["subgoals"][0]["disappearing_text"]]
    assert Verifier().check_task(goal, inputs()[1]).outcome == "failed"
    assert Verifier().check_task(goal, _frame("post", ("Next stage",))).passed
    assert not Verifier().check_task(result.workflow.final_check, _frame("fresh-final", ("Next stage",))).passed
    assert json.loads((directory / "generated_workflow.json").read_text("utf-8")) == result.workflow.model_dump(mode="json")


@pytest.mark.parametrize("candidate", ["unobserved marker", "pending stage", " Pending stage", "", "Caller finished", "Application title"])
def test_schema_decoder_cannot_bypass_exact_current_candidate_and_final_constraints(tmp_path, candidate):
    client = ScriptedClient([departure_ready(disappearing=candidate)])
    result = decompose(tmp_path, client)
    assert result.status == "failed" and result.workflow is None and result.model_requests == 1
    assert not (tmp_path / "decomposition/compiled_proposal.json").exists()
    assert json.loads((tmp_path / "decomposition/attempt_01_response.json").read_text("utf-8"))["content"] == departure_ready(disappearing=candidate)


@pytest.mark.parametrize("mutation", ["extra_goal", "no_goal", "extra_key", "legacy_checks", "duplicate_key", "missing_reason", "message", "coordinates"])
def test_native_schema_and_local_validation_both_refuse_invalid_declarations(tmp_path, mutation):
    proposal = json.loads(departure_ready())
    if mutation == "extra_goal":
        proposal["subgoals"] *= 2
    elif mutation == "no_goal":
        proposal["subgoals"] = []
    elif mutation == "extra_key":
        proposal["final_check"] = {"expect_text": ["fake success"]}
    elif mutation == "legacy_checks":
        proposal["subgoals"][0]["expect_text"] = ["invented future"]
    elif mutation == "missing_reason":
        proposal.pop("reason")
    elif mutation == "message":
        proposal["subgoals"][0]["instruction"] = "Send a message"
    elif mutation == "coordinates":
        proposal["subgoals"][0]["instruction"] = "Click (80, 100)"
    text = json.dumps(proposal)
    if mutation == "duplicate_key":
        text = text.replace('"status": "ready"', '"status":"ready","status":"refused"')
    result = decompose(tmp_path, ScriptedClient([text]))
    assert result.status == "failed" and result.workflow is None


@pytest.mark.parametrize("status", ["refused", "needs_clarification"])
def test_nonready_native_declaration_has_reason_and_no_compiled_executable_goal(tmp_path, status):
    native = {"status": status, "reason": "No justified observed departure", "subgoals": []}
    result = decompose(tmp_path, ScriptedClient([json.dumps(native)]))
    assert result.status == status and result.workflow is None and result.native_proposal == native
    mapping = json.loads((tmp_path / "decomposition/compiled_proposal.json").read_text("utf-8"))
    assert mapping["mapping"] == [] and mapping["compiled_proposal"]["subgoals"] == []


@pytest.mark.parametrize("fault", ["no_candidates", "no_elements", "perception_error", "no_window", "unstable", "unsupported_provider"])
def test_unavailable_snapshot_or_schema_transport_stops_before_any_model_request(tmp_path, fault):
    _, frame = inputs()
    client = ScriptedClient([])
    if fault == "no_candidates":
        frame.elements = [item for item in frame.elements if item.text in {"Application title", "Caller finished"}]
    elif fault == "no_elements":
        frame.elements = []
    elif fault == "perception_error":
        frame.errors.append("OCR unavailable")
    elif fault == "no_window":
        frame.window_id = ""
    elif fault == "unstable":
        frame.foreground_stable = False
    else:
        client.supports_strict_json_schema = False
    result = decompose(tmp_path, client, frame=frame)
    assert result.status == "failed" and result.model_requests == result.format_attempts == 0
    assert client.calls == [] and result.workflow is None
    assert (tmp_path / "decomposition/result.json").is_file()


def test_endpoint_rejecting_json_schema_retains_http_failure_without_plain_json_fallback(tmp_path):
    requests = []

    def endpoint(request):
        requests.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "json_schema is unsupported", "type": "invalid_request_error"}})

    client = OpenAICompatibleClient(model_name="offline", api_key="offline", max_retries=0)
    client._client = OpenAI(api_key="offline", base_url="http://offline.invalid/v1", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))
    result = decompose(tmp_path, client)
    assert result.status == "failed" and result.model_requests == len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"
    transport = json.loads((tmp_path / "decomposition/transport_requests.jsonl").read_text("utf-8"))
    assert transport["http_responses"][0]["status_code"] == 400
    assert "unsupported" in transport["http_responses"][0]["body"]["error"]["message"]


def test_default_api_keeps_postconditions_and_does_not_silently_compile_legacy_checks(tmp_path):
    request, frame = inputs()
    client = ScriptedClient([ready()])
    result = WorkflowDecomposer(client).decompose(request, evidence_directory=tmp_path / "legacy", deadline=1e12,
                                                 initial_observation=frame)
    assert result.status == "ready" and result.protocol_mode == "postconditions"
    assert client.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}
    assert result.native_proposal is None and result.workflow.subgoals[0].expect_text == ["Stage 1"]
    assert not (tmp_path / "legacy/compiled_proposal.json").exists()


@pytest.mark.parametrize("change", ["window", "substring_only"])
def test_automatic_selected_marker_and_scope_must_still_match_exact_fresh_before(tmp_path, change):
    request, initial = inputs()
    before = _frame("fresh-before", ("Pending stage extension",) if change == "substring_only" else ("Pending stage",))
    if change == "window":
        before.window_id = "another-window"
    frames = iter((initial, before))
    result = AutomaticWorkflowRunner(tmp_path / "automatic", ScriptedClient([departure_ready()]),
                                     lambda *_: pytest.fail("no child input after scope changed"),
                                     initial_observer=lambda _: next(frames)).run(
        request, ExecutionOptions(execute=True, max_planning_attempts=4))
    assert result.status == "failed" and result.model_requests == 1 and result.total_actions == 0
    assert result.decomposition_protocol_mode == "observed_departure"
    assert result.workflow_result is None and "changed before contract freezing" in result.error


def test_native_reason_cannot_turn_a_proposal_into_user_final_success(tmp_path):
    native = json.loads(departure_ready())
    native["reason"] = "All stages completed"
    result = decompose(tmp_path, ScriptedClient([json.dumps(native)]))
    assert result.status == "ready" and result.workflow.final_check.expect_text == ["Caller finished"]
    assert not Verifier().check_task(result.workflow.final_check, _frame("independent-final", ("Next stage",))).passed


def test_observed_protocol_requires_its_own_initial_snapshot_before_dispatch(tmp_path):
    request, _ = inputs()
    client = ScriptedClient([])
    result = WorkflowDecomposer(client).decompose(request, evidence_directory=tmp_path / "missing", deadline=1e12,
                                                 protocol_mode="observed_departure")
    assert result.status == "failed" and result.model_requests == 0 and client.calls == []
    assert "requires" in result.error


def test_constraint_repair_keeps_candidate_schema_original_reply_and_shared_request_cap(tmp_path):
    request, frame = inputs()
    client = ScriptedClient([departure_ready(disappearing="Unseen stage"), departure_ready(), departure_ready()])
    result = WorkflowDecomposer(client).decompose(request, evidence_directory=tmp_path / "repair", deadline=1e12,
                                                 initial_observation=frame, protocol_mode="observed_departure",
                                                 max_model_requests=2)
    assert result.status == "ready" and result.model_requests == result.format_attempts == 2
    assert len(client.calls) == 2 and client.calls[0]["kwargs"] == client.calls[1]["kwargs"]
    correction = json.loads(client.calls[1]["messages"][-1]["content"])["context"]
    assert correction["previous_response"] == departure_ready(disappearing="Unseen stage")
    assert correction["final_checks"] == {"expect_text": request.expect_text, "forbid_text": request.forbid_text}
    assert "exact current candidate" in correction["constraint_or_format_error"]


@pytest.mark.parametrize("mutation", ["rewrite", "leading_space", "trailing_space", "truncated"])
def test_native_instruction_cannot_rewrite_or_normalize_the_caller_anchor(tmp_path, mutation):
    anchor = _caller_anchored_instruction(inputs()[0])
    instruction = {"rewrite": "Choose an unrelated option", "leading_space": " " + anchor,
                   "trailing_space": anchor + " ", "truncated": anchor[:-5]}[mutation]
    result = decompose(tmp_path, ScriptedClient([departure_ready(instruction)]))
    assert result.status == "failed" and result.workflow is None
    assert "exactly match" in result.error


def test_full_original_caller_request_is_recorded_or_explicitly_refused_without_truncation(tmp_path):
    request = WorkflowRequest(instruction="Long original request " + "z" * 3900, expect_text=["Finished"])
    client = ScriptedClient([])
    result = WorkflowDecomposer(client).decompose(request, evidence_directory=tmp_path / "long", deadline=1e12,
                                                 protocol_mode="observed_departure", initial_observation=inputs()[1])
    assert result.status == "failed" and result.model_requests == 0 and client.calls == []
    assert "will not be truncated" in result.error
    assert json.loads((tmp_path / "long/request.json").read_text("utf-8"))["instruction"] == request.instruction


def step_plan(count):
    return json.dumps({"task_id": "one", "instruction": "Advance current stage", "steps": [
        {"step_id": f"s{number}", "description": "Choose the requested option", "action_type": "click",
         "target_text": "Available control", "arguments": {}} for number in range(count)
    ]})


@pytest.mark.parametrize("count", [1, 2])
def test_real_task_planner_enforces_one_step_per_call_and_restores_original_limit(tmp_path, count):
    _, raw, _, derived, contract = progress_inputs(tmp_path)
    child = child_with_post(tmp_path, _frame("post", ("Next stage",)))
    client = ScriptedClient([step_plan(count)])
    child.planner = TaskPlanner(client, max_steps=7, max_format_retries=0)
    original_run = child.run
    calls = []

    def run(task, options):
        result = child.planner.plan(task.instruction)
        calls.append(result)
        outcome = original_run(task, options)
        return outcome if result.ok else outcome.model_copy(update={"status": "blocked", "steps": []})

    child.run = run
    wrapper = _ProgressChild(child, raw.subgoals[0], derived.subgoals[0], contract, [], 1, single_step_planning=True)
    outcome = wrapper.run(derived.subgoals[0], ExecutionOptions(execute=True))
    assert child.planner.max_steps == 7 and "plan" not in child.planner.__dict__
    assert json.loads(client.calls[0]["messages"][-1]["content"])["context"]["max_steps"] == 1
    assert calls[0].ok is (count == 1)
    assert (outcome.status == "succeeded") is (count == 1)
    saved = json.loads((tmp_path / "child/single_step_planning_limit.json").read_text("utf-8"))
    assert saved["original_max_steps"] == saved["restored_max_steps"] == 7 and saved["restored"]
    assert saved["applied_max_steps"] == saved["calls"][0]["max_steps"] == 1
    assert saved["calls"][0]["returned_steps"] == (1 if count == 1 else None)


@pytest.mark.parametrize("failure", [RuntimeError("child failed"), KeyboardInterrupt()])
def test_single_step_limit_is_restored_when_child_raises_or_is_cancelled(tmp_path, failure):
    _, raw, _, derived, contract = progress_inputs(tmp_path)
    child = child_with_post(tmp_path, _frame("post", ("Next stage",)))
    child.planner = TaskPlanner(ScriptedClient([]), max_steps=6)

    def fail(*_args, **_kwargs):
        assert child.planner.max_steps == 1
        raise failure

    child.run = fail
    wrapper = _ProgressChild(child, raw.subgoals[0], derived.subgoals[0], contract, [], 1, single_step_planning=True)
    with pytest.raises(type(failure)):
        wrapper.run(derived.subgoals[0], ExecutionOptions(execute=True))
    assert child.planner.max_steps == 6 and "plan" not in child.planner.__dict__
    assert json.loads((tmp_path / "child/single_step_planning_limit.json").read_text("utf-8"))["restored"]


def test_custom_child_without_controlled_real_planner_is_refused_before_run(tmp_path):
    _, raw, _, derived, contract = progress_inputs(tmp_path)
    child = child_with_post(tmp_path, _frame("post", ("Next stage",)))
    child.run = lambda *_a, **_k: pytest.fail("unsupported child must not run or observe")
    with pytest.raises(ValueError, match="real TaskPlanner"):
        _ProgressChild(child, raw.subgoals[0], derived.subgoals[0], contract, [], 1, single_step_planning=True)


@pytest.mark.parametrize("repeat", [False, True])
def test_same_caller_anchor_allows_new_stage_marker_but_never_replays_completed_marker(tmp_path, repeat):
    request, initial = inputs()
    clock, state = Clock(), []
    second_marker = "Pending stage" if repeat else "Next stage"
    observations = iter((initial, _frame("before-1", ("Pending stage",)),
                         _frame("initial-2", (second_marker,)), _frame("before-2", (second_marker,))))
    client = ScriptedClient([departure_ready(), "plan one", departure_ready(disappearing=second_marker), "plan two"])
    result = AutomaticWorkflowRunner(
        tmp_path / "stages", client,
        child_factory(state, clock, action_attempts=1,
                      final_text=lambda: "Next stage" if len(state) == 1 else "Caller finished"),
        clock=clock, initial_observer=lambda _: next(observations), max_decomposition_rounds=2,
    ).run(request, ExecutionOptions(execute=True, max_planning_attempts=6, max_actions=3))
    if repeat:
        assert result.status == "blocked" and "repeated" in result.error and len(state) == 1
        assert result.total_actions == 1 and result.model_requests == 3
    else:
        assert result.status == "succeeded" and len(state) == result.total_actions == 2
        assert state[0]["task"].instruction == state[1]["task"].instruction == _caller_anchored_instruction(request)
        assert result.model_requests == result.planning_calls == 4
        request_two = json.loads(client.calls[2]["messages"][-1]["content"])["context"]
        assert "instruction" not in request_two["completed_subgoals"][0]
        assert request_two["completed_subgoals"][0]["progress_checks"]["forbid_text"] == ["Pending stage"]
        assert len(result.completed_subgoals) == 2 and result.workflow_result.final_verification.passed


def test_shared_http_and_action_budget_can_stop_before_a_second_anchored_stage(tmp_path):
    request, initial = inputs()
    clock, state = Clock(), []
    client = ScriptedClient([departure_ready(), "plan one", departure_ready(disappearing="Next stage")])
    result = AutomaticWorkflowRunner(
        tmp_path / "budget", client, child_factory(state, clock, action_attempts=1, final_text="Next stage"),
        clock=clock, initial_observer=lambda _: initial, max_model_requests=2,
    ).run(request, ExecutionOptions(execute=True, max_planning_attempts=8, max_actions=3))
    assert result.status == "blocked" and result.total_actions == 1
    assert result.model_requests == result.planning_calls == len(client.calls) == 2 and len(state) == 1
    assert "model request budget" in result.error


def test_delayed_anchored_child_factory_cannot_extend_original_deadline_or_observe(tmp_path):
    from test_runtime_runner import FakeExecutor, FakeObserver

    from gui_agent.recording import RunSession
    from gui_agent.runtime.action_adapter import ActionAdapter
    from gui_agent.runtime.recorder import TaskRecorder
    from gui_agent.runtime.runner import TaskRunner

    request, initial = inputs()
    clock, created = Clock(), []
    observer, executor = FakeObserver([initial]), FakeExecutor()
    client = ScriptedClient([departure_ready()])

    def factory(_task, _index, directory, budgeted):
        clock.advance(6)
        planner = TaskPlanner(budgeted, max_steps=9, max_format_retries=0)
        child = TaskRunner(observer=observer, planner=planner, executor=executor,
                           adapter=ActionAdapter(platform="win32"), verifier=Verifier(),
                           recorder=TaskRecorder(RunSession(session_id=directory.name, directory=directory)),
                           clock=clock, sleep=lambda _: None, absolute_deadline=budgeted.deadline)
        created.append(child)
        return child

    result = AutomaticWorkflowRunner(tmp_path / "deadline", client, factory, clock=clock,
                                     initial_observer=lambda _: initial).run(
        request, ExecutionOptions(execute=True, confirm=False, max_planning_attempts=6, task_timeout_seconds=5))
    assert result.status == "timed_out" and result.model_requests == result.planning_calls == 1
    assert observer.calls == 0 and executor.actions == []
    assert created[0].absolute_deadline == 105 and created[0].planner.max_steps == 9


def test_limit_audit_failure_restores_planner_and_keeps_known_receipts(tmp_path, monkeypatch):
    from gui_agent.planning import workflow_decomposer as module

    _, raw, _, derived, contract = progress_inputs(tmp_path)
    child = child_with_post(tmp_path, _frame("post", ("Next stage",)))
    child.planner = TaskPlanner(ScriptedClient([]), max_steps=6)
    original = module._write_json

    def fail(path, value):
        if path.name == "single_step_planning_limit.json" and value["restored"]:
            raise PermissionError("injected final limit audit write failure")
        return original(path, value)

    monkeypatch.setattr(module, "_write_json", fail)
    result = _ProgressChild(child, raw.subgoals[0], derived.subgoals[0], contract, [], 1,
                            single_step_planning=True).run(derived.subgoals[0], ExecutionOptions(execute=True))
    assert result.status == "failed" and result.action_count == result.planning_calls == 1
    assert child.planner.max_steps == 6 and "plan" not in child.planner.__dict__
    assert "single-step limit audit failed" in result.error
