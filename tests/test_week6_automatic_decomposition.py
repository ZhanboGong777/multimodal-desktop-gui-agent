"""Automatic decomposition: adversarial responses, independent checks, real HTTP accounting."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from test_runtime_runner import FakeExecutor, FakeObserver, FakePlanner, _frame

from gui_agent.models.base import ModelClient, ModelError, ModelResponse
from gui_agent.models.openai_compatible import OpenAICompatibleClient
from gui_agent.planning.workflow_decomposer import (
    AutomaticWorkflowRunner,
    BudgetedModelClient,
    DecompositionProposal,
    WorkflowDecomposer,
    WorkflowRequest,
)
from gui_agent.recording import RunSession
from gui_agent.runtime.action_adapter import ActionAdapter
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.runner import TaskRunner
from gui_agent.runtime.schemas import (
    ExecutionOptions,
    StepRecord,
    TaskRunResult,
    TaskSpec,
    VerificationResult,
)
from gui_agent.runtime.verification import Verifier


class Clock:
    def __init__(self):
        self.seconds = 100.0

    def __call__(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds


def ready(*instructions):
    return json.dumps({"status": "ready", "subgoals": [
        {"instruction": instruction, "expect_text": [f"Stage {index}"], "forbid_text": []}
        for index, instruction in enumerate(instructions or ("Open the local editor",), 1)
    ]})


class ScriptedClient(ModelClient):
    name = "test_explicit_transport"
    transport_attempts_are_explicit = True

    def __init__(self, responses, *, clock=None, costs=(), max_retries=0):
        super().__init__(model_name="offline-scripted", max_retries=max_retries, timeout_seconds=90)
        self.responses = list(responses)
        self.clock, self.costs, self.calls = clock, list(costs), []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs, "timeout": self.timeout_seconds})
        if self.clock and self.costs:
            self.clock.advance(self.costs.pop(0))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return ModelResponse(content=response, provider=self.name, model_name=self.model_name,
                             raw_response={"unabridged": response})


@pytest.fixture
def request_spec():
    return WorkflowRequest(workflow_id="LocalDemo", instruction="Open an editor and finish a local draft",
                           expect_text=["USER_FINAL"], forbid_text=["USER_ERROR"])


@pytest.mark.parametrize("mutation", [
    {"instruction": " "}, {"instruction": "send a message"}, {"instruction": "发送邮件"},
    {"expect_text": [], "forbid_text": []}, {"expect_text": [" "]},
    {"workflow_id": "../old"}, {"workflow_id": "CON"}, {"extra": "untrusted"},
    {"expect_text": ["same"], "forbid_text": ["same"]},
])
def test_request_requires_safe_identity_clear_scope_and_caller_final_rules(request_spec, mutation):
    with pytest.raises(ValueError):
        WorkflowRequest.model_validate({**request_spec.model_dump(), **mutation})


def test_actual_model_proposal_order_retained_and_user_final_rules_cannot_be_rewritten(tmp_path, request_spec):
    client = ScriptedClient([ready("Open the editor", "Enter a draft", "Complete the local draft")])
    result = WorkflowDecomposer(client).decompose(request_spec, evidence_directory=tmp_path / "dec", deadline=1e12)
    assert result.status == "ready" and result.model_requests == result.format_attempts == 1
    assert [goal.instruction for goal in result.workflow.subgoals] == [
        "Open the editor", "Enter a draft", "Complete the local draft",
    ]
    assert result.workflow.final_check.expect_text == ["USER_FINAL"]
    assert result.workflow.final_check.forbid_text == ["USER_ERROR"]
    assert all(goal.risk == "medium" for goal in result.workflow.subgoals)
    result.workflow.subgoals[-1].expect_text.append("invented completion")
    assert result.workflow.final_check.expect_text == ["USER_FINAL"]
    raw = json.loads((tmp_path / "dec/attempt_01_response.json").read_text("utf-8"))
    assert raw["content"] == ready("Open the editor", "Enter a draft", "Complete the local draft")
    assert raw["raw_response"]["unabridged"] == raw["content"]
    assert result.intermediate_check_source == "model_proposal_untrusted"
    assert result.final_check_source == "caller_supplied_unchanged"


@pytest.mark.parametrize("bad", [
    '```json\n{"status":"ready","subgoals":[]}\n```',
    '{"status":"ready","status":"refused","subgoals":[]}',
    json.dumps({"status": "ready", "subgoals": []}),
    json.dumps({"status": "ready", "final_check": {"expect_text": ["fake"]}, "subgoals": []}),
    json.dumps({"status": "ready", "subgoals": [{"instruction": "Open editor", "expect_text": []}]}),
    json.dumps({"status": "ready", "subgoals": [{"instruction": "Click (100, 200)", "expect_text": ["ok"]}]}),
    json.dumps({"status": "ready", "subgoals": [{"instruction": "Click obs-0001-e001", "expect_text": ["ok"]}]}),
    json.dumps({"status": "ready", "subgoals": [{"instruction": "Send a message", "expect_text": ["sent"]}]}),
    json.dumps({"status": "ready", "subgoals": [{"instruction": "Open editor", "expect_text": ["ok"], "max_actions": 999}]}),
    json.dumps({"status": "needs_clarification", "reason": "", "subgoals": []}),
])
def test_invalid_response_is_not_executable_and_every_raw_error_is_retained(tmp_path, request_spec, bad):
    client = ScriptedClient([bad])
    result = WorkflowDecomposer(client, max_format_retries=0).decompose(
        request_spec, evidence_directory=tmp_path / "dec", deadline=1e12,
    )
    assert result.status == "failed" and result.workflow is None
    assert result.model_requests == 1 and result.error
    assert json.loads((tmp_path / "dec/attempt_01_response.json").read_text("utf-8"))["content"] == bad
    assert json.loads((tmp_path / "dec/attempt_01_error.json").read_text("utf-8"))["error"]
    assert not (tmp_path / "dec/generated_workflow.json").exists()


def test_format_repair_has_original_instruction_fixed_final_checks_and_counted_requests(tmp_path, request_spec):
    client = ScriptedClient(["invalid JSON", ready("Open editor")])
    result = WorkflowDecomposer(client).decompose(request_spec, evidence_directory=tmp_path / "dec", deadline=1e12)
    assert result.status == "ready" and result.decomposition_calls == 1
    assert result.format_attempts == result.model_requests == len(client.calls) == 2
    second = json.loads(client.calls[1]["messages"][-1]["content"])
    assert second["instruction"] == request_spec.instruction
    assert second["context"]["previous_response"] == "invalid JSON"
    assert second["context"]["final_checks"] == {"expect_text": ["USER_FINAL"], "forbid_text": ["USER_ERROR"]}
    assert (tmp_path / "dec/attempt_01_error.json").is_file()
    assert (tmp_path / "dec/attempt_02_response.json").is_file()


@pytest.mark.parametrize("status", ["refused", "needs_clarification"])
def test_model_can_refuse_ambiguous_or_unsupported_task_without_inventing_goals(tmp_path, request_spec, status):
    client = ScriptedClient([json.dumps({"status": status, "reason": "target application unclear", "subgoals": []})])
    result = WorkflowDecomposer(client).decompose(request_spec, evidence_directory=tmp_path / "dec", deadline=1e12)
    assert result.status == status and result.workflow is None and len(client.calls) == 1


def test_subgoal_limit_and_retry_limit_are_bounded_and_strict():
    for kwargs in ({"max_subgoals": 0}, {"max_subgoals": 33}, {"max_subgoals": True},
                   {"max_format_retries": -1}, {"max_format_retries": 3}):
        with pytest.raises(ValueError):
            WorkflowDecomposer(ScriptedClient([]), **kwargs)
    with pytest.raises(ValueError):
        DecompositionProposal.model_validate_json(ready(*[f"stage {i}" for i in range(33)]))


def test_transport_retry_and_format_retry_cannot_exceed_shared_request_budget(tmp_path, request_spec):
    client = ScriptedClient([ModelError("temporary transport failure"), "bad JSON", ready("Never reached")], max_retries=1)
    result = WorkflowDecomposer(client).decompose(request_spec, evidence_directory=tmp_path / "dec", deadline=1e12,
                                                 max_model_requests=2)
    assert result.status == "failed" and result.model_requests == len(client.calls) == 2
    records = [json.loads(line) for line in (tmp_path / "dec/transport_requests.jsonl").read_text("utf-8").splitlines()]
    assert len(records) == 2 and "temporary transport failure" in records[0]["error"]
    assert records[1]["response"]["content"] == "bad JSON"


def test_expired_decomposition_sends_zero_requests(tmp_path, request_spec):
    clock, client = Clock(), ScriptedClient([])
    result = WorkflowDecomposer(client, clock=clock).decompose(request_spec, evidence_directory=tmp_path / "dec", deadline=100)
    assert result.status == "timed_out" and result.model_requests == result.format_attempts == 0
    assert client.calls == []


def test_late_provider_reply_is_kept_but_never_accepted(tmp_path, request_spec):
    clock = Clock()
    client = ScriptedClient([ready("Open editor")], clock=clock, costs=[6])
    result = WorkflowDecomposer(client, clock=clock).decompose(request_spec, evidence_directory=tmp_path / "dec", deadline=105)
    assert result.status == "timed_out" and result.workflow is None
    assert result.model_requests == 1 and result.elapsed_ms == 6000
    assert client.calls[0]["timeout"] == 5 and client.timeout_seconds == 90
    raw = json.loads((tmp_path / "dec/transport_requests.jsonl").read_text("utf-8"))
    assert raw["response"]["content"] == ready("Open editor")


def test_real_sdk_http_hook_disables_hidden_retries_and_limits_timeout_and_requests(tmp_path):
    clock = Clock()
    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "injected", "type": "server_error"}})

    client = OpenAICompatibleClient(api_key="local-test", base_url="http://fixture.invalid/v1",
                                    model_name="test", max_retries=2, timeout_seconds=90, environ={})
    sdk = OpenAI(api_key="local-test", max_retries=4, timeout=90,
                 http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))
    client._client = sdk
    before_hooks = list(sdk._client.event_hooks["request"])
    budgeted = BudgetedModelClient(client, max_requests=1, deadline=120, clock=clock,
                                  audit_path=tmp_path / "http.jsonl")
    response = budgeted.generate_text("one bounded call")
    assert not response.ok and len(requests) == budgeted.request_count == client.request_count == 1
    assert all(0 < value <= 20 for value in requests[0].extensions["timeout"].values())
    assert client._client is sdk and sdk.max_retries == 4
    assert client.timeout_seconds == 90 and client.max_retries == 2
    assert sdk._client.event_hooks["request"] == before_hooks
    raw = json.loads((tmp_path / "http.jsonl").read_text("utf-8"))
    assert raw["count_source"] == "httpx_request_event" and raw["requests_made"] == 1
    assert "injected" in raw["error"]
    sdk.close()


def test_sdk_rechecks_deadline_after_payload_preparation_before_actual_http():
    clock = Clock()
    requests = []
    client = OpenAICompatibleClient(api_key="local", model_name="test", max_retries=0, environ={})
    sdk = OpenAI(api_key="local", http_client=httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(500),
    )))
    client._client = sdk
    original = client.to_vision_messages

    def late_payload(messages, image):
        clock.advance(11)
        return original(messages, image)

    client.to_vision_messages = late_payload
    response = BudgetedModelClient(client, max_requests=3, deadline=110, clock=clock).generate_text("late")
    assert not response.ok and requests == [] and client.request_count == 0
    sdk.close()


def test_unknown_sdk_transport_is_refused_instead_of_claiming_actual_http_counts():
    client = SimpleNamespace(name="langchain", timeout_seconds=30)
    with pytest.raises(ValueError, match="explicitly counted transport"):
        BudgetedModelClient(client, max_requests=5, deadline=1e12)


def child_factory(state, clock, *, final_text="USER_FINAL", cost=1, action_attempts=0):
    def factory(task, index, directory, client):
        state.append({"task": task, "client": client, "options": None})
        recorder = TaskRecorder(RunSession(session_id=directory.name, directory=directory))

        class Child:
            observer = SimpleNamespace(observe=lambda: _frame(
                "fresh-final", (final_text() if callable(final_text) else final_text,),
            ))
            verifier = Verifier()

            def run(self, goal, options, **callbacks):
                state[-1]["options"] = options
                response = client.generate_text(f"Plan {goal.instruction}")
                clock.advance(cost)
                return TaskRunResult(run_id=directory.name, case_id=goal.case_id, instruction=goal.instruction,
                                     execute=options.execute, planning_calls=1,
                                     steps=[StepRecord(index=number + 1, step_id=f"a{number}", description="explicit simulated input",
                                                       action_type="wait", action_result={"success": True})
                                            for number in range(action_attempts)],
                                     status=("succeeded" if options.execute else "dry_run_completed") if response.ok else "failed",
                                     verification=VerificationResult(outcome="passed"))

        child = Child()
        child.recorder = recorder
        return child

    return factory


def test_automatic_runner_shares_time_calls_requests_and_does_not_trust_model_final(tmp_path, request_spec):
    clock, state = Clock(), []
    client = ScriptedClient([ready("Open editor", "Complete draft"), "stage plan", "stage plan"], clock=clock, costs=[3, 1, 1])
    result = AutomaticWorkflowRunner(tmp_path / "auto", client, child_factory(state, clock), clock=clock).run(
        request_spec, ExecutionOptions(execute=True, task_timeout_seconds=20, max_planning_attempts=5),
    )
    assert result.status == "succeeded" and result.model_requests == result.planning_calls == 3
    assert 0 < state[0]["options"].task_timeout_seconds <= 17
    assert state[0]["options"].max_planning_attempts == 4
    assert state[1]["options"].max_planning_attempts == 3
    assert result.elapsed_ms == 7000
    assert result.workflow_result.final_observation_id == "fresh-final"
    assert result.workflow_result.final_verification.passed
    assert (tmp_path / "auto/decomposition/request.json").is_file()
    assert (tmp_path / "auto/workflow/workflow_summary.json").is_file()
    assert (tmp_path / "auto/automatic_workflow_summary.json").is_file()


def test_all_generated_subgoals_pass_but_independent_caller_final_check_fails(tmp_path, request_spec):
    clock, state = Clock(), []
    client = ScriptedClient([ready("Open editor"), "stage plan"])
    result = AutomaticWorkflowRunner(tmp_path / "auto", client,
                                     child_factory(state, clock, final_text="Stage 1"), clock=clock,
                                     max_decomposition_rounds=1).run(
        request_spec, ExecutionOptions(execute=True, max_planning_attempts=3),
    )
    assert result.status == "blocked" and result.workflow_result.subgoals[0].status == "completed"
    assert not result.workflow_result.final_verification.passed


def test_decomposition_consumes_last_request_so_no_child_or_observer_is_constructed(tmp_path, request_spec):
    clock, state = Clock(), []
    client = ScriptedClient([ready("Open editor")])
    result = AutomaticWorkflowRunner(tmp_path / "auto", client, child_factory(state, clock),
                                     max_model_requests=1, clock=clock).run(
        request_spec, ExecutionOptions(max_planning_attempts=2),
    )
    assert result.status == "blocked" and result.model_requests == 1
    assert state == [] and not (tmp_path / "auto/workflow").exists()


def test_late_decomposition_never_enters_execution_and_dry_run_never_succeeds(tmp_path, request_spec):
    clock, state = Clock(), []
    late = ScriptedClient([ready("Open editor")], clock=clock, costs=[10])
    result = AutomaticWorkflowRunner(tmp_path / "late", late, child_factory(state, clock), clock=clock).run(
        request_spec, ExecutionOptions(max_planning_attempts=3, task_timeout_seconds=5),
    )
    assert result.status == "timed_out" and state == [] and result.elapsed_ms == 10000
    client = ScriptedClient([ready("Open editor"), "stage plan"])
    result = AutomaticWorkflowRunner(tmp_path / "dry", client, child_factory(state, clock), clock=clock).run(
        request_spec, ExecutionOptions(max_planning_attempts=3, execute=False),
    )
    assert result.status == "dry_run_completed" and result.workflow_result.final_verification is None


def test_existing_root_and_invalid_budget_are_refused_before_any_model_call(tmp_path, request_spec):
    client = ScriptedClient([])
    directory = tmp_path / "old"
    directory.mkdir()
    old = directory / "evidence"
    old.write_bytes(b"untouched old evidence")
    with pytest.raises(FileExistsError):
        AutomaticWorkflowRunner(directory, client, lambda *_: None).run(request_spec, ExecutionOptions(max_planning_attempts=3))
    with pytest.raises(ValueError, match="at least one action planning"):
        AutomaticWorkflowRunner(tmp_path / "invalid", client, lambda *_: None).run(request_spec, ExecutionOptions(max_planning_attempts=1))
    assert client.calls == [] and old.read_bytes() == b"untouched old evidence"
    assert list(directory.iterdir()) == [old] and not (tmp_path / "invalid").exists()


def test_initial_screen_is_available_to_decomposition_with_budget_and_readable_parent_monitor(tmp_path, request_spec):
    clock, state = Clock(), []
    client = ScriptedClient([ready("Open editor"), "stage plan"])
    directory = tmp_path / "grounded"

    def initial_observer(observation_directory):
        status = read_run_status(directory)
        assert status["trusted"] and status["state"] == "active"
        assert status["snapshot"]["stage"] == "observing"
        assert observation_directory == directory / "initial_observation"
        clock.advance(4)
        return _frame("initial-screen", ("Visible editor", "Open draft"))

    result = AutomaticWorkflowRunner(directory, client, child_factory(state, clock), clock=clock,
                                     initial_observer=initial_observer).run(
        request_spec, ExecutionOptions(task_timeout_seconds=20, max_planning_attempts=4),
    )
    assert result.status == "dry_run_completed"
    request = json.loads(client.calls[0]["messages"][-1]["content"])
    texts = [element["text"] for element in request["context"]["initial_observation"]["elements"]]
    assert texts == ["Visible editor", "Open draft"]
    assert state[0]["options"].task_timeout_seconds <= 16
    assert (directory / "initial_observation/observation.json").is_file()
    status = read_run_status(directory)
    assert status["state"] == "finished" and status["trusted"]
    assert status["snapshot"]["planning_pass"] == 2


@pytest.mark.parametrize("failure", ["exception", "perception_error", "late"])
def test_initial_observation_failure_never_loads_model_or_constructs_child_and_monitor_finishes(tmp_path, request_spec, failure):
    clock, state = Clock(), []
    client = ScriptedClient([])
    directory = tmp_path / "failed_observation"

    def initial_observer(_directory):
        if failure == "exception":
            raise RuntimeError("injected capture failure")
        frame = _frame("initial", ("visible",))
        if failure == "perception_error":
            frame.errors.append("OCR unavailable")
        else:
            clock.advance(12)
        return frame

    result = AutomaticWorkflowRunner(directory, client, child_factory(state, clock), clock=clock,
                                     initial_observer=initial_observer).run(
        request_spec, ExecutionOptions(task_timeout_seconds=10, max_planning_attempts=3),
    )
    assert result.status in {"failed", "blocked", "timed_out"}
    assert client.calls == [] and result.model_requests == 0 and state == []
    status = read_run_status(directory)
    assert status["state"] == "finished" and status["trusted"]
    assert (directory / "automatic_workflow_summary.json").is_file()


def test_decomposition_refusal_keeps_finished_blocked_parent_status(tmp_path, request_spec):
    clock = Clock()
    client = ScriptedClient([json.dumps({"status": "refused", "reason": "outside local scope", "subgoals": []})])
    directory = tmp_path / "refused"
    result = AutomaticWorkflowRunner(directory, client, lambda *_: pytest.fail("no child"), clock=clock).run(
        request_spec, ExecutionOptions(max_planning_attempts=3),
    )
    assert result.status == "refused"
    status = read_run_status(directory)
    assert status["state"] == "finished" and status["trusted"]
    assert status["snapshot"]["task_status"] == "blocked"
    assert status["snapshot"]["detail_code"] == "decomposition_refused"


def second_ready(instruction="Complete the remaining draft"):
    return json.dumps({"status": "ready", "subgoals": [
        {"instruction": instruction, "expect_text": ["Stage 2"], "forbid_text": []},
    ]})


def test_fresh_screen_rolling_decomposition_finishes_remaining_work_without_resetting_budgets(tmp_path, request_spec):
    clock, state = Clock(), []
    client = ScriptedClient([ready("Open editor"), "first action plan", second_ready(), "second action plan"],
                            clock=clock, costs=[1, 1, 1, 1])
    factory = child_factory(state, clock, action_attempts=1,
                            final_text=lambda: "Stage 1" if len(state) == 1 else "USER_FINAL")
    directory = tmp_path / "rolling"
    result = AutomaticWorkflowRunner(directory, client, factory, clock=clock).run(
        request_spec, ExecutionOptions(execute=True, max_planning_attempts=6, max_actions=3,
                                       task_timeout_seconds=20),
    )
    assert result.status == "succeeded" and result.model_requests == result.planning_calls == 4
    assert result.total_actions == 2 and len(result.decompositions) == len(result.workflow_rounds) == 2
    assert len(result.completed_subgoals) == 2
    assert result.workflow_rounds[0].status == "failed" and result.workflow_rounds[1].status == "succeeded"
    assert [entry["task"].instruction for entry in state] == ["Open editor", "Complete the remaining draft"]
    assert [entry["options"].max_actions for entry in state] == [3, 2]
    assert [entry["options"].max_planning_attempts for entry in state] == [5, 3]
    assert state[1]["options"].task_timeout_seconds < state[0]["options"].task_timeout_seconds
    second_request = json.loads(client.calls[2]["messages"][-1]["content"])
    assert second_request["instruction"] == request_spec.instruction
    assert second_request["context"]["completed_subgoals"][0]["instruction"] == "Open editor"
    assert second_request["context"]["initial_observation"]["elements"][0]["text"] == "Stage 1"
    assert (directory / "decomposition_round_002/attempt_01_response.json").is_file()
    assert (directory / "workflow_round_002/workflow_summary.json").is_file()
    assert read_run_status(directory)["snapshot"]["planning_pass"] == 4
    assert read_run_status(directory)["snapshot"]["actions_used"] == 2


@pytest.mark.parametrize("duplicate", ["instruction", "checks"])
def test_completed_stage_cannot_be_replayed_by_a_new_decomposition(tmp_path, request_spec, duplicate):
    clock, state = Clock(), []
    repeated = second_ready("Open editor") if duplicate == "instruction" else ready("Paraphrased same stage")
    client = ScriptedClient([ready("Open editor"), "action plan", repeated])
    result = AutomaticWorkflowRunner(tmp_path / "duplicate", client,
                                     child_factory(state, clock, final_text="Stage 1"), clock=clock).run(
        request_spec, ExecutionOptions(execute=True, max_planning_attempts=6),
    )
    assert result.status == "blocked" and "already completed" in result.error
    assert len(state) == 1 and result.model_requests == result.planning_calls == 3
    assert len(result.decompositions) == 2 and len(result.workflow_rounds) == 1


def test_rolling_refuses_new_round_when_shared_planning_budget_cannot_cover_it(tmp_path, request_spec):
    clock, state = Clock(), []
    client = ScriptedClient([ready("Open editor"), "action plan"])
    result = AutomaticWorkflowRunner(tmp_path / "budget", client,
                                     child_factory(state, clock, final_text="Stage 1"), clock=clock).run(
        request_spec, ExecutionOptions(execute=True, max_planning_attempts=3),
    )
    assert result.status == "blocked" and "planning budget" in result.error
    assert result.planning_calls == result.model_requests == 2 and len(state) == 1


def test_rolling_stops_at_declared_round_limit_and_retains_every_failed_independent_final(tmp_path, request_spec):
    clock, state = Clock(), []
    client = ScriptedClient([ready("Open editor"), "first action", second_ready(), "second action"])
    result = AutomaticWorkflowRunner(tmp_path / "limited", client,
                                     child_factory(state, clock, final_text="Still incomplete"), clock=clock,
                                     max_decomposition_rounds=2).run(
        request_spec, ExecutionOptions(execute=True, max_planning_attempts=10),
    )
    assert result.status == "blocked" and "rounds exhausted" in result.error
    assert result.planning_calls == result.model_requests == 4 and len(state) == 2
    assert len(result.workflow_rounds) == 2
    assert all(item.final_verification.outcome == "failed" for item in result.workflow_rounds)


def test_full_http_response_survives_sdk_finish_reason_length_error(tmp_path):
    content = "full retained output " * 150
    payload = {"id": "test-long", "object": "chat.completion", "created": 0, "model": "test",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}]}
    client = OpenAICompatibleClient(api_key="local", model_name="test", max_retries=0, environ={})
    sdk = OpenAI(api_key="local", http_client=httpx.Client(transport=httpx.MockTransport(
        lambda _request: httpx.Response(200, json=payload),
    )))
    client._client = sdk
    budgeted = BudgetedModelClient(client, max_requests=1, deadline=1e12, audit_path=tmp_path / "trace.jsonl")
    response = budgeted.generate_text("long response")
    assert not response.ok and budgeted.request_count == 1
    trace = json.loads((tmp_path / "trace.jsonl").read_text("utf-8"))
    assert trace["http_responses"][0]["status_code"] == 200
    assert trace["http_responses"][0]["body"] == payload
    assert trace["http_responses"][0]["body"]["choices"][0]["message"]["content"] == content
    sdk.close()


@pytest.mark.parametrize("deadline_offset", [0, -1])
def test_real_task_runner_expired_absolute_deadline_performs_zero_observations_and_zero_inputs(tmp_path, deadline_offset):
    clock = Clock()
    observer = FakeObserver([_frame("must-not-observe", ("Ready",))])
    planner, executor = FakePlanner(None), FakeExecutor()
    session = RunSession.create(tmp_path, session_id="expired-real-runtime")
    runner = TaskRunner(observer=observer, planner=planner, executor=executor,
                        adapter=ActionAdapter(platform="win32"), verifier=Verifier(),
                        recorder=TaskRecorder(session), clock=clock, sleep=lambda _: None,
                        absolute_deadline=clock() + deadline_offset)
    result = runner.run(TaskSpec(case_id="LOCAL", instruction="Reach Done", expect_text=["Done"]),
                        ExecutionOptions(task_timeout_seconds=1000, execute=True, confirm=False))
    assert result.status == "timed_out"
    assert observer.calls == 0 and planner.contexts == [] and executor.actions == []


def test_delayed_automatic_factory_cannot_give_real_task_runner_a_new_relative_budget(tmp_path, request_spec):
    clock = Clock()
    observer = FakeObserver([_frame("must-not-observe", ("Ready",))])
    planner, executor = FakePlanner(None), FakeExecutor()
    client = ScriptedClient([ready("Open editor")])
    created = []

    def factory(_task, _index, directory, budgeted_client):
        clock.advance(6)
        runner = TaskRunner(observer=observer, planner=planner, executor=executor,
                            adapter=ActionAdapter(platform="win32"), verifier=Verifier(),
                            recorder=TaskRecorder(RunSession(session_id=directory.name, directory=directory)),
                            clock=clock, absolute_deadline=budgeted_client.deadline)
        created.append(runner)
        return runner

    result = AutomaticWorkflowRunner(tmp_path / "delayed-factory", client, factory, clock=clock).run(
        request_spec, ExecutionOptions(execute=True, max_planning_attempts=5, task_timeout_seconds=5),
    )
    assert result.status == "timed_out" and result.model_requests == result.planning_calls == 1
    assert len(created) == 1 and created[0].absolute_deadline == 105
    assert observer.calls == 0 and planner.contexts == [] and executor.actions == []


def proposal_with_checks(*, expected=(), forbidden=()):
    return json.dumps({"status": "ready", "subgoals": [{
        "instruction": "Advance the current local stage", "expect_text": list(expected),
        "forbid_text": list(forbidden),
    }]})


@pytest.mark.parametrize("forbidden", ["USER_FINAL", "user_final", "final"])
def test_intermediate_forbid_cannot_conflict_with_caller_final_required_text(tmp_path, request_spec, forbidden):
    client = ScriptedClient([proposal_with_checks(expected=["new progress"], forbidden=[forbidden])])
    result = WorkflowDecomposer(client, max_format_retries=0).decompose(
        request_spec, evidence_directory=tmp_path / "conflict", deadline=1e12,
    )
    assert result.status == "failed" and result.workflow is None and result.model_requests == 1
    assert "conflicts with caller final expect_text" in result.error
    assert forbidden in result.error and "USER_FINAL" in result.error
    original = json.loads((tmp_path / "conflict/request.json").read_text("utf-8"))
    assert original["expect_text"] == ["USER_FINAL"] and original["forbid_text"] == ["USER_ERROR"]


def test_bad_current_button_expected_text_is_repaired_before_execution_with_same_current_frame(tmp_path, request_spec):
    bad = proposal_with_checks(expected=["Advance stage"])
    good = proposal_with_checks(expected=["Advance stage"], forbidden=["Pending stage"])
    client = ScriptedClient([bad, good])
    observation = _frame("initial", ("Advance stage", "Pending stage"))
    result = WorkflowDecomposer(client).decompose(
        request_spec, evidence_directory=tmp_path / "repair", deadline=1e12, initial_observation=observation,
    )
    assert result.status == "ready" and result.format_attempts == result.model_requests == 2
    assert result.workflow.subgoals[0].expect_text == ["Advance stage"]
    assert result.workflow.subgoals[0].forbid_text == ["Pending stage"]
    assert result.workflow.final_check.expect_text == request_spec.expect_text
    second_request = json.loads(client.calls[1]["messages"][-1]["content"])
    assert "complete conjunction of checks is already satisfied" in second_request["context"]["constraint_or_format_error"]
    assert second_request["context"]["previous_response"] == bad
    assert second_request["context"]["initial_observation"]["elements"][0]["text"] == "Advance stage"
    assert json.loads((tmp_path / "repair/attempt_01_response.json").read_text("utf-8"))["content"] == bad
    assert json.loads((tmp_path / "repair/attempt_02_response.json").read_text("utf-8"))["content"] == good


def test_default_two_constraint_repairs_are_counted_and_all_failed_proposals_remain_nonexecutable(tmp_path, request_spec):
    bad = proposal_with_checks(expected=["Advance stage"])
    client = ScriptedClient([bad, bad, bad, ready("never requested")])
    result = WorkflowDecomposer(client).decompose(
        request_spec, evidence_directory=tmp_path / "limit", deadline=1e12,
        initial_observation=_frame("initial", ("Advance stage", "Pending stage")),
    )
    assert result.status == "failed" and result.workflow is None
    assert len(client.calls) == result.model_requests == result.format_attempts == 3
    assert len(client.responses) == 1
    assert len(list((tmp_path / "limit").glob("attempt_*_response.json"))) == 3
    assert len(list((tmp_path / "limit").glob("attempt_*_error.json"))) == 3


def test_fixed_final_context_is_allowed_with_unsatisfied_progress_but_cannot_prove_progress_alone(tmp_path, request_spec):
    observation = _frame("initial", ("USER_FINAL", "Pending stage"))
    context_only = proposal_with_checks(expected=["USER_FINAL"])
    progress = proposal_with_checks(expected=["USER_FINAL"], forbidden=["Pending stage"])
    client = ScriptedClient([context_only, progress])
    result = WorkflowDecomposer(client).decompose(
        request_spec, evidence_directory=tmp_path / "context", deadline=1e12, initial_observation=observation,
    )
    assert result.status == "ready" and result.model_requests == 2
    first_error = json.loads((tmp_path / "context/attempt_01_error.json").read_text("utf-8"))["error"]
    assert "already satisfied" in first_error
    assert result.workflow.subgoals[0].expect_text == ["USER_FINAL"]
    assert result.workflow.subgoals[0].forbid_text == ["Pending stage"]


def test_constraint_failures_never_construct_child_or_rewrite_caller_final_rules(tmp_path, request_spec):
    bad = proposal_with_checks(expected=["Advance stage"], forbidden=["USER_FINAL"])
    client = ScriptedClient([bad, bad, bad])
    directory = tmp_path / "no-execution"
    result = AutomaticWorkflowRunner(
        directory, client, lambda *_: pytest.fail("invalid proposal must never enter a child"),
        initial_observer=lambda _: _frame("initial", ("Advance stage", "Pending stage")),
    ).run(request_spec, ExecutionOptions(execute=True, max_planning_attempts=6))
    assert result.status == "failed" and result.model_requests == 3
    assert result.planning_calls == 1 and result.workflow_result is None
    assert not (directory / "workflow").exists()
    assert read_run_status(directory)["snapshot"]["task_status"] == "failed"
    fixed = json.loads((directory / "decomposition/request.json").read_text("utf-8"))
    assert fixed["expect_text"] == ["USER_FINAL"] and fixed["forbid_text"] == ["USER_ERROR"]


def test_existing_expected_and_visible_forbidden_marker_form_a_pending_then_verified_conjunction(tmp_path, request_spec):
    before = _frame("before", ("Selected option", "Alternative available"))
    after = _frame("after", ("Selected option",))
    client = ScriptedClient([proposal_with_checks(expected=["Selected option"], forbidden=["Alternative available"])])
    result = WorkflowDecomposer(client).decompose(
        request_spec, evidence_directory=tmp_path / "conjunction", deadline=1e12, initial_observation=before,
    )
    assert result.status == "ready" and result.model_requests == 1
    task = result.workflow.subgoals[0]
    verifier = Verifier()
    before_result = verifier.check_task(task, before)
    after_result = verifier.check_task(task, after)
    assert before_result.outcome == "failed" and before_result.evidence["unexpected"] == ["Alternative available"]
    assert after_result.passed and after_result.evidence["observation_id"] == "after"
    assert not verifier.check_task(result.workflow.final_check, after).passed
    assert result.workflow.final_check.expect_text == request_spec.expect_text


def test_existing_expected_with_additional_absent_expected_is_pending_until_both_are_observed(tmp_path, request_spec):
    before = _frame("before", ("Retained context",))
    after = _frame("after", ("Retained context", "New progress"))
    client = ScriptedClient([proposal_with_checks(expected=["Retained context", "New progress"])])
    result = WorkflowDecomposer(client).decompose(
        request_spec, evidence_directory=tmp_path / "new-expected", deadline=1e12, initial_observation=before,
    )
    assert result.status == "ready" and result.model_requests == 1
    verifier = Verifier()
    task = result.workflow.subgoals[0]
    assert verifier.check_task(task, before).evidence["missing"] == ["New progress"]
    assert verifier.check_task(task, after).passed
    assert not verifier.check_task(result.workflow.final_check, after).passed


def test_pending_conjunction_cannot_bypass_fresh_workflow_final_observation(tmp_path, request_spec):
    clock, state = Clock(), []
    before = _frame("initial-observation", ("Retained context", "Alternative available"))
    proposal = proposal_with_checks(expected=["Retained context"], forbidden=["Alternative available"])
    client = ScriptedClient([proposal, "child action plan"])
    result = AutomaticWorkflowRunner(
        tmp_path / "pending-conjunction", client,
        child_factory(state, clock, final_text="Retained context"), clock=clock,
        initial_observer=lambda _: before, max_decomposition_rounds=1,
    ).run(request_spec, ExecutionOptions(execute=True, max_planning_attempts=4))
    assert result.decomposition.status == "ready" and len(state) == 1
    assert result.workflow_result.subgoals[0].status == "completed"
    assert result.workflow_result.final_observation_id == "fresh-final"
    assert result.workflow_result.final_observation_id != before.observation_id
    assert result.workflow_result.final_verification.outcome == "failed"
    assert result.status == "blocked"
