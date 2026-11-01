"""Independent CPU counterexamples for the new Week6 acceptance contract.

All observations, model replies and executors below are offline doubles. These
tests verify control flow and evidence boundaries; they do not establish real
window success or the semantic adequacy of a live model's decomposition.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from test_week6_automatic_decomposition import _frame
from test_week6_rolling_actions import options, plan
from test_week6_semantic_subtasks import lab, proposal

from gui_agent.planning import semantic_workflow as semantic
from gui_agent.runtime.monitoring import read_run_status
from gui_agent.runtime.schemas import VerificationResult


def _audit_files(directory: Path):
    return sorted(directory.glob("goal_*/001_*/semantic_goal_audit.json"))


def test_review_bare_drag_is_a_primitive_name_not_a_semantic_objective():
    with pytest.raises(ValueError, match="semantic|primitive"):
        semantic.SemanticGoal(goal_id="BarePrimitive", instruction="drag", expect_text=["Changed"])


@pytest.mark.parametrize("identity", ["id", "instruction", "checks"])
def test_review_completed_objective_cannot_be_reintroduced_under_partial_renaming(identity):
    original = json.loads(proposal())["subgoals"][0]
    changed = {**original, "goal_id": "NewIdentity", "instruction": "Complete a renamed semantic objective",
               "expect_text": ["New postcondition"]}
    if identity == "id":
        changed["goal_id"] = original["goal_id"].lower()
    elif identity == "instruction":
        changed["instruction"] = original["instruction"].upper()
    else:
        changed["expect_text"] = [text.upper() for text in original["expect_text"]]
    parsed = semantic.SemanticPlan.model_validate(json.loads(proposal(changed)))
    request = semantic.WorkflowRequest(instruction="Set preference and persist it", expect_text=["Final accepted"])
    with pytest.raises(ValueError, match="completed"):
        semantic.SemanticDecomposer()._validate(parsed, request, _frame("before", ["Pending"]),
                                               [original], revision=True)


def test_review_failed_semantic_goal_is_not_rescued_by_visible_caller_final(tmp_path):
    native = json.loads(proposal())
    native["subgoals"][1]["expect_text"] = ["Never observed objective result"]
    runner, request, state, client, _ = lab(tmp_path,
        responses=[json.dumps(native), plan(), plan()], rounds=3, goal_calls=1)
    result = runner.run(request, options())
    assert state.stage == 2  # The fake screen now contains the caller final.
    assert result.status != "succeeded" and result.final_verification is None
    assert [entry["status"] for entry in result.goal_outcomes] == ["succeeded", "failed"]
    assert [entry["goal_id"] for entry in result.completed_subgoals] == ["SetPreference"]
    assert result.total_actions == state.attempts == 2
    assert result.model_requests == result.planning_calls == len(client.calls) == 3
    assert len(result.plans) == 1  # No post-hoc rewriting or retry of the failed plan.


def test_review_raw_action_prediction_failure_does_not_rewrite_the_semantic_goal(tmp_path):
    native_action = json.loads(plan())
    native_action["steps"][0]["expected_result"] = "Model predicted text that never exists"
    runner, request, state, client, _ = lab(tmp_path,
        responses=[proposal(), json.dumps(native_action), json.dumps(native_action)])
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 2
    assert result.planning_calls == len(client.calls) == 3
    audits = [json.loads(path.read_text("utf8")) for path in _audit_files(tmp_path / "semantic")]
    assert len(audits) == 2
    for audit in audits:
        assert audit["eligible_for_semantic_credit"]
        assert audit["semantic_postcondition_verification"]["outcome"] == "passed"
        assert audit["steps"][0]["verification"]["outcome"] == "failed"
        assert audit["frozen_task"]["expect_text"] != [native_action["steps"][0]["expected_result"]]


def test_review_delegate_verifier_cannot_supply_false_semantic_success(tmp_path):
    native = json.loads(proposal())
    native["subgoals"][0]["expect_text"] = ["Never observed objective result"]
    runner, request, state, _, _ = lab(tmp_path,
        responses=[json.dumps(native), plan()], goal_calls=1)
    factory = runner.runner_factory

    class LyingVerifier(semantic.Verifier):
        def check_task(self, _task, observation):
            return VerificationResult(outcome="passed", evidence={"observation_id": observation.observation_id})

    def wrapped(task, index, directory, client):
        child = factory(task, index, directory, client)
        child.verifier = LyingVerifier()
        return child

    runner.runner_factory = wrapped
    result = runner.run(request, options())
    assert result.status != "succeeded" and not result.completed_subgoals
    assert result.total_actions == state.attempts == 1


def test_review_factory_cannot_replace_a_frozen_goal_with_preexisting_context(tmp_path):
    runner, request, state, client, _ = lab(tmp_path)
    factory = runner.runner_factory

    def wrapped(task, index, directory, budgeted):
        task.expect_text[:] = ["Advance control"]
        return factory(task, index, directory, budgeted)

    runner.runner_factory = wrapped
    result = runner.run(request, options())
    assert result.status != "succeeded" and not result.completed_subgoals
    assert state.attempts == result.total_actions == 0
    assert len(client.calls) == result.model_requests == result.planning_calls == 1


def test_review_caller_owned_request_alias_is_not_a_way_to_change_final_mid_run(tmp_path):
    runner, request, state, _, _ = lab(tmp_path)
    original = request.model_copy(deep=True)

    def callback(_event):
        request.expect_text[:] = ["Never accepted by frozen caller request"]
        request.forbid_text.clear()

    result = runner.run(request, options(), on_decomposition=callback)
    assert result.status == "succeeded" and state.attempts == 2
    stored = json.loads((tmp_path / "semantic/semantic_request.json").read_text("utf8"))
    assert stored["caller_request"] == original.model_dump(mode="json")
    final = json.loads((tmp_path / "semantic/semantic_plan_round_001/compiled_workflow.json").read_text("utf8"))
    assert final["workflow"]["final_check"]["expect_text"] == original.expect_text
    assert final["workflow"]["final_check"]["forbid_text"] == original.forbid_text


def test_review_decomposition_result_write_failure_preserves_consumed_counts(tmp_path, monkeypatch):
    runner, request, state, client, _ = lab(tmp_path)
    write = semantic._write_new_json

    def fail_final_plan_record(path, value):
        if path.name == "result.json" and path.parent.name.startswith("semantic_plan_round_"):
            raise PermissionError("independent injected result persistence fault")
        return write(path, value)

    monkeypatch.setattr(semantic, "_write_new_json", fail_final_plan_record)
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == 0
    assert result.model_requests == result.planning_calls == len(client.calls) == 1
    assert "persist" in result.error.lower() or "audit" in result.error.lower()


@pytest.mark.parametrize("kind", ["goal_audit", "goal_ledger", "root_summary"])
def test_review_post_input_persistence_failure_keeps_known_receipts_without_success(tmp_path, monkeypatch, kind):
    runner, request, state, client, _ = lab(tmp_path)
    if kind == "goal_audit":
        original = semantic._write_new_json

        def write(path, value):
            if path.name == "semantic_goal_audit.json":
                raise PermissionError("independent injected semantic goal audit fault")
            return original(path, value)

        monkeypatch.setattr(semantic, "_write_new_json", write)
    else:
        original = semantic._write_json
        target = "semantic_goal_ledger.json" if kind == "goal_ledger" else "automatic_workflow_summary.json"

        def write(path, value):
            if path.name == target:
                raise PermissionError("independent injected root record fault")
            return original(path, value)

        monkeypatch.setattr(semantic, "_write_json", write)
    result = runner.run(request, options())
    assert result.status != "succeeded"
    assert result.total_actions == state.attempts > 0
    assert result.model_requests == result.planning_calls == len(client.calls)
    assert all(child.planner.max_steps == 7 and "plan" not in child.planner.__dict__ for child in state.children)


def test_review_callback_keyboard_interrupt_does_not_erase_confirmed_inputs_or_replay(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, rounds=3)

    def interrupt(event):
        if event["actions"]:
            raise KeyboardInterrupt

    result = runner.run(request, options(), on_checkpoint=interrupt)
    assert result.status == "cancelled"
    assert result.total_actions == state.attempts == 1
    assert result.model_requests == result.planning_calls == len(client.calls) == 2
    assert len(result.plans) == 1 and not result.completed_subgoals
    assert state.children[0].planner.max_steps == 7 and "plan" not in state.children[0].planner.__dict__


def test_review_duplicate_json_keys_are_not_normalized_into_an_accepted_plan(tmp_path):
    native = proposal().replace('"status": "ready"', '"status": "refused", "status": "ready"')
    runner, request, state, client, _ = lab(tmp_path, responses=[native])
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == 0 and not result.completed_subgoals
    assert result.planning_calls == len(client.calls) == 1
    saved = json.loads((tmp_path / "semantic/semantic_plan_round_001/attempt_01_response.json").read_text("utf8"))
    assert saved["content"] == native
    assert "duplicate" in result.plans[0].error.lower()


def test_review_unsupported_schema_transport_does_not_spend_http_or_input(tmp_path):
    runner, request, state, client, _ = lab(tmp_path)
    client.supports_strict_json_schema = False
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == 0
    assert result.model_requests == result.planning_calls == len(client.calls) == 0


def test_review_wait_receipt_alone_is_not_semantic_input_credit(tmp_path):
    action = json.loads(plan())
    action["steps"][0].update(action_type="wait", target_text="", arguments={"seconds": 0.01})
    runner, request, state, _, _ = lab(tmp_path,
        responses=[proposal(), json.dumps(action)], goal_calls=1)
    result = runner.run(request, options())
    assert state.stage == 1  # Even an independently changed screen does not make wait an OS input.
    assert result.status != "succeeded" and not result.completed_subgoals
    audit = json.loads(_audit_files(tmp_path / "semantic")[0].read_text("utf8"))
    assert audit["confirmed_input_count"] == 0 and not audit["eligible_for_semantic_credit"]


def test_review_two_primitive_steps_cannot_escape_the_one_step_planning_contract(tmp_path):
    runner, request, state, client, _ = lab(tmp_path, responses=[proposal(), plan(2)])
    result = runner.run(request, options())
    assert result.status != "succeeded" and state.attempts == 0
    assert result.planning_calls == result.model_requests == len(client.calls) == 2
    assert state.children[0].planner.max_steps == 7 and "plan" not in state.children[0].planner.__dict__


@pytest.mark.parametrize("reuse", ["time", "path", "id"])
def test_review_child_reused_capture_provenance_stops_before_credit(tmp_path, reuse):
    runner, request, state, client, _ = lab(tmp_path, rounds=3)
    factory = runner.runner_factory

    def wrapped(task, index, directory, budgeted):
        child = factory(task, index, directory, budgeted)
        capture, prior = child.observer.observe, []

        def observe(**kwargs):
            frame = capture(**kwargs)
            if prior and state.attempts:
                if reuse == "time":
                    frame.captured_at = prior[0].captured_at
                elif reuse == "path":
                    frame.image_path = prior[0].image_path
                else:
                    frame.observation_id = prior[0].observation_id
            prior.append(frame.model_copy(deep=True))
            return frame

        child.observer.observe = observe
        return child

    runner.runner_factory = wrapped
    result = runner.run(request, options())
    assert result.status != "succeeded" and not result.completed_subgoals
    assert result.total_actions == state.attempts == 1
    assert len(result.plans) == 1 and len(client.calls) == result.model_requests == 2


def test_review_root_monitor_agrees_with_verified_goal_ledger_and_shared_counts(tmp_path):
    runner, request, state, _, _ = lab(tmp_path)
    result = runner.run(request, options())
    assert result.status == "succeeded"
    directory = tmp_path / "semantic"
    status = read_run_status(directory)
    final = status["snapshot"]
    assert final["subgoals_total"] == final["subgoals_completed"] == len(result.completed_subgoals) == 2
    assert final["actions_used"] == result.total_actions == state.attempts
    assert final["planning_pass"] == result.planning_calls
    events = [json.loads(line) for line in (directory / "status_events.jsonl").read_text("utf8").splitlines()]
    active_ids = {entry["subgoal_id"] for entry in events if entry["subgoal_id"]}
    assert active_ids == {"SetPreference", "PersistConfig"}
    assert all(entry["subgoals_total"] >= entry["subgoals_completed"] for entry in events)
    assert any(entry["subgoals_completed"] == 1 for entry in events)
    assert final["event"] == "finished" and final["task_status"] == result.status


def test_review_monitor_revision_keeps_confirmed_goal_count_monotonic(tmp_path):
    original = json.loads(proposal())
    original["subgoals"][1]["expect_text"] = ["Unobserved former objective"]
    revised = proposal({"goal_id": "Remaining", "instruction": "Persist the remaining approved configuration",
                        "expect_text": ["Verified part 3"], "forbid_text": []})
    runner, request, state, _, _ = lab(tmp_path,
        responses=[json.dumps(original), plan(), plan(), revised, plan()], rounds=2, goal_calls=1)
    request.expect_text = ["Verified part 3"]
    result = runner.run(request, options())
    assert result.status == "succeeded" and state.attempts == 3
    events = [json.loads(line) for line in (tmp_path / "semantic/status_events.jsonl").read_text("utf8").splitlines()]
    completed = [event["subgoals_completed"] for event in events]
    assert completed == sorted(completed) and completed[-1] == len(result.completed_subgoals) == 2
    assert all(event["subgoals_total"] >= event["subgoals_completed"] for event in events)
    assert events[-1]["actions_used"] == result.total_actions
    assert events[-1]["planning_pass"] == result.planning_calls


@pytest.fixture
def audit_reader():
    source = Path(__file__).resolve().parents[1] / "scripts/week6_audit_semantic_acceptance.py"
    spec = importlib.util.spec_from_file_location("independent_semantic_live_auditor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_reader_rejects_cross_run_evidence_even_if_the_other_file_exists(tmp_path, audit_reader):
    container = tmp_path / "selected"
    (container / "run").mkdir(parents=True)
    other = tmp_path / "other_run.json"
    other.write_text('{"passed":true}', encoding="utf8")
    evidence = audit_reader.Evidence(container)
    with pytest.raises(audit_reader.AuditError, match="escaped"):
        evidence.load(other)
    assert evidence.files == {}


def test_audit_reader_does_not_repair_duplicate_json_keys(tmp_path, audit_reader):
    container = tmp_path / "selected"
    (container / "run").mkdir(parents=True)
    target = container / "run/record.json"
    target.write_text('{"status":"failed","status":"succeeded"}', encoding="utf8")
    with pytest.raises(audit_reader.AuditError, match="duplicate"):
        audit_reader.Evidence(container).load(target)


def test_audit_reader_detects_source_evidence_changes_during_one_read_session(tmp_path, audit_reader):
    container = tmp_path / "selected"
    (container / "run").mkdir(parents=True)
    target = container / "run/record.json"
    target.write_text('{"counter":1}', encoding="utf8")
    evidence = audit_reader.Evidence(container)
    assert evidence.load(target) == {"counter": 1}
    target.write_text('{"counter":2}', encoding="utf8")
    with pytest.raises(audit_reader.AuditError, match="changed_while"):
        evidence.load(target)


def test_audit_reader_requires_real_owner_receipt_before_reading_a_fixture(tmp_path, audit_reader):
    container = tmp_path / "selected"
    (container / "run").mkdir(parents=True)
    (container / "run/acceptance_observed.json").write_text('{"runtime_pass":true}', encoding="utf8")
    evidence = audit_reader.Evidence(container)
    with pytest.raises(audit_reader.AuditError, match="missing_evidence:owner_claim"):
        audit_reader.audit(container, tmp_path / "not_read_owner.py", tmp_path / "not_read_fixture", evidence)
    assert not any("fixture" in name for name in evidence.files)


@pytest.mark.parametrize("code", [None, False, 1, 15])
def test_audit_reader_rejects_unknown_boolean_or_nonzero_child_exit(tmp_path, audit_reader, code):
    container = tmp_path / "selected"
    (container / "run").mkdir(parents=True)
    for name, value in {
        "owner_claim.json": {"schema": "week6.semantic.live.claim.v1"},
        "owner_start.json": {},
        "owner_receipt.json": {"schema": "week6.semantic.live.actual_exit.v1", "child_exit_code": code},
    }.items():
        (container / name).write_text(json.dumps(value), encoding="utf8")
    (container / "stdout.txt").write_text("retained ended output", encoding="utf8")
    with pytest.raises(audit_reader.AuditError, match="real_exit_zero"):
        audit_reader.owner_audit(audit_reader.Evidence(container), tmp_path / "unread_owner.py", tmp_path / "unread_fixture")


@pytest.mark.parametrize("large", [False, True])
def test_audit_reader_v2_repair_keeps_full_archive_and_exact_context_binding(tmp_path, audit_reader, large):
    container = tmp_path / "selected"
    directory = container / "run/automatic/semantic_plan_round_001"
    directory.mkdir(parents=True)
    content = "original reply " * (200 if large else 2)
    response_path = directory / "attempt_01_response.json"
    response_path.write_text(json.dumps({"content": content}), encoding="utf8")
    (directory / "attempt_01_error.json").write_text('{"error":"constraint rejected"}', encoding="utf8")
    context = {"previous_response_record": str(response_path.resolve()),
               "previous_response_characters": len(content), "constraint_error": "constraint rejected"}
    if not large:
        context["previous_response"] = content
    evidence = audit_reader.Evidence(container)
    audit_reader.audit_repair_context(evidence, directory, 2, context)
    assert len(evidence.files) == 2
    assert json.loads(response_path.read_text("utf8"))["content"] == content


@pytest.mark.parametrize("tamper", ["path", "count", "large_inline", "error", "small_missing"])
def test_audit_reader_v2_repair_rejects_unbound_or_silently_truncated_context(tmp_path, audit_reader, tamper):
    container = tmp_path / "selected"
    directory = container / "run/automatic/semantic_plan_round_001"
    directory.mkdir(parents=True)
    content = "original reply " * (200 if tamper == "large_inline" else 2)
    response_path = directory / "attempt_01_response.json"
    response_path.write_text(json.dumps({"content": content}), encoding="utf8")
    (directory / "attempt_01_error.json").write_text('{"error":"constraint rejected"}', encoding="utf8")
    context = {"previous_response_record": str(response_path.resolve()),
               "previous_response_characters": len(content), "constraint_error": "constraint rejected",
               "previous_response": content}
    if tamper == "path":
        context["previous_response_record"] = str(tmp_path / "another-run.json")
    elif tamper == "count":
        context["previous_response_characters"] -= 1
    elif tamper == "error":
        context["constraint_error"] = "a different rejection"
    elif tamper == "small_missing":
        context.pop("previous_response")
    with pytest.raises(audit_reader.AuditError):
        audit_reader.audit_repair_context(audit_reader.Evidence(container), directory, 2, context)


def test_audit_reader_v2_mapping_separates_raw_whitespace_from_validated_instruction(audit_reader):
    native = json.loads(proposal())
    native["subgoals"][0]["instruction"] = "  Set the requested preference  "
    parsed = semantic.SemanticPlan.model_validate(native)
    request = semantic.WorkflowRequest(instruction="Set preference and persist it", expect_text=["Final accepted"])
    tasks = [semantic._goal_task(goal, request) for goal in parsed.subgoals]
    mapping = audit_reader.expected_native_mapping(native, parsed, tasks)
    assert mapping[0]["native_instruction"] == "  Set the requested preference  "
    assert mapping[0]["validated_instruction"] == "Set the requested preference"
    assert mapping[0]["compiled_instruction"] == tasks[0].instruction


def _text_binding_arguments():
    frame = _frame("original", ["Original complete element evidence"])
    frame.image_path = "/original/frame.png"
    saved = {"input_modality": "screen_text_context", "image_attached": False, "image_path": None,
             "snapshot_image_path_provenance": frame.image_path,
             "context": {"input_modality": "screen_text_context", "image_attached": False,
                         "current_observation": frame.model_dump(mode="json")}}
    protocol = {"input_modality": "screen_text_context", "image_attached": False,
                "snapshot_image_path_provenance": frame.image_path}
    record = {"input_modality": "screen_text_context", "image_attached": False}
    return saved, protocol, record, frame


def test_audit_reader_v3_text_input_preserves_original_snapshot_without_attaching_pixels(audit_reader):
    saved, protocol, record, frame = _text_binding_arguments()
    audit_reader.audit_highlevel_text_binding(saved, protocol, record, frame)
    call = {"kind": "semantic_decomposition", "image_path": None}
    messages = [{"role": "user", "content": json.dumps({"instruction": "Original caller",
        "context": saved["context"]})}]
    audit_reader.audit_message_modality(call, messages)
    assert saved["context"]["current_observation"]["image_path"] == frame.image_path


@pytest.mark.parametrize("tamper", ["pixels", "flag", "modality", "provenance", "snapshot"])
def test_audit_reader_v3_rejects_faked_text_flags_or_dropped_source_evidence(audit_reader, tamper):
    saved, protocol, record, frame = _text_binding_arguments()
    if tamper == "pixels":
        saved["image_path"] = frame.image_path
    elif tamper == "flag":
        saved["context"]["image_attached"] = True
    elif tamper == "modality":
        record["input_modality"] = "other"
    elif tamper == "provenance":
        saved["snapshot_image_path_provenance"] = "/another/frame.png"
    else:
        saved["context"]["current_observation"]["elements"] = []
    with pytest.raises(audit_reader.AuditError):
        audit_reader.audit_highlevel_text_binding(saved, protocol, record, frame)


@pytest.mark.parametrize("path", [None, "", "/another/frame.png"])
def test_audit_reader_v3_lowlevel_cannot_drop_or_borrow_the_action_image(audit_reader, path):
    frame = _frame("original", ["Actual current action target"])
    frame.image_path = "/original/frame.png"
    with pytest.raises(audit_reader.AuditError, match="actual_saved_image"):
        audit_reader.audit_lowlevel_image_binding({"image_path": path}, frame)


@pytest.mark.parametrize("kind,image_path,payload", [
    ("semantic_decomposition", None, {"image_path": "/unexpected.png"}),
    ("primitive_action_plan", "/actual.png", {"instruction": "Image missing"}),
])
def test_audit_reader_v3_preencoding_message_cannot_switch_modalities(audit_reader, kind, image_path, payload):
    with pytest.raises(audit_reader.AuditError):
        audit_reader.audit_message_modality({"kind": kind, "image_path": image_path},
            [{"role": "user", "content": json.dumps(payload)}])
