"""Pre-execution progress witnesses keep raw checks, scope, receipts and final rules separate."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_week6_automatic_decomposition import _frame

from gui_agent.planning import workflow_decomposer as module
from gui_agent.planning.workflow_decomposer import (
    WorkflowRequest,
    _derive_progress_contract,
    _ObservedFrames,
    _ProgressChild,
    _ProgressVerifier,
)
from gui_agent.recording import RunSession
from gui_agent.runtime.recorder import TaskRecorder
from gui_agent.runtime.schemas import ExecutionOptions, StepRecord, TaskRunResult, TaskSpec
from gui_agent.runtime.verification import Verifier
from gui_agent.runtime.workflow import WorkflowSpec
from gui_agent.schemas import ActionResult, DesktopAction


def inputs():
    request = WorkflowRequest(instruction="Advance a local stage and finish", expect_text=["Caller finished"])
    raw = WorkflowSpec(workflow_id="Local", instruction=request.instruction,
                       subgoals=[TaskSpec(case_id="G01", instruction="Advance the local stage",
                                          expect_text=["Advance control"], forbid_text=["Pending stage"])],
                       final_check=TaskSpec(case_id="FINAL", instruction="Read caller rules", expect_text=request.expect_text))
    before = _frame("contract-before", ("Advance control", "Pending stage"))
    return request, raw, before


def contract(tmp_path):
    request, raw, before = inputs()
    derived, record = _derive_progress_contract(raw, request, before, tmp_path / "contract.json")
    return request, raw, before, derived, record


def test_preexecution_record_is_exclusive_and_keeps_raw_derived_context_and_final_distinct(tmp_path):
    request, raw, before, derived, record = contract(tmp_path)
    assert raw.subgoals[0].expect_text == ["Advance control"]
    assert derived.subgoals[0].expect_text == []
    assert derived.subgoals[0].forbid_text == ["Pending stage"]
    assert derived.final_check.model_dump() == raw.final_check.model_dump()
    assert derived.final_check.expect_text == request.expect_text
    assert record.context_expect_text == ["Advance control"]
    assert record.before_observation.observation_id == before.observation_id
    original_bytes = (tmp_path / "contract.json").read_bytes()
    data = json.loads(original_bytes)
    assert data["schema_version"] == "week6.preexecution_progress.v1"
    assert data["raw_workflow"]["subgoals"][0]["expect_text"] == ["Advance control"]
    assert data["derived_workflow"]["subgoals"][0]["expect_text"] == []
    with pytest.raises(FileExistsError):
        _derive_progress_contract(raw, request, before, tmp_path / "contract.json")
    assert (tmp_path / "contract.json").read_bytes() == original_bytes


def test_caller_final_expected_and_future_expected_are_never_dropped(tmp_path):
    request, raw, before = inputs()
    raw.subgoals[0].expect_text = ["Advance control", "Caller finished", "Future progress"]
    before.elements.extend(_frame("extra", ("Caller finished",)).elements)
    derived, record = _derive_progress_contract(raw, request, before, tmp_path / "contract.json")
    assert record.context_expect_text == ["Advance control"]
    assert derived.subgoals[0].expect_text == ["Caller finished", "Future progress"]
    post = _frame("after", ("Caller finished",))
    assert Verifier().check_task(derived.subgoals[0], post).outcome == "failed"


@pytest.mark.parametrize("fault", ["blank", "error", "missing_window", "unstable", "missing_geometry", "satisfied", "final_conflict"])
def test_untrusted_or_nonprogressing_before_is_refused_without_a_contract(tmp_path, fault):
    request, raw, before = inputs()
    if fault == "blank":
        before.elements.clear()
        before.window_title = "Readable title only"
    elif fault == "error":
        before.errors.append("injected")
    elif fault == "missing_window":
        before.window_id = ""
    elif fault == "unstable":
        before.foreground_stable = False
    elif fault == "missing_geometry":
        before.window_bounds = None
    elif fault == "satisfied":
        raw.subgoals[0].forbid_text = []
    else:
        raw.subgoals[0].forbid_text.append("Caller finished")
    with pytest.raises(ValueError):
        _derive_progress_contract(raw, request, before, tmp_path / "contract.json")
    assert not (tmp_path / "contract.json").exists()


def child_with_post(tmp_path, post, *, status=None, confirmed=True, unknown=0, action_types=("click",)):
    recorder = TaskRecorder(RunSession(session_id="child", directory=tmp_path / "child"))

    class Child:
        observer = SimpleNamespace(observe=lambda: post)
        verifier = Verifier()

        def run(self, task, options):
            frame = self.observer.observe()
            verification = self.verifier.check_task(task, frame)
            return TaskRunResult(run_id="child", case_id=task.case_id, instruction=task.instruction,
                                 execute=options.execute, status=status or ("succeeded" if verification.passed else "failed"),
                                 planning_calls=1, verification=verification, unconfirmed_action_attempts=unknown,
                                 steps=[StepRecord(index=index + 1, step_id=f"known-{index}", description="confirmed simulated local input",
                                                   action_type=kind, action_result=ActionResult(
                                                       success=True, dry_run=False,
                                                       action=DesktopAction(action_type=kind, x=20, y=20),
                                                   ).model_dump(mode="json")) for index, kind in enumerate(action_types)]
                                 if confirmed else [])

    child = Child()
    child.recorder = recorder
    return child


def test_raw_false_progress_true_is_recorded_without_claiming_raw_or_caller_success(tmp_path):
    _, raw, _, derived, record = contract(tmp_path)
    post = _frame("post", ("Next local stage",))
    audits = []
    child = _ProgressChild(child_with_post(tmp_path, post), raw.subgoals[0], derived.subgoals[0], record, audits, 1)
    result = child.run(derived.subgoals[0], ExecutionOptions(execute=True))
    assert result.status == "succeeded" and result.action_count == 1
    assert audits[0]["raw_proposal_verification"]["outcome"] == "failed"
    assert audits[0]["progress_contract_verification"]["outcome"] == "passed"
    assert audits[0]["eligible_for_progress_credit"]
    assert Verifier().check_task(raw.subgoals[0], post).outcome == "failed"  # explicit/manual rule remains strict
    assert Verifier().check_task(derived.final_check, post).outcome == "failed"
    saved = json.loads((tmp_path / "child/progress_verifications.json").read_text("utf-8"))
    assert saved["raw_proposal_verification"]["outcome"] != saved["progress_contract_verification"]["outcome"]


@pytest.mark.parametrize("fault", ["title_only", "error", "same_id", "older", "other_window", "unstable", "geometry"])
def test_progress_needs_fresh_readable_matching_window_post_evidence(tmp_path, fault):
    _, raw, before, derived, record = contract(tmp_path)
    post = _frame("post", ("Next local stage",))
    if fault == "title_only":
        post.elements.clear()
        post.window_title = "The expected application"
    elif fault == "error":
        post.errors.append("injected OCR failure")
    elif fault == "same_id":
        post.observation_id = before.observation_id
    elif fault == "older":
        from datetime import timedelta

        post.captured_at = before.captured_at - timedelta(seconds=1)
    elif fault == "other_window":
        post.window_id = "another-window"
    elif fault == "unstable":
        post.foreground_stable = False
    else:
        post.window_bounds = None
    verifier = _ProgressVerifier(Verifier(), record, derived.subgoals[0].case_id)
    assert verifier.check_task(derived.subgoals[0], post).outcome == "inconclusive"
    audits = []
    result = _ProgressChild(child_with_post(tmp_path, post), raw.subgoals[0], derived.subgoals[0], record, audits, 1).run(
        derived.subgoals[0], ExecutionOptions(execute=True))
    assert result.status != "succeeded" and not audits[0]["eligible_for_progress_credit"]


@pytest.mark.parametrize("status,confirmed,unknown", [("succeeded", False, 0), ("succeeded", True, 1),
                                                       ("cancelled", True, 0), ("timed_out", True, 0), ("failed", True, 0)])
def test_no_new_credit_for_unknown_unconfirmed_cancelled_timedout_or_failed_input(tmp_path, status, confirmed, unknown):
    _, raw, _, derived, record = contract(tmp_path)
    audits = []
    result = _ProgressChild(child_with_post(tmp_path, _frame("post", ("Next stage",)),
                                          status=status, confirmed=confirmed, unknown=unknown),
                            raw.subgoals[0], derived.subgoals[0], record, audits, 1).run(
        derived.subgoals[0], ExecutionOptions(execute=True))
    assert result.status != "succeeded" and not audits[0]["eligible_for_progress_credit"]
    assert result.unconfirmed_action_attempts == unknown


def test_post_audit_write_failure_preserves_known_actions_and_planning_counts(tmp_path, monkeypatch):
    _, raw, _, derived, record = contract(tmp_path)
    original = module._write_new_json

    def fail(path, value):
        if path.name == "progress_verifications.json":
            raise PermissionError("injected audit persistence failure")
        return original(path, value)

    monkeypatch.setattr(module, "_write_new_json", fail)
    result = _ProgressChild(child_with_post(tmp_path, _frame("post", ("Next stage",))),
                            raw.subgoals[0], derived.subgoals[0], record, [], 1).run(
        derived.subgoals[0], ExecutionOptions(execute=True))
    assert result.status == "failed" and result.action_count == result.planning_calls == 1
    assert result.unconfirmed_action_attempts == 0 and "audit failed" in result.error


def test_observation_cache_refuses_conflicting_evidence_under_a_reused_id():
    first, second = _frame("same", ("First",)), _frame("same", ("Different",))
    frames = iter((first, second))
    observer = _ObservedFrames(SimpleNamespace(observe=lambda: next(frames)))
    assert observer.observe().elements[0].text == "First"
    with pytest.raises(ValueError, match="reused"):
        observer.observe()
    assert observer.frames["same"].elements[0].text == "First"


@pytest.mark.parametrize("action_types,eligible,input_count", [(("wait",), False, 0), (("click", "wait"), True, 1)])
def test_only_wait_is_not_input_credit_but_confirmed_click_with_wait_is_allowed(tmp_path, action_types, eligible, input_count):
    _, raw, _, derived, record = contract(tmp_path)
    audits = []
    result = _ProgressChild(child_with_post(tmp_path, _frame("post", ("Next stage",)), action_types=action_types),
                            raw.subgoals[0], derived.subgoals[0], record, audits, 1).run(
        derived.subgoals[0], ExecutionOptions(execute=True))
    assert audits[0]["eligible_for_progress_credit"] is eligible
    assert audits[0]["confirmed_action_count"] == len(action_types)
    assert audits[0]["confirmed_input_count"] == input_count
    assert (result.status == "succeeded") is eligible
