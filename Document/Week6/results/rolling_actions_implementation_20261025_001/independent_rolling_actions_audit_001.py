"""Independent CPU checks with invented observations and offline replies; no GUI/model."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[2]
REPO = BASE / "week6-worktree"
sys.path.insert(0, str(REPO / "src"))
runtime = types.ModuleType("gui_agent.runtime")
runtime.__path__ = [str(REPO / "src/gui_agent/runtime")]
sys.modules["gui_agent.runtime"] = runtime
m = importlib.import_module("gui_agent.planning.workflow_decomposer")
s = importlib.import_module("gui_agent.runtime.schemas")
c = importlib.import_module("gui_agent.schemas")
b = importlib.import_module("gui_agent.models.base")
w = importlib.import_module("gui_agent.runtime.workflow")
recording = importlib.import_module("gui_agent.runtime.recorder")
sessions = importlib.import_module("gui_agent.recording")
Verifier = importlib.import_module("gui_agent.runtime.verification").Verifier
SOURCE = REPO / "src/gui_agent/planning/workflow_decomposer.py"
SOURCE_BEFORE = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
ROWS = []
EPOCH = datetime.now(UTC)


def frame(index, words=("Pending operation",), *, oid=None, path=None, **changes):
    value = s.ObservationSnapshot(
        observation_id=oid or f"obs-{index:04d}", captured_at=EPOCH + timedelta(seconds=index),
        image_path=path or str(BASE / "tmp" / f"cpu-only-invented-{index}.png"),
        screen_info=c.ScreenInfo(screenshot_width=800, screenshot_height=600, control_width=800, control_height=600),
        window_title="CPU imagined application", window_id="CPU-window", foreground_stable=True,
        window_bounds=c.BoundingBox(left=0, top=0, right=800, bottom=600),
        elements=[s.ElementRef(element_id=f"E-{index}-{i}", text=text,
            bounding_box=c.BoundingBox(left=1, top=1 + 20*i, right=200, bottom=20 + 20*i),
            center=c.Point(x=100, y=10 + 20*i), confidence=1) for i, text in enumerate(words)],
    )
    return value.model_copy(update=changes)


class Offline(b.ModelClient):
    name = "mock"
    def __init__(self, count=1):
        super().__init__(model_name="CPU-offline-no-model", max_retries=0)
        self.calls, self.count = [], count
    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return b.ModelResponse(content=json.dumps({"task_id":"CPU", "instruction":"untrusted echo", "steps":[
            {"step_id":str(i), "description":"Invented CPU click", "action_type":"click", "target_text":"CPU control"}
            for i in range(self.count)]}), model_name=self.model_name, provider=self.name)


REQUEST = m.WorkflowRequest(workflow_id="CPU_Rolling", instruction="Finish the whole offline caller operation", expect_text=["Caller finished"], forbid_text=["Danger pending"])
OPTIONS = s.ExecutionOptions(execute=True, max_actions=8, max_planning_attempts=6, task_timeout_seconds=60)


with TemporaryDirectory(prefix="rolling-independent-", dir=Path(__file__).resolve().parent) as temporary:
    ROOT = Path(temporary).resolve()

    def run(name, *, status="succeeded", unknown=0, kinds=("click",), receipt_dry=False,
            post=None, final=None, initial=None, forged=False, plan_steps=1, failure=None,
            record_failure=None, calls=1, execute=True, max_requests=8, unsupported=False,
            workflow_write_failure=False, always_pass_verifier=False):
        directory = ROOT / name
        client = Offline(plan_steps)
        created, observed, requested = [], [], []
        snapshots = iter((post or frame(2, ("Caller finished",), oid="obs-0001"), final or frame(3, ("Caller finished",))))
        class Observer:
            def observe(self, **kwargs):
                current = next(snapshots)
                observed.append(current)
                return current
        class Child(m.TaskRunner):
            def __init__(self, task, childdir, budgeted):
                self.planner = m.TaskPlanner(budgeted, max_steps=7, max_format_retries=0)
                self.observer = Observer()
                self.verifier = Verifier()
                if always_pass_verifier:
                    class UntrustedVerifier(Verifier):
                        def check_task(self, task, observation):
                            return s.VerificationResult(outcome="passed", evidence={"observation_id":observation.observation_id})
                    self.verifier = UntrustedVerifier()
                self.recorder = recording.TaskRecorder(sessions.RunSession(session_id=name, directory=childdir))
                self.absolute_deadline = budgeted.deadline
            def run(self, task, options, **callbacks):
                requested.append((task.model_copy(deep=True), options.model_copy(deep=True)))
                if failure:
                    raise failure("CPU injected interruption")
                planned = [self.planner.plan(task.instruction, context={"CPU":True}, task_id=task.case_id) for _ in range(calls)]
                current = self.observer.observe()
                verdict = (s.VerificationResult(outcome="passed", evidence={"observation_id":current.observation_id})
                           if forged else self.verifier.check_task(task, current))
                steps = [s.StepRecord(index=i+1, step_id=f"CPU-{i}", description="Invented offline receipt", action_type=kind,
                    observation_id=current.observation_id, after_observation_id=current.observation_id,
                    action_result=c.ActionResult(success=True, dry_run=receipt_dry,
                        action=c.DesktopAction(action_type=kind, x=20, y=20)).model_dump(mode="json"))
                    for i, kind in enumerate(kinds)] if all(item.ok for item in planned) else []
                outcome = s.TaskRunResult(run_id=name, case_id=task.case_id, instruction=task.instruction,
                    status=status if all(item.ok for item in planned) else "blocked", execute=options.execute,
                    steps=steps, planning_calls=calls, verification=verdict, unconfirmed_action_attempts=unknown)
                for step in steps:
                    self.recorder.append_step(step)
                self.recorder.write_summary(outcome)
                return outcome
        def factory(task, index, childdir, budgeted):
            child = Child(task, childdir, budgeted)
            created.append(child)
            return SimpleNamespace(planner=child.planner) if unsupported else child
        initial_frames = iter(initial if isinstance(initial, tuple) else (initial or frame(1, oid="obs-0001"),))
        old_new, old_write, old_summary = m._write_new_json, m._write_json, w.WorkflowRunner._write_summary
        def fail_new(path, payload):
            if path.name == record_failure:
                raise PermissionError("CPU audit write failure")
            return old_new(path, payload)
        def fail_write(path, payload):
            if path.name == record_failure:
                raise PermissionError("CPU update write failure")
            return old_write(path, payload)
        def fail_summary(self, result):
            if result.total_actions:
                raise PermissionError("CPU outer summary failure after receipt")
            return old_summary(self, result)
        result, error = None, None
        try:
            m._write_new_json, m._write_json = fail_new, fail_write
            if workflow_write_failure:
                w.WorkflowRunner._write_summary = fail_summary
            result = m.AutomaticWorkflowRunner(directory, client, factory, max_model_requests=max_requests,
                initial_observer=lambda _: next(initial_frames), protocol_mode="rolling_actions").run(
                    REQUEST, OPTIONS.model_copy(update={"execute":execute}))
        except BaseException as exc:
            error = type(exc).__name__
        finally:
            m._write_new_json, m._write_json, w.WorkflowRunner._write_summary = old_new, old_write, old_summary
        return SimpleNamespace(result=result, error=error, created=created, observed=observed, requested=requested,
                               client=client, directory=directory)

    def check(name, condition, detail=None):
        ROWS.append({"case":name, "passed":bool(condition), "detail":detail})

    good = run("good")
    result = good.result
    check("valid_whole_caller_goal_has_no_native_decomposition_or_semantic_subgoal_claim", result is not None
        and result.status == "succeeded" and result.decomposition.status == "not_requested"
        and result.decomposition.decomposition_calls == 0 and result.completed_subgoals == []
        and result.progress_verifications == [] and result.completion_basis == "executed_caller_goal"
        and len(good.observed) == 2 and result.total_actions == 1 and result.model_requests == 1)
    check("caller_checks_and_full_instruction_remain_immutable", result is not None and good.requested
        and good.requested[0][0].expect_text == REQUEST.expect_text and good.requested[0][0].forbid_text == REQUEST.forbid_text
        and good.requested[0][0].instruction == m._caller_anchored_instruction(REQUEST, max_length=None)
        and result.decomposition.workflow.final_check.expect_text == REQUEST.expect_text
        and result.decomposition.workflow.final_check.forbid_text == REQUEST.forbid_text)
    check("single_step_plan_record_and_planner_restoration", result is not None and result.rolling_actions_audit is not None
        and good.created[0].planner.max_steps == 7 and "plan" not in good.created[0].planner.__dict__
        and len(result.rolling_actions_audit.plans) == 1
        and result.rolling_actions_audit.plans[0].instruction == good.requested[0][0].instruction
        and result.rolling_actions_audit.plans[0].returned_steps == 1)
    for label, status, unknown in (("failed", "failed",0),("blocked","blocked",0),("cancelled","cancelled",0),("timed_out","timed_out",0),("unknown","succeeded",1)):
        out = run(label, status=status, unknown=unknown)
        check(f"{label}_child_cannot_be_rescued_by_final", out.result is not None and out.result.status != "succeeded"
            and out.result.total_actions == 1 and len(out.observed) == 1
            and out.result.unconfirmed_action_attempts == unknown)
    for label, kwargs in (("wait_only",{"kinds":("wait",)}), ("dry_receipt",{"receipt_dry":True}), ("no_input",{"kinds":()})):
        out = run(label, **kwargs)
        check(f"{label}_earns_no_executed_goal_credit", out.result is not None and out.result.status != "succeeded"
            and len(out.observed) == 1 and out.result.rolling_actions_audit is not None
            and out.result.rolling_actions_audit.confirmed_input_count == 0)
    out = run("click_wait", kinds=("click","wait"))
    check("real_input_can_accompany_wait_without_counting_wait_as_input", out.result.status == "succeeded"
        and out.result.total_actions == 2 and out.result.rolling_actions_audit.confirmed_input_count == 1)
    for label, changes in (("blank",{"elements":[]}), ("ocr_error",{"errors":["CPU error"]}),
                           ("window_changed",{"window_id":"other"}), ("unstable",{"foreground_stable":False}),
                           ("missing_bounds",{"window_bounds":None})):
        out = run(label, post=frame(2,("Caller finished",),**changes))
        check(f"{label}_post_blocks", out.result is not None and out.result.status != "succeeded" and len(out.observed) == 1)
    out = run("false_verdict", post=frame(2,("Caller still pending",)), forged=True)
    check("forged_child_pass_on_unmet_cached_caller_rules_blocks_independent_final", out.result is not None
        and out.result.status != "succeeded" and len(out.observed) == 1)
    out = run("false_final_verdict", final=frame(3,("Caller still pending",)), always_pass_verifier=True)
    check("custom_verifier_cannot_forge_immutable_fresh_final", out.result is not None and out.result.status != "succeeded"
        and out.result.total_actions == 1 and len(out.observed) == 2)
    for label, last in (("fresh_final_fails",frame(3,("Caller still pending",))),
                       ("stale_final_time",frame(3,("Caller finished",),captured_at=EPOCH+timedelta(seconds=1.5))),
                       ("reused_final_path",frame(3,("Caller finished",),path=frame(2).image_path)),
                       ("reused_final_id",frame(3,("Caller finished",),oid="obs-0001"))):
        out = run(label, final=last)
        check(label, out.result is not None and out.result.status != "succeeded" and out.result.total_actions == 1)
    out = run("audit_failure", record_failure="rolling_actions_audit.json")
    check("post_input_audit_failure_preserves_confirmed_lower_bound", out.result is not None and out.result.status == "failed"
        and out.result.total_actions == out.result.planning_calls == 1 and len(out.observed) == 1)
    out = run("outer_summary_failure", workflow_write_failure=True)
    check("outer_workflow_summary_failure_preserves_confirmed_lower_bound", out.result is not None and out.result.status == "failed"
        and out.result.total_actions >= 1 and out.result.planning_calls >= 1,
        {"status":out.result.status if out.result else None,"actions":out.result.total_actions if out.result else None})
    out = run("unsupported", unsupported=True)
    check("unsupported_custom_runner_refused_before_planning", out.result is not None and out.result.status == "failed"
        and not out.client.calls and not out.observed and out.result.total_actions == 0)
    for label, error in (("exception_restore",RuntimeError),("cancel_restore",KeyboardInterrupt)):
        out = run(label,failure=error)
        check(label, out.result is not None and out.result.status != "succeeded" and out.created[0].planner.max_steps == 7
            and "plan" not in out.created[0].planner.__dict__ and not out.client.calls)
    out = run("two_steps",plan_steps=2)
    check("two_model_steps_rejected_before_input", out.result is not None and out.result.status != "succeeded"
        and out.result.total_actions == 0 and out.created[0].planner.max_steps == 7)
    out = run("preexisting",initial=(frame(1,("Caller finished",),oid="obs-0001"),frame(2,("Caller finished",),oid="obs-0001")))
    check("preexisting_final_is_explicit_two_fresh_capture_zero_input_zero_model", out.result is not None
        and out.result.status == "succeeded" and out.result.already_satisfied
        and out.result.completion_basis == "preexisting_caller_final"
        and out.result.total_actions == out.result.model_requests == 0 and not out.created)
    out = run("stale_preexisting",initial=(frame(1,("Caller finished",)),frame(1,("Caller finished",))))
    check("preexisting_stale_reused_frame_cannot_pass", out.result is not None and out.result.status != "succeeded"
        and out.result.total_actions == out.result.model_requests == 0)
    out = run("dry_run",execute=False,receipt_dry=True,status="dry_run_completed")
    check("dry_run_never_labeled_executed_success", out.result is not None and out.result.status == "dry_run_completed"
        and out.result.completion_basis == "" and len(out.observed) == 1)
    check("programmatic_legacy_default_retained", inspect.signature(m.AutomaticWorkflowRunner).parameters["protocol_mode"].default == "observed_departure")

payload = {"schema":"week6.independent_rolling_actions_review.v1",
    "mode":"CPU_fabricated_snapshots_and_offline_transport_and_TaskRunner_subclass",
    "source_sha256_before":SOURCE_BEFORE,"source_sha256_after":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "source_unchanged":SOURCE_BEFORE==hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "checks":ROWS,"all_checks_passed":all(row["passed"] for row in ROWS),
    "actual_model_requests":0,"actual_HTTP_requests":0,"desktop_capture_count":0,"desktop_input_count":0,"real_GUI_success":False}
target = Path(__file__).with_suffix(".json")
with target.open("x",encoding="utf-8") as handle:
    handle.write(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(payload,ensure_ascii=False))
raise SystemExit(0 if payload["all_checks_passed"] and payload["source_unchanged"] else 1)
