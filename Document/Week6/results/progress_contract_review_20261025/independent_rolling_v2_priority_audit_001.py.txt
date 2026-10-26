"""Independent caller-priority checks using real TaskRunner and CPU-only adapters."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
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
Recorder = importlib.import_module("gui_agent.runtime.recorder").TaskRecorder
Session = importlib.import_module("gui_agent.recording").RunSession
ActionAdapter = importlib.import_module("gui_agent.runtime.action_adapter").ActionAdapter
Verifier = importlib.import_module("gui_agent.runtime.verification").Verifier
WATCHED = [REPO / "src/gui_agent/planning/workflow_decomposer.py", REPO / "src/gui_agent/runtime/runner.py"]
BEFORE = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in WATCHED}
ROWS = []
EPOCH = datetime(2026, 10, 25, tzinfo=UTC)


class Clock:
    def __init__(self): self.value = 100.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


class Offline(b.ModelClient):
    name = "mock"
    def __init__(self, *, expected="Phantom nebula sentinel", kind="click", count=1, clock=None, cost=0):
        super().__init__(model_name="CPU-only-offline", max_retries=0, timeout_seconds=90)
        self.calls, self.expected, self.kind, self.count, self.clock, self.cost = [], expected, kind, count, clock, cost
    def complete(self, messages, **kwargs):
        self.calls.append({"messages":messages,"kwargs":kwargs})
        self.clock.advance(self.cost)
        text = json.dumps({"task_id":"CPU", "instruction":"untrusted model echo", "steps":[
            {"step_id":f"S{i}","description":"CPU operation", "action_type":self.kind,
             "target_text":"Advance control", "arguments":{},"expected_result":self.expected}
            for i in range(self.count)]})
        return b.ModelResponse(content=text, provider=self.name, model_name=self.model_name, raw_response={"CPU":text})


REQUEST = m.WorkflowRequest(workflow_id="CPU_Priority", instruction="Complete the original offline caller operation",
    expect_text=["Caller finished"],forbid_text=["Caller error"])


with TemporaryDirectory(prefix="priority-independent-",dir=Path(__file__).resolve().parent) as temp:
    ROOT = Path(temp).resolve()
    def run(name, *, required=1, expected="Phantom nebula sentinel", action_kind="click", executor_failure=False,
            unknown=False, declined=False, execute=True, transient=False, post_fault=None, final_fault=None,
            caller_recedes=False, max_actions=8, max_plans=1, max_requests=8, count=1, model_cost=0, legacy=False):
        directory = ROOT / name
        state = SimpleNamespace(stage=0, attempts=0, captures=0, children=[], planners=[], childdir=None)
        clock = Clock()
        client = Offline(expected=expected,kind=action_kind,count=count,clock=clock,cost=model_cost)
        def frame(sequence, oid, origin):
            words = ["Advance control", "Caller finished" if state.stage >= required else "Caller pending"]
            observed = s.ObservationSnapshot(observation_id=oid,captured_at=EPOCH+timedelta(seconds=sequence),
                image_path=f"/CPU-invented/{name}/{origin}/{sequence}.png",
                screen_info=c.ScreenInfo(screenshot_width=800,screenshot_height=600,control_width=800,control_height=600),
                window_title="CPU application",window_id="CPU-window",foreground_stable=True,
                window_bounds=c.BoundingBox(left=0,top=0,right=800,bottom=600),
                elements=[s.ElementRef(element_id=f"{oid}-{i}",text=word,
                    bounding_box=c.BoundingBox(left=10,top=10+40*i,right=300,bottom=40+40*i),
                    center=c.Point(x=155,y=25+40*i),confidence=1) for i,word in enumerate(words)])
            return observed
        class Observer:
            def observe(self,**kwargs):
                state.captures += 1
                observed = frame(state.captures+1,f"obs-{state.captures:04d}","child")
                if state.captures == 3 and post_fault:
                    if post_fault == "blank": observed.elements=[]
                    elif post_fault == "window": observed.window_id="Other-window"
                    elif post_fault == "error": observed.errors=["CPU missing OCR"]
                if caller_recedes and state.captures >= 4:
                    observed.elements[1].text="Caller pending"
                if final_fault and state.childdir is not None and (state.childdir/"rolling_actions_audit.json").exists():
                    if final_fault == "blank": observed.elements=[]
                    elif final_fault == "window": observed.window_id="Other-window"
                    elif final_fault == "stale": observed.captured_at=EPOCH
                return observed
        class Executor:
            def execute(self,action,*,dry_run=False,screen=None):
                if dry_run: return c.ActionResult(action=action,success=True,dry_run=True)
                state.attempts += 1
                if unknown:
                    state.stage=required
                    raise RuntimeError("CPU unknown executor completion")
                if executor_failure:
                    state.stage=required
                    return c.ActionResult(action=action,success=False,dry_run=False,error="known CPU executor failure")
                if transient and state.attempts==1:
                    return c.ActionResult(action=action,success=False,dry_run=False,error="known CPU transient failure")
                state.stage += 1
                return c.ActionResult(action=action,success=True,dry_run=False)
        def factory(task,index,childdir,budgeted=None):
            state.childdir=childdir
            planner=m.TaskPlanner(budgeted or client,max_steps=6,max_format_retries=0)
            child=m.TaskRunner(observer=Observer(),planner=planner,adapter=ActionAdapter(platform="win32"),
                executor=Executor(),verifier=Verifier(),recorder=Recorder(Session(session_id=name,directory=childdir)),
                clock=clock,sleep=lambda _:None)
            state.children.append(child);state.planners.append(planner)
            return child
        options=s.ExecutionOptions(execute=execute,confirm=declined,max_actions=max_actions,max_planning_attempts=max_plans,
            task_timeout_seconds=5 if model_cost else 60,verification_timeout_seconds=0)
        if legacy:
            directory.mkdir()
            child=factory(None,0,directory)
            result=child.run(s.TaskSpec(case_id="Legacy",instruction=REQUEST.instruction,
                expect_text=REQUEST.expect_text,forbid_text=REQUEST.forbid_text),options)
            return SimpleNamespace(result=result,state=state,client=client,audit=None,child=result,compilation=None)
        result=m.AutomaticWorkflowRunner(directory,client,factory,initial_observer=lambda _:frame(0,"obs-0001","root"),
            clock=clock,max_model_requests=max_requests,protocol_mode="rolling_actions").run(REQUEST,options,
                confirm=(lambda _:False) if declined else None)
        child=None
        if state.childdir is not None and (state.childdir/"task_summary.json").exists():
            child=json.loads((state.childdir/"task_summary.json").read_text(encoding="utf-8"))
        compilation=json.loads((directory/"rolling_compilation.json").read_text(encoding="utf-8"))
        return SimpleNamespace(result=result,state=state,client=client,audit=result.rolling_actions_audit,child=child,compilation=compilation)
    def check(name,passed,detail=None): ROWS.append({"case":name,"passed":bool(passed),"detail":detail})

    good=run("bad_prediction_good_caller")
    check("model_step_miss_does_not_trigger_extra_plan_or_input_after_actual_caller_goal",good.result.status=="succeeded"
        and good.state.attempts==good.result.total_actions==good.result.planning_calls==good.result.model_requests==1
        and len(good.client.calls)==1 and good.result.completion_basis=="executed_caller_goal")
    check("raw_failed_model_step_is_preserved_beside_passed_caller_and_independent_final",good.audit is not None
        and good.audit.actions[0].verification.outcome=="failed"
        and good.child["steps"][0]["verification"]["outcome"]=="failed"
        and good.audit.caller_verification.outcome==good.result.final_verification.outcome=="passed"
        and good.audit.post_observation_id!=good.result.workflow_result.final_observation_id)
    check("v2_priority_is_declared_before_execution_only_on_caller_child",good.compilation["workflow"]["subgoals"][0]["gate_on_step_verification"] is False
        and good.compilation["workflow"]["final_check"]["expect_text"]==REQUEST.expect_text
        and good.result.decomposition.protocol_version=="week6.rolling_actions.v2"
        and good.result.completed_subgoals==good.result.progress_verifications==[])
    check("planner_one_step_limit_is_restored_after_model_step_miss",good.state.planners[0].max_steps==6
        and "plan" not in good.state.planners[0].__dict__
        and good.audit.plans[0].max_steps==good.audit.plans[0].returned_steps==1)
    legacy=run("legacy_strict",legacy=True)
    check("manual_TaskRunner_default_keeps_strict_model_step_gate",legacy.result.status=="failed"
        and legacy.state.attempts==1 and legacy.result.steps[0].verification.outcome=="failed")
    failed=run("failed_receipt",executor_failure=True)
    check("failed_executor_receipt_cannot_be_rescued_by_visible_caller_text",failed.result.status!="succeeded"
        and failed.state.attempts==failed.result.total_actions==1 and failed.result.final_verification is None
        and not failed.audit.eligible_for_final_verification)
    unknown=run("unknown_receipt",unknown=True,max_plans=4)
    check("unknown_dispatch_stops_without_replanning_or_caller_rescue",unknown.result.status!="succeeded"
        and unknown.state.attempts==unknown.result.model_requests==1 and unknown.result.unconfirmed_action_attempts==1
        and unknown.result.final_verification is None)
    cancelled=run("declined",declined=True)
    check("declined_confirmation_stays_cancelled_without_input",cancelled.result.status=="cancelled"
        and cancelled.state.attempts==0 and cancelled.result.final_verification is None)
    dry=run("dry",execute=False)
    check("dry_run_is_never_executed_caller_success",dry.result.status=="dry_run_completed" and dry.state.attempts==0
        and dry.result.completion_basis=="")
    waited=run("wait",action_kind="wait")
    check("wait_alone_cannot_gain_caller_input_credit",waited.result.status!="succeeded"
        and waited.audit.confirmed_input_count==0 and not waited.audit.eligible_for_final_verification)
    for fault in ("blank","window","error"):
        out=run("post_"+fault,post_fault=fault)
        check("post_"+fault+"_cannot_gain_caller_credit",out.result.status!="succeeded" and out.state.attempts==1
            and out.result.total_actions==1 and out.result.model_requests==1)
    for fault in ("blank","window","stale"):
        out=run("final_"+fault,final_fault=fault)
        check("independent_final_"+fault+"_blocks",out.result.status!="succeeded" and out.state.attempts==1
            and out.result.total_actions==1)
    receded=run("caller_recedes",caller_recedes=True)
    check("failed_raw_step_plus_caller_failure_does_not_become_success",receded.result.status!="succeeded"
        and receded.result.total_actions==receded.state.attempts==1 and receded.result.model_requests==1)
    for cap in ("actions","planning","http"):
        out=run("cap_"+cap,required=2,max_actions=1 if cap=="actions" else 8,
            max_plans=1 if cap=="planning" else 4,max_requests=1 if cap=="http" else 8)
        check("caller_priority_does_not_renew_"+cap+"_budget",out.result.status!="succeeded"
            and out.state.attempts==out.result.total_actions==out.result.model_requests==1)
    late=run("late_model",model_cost=6)
    check("late_model_reply_cannot_dispatch_or_renew_deadline",late.result.status=="timed_out"
        and late.state.attempts==0 and late.result.model_requests==1)
    two=run("two_steps",count=2)
    check("two_model_steps_still_refused_before_any_input",two.result.status!="succeeded" and two.state.attempts==0
        and two.result.total_actions==0 and two.state.planners[0].max_steps==6)
    recovered=run("known_transient",transient=True,max_plans=4)
    check("known_failed_receipt_needs_a_new_successful_input_before_caller_completion",recovered.result.status=="succeeded"
        and recovered.state.attempts==recovered.result.total_actions==recovered.result.model_requests==2
        and [step.action_result["success"] for step in recovered.audit.actions]==[False,True]
        and recovered.audit.confirmed_input_count==1
        and recovered.audit.actions[-1].verification.outcome=="failed")

AFTER={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in WATCHED}
payload={"schema":"week6.independent_rolling_v2_priority_review.v1",
    "mode":"CPU_actual_TaskRunner_with_fabricated_frames_and_offline_model_and_executor",
    "source_sha256_before":BEFORE,"source_sha256_after":AFTER,"source_unchanged":BEFORE==AFTER,
    "checks":ROWS,"all_checks_passed":all(row["passed"] for row in ROWS),
    "actual_model_requests":0,"actual_HTTP_requests":0,"desktop_capture_count":0,"desktop_input_count":0,"real_GUI_success":False}
with Path(__file__).with_suffix(".json").open("x",encoding="utf-8") as handle:
    handle.write(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(payload,ensure_ascii=False))
raise SystemExit(0 if payload["all_checks_passed"] and payload["source_unchanged"] else 1)
