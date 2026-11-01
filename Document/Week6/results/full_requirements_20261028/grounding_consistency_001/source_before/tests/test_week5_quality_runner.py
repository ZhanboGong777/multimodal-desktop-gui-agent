"""Week 5 read-only quality preflight regressions using existing synthetic helpers."""

from __future__ import annotations

import json

import pytest
from test_runtime_runner import (
    FakeExecutor,
    FakeObserver,
    FakePlanner,
    SimulatedChat,
    _plan,
    _runner,
    _simulated_t04,
    _StepClock,
)

from gui_agent.planning.schemas import PlanStep
from gui_agent.runtime import runner as runner_module
from gui_agent.runtime.schemas import ExecutionOptions, TaskSpec, VerificationResult


# ───────── read-only initial message capture quality ─────────
class _QualityClock:
    now = 0.0

    def __call__(self):
        return self.now


def _quality_preflight_case(tmp_path, *, frames=None, observer=None, clock=None):
    """Real prepared PNGs, fake semantic refusal; no OS, pointer or model call."""
    source = SimulatedChat(tmp_path).observe()
    observer = observer or FakeObserver(frames or [source])
    planner = FakePlanner(_plan(PlanStep(step_id="finish", description="done", action_type="finish")))
    executor = FakeExecutor()
    runner, recorder = _runner(tmp_path, observer, planner, executor, clock=clock)
    task = TaskSpec(case_id="T04", instruction="send a message", expect_text=["independent-marker"],
                    message_conversation="文件传输助手", success_rules=["strict message sent"])
    context_calls = []
    def refuse_context(spec, snapshot, **kwargs):
        context_calls.append((spec, snapshot, kwargs))
        return VerificationResult(outcome="failed", method="automatic", detail="independent semantic refusal")
    runner.verifier.check_message_context = refuse_context
    options = ExecutionOptions(execute=True, confirm=False, max_planning_attempts=3,
                               task_timeout_seconds=5, verification_timeout_seconds=0)
    return runner, task, options, observer, planner, executor, recorder, context_calls, source


def _quality_ledger(recorder):
    return json.loads((recorder.directory / "screenshot_quality_initial.json").read_text(encoding="utf-8"))


def test_message_quality_wrong_foreground_stops_before_semantic_or_planning(tmp_path):
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path)
    observer.frames = [source.model_copy(update={"window_title": "ChatGPT", "window_class": "Chrome_WidgetWin_1"})]
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "blocked" and "foreground_application_obstruction" in result.stop_reason
    assert observer.calls == 1 and not calls and not planner.contexts and not executor.actions
    assert ledger["capture_attempts"] == 1 and ledger["outcome"] == "blocked"
    assert ledger["samples"][0]["quality"]["evidence"]["window_title"] == "ChatGPT"
    assert (recorder.directory / f"{source.observation_id}.json").exists()


@pytest.mark.parametrize("title,class_name", [
    ("ChatGPT contacts", "Chrome_WidgetWin_1"), ("ChatGPT", "Qt51514QWindowIcon"),
])
def test_message_quality_does_not_guess_obstruction_from_title_substrings(tmp_path, title, class_name):
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path)
    observer.frames = [source.model_copy(update={"window_title": title, "window_class": class_name})]
    result = runner.run(task, options)
    assert result.status == "blocked" and "independent semantic refusal" in result.stop_reason
    assert _quality_ledger(recorder)["outcome"] == "passed"
    assert observer.calls == 1 and len(calls) == 1 and not planner.contexts and not executor.actions


def test_message_quality_transient_capture_gets_one_fresh_frame_then_semantic_refusal(tmp_path):
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path)
    observer.frames = [
        source.model_copy(update={"foreground_stable": False}),
        source.model_copy(update={"observation_id": "quality-0002"}),
    ]
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "blocked" and "independent semantic refusal" in result.stop_reason
    assert observer.calls == 2 and len(calls) == 1
    assert calls[0][1].observation_id == "quality-0002" and calls[0][2] == {"require_empty": True}
    assert ledger["outcome"] == "passed" and ledger["capture_attempts"] == 2
    assert [sample["quality"]["code"] for sample in ledger["samples"]] == [
        "foreground_capture_unstable", "basic_frame_integrity_passed",
    ]
    assert not planner.contexts and not executor.actions
    assert (recorder.directory / "quality-0002.json").exists()
    assert task.expect_text[0] not in json.dumps(ledger), "quality never receives or reconstructs the target marker"


@pytest.mark.parametrize("transient", ["uniform", "unstable"])
def test_message_quality_stops_after_three_captures_without_semantic_or_actions(tmp_path, transient):
    from PIL import Image
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path)
    if transient == "uniform":
        Image.new("RGB", (800, 500), "white").save(source.image_path)
    else:
        source = source.model_copy(update={"foreground_stable": False})
    observer.frames = [source.model_copy(update={"observation_id": f"quality-{index:04d}"}) for index in range(1, 4)]
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "blocked" and ledger["code"] == "quality_capture_limit_reached"
    assert observer.calls == ledger["capture_attempts"] == ledger["max_captures"] == 3
    assert [sample["observation_id"] for sample in ledger["samples"]] == ["quality-0001", "quality-0002", "quality-0003"]
    assert not calls and not planner.contexts and not executor.actions
    assert ledger["gui_actions"] == ledger["model_calls"] == 0


@pytest.mark.parametrize("field,value", [
    ("window_id", ""), ("window_title", ""), ("window_class", ""), ("window_bounds", None),
])
def test_message_quality_missing_foreground_evidence_cannot_be_recaptured_into_authorization(tmp_path, field, value):
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path)
    observer.frames = [source.model_copy(update={field: value}), source]
    result = runner.run(task, options)
    assert result.status == "blocked" and observer.calls == 1
    assert _quality_ledger(recorder)["code"] == "foreground_evidence_missing"
    assert not calls and not planner.contexts and not executor.actions


@pytest.mark.parametrize("change", ["handle", "title", "class", "bounds", "screen"])
def test_message_quality_retry_cannot_follow_another_window_or_geometry(tmp_path, change):
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path)
    updates = {
        "handle": {"window_id": "other:2"}, "title": {"window_title": "Other chat"},
        "class": {"window_class": "OtherWindow"},
        "bounds": {"window_bounds": source.window_bounds.model_copy(update={"left": source.window_bounds.left + 1})},
        "screen": {"screen_info": source.screen_info.model_copy(update={"monitor_left": 1})},
    }[change]
    observer.frames = [source.model_copy(update={"foreground_stable": False}),
                       source.model_copy(update={"observation_id": "quality-0002", **updates})]
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "blocked" and observer.calls == 2
    assert ledger["code"] == "foreground_changed_during_quality_recheck" and len(ledger["samples"]) == 2
    assert ledger["samples"][1]["observation_id"] == "quality-0002"
    assert not calls and not planner.contexts and not executor.actions


def test_message_quality_initial_budget_expiry_starts_no_capture(tmp_path):
    runner, task, options, observer, planner, executor, recorder, calls, _ = _quality_preflight_case(tmp_path, clock=_StepClock(step=10))
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "timed_out" and observer.calls == ledger["capture_attempts"] == 0
    assert ledger["code"] == "task_budget_exhausted_before_capture"
    assert not calls and not planner.contexts and not executor.actions


@pytest.mark.parametrize("expired_capture", [1, 2])
def test_message_quality_capture_consuming_budget_keeps_sample_and_stops(tmp_path, expired_capture):
    clock = _QualityClock()
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path, clock=clock)
    observer.frames = [source.model_copy(update={"foreground_stable": expired_capture != 2}),
                       source.model_copy(update={"observation_id": "quality-0002"})]
    observe = observer.observe
    def slow_capture(**kwargs):
        snapshot = observe(**kwargs)
        if observer.calls == expired_capture:
            clock.now = 10
        return snapshot
    observer.observe = slow_capture
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "timed_out" and observer.calls == expired_capture
    assert ledger["capture_attempts"] == expired_capture and ledger["outcome"] == "timed_out"
    assert ledger["samples"][-1]["image_path"] == source.image_path
    assert ledger["samples"][-1]["outcome"] == "timed_out"
    assert not calls and not planner.contexts and not executor.actions


def test_message_quality_cpu_check_cannot_authorize_after_budget_expiry(tmp_path, monkeypatch):
    clock = _QualityClock()
    runner, task, options, observer, planner, executor, recorder, calls, _ = _quality_preflight_case(tmp_path, clock=clock)
    original = runner_module.assess_screenshot_quality
    def slow_quality(snapshot):
        assessment = original(snapshot)
        clock.now = 10
        return assessment
    monkeypatch.setattr(runner_module, "assess_screenshot_quality", slow_quality)
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "timed_out" and observer.calls == 1
    assert ledger["samples"][0]["quality"]["outcome"] == "passed"
    assert ledger["samples"][0]["outcome"] == "timed_out"
    assert not calls and not planner.contexts and not executor.actions


def test_message_quality_evidence_io_cannot_start_semantic_request_after_budget(tmp_path, monkeypatch):
    clock = _QualityClock()
    runner, task, options, observer, planner, executor, recorder, calls, _ = _quality_preflight_case(tmp_path, clock=clock)
    save_json = recorder.session.save_json
    def slow_quality_record(filename, payload):
        result = save_json(filename, payload)
        if filename == "screenshot_quality_initial.json":
            clock.now = 10
        return result
    monkeypatch.setattr(recorder.session, "save_json", slow_quality_record)
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "timed_out" and observer.calls == 1
    assert ledger["code"] == "task_budget_exhausted_during_quality_recording"
    assert ledger["samples"][0]["quality"]["outcome"] == "passed"
    assert ledger["samples"][0]["outcome"] == "timed_out"
    assert not calls and not planner.contexts and not executor.actions


@pytest.mark.parametrize("expired", [False, True])
def test_message_quality_reobservation_error_preserves_prior_frame_and_failure(tmp_path, expired):
    clock = _QualityClock()
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path, clock=clock)
    observer.frames = [source.model_copy(update={"foreground_stable": False})]
    observe = observer.observe
    def broken_recapture(**kwargs):
        if observer.calls:
            observer.calls += 1
            if expired:
                clock.now = 10
            raise RuntimeError("saved monitor capture unavailable")
        return observe(**kwargs)
    observer.observe = broken_recapture
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == ("timed_out" if expired else "blocked")
    assert observer.calls == ledger["capture_attempts"] == 2
    assert ledger["samples"][0]["quality"]["code"] == "foreground_capture_unstable"
    assert ledger["samples"][1]["capture_error"] == "RuntimeError: saved monitor capture unavailable"
    assert not calls and not planner.contexts and not executor.actions


def test_message_quality_initial_capture_failure_is_auditable(tmp_path):
    runner, task, options, observer, planner, executor, recorder, calls, _ = _quality_preflight_case(tmp_path)
    def broken_capture(**kwargs):
        observer.calls += 1
        raise RuntimeError("first capture unavailable")
    observer.observe = broken_capture
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "blocked" and observer.calls == ledger["capture_attempts"] == 1
    assert ledger["samples"][0]["capture_error"] == "RuntimeError: first capture unavailable"
    assert not calls and not planner.contexts and not executor.actions


def test_message_quality_interruption_during_recapture_keeps_ledger_and_cancelled_summary(tmp_path):
    runner, task, options, observer, planner, executor, recorder, calls, source = _quality_preflight_case(tmp_path)
    observer.frames = [source.model_copy(update={"foreground_stable": False})]
    observe = observer.observe
    def interrupted_recapture(**kwargs):
        if observer.calls:
            observer.calls += 1
            raise KeyboardInterrupt
        return observe(**kwargs)
    observer.observe = interrupted_recapture
    result = runner.run(task, options)
    ledger = _quality_ledger(recorder)
    assert result.status == "cancelled" and ledger["outcome"] == "cancelled"
    assert len(ledger["samples"]) == 2 and ledger["samples"][1]["capture_error"] == "KeyboardInterrupt"
    assert not calls and not planner.contexts and not executor.actions


def test_message_quality_pass_does_not_retry_or_override_independent_draft_failure(tmp_path):
    runner, task, options, chat, client, recorder = _simulated_t04(tmp_path)
    draft_calls = []
    def refuse_draft(spec, frame):
        draft_calls.append(frame.observation_id)
        return VerificationResult(outcome="failed", method="automatic",
                                  detail="current visible glyph evidence differs; cursor state unknown")
    runner.verifier.check_message_draft = refuse_draft
    options.max_planning_attempts = 3
    result = runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)
    assert result.status == "failed" and not chat.sent
    assert [action.action_type for action, _ in chat.actions] == ["click", "type_text", "move"]
    assert len(draft_calls) == 1 and len(client.planning_contexts) == 1
    ledger = _quality_ledger(recorder)
    assert ledger["outcome"] == "passed" and ledger["capture_attempts"] == 1
    assert ledger["send_authorization"] is False


@pytest.mark.parametrize("execute,is_message", [(False, True), (True, False)])
def test_capture_quality_does_not_change_dry_run_or_nonmessage_paths(tmp_path, execute, is_message):
    runner, task, options, observer, planner, _, recorder, calls, source = _quality_preflight_case(tmp_path)
    observer.frames = [source.model_copy(update={"window_id": "", "image_path": None})]
    if not is_message:
        task.message_conversation = ""
        task.case_id = "OTHER"
    options.execute = execute
    options.max_planning_attempts = 1
    runner.run(task, options)
    assert planner.contexts
    assert all(not kwargs.get("require_empty") for _, _, kwargs in calls), "no new initial context gate"
    assert not (recorder.directory / "screenshot_quality_initial.json").exists()
