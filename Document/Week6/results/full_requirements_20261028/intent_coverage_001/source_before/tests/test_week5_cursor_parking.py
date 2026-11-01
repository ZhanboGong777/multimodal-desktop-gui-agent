"""Offline geometry and failure-path regressions; no real desktop or model."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image, ImageDraw
from test_runtime_runner import _simulated_t04

from gui_agent.coordinates import is_inside_screen, screenshot_to_control
from gui_agent.runtime import runner as runner_module
from gui_agent.runtime.cursor_parking import (
    CursorParkingDecision,
    CursorParkingError,
    plan_cursor_parking,
    verify_parking_frame,
)
from gui_agent.runtime.schemas import ObservationSnapshot, VerificationResult
from gui_agent.schemas import ActionResult, BoundingBox, ScreenInfo


def scene(tmp_path, *, bounds=None, scale=(1, 1), offset=(0, 0), oid="parking-1"):
    bounds = bounds or BoundingBox(left=160, top=80, right=600, bottom=400)
    image = Image.new("RGB", (800, 500), "#777777")
    draw = ImageDraw.Draw(image)
    draw.rectangle((bounds.left, bounds.top, bounds.right - 1, bounds.bottom - 1),
                   fill="white", outline="black", width=2)
    draw.text((bounds.left + 10, bounds.top + 10), "real pixel fixture", fill="black")
    path = tmp_path / f"{oid}.png"
    image.save(path)
    return ObservationSnapshot(
        observation_id=oid, captured_at=datetime(2026, 10, 9, tzinfo=UTC), image_path=str(path),
        screen_info=ScreenInfo(screenshot_width=800, screenshot_height=500,
                               control_width=int(800 * scale[0]), control_height=int(500 * scale[1]),
                               monitor_left=offset[0], monitor_top=offset[1]),
        window_id="real-fixture-handle", window_title="WeChat", window_class="QtFixture",
        window_bounds=bounds, foreground_stable=True,
    )


@pytest.mark.parametrize("scale,offset", [
    ((1, 1), (0, 0)), ((1.5, 1.25), (-1600, 250)), ((0.5, 0.75), (800, -600)),
])
def test_parking_uses_native_geometry_and_real_monitor_transform(tmp_path, scale, offset):
    frame = scene(tmp_path, scale=scale, offset=offset)
    decision = plan_cursor_parking(frame, step_id="internal-parking")
    assert decision.outcome == "planned"
    resolved, bounds = decision.resolved, frame.window_bounds
    assert resolved.action.action_type == "move" and resolved.element_id is None
    assert resolved.control_point == screenshot_to_control(resolved.screenshot_point, frame.screen_info)
    assert is_inside_screen(resolved.control_point, frame.screen_info)
    p = resolved.screenshot_point
    assert not (bounds.left <= p.x < bounds.right and bounds.top <= p.y < bounds.bottom)
    assert decision.evidence["clearance_control_pixels"] >= 32
    assert not decision.evidence["send_authorization"]
    assert plan_cursor_parking(frame, step_id="internal-parking") == decision


@pytest.mark.parametrize("bounds", [
    BoundingBox(left=0, top=0, right=800, bottom=500),
    BoundingBox(left=40, top=40, right=760, bottom=460),
])
def test_no_measured_safe_exterior_never_invents_an_inside_point(tmp_path, bounds):
    decision = plan_cursor_parking(scene(tmp_path, bounds=bounds), step_id="parking")
    assert decision.outcome == "skipped" and decision.resolved is None
    assert decision.code == "no_safe_window_exterior"


@pytest.mark.parametrize("bad", ["missing_image", "unstable", "degraded", "bad_scale"])
def test_unusable_geometry_never_plans_a_move(tmp_path, bad):
    frame = scene(tmp_path)
    update = {"missing_image": {"image_path": None}, "unstable": {"foreground_stable": False},
              "degraded": {"errors": ["capture degraded"]},
              "bad_scale": {"screen_info": frame.screen_info.model_copy(update={"scale_x": 9})}}[bad]
    decision = plan_cursor_parking(frame.model_copy(update=update), step_id="parking")
    assert decision.outcome == "blocked" and decision.resolved is None


def test_geometry_interface_rejects_task_labels_and_preserves_input_pixels(tmp_path):
    frame = scene(tmp_path)
    from pathlib import Path
    before = Path(frame.image_path).read_bytes()
    with pytest.raises(TypeError):
        plan_cursor_parking(frame, step_id="parking", expected_marker="LABEL_NOT_INPUT")
    plan_cursor_parking(frame, step_id="parking")
    assert Path(frame.image_path).read_bytes() == before


@pytest.mark.parametrize("change", ["handle", "title", "class", "bounds", "screen", "stale"])
def test_postmove_capture_requires_fresh_unchanged_foreground_and_screen(tmp_path, change):
    reference = scene(tmp_path)
    current = reference.model_copy(update={"observation_id": "parking-2",
                                           "captured_at": reference.captured_at + timedelta(seconds=1)})
    updates = {
        "handle": {"window_id": "different"}, "title": {"window_title": "Other receiver"},
        "class": {"window_class": "OtherClass"},
        "bounds": {"window_bounds": reference.window_bounds.model_copy(update={"left": 159})},
        "screen": {"screen_info": reference.screen_info.model_copy(update={"monitor_left": -800})},
        "stale": {"observation_id": reference.observation_id},
    }
    with pytest.raises(CursorParkingError):
        verify_parking_frame(reference, current.model_copy(update=updates[change]))
    assert verify_parking_frame(reference, current)["source_image_sha256"]


class Clock:
    value = 0.0

    def __call__(self):
        return self.value


def run_case(runner, task, options):
    return runner.run(task, options, confirm=lambda _: True, high_risk_confirm=lambda _: True)


def ledger(recorder):
    return json.loads((recorder.directory / "cursor_parking.json").read_text(encoding="utf-8"))


def test_counted_parking_before_strict_draft_uses_fresh_pre_and_post_frames(tmp_path):
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path)
    checked = []
    draft = runner.verifier.check_message_draft

    def read_draft(spec, frame):
        checked.append(frame.observation_id)
        assert [a.action_type for a, _ in chat.actions] == ["click", "type_text", "move"]
        return draft(spec, frame)

    runner.verifier.check_message_draft = read_draft
    result = run_case(runner, task, options)
    assert result.status == "succeeded", result.notes
    assert result.action_count == 4 and [a.action_type for a, _ in chat.actions] == [
        "click", "type_text", "move", "click",
    ]
    evidence = ledger(recorder)
    move = next(s for s in result.steps if s.step_id == "runtime_cursor_parking")
    assert move.action_result["success"] and move.after_observation_id != move.observation_id
    assert evidence["before"]["observation_id"] != evidence["typing_after_observation_id"]
    assert evidence["after"]["observation_id"] == move.after_observation_id
    assert evidence["outcome"] == "completed" and evidence["dispatch_attempted"]
    assert evidence["send_authorization"] is False and "key_up" in evidence["modifier_cleanup"]
    assert checked[0] != move.observation_id
    summary = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == recorder.read_steps()


def test_one_remaining_send_budget_skips_parking_and_keeps_existing_strict_path(tmp_path):
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path)
    options.max_actions = 3
    result = run_case(runner, task, options)
    assert result.status == "succeeded" and result.action_count == 3
    assert [a.action_type for a, _ in chat.actions] == ["click", "type_text", "click"]
    assert ledger(recorder)["code"] == "reserve_one_send_action"
    assert not ledger(recorder)["dispatch_attempted"]


def test_no_exterior_skip_does_not_become_send_authorization_or_new_blocked_gate(tmp_path, monkeypatch):
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path)
    original = runner_module.plan_cursor_parking

    def no_space(frame, **kwargs):
        decision = original(frame, **kwargs)
        return CursorParkingDecision("skipped", "no_safe_window_exterior", None, decision.evidence)

    monkeypatch.setattr(runner_module, "plan_cursor_parking", no_space)
    result = run_case(runner, task, options)
    assert result.status == "succeeded" and result.action_count == 3
    assert not any(a.action_type == "move" for a, _ in chat.actions)
    assert ledger(recorder)["outcome"] == "skipped"


@pytest.mark.parametrize("fault", ["failure", "unknown", "exception", "dry_result", "interrupt"])
def test_failed_or_unknown_move_is_counted_once_blocks_send_and_never_retries(tmp_path, fault):
    runner, task, options, chat, client, recorder = _simulated_t04(tmp_path)
    options.max_planning_attempts = 3
    original = chat.execute

    def execute(action, **kwargs):
        if action.action_type != "move":
            return original(action, **kwargs)
        original(action, **kwargs)
        if fault == "unknown":
            return None
        if fault == "exception":
            raise OSError("backend result unknown")
        if fault == "interrupt":
            raise KeyboardInterrupt
        return ActionResult(action=action, success=fault != "failure", dry_run=fault == "dry_result", error="move fault")

    chat.execute = execute
    result = run_case(runner, task, options)
    assert result.status == ("cancelled" if fault == "interrupt" else "failed")
    assert result.action_count == 3 and not chat.sent
    assert [a.action_type for a, _ in chat.actions].count("move") == 1
    assert len(client.planning_contexts) == 1 and runner._message_send_blocked
    evidence = ledger(recorder)
    assert evidence["dispatch_attempted"] and evidence["outcome"] != "completed"
    frame = next(reversed(chat.frames.values()))[0]
    runner._park_message_cursor(frame, frame, options, runner._timings.started, result.steps)
    assert [a.action_type for a, _ in chat.actions].count("move") == 1


@pytest.mark.parametrize("fault", ["handle", "title", "bounds", "screen", "capture_error", "stale", "degraded", "header"])
def test_changed_postmove_evidence_cannot_send_or_replan(tmp_path, fault):
    runner, task, options, chat, client, recorder = _simulated_t04(tmp_path)
    options.max_planning_attempts = 3
    original = chat.observe
    changed = False

    def observe(**kwargs):
        nonlocal changed
        moved = any(a.action_type == "move" for a, _ in chat.actions)
        if moved and not changed and fault == "capture_error":
            changed = True
            raise OSError("postmove capture unavailable")
        frame = original(**kwargs)
        if moved and not changed:
            changed = True
            updates = {
                "handle": {"window_id": "other-handle"}, "title": {"window_title": "other"},
                "bounds": {"window_bounds": frame.window_bounds.model_copy(update={"left": frame.window_bounds.left - 1})},
                "screen": {"screen_info": frame.screen_info.model_copy(update={"monitor_left": -800})},
                "stale": {"observation_id": next(s for s in recorder.read_steps() if s["action_type"] == "type_text")["after_observation_id"]},
                "degraded": {"errors": ["degraded capture"]}, "header": {},
            }
            if fault == "header":
                with Image.open(frame.image_path) as pixels:
                    pixels = pixels.copy()
                header = chat.frames[frame.observation_id][1]["header_box"]
                pixels.putpixel((header[0] + 1, header[1] + 1), (0, 0, 0))
                pixels.save(frame.image_path)
            frame = frame.model_copy(update=updates[fault])
        return frame

    chat.observe = observe
    result = run_case(runner, task, options)
    assert result.status == "failed" and result.action_count == 3 and not chat.sent
    assert len(client.planning_contexts) == 1 and ledger(recorder)["outcome"] == "failed"


@pytest.mark.parametrize("stage", ["ready_io", "dispatch", "post_capture", "final_io"])
def test_parking_budget_includes_dispatch_capture_and_last_evidence_save(tmp_path, stage):
    clock = Clock()
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path, clock=clock)
    original_execute, original_observe = chat.execute, chat.observe
    original_save = recorder.session.save_json
    save_calls = 0

    def execute(action, **kwargs):
        result = original_execute(action, **kwargs)
        if action.action_type == "move" and stage == "dispatch":
            clock.value += 100
        return result

    def observe(**kwargs):
        result = original_observe(**kwargs)
        if stage == "post_capture" and any(a.action_type == "move" for a, _ in chat.actions):
            clock.value += 100
        return result

    def save(name, payload):
        nonlocal save_calls
        if name == "cursor_parking.json":
            save_calls += 1
            if stage == "ready_io" and payload["outcome"] == "ready_to_dispatch":
                clock.value += 100
            # ready, completed, then finally: target only the final persist.
            if stage == "final_io" and save_calls == 3:
                clock.value += 100
        return original_save(name, payload)

    chat.execute, chat.observe, recorder.session.save_json = execute, observe, save
    result = run_case(runner, task, options)
    assert result.status == "timed_out" and not chat.sent
    assert result.action_count == (2 if stage == "ready_io" else 3)
    assert ledger(recorder)["outcome"] == "timed_out"
    assert result.elapsed_ms >= 100_000
    if stage == "final_io":
        move = next(s for s in result.steps if s.action_type == "move")
        assert move.error and move.verification.outcome == "failed"
        assert move.elapsed_ms >= 100_000 and result.failed_step_id == move.step_id
        summary = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))
        assert summary["steps"] == recorder.read_steps()


def test_slow_step_verification_gets_another_fresh_capture_before_move(tmp_path):
    clock = Clock()
    runner, task, options, _chat, _, recorder = _simulated_t04(tmp_path, clock=clock)
    original_check = runner.verifier.check_step
    observed = []

    def check(**kwargs):
        if kwargs["action_type"] == "type_text":
            observed.append(kwargs["after"].observation_id)
            clock.value += 5
        return original_check(**kwargs)

    runner.verifier.check_step = check
    assert run_case(runner, task, options).status == "succeeded"
    assert ledger(recorder)["before"]["observation_id"] != observed[0]


def test_parking_completion_does_not_override_independent_draft_refusal(tmp_path):
    runner, task, options, chat, client, recorder = _simulated_t04(tmp_path)
    runner.verifier.check_message_draft = lambda *_: VerificationResult(
        outcome="failed", detail="actual current draft/IME/Send state cannot authorise Send",
    )
    result = run_case(runner, task, options)
    assert result.status == "failed" and not chat.sent and result.action_count == 3
    assert ledger(recorder)["outcome"] == "completed" and len(client.planning_contexts) == 1


def test_reusing_post_type_capture_cannot_plan_or_dispatch_parking(tmp_path):
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path)
    original_check, original_observe = runner.verifier.check_step, chat.observe
    post_type = []

    def check(**kwargs):
        if kwargs["action_type"] == "type_text":
            post_type.append(kwargs["after"])
        return original_check(**kwargs)

    def observe(**kwargs):
        return post_type[0] if post_type else original_observe(**kwargs)

    runner.verifier.check_step, chat.observe = check, observe
    result = run_case(runner, task, options)
    assert result.status == "failed" and result.action_count == 2 and not chat.sent
    assert ledger(recorder)["outcome"] == "failed" and not ledger(recorder)["dispatch_attempted"]


def test_final_evidence_io_error_marks_move_failed_and_keeps_count(tmp_path):
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path)
    original = recorder.session.save_json
    calls = 0

    def save(name, payload):
        nonlocal calls
        if name == "cursor_parking.json":
            calls += 1
            if calls == 3:
                raise OSError("final parking evidence unavailable")
        return original(name, payload)

    recorder.session.save_json = save
    result = run_case(runner, task, options)
    assert result.status == "failed" and result.action_count == 3 and not chat.sent
    move = next(s for s in result.steps if s.action_type == "move")
    assert move.error and move.verification.outcome == "failed" and result.failed_step_id == move.step_id
    assert ledger(recorder)["outcome"] == "failed"
    summary = json.loads((recorder.directory / "task_summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == recorder.read_steps()


def test_step_log_io_budget_failure_is_separate_without_duplicate_move_count(tmp_path):
    clock = Clock()
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path, clock=clock)
    original = recorder.append_step

    def append(record):
        if record.step_id == "runtime_cursor_parking":
            clock.value += 100
        return original(record)

    recorder.append_step = append
    result = run_case(runner, task, options)
    assert result.status == "timed_out" and result.action_count == 3 and not chat.sent
    assert result.failed_step_id == "runtime_cursor_parking_evidence"
    assert sum(s.action_type == "move" for s in result.steps) == 1
    assert result.steps[-1].action_result is None
    assert ledger(recorder)["code"] == "parking_step_log_budget_exhausted"


@pytest.mark.parametrize("stage", ["ready", "post_capture"])
def test_interrupting_parking_evidence_or_capture_keeps_attempt_denominator(tmp_path, stage):
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path)
    original_save, original_observe = recorder.session.save_json, chat.observe
    interrupted = False

    def save(name, payload):
        nonlocal interrupted
        if (stage == "ready" and name == "cursor_parking.json"
                and payload["outcome"] == "ready_to_dispatch" and not interrupted):
            interrupted = True
            raise KeyboardInterrupt
        return original_save(name, payload)

    def observe(**kwargs):
        nonlocal interrupted
        if stage == "post_capture" and any(a.action_type == "move" for a, _ in chat.actions) and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_observe(**kwargs)

    recorder.session.save_json, chat.observe = save, observe
    result = run_case(runner, task, options)
    assert result.status == "cancelled" and not chat.sent
    assert result.action_count == (2 if stage == "ready" else 3)
    assert ledger(recorder)["outcome"] == "cancelled"


def test_operator_interrupt_takes_precedence_over_simultaneously_expired_budget(tmp_path):
    clock = Clock()
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path, clock=clock)
    original = chat.execute

    def execute(action, **kwargs):
        result = original(action, **kwargs)
        if action.action_type == "move":
            clock.value += 100
            raise KeyboardInterrupt
        return result

    chat.execute = execute
    result = run_case(runner, task, options)
    assert result.status == "cancelled" and result.action_count == 3 and not chat.sent
    assert ledger(recorder)["outcome"] == "cancelled"


@pytest.mark.parametrize("stage", ["before_move", "after_move"])
def test_os_foreground_recheck_blocks_move_or_send_without_replanning(tmp_path, stage):
    runner, task, options, chat, client, recorder = _simulated_t04(tmp_path)
    original_execute, original_save = chat.execute, recorder.session.save_json
    original_foreground = chat.foreground_matches
    foreground_changed = False

    def foreground_matches(snapshot):
        return original_foreground(snapshot) and not foreground_changed

    def execute(action, **kwargs):
        nonlocal foreground_changed
        result = original_execute(action, **kwargs)
        if stage == "after_move" and action.action_type == "move":
            foreground_changed = True
        return result

    def save(name, payload):
        nonlocal foreground_changed
        if stage == "before_move" and name == "cursor_parking.json" and payload["outcome"] == "ready_to_dispatch":
            foreground_changed = True
        return original_save(name, payload)

    chat.foreground_matches = foreground_matches
    chat.execute, recorder.session.save_json = execute, save
    result = run_case(runner, task, options)
    assert result.status == "failed" and not chat.sent and len(client.planning_contexts) == 1
    assert result.action_count == (2 if stage == "before_move" else 3)
    assert ledger(recorder)["outcome"] == "failed"


@pytest.mark.parametrize("fault", ["dry_run", "typing_failed"])
def test_dry_run_and_unsuccessful_typing_never_park(tmp_path, fault):
    runner, task, options, chat, _, recorder = _simulated_t04(tmp_path)
    if fault == "dry_run":
        options.execute = False
    else:
        original = chat.execute
        def execute(action, **kwargs):
            if action.action_type == "type_text":
                chat.actions.append((action, kwargs.get("dry_run")))
                return ActionResult(action=action, success=False, dry_run=False, error="typing failed")
            return original(action, **kwargs)
        chat.execute = execute
    run_case(runner, task, options)
    assert not (recorder.directory / "cursor_parking.json").exists()
    assert not any(a.action_type == "move" for a, _ in chat.actions)
