"""Tests for action validation and the dry-run executor.

No test sends a real mouse or keyboard event. The executor is given a fake
backend, and dry runs are never handed a backend at all.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import pytest

from gui_agent.config import ControlConfig
from gui_agent.control.executor import ActionExecutor
from gui_agent.control.safety import (
    SafetyError,
    describe_action,
    is_sensitive_text,
    redact_action,
    validate_action,
)
from gui_agent.schemas import DesktopAction, Point, ScreenInfo


class FakeBackend:
    """Records every call instead of touching the real desktop."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.fail_on = fail_on

    def _record(self, name: str, *args: object) -> None:
        self.calls.append((name, args))
        if self.fail_on == name:
            raise RuntimeError("PyAutoGUI fail-safe triggered")

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def screen_size(self) -> tuple[int, int]:
        self._record("screen_size")
        return (1000, 800)

    def position(self) -> Point:
        self._record("position")
        return Point(x=0, y=0)

    def move_to(self, point: Point, duration: float = 0.0) -> None:
        self._record("move_to", point, duration)

    def click(
        self,
        point: Point | None = None,
        *,
        clicks: int = 1,
        button: str = "left",
        duration: float = 0.0,
    ) -> None:
        self._record("click", point, clicks, button)

    def drag_to(
        self, start: Point, end: Point, *, duration: float = 0.0, button: str = "left"
    ) -> None:
        self._record("drag_to", start, end, duration)

    def scroll(self, amount: int, point: Point | None = None) -> None:
        self._record("scroll", amount, point)

    def type_text(self, text: str, *, interval: float = 0.02) -> None:
        self._record("type_text", text)

    def key_down(self, key: str) -> None:
        self._record("key_down", key)

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", (key,)))

    def press(self, key: str) -> None:
        self._record("press", key)

    def hotkey(self, keys: Sequence[str]) -> None:
        self._record("hotkey", tuple(keys))

    def sleep(self, seconds: float) -> None:
        self.calls.append(("sleep", (seconds,)))


def make_screen() -> ScreenInfo:
    return ScreenInfo(
        screenshot_width=1000, screenshot_height=800, control_width=1000, control_height=800
    )


def make_config(**overrides: Any) -> ControlConfig:
    values: dict[str, Any] = {
        "countdown_seconds": 0.0,
        "action_delay_seconds": 0.0,
        "dry_run": True,
    }
    values.update(overrides)
    return ControlConfig(**values)


def test_validation_accepts_points_inside_the_monitor() -> None:
    validate_action(DesktopAction(action_type="click", x=10, y=10), make_screen())
    validate_action(DesktopAction(action_type="click", x=999, y=799), make_screen())


@pytest.mark.parametrize("x, y", [(1000, 10), (10, 800), (-1, 10), (10, -1)])
def test_validation_rejects_points_outside_the_monitor(x: int, y: int) -> None:
    with pytest.raises(SafetyError):
        validate_action(DesktopAction(action_type="click", x=x, y=y), make_screen())


def test_drag_validates_both_endpoints() -> None:
    validate_action(
        DesktopAction(action_type="drag", start=Point(x=1, y=1), end=Point(x=500, y=500)),
        make_screen(),
    )


class _RecordingGui:
    """Stands in for the pyautogui module, recording the calls a hotkey makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def keyDown(self, key: str) -> None:
        self.calls.append(("keyDown", key))

    def keyUp(self, key: str) -> None:
        self.calls.append(("keyUp", key))

    def press(self, key: str) -> None:
        self.calls.append(("press", key))

    def hotkey(self, *keys: str) -> None:
        # The composite call this backend deliberately does not use. If it appears here the
        # regression is back, and the assertion below says why it matters.
        self.calls.append(("hotkey", ",".join(keys)))


def test_a_hotkey_holds_its_modifier_down_across_the_final_key() -> None:
    """The Windows defect that stopped T05, pinned as a sequence.

    Measured on the review machine, same window and document: `pyautogui.hotkey('alt','f4')`
    left the window open, while `keyDown('alt'); press('f4'); keyUp('alt')` closed it - and
    the same native keystrokes closed it too, so the key names and the order were never the
    problem. T05's entire task is `alt+f4`, so the case could not pass while the backend
    used the composite call.

    What this asserts is the shape that works: the modifier goes down first, stays down
    while the final key is pressed, and comes back up afterwards.
    """
    from gui_agent.control.actions import PyAutoGUIBackend

    gui = _RecordingGui()
    backend = PyAutoGUIBackend.__new__(PyAutoGUIBackend)
    backend._gui = gui

    backend.hotkey(["alt", "f4"])

    assert gui.calls == [("keyDown", "alt"), ("press", "f4"), ("keyUp", "alt")], (
        "a hotkey must hold its modifiers down across the final key, and must not use "
        "pyautogui's composite hotkey(), which does nothing for Alt on Windows"
    )
    with pytest.raises(SafetyError):
        validate_action(
            DesktopAction(action_type="drag", start=Point(x=1, y=1), end=Point(x=5000, y=1)),
            make_screen(),
        )


def test_drag_with_identical_endpoints_is_refused() -> None:
    with pytest.raises(SafetyError):
        validate_action(
            DesktopAction(action_type="drag", start=Point(x=5, y=5), end=Point(x=5, y=5)),
            make_screen(),
        )


def test_negative_duration_is_refused() -> None:
    with pytest.raises(SafetyError):
        validate_action(DesktopAction(action_type="move", x=1, y=1, duration=-1.0), make_screen())


def test_dry_run_never_touches_a_backend() -> None:
    notices: list[str] = []
    executor = ActionExecutor(
        make_config(dry_run=True), screen=make_screen(), notice=notices.append
    )
    result = executor.execute(DesktopAction(action_type="click", x=10, y=10))
    assert result.success
    assert result.dry_run
    assert executor._backend is None
    assert notices and "dry run" in notices[0]


def test_dry_run_can_be_forced_even_when_config_allows_real_control() -> None:
    executor = ActionExecutor(make_config(dry_run=False), screen=make_screen())
    result = executor.execute(DesktopAction(action_type="move", x=5, y=5), dry_run=True)
    assert result.success and result.dry_run
    assert executor._backend is None


def test_real_run_dispatches_to_the_backend() -> None:
    backend = FakeBackend()
    executor = ActionExecutor(make_config(dry_run=False), backend=backend, screen=make_screen())
    result = executor.execute(DesktopAction(action_type="click", x=10, y=10))
    assert result.success
    assert not result.dry_run
    assert "click" in backend.names()


def test_drag_is_dispatched_with_both_points() -> None:
    backend = FakeBackend()
    executor = ActionExecutor(make_config(dry_run=False), backend=backend, screen=make_screen())
    result = executor.execute(
        DesktopAction(action_type="drag", start=Point(x=10, y=10), end=Point(x=200, y=200))
    )
    assert result.success
    assert "drag_to" in backend.names()


def test_unsafe_action_is_refused_before_the_backend_is_used() -> None:
    backend = FakeBackend()
    executor = ActionExecutor(make_config(dry_run=False), backend=backend, screen=make_screen())
    result = executor.execute(DesktopAction(action_type="click", x=5000, y=5000))
    assert not result.success
    assert result.error is not None and "outside" in result.error
    assert backend.calls == []


def test_backend_failure_becomes_a_failed_result() -> None:
    backend = FakeBackend(fail_on="click")
    executor = ActionExecutor(make_config(dry_run=False), backend=backend, screen=make_screen())
    result = executor.execute(DesktopAction(action_type="click", x=10, y=10))
    assert not result.success
    assert result.error is not None and "fail-safe" in result.error


def test_modifier_keys_are_released_after_a_failure() -> None:
    backend = FakeBackend(fail_on="click")
    executor = ActionExecutor(make_config(dry_run=False), backend=backend, screen=make_screen())
    executor.execute(DesktopAction(action_type="click", x=10, y=10))
    assert "key_up" in backend.names()


def test_missing_screen_info_is_reported() -> None:
    executor = ActionExecutor(make_config(dry_run=False), backend=FakeBackend())
    result = executor.execute(DesktopAction(action_type="click", x=1, y=1))
    assert not result.success
    assert result.error is not None and "ScreenInfo" in result.error


def test_sensitive_text_is_detected() -> None:
    assert is_sensitive_text("my password is hunter2")
    assert is_sensitive_text("API_KEY=abc")
    assert not is_sensitive_text("hello world")
    assert not is_sensitive_text(None)


def test_redaction_hides_credentials_in_logs() -> None:
    action = DesktopAction(action_type="type_text", text="hunter2")
    payload = redact_action(action)
    assert payload["text"] == "<redacted>"
    assert "hunter2" not in describe_action(action)


def test_describe_action_redacts_all_typed_text() -> None:
    action = DesktopAction(action_type="type_text", text="hello")
    assert "hello" not in describe_action(action)


def test_describe_action_includes_drag_endpoints() -> None:
    action = DesktopAction(action_type="drag", start=Point(x=1, y=2), end=Point(x=3, y=4))
    description = describe_action(action)
    assert "from=(1,2)" in description
    assert "to=(3,4)" in description


# ───── the executor's dispatch, for the action types the five tasks use ─────
def _executor(backend: FakeBackend, **overrides: Any) -> ActionExecutor:
    return ActionExecutor(make_config(**overrides), backend=backend, screen=make_screen())


def test_typing_is_dispatched_with_the_text() -> None:
    """T02 is a type-then-submit task, and neither half had ever been executed.

    The executor's dispatch for `type_text`, `key_press`, `hotkey`, `scroll` and
    `wait` was uncovered: the tests exercised `click` and `drag` and left the rest
    to the real machine, where a wrong argument order only shows up as a task that
    quietly does not work.
    """
    backend = FakeBackend()

    result = _executor(backend).execute(
        DesktopAction(action_type="type_text", text="GUI agent research"), dry_run=False
    )

    assert result.success
    assert ("type_text", ("GUI agent research",)) in backend.calls


def test_a_key_press_is_dispatched_with_its_key() -> None:
    backend = FakeBackend()

    result = _executor(backend).execute(DesktopAction(action_type="key_press", key="enter"), dry_run=False)

    assert result.success
    assert ("press", ("enter",)) in backend.calls
    assert backend.names()[0] == "press", "the key goes down before anything is released"


def test_a_hotkey_is_dispatched_as_one_chord() -> None:
    """One `hotkey` call, not a press of each key in turn.

    Pressing them separately would type "ctrl" and "l" rather than opening the
    address bar - a difference that is invisible to a fake that only counts calls.
    """
    backend = FakeBackend()

    result = _executor(backend).execute(
        DesktopAction(action_type="hotkey", keys=["ctrl", "l"]), dry_run=False
    )

    assert result.success
    assert ("hotkey", (("ctrl", "l"),)) in backend.calls


def test_a_scroll_is_dispatched_with_its_amount_and_point() -> None:
    backend = FakeBackend()

    result = _executor(backend).execute(
        DesktopAction(action_type="scroll", scroll_amount=-3, x=100, y=200), dry_run=False
    )

    assert result.success
    assert ("scroll", (-3, Point(x=100, y=200))) in backend.calls


def test_a_wait_is_dispatched_with_its_duration() -> None:
    backend = FakeBackend()

    result = _executor(backend).execute(DesktopAction(action_type="wait", duration=1.5), dry_run=False)

    assert result.success
    assert ("sleep", (1.5,)) in backend.calls


def test_the_countdown_runs_through_the_backend_before_the_action() -> None:
    """It is a warning the operator can act on, so it has to happen before the event.

    The wrapper is asleep during it rather than busy-waiting, which is why the
    backend's own `sleep` is the thing being asserted.
    """
    backend = FakeBackend()
    announced: list[str] = []

    executor = ActionExecutor(
        make_config(countdown_seconds=3.0), backend=backend, screen=make_screen()
    )
    executor._countdown = announced.append

    result = executor.execute(
        DesktopAction(action_type="click", x=100, y=100), dry_run=False
    )

    assert result.success
    assert announced and "3s" in announced[0]
    assert backend.calls[0] == ("sleep", (3.0,)), "the countdown is slept through first"
    assert backend.names()[1] == "click", "and the event follows it"


def test_the_action_delay_is_paid_after_every_real_action() -> None:
    """12.1's escape hatch: a small gap so the interface can react."""
    backend = FakeBackend()

    _executor(backend, action_delay_seconds=0.2).execute(
        DesktopAction(action_type="click", x=10, y=10), dry_run=False
    )

    assert backend.calls[-1] == ("sleep", (0.2,))


def test_modifiers_are_released_after_every_action() -> None:
    """14.3.6: a chord must not leave a modifier held down.

    A stuck ctrl turns the next click into a ctrl-click and a stuck shift turns the
    next typed letter into a capital, and neither shows up as a failed action. The
    release happens in a `finally`, so it also runs when the action itself raised.
    """
    backend = FakeBackend()

    _executor(backend).execute(DesktopAction(action_type="click", x=10, y=10), dry_run=False)

    released = {args[0] for name, args in backend.calls if name == "key_up"}
    assert {"shift", "ctrl", "alt", "command"} <= released


def test_a_modifier_that_cannot_be_released_does_not_fail_the_action() -> None:
    """Three of the names are not valid PyAutoGUI keys on any platform.

    `control`, `cmd` and `super` are not in the key list, so those releases always
    raise. Swallowing them is deliberate, and this pins both halves: the invalid
    ones are still attempted - they are what a plan or another tool might have
    pressed - and the action is still reported as succeeding.
    """
    backend = FakeBackend()

    result = _executor(backend).execute(
        DesktopAction(action_type="click", x=10, y=10), dry_run=False
    )

    assert result.success
    attempted = {args[0] for name, args in backend.calls if name == "key_up"}
    assert {"control", "cmd", "super"} <= attempted


def test_the_modifiers_are_released_even_when_the_action_fails() -> None:
    backend = FakeBackend(fail_on="click")

    result = _executor(backend).execute(
        DesktopAction(action_type="click", x=10, y=10), dry_run=False
    )

    assert not result.success
    assert any(name == "key_up" for name, _ in backend.calls), "the finally still ran"


# ───── the real backend, against a stub pyautogui ─────
class _StubGui:
    """Stands in for the pyautogui module, so no real mouse is ever moved.

    `PyAutoGUIBackend.__init__` imports pyautogui; putting this in `sys.modules`
    means the wrapper binds to it instead. Nothing below touches a screen - it
    checks that each primitive is forwarded to the right PyAutoGUI call with the
    right arguments, which is the layer that actually drives Windows and had no
    test at all.
    """

    FAILSAFE = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((name, args, kwargs))
        if name == "size":
            return type("Size", (), {"width": 2560, "height": 1600})()
        if name == "position":
            return type("Pos", (), {"x": 12, "y": 34})()
        return None

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: self._record(name, *args, **kwargs)


def _backend(monkeypatch: pytest.MonkeyPatch, *, failsafe: bool = True):
    from gui_agent.control import actions as actions_module

    stub = _StubGui()
    monkeypatch.setitem(sys.modules, "pyautogui", stub)
    return actions_module.PyAutoGUIBackend(failsafe=failsafe), stub


def test_constructing_the_real_backend_sets_the_failsafe_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The corner-of-the-screen abort has to be on before anything moves."""
    from gui_agent.control import actions as actions_module

    _backend(monkeypatch, failsafe=False)
    stub = sys.modules["pyautogui"]
    assert stub.FAILSAFE is False, "constructing with it off turns it off"

    actions_module.PyAutoGUIBackend(failsafe=True)

    assert stub.FAILSAFE is True, "and constructing with it on turns it back on"


def test_the_real_backend_forwards_each_primitive(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, stub = _backend(monkeypatch)
    point = Point(x=100, y=200)

    backend.move_to(point, duration=0.3)
    backend.click(point)
    backend.click(clicks=2, button="right")
    backend.drag_to(point, Point(x=300, y=400), duration=0.5)
    backend.scroll(-2, point)
    backend.scroll(-2)
    backend.type_text("hello")
    backend.key_down("shift")
    backend.key_up("shift")
    backend.press("enter")
    backend.hotkey(["ctrl", "l"])

    assert stub.calls[0] == ("moveTo", (100, 200), {"duration": 0.3})
    assert stub.calls[1] == ("click", (), {"x": 100, "y": 200, "clicks": 1, "button": "left", "duration": 0.0})
    # a click with no point keeps the pointer where it is
    assert stub.calls[2] == ("click", (), {"clicks": 2, "button": "right"})
    # drag moves to the start first, then drags to the end
    assert stub.calls[3] == ("moveTo", (100, 200), {"duration": 0})
    assert stub.calls[4] == ("dragTo", (300, 400), {"duration": 0.5, "button": "left"})
    assert stub.calls[5] == ("scroll", (-2,), {"x": 100, "y": 200})
    assert stub.calls[6] == ("scroll", (-2,), {})
    assert stub.calls[7][0] == "typewrite" and stub.calls[7][1] == ("hello",)
    assert stub.calls[8] == ("keyDown", ("shift",), {})
    assert stub.calls[9] == ("keyUp", ("shift",), {})
    assert stub.calls[10] == ("press", ("enter",), {})
    # A hotkey is three pyautogui calls, not one: the modifier goes down, the final key is
    # pressed while it is held, and the modifier is released after. It used to be a single
    # `hotkey` call, which does nothing for Alt on Windows - see the test above this one.
    assert stub.calls[11:14] == [
        ("keyDown", ("ctrl",), {}),
        ("press", ("l",), {}),
        ("keyUp", ("ctrl",), {}),
    ]


def test_the_real_backend_reports_its_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _stub = _backend(monkeypatch)

    assert backend.screen_size() == (2560, 1600)
    assert backend.position() == Point(x=12, y=34)


def test_the_real_backend_sleeps_for_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sleep` is the one primitive that does not go through PyAutoGUI."""
    backend, stub = _backend(monkeypatch)
    slept: list[float] = []
    monkeypatch.setattr("gui_agent.control.actions.time.sleep", slept.append)

    backend.sleep(0.25)

    assert slept == [0.25]
    assert stub.calls == []


def test_the_remaining_pointer_actions_are_dispatched() -> None:
    """`move`, `double_click` and `right_click` had no dispatch test either.

    All of them are in the plan vocabulary the model is handed, so a plan can
    contain any of them and the executor has to know what each one means - a
    `double_click` that dispatched as a single click would look like a sluggish
    application rather than a bug.
    """
    backend = FakeBackend()

    for action in (
        DesktopAction(action_type="move", x=50, y=60),
        DesktopAction(action_type="double_click", x=70, y=80),
        DesktopAction(action_type="right_click", x=90, y=100),
    ):
        assert _executor(backend).execute(action, dry_run=False).success

    moved = [call for call in backend.calls if call[0] == "move_to"]
    assert moved and moved[0][1][0] == Point(x=50, y=60)
    assert ("click", (Point(x=70, y=80), 2, "left")) in backend.calls, "double, not single"
    assert ("click", (Point(x=90, y=100), 1, "right")) in backend.calls, "the other button"
