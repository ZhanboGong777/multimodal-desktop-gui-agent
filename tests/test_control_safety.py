"""Tests for action validation and the dry-run executor.

No test sends a real mouse or keyboard event. The executor is given a fake
backend, and dry runs are never handed a backend at all.
"""

from __future__ import annotations

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
    action = DesktopAction(action_type="type_text", text="password hunter2")
    payload = redact_action(action)
    assert payload["text"] == "<redacted>"
    assert "hunter2" not in describe_action(action)


def test_describe_action_keeps_ordinary_text() -> None:
    action = DesktopAction(action_type="type_text", text="hello")
    assert "hello" in describe_action(action)


def test_describe_action_includes_drag_endpoints() -> None:
    action = DesktopAction(action_type="drag", start=Point(x=1, y=2), end=Point(x=3, y=4))
    description = describe_action(action)
    assert "from=(1,2)" in description
    assert "to=(3,4)" in description
