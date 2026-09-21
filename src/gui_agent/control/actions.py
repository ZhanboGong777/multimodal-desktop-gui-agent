"""Thin wrappers around PyAutoGUI primitives.

Every direct PyAutoGUI call lives here so the executor can be tested with a fake
backend that never moves the real mouse.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from ..schemas import Point

DEFAULT_TYPE_INTERVAL = 0.02


class ControlBackend(Protocol):
    """The surface the executor relies on."""

    def screen_size(self) -> tuple[int, int]: ...
    def position(self) -> Point: ...
    def move_to(self, point: Point, duration: float = 0.0) -> None: ...
    def click(
        self,
        point: Point | None = None,
        *,
        clicks: int = 1,
        button: str = "left",
        duration: float = 0.0,
    ) -> None: ...
    def drag_to(
        self, start: Point, end: Point, *, duration: float = 0.0, button: str = "left"
    ) -> None: ...
    def scroll(self, amount: int, point: Point | None = None) -> None: ...
    def type_text(self, text: str, *, interval: float = DEFAULT_TYPE_INTERVAL) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def press(self, key: str) -> None: ...
    def hotkey(self, keys: Sequence[str]) -> None: ...
    def sleep(self, seconds: float) -> None: ...


class PyAutoGUIBackend:
    """Real backend. Constructing it enables PyAutoGUI's fail-safe corner."""

    def __init__(self, failsafe: bool = True) -> None:
        import pyautogui

        self._gui = pyautogui
        pyautogui.FAILSAFE = failsafe

    def screen_size(self) -> tuple[int, int]:
        size = self._gui.size()
        return int(size.width), int(size.height)

    def position(self) -> Point:
        current = self._gui.position()
        return Point(x=int(current.x), y=int(current.y))

    def move_to(self, point: Point, duration: float = 0.0) -> None:
        self._gui.moveTo(point.x, point.y, duration=duration)

    def click(
        self,
        point: Point | None = None,
        *,
        clicks: int = 1,
        button: str = "left",
        duration: float = 0.0,
    ) -> None:
        if point is None:
            self._gui.click(clicks=clicks, button=button)
        else:
            self._gui.click(x=point.x, y=point.y, clicks=clicks, button=button, duration=duration)

    def drag_to(
        self,
        start: Point,
        end: Point,
        *,
        duration: float = 0.0,
        button: str = "left",
    ) -> None:
        self._gui.moveTo(start.x, start.y, duration=0)
        self._gui.dragTo(end.x, end.y, duration=duration, button=button)

    def scroll(self, amount: int, point: Point | None = None) -> None:
        if point is None:
            self._gui.scroll(amount)
        else:
            self._gui.scroll(amount, x=point.x, y=point.y)

    def type_text(self, text: str, *, interval: float = DEFAULT_TYPE_INTERVAL) -> None:
        self._gui.typewrite(text, interval=interval)

    def key_down(self, key: str) -> None:
        self._gui.keyDown(key)

    def key_up(self, key: str) -> None:
        self._gui.keyUp(key)

    def press(self, key: str) -> None:
        self._gui.press(key)

    def hotkey(self, keys: Sequence[str]) -> None:
        self._gui.hotkey(*keys)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class DesktopActions:
    """The ten Week 2 primitives, expressed against a backend."""

    def __init__(self, backend: ControlBackend) -> None:
        self.backend = backend

    def move_to(self, point: Point, duration: float = 0.0) -> None:
        self.backend.move_to(point, duration)

    def click(
        self, point: Point, *, clicks: int = 1, button: str = "left", duration: float = 0.0
    ) -> None:
        self.backend.click(point, clicks=clicks, button=button, duration=duration)

    def double_click(self, point: Point, *, duration: float = 0.0) -> None:
        self.backend.click(point, clicks=2, button="left", duration=duration)

    def right_click(self, point: Point, *, duration: float = 0.0) -> None:
        self.backend.click(point, clicks=1, button="right", duration=duration)

    def drag_to(self, start: Point, end: Point, *, duration: float = 0.0) -> None:
        self.backend.drag_to(start, end, duration=duration)

    def scroll(self, amount: int, point: Point | None = None) -> None:
        self.backend.scroll(amount, point)

    def type_text(self, text: str, *, interval: float = DEFAULT_TYPE_INTERVAL) -> None:
        self.backend.type_text(text, interval=interval)

    def press_key(self, key: str) -> None:
        self.backend.press(key)

    def hotkey(self, keys: Sequence[str]) -> None:
        self.backend.hotkey(keys)

    def wait(self, seconds: float) -> None:
        self.backend.sleep(seconds)
