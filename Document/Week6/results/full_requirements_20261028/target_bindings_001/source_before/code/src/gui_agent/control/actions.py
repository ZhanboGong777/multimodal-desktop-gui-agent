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


def ascii_input_mode() -> bool:
    """Put the foreground window's input method into ASCII mode, if it has one.

    Returns whether a conversion mode was found and set. Safe to call on a platform or a
    window without an IME: it reports False and changes nothing.

    Why this exists, measured on the Windows review machine: with Microsoft Pinyin active,
    typing the Latin string "GUI agent research" produced **"GUI阿根廷research"** in the
    field - the IME converted `agent` to Chinese - and the search that followed used that
    text, so the case's success rule could never match. The same mechanism explains every
    odd string in this project's records ('researchGUI阿根廷researchresearchGUlresearch',
    'guivant', '阿根廷文'), all of which had been read as model or OCR faults.

    **It is not wired into `type_text`, and that is deliberate.** Measured on the two
    applications that matter here: Chrome returns 0 from `ImmGetContext` for its window -
    it uses TSF and composes input itself - so this call cannot reach it, while calling it
    before typing on that same window produced no text at all. An IME that ignores the call
    is the common case for browsers, so the honest state is a helper that works for classic
    Win32 windows and a documented limit for the rest. What works for a browser is to leave
    the machine's input method switched to English before the run, which is an operator
    step, not something this module can assert.
    """
    import sys

    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    try:
        imm32 = ctypes.windll.imm32  # type: ignore[attr-defined]
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    except AttributeError:
        return False
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False
    himc = imm32.ImmGetContext(wintypes.HWND(hwnd))
    if not himc:
        return False
    try:
        # IME_CMODE_ALPHANUMERIC (0) clears the native/conversion bits, so keystrokes arrive
        # as the characters that were sent.
        return bool(imm32.ImmSetConversionStatus(himc, 0, 0))
    finally:
        imm32.ImmReleaseContext(wintypes.HWND(hwnd), himc)


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
        # Pressed by hand rather than through pyautogui's own `hotkey`, which silently does
        # nothing for the Alt combinations on Windows - and T05's whole task is `alt+f4`.
        #
        # Measured on the Windows review machine, same window, same document, three ways:
        #
        #   pyautogui.hotkey('alt', 'f4')                        window still open
        #   pyautogui.keyDown('alt'); press('f4'); keyUp('alt')  window closed
        #   native keybd_event(alt); keybd_event(f4)             window closed
        #
        # So the key names and the sequence are both fine and the fault is in the library's
        # composite call. The loop below is the middle form, which is the one that works and
        # the one this class can already express: this backend holds the modifier down,
        # presses the final key, and releases the modifiers in reverse.
        #
        # `press` on a modifier is also avoided - it would release the key before the next
        # one goes down, which is what a hotkey must not do.
        if not keys:
            return
        modifiers, final = list(keys[:-1]), keys[-1]
        for key in modifiers:
            self._gui.keyDown(key)
        try:
            self._gui.press(final)
        finally:
            for key in reversed(modifiers):
                self._gui.keyUp(key)

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
