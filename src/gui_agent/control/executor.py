"""Execute :class:`DesktopAction` objects, dry-run by default.

Real control is opt-in. The executor validates coordinates before touching the
backend, releases modifier keys in ``finally`` so nothing stays stuck, converts
every failure into an :class:`ActionResult`, and never writes credentials to the
run metadata.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from ..config import ControlConfig
from ..schemas import ActionResult, DesktopAction, Point, ScreenInfo
from .actions import ControlBackend, DesktopActions, PyAutoGUIBackend
from .safety import SafetyError, redact_action, validate_action

STICKY_KEYS = ("shift", "ctrl", "control", "alt", "option", "cmd", "command", "win", "super")


class Notifier(Protocol):
    def __call__(self, message: str) -> None: ...


class ActionExecutor:
    """Runs one action at a time and always returns an ActionResult."""

    def __init__(
        self,
        config: ControlConfig | None = None,
        *,
        backend: ControlBackend | None = None,
        screen: ScreenInfo | None = None,
        notice: Notifier | None = None,
        countdown: Notifier | None = None,
    ) -> None:
        self.config = config or ControlConfig()
        self._backend = backend
        self._screen = screen
        self._notice = notice or (lambda message: None)
        self._countdown = countdown

    @property
    def screen(self) -> ScreenInfo | None:
        return self._screen

    def update_screen(self, screen: ScreenInfo) -> None:
        self._screen = screen

    def _require_backend(self) -> ControlBackend:
        if self._backend is None:
            self._backend = PyAutoGUIBackend(failsafe=self.config.pyautogui_failsafe)
        return self._backend

    def _release_modifiers(self, backend: ControlBackend) -> None:
        for key in STICKY_KEYS:
            try:
                backend.key_up(key)
            except Exception:  # noqa: BLE001, S112 - releasing a key that was never pressed
                continue

    def _dispatch(self, action: DesktopAction, actions: DesktopActions) -> None:
        kind = action.action_type
        if kind == "move":
            actions.move_to(
                Point(x=action.x, y=action.y), action.duration or self.config.move_duration_seconds
            )
        elif kind == "click":
            actions.click(Point(x=action.x, y=action.y))
        elif kind == "double_click":
            actions.double_click(Point(x=action.x, y=action.y))
        elif kind == "right_click":
            actions.right_click(Point(x=action.x, y=action.y))
        elif kind == "drag":
            assert action.start is not None and action.end is not None
            actions.drag_to(
                action.start,
                action.end,
                duration=action.duration or self.config.drag_duration_seconds,
            )
        elif kind == "scroll":
            point = None if action.x is None or action.y is None else Point(x=action.x, y=action.y)
            actions.scroll(action.scroll_amount or 0, point)
        elif kind == "type_text":
            actions.type_text(action.text or "")
        elif kind == "key_press":
            actions.press_key(action.key or "")
        elif kind == "hotkey":
            actions.hotkey(list(action.keys or []))
        elif kind == "wait":
            actions.wait(action.duration or self.config.action_delay_seconds)
        else:  # pragma: no cover - the schema already restricts the literal
            raise SafetyError(f"unsupported action_type: {kind}")

    def execute(
        self,
        action: DesktopAction,
        *,
        dry_run: bool | None = None,
        screen: ScreenInfo | None = None,
    ) -> ActionResult:
        """Validate and (unless dry-run) perform ``action``."""
        run_dry = self.config.dry_run if dry_run is None else dry_run
        active_screen = screen or self._screen
        started_at = datetime.now(UTC)

        if active_screen is None:
            return ActionResult(
                success=False,
                dry_run=run_dry,
                action=action,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                error="no ScreenInfo available; capture the screen before executing actions",
            )

        try:
            validate_action(action, active_screen)
        except SafetyError as exc:
            self._notice(f"refused: {exc}")
            return ActionResult(
                success=False,
                dry_run=run_dry,
                action=action,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                error=str(exc),
                metadata={"plan": redact_action(action)},
            )

        if run_dry:
            self._notice("dry run: no real mouse or keyboard event was sent")
            return ActionResult(
                success=True,
                dry_run=True,
                action=action,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                metadata={"plan": redact_action(action)},
            )

        backend = self._require_backend()
        if self._countdown is not None and self.config.countdown_seconds > 0:
            self._countdown(f"executing in {self.config.countdown_seconds:.0f}s")
            backend.sleep(self.config.countdown_seconds)

        error: str | None = None
        try:
            self._dispatch(action, DesktopActions(backend))
        except Exception as exc:  # noqa: BLE001 - FailSafe and backend failures both land here
            error = f"{type(exc).__name__}: {exc}"
            self._notice(f"action failed: {error}")
        finally:
            self._release_modifiers(backend)
            if self.config.action_delay_seconds > 0:
                backend.sleep(self.config.action_delay_seconds)

        return ActionResult(
            success=error is None,
            dry_run=False,
            action=action,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            error=error,
            metadata={"plan": redact_action(action)},
        )
