"""Shared data structures for perception, control and reporting.

These models are the single interface between the perception layer, the control
layer and the demo scripts. Modules must exchange these objects instead of
untyped dictionaries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionType = Literal[
    "move",
    "click",
    "double_click",
    "right_click",
    "drag",
    "scroll",
    "type_text",
    "key_press",
    "hotkey",
    "wait",
]
ElementSource = Literal["ocr", "contour", "manual"]

# Actions that are addressed by a single absolute screen coordinate.
POINT_ACTIONS = frozenset({"move", "click", "double_click", "right_click", "scroll"})


class SchemaModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Point(SchemaModel):
    """A single coordinate in either screenshot or control space."""

    x: int
    y: int


class BoundingBox(SchemaModel):
    """Axis-aligned box around one interface element."""

    left: int
    top: int
    right: int
    bottom: int

    @model_validator(mode="after")
    def _check_geometry(self) -> BoundingBox:
        if self.right <= self.left:
            raise ValueError(f"right ({self.right}) must be greater than left ({self.left})")
        if self.bottom <= self.top:
            raise ValueError(f"bottom ({self.bottom}) must be greater than top ({self.top})")
        return self

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> Point:
        return Point(x=(self.left + self.right) // 2, y=(self.top + self.bottom) // 2)


class ScreenInfo(SchemaModel):
    """Geometry needed to map screenshot pixels onto control coordinates."""

    screenshot_width: int = Field(gt=0)
    screenshot_height: int = Field(gt=0)
    control_width: int = Field(gt=0)
    control_height: int = Field(gt=0)
    monitor_index: int = Field(default=1, ge=0)
    monitor_left: int = 0
    monitor_top: int = 0
    scale_x: float | None = None
    scale_y: float | None = None

    @model_validator(mode="after")
    def _default_scales(self) -> ScreenInfo:
        if self.scale_x is None:
            self.scale_x = self.control_width / self.screenshot_width
        if self.scale_y is None:
            self.scale_y = self.control_height / self.screenshot_height
        return self


class UIElement(SchemaModel):
    """One interface element found by OCR or by contour detection."""

    text: str
    bounding_box: BoundingBox
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    center: Point | None = None
    source: ElementSource = "ocr"

    @model_validator(mode="after")
    def _default_center(self) -> UIElement:
        if self.center is None:
            self.center = self.bounding_box.center
        return self


class PerceptionResult(SchemaModel):
    """Everything produced by a single perception pass."""

    screen_info: ScreenInfo
    timestamp: datetime = Field(default_factory=datetime.now)
    image_path: str | None = None
    elements: list[UIElement] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class DesktopAction(SchemaModel):
    """A desktop action that is ready to be executed."""

    action_type: ActionType
    x: int | None = None
    y: int | None = None
    start: Point | None = None
    end: Point | None = None
    text: str | None = None
    key: str | None = None
    keys: list[str] | None = None
    scroll_amount: int | None = None
    duration: float | None = None
    target_description: str | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> DesktopAction:
        kind = self.action_type
        if kind in POINT_ACTIONS and (self.x is None or self.y is None):
            raise ValueError(f"{kind} requires both x and y")
        if kind == "drag" and (self.start is None or self.end is None):
            raise ValueError("drag requires both start and end (a single point is not enough)")
        if kind == "type_text" and not (self.text or "").strip():
            raise ValueError("type_text requires non-empty text")
        if kind == "key_press" and not self.key:
            raise ValueError("key_press requires a key")
        if kind == "hotkey" and not self.keys:
            raise ValueError("hotkey requires at least one key")
        return self

    @property
    def points(self) -> list[Point]:
        """Every screen coordinate this action touches, for safety checks."""
        found: list[Point] = []
        if self.x is not None and self.y is not None:
            found.append(Point(x=self.x, y=self.y))
        if self.start is not None:
            found.append(self.start)
        if self.end is not None:
            found.append(self.end)
        return found


class ActionResult(SchemaModel):
    """Outcome of one control attempt."""

    success: bool
    dry_run: bool
    action: DesktopAction | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    before_image_path: str | None = None
    after_image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
