"""Typed configuration loading for the GUI agent."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class ConfigModel(BaseModel):
    """Base config model that rejects unknown or misspelled keys."""

    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(ConfigModel):
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"


class CaptureConfig(ConfigModel):
    frame_count: int = Field(default=10, ge=1)
    interval_seconds: float = Field(default=0.2, ge=0.0)
    save_frames: bool = False


class PreprocessingConfig(ConfigModel):
    grayscale: bool = False
    resize_scale: float = Field(default=1.0, gt=0.0)
    gaussian_blur: bool = False
    otsu_threshold: bool = False


class OcrConfig(ConfigModel):
    engine: Literal["paddleocr", "tesseract"] = "tesseract"
    fallback_engine: Literal["paddleocr", "tesseract", "none"] = "tesseract"
    languages: list[str] = Field(default_factory=lambda: ["en"])
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # PaddleOCR's document pipeline (orientation, unwarping, textline angle) is
    # built for photographed paper. A screen capture is already flat and upright,
    # so it adds model downloads and can distort the image; keep it off unless a
    # task really needs it.
    use_doc_preprocessing: bool = False


class UiDetectionConfig(ConfigModel):
    enabled: bool = True
    min_area: int = Field(default=100, ge=0)
    max_area_ratio: float = Field(default=0.5, gt=0.0, le=1.0)


class PerceptionConfig(ConfigModel):
    screenshot_backend: Literal["mss"] = "mss"
    monitor_index: int = Field(default=1, ge=0)
    output_directory: Path = Path("outputs/week2")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    ui_detection: UiDetectionConfig = Field(default_factory=UiDetectionConfig)


class ControlConfig(ConfigModel):
    dry_run: bool = True
    pyautogui_failsafe: bool = True
    action_delay_seconds: float = Field(default=0.2, ge=0.0)
    countdown_seconds: float = Field(default=3.0, ge=0.0)
    move_duration_seconds: float = Field(default=0.3, ge=0.0)
    drag_duration_seconds: float = Field(default=0.5, ge=0.0)


class OutputConfig(ConfigModel):
    directory: Path = Path("outputs/week2")
    save_before_image: bool = True
    save_after_image: bool = True
    save_annotated_image: bool = True
    save_run_summary: bool = True


class LoggingConfig(ConfigModel):
    level: str = "INFO"
    directory: Path = Path("outputs/week2")


class Config(ConfigModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config.model_validate(raw)
