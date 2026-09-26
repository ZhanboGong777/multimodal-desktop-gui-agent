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
    # How much of its bounding box a contour must fill to count as a rectangle.
    # Off by default: the right value depends heavily on the theme behind the
    # screen (0.55 removes almost nothing on a light UI and about 85% of the
    # candidates on a dark one), so it is a tuning knob, not a safe default.
    min_rectangularity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Keep the largest N candidates; the detector sorts by area first, so a lower
    # cap drops the small noisy boxes and keeps the meaningful ones. This is the
    # portable way to make the annotated image readable.
    max_candidates: int = Field(default=200, gt=0)
    # Drop candidates that overlap an OCR region by this fraction of the smaller box.
    exclusion_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


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


class DatasetConfig(ConfigModel):
    """Where a preparation run reads from and writes to."""

    name: Literal["screenagent", "mind2web", "webarena"] = "screenagent"
    split: str = "train"
    sample_limit: int = Field(default=20, ge=1)
    raw_directory: str = "data/raw"
    processed_directory: str = "data/processed"


class ModelConfig(ConfigModel):
    """Which multimodal backend to talk to, and how to reach it."""

    provider: Literal["mock", "openai_compatible"] = "mock"
    model_name: str = "mock-vision-model"
    # Credentials never live here: they are read from the environment.
    base_url: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_retries: int = Field(default=1, ge=0)


class PlanningConfig(ConfigModel):
    """Limits the planner enforces before a plan may leave the module."""

    max_steps: int = Field(default=10, ge=1)
    require_structured_output: bool = True
    allow_real_execution: bool = False


class Config(ConfigModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config.model_validate(raw)
