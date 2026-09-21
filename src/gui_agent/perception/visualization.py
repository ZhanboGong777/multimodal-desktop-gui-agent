"""Draw bounding boxes onto a copy of a screenshot.

This is graded Week 2 output, not throw-away debug code: annotated images are the
visual evidence that OCR text and contour candidates were located correctly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..schemas import UIElement

OCR_COLOR = (235, 64, 52)
CONTOUR_COLOR = (46, 134, 222)
LABEL_BACKGROUND = (0, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_PADDING = 2
DEFAULT_LINE_WIDTH = 2
DEFAULT_FONT_SIZE = 14

# Candidates in preference order; the first one that loads wins.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/arial.ttf",
)


def load_label_font(size: int = DEFAULT_FONT_SIZE) -> ImageFont.ImageFont:
    """Load a Unicode-capable font, falling back to Pillow's built-in bitmap font."""
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:  # noqa: BLE001, S112 - try the next candidate font
            continue
    return ImageFont.load_default()


def color_for(element: UIElement) -> tuple[int, int, int]:
    return OCR_COLOR if element.source == "ocr" else CONTOUR_COLOR


def label_for(
    element: UIElement,
    *,
    index: int | None = None,
    show_confidence: bool = False,
) -> str:
    """Build the label shown above a box; may be empty."""
    parts: list[str] = []
    if index is not None:
        parts.append(f"#{index}")
    if element.text:
        parts.append(element.text)
    if show_confidence:
        parts.append(f"{element.confidence:.2f}")
    return " ".join(parts)


def draw_bounding_boxes(
    image: Image.Image,
    elements: Iterable[UIElement],
    *,
    show_labels: bool = True,
    show_confidence: bool = False,
    show_index: bool = False,
    line_width: int = DEFAULT_LINE_WIDTH,
    font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    """Return a new annotated image; the input image is never modified."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    label_font = font or load_label_font()

    for position, element in enumerate(elements):
        box = element.bounding_box
        color = color_for(element)
        draw.rectangle((box.left, box.top, box.right, box.bottom), outline=color, width=line_width)
        if not show_labels:
            continue
        text = label_for(
            element,
            index=position if show_index else None,
            show_confidence=show_confidence,
        )
        if not text:
            continue
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        text_width, text_height = right - left, bottom - top
        label_top = max(0, box.top - text_height - 2 * LABEL_PADDING)
        draw.rectangle(
            (
                box.left,
                label_top,
                box.left + text_width + 2 * LABEL_PADDING,
                label_top + text_height + 2 * LABEL_PADDING,
            ),
            fill=LABEL_BACKGROUND,
        )
        draw.text(
            (box.left + LABEL_PADDING, label_top + LABEL_PADDING),
            text,
            fill=LABEL_TEXT_COLOR,
            font=label_font,
        )

    return canvas


def save_annotated_image(image: Image.Image, path: str | Path) -> Path:
    """Write an annotated image to disk, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


def summarize_sources(elements: Sequence[UIElement]) -> dict[str, int]:
    """Count elements per source, for logs and run summaries."""
    summary: dict[str, int] = {}
    for element in elements:
        summary[element.source] = summary.get(element.source, 0) + 1
    return summary
