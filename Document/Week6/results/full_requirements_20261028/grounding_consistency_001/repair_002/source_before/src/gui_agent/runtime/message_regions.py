"""Small, frame-backed image regions for visual message verification.

These are geometry candidates, not recognised controls. The visual assessor
must still confirm that the header is readable and the bottom region is an
editor. Crops and their layout transforms are evidence only: they never create
an action, a new element id, or a point that the executor may dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..schemas import BoundingBox
from .schemas import ElementRef, ObservationSnapshot

_MIN_CONFIDENCE = 0.35
_MAX_MESSAGES = 10
_HEADER_PADDING = 4
_LABEL_HEIGHT = 24
_MESSAGE_HALO = 32
_COMPOSER_BOTTOM_FRACTION = 0.85
# T04_20261009_083042_910048_75cebb04 exposed a titlebar X at 1.44%
# window height above the chat title at 7.11%-9.58%. Across 106 preceding
# usable real regions plus this frame the title occupied 6.90%-9.68%, beginning within
# 1.31% of the observed editor's left edge. These conservative layout bands
# isolate supported chat titles; they do not recognise recipient text.
_HEADER_TOP_FRACTION = 0.05
_HEADER_MAX_TOP_FRACTION = 0.12
_HEADER_BOTTOM_FRACTION = 0.18
_HEADER_ANCHOR_LEFT_FRACTION = 0.20
_HEADER_FRAGMENT_GAP_HEIGHTS = 2.0


class MessageRegionError(ValueError):
    """The recorded observation cannot provide unambiguous visual regions."""


@dataclass(frozen=True)
class MessageRegions:
    """Observed boxes from one frame, still subject to semantic assessment."""

    observation_id: str
    header_box: BoundingBox
    header_element_ids: tuple[str, ...]
    composer: ElementRef
    messages: tuple[ElementRef, ...]


@dataclass(frozen=True)
class RegionPanel:
    """A native-pixel crop and its position on the evidence contact sheet."""

    label: str
    source_box: BoundingBox
    panel_box: BoundingBox
    candidate_ids: tuple[str, ...]
    candidate_boxes: dict[str, BoundingBox]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_box": self.source_box.model_dump(),
            "panel_box": self.panel_box.model_dump(),
            "candidate_ids": list(self.candidate_ids),
            "candidate_boxes": {
                element_id: box.model_dump() for element_id, box in self.candidate_boxes.items()
            },
            "source_to_panel": {
                "scale_x": 1.0,
                "scale_y": 1.0,
                "translate_x": self.panel_box.left - self.source_box.left,
                "translate_y": self.panel_box.top - self.source_box.top,
            },
        }


@dataclass(frozen=True)
class RenderedMessageRegions:
    """Saved visual evidence, including every crop's reversible layout mapping."""

    observation_id: str
    image_path: Path
    image_size: tuple[int, int]
    panels: tuple[RegionPanel, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "image_path": str(self.image_path),
            "image_size": list(self.image_size),
            "native_pixels": True,
            "panels": [panel.as_dict() for panel in self.panels],
        }


def _inside(inner: BoundingBox, outer: BoundingBox) -> bool:
    return (
        outer.left <= inner.left < inner.right <= outer.right
        and outer.top <= inner.top < inner.bottom <= outer.bottom
    )


def _foreground(observation: ObservationSnapshot) -> BoundingBox:
    bounds = observation.window_bounds
    if (
        observation.errors
        or not observation.foreground_stable
        or not observation.window_id
        or not observation.window_title
        or not observation.window_class
        or bounds is None
    ):
        raise MessageRegionError("stable, readable foreground identity and bounds are required")
    screen = BoundingBox(
        left=0, top=0, right=observation.screen_info.screenshot_width,
        bottom=observation.screen_info.screenshot_height,
    )
    if not _inside(bounds, screen):
        raise MessageRegionError("foreground bounds extend outside the recorded screenshot")
    ids = [item.element_id for item in observation.elements]
    if len(ids) != len(set(ids)) or any(
        not element_id.startswith(f"{observation.observation_id}-e") for element_id in ids
    ):
        raise MessageRegionError("element ids must be unique and belong to this observation")
    return bounds


def _eligible(element: ElementRef, foreground: BoundingBox) -> bool:
    return (
        element.source in {"ocr", "contour"}
        and element.confidence >= _MIN_CONFIDENCE
        and element.center == element.bounding_box.center
        and _inside(element.bounding_box, foreground)
    )


def _same_row(first: BoundingBox, second: BoundingBox) -> bool:
    overlap = min(first.bottom, second.bottom) - max(first.top, second.top)
    return overlap >= min(first.height, second.height) * 0.5


def _duplicate_border(first: BoundingBox, second: BoundingBox) -> bool:
    """Two nearly coincident contour borders are one evidence crop, not two messages."""
    return all(
        abs(getattr(first, edge) - getattr(second, edge)) <= 2
        for edge in ("left", "top", "right", "bottom")
    )


def _header_row(
    elements: list[ElementRef], foreground: BoundingBox, body: BoundingBox,
) -> list[ElementRef]:
    """Require one left-anchored content-title row, separate from window chrome.

    The editor supplies the actual chat column. Only complete OCR boxes in a
    bounded content band can supply evidence; disconnected right-side toolbar
    glyphs are not appended to a title. Missing or competing left-anchored rows
    remain ambiguous, even when their OCR strings happen to be identical.
    """
    # Do not turn text inside an actual compact transcript-like border into a
    # title if title OCR is missing. Full-width pane/window borders and the
    # bottom editor are excluded from this conservative enclosure veto.
    enclosures = [
        item.bounding_box for item in elements
        if item.source == "contour" and not item.text.strip()
        and body.left <= item.bounding_box.left < item.bounding_box.right <= body.right
        and item.bounding_box.bottom < body.top
        and item.bounding_box.width <= body.width * 0.90
        and item.bounding_box.height <= foreground.height * 0.25
        and item.bounding_box.width * item.bounding_box.height
        <= body.width * (body.top - foreground.top) * 0.50
    ]
    candidates = [
        item for item in elements
        if item.source == "ocr" and item.text.strip()
        and body.left <= item.bounding_box.left < item.bounding_box.right <= body.right
        and item.bounding_box.top >= foreground.top + foreground.height * _HEADER_TOP_FRACTION
        and item.bounding_box.top <= foreground.top + foreground.height * _HEADER_MAX_TOP_FRACTION
        and item.bounding_box.bottom <= foreground.top + foreground.height * _HEADER_BOTTOM_FRACTION
        and not any(_inside(item.bounding_box, enclosure) for enclosure in enclosures)
    ]
    anchors = [
        item for item in candidates
        if item.bounding_box.left <= body.left + body.width * _HEADER_ANCHOR_LEFT_FRACTION
    ]
    if not anchors:
        raise MessageRegionError("no upper OCR row is available in the bounded chat content-title band")
    first = min(anchors, key=lambda item: (item.bounding_box.top, item.bounding_box.left))
    if any(not _same_row(first.bounding_box, item.bounding_box) for item in anchors):
        raise MessageRegionError("exactly one content-title OCR row is required; competing rows are ambiguous")
    row = sorted(
        (item for item in candidates if _same_row(first.bounding_box, item.bounding_box)),
        key=lambda item: (item.bounding_box.left, item.element_id),
    )
    segments: list[list[ElementRef]] = []
    for item in row:
        if segments:
            previous = segments[-1]
            right = max(member.bounding_box.right for member in previous)
            height = max(item.bounding_box.height, *(member.bounding_box.height for member in previous))
            if item.bounding_box.left - right <= height * _HEADER_FRAGMENT_GAP_HEIGHTS:
                previous.append(item)
                continue
        segments.append([item])
    anchor_ids = {item.element_id for item in anchors}
    anchored = [segment for segment in segments if any(item.element_id in anchor_ids for item in segment)]
    if len(anchored) != 1:
        raise MessageRegionError("exactly one contiguous content-title OCR row is required")
    return anchored[0]


def select_message_regions(observation: ObservationSnapshot) -> MessageRegions:
    """Select bounded visual candidates without using expected recipient or marker.

    The wide contour reaching the foreground's bottom band must be unique.
    A wide transcript bubble above that band is still a message candidate.
    Header geometry comes from a unique left-anchored content-title OCR row
    below window chrome within that contour's horizontal span; OCR strings are
    never copied into the assessor's image. Message candidates remain actual
    contours between those regions, with enclosing transcript frames excluded.
    """
    foreground = _foreground(observation)
    elements = [item for item in observation.elements if _eligible(item, foreground)]
    composers = [
        item for item in elements
        if item.source == "contour" and not item.text.strip()
        and item.bounding_box.width >= foreground.width * 0.45
        and item.bounding_box.top >= foreground.top + foreground.height * 0.60
        # In T04_20261007_140104_387834_d3da5521, a 604-pixel bubble crossed
        # foreground.width * 0.45 = 603.9 but ended at 773, above editor 1019.
        # Require the existing observed contour to reach the window's bottom band;
        # do not choose whichever candidate happens to be nearest the bottom.
        and item.bounding_box.bottom >= foreground.top + foreground.height * _COMPOSER_BOTTOM_FRACTION
        and item.bounding_box.height <= foreground.height * 0.40
    ]
    if len(composers) != 1:
        raise MessageRegionError("exactly one bottom wide contour candidate is required")
    composer = composers[0]
    body = composer.bounding_box
    row = _header_row(elements, foreground, body)
    header = BoundingBox(
        left=max(body.left, min(item.bounding_box.left for item in row) - _HEADER_PADDING),
        top=max(foreground.top, min(item.bounding_box.top for item in row) - _HEADER_PADDING),
        right=min(body.right, max(item.bounding_box.right for item in row) + _HEADER_PADDING),
        bottom=min(body.top, max(item.bounding_box.bottom for item in row) + _HEADER_PADDING),
    )
    if header.bottom >= body.top:
        raise MessageRegionError("header padding leaves no transcript region above the candidate composer")
    transcript = BoundingBox(
        left=body.left, top=header.bottom, right=body.right, bottom=body.top,
    )
    messages = [
        item for item in elements
        if item.source == "contour" and not item.text.strip()
        and _inside(item.bounding_box, transcript)
        and item.bounding_box.width >= body.width * 0.20
        and item.bounding_box.height >= 18
        and item.bounding_box.width * item.bounding_box.height
        <= transcript.width * transcript.height * 0.50
    ]
    # Prefer the larger observed border when the detector kept both inner and
    # outer borders. This does not synthesize a new box or transfer an action id.
    messages.sort(key=lambda item: (
        -item.bounding_box.width * item.bounding_box.height, item.element_id,
    ))
    unique: list[ElementRef] = []
    for item in messages:
        if not any(_duplicate_border(item.bounding_box, other.bounding_box) for other in unique):
            unique.append(item)
    unique.sort(key=lambda item: (
        -item.bounding_box.bottom, -item.bounding_box.top, item.element_id,
    ))
    return MessageRegions(
        observation_id=observation.observation_id,
        header_box=header,
        header_element_ids=tuple(item.element_id for item in row),
        composer=composer,
        messages=tuple(unique[:_MAX_MESSAGES]),
    )


def render_message_regions(
    observation: ObservationSnapshot,
    regions: MessageRegions,
    output_path: Path,
    *,
    include_messages: bool = True,
) -> RenderedMessageRegions:
    """Stack unscaled crops with neutral labels; never render OCR strings or targets.

    ``include_messages=False`` isolates initial recipient/editor assessment from
    older transcript content. Source pixels are untouched; only the surrounding
    matte and label strips are new. Every panel records its original bounds and
    its translation into the contact sheet.
    """
    if regions != select_message_regions(observation):
        raise MessageRegionError("regions must be selected from this exact observation")
    if not observation.image_path:
        raise MessageRegionError("a recorded screenshot is required to render regions")
    destination = Path(output_path)
    source_path = Path(observation.image_path)
    if destination.resolve() == source_path.resolve():
        raise MessageRegionError("the evidence sheet must not overwrite its source screenshot")
    if destination.suffix.casefold() != ".png":
        raise MessageRegionError("the evidence sheet must be saved as a lossless PNG")
    try:
        with Image.open(source_path) as image:
            expected = (
                observation.screen_info.screenshot_width,
                observation.screen_info.screenshot_height,
            )
            if image.size != expected:
                raise MessageRegionError("screenshot dimensions disagree with the observation")
            pixels = image.convert("RGB")
        crops = [
            (
                "HEADER", regions.header_box, regions.header_element_ids,
                {
                    element_id: observation.element(element_id).bounding_box
                    for element_id in regions.header_element_ids
                },
            ),
            (
                "COMPOSER", regions.composer.bounding_box, (regions.composer.element_id,),
                {regions.composer.element_id: regions.composer.bounding_box},
            ),
        ]
        if include_messages:
            for item in regions.messages:
                box = item.bounding_box
                # Adjacent pixels may carry pending/failed-send indicators. The
                # crop may include them; the candidate box itself never expands.
                halo = BoundingBox(
                    left=max(regions.composer.bounding_box.left, box.left - _MESSAGE_HALO),
                    top=max(regions.header_box.bottom, box.top - _MESSAGE_HALO),
                    right=min(regions.composer.bounding_box.right, box.right + _MESSAGE_HALO),
                    bottom=min(regions.composer.bounding_box.top, box.bottom + _MESSAGE_HALO),
                )
                crops.append((
                    f"MESSAGE:{item.element_id}", halo, (item.element_id,),
                    {item.element_id: box},
                ))
        width = max(box.width for _, box, _, _ in crops)
        height = sum(box.height + _LABEL_HEIGHT for _, box, _, _ in crops)
        sheet = Image.new("RGB", (width, height), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        panels: list[RegionPanel] = []
        y = 0
        for label, box, candidate_ids, candidate_boxes in crops:
            draw.text((4, y + 4), label, fill=(0, 0, 0))
            y += _LABEL_HEIGHT
            panel_box = BoundingBox(left=0, top=y, right=box.width, bottom=y + box.height)
            sheet.paste(pixels.crop((box.left, box.top, box.right, box.bottom)), (0, y))
            panels.append(RegionPanel(label, box, panel_box, candidate_ids, candidate_boxes))
            y += box.height
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, format="PNG")
    except (OSError, ValueError) as exc:
        if isinstance(exc, MessageRegionError):
            raise
        raise MessageRegionError(f"cannot render recorded message regions: {exc}") from exc
    return RenderedMessageRegions(
        observation_id=observation.observation_id,
        image_path=destination,
        image_size=(width, height),
        panels=tuple(panels),
    )
