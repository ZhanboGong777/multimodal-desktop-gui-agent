"""Capture the screen, run OCR and contour detection, and save the results.

Example:
    python scripts/week2_perception_demo.py
    python scripts/week2_perception_demo.py --region 0 0 800 600 --save-intermediate
    python scripts/week2_perception_demo.py --frames 10 --interval 0.2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import Config, load_config
from gui_agent.perception.capture import (
    CaptureRegion,
    CaptureResult,
    CaptureSequence,
    capture_frames,
    capture_monitor,
    capture_region,
)
from gui_agent.perception.ocr import OcrError, create_ocr_engine
from gui_agent.perception.preprocessing import apply_preprocessing
from gui_agent.perception.ui_detection import detect_ui_candidates
from gui_agent.perception.visualization import (
    draw_bounding_boxes,
    save_annotated_image,
    summarize_sources,
)
from gui_agent.recording import (
    ANNOTATED_IMAGE,
    BEFORE_IMAGE,
    RunSession,
    build_run_summary,
)
from gui_agent.schemas import PerceptionResult, UIElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--monitor", type=int, default=None, help="monitor index (default from config)"
    )
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="capture only this region instead of the whole monitor",
    )
    parser.add_argument("--ocr-engine", choices=["paddleocr", "tesseract"], default=None)
    parser.add_argument("--language", action="append", default=None, help="repeatable OCR language")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument(
        "--save-intermediate", action="store_true", help="also store the preprocessed image"
    )
    parser.add_argument("--no-ui-detection", action="store_true")
    parser.add_argument(
        "--max-ui-candidates",
        type=int,
        default=None,
        help="cap the number of contour candidates (default from config)",
    )
    parser.add_argument(
        "--label-all",
        action="store_true",
        help="label every element, including contour candidates (clutters the image)",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="capture a sequence of this many frames and work on the last one",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="seconds between frames (default from config; only used with --frames > 1)",
    )
    return parser.parse_args()


def capture(
    config: Config, args: argparse.Namespace
) -> tuple[CaptureResult, CaptureSequence | None]:
    """Capture one frame, or a short sequence when ``--frames`` asks for more.

    Returns the frame the perception pipeline should work on (the most recent
    one) plus the sequence itself when more than one frame was requested, so its
    timing can be reported.
    """
    monitor_index = args.monitor if args.monitor is not None else config.perception.monitor_index
    region = CaptureRegion(*args.region) if args.region else None

    if args.frames <= 1:
        if region is not None:
            return capture_region(region, monitor_index=monitor_index), None
        return capture_monitor(monitor_index), None

    interval = (
        args.interval if args.interval is not None else config.perception.capture.interval_seconds
    )
    sequence = capture_frames(args.frames, interval, monitor_index=monitor_index, region=region)
    return sequence[-1], sequence


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.ocr_engine:
        config.perception.ocr.engine = args.ocr_engine
    if args.language:
        config.perception.ocr.languages = args.language
    if args.min_confidence is not None:
        config.perception.ocr.min_confidence = args.min_confidence
    if args.no_ui_detection:
        config.perception.ui_detection.enabled = False

    interval = (
        args.interval if args.interval is not None else config.perception.capture.interval_seconds
    )
    started = time.perf_counter()
    session = RunSession.create(config.perception.output_directory, args.session_id)
    log: list[str] = []

    def note(message: str) -> None:
        print(message)
        log.append(message)

    note(f"session directory: {session.directory}")
    frame, sequence = capture(config, args)
    if sequence is not None:
        note(
            f"captured {len(sequence)} frames in {sequence.elapsed_seconds:.2f} s "
            f"(interval {interval:.2f} s); working on the last one"
        )
        note(
            f"capture-only throughput {sequence.capture_only_fps:.1f} FPS, "
            f"effective sequence rate {sequence.effective_fps:.2f} FPS"
        )
    note(f"captured {frame.image.width}x{frame.image.height} in {frame.capture_time_ms:.1f} ms")
    note(f"screen info: {frame.screen_info.model_dump_json()}")

    before_path = session.path_for(BEFORE_IMAGE)
    frame.image.save(before_path)

    processed = apply_preprocessing(frame.image, config.perception.preprocessing)
    if args.save_intermediate:
        intermediate = session.path_for("preprocessed.png")
        processed.save(intermediate)
        note(f"preprocessed image: {intermediate}")

    selection = create_ocr_engine(config.perception.ocr)
    for notice in selection.notices:
        note(f"notice: {notice}")
    shown_notices = len(selection.notices)
    try:
        ocr_output = selection.engine.recognize(
            processed,
            languages=config.perception.ocr.languages,
            min_confidence=config.perception.ocr.min_confidence,
        )
    except OcrError as exc:
        note(f"OCR failed: {exc}")
        session.save_log(log)
        return 1
    for notice in selection.notices[shown_notices:]:
        note(f"notice: {notice}")
    note(
        f"OCR backend '{ocr_output.engine}' found {len(ocr_output.elements)} text regions "
        f"in {ocr_output.elapsed_ms:.1f} ms"
    )

    elements: list[UIElement] = list(ocr_output.elements)
    if config.perception.ui_detection.enabled:
        candidates = detect_ui_candidates(
            frame.image,
            min_area=config.perception.ui_detection.min_area,
            max_area_ratio=config.perception.ui_detection.max_area_ratio,
            min_rectangularity=config.perception.ui_detection.min_rectangularity,
            max_candidates=(
                args.max_ui_candidates
                if args.max_ui_candidates is not None
                else config.perception.ui_detection.max_candidates
            ),
            # Do not re-frame text that OCR has already located.
            exclude=[element.bounding_box for element in ocr_output.elements],
            exclusion_threshold=config.perception.ui_detection.exclusion_threshold,
        )
        note(f"contour detection proposed {len(candidates)} non-text candidates")
        elements.extend(candidates)

    for element in elements[:15]:
        box = element.bounding_box
        note(
            f"  [{element.source}] {element.text!r} conf={element.confidence:.2f} "
            f"box=({box.left},{box.top},{box.right},{box.bottom}) center=({element.center.x},{element.center.y})"
        )
    if len(elements) > 15:
        note(f"  ... {len(elements) - 15} more")

    annotated = draw_bounding_boxes(
        frame.image,
        elements,
        label_sources=("contour", "ocr", "manual") if args.label_all else ("ocr",),
        show_confidence=True,
    )
    annotated_path = save_annotated_image(annotated, session.path_for(ANNOTATED_IMAGE))
    note(f"annotated image: {annotated_path}")

    result = PerceptionResult(
        screen_info=frame.screen_info,
        image_path=str(before_path),
        elements=elements,
        processing_time_ms=(time.perf_counter() - started) * 1000.0,
        errors=list(ocr_output.errors),
    )
    session.save_perception(result)
    session.save_summary(
        build_run_summary(
            selected_target=None,
            action=None,
            result=None,
            capture_time_ms=frame.capture_time_ms,
            ocr_time_ms=ocr_output.elapsed_ms,
            total_time_ms=result.processing_time_ms,
            element_count=len(elements),
            extra={"sources": summarize_sources(elements)},
        )
    )
    session.save_log(log)
    note(f"perception JSON: {session.path_for('perception.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
