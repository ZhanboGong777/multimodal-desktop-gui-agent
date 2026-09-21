"""Closed loop: capture, recognise, locate a text target, plan, optionally act.

The default is ``--move-only`` plus a dry run, so the pointer is only moved when
both ``--move-only`` and ``--execute`` are supplied.

Examples:
    python scripts/week2_closed_loop_demo.py --target-text README
    python scripts/week2_closed_loop_demo.py --target-text README --move-only --execute
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gui_agent.config import load_config
from gui_agent.control.executor import ActionExecutor
from gui_agent.coordinates import screenshot_to_control_clamped
from gui_agent.perception.capture import capture_monitor
from gui_agent.perception.grounding import find_text, format_candidates
from gui_agent.perception.ocr import OcrError, create_ocr_engine
from gui_agent.perception.preprocessing import apply_preprocessing
from gui_agent.perception.visualization import (
    draw_bounding_boxes,
    save_annotated_image,
)
from gui_agent.recording import (
    AFTER_IMAGE,
    ANNOTATED_IMAGE,
    BEFORE_IMAGE,
    RunSession,
    build_run_summary,
)
from gui_agent.schemas import DesktopAction, PerceptionResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--target-text", required=True)
    parser.add_argument(
        "--match-mode",
        choices=["exact", "contains", "case_insensitive"],
        default="contains",
    )
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument(
        "--move-only", action="store_true", help="move the pointer instead of clicking"
    )
    parser.add_argument("--execute", action="store_true", help="really send mouse events")
    parser.add_argument("--session-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.min_confidence is not None:
        config.perception.ocr.min_confidence = args.min_confidence

    started = time.perf_counter()
    session = RunSession.create(config.perception.output_directory, args.session_id)
    lines: list[str] = []

    def note(message: str) -> None:
        print(message)
        lines.append(message)

    frame = capture_monitor(config.perception.monitor_index)
    frame.image.save(session.path_for(BEFORE_IMAGE))
    note(f"captured {frame.image.width}x{frame.image.height} in {frame.capture_time_ms:.1f} ms")

    processed = apply_preprocessing(frame.image, config.perception.preprocessing)
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
        session.save_log(lines)
        return 1
    for notice in selection.notices[shown_notices:]:
        note(f"notice: {notice}")
    note(
        f"OCR backend '{ocr_output.engine}' found {len(ocr_output.elements)} regions "
        f"in {ocr_output.elapsed_ms:.1f} ms"
    )

    result = find_text(
        ocr_output.elements,
        args.target_text,
        match_mode=args.match_mode,
        min_confidence=config.perception.ocr.min_confidence,
        candidate_index=args.candidate_index,
        screen_info=frame.screen_info,
    )
    for line in format_candidates(result):
        note(line)
    if result.message:
        note(f"grounding: {result.message}")

    annotated = draw_bounding_boxes(
        frame.image,
        [match.element for match in result.candidates],
        show_confidence=True,
        show_index=True,
    )
    note(f"annotated image: {save_annotated_image(annotated, session.path_for(ANNOTATED_IMAGE))}")

    selected = result.selected
    if selected is None:
        note("no target found; nothing to execute")
        session.save_log(lines)
        session.save_summary(
            build_run_summary(
                selected_target=args.target_text,
                action=None,
                result=None,
                capture_time_ms=frame.capture_time_ms,
                ocr_time_ms=ocr_output.elapsed_ms,
                total_time_ms=(time.perf_counter() - started) * 1000.0,
                element_count=len(ocr_output.elements),
                error=result.message or "target not found",
            )
        )
        return 1

    control_point = selected.control_center or screenshot_to_control_clamped(
        selected.screenshot_center, frame.screen_info
    )
    note(
        f"selected target {selected.matched_text!r}: screenshot center "
        f"({selected.screenshot_center.x},{selected.screenshot_center.y}) -> control "
        f"({control_point.x},{control_point.y})"
    )

    action_type = "move" if args.move_only else "click"
    action = DesktopAction(
        action_type=action_type,  # type: ignore[arg-type]
        x=control_point.x,
        y=control_point.y,
        target_description=selected.matched_text,
    )
    note(f"planned action: {action_type} at ({control_point.x},{control_point.y})")

    dry_run = not args.execute
    if args.execute:
        note("--execute given: move the pointer to a screen corner to abort")

    executor = ActionExecutor(config.control, screen=frame.screen_info, notice=note, countdown=note)
    action_result = executor.execute(action, dry_run=dry_run)
    note(
        f"success={action_result.success} dry_run={action_result.dry_run} error={action_result.error}"
    )

    if not dry_run:
        time.sleep(config.control.action_delay_seconds)
    after = capture_monitor(config.perception.monitor_index)
    after.image.save(session.path_for(AFTER_IMAGE))
    note(f"after image: {session.path_for(AFTER_IMAGE)}")

    session.save_perception(
        PerceptionResult(
            screen_info=frame.screen_info,
            image_path=str(session.path_for(BEFORE_IMAGE)),
            elements=ocr_output.elements,
            processing_time_ms=ocr_output.elapsed_ms,
            errors=list(ocr_output.errors),
        )
    )
    session.save_action(action, action_result)
    session.save_summary(
        build_run_summary(
            selected_target=selected.matched_text,
            action=action,
            result=action_result,
            capture_time_ms=frame.capture_time_ms + after.capture_time_ms,
            ocr_time_ms=ocr_output.elapsed_ms,
            total_time_ms=(time.perf_counter() - started) * 1000.0,
            element_count=len(ocr_output.elements),
            error=action_result.error,
            extra={"candidate_count": len(result.candidates)},
        )
    )
    session.save_log(lines)
    note(f"run record: {session.directory}")
    return 0 if action_result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
