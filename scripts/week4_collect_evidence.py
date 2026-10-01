"""Copy a finished run's text records into the repository as evidence.

`outputs/` is not tracked, so a run id quoted in the test report resolves to
nothing as soon as the report is read anywhere but the machine that produced it.
The hand-off asks for a report whose run ids can be traced back to their evidence,
which means the evidence has to travel with the repository.

Two files are copied: ``task_summary.json`` (the verdict, timings and notes) and
``steps.jsonl`` (one line per step). The screenshots and the per-frame observation
files are deliberately left behind - they are a picture of the whole desktop, and
publishing the whole desktop is not what "attach the evidence" should mean.

    python scripts/week4_collect_evidence.py --latest T01
    python scripts/week4_collect_evidence.py --session outputs/week4/T01_20261001_130401
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = REPO_ROOT / "outputs" / "week4"
DEFAULT_DESTINATION = REPO_ROOT / "Document" / "Week4" / "evidence"

#: Copied verbatim when present. Anything else in the session directory - images,
#: ``obs-NNNN.json`` - is screen content and stays out.
RECORDS = ("task_summary.json", "steps.jsonl")

EXIT_OK = 0
EXIT_ERROR = 2


class EvidenceError(RuntimeError):
    """Raised when there is no run to collect, or the run did not finish."""


def latest_session(case_id: str, root: Path = DEFAULT_SESSION_ROOT) -> Path:
    """Newest session directory for a case.

    Session names are ``<case>_<timestamp>``, so lexical order is chronological
    order and the last one is the most recent run.
    """
    root = Path(root)
    candidates = sorted(path for path in root.glob(f"{case_id.upper()}_*") if path.is_dir())
    if not candidates:
        raise EvidenceError(f"no {case_id.upper()} session under {root}")
    return candidates[-1]


def run_id_of(session: Path) -> str:
    """The run id recorded in a session's summary, or an empty string."""
    summary = Path(session) / "task_summary.json"
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("run_id", ""))


def collect(session: Path, destination: Path, *, include_steps: bool = True) -> list[Path]:
    """Copy the text records of ``session`` under ``destination``/<session name>.

    Returns the files written. Raises :class:`EvidenceError` when the session has
    no summary: a run without one did not finish, and a half-run is not evidence.
    """
    session = Path(session)
    if not session.is_dir():
        raise EvidenceError(f"{session} is not a directory")
    summary = session / "task_summary.json"
    if not summary.is_file():
        raise EvidenceError(f"{summary} is missing; the run did not finish")

    target = Path(destination) / session.name
    target.mkdir(parents=True, exist_ok=True)
    wanted = RECORDS if include_steps else RECORDS[:1]
    copied = [
        shutil.copy2(session / name, target / name)
        for name in wanted
        if (session / name).is_file()
    ]
    return copied


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy a run's text records into Document/Week4/evidence/.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", type=Path, help="a session directory under outputs/week4")
    source.add_argument("--latest", metavar="CASE", help="the newest session for a case, e.g. T01")
    parser.add_argument(
        "--session-root",
        type=Path,
        default=DEFAULT_SESSION_ROOT,
        help="where --latest looks (default: outputs/week4)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="where the copy goes (default: Document/Week4/evidence)",
    )
    parser.add_argument(
        "--no-steps",
        action="store_true",
        help="copy only the summary, not the per-step log",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        session = (
            latest_session(args.latest, args.session_root) if args.latest else Path(args.session)
        )
        copied = collect(session, args.destination, include_steps=not args.no_steps)
    except EvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    run_id = run_id_of(session)
    print(f"session : {session}")
    print(f"run id  : {run_id or '(not recorded)'}")
    for path in copied:
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        print(f"copied  : {shown}")
    print("\nQuote the run id in the report; the path above is what it resolves to.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
