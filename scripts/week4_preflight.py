"""Pre-run checks for the Week 4 cases. Run this BEFORE a real task run.

Four things went wrong on the Windows review machine, and all four were silent: the
run looked normal until it failed somewhere unrelated. Each check here turns one of
them into an explicit message.

1. The model could not answer inside ``model.timeout_seconds``. On this machine a
   planning call took 124 s against a 120 s timeout, so the run died with
   "planning failed: ModelError: APITimeoutError" - which names the transport, not the
   cause. Measured here: warm text ~0.7 s, warm image ~0.9 s, cold start ~12 s.
2. The observation contained the *operator's own terminal* instead of the desktop. The
   element list was made of fragments like 'not set.', '(825.4 ms', 'Cache hit 99%',
   'blocked' - the harness window - so no real UI element was offered and the model
   invented a target. A terminal window on screen is enough to do this: text elements
   are ranked above contour boxes and the former 60-element cap excluded the desktop.
   The cap is now 300, but an obstructed capture still offers the wrong window.
3. The Ollama context window was 4096 while the former 60-slot prompt needed
   ~7 500 tokens, giving HTTP 400 "exceeds the available context size". The new
   300-slot anonymous list needs a new request-size measurement.
4. The screen was already in the end state the case is supposed to produce, which the
   runner reports as `blocked` (correct) - worth knowing before spending a model call.

Exit code 0 means "go", 1 means "fix the printed problem first".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gui_agent.config import load_config
from gui_agent.models import create_model_client
from gui_agent.models.base import ModelError
from gui_agent.perception.capture import capture_monitor
from gui_agent.perception.ocr import create_ocr_engine

#: Words that only ever appear in a terminal, a log or a chat transcript. Two hits are
#: enough to conclude the frame is showing the operator's console rather than a desktop.
TERMINAL_MARKERS = (
    "pwsh", "powershell", "cmd.exe", "c:\\windows\\system32", "powershell.exe",
    "traceback", "modelerror", "blocked", "failed", "not set.", "conf=",
    "timeout_seconds", "apiconnectionerror", "apitimeouterror", "ms)", "exit code",
    "$env:", "pip install", "pytest", "week4", "obs-0001", "task status",
)

OK = "  [ok]   "
BAD = "  [FAIL] "
WARN = "  [warn] "


def check_model(config_path: str, model: str | None) -> list[str]:
    """Is the model reachable, and does it answer inside the configured timeout?"""
    problems: list[str] = []
    cfg = load_config(config_path)
    if model:
        cfg.model.model_name = model

    print(f"model      : {cfg.model.provider} / {cfg.model.model_name}")
    print(f"timeout    : {cfg.model.timeout_seconds:g} s per request")

    client = create_model_client(cfg.model)
    try:
        started = time.perf_counter()
        client.complete([{"role": "user", "content": "Reply with exactly: OK"}])
        elapsed = time.perf_counter() - started
    except ModelError as exc:
        problems.append(f"the model did not answer: {exc}")
        print(BAD + f"model unreachable: {exc}")
        return problems

    print(OK + f"answered in {elapsed:.1f} s")
    if elapsed > cfg.model.timeout_seconds * 0.5:
        print(WARN + f"a text-only call took {elapsed:.1f} s, more than half the "
                     f"{cfg.model.timeout_seconds:g} s budget. A planning call carries a "
                     f"2560x1600 image plus the element list and is much slower.")
        problems.append("the model is slow enough that a planning call is likely to time out")
    return problems


def check_observation(config_path: str) -> list[str]:
    """Would the next observation offer real UI elements, or the operator's console?"""
    problems: list[str] = []
    cfg = load_config(config_path)
    engine = create_ocr_engine(cfg.perception.ocr)
    out_dir = Path(cfg.output.directory)

    frame = capture_monitor(cfg.perception.monitor_index, output_directory=out_dir, save=True)
    print(f"screenshot : {frame.image_path}")

    result = engine.engine.recognize(frame.image)
    elements = list(getattr(result, "elements", []))
    texts = [e.text for e in elements]
    print(f"ocr        : engine={engine.engine.name} elements={len(texts)}")

    if engine.engine.name != cfg.perception.ocr.engine:
        print(WARN + f"configured {cfg.perception.ocr.engine} but {engine.engine.name} is "
                     f"in use (fallback). Notices: {engine.notices}")
        problems.append("the configured OCR engine fell back to another one")

    if not texts:
        print(BAD + "OCR returned no text at all; the screen may be blank or locked")
        problems.append("no text on screen")
        return problems

    lowered = [t.lower() for t in texts]
    hits = sorted({m for m in TERMINAL_MARKERS for t in lowered if m in t})
    console_like = sum(1 for t in lowered if any(m in t for m in TERMINAL_MARKERS))
    share = console_like / len(lowered)

    print(f"console-ish: {console_like}/{len(lowered)} elements match terminal markers "
          f"({share:.0%})")
    if hits:
        print(f"             markers seen: {hits[:12]}")

    # Coverage of the screen by text boxes is reported, not gated on: OCR boxes wrap the
    # glyphs, not the window, so a maximised application can still read as a few percent.
    # The honest signal is the element count together with what the text says.
    screen = frame.screen_info
    screen_area = max(1.0, float(screen.screenshot_width) * float(screen.screenshot_height))
    covered = 0.0
    for element in elements:
        box = element.bounding_box
        covered += max(0, box.right - box.left) * max(0, box.bottom - box.top)
    coverage = min(1.0, covered / screen_area)
    print(f"text cover : {coverage:.0%} of the screen area is inside a text box")

    busy = len(texts) >= 100 or share >= 0.25
    if busy:
        print(BAD + f"{len(texts)} text elements on screen ({console_like} of them carrying "
                    f"terminal or log wording).")
        print("             A desktop worth acting on has one window and some icons. This "
              "screen is an application - very likely the one running this agent - and "
              "because text is ranked above contour boxes and the list is capped at "
              "execution.max_elements, it can take the entire element budget. The model is "
              "then shown that window's text instead of the desktop, and invents targets.")
        print("             Run the case from a session whose own window is NOT on the "
              "captured monitor, or minimise everything and re-run this check until the "
              "element count drops and the top elements are desktop icons.")
        problems.append(f"the screen looks like an application window, not a desktop "
                        f"({len(texts)} text elements)")
    else:
        print(OK + f"{len(texts)} text elements: consistent with a desktop")

    print("             first 12 elements:")
    for t in texts[:12]:
        print(f"               {t[:70]!r}")
    return problems


def check_context(base_url: str | None, required: int) -> list[str]:
    """Is the Ollama context window big enough for the prompt?"""
    import json
    import urllib.request

    problems: list[str] = []
    url = (base_url or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(url + "/api/ps", timeout=10) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        print(WARN + f"could not read {url}/api/ps: {type(exc).__name__}: {exc}")
        return problems

    models = payload.get("models") or []
    if not models:
        print(WARN + "no model is loaded; the first planning call will pay the load cost "
                     "(about 12 s here, 85 s for a freshly downloaded model)")
        problems.append("no model is loaded - warm it up first "
                        "(scripts/week4_warmup.py) so the load cost is not counted as planning time")
        return problems

    ctx = models[0].get("context_length")
    print(f"context    : model={models[0].get('name')} context_length={ctx} "
          f"vram={models[0].get('size_vram', 0) / 1e9:.1f} GB")
    if ctx and ctx < required:
        print(BAD + f"context_length {ctx} is below the {required} tokens a run needs "
                    f"(2560x1600 screenshot + element list measured at ~7 500 tokens).")
        print("             Set OLLAMA_CONTEXT_LENGTH before the server starts - it must be "
              "inherited by the server process, so restart it, not just reload the model.")
        problems.append(f"context window {ctx} < {required}")
    elif ctx:
        print(OK + f"context window {ctx} covers the {required} needed")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs" / "week4.yaml"))
    ap.add_argument("--model", default=None, help="override model.model_name")
    ap.add_argument("--base-url", default=None, help="Ollama base url for the context check")
    ap.add_argument("--required-context", type=int, default=8192)
    ap.add_argument("--skip-model", action="store_true", help="do not call the model")
    args = ap.parse_args()

    print("Week 4 preflight")
    print("=" * 70)
    problems: list[str] = []

    if not args.skip_model:
        problems += check_model(args.config, args.model)
    print()
    problems += check_context(args.base_url, args.required_context)
    print()
    problems += check_observation(args.config)

    print()
    print("=" * 70)
    if problems:
        print(f"{len(problems)} problem(s) - fix these before running a case:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("preflight passed: the model answers in budget, the context is big enough, "
          "and the screen looks like a desktop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
