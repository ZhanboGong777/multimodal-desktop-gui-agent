"""Answer one question before a case spends a model call: is the browser in the frame?

Two rounds were spent inferring this from `obs-*.json` after the fact. The symptom is
never an error: the run plans happily against desktop icons - 'Microsoft', 'Edge',
'Chrome' - because a minimised window contributes no text, and then fails task
verification with something like "expected on screen but not found: GUI agent research".
The cause is always that the target window was not on screen when the capture happened,
and the fix is always the same: restore it and check again.

So this takes a frame, runs the project's own OCR over it, and looks for the marks a
browser window leaves - an address bar, a tab strip, a search field. It retries with the
window restored. Exit 0 means the frame contains a browser.

Usage: python week4_capture_probe.py [--retries 3]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Developer\multimodal-desktop-gui-agent")
sys.path.insert(0, str(REPO / "src"))

from gui_agent.config import load_config
from gui_agent.perception.capture import capture_monitor
from gui_agent.perception.ocr import create_ocr_engine

#: Text that only a browser window carries. A desktop shortcut labelled 'Chrome' is not
#: enough - the point is to distinguish the window from its icon, which is exactly the
#: distinction the model could not make from the element list it was given.
BROWSER_MARKS = (
    "http", "www.", ".com", "搜索", "search", "地址", "address",
    "新标签页", "new tab", "google", "bing", "baidu",
)


def restore_browser() -> bool:
    """Bring Chrome (or Edge) back and activate it; True when a window was found."""
    script = (
        "$sig='[DllImport(\"user32.dll\")] public static extern bool ShowWindow(IntPtr h,int c);"
        "[DllImport(\"user32.dll\")] public static extern bool IsIconic(IntPtr h);"
        "[DllImport(\"user32.dll\")] public static extern bool SetForegroundWindow(IntPtr h);';"
        "$t=Add-Type -MemberDefinition $sig -Name PB -Namespace Probe -PassThru;"
        "$p = Get-Process chrome,msedge -ErrorAction SilentlyContinue | "
        "Where-Object { $_.MainWindowHandle -ne 0 } | "
        "Sort-Object @{Expression={ if ($_.ProcessName -eq 'chrome') {0} else {1} }} | "
        "Select-Object -First 1;"
        "if (-not $p) { 'none'; exit }"
        "if ([Probe.PB]::IsIconic($p.MainWindowHandle)) {"
        "  [Probe.PB]::ShowWindow($p.MainWindowHandle, 9) | Out-Null; Start-Sleep -Milliseconds 700 }"
        "[Probe.PB]::SetForegroundWindow($p.MainWindowHandle) | Out-Null; $p.ProcessName"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(out) and out != "none"


def browser_in_frame() -> tuple[bool, list[str]]:
    """Capture, OCR, and report whether the frame looks like a browser window."""
    cfg = load_config(str(REPO / "configs" / "week4.yaml"))
    engine = create_ocr_engine(cfg.perception.ocr)
    frame = capture_monitor(
        cfg.perception.monitor_index,
        output_directory=Path(cfg.output.directory),
        save=True,
    )
    result = engine.engine.recognize(frame.image)
    texts = [e.text for e in getattr(result, "elements", [])]
    lowered = [t.casefold() for t in texts]
    hits = sorted({m for m in BROWSER_MARKS for t in lowered if m in t})
    return bool(hits), hits


def main() -> int:
    retries = 3
    if "--retries" in sys.argv:
        retries = int(sys.argv[sys.argv.index("--retries") + 1])

    for attempt in range(1, retries + 1):
        present, hits = browser_in_frame()
        print(f"attempt {attempt}: browser marks {hits if hits else 'none'}")
        if present:
            print("browser is in the frame")
            return 0
        restored = restore_browser()
        print(f"  not in frame; restore_browser -> {restored}")
        time.sleep(2)

    print("WARNING: no browser marks in the frame after "
          f"{retries} attempts; the case will plan against whatever is on screen")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
