"""Verify the published file bytes on CPU, without loading models or a desktop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Document/Week5/results/file_checksums.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(relative)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(relative)
    print(json.dumps({"status": "passed" if not failures else "failed",
                      "files_checked": len(manifest["sha256"]), "failures": failures,
                      "model_loads": 0, "desktop_inputs": 0}, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
