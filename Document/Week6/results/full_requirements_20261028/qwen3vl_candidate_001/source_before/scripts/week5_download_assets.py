"""Download pinned public Week 5 assets; never collect local desktop images."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENAGENT_REVISION = "2312c852908dda5fbc43d903fad5929a9dec649d"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
MODEL_SHARDS = {
    "model-00001-of-00002.safetensors": (
        3988609112, "994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9"
    ),
    "model-00002-of-00002.safetensors": (
        429441656, "92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download_dataset() -> None:
    destination = REPO_ROOT / "data/raw/screenagent"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"ScreenAgent-{SCREENAGENT_REVISION}.zip"
    url = f"https://codeload.github.com/niuzaisheng/ScreenAgent/zip/{SCREENAGENT_REVISION}"
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        started = time.perf_counter()
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            downloaded = 0
            next_report = 16 * 1024 * 1024
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"dataset {downloaded / 1024**2:.1f} MiB in "
                          f"{time.perf_counter() - started:.1f}s", flush=True)
                    next_report += 16 * 1024 * 1024
        with zipfile.ZipFile(partial) as source:
            bad = source.testzip()
            if bad:
                raise ValueError(f"corrupt archive member: {bad}")
        partial.replace(archive)
    root = (destination / "source").resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = []
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            pieces = Path(item.filename).parts
            if len(pieces) < 2 or item.is_dir():
                continue
            relative = Path(*pieces[1:])
            name = relative.as_posix()
            if not (
                name.startswith("data/ScreenAgent/train/")
                or name in {"data/ScreenAgent/test.zip", "README.md", "LICENSE"}
            ):
                continue
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe archive member: {item.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            content = source.read(item)
            target.write_bytes(content)
            files.append({"path": name, "bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest()})
    write_json(destination / "download_manifest.json", {
        "source": "https://github.com/niuzaisheng/ScreenAgent",
        "revision": SCREENAGENT_REVISION, "url": url,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "archive": str(archive), "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive), "extracted_files": files,
    })
    print(f"Dataset ready: {root}; {len(files)} files", flush=True)


def download_shard_in_ranges(destination: Path, name: str, size: int, digest: str) -> None:
    """Recover a public download using identical official ModelScope weights.

    Sizes and SHA256 values come from the pinned HF revision's LFS metadata.
    Every range is checked before assembling; the final file must match the
    pinned HF checksum, regardless of the serving mirror's branch name.
    """
    target = destination / name
    if target.exists() and target.stat().st_size == size and sha256(target) == digest:
        print(f"Verified existing {name}", flush=True)
        return
    parts = destination / ".download_parts" / name
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 64 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, size) - 1)
              for start in range(0, size, chunk_size)]

    def fetch(bounds: tuple[int, int]) -> Path:
        start, end = bounds
        part = parts / f"{start:012d}-{end:012d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        url = (f"https://modelscope.cn/models/{MODEL_ID}/resolve/master/{name}"
               f"?week5_range={start}")
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected_range = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"unexpected range response: {response.status}, "
                                         f"{response.headers.get('Content-Range')}")
                    temporary = part.with_suffix(".partial")
                    with temporary.open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                    if temporary.stat().st_size != end - start + 1:
                        raise ValueError("incomplete range")
                    temporary.replace(part)
                    print(f"{name}: range {start // chunk_size + 1}/{len(ranges)} ready", flush=True)
                    return part
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("unreachable range retry state")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        ordered_parts = list(executor.map(fetch, ranges))
    temporary = target.with_suffix(".safetensors.partial")
    with temporary.open("wb") as output:
        for part in ordered_parts:
            with part.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(chunk)
    if temporary.stat().st_size != size or sha256(temporary) != digest:
        raise ValueError(f"checksum mismatch for assembled {name}; no model file published")
    temporary.replace(target)
    print(f"Verified pinned HF SHA256: {name}", flush=True)


def download_model(mirror: str) -> None:
    from huggingface_hub import snapshot_download

    destination = REPO_ROOT / "models/Qwen2-VL-2B-Instruct"
    if mirror == "huggingface":
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(destination),
            token=False, max_workers=3,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
        )
    else:
        # Configuration/tokenizer files still come from the pinned HF revision.
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(destination),
            token=False, max_workers=2,
            allow_patterns=["*.json", "*.txt", "LICENSE", "README.md"],
        )
        for name, (size, digest) in MODEL_SHARDS.items():
            download_shard_in_ranges(destination, name, size, digest)
    for name, (size, digest) in MODEL_SHARDS.items():
        shard = destination / name
        if shard.stat().st_size != size or sha256(shard) != digest:
            raise ValueError(f"downloaded shard does not match the pinned HF revision: {name}")
    files = []
    for path in sorted(destination.iterdir()):
        if path.is_file() and path.name != "week5_download_manifest.json":
            files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(destination / "week5_download_manifest.json", {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "weight_serving_source": mirror, "pinned_hf_lfs": MODEL_SHARDS,
        "downloaded_at": datetime.now(UTC).isoformat(), "files": files,
    })
    print(f"Model ready: {destination}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["dataset", "model"])
    parser.add_argument("--mirror", choices=["huggingface", "modelscope"], default="huggingface")
    args = parser.parse_args()
    if args.asset == "dataset":
        download_dataset()
    else:
        download_model(args.mirror)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
