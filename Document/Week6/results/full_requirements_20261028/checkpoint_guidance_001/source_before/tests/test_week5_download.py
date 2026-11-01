"""Public model downloads cannot publish corrupted weights or escape their root."""

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[1] / "scripts/week5_download_assets.py"
    spec = importlib.util.spec_from_file_location("week5_downloader_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_existing_weight_is_reused_without_network(downloader, tmp_path, monkeypatch):
    content = b"verified weight"
    (tmp_path / "weight.safetensors").write_bytes(content)

    def unexpected(*args, **kwargs):
        raise AssertionError("valid local weights must not trigger a network request")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", unexpected)
    downloader.download_shard_in_ranges(
        tmp_path, "weight.safetensors", len(content), hashlib.sha256(content).hexdigest()
    )


def test_bad_range_checksum_cannot_publish_a_weight(downloader, tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers: ClassVar[dict[str, str]] = {"Content-Range": "bytes 0-2/3"}

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda *a, **k: Response(b"bad"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        downloader.download_shard_in_ranges(
            tmp_path, "weight.safetensors", 3, hashlib.sha256(b"yes").hexdigest()
        )
    assert not (tmp_path / "weight.safetensors").exists()
    assert (tmp_path / "weight.safetensors.partial").exists()


def test_dataset_archive_cannot_extract_outside_its_source_root(
    downloader, tmp_path, monkeypatch
):
    monkeypatch.setattr(downloader, "REPO_ROOT", tmp_path)
    raw = tmp_path / "data/raw/screenagent"
    raw.mkdir(parents=True)
    archive = raw / f"ScreenAgent-{downloader.SCREENAGENT_REVISION}.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            "ScreenAgent/data/ScreenAgent/train/../../../../escaped.bin", b"bad"
        )
    with pytest.raises(ValueError, match="unsafe archive member"):
        downloader.download_dataset()
    assert not (raw / "escaped.bin").exists()
