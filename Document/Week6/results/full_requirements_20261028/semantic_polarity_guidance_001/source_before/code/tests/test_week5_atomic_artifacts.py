"""CPU regressions for atomic training JSON under transient Windows readers."""

from __future__ import annotations

import errno
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gui_agent.finetuning import training


def windows_lock(code: int) -> PermissionError:
    error = PermissionError(errno.EACCES, "synthetic Windows destination lock")
    error.winerror = code
    return error


@pytest.fixture
def json_pair(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    old = {"status": "running", "updates": 44, "text": "原有记录"}
    new = {"status": "running", "updates": 45, "text": "新记录"}
    path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        training, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    return path, path.with_suffix(".json.tmp"), old, new, clock


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_lock_recovers_without_exposing_partial_json(
    json_pair, monkeypatch, winerror
):
    path, temporary, old, new, clock = json_pair
    original_replace = Path.replace
    attempts = []

    def replace(source, destination):
        assert source == temporary and destination == path
        assert json.loads(path.read_text(encoding="utf-8")) == old
        assert json.loads(temporary.read_text(encoding="utf-8")) == new
        attempts.append(source)
        if len(attempts) <= 3:
            raise windows_lock(winerror)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    training._write_json(path, new)
    assert len(attempts) == 4
    assert clock.sleeps == [0.025, 0.05, 0.1]
    assert json.loads(path.read_text(encoding="utf-8")) == new
    assert not temporary.exists()


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_persistent_windows_lock_is_bounded_and_keeps_both_valid_records(
    json_pair, monkeypatch, winerror
):
    path, temporary, old, new, clock = json_pair
    attempts = []
    error = windows_lock(winerror)

    def replace(source, destination):
        attempts.append((source, destination))
        assert json.loads(path.read_text(encoding="utf-8")) == old
        assert json.loads(temporary.read_text(encoding="utf-8")) == new
        raise error

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        training._write_json(path, new)
    assert caught.value is error
    assert len(attempts) == 8 and len(clock.sleeps) == 7
    assert sum(clock.sleeps) <= 2.0
    assert json.loads(path.read_text(encoding="utf-8")) == old
    assert json.loads(temporary.read_text(encoding="utf-8")) == new


@pytest.mark.parametrize("winerror", [None, 1, 2, 3, 87])
def test_other_permission_errors_are_not_retried(json_pair, monkeypatch, winerror):
    path, temporary, old, new, clock = json_pair
    error = PermissionError(errno.EACCES, "non-sharing permission error")
    if winerror is not None:
        error.winerror = winerror
    attempts = []

    def replace(source, destination):
        attempts.append((source, destination))
        raise error

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError) as caught:
        training._write_json(path, new)
    assert caught.value is error
    assert len(attempts) == 1 and not clock.sleeps
    assert json.loads(path.read_text(encoding="utf-8")) == old
    assert json.loads(temporary.read_text(encoding="utf-8")) == new


def test_io_error_is_not_retried(json_pair, monkeypatch):
    path, temporary, old, new, clock = json_pair
    error = OSError(errno.EIO, "disk I/O failure")
    attempts = []

    def replace(source, destination):
        attempts.append((source, destination))
        raise error

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError) as caught:
        training._write_json(path, new)
    assert caught.value is error
    assert len(attempts) == 1 and not clock.sleeps
    assert json.loads(path.read_text(encoding="utf-8")) == old
    assert json.loads(temporary.read_text(encoding="utf-8")) == new


def test_retry_budget_also_stops_after_slow_failed_replace(json_pair, monkeypatch):
    path, _, old, new, clock = json_pair
    attempts = []

    def replace(source, destination):
        attempts.append((source, destination))
        clock.now += 2.1
        raise windows_lock(5)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError):
        training._write_json(path, new)
    assert len(attempts) == 1 and not clock.sleeps
    assert json.loads(path.read_text(encoding="utf-8")) == old


def test_cancellation_during_retry_propagates_and_preserves_evidence(json_pair, monkeypatch):
    path, temporary, old, new, _ = json_pair
    attempts = []

    def replace(source, destination):
        attempts.append((source, destination))
        raise windows_lock(32)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(training.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training._write_json(path, new)
    assert len(attempts) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == old
    assert json.loads(temporary.read_text(encoding="utf-8")) == new


def test_invalid_payload_does_not_replace_existing_valid_json(json_pair, monkeypatch):
    path, temporary, old, _, clock = json_pair
    monkeypatch.setattr(
        Path, "replace", lambda *_args: pytest.fail("invalid JSON must never be published")
    )
    with pytest.raises(ValueError):
        training._write_json(path, {"loss": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == old
    assert not temporary.exists() and not clock.sleeps


@pytest.mark.skipif(sys.platform != "win32", reason="real Windows non-delete-sharing reader")
def test_real_windows_reader_hold_then_release_keeps_atomic_json(tmp_path, monkeypatch):
    import ctypes
    import threading
    from ctypes import wintypes

    path = tmp_path / "summary.json"
    old, new = {"updates": 44}, {"updates": 45}
    path.write_text(json.dumps(old), encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Read/write sharing is permitted; deletion/replacement sharing is withheld.
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    released = threading.Event()
    release_lock = threading.Lock()

    def release():
        with release_lock:
            if not released.is_set():
                if not kernel.CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                released.set()

    timer = threading.Timer(0.08, release)
    failures = []
    original_replace = Path.replace

    def replace(source, destination):
        try:
            return original_replace(source, destination)
        except PermissionError as exc:
            failures.append(exc.winerror)
            assert json.loads(path.read_text(encoding="utf-8")) == old
            assert json.loads(source.read_text(encoding="utf-8")) == new
            if len(failures) == 1:
                timer.start()
            raise

    monkeypatch.setattr(Path, "replace", replace)
    try:
        training._write_json(path, new)
        assert released.is_set() and 1 <= len(failures) <= 7
        assert all(code in {5, 32, 33} for code in failures)
        assert json.loads(path.read_text(encoding="utf-8")) == new
        assert not path.with_suffix(".json.tmp").exists()
    finally:
        timer.cancel()
        release()
        if timer.ident is not None:
            timer.join(timeout=1)
