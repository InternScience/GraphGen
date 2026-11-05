"""POSIX 执行方事务锁：线程/进程互斥、异常释放及不支持平台拒绝执行。"""
import multiprocessing
import threading

import pytest

from graphgen.utils import concurrency


def _hold(path, acquired, release):
    with concurrency.file_lock(path):
        acquired.set()
        assert release.wait(20)


def test_path_aliases_share_lock(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    assert concurrency.graph_lock_path(str(real)) == concurrency.graph_lock_path(str(alias))


def test_thread_serialization(tmp_path):
    entered = threading.Event()
    path = str(tmp_path / "lock")
    with concurrency.file_lock(path):
        contender = threading.Thread(target=lambda: _hold(path, entered, entered))
        contender.start()
        assert not entered.wait(0.2)
    contender.join(20)
    assert entered.is_set()
    assert not contender.is_alive()


def test_process_serialization(tmp_path):
    import fcntl

    ctx = multiprocessing.get_context("spawn")
    acquired, release = ctx.Event(), ctx.Event()
    path = str(tmp_path / "lock")
    child = ctx.Process(target=_hold, args=(path, acquired, release))
    child.start()
    try:
        assert acquired.wait(20)
        with open(path, "a+", encoding="utf-8") as fh:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release.set()
        child.join(20)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0


def test_exception_releases_lock(tmp_path):
    path = str(tmp_path / "lock")
    with pytest.raises(ValueError):
        with concurrency.file_lock(path):
            raise ValueError("injected")
    with concurrency.file_lock(path):
        pass


def test_unsupported_platform_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(concurrency, "fcntl", None)
    with pytest.raises(RuntimeError, match="require POSIX"):
        with concurrency.file_lock(str(tmp_path / "lock")):
            pytest.fail("unsupported platform silently entered transaction")
