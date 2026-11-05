"""图批量 loss 写回的执行方互斥（POSIX 本地文件系统）。

锁必须由实际执行 reload/update/flush 的进程持有，而非等待 RPC 的调用方。
调用方死亡不会提前释放执行方的锁；同一 Ray graph actor 内的其他方法由
官方同步 actor 调度串行，独立 actor/运行之间由此文件锁串行。
"""

import os
import threading
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # 非 POSIX 必须拒绝事务，不能静默降级成仅线程锁
    fcntl = None

_locks_guard = threading.Lock()
_locks = {}


def graph_lock_path(working_dir: str, namespace: str = "graph") -> str:
    return os.path.join(os.path.realpath(working_dir), f".{namespace}.graph.lock")


@contextmanager
def file_lock(path: str):
    if fcntl is None:
        raise RuntimeError("graph loss transactions require POSIX fcntl.flock")
    key = os.path.realpath(path)
    with _locks_guard:
        local_lock = _locks.setdefault(key, threading.Lock())
    with local_lock:
        os.makedirs(os.path.dirname(key), exist_ok=True)
        # 不删除锁文件：unlink 会让等待者与后来者锁定不同 inode。
        with open(key, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def graph_write_lock(working_dir: str, namespace: str = "graph"):
    with file_lock(graph_lock_path(working_dir, namespace)):
        yield
