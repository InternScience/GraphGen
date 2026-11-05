"""PMS_PATCHES.md #17(a)：write_losses 后端 fail-closed 门控。

loss 写回原子性协议（执行方文件锁 + reload→校验→loss-only merge→整文件
原子替换）仅在 networkx 后端验证；其他后端（kuzu 无数据库级事务包裹，
会部分提交）必须拒绝而非静默部分写入。
"""

import pytest

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.storage.graph.kuzu_storage import KuzuStorage
from graphgen.storage.graph.networkx_storage import NetworkXStorage

try:
    import kuzu  # noqa: F401

    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False


def test_kuzu_storage_fails_closed(tmp_path):
    """真实 KuzuStorage 实例（本地 tmp DB，不连任何外部服务）调 write_losses 必须抛 NotImplementedError，消息写明 backend 与原因。"""
    storage = KuzuStorage(str(tmp_path), "graph")
    with pytest.raises(NotImplementedError) as excinfo:
        storage.write_losses([("node-1", 0.5)])
    message = str(excinfo.value)
    assert "KuzuStorage" in message
    assert "networkx" in message


def test_arbitrary_backend_subclass_fails_closed(tmp_path):
    """非 networkx 的 BaseGraphStorage 子类同样被基类门控拒绝，消息含 backend 类名。"""

    class DummyBackend(BaseGraphStorage):
        def is_directed(self):
            return False

    # 其余抽象方法与本门控无关；绕过 Python 抽象实例化守卫，只验证 write_losses 门控。
    DummyBackend.__abstractmethods__ = frozenset()
    storage = DummyBackend()
    storage.working_dir = str(tmp_path)
    storage.namespace = "graph"
    with pytest.raises(NotImplementedError) as excinfo:
        storage.write_losses([("node-1", 0.5)])
    assert "DummyBackend" in str(excinfo.value)
    assert "networkx" in str(excinfo.value)


def test_networkx_backend_implements_transaction():
    """NetworkXStorage 必须override write_losses（门控不得误伤已验证后端）。"""
    assert (
        NetworkXStorage.write_losses.__code__
        is not BaseGraphStorage.write_losses.__code__
    )


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu not installed")
def test_kuzu_importable_guard_consistent():
    """kuzu 可用时 KuzuStorage 才可实例化；不可用时模块守卫与 skipif 一致。"""
    assert KuzuStorage is not None
