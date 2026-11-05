"""Judge/graphml 传输可靠性回归（PMS_PATCHES.md #16）。

覆盖独立功能审查发现的三处缺陷：
1. ``process()`` 对 run_concurrent 的 None 占位（输入行缺 description 等异常
   逃出内层 try 时产生）必须跳过，而不是 AttributeError 崩整批；
2. ``split()`` 缓存恢复行的边 index（历史 KV 缓存为 list）必须归一到对外
   JSON 字符串口径，避免与新产出混出 str/list 混型列炸 Arrow；
3. ``write_nx_graph`` 序列化失败时清理残留 tmp 文件。
"""

import json

import pandas as pd
import pytest

from graphgen.operators.judge.judge_service import JudgeService
from graphgen.utils import logger


class FakeKV:
    """最小 KV 存储：仅支持 get_by_id("_meta_forward") 与 get_by_ids。"""

    def __init__(self, meta_forward, rows):
        self._meta_forward = meta_forward
        self._rows = rows

    def get_by_id(self, key):
        return self._meta_forward if key == "_meta_forward" else None

    def get_by_ids(self, ids):
        return [self._rows.get(i) for i in ids]


class FakeGraph:
    """最小图存储：记录 update 调用，供写回断言。"""

    def __init__(self):
        self.nodes = {"n1": {"description": "node desc"}}
        self.edges = {("a", "b"): {"description": "edge desc"}}
        self.updated = []

    def reload(self):
        return None

    def index_done_callback(self):
        return None

    def get_node(self, key):
        data = self.nodes.get(key)
        return dict(data) if data else None

    def get_edge(self, src, tgt):
        data = self.edges.get((src, tgt))
        return dict(data) if data else None

    def update_node(self, key, data):
        self.nodes[key] = data
        self.updated.append((("node", key), data.get("loss")))

    def update_edge(self, src, tgt, data):
        self.edges[(src, tgt)] = data
        self.updated.append(((("edge", src, tgt)), data.get("loss")))


class FakeLLM:
    async def generate_topk_per_token(self, _prompt):
        class Candidate:
            top_candidates = [0.9, 0.1]

        return [Candidate()]


def make_service(meta_forward=None, rows=None):
    svc = JudgeService.__new__(JudgeService)
    svc.op_name = "judge"
    svc.working_dir = "cache"
    svc.kv_storage = FakeKV(meta_forward or {}, rows or {})
    svc.graph_storage = FakeGraph()
    svc.llm_client = FakeLLM()
    svc.logger = logger
    return svc


def test_process_skips_none_placeholders():
    """缺 description 的行 → run_concurrent None 占位 → process 跳过不崩。"""
    svc = make_service()
    batch = [
        # 无 description：KeyError 逃出内层 try，run_concurrent 留 None 占位
        {"_trace_id": "t-missing", "index": "n1"},
        {"_trace_id": "t-ok", "index": ("a", "b"), "description": "edge desc"},
    ]
    results, meta_update = svc.process(batch)

    assert results[0] is None
    assert isinstance(results[1]["index"], str)
    assert json.loads(results[1]["index"]) == ["a", "b"]
    assert results[1]["loss"] != 0.0
    assert set(meta_update) == {"t-ok"}


def test_split_normalizes_recovered_edge_index():
    """历史 KV 缓存的 list 边 index：写回按 tuple 落图，对外产出归一 JSON str。"""
    rows = {
        "unit-1": {
            "index": ["a", "b"],
            "description": "cached edge",
            "loss": 0.5,
            "_trace_id": "unit-1",
        }
    }
    svc = make_service(meta_forward={"t-src": ["unit-1"]}, rows=rows)
    batch = pd.DataFrame(
        [{"_trace_id": "t-src", "index": ["a", "b"], "description": "cached edge"}]
    )

    to_process, recovered = svc.split(batch)

    assert to_process.empty
    value = recovered.at[0, "index"]
    assert isinstance(value, str)
    assert json.loads(value) == ["a", "b"]
    # 写回仍按解码后的二元组分派落图
    assert (("edge", "a", "b"), 0.5) in svc.graph_storage.updated


def test_write_nx_graph_cleans_tmp_on_failure(tmp_path, monkeypatch):
    """序列化失败时不得残留 *.tmp（大图反复失败会占满磁盘）。"""
    import networkx as nx

    from graphgen.storage.graph.networkx_storage import NetworkXStorage

    graph = nx.Graph()
    graph.add_node("a")
    target = str(tmp_path / "graph.graphml")

    def boom(_graph, path, *_args, **_kwargs):
        # 模拟半写：先创建文件再抛异常，未修复时会残留 *.tmp
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<graphml partial")
        raise RuntimeError("serialize failed")

    monkeypatch.setattr(nx, "write_graphml", boom)
    with pytest.raises(RuntimeError):
        NetworkXStorage.write_nx_graph(graph, target)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers


def test_write_nx_graph_atomic_replace(tmp_path):
    """正常路径：目标文件完整落盘，无 tmp 残留。"""
    import networkx as nx

    from graphgen.storage.graph.networkx_storage import NetworkXStorage

    graph = nx.Graph()
    graph.add_node("a", description="d")
    target = str(tmp_path / "graph.graphml")

    NetworkXStorage.write_nx_graph(graph, target)

    assert tmp_path.joinpath("graph.graphml").is_file()
    assert not [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
