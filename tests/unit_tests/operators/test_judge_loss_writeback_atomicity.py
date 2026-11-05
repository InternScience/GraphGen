"""真实 JudgeService + NetworkX 文件回归：缓存/新判、共享/独立实例及跨进程。

最终协议为执行方 write_losses 单次 reload→loss-only merge→flush；judge
自身不持锁、不预先 reload。LLM 测试替身只返回确定性 Token，不访问模型。
真实 Ray 的 reload 插队和 caller-kill 交错另见 integration_tests。
"""
import math
import multiprocessing
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from graphgen.bases.datatypes import Token
from graphgen.operators.judge.judge_service import JudgeService
from graphgen.storage.graph.networkx_storage import NetworkXStorage
from graphgen.utils import logger


class FixedLLM:
    def __init__(self, probability, hook=None):
        self.probability, self.hook = probability, hook

    async def generate_topk_per_token(self, _prompt):
        if self.hook:
            self.hook()
        return [SimpleNamespace(top_candidates=[
            Token("yes", self.probability), Token("no", 1 - self.probability)
        ])]


class Cache:
    def get_by_id(self, key):
        return {"input": ["cached"]} if key == "_meta_forward" else None

    def get_by_ids(self, ids):
        return [{"index": "n1", "description": "cached", "loss": 0.5} for _ in ids]


def seed(directory):
    graph = NetworkXStorage(str(directory), "graph")
    for node in ("n1", "n2"):
        graph.upsert_node(node, {"description": f"desc {node}", "length": 10})
    graph.upsert_edge("n1", "n2", {"description": "edge", "length": 8})
    graph.index_done_callback()
    return graph


def service(directory, storage, probability=0.9, hook=None):
    result = JudgeService.__new__(JudgeService)
    result.op_name = "judge"
    result.working_dir = str(directory)
    result.graph_storage = storage
    result.llm_client = FixedLLM(probability, hook)
    result.kv_storage = Cache()
    result.logger = logger
    return result


def batch(index):
    return [{"index": index, "description": "description", "_trace_id": "input"}]


def losses(directory):
    fresh = NetworkXStorage(str(directory), "graph")
    return {node: data.get("loss") for node, data in fresh.get_all_nodes()}


def test_cached_writeback_during_judging_window_survives(tmp_path):
    seed(tmp_path)
    cached = service(tmp_path, NetworkXStorage(str(tmp_path), "graph"))
    storage = NetworkXStorage(str(tmp_path), "graph")

    def cache_hook():
        pending, recovered = cached.split(pd.DataFrame(batch("n1")))
        assert pending.empty and len(recovered) == 1

    service(tmp_path, storage, 0.3, cache_hook).process(batch("n2"))
    assert losses(tmp_path) == pytest.approx({"n1": 0.5, "n2": -math.log(0.3)})


@pytest.mark.parametrize("shared", [False, True])
def test_threaded_new_judgements_keep_all_losses(tmp_path, shared):
    seed(tmp_path)
    a = NetworkXStorage(str(tmp_path), "graph")
    b = a if shared else NetworkXStorage(str(tmp_path), "graph")
    in_llm, other_flushed = threading.Event(), threading.Event()
    errors = []

    def hook():
        in_llm.set()
        assert other_flushed.wait(30)

    def run():
        try:
            service(tmp_path, a, 0.9, hook).process(batch("n1"))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert in_llm.wait(30)
        service(tmp_path, b, 0.3).process(batch("n2"))
    finally:
        other_flushed.set()
        thread.join(30)
    assert not thread.is_alive() and not errors
    assert losses(tmp_path) == pytest.approx({"n1": -math.log(0.9), "n2": -math.log(0.3)})


def _child_judge(directory, in_llm, other_flushed):
    storage = NetworkXStorage(directory, "graph")

    def hook():
        in_llm.set()
        assert other_flushed.wait(30)

    service(directory, storage, 0.9, hook).process(batch("n1"))


def test_cross_process_runs_both_losses_persist(tmp_path):
    seed(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    in_llm, other_flushed = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_child_judge, args=(str(tmp_path), in_llm, other_flushed))
    child.start()
    try:
        assert in_llm.wait(30)
        service(tmp_path, NetworkXStorage(str(tmp_path), "graph"), 0.3).process(batch("n2"))
    finally:
        other_flushed.set()
        child.join(30)
        if child.is_alive():
            child.kill()
            child.join()
    assert child.exitcode == 0
    assert losses(tmp_path) == pytest.approx({"n1": -math.log(0.9), "n2": -math.log(0.3)})


def test_loss_only_merge_preserves_node_and_edge_attributes(tmp_path):
    graph = seed(tmp_path)
    assert graph.write_losses([("n1", 0.2), (("n1", "n2"), 0.7)]) == 2
    fresh = NetworkXStorage(str(tmp_path), "graph")
    assert fresh.get_node("n1") == {"description": "desc n1", "length": 10, "loss": 0.2}
    assert fresh.get_edge("n1", "n2") == {"description": "edge", "length": 8, "loss": 0.7}


def test_missing_target_cannot_silently_report_success(tmp_path):
    graph = seed(tmp_path)
    with pytest.raises(KeyError, match="target missing"):
        graph.write_losses([("n1", 0.2), ("missing", 0.7)])
    assert losses(tmp_path) == {"n1": None, "n2": None}
    assert graph.get_node("n1").get("loss") is None


@pytest.mark.parametrize("cached", [False, True])
def test_flush_failure_is_not_published_as_success(tmp_path, monkeypatch, cached):
    graph = seed(tmp_path)
    svc = service(tmp_path, graph)

    def fail():
        raise OSError("injected disk failure")

    monkeypatch.setattr(graph, "index_done_callback", fail)
    with pytest.raises(OSError, match="injected disk failure"):
        if cached:
            svc.split(pd.DataFrame(batch("n1")))
        else:
            svc.process(batch("n1"))
    assert losses(tmp_path) == {"n1": None, "n2": None}
