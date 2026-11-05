"""真实 Ray/proxy/JudgeService 回归（只允许显式指定独立测试集群）。

运行前设 GRAPHGEN_TEST_RAY_ADDRESS，禁止 address=auto，禁止连接生产集群。
测试通过子类/代理插桩编排危险交错，不向生产实现添加测试钩子。
"""
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pandas as pd
import pytest
import ray

from graphgen.bases.datatypes import Token
from graphgen.common.init_storage import (
    GraphStorageActor,
    RemoteGraphStorageProxy,
    StorageFactory,
    init_storage,
)
from graphgen.operators.judge.judge_service import JudgeService
from graphgen.storage.graph.networkx_storage import NetworkXStorage
from graphgen.utils import logger


ROOT = str(Path(__file__).resolve().parents[2])
LOSS = -__import__("math").log(0.9)


def wait_file(path, timeout=40):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"barrier not reached: {path}")
        time.sleep(0.01)


def seed(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    graph = NetworkXStorage(str(directory), "graph")
    for node in ("n1", "n2"):
        graph.upsert_node(node, {"description": f"desc {node}", "length": 10})
    graph.upsert_edge("n1", "n2", {"description": "edge desc", "length": 8})
    graph.index_done_callback()


def disk_loss(directory, index):
    graph = nx.read_graphml(Path(directory) / "graph.graphml")
    data = graph.nodes[index] if isinstance(index, str) else graph.edges[index]
    return data.get("loss")


class ControlledGraphActor(GraphStorageActor):
    """执行器内部插桩：暂停在 update 已成功、或快照已序列化但未 replace 时。"""

    def arm(self, mode, entered, release):
        if mode == "update":
            original_node = self.graph.update_node
            original_edge = self.graph.update_edge

            def pause():
                Path(entered).touch()
                wait_file(release)

            def update_node(*args):
                original_node(*args)
                pause()

            def update_edge(*args):
                original_edge(*args)
                pause()

            self.graph.update_node = update_node
            self.graph.update_edge = update_edge
        else:
            original_write = nx.write_graphml

            def write_snapshot(*args, **kwargs):
                original_write(*args, **kwargs)
                Path(entered).touch()
                wait_file(release)

            nx.write_graphml = write_snapshot


class GatedProxy(RemoteGraphStorageProxy):
    """基线时强制 reader.reload 完成后再 flush；最终单 RPC 不经过此门。

    actor 内 update 暂停保证 reader RPC 已排队；基线 update RPC 返回后，
    此代理让 judge 等 reader 完成才调用 flush，避免依靠调度概率复现。
    """

    def __init__(self, handle, reader_done=None):
        super().__init__(handle)
        self.reader_done = reader_done

    def update_node(self, *args):
        result = super().update_node(*args)
        if self.reader_done:
            wait_file(self.reader_done)
        return result

    def update_edge(self, *args):
        result = super().update_edge(*args)
        if self.reader_done:
            wait_file(self.reader_done)
        return result


class FixedLLM:
    async def generate_topk_per_token(self, _prompt):
        return [SimpleNamespace(top_candidates=[Token("yes", 0.9), Token("no", 0.1)])]


class Cache:
    def __init__(self, record):
        self.record = record

    def get_by_id(self, key):
        return {"input": ["cached"]} if key == "_meta_forward" else None

    def get_by_ids(self, ids):
        return [self.record for _key in ids]


class JudgeCaller:
    def run(self, directory, proxy, index, cached=False, started=None):
        if started:
            Path(started).touch()
        service = JudgeService.__new__(JudgeService)
        service.op_name = "judge"
        service.working_dir = str(directory)
        service.graph_storage = proxy
        service.llm_client = FixedLLM()
        service.logger = logger
        item = {"index": index, "description": "a genuine description", "_trace_id": "input"}
        if cached:
            service.kv_storage = Cache(dict(item, loss=LOSS))
            _pending, recovered = service.split(pd.DataFrame([item]))
            return len(recovered)
        results, _meta = service.process([item])
        return results[0]["loss"]


def read_stage(directory, stage, started, done):
    Path(started).touch()
    try:
        if stage == "graph_units":
            from graphgen.operators.graph_units.graph_units import graph_units

            # 真实 source、真实 factory/proxy/reload 和 Ray Data 输出。
            return len(graph_units(str(directory)).take_all())
        from graphgen.models import BFSPartitioner
        from graphgen.operators.partition.partition_service import PartitionService

        service = PartitionService.__new__(PartitionService)
        service.working_dir = str(directory)
        service.op_name = "partition"
        service.kg_instance = init_storage("networkx", str(directory), "graph")
        service.partitioner = BFSPartitioner()
        service.method_params = {}
        service.max_communities = None
        result, _meta = service.process([])
        return len(list(result))
    finally:
        Path(done).touch()


@pytest.fixture(scope="module", autouse=True)
def isolated_ray():
    address = os.environ.get("GRAPHGEN_TEST_RAY_ADDRESS")
    if not address:
        pytest.skip("requires explicit dedicated GRAPHGEN_TEST_RAY_ADDRESS")
    assert address != "auto", "never auto-connect to a shared cluster"
    assert not ray.is_initialized()

    def worker_setup():
        # 某些测试环境的默认 worker cwd=/，会让 NLTK 将全磁盘误视为 cwd。
        os.chdir(ROOT)

    paths = [ROOT, str(Path(__file__).parent)] + [p for p in __import__("sys").path if p]
    ray.init(
        address=address,
        namespace=f"loss-regression-{uuid.uuid4().hex}",
        log_to_driver=False,
        runtime_env={
            "worker_process_setup_hook": worker_setup,
            "env_vars": {"PYTHONPATH": os.pathsep.join(paths), "PYTHONSAFEPATH": "1"},
        },
    )
    yield
    ray.shutdown()


def named_controlled(directory):
    # 兼容未修复基线，以同一真实 factory actor 名插桩，非替换 factory。
    if hasattr(StorageFactory, "_actor_name"):
        name = StorageFactory._actor_name("Graph", "networkx", str(directory), "graph")
    else:
        name = "Actor_Graph_graph"
    return ray.remote(ControlledGraphActor).options(name=name, num_cpus=0).remote(
        "networkx", str(directory), "graph"
    )


@pytest.mark.parametrize("kv_backend", ["json_kv", "rocksdb"])
def test_real_factory_isolates_directories_and_reuses_path_aliases(tmp_path, kv_backend):
    a, b = tmp_path / "a", tmp_path / "b"
    seed(a)
    seed(b)
    ga = init_storage("networkx", str(a), "graph")
    gb = init_storage("networkx", str(b), "graph")
    alias = tmp_path / "alias"
    alias.symlink_to(a, target_is_directory=True)
    same = init_storage("networkx", str(alias), "graph")
    try:
        assert ga.actor._actor_id != gb.actor._actor_id
        assert same.actor._actor_id == ga.actor._actor_id
        ga.upsert_node("only-a", {"description": "a"})
        gb.upsert_node("only-b", {"description": "b"})
        ga.index_done_callback()
        gb.index_done_callback()
        assert "only-a" in nx.read_graphml(a / "graph.graphml")
        assert "only-a" not in nx.read_graphml(b / "graph.graphml")
        ka = init_storage(kv_backend, str(a), "judge")
        kb = init_storage(kv_backend, str(b), "judge")
        assert ka.actor._actor_id != kb.actor._actor_id
        ka.upsert({"record": {"value": "a"}})
        kb.upsert({"record": {"value": "b"}})
        ka.index_done_callback()
        kb.index_done_callback()
        assert ka.get_by_id("record") == {"value": "a"}
        assert kb.get_by_id("record") == {"value": "b"}
        ray.kill(ka.actor)
        ray.kill(kb.actor)
    finally:
        ray.kill(ga.actor)
        if gb.actor._actor_id != ga.actor._actor_id:
            ray.kill(gb.actor)


@pytest.mark.parametrize("stage", ["graph_units", "partition"])
@pytest.mark.parametrize("cached", [False, True])
def test_real_judge_cache_fresh_and_reader_reload_do_not_lose_losses(tmp_path, stage, cached):
    seed(tmp_path)
    entered, release = tmp_path / "updated", tmp_path / "release"
    reader_started, reader_done = tmp_path / "reader-started", tmp_path / "reader-done"
    storage = named_controlled(tmp_path)
    ray.get(storage.arm.remote("update", str(entered), str(release)), timeout=40)
    caller_a = ray.remote(JudgeCaller).options(num_cpus=0).remote()
    caller_b = ray.remote(JudgeCaller).options(num_cpus=0).remote()
    index = ["n1", "n2"] if cached else "n1"
    proxy = GatedProxy(storage, str(reader_done))
    result = caller_a.run.remote(str(tmp_path), proxy, index, cached)
    reader = None
    try:
        wait_file(entered)
        reader = ray.remote(read_stage).options(num_cpus=0).remote(
            str(tmp_path), stage, str(reader_started), str(reader_done)
        )
        wait_file(reader_started)
        # 让 reader 的真实 reload RPC 到达正在执行的 actor。
        ray.wait([reader], timeout=0.5)
        other = caller_b.run.remote(str(tmp_path), GatedProxy(storage), "n2", not cached)
        release.touch()
        assert ray.get(reader, timeout=60) > 0
        ray.get([result, other], timeout=60)
        target = tuple(index) if isinstance(index, list) else index
        assert disk_loss(tmp_path, target) == pytest.approx(LOSS)
        assert disk_loss(tmp_path, "n2") == pytest.approx(LOSS)
    finally:
        release.touch()
        ray.kill(caller_a)
        ray.kill(caller_b)
        ray.kill(storage)


@pytest.mark.parametrize("cached", [False, True])
def test_killed_judge_caller_does_not_release_executor_transaction(tmp_path, cached):
    """A 快照写好但 replace 前杀 caller；A actor 仍活着。B 独立 actor
    写同目录：必须等 A 完成后重新加载，不能先完成再被 A 的旧快照覆盖。
    """
    seed(tmp_path)
    entered, release = tmp_path / "snapshot-ready", tmp_path / "release"
    b_started = tmp_path / "b-started"
    actor_a = ray.remote(ControlledGraphActor).options(num_cpus=0).remote(
        "networkx", str(tmp_path), "graph"
    )
    actor_b = ray.remote(GraphStorageActor).options(num_cpus=0).remote(
        "networkx", str(tmp_path), "graph"
    )
    ray.get(actor_a.arm.remote("flush", str(entered), str(release)), timeout=40)
    caller_a = ray.remote(JudgeCaller).options(num_cpus=0).remote()
    caller_b = ray.remote(JudgeCaller).options(num_cpus=0).remote()
    pending = caller_a.run.remote(str(tmp_path), GatedProxy(actor_a), "n1", cached)
    try:
        wait_file(entered)
        ray.kill(caller_a, no_restart=True)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(pending, timeout=20)
        other = caller_b.run.remote(
            str(tmp_path), GatedProxy(actor_b), "n2", not cached, str(b_started)
        )
        wait_file(b_started)
        ready, _ = ray.wait([other], timeout=0.6)
        blocked_until_executor_finishes = not ready
        release.touch()
        # A 的真实 I/O 必须仍在存活的 actor 中完成（不靠 caller 存活）。
        assert ray.get(actor_a.ready.remote(), timeout=40)
        ray.get(other, timeout=40)
        assert disk_loss(tmp_path, "n1") == pytest.approx(LOSS)
        assert disk_loss(tmp_path, "n2") == pytest.approx(LOSS)
        assert blocked_until_executor_finishes
    finally:
        release.touch()
        ray.kill(caller_b)
        ray.kill(actor_a)
        ray.kill(actor_b)
