from typing import Any, Dict, List, Set, Union

import ray

from graphgen.bases.base_storage import BaseGraphStorage, BaseKVStorage


class KVStorageActor:
    def __init__(self, backend: str, working_dir: str, namespace: str):
        if backend == "json_kv":
            from graphgen.storage import JsonKVStorage

            self.kv = JsonKVStorage(working_dir, namespace)
        elif backend == "rocksdb":
            from graphgen.storage import RocksDBKVStorage

            self.kv = RocksDBKVStorage(working_dir, namespace)
        else:
            raise ValueError(f"Unknown KV backend: {backend}")

    def data(self) -> Dict[str, Dict]:
        return self.kv.data

    def all_keys(self) -> list[str]:
        return self.kv.all_keys()

    def index_done_callback(self):
        return self.kv.index_done_callback()

    def get_by_id(self, id: str) -> Dict:
        return self.kv.get_by_id(id)

    def get_by_ids(self, ids: list[str], fields=None) -> list:
        return self.kv.get_by_ids(ids, fields)

    def get_all(self) -> Dict[str, Dict]:
        return self.kv.get_all()

    def filter_keys(self, data: list[str]) -> set[str]:
        return self.kv.filter_keys(data)

    def upsert(self, data: dict) -> dict:
        return self.kv.upsert(data)

    def update(self, data: dict):
        return self.kv.update(data)

    def delete(self, ids: list[str]):
        return self.kv.delete(ids)

    def drop(self):
        return self.kv.drop()

    def reload(self):
        return self.kv.reload()

    def ready(self) -> bool:
        return True


class GraphStorageActor:
    def __init__(self, backend: str, working_dir: str, namespace: str):
        if backend == "networkx":
            from graphgen.storage import NetworkXStorage

            self.graph = NetworkXStorage(working_dir, namespace)
        elif backend == "kuzu":
            from graphgen.storage import KuzuStorage

            self.graph = KuzuStorage(working_dir, namespace)
        else:
            raise ValueError(f"Unknown Graph backend: {backend}")

    def index_done_callback(self):
        return self.graph.index_done_callback()

    def is_directed(self) -> bool:
        return self.graph.is_directed()

    def get_all_node_degrees(self) -> Dict[str, int]:
        return self.graph.get_all_node_degrees()

    def get_node_count(self) -> int:
        return self.graph.get_node_count()

    def get_edge_count(self) -> int:
        return self.graph.get_edge_count()

    def get_connected_components(self, undirected: bool = True) -> List[Set[str]]:
        return self.graph.get_connected_components(undirected)

    def has_node(self, node_id: str) -> bool:
        return self.graph.has_node(node_id)

    def has_edge(self, source_node_id: str, target_node_id: str):
        return self.graph.has_edge(source_node_id, target_node_id)

    def node_degree(self, node_id: str) -> int:
        return self.graph.node_degree(node_id)

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self.graph.edge_degree(src_id, tgt_id)

    def get_node(self, node_id: str) -> Any:
        return self.graph.get_node(node_id)

    def update_node(self, node_id: str, node_data: dict[str, str]):
        return self.graph.update_node(node_id, node_data)

    def get_all_nodes(self) -> Any:
        return self.graph.get_all_nodes()

    def get_edge(self, source_node_id: str, target_node_id: str):
        return self.graph.get_edge(source_node_id, target_node_id)

    def update_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        return self.graph.update_edge(source_node_id, target_node_id, edge_data)

    def get_all_edges(self) -> Any:
        return self.graph.get_all_edges()

    def get_node_edges(self, source_node_id: str) -> Any:
        return self.graph.get_node_edges(source_node_id)

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        return self.graph.upsert_node(node_id, node_data)

    def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        return self.graph.upsert_edge(source_node_id, target_node_id, edge_data)

    def delete_node(self, node_id: str):
        return self.graph.delete_node(node_id)

    def get_neighbors(self, node_id: str) -> List[str]:
        return self.graph.get_neighbors(node_id)

    def reload(self):
        return self.graph.reload()

    def ready(self) -> bool:
        return True


class RemoteKVStorageProxy(BaseKVStorage):
    def __init__(self, actor_handle: ray.actor.ActorHandle):
        super().__init__()
        self.actor = actor_handle

    def data(self) -> Dict[str, Any]:
        return ray.get(self.actor.data.remote())

    def all_keys(self) -> list[str]:
        return ray.get(self.actor.all_keys.remote())

    def index_done_callback(self):
        return ray.get(self.actor.index_done_callback.remote())

    def get_by_id(self, id: str) -> Union[Any, None]:
        return ray.get(self.actor.get_by_id.remote(id))

    def get_by_ids(self, ids: list[str], fields=None) -> list[Any]:
        return ray.get(self.actor.get_by_ids.remote(ids, fields))

    def get_all(self) -> Dict[str, Any]:
        return ray.get(self.actor.get_all.remote())

    def filter_keys(self, data: list[str]) -> set[str]:
        return ray.get(self.actor.filter_keys.remote(data))

    def upsert(self, data: Dict[str, Any]):
        return ray.get(self.actor.upsert.remote(data))

    def update(self, data: Dict[str, Any]):
        return ray.get(self.actor.update.remote(data))

    def delete(self, ids: list[str]):
        return ray.get(self.actor.delete.remote(ids))

    def drop(self):
        return ray.get(self.actor.drop.remote())

    def reload(self):
        return ray.get(self.actor.reload.remote())


class RemoteGraphStorageProxy(BaseGraphStorage):
    def __init__(self, actor_handle: ray.actor.ActorHandle):
        super().__init__()
        self.actor = actor_handle

    def index_done_callback(self):
        return ray.get(self.actor.index_done_callback.remote())

    def is_directed(self) -> bool:
        return ray.get(self.actor.is_directed.remote())

    def get_all_node_degrees(self) -> Dict[str, int]:
        return ray.get(self.actor.get_all_node_degrees.remote())

    def get_node_count(self) -> int:
        return ray.get(self.actor.get_node_count.remote())

    def get_edge_count(self) -> int:
        return ray.get(self.actor.get_edge_count.remote())

    def get_connected_components(self, undirected: bool = True) -> List[Set[str]]:
        return ray.get(self.actor.get_connected_components.remote(undirected))

    def has_node(self, node_id: str) -> bool:
        return ray.get(self.actor.has_node.remote(node_id))

    def has_edge(self, source_node_id: str, target_node_id: str):
        return ray.get(self.actor.has_edge.remote(source_node_id, target_node_id))

    def node_degree(self, node_id: str) -> int:
        return ray.get(self.actor.node_degree.remote(node_id))

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return ray.get(self.actor.edge_degree.remote(src_id, tgt_id))

    def get_node(self, node_id: str) -> Any:
        return ray.get(self.actor.get_node.remote(node_id))

    def update_node(self, node_id: str, node_data: dict[str, str]):
        return ray.get(self.actor.update_node.remote(node_id, node_data))

    def get_all_nodes(self) -> Any:
        return ray.get(self.actor.get_all_nodes.remote())

    def get_edge(self, source_node_id: str, target_node_id: str):
        return ray.get(self.actor.get_edge.remote(source_node_id, target_node_id))

    def update_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        return ray.get(
            self.actor.update_edge.remote(source_node_id, target_node_id, edge_data)
        )

    def get_all_edges(self) -> Any:
        return ray.get(self.actor.get_all_edges.remote())

    def get_node_edges(self, source_node_id: str) -> Any:
        return ray.get(self.actor.get_node_edges.remote(source_node_id))

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        return ray.get(self.actor.upsert_node.remote(node_id, node_data))

    def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        return ray.get(
            self.actor.upsert_edge.remote(source_node_id, target_node_id, edge_data)
        )

    def delete_node(self, node_id: str):
        return ray.get(self.actor.delete_node.remote(node_id))

    def get_neighbors(self, node_id: str) -> List[str]:
        return ray.get(self.actor.get_neighbors.remote(node_id))

    def reload(self):
        return ray.get(self.actor.reload.remote())


class StorageFactory:
    """
    Factory class to create storage instances based on backend.
    """

    @staticmethod
    def create_storage(backend: str, working_dir: str, namespace: str):
        if backend in ["json_kv", "rocksdb"]:
            actor_name = f"Actor_KV_{namespace}"
            actor_class = KVStorageActor
            proxy_class = RemoteKVStorageProxy
        elif backend in ["networkx", "kuzu"]:
            actor_name = f"Actor_Graph_{namespace}"
            actor_class = GraphStorageActor
            proxy_class = RemoteGraphStorageProxy
        else:
            raise ValueError(f"Unknown storage backend: {backend}")
        try:
            actor_handle = ray.get_actor(actor_name)
        except ValueError:
            actor_handle = (
                ray.remote(actor_class)
                .options(
                    name=actor_name,
                    get_if_exists=True,
                )
                .remote(backend, working_dir, namespace)
            )
            ray.get(actor_handle.ready.remote())
        return proxy_class(actor_handle)


def init_storage(backend: str, working_dir: str, namespace: str):
    return StorageFactory.create_storage(backend, working_dir, namespace)
