import inspect
import logging
import os
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Set

import ray
import ray.data
from ray.data import DataContext
from ray.data.block import Block
from ray.data.datasource.filename_provider import FilenameProvider

from graphgen.bases import Config, Node
from graphgen.common import init_llm, init_storage
from graphgen.utils import logger


class NodeFilenameProvider(FilenameProvider):
    def __init__(self, node_id: str):
        self.node_id = node_id

    def get_filename_for_block(
        self, block: Block, write_uuid: str, task_index: int, block_index: int
    ) -> str:
        # format: {node_id}_{write_uuid}_{task_index:06}_{block_index:06}.jsonl
        return f"{self.node_id}_{write_uuid}_{task_index:06d}_{block_index:06d}.jsonl"

    def get_filename_for_row(
        self,
        row: Dict[str, Any],
        write_uuid: str,
        task_index: int,
        block_index: int,
        row_index: int,
    ) -> str:
        raise NotImplementedError(
            f"Row-based filenames are not supported by write_json. "
            f"Node: {self.node_id}, write_uuid: {write_uuid}"
        )


class Engine:
    def __init__(
        self, config: Dict[str, Any], functions: Dict[str, Callable], **ray_init_kwargs
    ):
        self.config = Config(**config)
        self.global_params = self.config.global_params
        self.functions = functions
        self.datasets: Dict[str, ray.data.Dataset] = {}
        self.llm_actors = {}
        self.storage_actors = {}

        ctx = DataContext.get_current()
        ctx.enable_rich_progress_bars = False
        ctx.use_ray_tqdm = False
        # Disable tensor extension casting to avoid conversion errors with complex types
        # (e.g., gene_synonyms, gene_names which are lists/arrays)
        ctx.enable_tensor_extension_casting = False
        ctx._metrics_export_port = 0  # Disable metrics exporter to avoid RpcError

        all_env_vars = os.environ.copy()
        if "runtime_env" not in ray_init_kwargs:
            ray_init_kwargs["runtime_env"] = {}

        existing_env_vars = ray_init_kwargs["runtime_env"].get("env_vars", {})
        ray_init_kwargs["runtime_env"]["env_vars"] = {
            **all_env_vars,
            **existing_env_vars,
        }

        if not ray.is_initialized():
            context = ray.init(
                ignore_reinit_error=True,
                logging_level=logging.ERROR,
                log_to_driver=True,
                **ray_init_kwargs,
            )
            logger.info("Ray Dashboard URL: %s", context.dashboard_url)

        self._init_llms()
        self._init_storage()

    def _init_llms(self):
        self.llm_actors["synthesizer"] = init_llm("synthesizer")
        self.llm_actors["trainee"] = init_llm("trainee")

    def _init_storage(self):
        kv_namespaces, graph_namespaces = self._scan_storage_requirements()
        working_dir = self.global_params["working_dir"]

        for node_id in kv_namespaces:
            proxy = init_storage(self.global_params["kv_backend"], working_dir, node_id)
            self.storage_actors[f"kv_{node_id}"] = proxy
            logger.info("Create KV Storage Actor: namespace=%s", node_id)

        for ns in graph_namespaces:
            proxy = init_storage(self.global_params["graph_backend"], working_dir, ns)
            self.storage_actors[f"graph_{ns}"] = proxy
            logger.info("Create Graph Storage Actor: namespace=%s", ns)

    def _scan_storage_requirements(self) -> tuple[set[str], set[str]]:
        kv_namespaces = set()
        graph_namespaces = set()

        for node in self.config.nodes:
            op_name = node.op_name
            if self._function_needs_param(op_name, "kv_backend"):
                kv_namespaces.add(op_name)
            if self._function_needs_param(op_name, "graph_backend"):
                graph_namespaces.add("graph")
        return kv_namespaces, graph_namespaces

    def _function_needs_param(self, op_name: str, param_name: str) -> bool:
        if op_name not in self.functions:
            return False

        func = self.functions[op_name]

        if inspect.isclass(func):
            try:
                sig = inspect.signature(func.__init__)
                return param_name in sig.parameters
            except (ValueError, TypeError):
                return False

        try:
            sig = inspect.signature(func)
            return param_name in sig.parameters
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _topo_sort(nodes: List[Node]) -> List[Node]:
        id_to_node: Dict[str, Node] = {}
        for n in nodes:
            id_to_node[n.id] = n

        indeg: Dict[str, int] = {nid: 0 for nid in id_to_node}
        adj: Dict[str, List[str]] = defaultdict(list)

        for n in nodes:
            nid = n.id
            deps: List[str] = n.dependencies
            uniq_deps: Set[str] = set(deps)
            for d in uniq_deps:
                if d not in id_to_node:
                    raise ValueError(
                        f"The dependency node id {d} of node {nid} is not defined in the configuration."
                    )
                indeg[nid] += 1
                adj[d].append(nid)

        zero_deg: deque = deque(
            [id_to_node[nid] for nid, deg in indeg.items() if deg == 0]
        )
        sorted_nodes: List[Node] = []

        while zero_deg:
            cur = zero_deg.popleft()
            sorted_nodes.append(cur)
            cur_id = cur.id
            for nb_id in adj.get(cur_id, []):
                indeg[nb_id] -= 1
                if indeg[nb_id] == 0:
                    zero_deg.append(id_to_node[nb_id])

        if len(sorted_nodes) != len(nodes):
            remaining = [nid for nid, deg in indeg.items() if deg > 0]
            raise ValueError(
                f"The configuration contains cycles, unable to execute. Remaining nodes with indegree > 0: {remaining}"
            )

        return sorted_nodes

    def _get_input_dataset(
        self, node: Node, initial_ds: ray.data.Dataset
    ) -> ray.data.Dataset:
        deps = node.dependencies

        if not deps:
            return initial_ds

        if len(deps) == 1:
            return self.datasets[deps[0]]

        main_ds = self.datasets[deps[0]]
        other_dss = [self.datasets[d] for d in deps[1:]]
        return main_ds.union(*other_dss)

    def _execute_node(self, node: Node, initial_ds: ray.data.Dataset):
        def _filter_kwargs(
            func_or_class: Callable,
            global_params: Dict[str, Any],
            func_params: Dict[str, Any],
        ) -> Dict[str, Any]:
            """
            1. global_params: only when specified in function signature, will be passed
            2. func_params: pass specified params first, then **kwargs if exists
            """
            try:
                sig = inspect.signature(func_or_class)
            except ValueError:
                return {}

            params = sig.parameters
            final_kwargs = {}

            has_var_keywords = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            valid_keys = set(params.keys())
            for k, v in global_params.items():
                if k in valid_keys:
                    final_kwargs[k] = v

            for k, v in func_params.items():
                if k in valid_keys or has_var_keywords:
                    final_kwargs[k] = v
            return final_kwargs

        if node.op_name not in self.functions:
            raise ValueError(f"Operator {node.op_name} not found for node {node.id}")

        op_handler = self.functions[node.op_name]
        node_params = _filter_kwargs(op_handler, self.global_params, node.params or {})

        if node.type == "source":
            self.datasets[node.id] = op_handler(**node_params)
            return

        input_ds = self._get_input_dataset(node, initial_ds)

        # if inspect.isclass(op_handler):
        execution_params = node.execution_params or {}
        replicas = execution_params.get("replicas", 1)
        batch_size = (
            int(execution_params.get("batch_size"))
            if "batch_size" in execution_params
            else "default"
        )
        compute_resources = execution_params.get("compute_resources", {})

        if node.type == "aggregate":
            self.datasets[node.id] = input_ds.repartition(1).map_batches(
                op_handler,
                compute=ray.data.ActorPoolStrategy(min_size=1, max_size=1),
                batch_size=None,  # aggregate processes the whole dataset at once
                num_gpus=compute_resources.get("num_gpus", 0)
                if compute_resources
                else 0,
                fn_constructor_kwargs=node_params,
                batch_format="pandas",
            )
        else:
            self.datasets[node.id] = input_ds.map_batches(
                op_handler,
                compute=ray.data.ActorPoolStrategy(min_size=1, max_size=replicas),
                batch_size=batch_size,
                num_gpus=compute_resources.get("num_gpus", 0)
                if compute_resources
                else 0,
                fn_constructor_kwargs=node_params,
                batch_format="pandas",
            )

    def execute(
        self, initial_ds: ray.data.Dataset, output_dir: str
    ) -> Dict[str, ray.data.Dataset]:
        sorted_nodes = self._topo_sort(self.config.nodes)

        for node in sorted_nodes:
            logger.info("Executing node %s of type %s", node.id, node.type)
            self._execute_node(node, initial_ds)
            if getattr(node, "save_output", False):
                node_output_path = os.path.join(output_dir, f"{node.id}")
                os.makedirs(node_output_path, exist_ok=True)
                logger.info("Saving output of node %s to %s", node.id, node_output_path)

                ds = self.datasets[node.id]
                ds.write_json(
                    node_output_path,
                    filename_provider=NodeFilenameProvider(node.id),
                    pandas_json_args_fn=lambda: {
                        "orient": "records",
                        "lines": True,
                        "force_ascii": False,
                    },
                )
                logger.info("Node %s output saved to %s", node.id, node_output_path)

                # ray will lazy read the dataset
                if os.path.exists(node_output_path) and os.listdir(node_output_path):
                    self.datasets[node.id] = ray.data.read_json(node_output_path)
                else:
                    self.datasets[node.id] = ray.data.from_items([])
                    logger.warning(
                        "Node %s output path %s is empty. Created an empty dataset.",
                        node.id,
                        node_output_path,
                    )

        return self.datasets
