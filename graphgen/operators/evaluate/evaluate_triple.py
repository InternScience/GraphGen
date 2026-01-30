from typing import Any

from graphgen.bases import BaseKVStorage
from graphgen.utils import logger, run_concurrent


def evaluate_triple(
    triple_evaluators: dict[str, Any],
    src_storage: BaseKVStorage,
    tgt_storage: BaseKVStorage,
) -> dict[str, Any]:
    forward_meta = tgt_storage.get_by_id("_meta_forward")

    tasks = []
    for chunk_id, unit_ids in forward_meta.items():
        chunk_content = str(src_storage.get_by_id(chunk_id))

        nodes = []
        edges = []

        for unit_id in unit_ids:
            unit_data = tgt_storage.get_by_id(unit_id)
            if "node" in unit_data and unit_data["node"]:
                nodes.append(unit_data["node"])
            if "edge" in unit_data and unit_data["edge"]:
                edges.append(unit_data["edge"])

        tasks.append((chunk_content, nodes, edges))

    results = {}
    for key, triple_evaluator in triple_evaluators.items():
        logger.info(f"Evaluating Triples with metric: {key}...")
        result = run_concurrent(
            triple_evaluator.evaluate,
            tasks,
            desc=f"Evaluating Triples with {key}",
        )
        results[key] = result
    return results
