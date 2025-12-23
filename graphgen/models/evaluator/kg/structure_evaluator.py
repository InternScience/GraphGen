from typing import Any, Dict, Optional

import networkx as nx
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

from graphgen.bases import BaseGraphStorage
from graphgen.utils import logger


def _convert_to_networkx(graph_storage: BaseGraphStorage) -> nx.DiGraph:
    """Convert graph storage to NetworkX graph."""
    G = nx.DiGraph()

    # Add nodes
    nodes = graph_storage.get_all_nodes() or []
    for node_id, node_data in nodes:
        if isinstance(node_data, dict):
            G.add_node(node_id, **node_data)
        else:
            G.add_node(node_id)

    # Add edges
    edges = graph_storage.get_all_edges() or []
    for src, dst, edge_data in edges:
        if isinstance(edge_data, dict):
            G.add_edge(src, dst, **edge_data)
        else:
            G.add_edge(src, dst)

    return G


class StructureEvaluator:
    """Evaluates structural robustness of the graph."""

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        noise_ratio_threshold: float = 0.15,
        largest_cc_ratio_threshold: float = 0.90,
        avg_degree_min: float = 2.0,
        avg_degree_max: float = 5.0,
        powerlaw_r2_threshold: float = 0.75,
    ):
        self.graph_storage = graph_storage
        self.noise_ratio_threshold = noise_ratio_threshold
        self.largest_cc_ratio_threshold = largest_cc_ratio_threshold
        self.avg_degree_min = avg_degree_min
        self.avg_degree_max = avg_degree_max
        self.powerlaw_r2_threshold = powerlaw_r2_threshold

    def evaluate(self) -> Dict[str, Any]:
        # Convert graph to NetworkX
        G = _convert_to_networkx(self.graph_storage)

        if G.number_of_nodes() == 0:
            return {"error": "Empty graph"}

        # Calculate metrics
        total_nodes = G.number_of_nodes()
        total_edges = G.number_of_edges()

        # Noise ratio: isolated nodes / total nodes
        isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
        noise_ratio = len(isolated_nodes) / total_nodes if total_nodes > 0 else 0

        # Largest connected component
        if G.is_directed():
            G_undirected = G.to_undirected()
        else:
            G_undirected = G

        connected_components = list(nx.connected_components(G_undirected))
        if connected_components:
            largest_cc = max(connected_components, key=len)
            largest_cc_ratio = (
                len(largest_cc) / total_nodes if total_nodes > 0 else 0
            )
        else:
            largest_cc_ratio = 0

        # Average node degree
        if total_nodes > 0:
            total_degree = sum(G.degree(n) for n in G.nodes())
            avg_degree = total_degree / total_nodes
        else:
            avg_degree = 0

        # Power law distribution R²
        powerlaw_r2 = self._calculate_powerlaw_r2(G)

        thresholds = {
            "noise_ratio": {
                "value": noise_ratio,
                "threshold": self.noise_ratio_threshold,
                "pass": noise_ratio < self.noise_ratio_threshold,
            },
            "largest_cc_ratio": {
                "value": largest_cc_ratio,
                "threshold": self.largest_cc_ratio_threshold,
                "pass": largest_cc_ratio > self.largest_cc_ratio_threshold,
            },
            "avg_degree": {
                "value": avg_degree,
                "threshold": (self.avg_degree_min, self.avg_degree_max),
                "pass": self.avg_degree_min <= avg_degree <= self.avg_degree_max,
            },
            "powerlaw_r2": {
                "value": powerlaw_r2,
                "threshold": self.powerlaw_r2_threshold,
                "pass": powerlaw_r2 > self.powerlaw_r2_threshold if powerlaw_r2 is not None else False,
            },
        }

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "isolated_nodes_count": len(isolated_nodes),
            "noise_ratio": noise_ratio,
            "largest_cc_ratio": largest_cc_ratio,
            "avg_degree": avg_degree,
            "powerlaw_r2": powerlaw_r2,
            "thresholds": thresholds,
        }

    def _calculate_powerlaw_r2(self, G: "nx.Graph") -> Optional[float]:
        """
        Calculate R² for power law distribution of node degrees.

        Returns:
            R² value if calculation successful, None otherwise
        """
        if stats is None:
            logger.warning("scipy not available, skipping power law R² calculation")
            return None

        degrees = [G.degree(n) for n in G.nodes()]
        if len(degrees) < 10:  # Need sufficient data points
            logger.warning("Insufficient nodes for power law fitting")
            return None

        # Filter out zero degrees for log fitting
        non_zero_degrees = [d for d in degrees if d > 0]
        if len(non_zero_degrees) < 5:
            return None

        try:
            # Fit power law: log(y) = a * log(x) + b
            log_degrees = np.log(non_zero_degrees)
            sorted_log_degrees = np.sort(log_degrees)
            x = np.arange(1, len(sorted_log_degrees) + 1)
            log_x = np.log(x)

            # Linear regression on log-log scale
            r_value, *_ = stats.linregress(log_x, sorted_log_degrees)
            r2 = r_value ** 2

            return float(r2)
        except Exception as e:
            logger.error(f"Power law R² calculation failed: {e}")
            return None
