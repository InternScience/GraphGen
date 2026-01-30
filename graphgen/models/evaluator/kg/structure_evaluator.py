from collections import Counter
from typing import Any, Dict, Optional

from graphgen.bases import BaseGraphStorage, BaseKGEvaluator
from graphgen.utils import logger


class StructureEvaluator(BaseKGEvaluator):
    """Evaluates structural robustness of the graph."""

    def __init__(
        self,
        noise_ratio_threshold: float = 0.15,
        largest_cc_ratio_threshold: float = 0.90,
        avg_degree_min: float = 2.0,
        avg_degree_max: float = 5.0,
        powerlaw_r2_threshold: float = 0.75,
    ):
        self.noise_ratio_threshold = noise_ratio_threshold
        self.largest_cc_ratio_threshold = largest_cc_ratio_threshold
        self.avg_degree_min = avg_degree_min
        self.avg_degree_max = avg_degree_max
        self.powerlaw_r2_threshold = powerlaw_r2_threshold

    def evaluate(self, kg: BaseGraphStorage) -> Dict[str, Any]:
        """
        Evaluate the structural robustness of the graph.
        :return: Dictionary of structural metrics and robustness verdict. The keys include:
            - total_nodes: Total number of nodes in the graph
            - total_edges: Total number of edges in the graph
            - noise_ratio: Ratio of isolated nodes to total nodes
            - largest_cc_ratio: Ratio of largest connected component size to total nodes
            - avg_degree: Average node degree
            - powerlaw_r2: R² value of power law fit to degree distribution
            - is_robust: Boolean indicating if the graph is structurally robust
        """
        total_nodes = kg.get_node_count()
        total_edges = kg.get_edge_count()
        degree_map = kg.get_all_node_degrees()

        # Noise ratio: isolated nodes / total nodes
        isolated_nodes = [nid for nid, deg in degree_map.items() if deg == 0]
        noise_ratio = len(isolated_nodes) / total_nodes

        # Largest connected component
        components = kg.get_connected_components(undirected=True)
        largest_cc_ratio = (
            len(max(components, key=len)) / total_nodes if components else 0
        )

        avg_degree = sum(degree_map.values()) / total_nodes
        powerlaw_r2 = self._calculate_powerlaw_r2(degree_map)

        results = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "noise_ratio": noise_ratio,
            "largest_cc_ratio": largest_cc_ratio,
            "avg_degree": avg_degree,
            "powerlaw_r2": powerlaw_r2,
            "is_robust": (
                noise_ratio < self.noise_ratio_threshold
                and largest_cc_ratio > self.largest_cc_ratio_threshold
                and self.avg_degree_min <= avg_degree <= self.avg_degree_max
                and (
                    powerlaw_r2 is not None and powerlaw_r2 > self.powerlaw_r2_threshold
                )
            ),
        }

        return results

    @staticmethod
    def _calculate_powerlaw_r2(degree_map: Dict[str, int]) -> Optional[float]:
        import numpy as np
        from scipy import stats

        degrees = [deg for deg in degree_map.values() if deg > 0]

        if len(degrees) < 10:
            logger.warning("Insufficient nodes for power law fitting")
            return None

        try:
            degree_counts = Counter(degrees)
            degree_values, frequencies = zip(*sorted(degree_counts.items()))

            if len(degree_values) < 3:
                logger.warning(
                    f"Insufficient unique degrees ({len(degree_values)}) for power law fitting. "
                    f"Graph may be too uniform."
                )
                return None

            # Fit power law: log(frequency) = a * log(degree) + b
            log_degrees = np.log(degree_values)
            log_frequencies = np.log(frequencies)

            # Linear regression on log-log scale
            result = stats.linregress(log_degrees, log_frequencies)
            r2 = result.rvalue**2

            return float(r2)
        except Exception as e:
            logger.error(f"Power law R² calculation failed: {e}")
            return None
