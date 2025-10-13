from typing import List

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community
from graphgen.models import BFSPartitioner


class ECEPartitioner(BFSPartitioner):
    """
    ECE partitioner that partitions the graph into communities based on Expected Calibration Error (ECE).
    We calculate ECE for edges in KG(represented as 'comprehension loss') and group edges with similar ECE values into the same community.
    1. Select a sampling strategy.
    2. Choose a unit based on the sampling strategy.
    2. Expand the community using BFS.
    3. When expending, prefer to add units with the sampling strategy.
    4. Stop when the max unit size is reached or the max input length is reached.
    (A unit is a node or an edge.)
    """

    # async def partition(
    #         self,
    #         g: BaseGraphStorage,
    #         *,
    # ):
    #     pass


# 修改
# max_depth 取消
# expand_method 改名为 xxx
# edge_sampling
# loss_strategy取消，因为node和edge可以看作同一种unit
# bidirectional 取消
# max_extra_edges 改名为 max_units_per_community
# max_tokens 改名为 max_tokens_per_community

# 可以退化成BFS
