from functools import lru_cache
from pathlib import Path

import networkx as nx

from config import get_logger
from src.interfaces import INetworkService, NetworkPath
from src.state.utils import plot_nodes


logger = get_logger(__name__)


class NetworkXService(INetworkService):
    """Fast, minimal wrapper around a MultiDiGraph with precomputed edge weights."""

    def __init__(self, graph: nx.MultiDiGraph):
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("NetworkXService requires an nx.MultiDiGraph")
        self.G = graph
        self._nodeset = set(graph.nodes)

    # TODO: Potential improvements:
    # add cache with duckDB
    # https://chatgpt.com/c/690f4825-01f4-832e-aefc-c6a34a6fdc40

    @lru_cache(maxsize=2_000_000)  # Est. 4GB on 36GB RAM; key is (o,d)
    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:
        # Fastest path by precomputed time
        # if origin_node not in self._nodeset:
        #     raise KeyError(f"origin_node {origin_node} not in graph")
        # if destination_node not in self._nodeset:
        #     raise KeyError(f"destination_node {destination_node} not in graph")
        try:
            nodes = nx.shortest_path(
                self.G, origin_node, destination_node, weight="travel_time"
            )
        except nx.NetworkXNoPath as _:
            return None

        # Sum time/dist along the chosen path (cheap Python loop)
        total_time = 0.0
        total_dist = 0.0
        G = self.G  # local ref for speed
        travel_times = []
        for u, v in zip(nodes, nodes[1:]):
            edge_attr = min(G[u][v].values(), key=lambda attrs: attrs["travel_time"])

            total_time += float(edge_attr["travel_time"])
            travel_times.append(float(edge_attr["travel_time"]))
            # if dist is None:
            #     raise ValueError(
            #         f"Edge ({u},{v}) missing 'length' attribute for distance"
            #     )
            total_dist += float(edge_attr.get("length"))

        return NetworkPath(
            nodes=tuple(nodes),
            travel_times=tuple(travel_times),
            travel_time=total_time,
            travel_distance=total_dist,
        )
