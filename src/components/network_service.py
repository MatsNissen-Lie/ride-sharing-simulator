from src.interfaces import INetworkService, NetworkPath
from functools import lru_cache
import networkx as nx


class NetworkXService(INetworkService):
    """Fast, minimal wrapper around a DiGraph with precomputed edge weights."""

    def __init__(self, graph: nx.DiGraph):
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("NetworkXService requires an nx.DiGraph")
        self.G = graph
        self._nodeset = set(graph.nodes)

    # Potential improvements:
    # add cache with duckDB
    # https://chatgpt.com/c/690f4825-01f4-832e-aefc-c6a34a6fdc40

    @lru_cache(maxsize=200_000)  # tune based on memory; key is (o,d)
    def get_shortest_path(self, origin_node: int, destination_node: int) -> NetworkPath:
        # Fastest path by precomputed time
        # if origin_node not in self._nodeset:
        #     raise KeyError(f"origin_node {origin_node} not in graph")
        # if destination_node not in self._nodeset:
        #     raise KeyError(f"destination_node {destination_node} not in graph")
        nodes = nx.shortest_path(
            self.G, origin_node, destination_node, weight="travel_time"
        )

        # Sum time/dist along the chosen path (cheap Python loop)
        total_time = 0.0
        total_dist = 0.0
        G = self.G  # local ref for speed
        for u, v in zip(nodes, nodes[1:]):
            edge_attr = min(G[u][v].values(), key=lambda attrs: attrs["travel_time"])
            total_time += edge_attr["travel_time"]
            # if dist is None:
            #     raise ValueError(
            #         f"Edge ({u},{v}) missing 'length' attribute for distance"
            #     )
            total_dist += float(edge_attr.get("length"))

        return NetworkPath(
            nodes=nodes, travel_time=total_time, travel_distance=total_dist
        )
