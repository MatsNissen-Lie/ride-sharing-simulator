# network_service.py
from functools import lru_cache
import networkx as nx
from typing import Optional
from src.database.shortest_path_db import DuckDBPathCache
from src.interfaces import INetworkService, NetworkPath
from src.interfaces.data_models import NetworkPathExtended


class NetworkXService(INetworkService):
    def __init__(self, graph: nx.MultiDiGraph):
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("NetworkXService requires an nx.MultiDiGraph")
        self.G = graph
        self._nodeset = set(graph.nodes)
        self._cache = DuckDBPathCache()
        self._ghash = self.graph_hash(graph)

    @lru_cache(maxsize=200_000)
    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> Optional[NetworkPath]:
        # L2: tiny row → fastest
        hit = self._cache.get_simple(origin_node, destination_node, self._ghash)
        if hit:
            t, d = hit
            # We don’t need nodes for the simple variant
            return NetworkPath(nodes=(), travel_time=t, travel_distance=d)

        # Compute
        try:
            nodes = nx.shortest_path(
                self.G, origin_node, destination_node, weight="travel_time"
            )
        except nx.NetworkXNoPath:
            return None

        G = self.G
        total_time = 0.0
        total_dist = 0.0
        for u, v in zip(nodes, nodes[1:]):
            data = min(G[u][v].values(), key=lambda a: a["travel_time"])
            total_time += float(data["travel_time"])
            total_dist += float(data.get("length", 0.0))

        self._cache.put_simple(
            origin_node, destination_node, self._ghash, total_time, total_dist
        )

        return NetworkPath(
            nodes=tuple(nodes), travel_time=total_time, travel_distance=total_dist
        )

    def get_shortest_path_with_travel_time(
        self, origin_node: int, destination_node: int
    ) -> Optional[NetworkPathExtended]:
        hit = self._cache.get_extended(origin_node, destination_node, self._ghash)
        if hit:
            nodes, edge_times, t, d = hit
            return NetworkPathExtended(
                nodes=tuple(nodes),
                travel_times=tuple(edge_times),
                travel_time=t,
                travel_distance=d,
            )

        # Compute
        try:
            nodes = nx.shortest_path(
                self.G, origin_node, destination_node, weight="travel_time"
            )
        except nx.NetworkXNoPath:
            return None

        G = self.G
        total_time = 0.0
        total_dist = 0.0
        edge_times = []
        for u, v in zip(nodes, nodes[1:]):
            data = min(G[u][v].values(), key=lambda a: a["travel_time"])
            t = float(data["travel_time"])
            total_time += t
            total_dist += float(data.get("length", 0.0))
            edge_times.append(t)

        self._cache.put_extended(
            origin_node,
            destination_node,
            self._ghash,
            nodes,
            edge_times,
            total_time,
            total_dist,
        )

        return NetworkPathExtended(
            nodes=tuple(nodes),
            travel_times=tuple(edge_times),
            travel_time=total_time,
            travel_distance=total_dist,
        )

    @staticmethod
    def graph_hash(G: nx.MultiDiGraph, weight_attr: str = "travel_time") -> str:
        # lightweight fingerprint for cache versioning
        return f"{G.number_of_nodes()}-{G.number_of_edges()}-{int(G.size(weight=weight_attr) or 0)}"
