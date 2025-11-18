from functools import lru_cache

import networkx as nx

from config import get_logger
from src.database.simple_db import DuckDBPathCache
from src.interfaces import INetworkService
from src.interfaces.data_models import NetworkPath, NetworkPathExtended

logger = get_logger(__name__)


class NetworkXService(INetworkService):
    """
    Minimal MultiDiGraph wrapper with:
      - L1: @lru_cache
      - L2: DuckDBPathCache (time_simple hot table; optional path_extended)
      - Fallback: compute with NetworkX, write-through to DuckDB
    """

    def __init__(self, graph: nx.MultiDiGraph, with_cache: bool = True):
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("NetworkXService requires an nx.MultiDiGraph")
        self.G = graph
        self._nodeset = set(graph.nodes)
        self._cache = DuckDBPathCache() if with_cache else None

    @lru_cache(maxsize=200_000)  # lean L1; rely on DuckDB for persistence
    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:
        # L2: DuckDB hot table
        if self._cache:
            hit = self._cache.get_simple(origin_node, destination_node)
            if hit:
                time_s, dist_m = hit
                return NetworkPath(travel_time=time_s, travel_distance=dist_m)

        # Compute on miss
        try:
            nodes = nx.shortest_path(
                self.G, origin_node, destination_node, weight="travel_time"
            )
        except nx.NetworkXNoPath:
            return None

        total_time = 0.0
        total_dist = 0.0
        G = self.G
        for u, v in zip(nodes, nodes[1:]):
            edge_attr = min(G[u][v].values(), key=lambda a: a["travel_time"])
            total_time += float(edge_attr["travel_time"])
            total_dist += float(edge_attr.get("length", 0.0))

        # Write-through to L2 (only the tiny row)
        if self._cache:
            try:
                self._cache.put_simple(
                    origin_node, destination_node, total_time, total_dist
                )
            except Exception as e:
                logger.warning(
                    f"DuckDB put_simple failed for ({origin_node},{destination_node}): {e}"
                )

        return NetworkPath(travel_time=total_time, travel_distance=total_dist)

    def get_shortest_path_with_travel_time(
        self, origin_node: int, destination_node: int
    ) -> NetworkPathExtended | None:
        # First try extended payload (nodes+edge_times)
        nodes_et = None
        if self._cache:
            nodes_et = self._cache.get_extended(origin_node, destination_node)

        if nodes_et:
            nodes, edge_times_seq = nodes_et
            edge_times = list(edge_times_seq)
            # We still want time+dist; try simple first for minimal IO
            time_dist = (
                self._cache.get_simple(origin_node, destination_node)
                if self._cache
                else None
            )
            if time_dist:
                time_s, dist_m = time_dist
                return NetworkPathExtended(
                    nodes=tuple(int(n) for n in nodes),
                    travel_times=tuple(float(t) for t in edge_times),
                    travel_time=float(time_s),
                    travel_distance=float(dist_m),
                )

        # Fall back to compute the full path
        try:
            nodes = nx.shortest_path(
                self.G, origin_node, destination_node, weight="travel_time"
            )
        except nx.NetworkXNoPath:
            return None

        total_time = 0.0
        total_dist = 0.0
        edge_times: list[float] = []
        G = self.G
        for u, v in zip(nodes, nodes[1:]):
            edge_attr = min(G[u][v].values(), key=lambda a: a["travel_time"])
            t = float(edge_attr["travel_time"])
            total_time += t
            total_dist += float(edge_attr.get("length", 0.0))
            edge_times.append(t)

        # Write-through:
        if self._cache:
            # Ensure minimal row exists
            try:
                self._cache.put_simple(
                    origin_node, destination_node, total_time, total_dist
                )
            except Exception as e:
                logger.warning(
                    f"DuckDB put_simple failed for ({origin_node},{destination_node}): {e}"
                )
            # And extended row (optional)
            try:
                self._cache.put_extended(
                    origin_node, destination_node, nodes, edge_times
                )
            except Exception as e:
                logger.warning(
                    f"DuckDB put_extended failed for ({origin_node},{destination_node}): {e}"
                )

        return NetworkPathExtended(
            nodes=tuple(nodes),
            travel_times=tuple(edge_times),
            travel_time=total_time,
            travel_distance=total_dist,
        )
