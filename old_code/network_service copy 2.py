from functools import lru_cache
from typing import override

import networkx as nx
import numpy as np

from config import get_logger
from src.database.simple_db import DuckDBPathCache
from src.interfaces import INetworkService
from src.interfaces.data_models import NetworkPath, NetworkPathExtended

logger = get_logger(__name__)
# set debug level for this module


class NetworkXService(INetworkService):
    """
    Minimal MultiDiGraph wrapper with:
      - L1: @lru_cache
      - L2: DuckDBPathCache (time_simple hot table; optional path_extended)
      - Fallback: compute with NetworkX, write-through to DuckDB
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph | None,
        cache: DuckDBPathCache | None = None,
    ):
        if graph is not None and not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("NetworkXService requires an nx.MultiDiGraph")
        self.G = graph
        self._nodeset = set(graph.nodes) if graph is not None else set()
        self._cache = cache if cache is not None else DuckDBPathCache()
        # L1 Manual Cache for Eager Prefetching
        self._manual_cache: dict[tuple[int, int], NetworkPath] = {}

    @lru_cache(maxsize=200_000)  # lean L1; rely on DuckDB for persistence
    def get_shortest_path(  # type: ignore[override]
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:
        # L1: Manual Cache (Eager Prefetch)
        if (origin_node, destination_node) in self._manual_cache:
            return self._manual_cache[(origin_node, destination_node)]

        # L2: DuckDB hot table

        hit = self._cache.get_simple(origin_node, destination_node)
        if hit:
            time_s, dist_m = hit
            # Check for "no path" marker
            if time_s < 0:
                return None
            return NetworkPath(travel_time=time_s, travel_distance=dist_m)
        else:
            if origin_node == destination_node:
                return NetworkPath(travel_time=0.0, travel_distance=0.0)
            print("Cache miss in L2 DuckDB for nodes", origin_node, destination_node)
            raise RuntimeError("I think we have compueted all pairs already")

        # Compute on miss
        try:
            nodes = nx.shortest_path(
                self.G, origin_node, destination_node, weight="travel_time"
            )
        except nx.NetworkXNoPath:
            # Store "no path" to prevent re-computation
            self._cache.put_no_path(origin_node, destination_node)
            return None

        total_time = 0.0
        total_dist = 0.0
        G = self.G
        for u, v in zip(nodes, nodes[1:]):
            edge_attr = min(G[u][v].values(), key=lambda a: a["travel_time"])
            total_time += float(edge_attr["travel_time"])
            total_dist += float(edge_attr.get("length", 0.0))

        # Write-through to L2 (only the tiny row)
        self._cache.put_simple(origin_node, destination_node, total_time, total_dist)

        return NetworkPath(travel_time=total_time, travel_distance=total_dist)

    def get_shortest_path_with_travel_time(
        self, origin_node: int, destination_node: int
    ) -> NetworkPathExtended | None:
        # 1) Single round-trip to cache
        time_s = dist_m = None
        nodes = edge_times = None

        time_s, dist_m, nodes, edge_times = self._cache.get_both(
            origin_node, destination_node
        )

        # Check for "no path" marker
        if time_s is not None and time_s < 0:
            return None

        # If we have everything, return immediately
        if (
            nodes is not None
            and edge_times is not None
            and time_s is not None
            and dist_m is not None
        ):
            return NetworkPathExtended(
                nodes=tuple(int(n) for n in nodes),
                travel_times=tuple(float(t) for t in edge_times),
                travel_time=float(time_s),
                travel_distance=float(dist_m),
            )
        if self.G is None:
            raise RuntimeError("Underlying graph is not initialized.")

        # 2) If we have nodes+edge_times but missing totals, derive what we can
        if nodes is not None and edge_times is not None:
            derived_time = float(sum(edge_times))
            # If distance missing, recompute distance cheaply by summing along nodes in G
            if dist_m is None:
                total_dist = 0.0
                G = self.G
                for u, v in zip(nodes, nodes[1:]):
                    # use the min-time edge (consistent with how path was produced)
                    edge_attr = min(G[u][v].values(), key=lambda a: a["travel_time"])
                    total_dist += float(edge_attr.get("length", 0.0))
                dist_m = total_dist
            if time_s is None:
                time_s = derived_time

            # Write-through minimal row if it was missing
            if time_s is not None and dist_m is not None:
                self._cache.put_simple(origin_node, destination_node, time_s, dist_m)

            return NetworkPathExtended(
                nodes=tuple(int(n) for n in nodes),
                travel_times=tuple(float(t) for t in edge_times),
                travel_time=float(time_s),
                travel_distance=float(dist_m),
            )

        # 3) Cache miss: compute fresh
        try:
            if self.G is None:
                raise RuntimeError("Underlying graph is not initialized.")
            nodes = nx.shortest_path(
                self.G, origin_node, destination_node, weight="travel_time"
            )
        except nx.NetworkXNoPath:
            # Store "no path" to prevent re-computation
            self._cache.put_no_path(origin_node, destination_node)
            return None

        total_time = 0.0
        total_dist = 0.0
        edge_times_list: list[float] = []
        G = self.G
        for u, v in zip(nodes, nodes[1:]):
            edge_attr = min(G[u][v].values(), key=lambda a: a["travel_time"])
            t = float(edge_attr["travel_time"])
            total_time += t
            total_dist += float(edge_attr.get("length", 0.0))
            edge_times_list.append(t)

        # 4) Write-through both layers
        self._cache.put_simple(origin_node, destination_node, total_time, total_dist)
        self._cache.put_extended(origin_node, destination_node, nodes, edge_times_list)

        return NetworkPathExtended(
            nodes=tuple(nodes),
            travel_times=tuple(edge_times_list),
            travel_time=total_time,
            travel_distance=total_dist,
        )

    def get_travel_time_matrix(
        self, origins: list[int], destinations: list[int]
    ) -> np.ndarray:
        # L2: Fetch from DuckDB
        matrix = self._cache.get_travel_time_matrix(origins, destinations)

        # OPTIONAL: Handle cache misses (where matrix value is inf)
        # If you want to compute missing paths on the fly and update the DB,
        # you would identify indices where matrix[i,j] == inf, compute them using NetworkX,
        # update the DB, and update the matrix.
        # For high-performance pruning, it is often better to assume
        # "inf" means "too far/unknown" and skip, OR compute strictly needed ones later.

        return matrix

    def prefetch_all_pairs(self, nodes: list[int]) -> None:
        """
        Eagerly fetches the travel time matrix for all combinations of the provided nodes
        and populates the internal manual cache.
        """
        if not nodes:
            return

        # 1. Fetch the raw matrix (N x N) from DuckDB
        # This is extremely fast even for 500x500
        matrix = self.get_travel_time_matrix(nodes, nodes)

        # 2. Bulk insert into manual_cache
        n = len(nodes)
        for i in range(n):
            u = nodes[i]
            for j in range(n):
                v = nodes[j]
                time_s = matrix[i, j]

                # Check for infinity (unreachable) or validity
                if time_s < 1e9:  # Assuming inf is high
                    # We store distance=0.0 because the matrix usually only gives time.
                    # If exact distance is needed for cost, the matrix query needs to return it too.
                    self._manual_cache[(u, v)] = NetworkPath(
                        travel_time=time_s, travel_distance=0.0
                    )

    def clear_manual_cache(self) -> None:
        """Clears the manual cache to free memory."""
        self._manual_cache.clear()
