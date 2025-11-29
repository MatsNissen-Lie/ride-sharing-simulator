from functools import lru_cache

import networkx as nx
import numpy as np

from config import get_logger
from src.database.simple_db import DuckDBPathCache
from src.interfaces import INetworkService
from src.interfaces.data_models import NetworkPath

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

    @lru_cache(maxsize=200_000)  # lean L1; rely on DuckDB for persistence
    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:
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
