import itertools
import itertools
from typing import Iterator, cast, Iterable

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
        db: DuckDBPathCache | None = None,
    ):
        if graph is not None and not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("NetworkXService requires an nx.MultiDiGraph")
        self.G = graph
        # L2: Persistent Storage
        self._db = db if db is not None else DuckDBPathCache()

        # L1: Fast RAM Cache.
        # Structure: {(origin, dest): (time_s, dist_m)}
        # We store tuples, not Objects, to save RAM and instantiation time.
        self._l1_cache: dict[tuple[int, int], tuple[float, float]] = {}

    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:

        # 1. Fast L1 Lookup (O(1))
        # We catch the KeyError explicitly; it is faster than .get() for high-hit-rate caches
        try:
            time_s, dist_m = self._l1_cache[(origin_node, destination_node)]
            if time_s < 0:
                return None  # Handle "No Path" marker
            return NetworkPath(travel_time=time_s, travel_distance=dist_m)
        except KeyError:
            pass

        # 2. L2 DuckDB Lookup
        hit = self._db.get_simple(origin_node, destination_node)
        if hit:
            time_s, dist_m = hit
            # Update L1 immediately for next time
            self._l1_cache[(origin_node, destination_node)] = (time_s, dist_m)

            if time_s < 0:
                return None
            return NetworkPath(travel_time=time_s, travel_distance=dist_m)

        # 3. Fallback: Calculation (Same as your original logic, simplified)
        if origin_node == destination_node:
            return NetworkPath(travel_time=0.0, travel_distance=0.0)

        # ... (Insert your path computation logic here) ...
        # log the nodes not found
        logger.warning(f"No path found between {origin_node} and {destination_node}")
        raise RuntimeError("Computed paths logic should not be needed here.")
        return None

    def prefetch_all_pairs(self, nodes: list[int]) -> None:
        """
        Populates the cache from the DB matrix WITHOUT python loops.
        """
        if not nodes:
            return

        # 1. Fetch Matrix (C-speed fetch)
        # Returns shape (N, N) floats
        matrix_time = self._db.get_travel_time_matrix(nodes, nodes)

        # 2. Flatten data for bulk insert
        # We need a flat list of values to match the flat list of keys
        flat_times = matrix_time.ravel(order="C")  # O(1) view if contiguous

        # 3. Create Keys Iterator (C-speed generation)
        # itertools.product creates the (u, v) pairs without storing them all in RAM at once
        keys_iterator = cast(
            Iterator[tuple[int, int]], itertools.product(nodes, repeat=2)
        )

        # 4. Handle Distances
        # If your matrix only gives time, we assume dist=0 or handle it.
        # If you have a distance matrix too, flatten it similarly.
        # For this example, we'll zip time with a default distance of 0.0
        # If you strictly need distance, fetch a distance matrix too.

        # Create generator of values: ((time, dist), (time, dist)...)
        # We use a generator expression so we don't allocate a massive list
        values_iterator: Iterator[tuple[float, float]] = (
            (float(t), 0.0) for t in flat_times
        )

        # 5. BULK UPDATE (The "Magic" Step)
        # dict.update consumes the zipped iterators in C.
        # No Python loop overhead. Fastest possible insertion.
        self._l1_cache.update(zip(keys_iterator, values_iterator))

        logger.info(f"Prefetched {len(nodes)**2} pairs into L1 cache.")

    def get_travel_time_matrix(
        self, origins: list[int], destinations: list[int]
    ) -> np.ndarray:
        # Check if we have these in L1 first?
        # Usually for matrices, it's faster to go straight to DuckDB
        # OR build a numpy array from L1 if you are sure L1 is full.
        return self._db.get_travel_time_matrix(origins, destinations)

    def get_shortest_path_with_travel_time(
        self, origin_node: int, destination_node: int
    ) -> NetworkPathExtended | None:
        # 1) Single round-trip to cache
        time_s = dist_m = None
        nodes = edge_times = None

        time_s, dist_m, nodes, edge_times = self._db.get_both(
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
                self._db.put_simple(origin_node, destination_node, time_s, dist_m)

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
            self._db.put_no_path(origin_node, destination_node)
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
        self._db.put_simple(origin_node, destination_node, total_time, total_dist)
        self._db.put_extended(origin_node, destination_node, nodes, edge_times_list)

        return NetworkPathExtended(
            nodes=tuple(nodes),
            travel_times=tuple(edge_times_list),
            travel_time=total_time,
            travel_distance=total_dist,
        )

    def clear_l1_cache(self) -> None:
        """Clears the manual cache to free memory."""
        self._l1_cache.clear()

    def get_travel_times_l1_cache_subset(
        self, nodes: Iterable[int]
    ) -> dict[tuple[int, int], float]:
        """
        Returns a dense dictionary of travel times.
        Raises KeyError if any pair is missing from the L1 cache.
        """
        # 1. Local variable optimization (avoids 'self.' lookup in the loop)
        main_cache = self._l1_cache

        # 2. Dictionary Comprehension
        # This executes the loop in C. It creates the tuple key, hashes it,
        # and looks up the value. If a key is missing, it raises KeyError immediately.
        subset = {
            (u, v): main_cache[(u, v)][0] for u, v in itertools.permutations(nodes, 2)
        }

        # 3. Handle Diagonals (Self-loops)
        # Permutations skips (u, u), but the solver might need to know that cost is 0.0
        # Doing this here is faster than doing "if u == v" inside the hot simulation loop.
        subset.update({(n, n): 0.0 for n in nodes})

        return subset
