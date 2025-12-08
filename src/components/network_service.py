import itertools
from typing import Iterable
import time
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
      - L1: Nuclear Cache (NumPy dense arrays)
      - L2: DuckDBPathCache (time_simple hot table; optional path_extended)
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph | None,
        db: DuckDBPathCache | None = None,
        max_node_id: int | None = None,
        prefetch: bool = True,
    ):
        if graph is not None and not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("NetworkXService requires an nx.MultiDiGraph")
        self.G = graph

        if max_node_id is not None:
            self.N = max_node_id
        elif graph is not None:
            self.N = max(graph.nodes) + 1
        else:
            self.N = 0

        # L2: Persistent Storage
        self._db = db if db is not None else DuckDBPathCache()

        # --- NUCLEAR CACHE SETUP ---
        # 4600^2 * 4 bytes = ~85 MB. Cheap!
        # We use NaN as the "Not Found / Not Calculated" sentinel.

        # check that n is not too large
        # 20k^2 * 4 bytes = ~1.6 GB
        if self.N > 20_000:
            raise ValueError(
                f"NetworkXService: Node count {self.N} is too large for nuclear cache."
            )

        # 1. Time Matrix (Float32 for speed/size)
        self._time_matrix = np.full((self.N, self.N), np.nan, dtype=np.float32)

        # 2. Distance Matrix (Optional, if you need it for costs)
        self._dist_matrix = np.full((self.N, self.N), np.nan, dtype=np.float32)

        # Lets make it nuclear by default
        if not prefetch:
            self.is_nuclear = False
            return
        logger.info("NetworkXService: Initializing Nuclear Cache...")
        start_time = time.time()
        self.prefetch_all_pairs(list(range(self.N)))
        logger.info(
            f"NetworkXService: Nuclear Cache initialized in {time.time() - start_time:.2f} seconds."
        )
        self.is_nuclear = True

    def get_shortest_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPath | None:
        # 1. Direct Array Access (Fastest possible single-item lookup)
        # Note: We check bounds only if you suspect invalid IDs
        t = float(self._time_matrix[origin_node, destination_node])

        # 2. Check for Cache Miss (NaN)
        if np.isnan(t):
            if self.is_nuclear:
                raise RuntimeError(
                    f"Nuclear cache miss detected. This should not happen. Nodes {origin_node} -> {destination_node}"
                )

            # L2 DuckDB Lookup (Fallback)
            hit = self._db.get_simple(origin_node, destination_node)
            if hit:
                time_s, dist_m = hit
                # Update L1 immediately for next time
                self._time_matrix[origin_node, destination_node] = time_s
                self._dist_matrix[origin_node, destination_node] = dist_m

                if time_s < 0:
                    return None
                return NetworkPath(travel_time=time_s, travel_distance=dist_m)

            # If not in DB either, return None (or raise error if strict)
            # User said "The db has all pairs already loaded", so this should imply
            # if it's not in DB, it doesn't exist or is unreachable.
            return None

        # 3. Check for No Path (inf)
        # We assume the DB returns inf for unreachable nodes, which we store in the matrix.
        # Also check for -1.0 which might be stored as "no path" sentinel in DB/matrix
        if t == float("inf") or t > 1e9 or t < 0:
            return None

        # 4. Return Path
        # We also need distance.
        d = float(self._dist_matrix[origin_node, destination_node])
        return NetworkPath(travel_time=t, travel_distance=d)

    def prefetch_all_pairs(self, nodes: list[int]) -> None:
        """
        Populates the NumPy Cache using advanced indexing (Vectorized).
        """
        if not nodes:
            return

        # 1. Fetch from DB (returns NxN matrix)
        # Ensure DB returns data in the exact order of 'nodes' list
        # This returns a matrix of travel times.
        db_matrix_time = self._db.get_travel_time_matrix(nodes, nodes)

        # 2. NUCLEAR UPDATE (No Loops)
        # We use NumPy "Fancy Indexing" to update the specific sub-grid instantly.
        # This operates at C-speed.

        # Create the grid of indices (ix_ handles the cross-product broadcasting)
        # This says: Update rows [nodes] and columns [nodes]
        grid_indices = np.ix_(nodes, nodes)

        # Bulk write
        self._time_matrix[grid_indices] = db_matrix_time

        # Note: We are not populating _dist_matrix here because get_travel_time_matrix
        # only returns times. Distances will be fetched on-demand via L2 fallback
        # if get_shortest_path is called.

        logger.info(f"Nuclear Cache: Updated {len(nodes)**2} entries.")

    def get_travel_times_l1_cache_subset(
        self, nodes: Iterable[int]
    ) -> dict[tuple[int, int], float]:
        """
        Extracts a subset of the matrix for the Route Planner.
        Blazing fast because it extracts a contiguous block (or indexed block)
        and zips it, skipping single-item lookups.
        """
        node_list = list(nodes)

        # 1. Extract Sub-Matrix (Vectorized)
        # This creates a small temporary copy, e.g., 6x6 floats
        sub_matrix = self._time_matrix[np.ix_(node_list, node_list)]

        # # Enforce 0.0 for diagonals (self-loops)
        # np.fill_diagonal(sub_matrix, 0.0) #TODO: Check if needed

        # 2. Flatten
        flat_values = sub_matrix.ravel()

        # 3. Create Dict
        # itertools.product matches the 'ravel' order (Row-Major)
        # We use dict(zip(...)) which is highly optimized in Python
        return dict(zip(itertools.product(node_list, repeat=2), flat_values))  # type: ignore

    def get_travel_time_matrix(
        self, origins: list[int], destinations: list[int]
    ) -> np.ndarray:
        if self.is_nuclear:
            # 1. Extract Sub-Matrix (Vectorized)
            # We copy to avoid modifying the cache when we replace NaNs
            sub_matrix = self._time_matrix[np.ix_(origins, destinations)].copy()

            # 2. Convert NaN (Nuclear "No Path") to inf (Standard "No Path")
            # This ensures consistency with DuckDBPathCache behavior
            sub_matrix[np.isnan(sub_matrix)] = float("inf")

            # 3. Enforce 0.0 for diagonals if origins and destinations overlap
            # This is a bit tricky with rectangular matrices, but if row_idx == col_idx (meaning same node), it should be 0.0
            # However, np.ix_ returns a copy, so we can modify it safely.
            # But we need to know WHICH elements correspond to self-loops.
            # A simple way is to iterate, but that defeats vectorization.
            # Actually, if the user asks for (u, u), it should already be 0.0 in _time_matrix.
            # So we just return the sub_matrix after NaN conversion.
            return sub_matrix

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
        if not self._db.read_only:
            self._db.put_simple(origin_node, destination_node, total_time, total_dist)
            self._db.put_extended(origin_node, destination_node, nodes, edge_times_list)

        return NetworkPathExtended(
            nodes=tuple(nodes),
            travel_times=tuple(edge_times_list),
            travel_time=total_time,
            travel_distance=total_dist,
        )

    def get_travel_time(self, origin: int, destination: int) -> float:
        """
        Returns the travel time between two nodes.
        Returns float('inf') if no path exists.
        """
        # 1. Direct Array Access
        t = float(self._time_matrix[origin, destination])

        # 2. Check for Cache Miss (NaN)
        if np.isnan(t):
            if self.is_nuclear:
                raise RuntimeError(
                    f"Nuclear cache miss detected. This should not happen. Nodes {origin} -> {destination}"
                )

            # L2 DuckDB Lookup (Fallback)
            hit = self._db.get_simple(origin, destination)
            if hit:
                time_s, dist_m = hit
                # Update L1 immediately for next time
                self._time_matrix[origin, destination] = time_s
                self._dist_matrix[origin, destination] = dist_m

                if time_s < 0:
                    return float("inf")
                return float(time_s)

            # If not in DB, assume no path
            return float("inf")

        # 3. Check for No Path (inf or negative sentinel)
        if t == float("inf") or t > 1e9 or t < 0:
            return float("inf")

        return t

    def clear_l1_cache(self) -> None:
        """Clears the manual cache to free memory."""
        # Reset to NaN
        self._time_matrix.fill(np.nan)
        self._dist_matrix.fill(np.nan)
        # Restore diagonals
        np.fill_diagonal(self._time_matrix, 0.0)
        np.fill_diagonal(self._dist_matrix, 0.0)

    def update_cache(self, cache: np.ndarray) -> None:
        """Updates the L1 cache with a given time matrix."""
        if cache.shape != (self.N, self.N):
            raise ValueError(
                f"Input time_matrix shape {cache.shape} does not match expected {(self.N, self.N)}"
            )
        self._time_matrix = cache

    def get_l1_cache(self):
        """Returns the current L1 cache."""
        return self._time_matrix

    def get_matrix_and_mapping(self) -> tuple[np.ndarray, dict[int, int]]:
        """
        Returns the raw travel time matrix and a mapping from Node ID -> Matrix Index.
        In this implementation, Node ID == Matrix Index (Identity Mapping).
        """
        # Identity mapping for 0..N-1
        # For performance, we can return a specialized dict-like object or just a dict.
        # Given the User Request emphasizes explicit robust mapping, we return a dict.
        # Use cache if N is large?
        if not hasattr(self, "_identity_mapping"):
            self._identity_mapping = {i: i for i in range(self.N)}
        return self._time_matrix, self._identity_mapping
