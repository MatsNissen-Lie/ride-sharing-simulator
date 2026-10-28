import hashlib
import itertools
import os
from array import array
from typing import Iterable, Iterator
import time
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from config import get_logger
from src.database.simple_db import DuckDBPathCache
from src.interfaces import INetworkService
from src.interfaces.data_models import NetworkPath, NetworkPathExtended

logger = get_logger(__name__)


def _save(path: Path, values: np.ndarray) -> None:
    """np.save via a temporary file, so parallel runs never read half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, values)
    os.replace(tmp, path)
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
        travel_time_factor: float = 1.0,
        tree_cache_dir: Path | None = None,
    ):
        """
        travel_time_factor scales every travel time this service hands out
        (matrix, paths, single lookups) relative to the free-flow times in the
        graph and database, e.g. 2.9 for Manhattan evening traffic. The stored
        database is never changed.

        With tree_cache_dir, the shortest-path trees of all origins are computed
        once per graph and kept there (named by a hash of the edge weights);
        otherwise each origin's tree is computed when first needed.
        """
        if graph is not None and not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("NetworkXService requires an nx.MultiDiGraph")
        if travel_time_factor <= 0:
            raise ValueError("travel_time_factor must be positive")
        self.G = graph
        self.travel_time_factor = travel_time_factor
        self._adjacency: csr_matrix | None = None  # fastest edge per node pair
        self._predecessors: dict[int, array] = {}  # shortest-path tree per origin
        self._tree_cache_dir = tree_cache_dir
        # next hop, path time, path length per node pair; False: not usable
        self._routes: tuple[np.ndarray, np.ndarray, np.ndarray] | bool | None = None
        self._trees: np.ndarray | None = None  # all trees, from tree_cache_dir
        self._fastest: dict[tuple[int, int], tuple[float, float]] | None = None
        self._fastest_scaled: dict[tuple[int, int], tuple[float, float]] = {}
        self._distances_loaded = False

        if max_node_id is not None:
            self.N = max_node_id
        elif graph is not None:
            self.N = max(graph.nodes) + 1
        else:
            self.N = 0

        # L2: Persistent Storage
        self._db = db if db is not None else DuckDBPathCache(read_only=True)

        # --- NUCLEAR CACHE SETUP ---
        # 4600^2 * 4 bytes = ~85 MB. Cheap!
        # We use NaN as the "Not Found / Not Calculated" sentinel.

        # check that n is not too large
        # 20k^2 * 4 bytes = ~1.6 GB
        if self.N > 20_000:
            raise ValueError(
                f"NetworkXService: Node count {self.N} is too large for nuclear cache."
            )

        # 1. Time Matrix (float64 so planned and replayed times agree)
        if prefetch:  # every entry is written below
            self._time_matrix = np.empty((self.N, self.N), dtype=np.float64)
        else:
            self._time_matrix = np.full((self.N, self.N), np.nan, dtype=np.float64)

        # 2. Distance Matrix (Optional, if you need it for costs)
        self._dist_matrix = np.full((self.N, self.N), np.nan, dtype=np.float64)

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
                time_s = self._scaled(time_s)
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
        if nodes == list(range(self.N)):
            self._time_matrix[...] = self._scaled(db_matrix_time)  # no index arrays
        else:
            self._time_matrix[grid_indices] = self._scaled(db_matrix_time)

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
            sub_matrix = self._time_matrix[np.ix_(origins, destinations)]  # a copy

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
        return self._scaled(self._db.get_travel_time_matrix(origins, destinations))

    def get_shortest_path_with_travel_time(
        self, origin_node: int, destination_node: int
    ) -> NetworkPathExtended | None:
        return self._get_cached_path(origin_node, destination_node)

    @lru_cache(maxsize=50_000)
    def _get_cached_path(
        self, origin_node: int, destination_node: int
    ) -> NetworkPathExtended | None:
        # 1. Check for basic connectivity first (Fast L1 lookup)
        # using the simple matrix lookup to fail fast if no path exists
        # or if nodes are invalid
        direct_time = self.get_travel_time(origin_node, destination_node)
        if direct_time == float("inf"):
            return None

        # 2. Compute Path (Dijkstra tree of the origin)
        if self.G is None:
            raise RuntimeError("Underlying graph is not initialized.")
        nodes = self._path_nodes(origin_node, destination_node)
        if nodes is None:
            return None

        # 3. Compute Metrics (on the fastest of parallel edges)
        total_time = 0.0
        total_dist = 0.0
        edge_times_list: list[float] = []
        fastest = self._fastest_edges()

        for u, v in zip(nodes, nodes[1:]):
            t, length = fastest[u, v]
            total_time += t
            total_dist += length
            edge_times_list.append(t)

        return NetworkPathExtended(
            nodes=tuple(nodes),
            travel_times=tuple(edge_times_list),
            travel_time=total_time,
            travel_distance=total_dist,
        )

    def _path_nodes(self, origin: int, destination: int) -> list[int] | None:
        """Nodes of the fastest path, read from the shortest-path tree of `origin`.

        Trees come from scipy's Dijkstra on the fastest edge per node pair and are
        kept per origin (at most N x N int32). Same paths as nx.shortest_path, which
        cost ~5 ms per call (checked on 60,000 pairs of the Manhattan graph).
        """
        pred = self._predecessors.get(origin)
        if pred is None:
            self._adjacency_matrix()
            if self._tree_cache_dir is not None and self._trees is None:
                self._trees = self._all_trees()
            if self._trees is not None:
                tree = self._trees[origin]
            else:
                _, tree = dijkstra(self._adjacency, indices=origin, return_predecessors=True)
            pred = array("i", tree.astype(np.int32).tobytes())  # fast item reads
            self._predecessors[origin] = pred
        if destination != origin and pred[destination] < 0:
            return None
        nodes = [destination]
        while nodes[-1] != origin:
            nodes.append(pred[nodes[-1]])
        nodes.reverse()
        return nodes

    def _adjacency_matrix(self) -> csr_matrix:
        """Free-flow time of the fastest edge per node pair, for scipy's Dijkstra."""
        if self._adjacency is None:
            times = {pair: t for pair, (t, _) in self._fastest_edges(scaled=False).items()}
            rows, cols = zip(*times)
            self._adjacency = csr_matrix(
                (list(times.values()), (rows, cols)), shape=(self.N, self.N)
            )
        return self._adjacency

    def _graph_key(self) -> str:
        key = hashlib.sha1()
        adjacency = self._adjacency_matrix()
        for part in (adjacency.indptr, adjacency.indices, adjacency.data):
            key.update(np.ascontiguousarray(part).tobytes())
        return f"{self.N}_{key.hexdigest()[:16]}"

    def _all_trees(self) -> np.ndarray:
        """Predecessor rows of every origin (N x N int32), read from the cache
        directory or computed (~2.4 s for Manhattan) and stored there."""
        path = self._tree_cache_dir / f"shortest_path_trees_{self._graph_key()}.npy"
        if path.exists():
            return np.load(path)  # whole, as in _route_arrays
        _, trees = dijkstra(self._adjacency, return_predecessors=True)
        _save(path, trees.astype(np.int32))
        return trees

    def path_time_and_distance(
        self, origin_node: int, destination_node: int
    ) -> tuple[float, float] | None:
        routes = self._route_arrays()
        if routes is None:
            return super().path_time_and_distance(origin_node, destination_node)
        _, time, length = routes
        travel_time = time[origin_node, destination_node]
        if travel_time != travel_time:  # NaN: no path
            return None
        return float(travel_time), float(length[origin_node, destination_node])

    def path_steps(self, origin_node: int, destination_node: int) -> Iterator[tuple[int, float]]:
        routes = self._route_arrays()
        if routes is None:
            yield from super().path_steps(origin_node, destination_node)
            return
        hop, time, _ = routes
        if time[origin_node, destination_node] != time[origin_node, destination_node]:
            return  # no path
        node = origin_node
        while node != destination_node:
            following = int(hop[node, destination_node])
            yield following, float(time[node, following])  # time of the one edge
            node = following

    def _route_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(next hop, path time, path length) for every node pair, or None.

        next_hop[o, d] is the node after o on the path of
        get_shortest_path_with_travel_time(o, d); time and length are that
        path's travel_time and travel_distance, summed edge by edge from o as
        _get_cached_path sums them. Walking next hops from o follows o's own path
        only if every node's tree agrees with the trees of the nodes before it
        (true on Manhattan: no exceptions among 21M pairs); this is checked
        when the arrays are built, and None (use the paths) otherwise. Built in
        ~6 s; kept in the tree cache directory when there is one.
        """
        if self._routes is not None:
            return self._routes or None
        key = self._graph_key()
        factor = f"{self.travel_time_factor:g}"
        names = (f"next_hop_{key}.npy", f"path_time_{key}_x{factor}.npy", f"path_length_{key}.npy")
        cache = self._tree_cache_dir
        if cache is not None and all((cache / n).exists() for n in names):
            # Read whole: one sequential read beats page faults all over the
            # files (~30 s on a cold disk cache), and plain arrays index faster.
            routes = tuple(np.load(cache / n) for n in names)
        else:
            routes = self._build_route_arrays()
            if routes is None:
                logger.warning("Shortest-path trees disagree; paths are built one by one.")
                self._routes = False
                return None
            if cache is not None:
                for name, values in zip(names, routes):
                    _save(cache / name, values)
        # get_shortest_path_with_travel_time gives no path where the travel-time
        # matrix has none; mark those pairs NaN so one lookup answers both.
        hop, time, length = routes
        no_path = ~np.isfinite(self._time_matrix) | (self._time_matrix < 0) | (self._time_matrix > 1e9)
        self._routes = hop, np.where(no_path, np.nan, time), length
        return self._routes

    def _build_route_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        n = self.N
        dist, pred = dijkstra(self._adjacency_matrix(), return_predecessors=True)
        order = np.argsort(dist, axis=1, kind="stable")  # parents before children
        edges = self._fastest_edges()
        keys = np.array([u * n + v for u, v in edges], dtype=np.int64)
        sort = np.argsort(keys)
        keys = keys[sort]
        edge_time = np.array([t for t, _ in edges.values()])[sort]
        edge_length = np.array([d for _, d in edges.values()])[sort]

        hop = np.full((n, n), -1, dtype=np.int32)
        time = np.full((n, n), np.nan)
        length = np.full((n, n), np.nan)
        rows = np.arange(n)
        time[rows, rows] = 0.0
        length[rows, rows] = 0.0
        for rank in range(1, n):
            node = order[:, rank]
            parent = pred[rows, node]
            o, node, parent = rows[parent >= 0], node[parent >= 0], parent[parent >= 0]
            wanted = parent.astype(np.int64) * n + node
            e = np.searchsorted(keys, wanted)
            if (keys[np.minimum(e, len(keys) - 1)] != wanted).any():
                raise RuntimeError("Tree edge missing from the fastest edges")
            time[o, node] = time[o, parent] + edge_time[e]
            length[o, node] = length[o, parent] + edge_length[e]
            hop[o, node] = np.where(parent == o, node, hop[o, parent])

        o, d = np.nonzero(pred >= 0)
        if np.isnan(time[o, d]).any():
            raise RuntimeError("Path times built out of order")
        first = hop[o, d]
        later = d != first
        if (pred[first[later], d[later]] != pred[o[later], d[later]]).any():
            return None
        return hop, time, length

    def _fastest_edges(self, scaled: bool = True) -> dict[tuple[int, int], tuple[float, float]]:
        """(travel time, length) of the fastest edge per node pair; the first one
        on ties, as min() over the parallel edges. Times scaled unless `scaled` is False."""
        if self._fastest is None:
            fastest: dict[tuple[int, int], dict] = {}
            for u, v, attr in self.G.edges(data=True):
                if (u, v) not in fastest or attr["travel_time"] < fastest[u, v]["travel_time"]:
                    fastest[u, v] = attr
            self._fastest = {
                pair: (float(a["travel_time"]), float(a.get("length", 0.0)))
                for pair, a in fastest.items()
            }
            self._fastest_scaled = {
                pair: (self._scaled(t), length) for pair, (t, length) in self._fastest.items()
            }
        return self._fastest_scaled if scaled else self._fastest

    def _scaled(self, free_flow):
        """The single place where free-flow times become simulated times.

        Negative values are "no path" sentinels and stay unchanged.
        """
        if self.travel_time_factor == 1.0:
            return free_flow
        if isinstance(free_flow, np.ndarray):
            scaled = free_flow * self.travel_time_factor
            np.copyto(scaled, free_flow, where=free_flow < 0)
            return scaled
        return free_flow if free_flow < 0 else free_flow * self.travel_time_factor

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
                time_s = self._scaled(time_s)
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

    def get_distance_matrix(self) -> np.ndarray:
        """All-pairs shortest-path distances in metres (loaded on first use)."""
        if not self._distances_loaded:
            nodes = list(range(self.N))
            self._dist_matrix[...] = self._db.get_distance_matrix(nodes, nodes)
            self._distances_loaded = True
        return self._dist_matrix

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
