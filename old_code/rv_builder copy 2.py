"""
rv_graph_builder
=================

This module provides an implementation of the pairwise request-vehicle (RV)
graph described in the supplemental material of Alonso-Mora et al. (2017).
The RV-graph captures which pairs of requests may be served together by a
single vehicle and which individual requests may be served by a given
vehicle. Each edge in the graph is annotated with the cost returned from
the single vehicle route planner, allowing subsequent steps (such as trip
construction and assignment) to reason about the feasibility and cost of
possible matches.

The implementation here does not impose any particular limit on the number
of edges per node; however, users may choose to prune the graph to speed up
subsequent computations. See the paper for details on how such pruning can
trade off optimality for runtime.

The central class exported by this module is :class:`RVGraphBuilder`, which
implements the :class:`IRVGraphBuilder` interface from ``src.interfaces``.
It relies on a provided :class:`RoutePlanner` to evaluate feasibility of
pairings. The resulting graph is returned as an instance of
:class:`RVGraph` defined below.
"""

from collections.abc import Iterable
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import get_logger
from config.config import OMEGA


from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    IRVGraphBuilder,
    Request,
)
from src.interfaces.graphs import RVGraph
from src.interfaces.services import INetworkService

from src.components.graph_builder.workers import (
    process_rr_chunk,
    process_rv_chunk,
    RRTask,
    RVTask,
    init_worker,
)

logger = get_logger("rv_graph_builder")


class RVGraphBuilderOld(IRVGraphBuilder):
    """Concrete implementation of :class:`IRVGraphBuilder`.
    (Docstring unchanged)
    """

    def __init__(
        self,
        route_planner: ISingleVehicleRoutePlanner,
        network: INetworkService,
    ) -> None:
        super().__init__(route_planner)
        self._network = network

    def build_graph(
        self, requests: set[Request], vehicles: set[VehicleAvailability]
    ) -> RVGraph:
        graph = RVGraph()
        req_list = list(requests)
        veh_list = list(vehicles)

        # --- 1. VECTORIZED PRUNING (The Filter) ---
        # (This is the code we wrote in the previous turn)

        req_origins = [r.origin_node for r in req_list]
        rr_matrix = self._network.get_travel_time_matrix(req_origins, req_origins)

        forward_ok = rr_matrix <= OMEGA
        backward_ok = rr_matrix.T <= OMEGA
        rr_mask = np.triu(forward_ok | backward_ok, k=1)
        rr_idx_i, rr_idx_j = np.where(rr_mask)

        # --- 2. PREPARE TASKS FOR MULTIPROCESSING ---

        rr_tasks = []
        for i, j in zip(rr_idx_i, rr_idx_j):
            rr_tasks.append(RRTask(req_list[i], req_list[j]))

        logger.info(
            f"Matrix pruning kept {len(rr_tasks)} RR pairs out of {len(req_list)**2}"
        )

        # --- 3. EXECUTE PARALLEL RR CHECKS ---

        # Chunk Size: Tasks are lightweight, so we want large chunks (e.g., 100-500)
        # to amortize the process start-up cost.
        chunk_size = 400
        rr_chunks = [
            rr_tasks[i : i + chunk_size] for i in range(0, len(rr_tasks), chunk_size)
        ]

        # Using 'spawn' context is default/safer on Mac, but ProcessPoolExecutor handles it.
        # Max workers = 12 (leave 2 cores for OS/UI)
        with ProcessPoolExecutor(max_workers=12, initializer=init_worker) as executor:
            future_to_chunk = {
                executor.submit(process_rr_chunk, chunk): chunk for chunk in rr_chunks
            }

            for future in as_completed(future_to_chunk):
                try:
                    results = future.result()
                    # Aggregate results back into the main graph
                    for r1_id, r2_id, feasible_route in results:
                        graph.add_rr_edge(r1_id, r2_id, feasible_route)
                except Exception as exc:
                    raise exc
                    logger.error(f"RR Worker generated an exception: {exc}")

        # --- 4. REPEAT FOR RV (Same Logic) ---

        rv_tasks = []
        if veh_list:
            veh_locs = [v.available_at_node for v in veh_list]
            veh_times = np.array([v.available_from_time for v in veh_list])
            req_times = np.array([r.request_time for r in req_list])

            rv_matrix = self._network.get_travel_time_matrix(veh_locs, req_origins)

            # Broadcasting logic
            wait_times = (veh_times[:, None] + rv_matrix) - req_times[None, :]
            rv_mask = wait_times <= OMEGA
            v_indices, r_indices = np.where(rv_mask)

            for v_i, r_i in zip(v_indices, r_indices):
                # Optional: Add the "Already Onboard" check here or in worker
                rv_tasks.append(RVTask(veh_list[v_i], req_list[r_i]))

        logger.info(f"Matrix pruning kept {len(rv_tasks)} RV pairs")

        # # --- PREFETCH FOR RV ---
        nodes_to_prefetch = set()
        for task in rv_tasks:
            nodes_to_prefetch.add(task.vehicle.available_at_node)
            nodes_to_prefetch.add(task.req.origin_node)
            nodes_to_prefetch.add(task.req.destination_node)
            for stop in task.vehicle.remaining_stops_to_serve:
                nodes_to_prefetch.add(stop.node)

        if hasattr(self._network, "prefetch_all_pairs") and nodes_to_prefetch:
            self._network.prefetch_all_pairs(list(nodes_to_prefetch))

        # Get cache copy to share
        l1_cache_copy = (
            self._network.get_l1_cache()
            if hasattr(self._network, "get_l1_cache")
            else self._network._l1_cache.copy()
        )

        # Calculate chunk size to distribute work evenly among 12 workers
        # We use max(1, ...) to avoid division by zero or empty chunks
        n_workers = 12
        total_tasks = len(rv_tasks)
        chunk_size = max(1, (total_tasks + n_workers - 1) // n_workers)
        chunk_size = 400

        rv_chunks = [
            rv_tasks[i : i + chunk_size] for i in range(0, len(rv_tasks), chunk_size)
        ]
        with ProcessPoolExecutor(
            max_workers=12, initializer=init_worker, initargs=(l1_cache_copy,)
        ) as executor:
            future_to_chunk = {
                executor.submit(process_rv_chunk, chunk): chunk for chunk in rv_chunks
            }
            for future in as_completed(future_to_chunk):
                try:
                    results = future.result()
                    for req_id, veh_id, feasible_route in results:
                        graph.add_rv_edge(req_id, veh_id, feasible_route)
                except Exception as exc:
                    raise exc
                    logger.error(f"RV Worker generated an exception: {exc}")

        return graph


class RVGraphBuilder(IRVGraphBuilder):
    def __init__(
        self,
        route_planner: ISingleVehicleRoutePlanner,
        network: INetworkService,
    ) -> None:
        super().__init__(route_planner)
        self._network = network

    def build_graph(
        self, requests: set[Request], vehicles: set[VehicleAvailability]
    ) -> RVGraph:
        graph = RVGraph()
        req_list = list(requests)
        veh_list = list(vehicles)

        # --- PHASE 1: VECTORIZED PRUNING (RR & RV) ---

        # 1a. RR Pruning
        req_origins = [r.origin_node for r in req_list]
        rr_matrix = self._network.get_travel_time_matrix(req_origins, req_origins)

        # Upper triangle logic (excluding diagonal k=1)
        rr_mask = np.triu((rr_matrix <= OMEGA) | (rr_matrix.T <= OMEGA), k=1)
        rr_indices = np.where(
            rr_mask
        )  # Returns tuple of arrays (row_indices, col_indices)

        # 1b. RV Pruning
        rv_indices = ([], [])  # Default empty if no vehicles
        if veh_list:
            veh_locs = [v.available_at_node for v in veh_list]

            # Use broadcasting for fast time checking
            veh_times = np.array([v.available_from_time for v in veh_list])[:, None]
            req_times = np.array([r.request_time for r in req_list])[None, :]

            rv_matrix = self._network.get_travel_time_matrix(veh_locs, req_origins)

            wait_times = (veh_times + rv_matrix) - req_times
            rv_indices = np.where(wait_times <= OMEGA)

        # --- PHASE 2: TASK GENERATION (List Comprehensions) ---

        # Fast generation of task objects
        rr_tasks = [RRTask(req_list[i], req_list[j]) for i, j in zip(*rr_indices)]

        rv_tasks = [RVTask(veh_list[i], req_list[j]) for i, j in zip(*rv_indices)]

        total_rr = len(rr_tasks)
        total_rv = len(rv_tasks)
        logger.info(f"Pruning kept: {total_rr} RR pairs, {total_rv} RV pairs.")

        # --- PHASE 3: UNIFIED PREFETCHING ---

        # Identify ALL unique nodes required for ALL tasks
        # Using a set comprehension is faster than looping
        nodes_to_prefetch = {r.origin_node for r in req_list} | {
            r.destination_node for r in req_list
        }

        # Add vehicle nodes
        for v in veh_list:
            nodes_to_prefetch.add(v.available_at_node)
            for stop in v.remaining_stops_to_serve:
                nodes_to_prefetch.add(stop.node)

        # Single DB Hit
        if hasattr(self._network, "prefetch_all_pairs") and nodes_to_prefetch:
            self._network.prefetch_all_pairs(list(nodes_to_prefetch))

        # --- PHASE 4: COMBINED PARALLEL EXECUTION ---

        # Prepare L1 cache snapshot ONCE to share with all workers
        l1_cache_copy = (
            self._network.get_l1_cache()
            if hasattr(self._network, "get_l1_cache")
            else self._network._l1_cache.copy()
        )

        # Create combined chunks
        # Structure: (TaskType, ChunkData)
        chunk_size = 400

        # Generator for chunks to avoid creating massive intermediate lists
        def chunk_generator():
            # Yield RR chunks
            for i in range(0, total_rr, chunk_size):
                yield ("RR", rr_tasks[i : i + chunk_size])
            # Yield RV chunks
            for i in range(0, total_rv, chunk_size):
                yield ("RV", rv_tasks[i : i + chunk_size])

        with ProcessPoolExecutor(
            max_workers=12, initializer=init_worker, initargs=(l1_cache_copy,)
        ) as executor:

            # Map futures to their type so we know how to process results
            future_map = {}

            for task_type, chunk in chunk_generator():
                if task_type == "RR":
                    future = executor.submit(process_rr_chunk, chunk)  # type: ignore
                else:
                    future = executor.submit(process_rv_chunk, chunk)  # type: ignore
                future_map[future] = task_type

            # Process results as they complete
            for future in as_completed(future_map):
                task_type = future_map[future]
                try:
                    results = future.result()

                    if task_type == "RR":
                        for r1_id, r2_id, feasible_route in results:
                            graph.add_rr_edge(r1_id, r2_id, feasible_route)
                    else:  # RV
                        for req_id, veh_id, feasible_route in results:
                            graph.add_rv_edge(req_id, veh_id, feasible_route)

                except Exception as exc:
                    logger.error(f"{task_type} Worker exception: {exc}")
                    raise exc

        return graph
