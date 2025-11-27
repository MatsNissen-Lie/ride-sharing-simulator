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
    Passenger,
    Request,
)
from src.interfaces.graphs import RVGraph
from src.interfaces.services import INetworkService

from src.components.graph_builder.workers import (
    process_rr_chunk,
    process_rv_chunk,
    RRTask,
    RVTask,
)

logger = get_logger("rv_graph_builder")


class RVGraphBuilder(IRVGraphBuilder):
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
        chunk_size = 200
        rr_chunks = [
            rr_tasks[i : i + chunk_size] for i in range(0, len(rr_tasks), chunk_size)
        ]

        # Using 'spawn' context is default/safer on Mac, but ProcessPoolExecutor handles it.
        # Max workers = 12 (leave 2 cores for OS/UI)
        with ProcessPoolExecutor(max_workers=12) as executor:
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

        rv_chunks = [
            rv_tasks[i : i + chunk_size] for i in range(0, len(rv_tasks), chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=12) as executor:
            future_to_chunk = {
                executor.submit(process_rv_chunk, chunk): chunk for chunk in rv_chunks
            }
            for future in as_completed(future_to_chunk):
                try:
                    results = future.result()
                    for req_id, veh_id, feasible_route in results:
                        graph.add_rv_edge(req_id, veh_id, feasible_route)
                except Exception as exc:
                    logger.error(f"RV Worker generated an exception: {exc}")

        return graph

    @staticmethod
    def _request_already_onboard(
        request: Request, passengers: Iterable[Passenger] | None
    ) -> bool:
        if passengers is None:
            return False
        for passenger in passengers or []:
            if request.id == passenger.original_request.id:
                return True
        return False

    @staticmethod
    def _vehicle_identifier(vehicle: VehicleAvailability) -> str:
        return vehicle.vehicle_id
