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
from typing import cast
import numpy as np

from config import get_logger
from config.config import OMEGA
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    IRVGraphBuilder,
    FeasibleRoute,
    Passenger,
    Request,
)
from src.interfaces.graphs import RVGraph
from src.interfaces.services import INetworkService


logger = get_logger("old rv builder")


# --- The Builder class is now simpler ---


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
        self,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
    ) -> RVGraph:
        graph = RVGraph()

        # Convert sets to stable lists for indexing
        req_list = list(requests)
        veh_list = list(vehicles)

        n_req = len(req_list)
        n_veh = len(veh_list)

        if n_req == 0:
            return graph

        # --- 1. DATA PREPARATION ---

        # Extract scalar arrays for broadcasting
        # Shape: (N_req,)
        req_times = np.array([r.request_time for r in req_list])
        req_origins = [r.origin_node for r in req_list]

        # Shape: (N_veh,)
        veh_times = np.array([v.available_from_time for v in veh_list])
        veh_locs = [v.available_at_node for v in veh_list]

        # --- 2. FETCH MATRICES ---

        # RR Matrix (N_req x N_req)
        # Row i, Col j = Travel time from Req[i] -> Req[j]
        rr_matrix = self._network.get_travel_time_matrix(req_origins, req_origins)

        # RV Matrix (N_veh x N_req)
        # Row i, Col j = Travel time from Veh[i] -> Req[j]
        rv_matrix = self._network.get_travel_time_matrix(veh_locs, req_origins)

        # --- 3. VECTORIZED R-R PRUNING ---

        # We need to check pairs (i, j) where i < j.
        # Condition: Valid if (Time_i_j <= Omega) OR (Time_j_i <= Omega)

        # Create boolean masks
        forward_ok = rr_matrix <= OMEGA
        backward_ok = rr_matrix.T <= OMEGA  # Transpose checks j -> i

        # Combine: Either direction fits?
        # We only care about the upper triangle (k=1) to avoid duplicates and self-loops
        rr_mask = np.triu(forward_ok | backward_ok, k=1)

        # Get indices where mask is True
        # valid_rr_indices is a tuple of arrays: (row_indices, col_indices)
        rr_idx_i, rr_idx_j = np.where(rr_mask)

        logger.debug(
            f"RR Pruning: Found {len(rr_idx_i)} candidate pairs out of {n_req * (n_req - 1) // 2} possible."
        )

        # Iterate ONLY the feasible candidates
        for i, j in zip(rr_idx_i, rr_idx_j):
            r1, r2 = req_list[i], req_list[j]

            # The matrix check above guarantees we pass the spatial filter.
            # Now we run the full constraints (capacity, exact delays).
            result = self._route_planner.check_feasibility({r1, r2})

            if result.is_feasible:
                graph.add_rr_edge(r1.id, r2.id, cast(FeasibleRoute, result))

        # --- 4. VECTORIZED R-V PRUNING ---

        if n_veh > 0:
            # We want to check: (VehTime + TravelTime - ReqTime) <= OMEGA

            # Broadcasting shapes:
            # veh_times[:, None]  -> (N_veh, 1)
            # rv_matrix           -> (N_veh, N_req)
            # req_times[None, :]  -> (1, N_req)

            # Resulting 'arrival_times' is (N_veh, N_req)
            arrival_times = veh_times[:, None] + rv_matrix

            # Wait times: arrival - request_time
            wait_times = arrival_times - req_times[None, :]

            # Boolean Mask
            rv_mask = wait_times <= OMEGA

            # Get indices
            rv_idx_v, rv_idx_r = np.where(rv_mask)

            logger.debug(
                f"RV Pruning: Found {len(rv_idx_v)} candidate pairs out of {n_veh * n_req} possible."
            )

            for v_idx, r_idx in zip(rv_idx_v, rv_idx_r):
                vehicle = veh_list[v_idx]
                req = req_list[r_idx]

                # Check if request is already onboard (kept for safety)
                if self._request_already_onboard(req, vehicle.passengers_onboard):
                    continue

                # Run expensive check
                result = self._route_planner.check_feasibility({req}, vehicle)

                if result.is_feasible:
                    graph.add_rv_edge(
                        req.id, vehicle.vehicle_id, cast(FeasibleRoute, result)
                    )

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
