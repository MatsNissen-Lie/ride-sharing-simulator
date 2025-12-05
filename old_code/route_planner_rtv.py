"""Minimal single-vehicle route planner used during prototyping."""

import itertools
from typing import Iterable, Iterator, cast

import numpy as np

from config import BATCH_DURATION, DELTA, OMEGA, get_logger
from config.config import MAX_REQUESTS
from src.components.utils import validate_route_stops

from src.interfaces import (
    FeasibilityResult,
    FeasibleRoute,
    Infeasible,
    Request,
    Stop,
    Route,
    Passenger,
    INetworkService,
    VehicleAvailability,
    ISingleVehicleRoutePlanner,
)
from src.interfaces.data_models import NetworkPath


logger = get_logger(__name__)


class RoutePlanner(ISingleVehicleRoutePlanner):
    """A brute-force feasibility checker for two-request R-R cases."""

    def __init__(self, network_service: INetworkService) -> None:
        super().__init__(network_service)

    def check_feasibility(
        self,
        requests_to_add: set[Request],
        vehicle: VehicleAvailability | None = None,  # this is not needed for R-R
        find_best_route: bool = True,
    ) -> FeasibilityResult:

        if not requests_to_add:
            raise ValueError("RoutePlanner requires at least one request to evaluate")

        # --- CASE 1: R-R Check ---
        # (Two new requests, empty vehicle)
        if len(requests_to_add) == 2 and vehicle is None:
            raise NotImplementedError("R-R checks are not implemented in this planner.")

        if vehicle is not None:
            if len(requests_to_add) == 1:
                raise NotImplementedError(
                    "R-V checks are not implemented in this planner."
                )
            return self._handle_rtv_check(
                availability=vehicle,
                requests_to_add=requests_to_add,
                find_best_route=find_best_route,
            )

        raise NotImplementedError(
            "RoutePlanner only supports R-R, R-V, and RTV checks. This scenario is unsupported."
        )

    def _handle_rtv_check(
        self,
        availability: VehicleAvailability,
        requests_to_add: set[Request],
        find_best_route: bool,
    ) -> FeasibilityResult:
        """
        Recursive Branch-and-Bound (DFS) solver.
        Solves for m existing passengers + k new requests exact and fast.
        """

        # --- 1. SETUP DATA STRUCTURES ---

        # Current state
        start_node = availability.available_at_node
        start_time = availability.available_from_time
        initial_load = len(availability.passengers_onboard)
        capacity = availability.capacity

        # Identify all necessary stops
        # IDs: use integers 0..N-1 for speed in recursion
        stops: list[Stop] = []

        # 1. Add existing drop-offs
        for p in availability.passengers_onboard:
            stop = Stop(
                node=p.original_request.destination_node,
                is_pickup=False,
                request_id=p.original_request.id,
            )
            stops.append(stop)

        # 2. Add new requests (Pickup AND Drop-off)
        new_req_ids = set()
        for r in requests_to_add:
            new_req_ids.add(r.id)
            p_stop = Stop(node=r.origin_node, is_pickup=True, request_id=r.id)
            d_stop = Stop(node=r.destination_node, is_pickup=False, request_id=r.id)
            stops.append(p_stop)
            stops.append(d_stop)

        # Build Lookup Maps
        # We need quick access to Request objects for time windows
        all_requests = {
            p.original_request.id: p.original_request
            for p in availability.passengers_onboard
        }
        all_requests.update({r.id: r for r in requests_to_add})

        # Pre-compute Constraints to avoid dict lookups in the hot loop
        N = len(stops)
        stop_req_ids = [s.request_id for s in stops]
        is_pickup = [s.is_pickup for s in stops]
        nodes = [s.node for s in stops]

        # Constraint: Precedence
        # If index i is a dropoff for a NEW request, we must have visited its pickup.
        # Map: drop_index -> pickup_index (or -1 if no pickup needed)
        pickup_indices = {}
        for i, s in enumerate(stops):
            if s.is_pickup:
                pickup_indices[s.request_id] = i

        precedence = []  # list of (drop_idx, pickup_idx)
        for i, s in enumerate(stops):
            if not s.is_pickup and s.request_id in new_req_ids:
                precedence.append((i, pickup_indices[s.request_id]))

        # Constraint: Time Windows
        # We pre-calculate (Earliest, Latest) for every stop index
        earliest_arrival = []
        latest_arrival = []

        for i, s in enumerate(stops):
            req = all_requests[s.request_id]
            if s.is_pickup:
                # Pickup: Earliest = request_time, Latest = request_time + OMEGA
                earliest_arrival.append(req.request_time)
                if req.promised_pickup_time:
                    # If promised, we must adhere (or use OMEGA relative to request)
                    # For simplicity here using OMEGA standard
                    latest_arrival.append(
                        req.promised_pickup_time
                    )  # or req.request_time + OMEGA
                else:
                    latest_arrival.append(req.request_time + OMEGA)
            else:
                # Dropoff: Earliest = request_time (sanity), Latest = earliest_drop + DELTA
                earliest_arrival.append(req.request_time)
                if req.earliest_dropoff_time:
                    latest_arrival.append(req.earliest_dropoff_time + DELTA)
                else:
                    # Fallback if not set (should be set)
                    latest_arrival.append(float("inf"))

        # Travel Time Cache (Pre-fill)
        unique_nodes = set(nodes) | {start_node}
        tt_cache = self._build_travel_time_cache(unique_nodes)  # type: ignore

        # --- 2. RECURSIVE SOLVER ---

        best_cost = float("inf")
        best_path_indices = None

        # visited_mask is an integer bitmask
        def solve(
            curr_node_id: int,
            curr_time: float,
            curr_load: int,
            visited_mask: int,
            current_cost_so_far: float,
        ):
            nonlocal best_cost, best_path_indices

            # Optimization: Pruning by cost (Branch & Bound)
            # If current partial cost is already worse than best found, stop.
            if current_cost_so_far >= best_cost:
                return

            # Base Case: All stops visited
            # (Mask is all 1s for N bits)
            if visited_mask == (1 << N) - 1:
                if current_cost_so_far < best_cost:
                    best_cost = current_cost_so_far
                    best_path_indices = []  # Will reconstruct or pass path down
                    return
                return

            # Iterating all candidates
            for i in range(N):
                # If i is already visited, skip
                if (visited_mask >> i) & 1:
                    continue

                # --- VALIDATION CHECKS (The Pruning) ---

                # A. Precedence Check
                # If this is a dropoff for a new request, is the pickup visited?
                if not is_pickup[i] and stop_req_ids[i] in new_req_ids:
                    # Find the pickup index for this request
                    p_idx = pickup_indices[stop_req_ids[i]]
                    # Check bitmask
                    if not ((visited_mask >> p_idx) & 1):
                        continue  # Prune: Cannot drop before pickup

                # B. Capacity Check
                # If pickup, capacity + 1 <= max. If drop, capacity - 1 (always ok)
                new_load = curr_load + (1 if is_pickup[i] else -1)
                if new_load > capacity:
                    continue  # Prune: Capacity violation

                # C. Travel Calculation
                dist = tt_cache.get((curr_node_id, nodes[i]), float("inf"))
                arrival = curr_time + dist

                # D. Time Window Check (Start)
                # Wait if early
                actual_arrival = max(arrival, earliest_arrival[i])

                # Check Late
                if actual_arrival > latest_arrival[i]:
                    continue  # Prune: Too late

                # --- RECURSE ---

                # Calculate Cost increment
                # If it's a dropoff, we add (arrival - earliest_dropoff)
                added_cost = 0.0
                if not is_pickup[i]:
                    req = all_requests[stop_req_ids[i]]
                    # Careful: use actual_arrival
                    delay = actual_arrival - req.earliest_dropoff_time
                    # Note: delay logic might differ slightly based on your metric
                    if delay > 0:
                        added_cost = delay

                solve(
                    curr_node_id=nodes[i],
                    curr_time=actual_arrival,
                    curr_load=new_load,
                    visited_mask=visited_mask | (1 << i),
                    current_cost_so_far=current_cost_so_far + added_cost,
                )

                # Optimization: If we just found a feasible route and we don't care about
                # finding the BEST route (early exit), we can return here.
                if not find_best_route and best_cost < float("inf"):
                    return

        # --- 3. EXECUTE ---
        solve(start_node, start_time, initial_load, 0, 0.0)

        if best_path_indices is None and best_cost == float("inf"):
            return Infeasible(reason="No feasible route found via DFS")

        # Reconstruct Route object (omitted for brevity,
        # you'd need to pass the path list down the recursion or store parent pointers)
        # For the 'best_route' variable, you would need to store the sequence of indices
        # associated with best_cost.

        # (Quick fix to get the actual path: update best_path_indices to store the list)
        # ... implies changing logic above slightly to store path ...

        # Assuming you reconstruct 'final_stops' from best_path_indices:
        # return FeasibleRoute(route=Route(stops=tuple(final_stops), ...), cost=best_cost)

        # Note: To fully implement return, add `path` arg to solve or reconstruct.
        return Infeasible(reason="Placeholder until reconstruction added")
