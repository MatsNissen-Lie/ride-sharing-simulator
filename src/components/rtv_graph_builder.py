from config.config import MAX_REQUESTS, MAX_REQUESTS_ABOVE_CAPACITY
from src.components.rv_graph_builder import RVGraph
from src.interfaces import FeasibleRoute
from collections import defaultdict
from typing import Set, Dict, cast

from config import get_logger
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    # FeasibleRoute,
    Request,
)
from src.interfaces.graphs import RTVGraph, TripId
from src.interfaces.services import IRTVGraphBuilder


logger = get_logger(__name__)


class RTVGraphBuilder(IRTVGraphBuilder):
    """
    Concrete implementation of IRTVGraphBuilder.

    Implements Algorithm 1 in Alonso-Mora et al. for a single batch:
    - For each vehicle v:
      * Build T_1 (trips of size 1) from R-V edges in the RV-graph.
      * Build T_2 (trips of size 2) using R-R edges + travel(v, {r1,r2}).
      * For k >= 3, build T_k from T_{k-1} using the sub-trip lemma and
        travel(v, T) feasibility.
    """

    def __init__(self, route_planner: ISingleVehicleRoutePlanner) -> None:
        super().__init__(route_planner)

    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: Set[VehicleAvailability],
        requests: set[Request],
        max_trip_size: int,
    ) -> RTVGraph:
        rtv_graph = RTVGraph()

        # Convenience: map vehicle_id -> VehicleAvailability
        vehicles_by_id: Dict[str, VehicleAvailability] = {
            v.vehicle_id: v for v in vehicles
        }

        requests_by_id: dict[str, Request] = {r.id: r for r in requests}

        # For each vehicle, we build trips incrementally (Algorithm 1)
        for v_id, v_avail in vehicles_by_id.items():

            self._build_trips_for_vehicle(
                v_id=v_id,
                v_avail=v_avail,
                rv_graph=rv_graph,
                requests_by_id=requests_by_id,
                max_trip_size=max_trip_size,
                rtv_graph=rtv_graph,
            )

        return rtv_graph

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trips_for_vehicle(
        self,
        v_id: str,
        v_avail: VehicleAvailability,
        rv_graph: RVGraph,
        requests_by_id: Dict[str, Request],
        max_trip_size: int,
        rtv_graph: RTVGraph,
    ) -> None:
        """
        Implements Algorithm 1 for a single vehicle v.
        """
        m = len(v_avail.passengers_onboard)
        # Capacity check: limit new requests to available seats
        # Also hard-cap at 4 per Alonso-Mora logic if needed, or use vehicle capacity
        # available_seats = max(v_avail.capacity - m, MAX_REQUESTS_ABOVE_CAPACITY)
        available_seats = v_avail.capacity - m
        limit_k = min(max_trip_size, available_seats, MAX_REQUESTS)

        if limit_k <= 0:
            return

        # T_k per vehicle: k -> set of TripId
        T_by_size: Dict[int, Set[TripId]] = defaultdict(set)

        # --- [Step 1] Trips of size 1 ---
        # Use RV-graph R-V edges
        relevant_requests = rv_graph.get_requests_for_vehicle(v_id)

        for r_id in relevant_requests:
            trip: TripId = frozenset({r_id})
            rv_route = rv_graph.rv_details.get((r_id, v_id))

            if rv_route:
                T_by_size[1].add(trip)
                rtv_graph.add_trip_for_vehicle(trip, v_id, rv_route)

        if limit_k == 1 or not T_by_size[1]:
            return

        # --- [Step 2] Trips of size 2 ---
        # Only pairs {r1}, {r2} where both are valid individually AND shareable
        single_trips = list(T_by_size[1])

        for i in range(len(single_trips)):
            for j in range(i + 1, len(single_trips)):
                t1 = single_trips[i]
                t2 = single_trips[j]
                # Unpack single IDs
                r1_id = next(iter(t1))
                r2_id = next(iter(t2))

                # Pruning: Check R-R edge in RVGraph
                if r2_id not in rv_graph.request_request.get(r1_id, set()):
                    continue

                # Candidate trip
                trip_2: TripId = frozenset({r1_id, r2_id})

                # Feasibility check
                reqs = {requests_by_id[r1_id], requests_by_id[r2_id]}
                result = self._route_planner.check_feasibility(reqs, vehicle=v_avail)

                if result.is_feasible:
                    feasible = cast(FeasibleRoute, result)
                    T_by_size[2].add(trip_2)
                    rtv_graph.add_trip_for_vehicle(trip_2, v_id, feasible)

        if limit_k == 2 or not T_by_size[2]:
            return

        # TODO: THINK I HAVE BEEN THINKING ABOUT HOW WE BUILD TRIPS.
        # Right now we build trips per vehicle, but we could build trips globally
        # and then assign them to vehicles that can serve them.
        # Do we reuse the information from smaller trips across vehicles?
        # DO WE reuse the TRIPS OF SIZE n-1 when building trips of size n

        # --- [Step 3] Trips of size k >= 3 ---
        for k in range(3, limit_k + 1):
            prev_trips = T_by_size[k - 1]
            if not prev_trips:
                break

            candidates: Set[TripId] = set()

            # Generate candidates: T_{k-1} U {r}
            for base_trip in prev_trips:
                for single_trip in T_by_size[1]:
                    r_new = next(iter(single_trip))

                    # If r is already in the trip, skip
                    if r_new in base_trip:
                        continue

                    # Construct new trip
                    new_trip = base_trip | {r_new}

                    # Optimization: We rely on the 'candidates' set to deduplicate.
                    # However, before adding to candidates, we can perform the
                    # Sub-trip Lemma check RIGHT HERE to fail fast,
                    # or keep it in the second loop as you have it.
                    # Keeping it separate (as you have) is often cleaner for debugging.
                    candidates.add(new_trip)

            if not candidates:
                break

            for trip in candidates:
                # Sub-trip Lemma Check:
                # We must verify that ALL subsets of size k-1 are valid valid trips.
                is_valid_candidate = True

                for r_removed in trip:
                    # OPTIMIZATION: We know at least one subset is valid (the one that generated this candidate).
                    # If you want to be strictly rigorous, check them all.
                    # But practically, usually checking the *other* subsets is what catches invalid trips.

                    sub_trip = trip - {r_removed}
                    if sub_trip not in prev_trips:
                        is_valid_candidate = False
                        break

                if not is_valid_candidate:
                    continue

                # Feasibility Check (Expensive)
                # Only done if the structural (graph) checks pass
                reqs = {requests_by_id[r] for r in trip}
                result = self._route_planner.check_feasibility(reqs, vehicle=v_avail)

                if result.is_feasible:
                    feasible = cast(FeasibleRoute, result)
                    T_by_size[k].add(trip)
                    rtv_graph.add_trip_for_vehicle(trip, v_id, feasible)
