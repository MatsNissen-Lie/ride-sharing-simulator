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

    def _build_trips_for_vehicle(  # TODO: I NEED TO MAKE THIS FUNCTION WORK
        self,
        v_id: str,
        v_avail: VehicleAvailability,
        rv_graph: RVGraph,
        requests_by_id: Dict[str, Request],
        max_trip_size: int,
        rtv_graph: RTVGraph,
    ) -> None:
        """
        Implements Algorithm 1 for a single vehicle v:

        - Build T_1 from e(r,v) edges in RV-graph (and reuse the stored
          FeasibleRoute from rv_graph.rv_details).
        - Build T_2 from pairs in T_1 that are adjacent in the RV R-R graph.
        - Build T_k using sub-trip lemma and travel(v, T) feasibility.
        """
        m = len(v_avail.passengers_onboard)

        num_requests_to_add = min(max_trip_size, 4, v_avail.capacity - m)

        if num_requests_to_add <= 0:
            # Vehicle is effectively full, skip it
            return

        # T_k per vehicle: k -> set of TripId
        T_by_size: Dict[int, Set[TripId]] = defaultdict(set)

        # --- [Add trips of size one]  (Algorithm 1, lines 324–328) ---
        # For all e(r, v) edges in RV-graph
        for r_id, vehicle_ids in rv_graph.request_vehicle.items():
            if v_id not in vehicle_ids:
                continue

            trip: TripId = frozenset({r_id})
            # We already have the single-request FeasibleRoute in rv_graph.rv_details
            rv_route = rv_graph.rv_details.get((r_id, v_id))
            if rv_route is None:
                # This should not happen if RVGraphBuilder stored details consistently.
                logger.warning(
                    "Missing FeasibleRoute for (r=%s, v=%s) in RVGraph.rv_details",
                    r_id,
                    v_id,
                )
                continue

            T_by_size[1].add(trip)
            rtv_graph.add_trip_for_vehicle(trip, v_id, rv_route)

        if num_requests_to_add == 1 or not T_by_size[1]:
            # Either no single trips for this vehicle, or we are limited to size 1.
            return

        # --- [Add trips of size two]  (Algorithm 1, lines 330–338) ---

        # We only consider pairs {r1}, {r2} in T_1 such that e(r1, r2) exists in RV R-R graph.
        # For each such pair we check travel(v, {r1, r2}) via the route planner.
        size1_trips = list(T_by_size[1])

        for i in range(len(size1_trips)):
            for j in range(i + 1, len(size1_trips)):
                t1 = size1_trips[i]
                t2 = size1_trips[j]
                (r1_id,) = tuple(t1)
                (r2_id,) = tuple(t2)

                # Check if R-R edge exists in RVGraph
                if r2_id not in rv_graph.request_request.get(r1_id, set()):
                    continue

                trip_2: TripId = frozenset({r1_id, r2_id})

                # Build Request objects set for {r1, r2}
                try:
                    r1 = requests_by_id[r1_id]
                    r2 = requests_by_id[r2_id]
                except KeyError:
                    logger.error(
                        "Request id not found in rv_graph.requests_by_id: %s or %s",
                        r1_id,
                        r2_id,
                    )
                    continue

                result = self._route_planner.check_feasibility(
                    {r1, r2},
                    vehicle=v_avail,
                )

                if not result.is_feasible:
                    continue

                feasible = cast(FeasibleRoute, result)
                T_by_size[2].add(trip_2)
                rtv_graph.add_trip_for_vehicle(trip_2, v_id, feasible)

        if max_trip_size == 2 or not T_by_size[2]:
            return

        # --- [Add trips of size k >= 3]  (Algorithm 1, lines 340–357) ---

        # Continue growing trips using the sub-trip lemma up to max_trip_size.

        for k in range(3, max_trip_size + 1):
            prev_trips = list(T_by_size[k - 1])
            if not prev_trips:
                break  # no larger trips possible for this vehicle

            candidates: Set[TripId] = set()

            # Generate candidate unions T_1 ∪ T_2 with |union| = k
            for t1 in prev_trips:
                for t2 in prev_trips:
                    union_trip = t1 | t2
                    if len(union_trip) == k:
                        candidates.add(frozenset(union_trip))

            if not candidates:
                break

            for trip in candidates:
                # Sub-feasibility check: all (k-1)-size subtrips must be in T_{k-1}
                all_subtrips_ok = True
                for r_id in trip:
                    subtrip = frozenset(trip - {r_id})
                    if subtrip not in T_by_size[k - 1]:
                        all_subtrips_ok = False
                        break

                if not all_subtrips_ok:
                    continue

                # Build Request objects set for this trip
                try:
                    reqs = {requests_by_id[r_id] for r_id in trip}
                except KeyError as e:
                    logger.error(
                        "Request id %s not found in rv_graph.requests_by_id", e.args[0]
                    )
                    continue

                result = self._route_planner.check_feasibility(
                    reqs,
                    vehicle=v_avail,
                )
                if not result.is_feasible:
                    continue

                feasible = cast(FeasibleRoute, result)
                T_by_size[k].add(trip)
                rtv_graph.add_trip_for_vehicle(trip, v_id, feasible)
