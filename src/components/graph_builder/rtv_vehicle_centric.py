"""Vehicle-centric RTV graph construction (Algorithm 1 in Alonso-Mora et al.).

For each vehicle independently:
  * T_1: requests with an RV edge to the vehicle.
  * T_2: pairs of those requests joined by an RR edge, checked with travel(v, T).
  * T_k (k >= 3): Apriori join of T_{k-1}; a candidate is only checked when all
    of its (k-1)-subsets are feasible for this vehicle (sub-trip lemma).
"""

from collections import defaultdict
from typing import cast

from config import get_logger
from config.config import MAX_REQUESTS_ABOVE_CAPACITY
from src.components.trip_solver import TripSolver
from src.interfaces import FeasibleRoute, Request, VehicleAvailability
from src.interfaces.graphs import RTVGraph, RVGraph, TripId
from src.interfaces.services import IRTVGraphBuilder

logger = get_logger(__name__)


class VehicleCentricRTVGraphBuilder(IRTVGraphBuilder):
    """Builds trips per vehicle, following the paper's Algorithm 1."""

    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: set[VehicleAvailability],
        requests: set[Request],
    ) -> RTVGraph:
        rtv_graph = RTVGraph()
        requests_by_id = {r.id: r for r in requests}
        time_matrix = self._network.get_l1_cache()
        for vehicle in vehicles:
            self._build_trips_for_vehicle(
                vehicle, rv_graph, requests_by_id, rtv_graph, time_matrix
            )
        return rtv_graph

    def _build_trips_for_vehicle(
        self,
        vehicle: VehicleAvailability,
        rv_graph: RVGraph,
        requests_by_id: dict[str, Request],
        rtv_graph: RTVGraph,
        time_matrix,
    ) -> None:
        v_id = vehicle.vehicle_id
        limit_k = (
            vehicle.capacity
            - len(vehicle.passengers_onboard)
            + MAX_REQUESTS_ABOVE_CAPACITY
        )
        if limit_k <= 0:
            return

        trips_by_size: dict[int, set[TripId]] = defaultdict(set)

        # T_1: one trip per RV edge.
        singles = set()
        for r_id in rv_graph.get_requests_for_vehicle(v_id):
            rv_route = rv_graph.rv_details.get((r_id, v_id))
            if rv_route:
                trip: TripId = frozenset({r_id})
                trips_by_size[1].add(trip)
                rtv_graph.add_trip_for_vehicle(trip, v_id, rv_route)
                singles.add(r_id)

        if limit_k == 1 or not singles:
            return

        # T_2: RR-adjacent pairs of requests this vehicle can serve.
        for r1 in sorted(singles):
            for r2 in rv_graph.request_request.get(r1, set()):
                if r2 not in singles or r2 <= r1:
                    continue
                trip = frozenset({r1, r2})
                reqs = [requests_by_id[r] for r in sorted(trip)]
                result = TripSolver.solve(reqs, vehicle, time_matrix)
                if result.is_feasible:
                    trips_by_size[2].add(trip)
                    rtv_graph.add_trip_for_vehicle(
                        trip, v_id, cast(FeasibleRoute, result)
                    )

        # T_k for k >= 3: join trips sharing their first k-2 sorted requests.
        for k in range(3, limit_k + 1):
            if not trips_by_size[k - 1]:
                break
            prev = sorted(sorted(t) for t in trips_by_size[k - 1])
            for i, trip_a in enumerate(prev):
                for trip_b in prev[i + 1 :]:
                    if trip_a[:-1] != trip_b[:-1]:
                        break
                    r_x, r_y = trip_a[-1], trip_b[-1]
                    prefix = trip_a[:-1]
                    # Sub-trip lemma: every (k-1)-subset must already be feasible.
                    if any(
                        frozenset(prefix[:idx] + prefix[idx + 1 :] + [r_x, r_y])
                        not in trips_by_size[k - 1]
                        for idx in range(len(prefix))
                    ):
                        continue
                    trip_list = trip_a + [r_y]
                    trip = frozenset(trip_list)
                    reqs = [requests_by_id[r] for r in trip_list]
                    result = TripSolver.solve(reqs, vehicle, time_matrix)
                    if result.is_feasible:
                        trips_by_size[k].add(trip)
                        rtv_graph.add_trip_for_vehicle(
                            trip, v_id, cast(FeasibleRoute, result)
                        )
