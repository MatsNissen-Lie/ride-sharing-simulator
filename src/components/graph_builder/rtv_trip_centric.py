"""Trip-centric RTV graph construction.

Builds trips once for the whole fleet instead of per vehicle. Each trip keeps
the set of vehicles that can serve it; a size-k candidate is checked only on
vehicles that can serve every one of its (k-1)-subsets. Produces the same RTV
graph as the vehicle-centric builder, with fewer repeated joins.
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


class TripCentricRTVGraphBuilder(IRTVGraphBuilder):
    """Builds trips globally and tracks candidate vehicles per trip."""

    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: set[VehicleAvailability],
        requests: set[Request],
    ) -> RTVGraph:
        rtv_graph = RTVGraph()
        req_map = {r.id: r for r in requests}
        veh_map = {v.vehicle_id: v for v in vehicles}
        time_matrix = self._network.get_l1_cache()

        def trip_limit(vehicle: VehicleAvailability) -> int:
            seats = vehicle.capacity - len(vehicle.passengers_onboard)
            return seats + MAX_REQUESTS_ABOVE_CAPACITY

        # trip -> vehicles that can serve it
        trip_vehicles: dict[TripId, frozenset[str]] = {}
        trips_by_size: dict[int, set[TripId]] = defaultdict(set)

        # T_1: one trip per request, with every RV edge.
        for r_id, veh_ids in rv_graph.request_vehicle.items():
            trip: TripId = frozenset([r_id])
            valid = set()
            for vid in veh_ids:
                rv_route = rv_graph.rv_details.get((r_id, vid))
                if rv_route and trip_limit(veh_map[vid]) >= 1:
                    rtv_graph.add_trip_for_vehicle(trip, vid, rv_route)
                    valid.add(vid)
            if valid:
                trips_by_size[1].add(trip)
                trip_vehicles[trip] = frozenset(valid)

        # T_2: RR-adjacent pairs, checked on vehicles that can serve both.
        for r1, r2, common in rv_graph.get_all_unique_shareable_pairs_with_vehicles():
            trip = frozenset((r1, r2))
            reqs = [req_map[r] for r in sorted(trip)]
            valid = set()
            for vid in common:
                vehicle = veh_map[vid]
                if trip_limit(vehicle) < 2:
                    continue
                result = TripSolver.solve(reqs, vehicle, time_matrix)
                if result.is_feasible:
                    rtv_graph.add_trip_for_vehicle(
                        trip, vid, cast(FeasibleRoute, result)
                    )
                    valid.add(vid)
            if valid:
                trips_by_size[2].add(trip)
                trip_vehicles[trip] = frozenset(valid)

        max_k = max((trip_limit(v) for v in vehicles), default=0)
        # T_k for k >= 3: join trips sharing their first k-2 sorted requests.
        for k in range(3, max_k + 1):
            if not trips_by_size[k - 1]:
                break
            prev = sorted(sorted(t) for t in trips_by_size[k - 1])
            for i, trip_a in enumerate(prev):
                for trip_b in prev[i + 1 :]:
                    if trip_a[:-1] != trip_b[:-1]:
                        break
                    r_x, r_y = trip_a[-1], trip_b[-1]
                    candidates = set(trip_vehicles[frozenset(trip_a)])
                    candidates &= trip_vehicles[frozenset(trip_b)]
                    prefix = trip_a[:-1]
                    # Sub-trip lemma, per vehicle: intersect over every (k-1)-subset.
                    for idx in range(len(prefix)):
                        if not candidates:
                            break
                        sub = frozenset(prefix[:idx] + prefix[idx + 1 :] + [r_x, r_y])
                        candidates &= trip_vehicles.get(sub, frozenset())
                    if not candidates:
                        continue

                    trip_list = trip_a + [r_y]
                    trip = frozenset(trip_list)
                    reqs = [req_map[r] for r in trip_list]
                    valid = set()
                    for vid in candidates:
                        vehicle = veh_map[vid]
                        if trip_limit(vehicle) < k:
                            continue
                        result = TripSolver.solve(reqs, vehicle, time_matrix)
                        if result.is_feasible:
                            rtv_graph.add_trip_for_vehicle(
                                trip, vid, cast(FeasibleRoute, result)
                            )
                            valid.add(vid)
                    if valid:
                        trips_by_size[k].add(trip)
                        trip_vehicles[trip] = frozenset(valid)

        return rtv_graph
