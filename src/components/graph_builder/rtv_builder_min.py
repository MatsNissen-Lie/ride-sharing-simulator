from collections import defaultdict
from dataclasses import dataclass
from typing import FrozenSet

from config import get_logger
from config.config import (
    MAX_REQUESTS_ABOVE_CAPACITY,
    VEHICLE_CAPACITY,
)
from src.components.graph_builder.trip_graph import TripGraph
from src.interfaces import (
    ISingleVehicleRoutePlanner,
    VehicleAvailability,
    Request,
)

from src.interfaces.graphs import RTVGraph, TripId, RVGraph
from src.interfaces.services import IRTVGraphBuilder

logger = get_logger(__name__)


@dataclass(frozen=True)
class GlobalTrip:
    """
    Represents a Trip (set of requests) that is feasible for at least one vehicle.
    Used for the Apriori expansion.
    """

    id: TripId
    candidate_vehicle_ids: FrozenSet[str]

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def request_ids(self) -> FrozenSet[str]:
        return self.id


class TripCentricRTVGraphBuilder(IRTVGraphBuilder):
    """
    RTV-Graph Builder using the Trip-Centric (Combinatorial) approach.
    """

    def __init__(self, route_planner: ISingleVehicleRoutePlanner) -> None:
        super().__init__(route_planner)
        self._network = route_planner._network

    def build_graph(
        self,
        rv_graph: RVGraph,
        vehicles: set[VehicleAvailability],
        requests: set[Request],
    ) -> RTVGraph:
        """
        Trip-Centric RTV Construction using the "Old Code" Solver Baseline.
        """

        # --- 1. SETUP ---
        rtv_graph = RTVGraph()

        # Fast Lookups
        req_map: dict[str, Request] = {r.id: r for r in requests}
        veh_map: dict[str, VehicleAvailability] = {v.vehicle_id: v for v in vehicles}

        # Global Trips: Size -> { TripID -> GlobalTrip }
        # This stores only VALID trips that survived feasibility checks

        T_by_size: dict[int, set[TripId]] = defaultdict(set)
        global_trips: dict[TripId, GlobalTrip] = defaultdict()

        limit_k = VEHICLE_CAPACITY + MAX_REQUESTS_ABOVE_CAPACITY

        valid_singles = set()

        # --- 2. PHASE 1: SIZE 1 TRIPS ---
        for r_id, veh_ids in rv_graph.request_vehicle.items():
            t_id: TripId = frozenset([r_id])
            valid_singles.add(r_id)
            for vid in veh_ids:
                rv_route = rv_graph.rv_details.get((r_id, vid))
                if rv_route:
                    rtv_graph.add_trip_for_vehicle(t_id, vid, rv_route)

            T_by_size[1].add(t_id)
            global_trips[t_id] = GlobalTrip(
                id=t_id,
                candidate_vehicle_ids=frozenset(veh_ids),
            )

        sorted_singles = sorted(list(valid_singles))

        for r1 in sorted_singles:
            # Get all neighbors of r1 from the RV-graph
            neighbors = rv_graph.request_request.get(r1, set())
            potential_r2 = [r2 for r2 in neighbors if r2 in valid_singles and r1 < r2]
            for r2 in potential_r2:
                candidate_trip = frozenset({r1, r2})
                # We need to look at the intersection of vehicles that served r1 and r2
                common_vehs = global_trips[
                    frozenset([r1])
                ].candidate_vehicle_ids.intersection(
                    global_trips[frozenset([r2])].candidate_vehicle_ids
                )
                valid_vehs = set()

                request_graph = TripGraph.from_requests(
                    {req_map[r] for r in candidate_trip}, self._network
                )
                for vid in common_vehs:
                    vehicle: VehicleAvailability = veh_map.get(vid)  # type: ignore // safe because of RV check

                    available_seats = vehicle.capacity - len(vehicle.passengers_onboard)
                    limit_k = max(available_seats, MAX_REQUESTS_ABOVE_CAPACITY)
                    if len(candidate_trip) > limit_k:
                        continue

                    result = request_graph.solve(vehicle, self._network)

                    if result.is_feasible:
                        T_by_size[2].add(candidate_trip)
                        rtv_graph.add_trip_for_vehicle(candidate_trip, vid, result)  # type: ignore
                        valid_vehs.add(vid)

                global_trips[candidate_trip] = GlobalTrip(
                    id=candidate_trip,
                    candidate_vehicle_ids=frozenset(valid_vehs),
                )

        # --- 3. EXPANSION LOOP (k=2 to Limit) ---
        for k in range(3, limit_k + 1):
            if not T_by_size[k - 1]:
                break

            # We iterate over existing valid trips and try to extend them
            # Optimization: Use a sorted list of trips to avoid duplicate pairs {A,B} vs {B,A}
            # (Similar to the "Old Code" lexicographic check)

            prev_trips_list = sorted([sorted(list(t)) for t in T_by_size[k - 1]])

            n = len(prev_trips_list)

            for i in range(n):
                trip_a = prev_trips_list[i]

                for j in range(i + 1, n):
                    trip_b = prev_trips_list[j]

                    # OPTIMIZATION: Ensure they share k-2 requests (Apriori)
                    if trip_a[:-1] != trip_b[:-1]:
                        break

                    # Create candidate trip by merging
                    new_req_id = trip_b[-1]
                    candidate_trip = frozenset(trip_a + [new_req_id])

                    # Determine common vehicles that served both trips
                    candidate_vehs = set(
                        global_trips[
                            frozenset(trip_a)
                        ].candidate_vehicle_ids.intersection(
                            global_trips[frozenset(trip_b)].candidate_vehicle_ids
                        )
                    )
                    if not candidate_vehs:
                        continue

                    is_structure_valid = True
                    prefix = trip_a[:-1]
                    for idx in range(len(prefix)):
                        # Remove element at idx from prefix
                        sub_prefix = prefix[:idx] + prefix[idx + 1 :]

                        # The sub-trip is: Modified Prefix + {x, y}
                        # We must sort it or use set to match the key format
                        sub_trip_id = frozenset(sub_prefix + [trip_a[-1], trip_b[-1]])
                        # A. Does this sub-trip exist globally?
                        if sub_trip_id not in global_trips:
                            is_structure_valid = False
                            break

                        # B. PER VEHICLE CHECK
                        candidate_vehs &= global_trips[
                            sub_trip_id
                        ].candidate_vehicle_ids

                        if not candidate_vehs:
                            is_structure_valid = False
                            break
                    if not is_structure_valid:
                        continue

                    valid_vehs = set()
                    request_graph = TripGraph.from_requests(
                        {req_map[r] for r in candidate_trip}, self._network
                    )
                    for vid in candidate_vehs:
                        vehicle: VehicleAvailability = veh_map.get(vid)  # type: ignore // safe because of RV check

                        available_seats = vehicle.capacity - len(
                            vehicle.passengers_onboard
                        )
                        limit_k = max(available_seats, MAX_REQUESTS_ABOVE_CAPACITY)
                        if len(candidate_trip) > limit_k:
                            continue

                        result = request_graph.solve(vehicle, self._network)

                        if result.is_feasible:
                            T_by_size[k].add(candidate_trip)
                            rtv_graph.add_trip_for_vehicle(
                                candidate_trip, vid, result  # type: ignore
                            )  # type: ignore
                            valid_vehs.add(vid)
                    global_trips[candidate_trip] = GlobalTrip(
                        id=candidate_trip,
                        candidate_vehicle_ids=frozenset(valid_vehs),
                    )

        return rtv_graph
