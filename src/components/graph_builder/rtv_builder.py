from config.config import MAX_REQUESTS
from old_code.rv_graph_builder import RVGraph
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

        if not self._route_planner._network.is_nuclear:
            # --- STEP 0: PREFETCH / CACHE WARMING ---
            # Identify all unique nodes involved in this batch
            unique_nodes = set()

            # 1. Request Origins and Destinations
            for req in requests:
                unique_nodes.add(req.origin_node)
                unique_nodes.add(req.destination_node)

            # 2. Vehicle Locations
            for veh in vehicles:
                unique_nodes.add(veh.available_at_node)
            node_list = list(unique_nodes)
            # 3. Trigger the Big Fetch
            # This will populate the RoutePlanner's internal cache
            if hasattr(self._route_planner._network, "prefetch_all_pairs"):
                self._route_planner._network.prefetch_all_pairs(node_list)
            logger.info(
                f"Prefetched routes for {len(node_list)} unique nodes ({len(node_list)**2} pairs)"
            )

        try:
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
        except Exception as e:
            logger.error(f"Error building RTV graph: {e}")
            raise e

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
        Implements Algorithm 1 using Clique Expansion + Lexicographic Ordering.
        We only iterate over the intersection of neighbors in the RV-graph.
        """
        m = len(v_avail.passengers_onboard)
        available_seats = v_avail.capacity - m
        limit_k = min(max_trip_size, available_seats, MAX_REQUESTS)

        if limit_k <= 0:
            return

        # T_by_size: k -> Set[TripId]
        T_by_size: Dict[int, Set[TripId]] = defaultdict(set)

        # Cache valid single requests for this vehicle for fast intersection
        # This represents "Nodes in the RV-graph that are also valid for this vehicle"
        valid_singles = set()

        # --- [Step 1] Trips of size 1 ---
        relevant_requests = rv_graph.get_requests_for_vehicle(v_id)
        for r_id in relevant_requests:
            trip: TripId = frozenset({r_id})
            rv_route = rv_graph.rv_details.get((r_id, v_id))
            if rv_route:
                T_by_size[1].add(trip)
                rtv_graph.add_trip_for_vehicle(trip, v_id, rv_route)
                valid_singles.add(r_id)

        if limit_k == 1 or not T_by_size[1]:
            return

        # --- [Step 2 & 3] General Loop for k >= 2 ---
        # We can unify Step 2 and 3 because the intersection logic works for size 1 too.

        for k in range(2, limit_k + 1):
            prev_trips_set = T_by_size[k - 1]
            if not prev_trips_set:
                break

            for base_trip in prev_trips_set:
                max_req_in_base = max(base_trip)

                # 1. CLIQUE OPTIMIZATION (The "Not Dumb" Part)
                # A new request must be a neighbor of EVERY request in the current base_trip.
                # We start the intersection with 'valid_singles' to ensure we only pick
                # requests this vehicle can actually serve.

                common_neighbors = valid_singles.copy()

                # Intersect with neighbors of each request in the trip
                for r_existing in base_trip:
                    r_neighbors = rv_graph.request_request.get(r_existing, set())
                    common_neighbors.intersection_update(r_neighbors)

                    # Optimization: Fail fast if intersection becomes empty
                    if not common_neighbors:
                        break

                if not common_neighbors:
                    continue

                # 2. Iterate over the valid clique candidates
                for r_new in common_neighbors:

                    # Enforce Ordering (prevents duplicates like {A,B}+{C} vs {A,C}+{B})
                    if r_new <= max_req_in_base:
                        continue

                    # 3. Sub-trip Lemma (Lemma 2)
                    # We know 'base_trip' exists. We verify other sub-trips.
                    is_valid_structure = True

                    # Only check if k > 2. For k=2, base_trip is size 1.
                    # If we form {A, B}, the sub-trips are {A} and {B}.
                    # We know {A} exists (it's base_trip).
                    # We know {B} exists (it's in valid_singles).
                    # So structural check is implicitly passed for k=2!

                    if k > 2:
                        for r_old in base_trip:
                            # Construct the check_trip: (base_trip \ {r_old}) U {r_new}
                            # We check if this specific combination was found in the previous step
                            check_trip = frozenset((base_trip - {r_old}) | {r_new})

                            if check_trip not in prev_trips_set:
                                is_valid_structure = False
                                break

                    if not is_valid_structure:
                        continue

                    # 4. Feasibility Check (Constraint Satisfaction)
                    new_trip = base_trip | {r_new}
                    reqs = {requests_by_id[r] for r in new_trip}

                    # This is the heavy calculation
                    result = self._route_planner.check_feasibility(
                        reqs, vehicle=v_avail
                    )

                    if result.is_feasible:
                        T_by_size[k].add(new_trip)
                        rtv_graph.add_trip_for_vehicle(
                            new_trip, v_id, cast(FeasibleRoute, result)
                        )
