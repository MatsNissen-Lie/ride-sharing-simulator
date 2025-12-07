from config.config import MAX_REQUESTS, OMEGA, DELTA
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
            if not T_by_size[k - 1]:
                break

            # Convert previous trips to sorted lists for consistent indexing
            # (Assuming T_by_size[k-1] contains frozensets or tuples)
            prev_trips_list = sorted([sorted(list(t)) for t in T_by_size[k - 1]])

            n = len(prev_trips_list)

            for i in range(n):
                trip_a = prev_trips_list[i]

                for j in range(i + 1, n):
                    trip_b = prev_trips_list[j]

                    # OPTIMIZATION:
                    # Two trips of size k-1 are candidates to merge ONLY if
                    # their first k-2 elements are identical.
                    # Since the list is sorted, if the prefix doesn't match,
                    # we can break the inner loop immediately (for this block).

                    # Check prefix (all elements except the last one)
                    if trip_a[:-1] != trip_b[:-1]:
                        break

                    # 2. Identify the new elements being combined
                    # For k=2: trip_a=[r_x], trip_b=[r_y]. We are combining r_x and r_y.
                    r_x = trip_a[-1]
                    r_y = trip_b[-1]

                    # --- FIX START ---
                    # 3. RV-Graph Connectivity Check (The Missing Link)
                    # For k=2, we MUST verify that r_x and r_y are actually neighbors in the RV graph.
                    # For k>2, this is implicitly guaranteed by the sub-trip existence check below.
                    if k == 2:
                        # Check if r_y is in r_x's neighbor list (or vice versa)
                        # Note: rv_graph.request_request should be a dict of sets
                        neighbors_of_x = rv_graph.request_request.get(r_x, set())
                        if r_y not in neighbors_of_x:
                            continue
                    # --- FIX END ---

                    # If we are here, trip_a and trip_b differ only by their last element.
                    # trip_a = [r1, r2, ..., r_{k-2}, r_x]
                    # trip_b = [r1, r2, ..., r_{k-2}, r_y]
                    # Union  = [r1, r2, ..., r_{k-2}, r_x, r_y]

                    # Create the candidate trip of size k
                    # (trip_b[-1] is guaranteed to be > trip_a[-1] because of sorting)
                    candidate_trip = frozenset(trip_a + [trip_b[-1]])

                    # 3. Sub-trip Lemma (Lemma 2 from Paper)
                    # We constructed this candidate from two existing sub-trips (trip_a and trip_b).
                    # Now we must ensure ALL other (k-2) sub-trips of size k-1 exist in T_{k-1}.

                    is_valid_structure = True

                    # We only need to check combinations we haven't already implicitly verified.
                    # We know sub-trip excluding r_y exists (it's trip_a).
                    # We know sub-trip excluding r_x exists (it's trip_b).
                    # We must check the removal of any of the "prefix" elements.

                    if k > 2:
                        # The prefix is trip_a[:-1]
                        prefix = trip_a[:-1]
                        last_a = trip_a[-1]
                        last_b = trip_b[-1]

                        # Iterate through removing each element of the prefix
                        for idx in range(len(prefix)):
                            # Construct a sub-trip by removing prefix[idx]
                            # New sub-trip: prefix without idx + {last_a, last_b}
                            sub_prefix = prefix[:idx] + prefix[idx + 1 :]
                            check_subtrip = frozenset(sub_prefix + [last_a, last_b])

                            if check_subtrip not in T_by_size[k - 1]:
                                is_valid_structure = False
                                break

                    if not is_valid_structure:
                        continue

                    # 4. Feasibility Check (Constraint Satisfaction)
                    # This is the heavy calculation - only performed if structural integrity passes
                    reqs = {requests_by_id[r] for r in candidate_trip}

                    result = self._route_planner.check_feasibility(
                        reqs, vehicle=v_avail
                    )

                    if result.is_feasible:
                        # DEBUG: Check if it works as a clique (Pairwise RV-graph connectivity)
                        # If it is feasible, they SHOULD be neighbors in the RV graph.
                        # If not, we have a bug in RV graph construction or an inconsistency.
                        # all_neighbors = True
                        # missing_pair = ("", "")

                        # trip_list = list(candidate_trip)
                        # for idx_1 in range(len(trip_list)):
                        #     for idx_2 in range(idx_1 + 1, len(trip_list)):
                        #         r1 = trip_list[idx_1]
                        #         r2 = trip_list[idx_2]
                        #         if r2 not in rv_graph.request_request.get(r1, set()):
                        #             all_neighbors = False
                        #             missing_pair = (r1, r2)
                        #             break
                        #     if not all_neighbors:
                        #         break

                        # if not all_neighbors:
                        #     r1, r2 = missing_pair
                        #     logger.critical(f"Feasible trip found but missing RV-graph edge between {r1} and {r2}!")

                        #     # Log neighbors
                        #     logger.error(f"Neighbors of r1 ({r1}): {rv_graph.request_request.get(r1, set())}")
                        #     logger.error(f"Neighbors of r2 ({r2}): {rv_graph.request_request.get(r2, set())}")

                        #     req1 = requests_by_id[r1]
                        #     req2 = requests_by_id[r2]

                        #     # Direct travel times between critical points
                        #     try:
                        #         path_o1_o2 = self._route_planner._network.get_shortest_path(req1.origin_node, req2.origin_node)
                        #         tt_o1_o2 = path_o1_o2.travel_time if path_o1_o2 else -1.0

                        #         path_d1_o2 = self._route_planner._network.get_shortest_path(req1.destination_node, req2.origin_node)
                        #         tt_d1_o2 = path_d1_o2.travel_time if path_d1_o2 else -1.0

                        #         logger.error("Travel Times:")
                        #         logger.error(f"  {req1.origin_node} (O1) -> {req2.origin_node} (O2): {tt_o1_o2}")
                        #         logger.error(f"  {req1.destination_node} (D1) -> {req2.origin_node} (O2): {tt_d1_o2}")
                        #     except Exception as e:
                        #         logger.error(f"Could not fetch travel times: {e}")

                        #     # Log Temporal Details
                        #     logger.error("Temporal Details:")
                        #     logger.error(f"  Config OMEGA (Max Wait): {OMEGA}")
                        #     logger.error(f"  Config DELTA (Max Delay): {DELTA}")
                        #     logger.error(f"  r1 Time: {req1.request_time}")
                        #     logger.error(f"  r2 Time: {req2.request_time}")

                        #     logger.error("Vehicle Details:")
                        #     logger.error(f"  ID: {v_id}")
                        #     logger.error(f"  Available At Node: {v_avail.available_at_node}")
                        #     logger.error(f"  Available From Time: {v_avail.available_from_time}")

                        #     logger.error(f"Candidate trip: {candidate_trip}")
                        #     logger.error(f"Resulting route: {result}")
                        #     raise Exception("Found feasible trip that violates RV clique property. RV Graph incomplete?")

                        T_by_size[k].add(candidate_trip)
                        rtv_graph.add_trip_for_vehicle(
                            candidate_trip, v_id, cast(FeasibleRoute, result)
                        )
