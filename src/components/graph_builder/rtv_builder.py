from config.config import MAX_REQUESTS_ABOVE_CAPACITY
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
        rtv_graph: RTVGraph,
    ) -> None:
        """
        Implements Algorithm 1 using Clique Expansion + Lexicographic Ordering.
        We only iterate over the intersection of neighbors in the RV-graph.
        """
        from src.components.search_algorithm.dfs_bnb import TripSolver as Original_BnB

        from src.components.search_algorithm.dfs_bnb_memo import (
            TripSolver as AStarDFSSolver,
        )
        from src.components.search_algorithm.bfs_astar import TripSolver as BFSSolver
        from src.components.search_algorithm.bfs_astar_monotonic import (
            TripSolver as BfsMono,
        )
        from src.components.search_algorithm.bfs_astar_gradient import (
            TripSolver as BfsGradient,
        )

        TripSolverClass = AStarDFSSolver
        TripSolverClass = Original_BnB
        TripSolverClass = BfsMono
        TripSolverClass = BFSSolver
        TripSolverClass = BfsGradient

        m = len(v_avail.passengers_onboard)
        available_seats = v_avail.capacity - m
        limit_k = max(available_seats, MAX_REQUESTS_ABOVE_CAPACITY)

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

        # --- [Step 2] We keep it seperate from the general Apriori Loop because we have the RV-graph ---
        # Sort valid_singles for deterministic ordering (optional but good for consistency)
        sorted_singles = sorted(list(valid_singles))

        for r1 in sorted_singles:
            # Get all neighbors of r1 from the RV-graph
            neighbors = rv_graph.request_request.get(r1, set())
            # we need to ensure that it is also for the correct vehicle

            # Intersect neighbors with valid_singles to ensure the other request is also serviceable
            # We enforce r1 < r2 to avoid duplicates ({A,B} vs {B,A}) and self-loops
            potential_r2 = [r2 for r2 in neighbors if r2 in valid_singles and r1 < r2]

            for r2 in potential_r2:
                # We know r1 and r2 are neighbors (graph) and valid singles (vehicle).
                # We just need to check if they can fit together in this specific vehicle.
                candidate_trip = frozenset({r1, r2})

                reqs = {requests_by_id[r] for r in candidate_trip}
                # result = self._route_planner.check_feasibility(reqs, vehicle=v_avail)
                # result = TripSolver.solve(
                #     reqs, v_avail, self._route_planner._network
                # )
                result = TripSolverClass.solve(
                    reqs, v_avail, self._route_planner._network.get_l1_cache()
                )

                if result.is_feasible:
                    T_by_size[2].add(candidate_trip)
                    rtv_graph.add_trip_for_vehicle(
                        candidate_trip, v_id, cast(FeasibleRoute, result)
                    )

        # --- [Step 3] General Apriori Loop for k >= 3 ---
        for k in range(3, limit_k + 1):
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

                    # Create the candidate trip of size k
                    # (trip_b[-1] is guaranteed to be > trip_a[-1] because of sorting)
                    candidate_trip = frozenset(trip_a + [trip_b[-1]])

                    # 3. Sub-trip Lemma (Lemma 2 from Paper)
                    # We constructed this candidate from two existing sub-trips (trip_a and trip_b).
                    # Now we must ensure ALL other (k-2) sub-trips of size k-1 exist in T_{k-1}.

                    is_valid_structure = True
                    prefix = trip_a[:-1]

                    # We need to verify that merging r_x and r_y into the prefix
                    # doesn't violate any other subset constraints.
                    for idx in range(len(prefix)):
                        # Remove one element from the prefix
                        sub_prefix = prefix[:idx] + prefix[idx + 1 :]
                        # Check if the remaining prefix + {rx, ry} was a valid trip in the previous round
                        check_subtrip = frozenset(sub_prefix + [r_x, r_y])

                        if check_subtrip not in T_by_size[k - 1]:
                            is_valid_structure = False
                            break

                    if not is_valid_structure:
                        continue

                    # 4. Feasibility Check (Constraint Satisfaction)
                    # This is the heavy calculation - only performed if structural integrity passes
                    reqs = {requests_by_id[r] for r in candidate_trip}

                    result = TripSolverClass.solve(
                        reqs, v_avail, self._route_planner._network.get_l1_cache()
                    )

                    if result.is_feasible:
                        T_by_size[k].add(candidate_trip)
                        rtv_graph.add_trip_for_vehicle(
                            candidate_trip, v_id, cast(FeasibleRoute, result)
                        )
