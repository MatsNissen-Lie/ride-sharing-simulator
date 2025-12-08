"""Minimal single-vehicle route planner used during prototyping."""

from dataclasses import dataclass, field
from typing import List, Optional
from config import DELTA, OMEGA, get_logger

from src.interfaces import (
    FeasibilityResult,
    FeasibleRoute,
    Infeasible,
    Request,
    Stop,
    Route,
    INetworkService,
    VehicleAvailability,
    ISingleVehicleRoutePlanner,
)

logger = get_logger(__name__)


@dataclass(order=True)
class SearchNode:
    """Represents a state in the A* search tree."""

    f_score: float  # Priority (g + h)
    g_score: float = field(compare=False)  # Actual cost so far (cumulative delay)
    time: float = field(compare=False)  # Current time at this node
    load: int = field(compare=False)  # Current passenger count
    mask: int = field(compare=False)  # Bitmask of visited stops
    curr_node_idx: int = field(compare=False)  # Index of the stop we are currently at
    parent: Optional["SearchNode"] = field(compare=False, default=None)

    def get_path(self, stops: list[Stop]) -> list[Stop]:
        """Reconstructs the path by backtracking."""
        path = []
        curr = self
        while curr.parent is not None:
            # We skip the root/dummy start node in the path list
            # providing the stop indices align with the stops list
            if curr.curr_node_idx >= 0:
                path.append(stops[curr.curr_node_idx])
            curr = curr.parent
        return path[::-1]


class RoutePlanner(ISingleVehicleRoutePlanner):
    """A brute-force feasibility checker for two-request R-R cases."""

    def __init__(self, network_service: INetworkService) -> None:
        super().__init__(network_service)

    def check_feasibility(
        self,
        requests_to_add: set[Request],
        vehicle: VehicleAvailability | None = None,
        find_best_route: bool = True,
    ) -> FeasibilityResult:

        if not requests_to_add:
            raise ValueError("Requires at least one request.")

        # --- CASE 1: R-R Check (No Vehicle) ---
        if vehicle is None:
            if len(requests_to_add) != 2:
                # If you ever need 3+ request pooling without a vehicle,
                # you'd implement a generic RR-solver here.
                raise NotImplementedError(
                    "Only pairwise R-R supported without vehicle."
                )
            raise NotImplementedError("R-R checks are not implemented in this planner.")

        # --- CASE 2: General Vehicle + N Requests (RV & RTV) ---
        # Whether it is 1 request (RV) or 3 requests (RTV), the logic is identical.
        _ = find_best_route  # Currently unused
        return self._handle_rtv_check(vehicle, requests_to_add)

    def _handle_rtv_check(
        self,
        availability: VehicleAvailability,
        requests_to_add: set[Request],
        # find_best_route: bool,
    ) -> FeasibilityResult:
        """
        Unified DFS Branch-and-Bound solver for RV and RTV.
        """
        # --- 1. SETUP ---
        start_node = availability.available_at_node
        start_time = availability.available_from_time
        initial_load = len(availability.passengers_onboard)
        capacity = availability.capacity

        # Collect all stops (Existing Drop-offs + New Pickups/Drop-offs)
        stops: list[Stop] = []

        # Existing passengers (Drop-off only)
        for p in availability.passengers_onboard:
            stops.append(
                Stop(p.original_request.destination_node, False, p.original_request.id)
            )

        # New requests (Pickup + Drop-off)
        new_req_ids = set()
        for r in requests_to_add:
            new_req_ids.add(r.id)
            stops.append(Stop(r.origin_node, True, r.id))
            stops.append(Stop(r.destination_node, False, r.id))

        N = len(stops)

        # Mappings & Constraints
        all_reqs = {
            p.original_request.id: p.original_request
            for p in availability.passengers_onboard
        }
        all_reqs.update({r.id: r for r in requests_to_add})

        # Arrays for fast indexing (avoid dict lookups in loop)
        s_node = [s.node for s in stops]
        s_is_pick = [s.is_pickup for s in stops]
        s_req_id = [s.request_id for s in stops]

        # Time Windows
        t_earliest = []
        t_latest = []

        for i in range(N):
            r = all_reqs[s_req_id[i]]
            if s_is_pick[i]:
                t_earliest.append(r.request_time)
                # Strict wait time constraint
                t_latest.append(r.request_time + OMEGA)
            else:
                t_earliest.append(r.request_time)  # Can't drop off before requested
                # Max detour constraint
                limit = getattr(
                    r, "earliest_dropoff_time", None
                )  # pre-calculated direct arrival
                if limit is None:
                    raise ValueError(
                        f"Earliest dropoff time not set for request {r.id}"
                    )
                t_latest.append((limit + DELTA) if limit else float("inf"))

        # Precedence: Pickup index for every dropoff index (if applicable)
        # We map RequestID -> Pickup Index
        pick_map = {}
        for i in range(N):
            if s_is_pick[i]:
                pick_map[s_req_id[i]] = i

        # Cache Travel Times
        unique = set(s_node) | {start_node}
        tt = self._build_travel_time_cache(unique)

        # --- 2. DFS STATE ---
        best_cost = float("inf")
        best_path: list[int] = []
        full_mask = (1 << N) - 1

        def solve(u_node, u_time, load, mask, cost, path):
            nonlocal best_cost, best_path

            # Pruning
            if cost >= best_cost:
                return

            # Goal
            if mask == full_mask:
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)
                return

            # Branching
            for i in range(N):
                # If not visited
                if not ((mask >> i) & 1):

                    # A. Precedence (Dropoff requires visited Pickup)
                    # Existing passengers don't have pickups in 'stops', so check 'in new_req_ids'
                    if not s_is_pick[i] and s_req_id[i] in new_req_ids:
                        p_idx = pick_map[s_req_id[i]]
                        if not ((mask >> p_idx) & 1):
                            continue

                    # B. Capacity
                    # Pickup +1, Dropoff -1
                    new_load = load + (1 if s_is_pick[i] else -1)
                    if new_load > capacity:
                        continue

                    # C. Time
                    travel = tt.get((u_node, s_node[i]), float("inf"))
                    arr = u_time + travel
                    # Wait if early (not applicable for dropoffs usually, but logic holds)
                    act_arr = max(arr, t_earliest[i])

                    if act_arr > t_latest[i]:
                        continue

                    # D. Cost Function (Objective: Minimize Total Delay)
                    # Cost = Cumulative Travel Time + Wait Time?
                    # Standard: Sum of (Arrival_Dropoff - Request_Time)
                    # Your generic metric: Sum of Travel Times + Lateness Penalty

                    # Simple accumulation:
                    new_cost = cost + travel
                    # Add wait time at pickup (if driver arrives early)
                    if act_arr > arr:
                        raise ValueError(
                            "Unexpected wait time at pickup stop. Should not happen as vehicles are dispatched after request time."
                        )
                        new_cost += act_arr - arr

                    path.append(i)
                    solve(s_node[i], act_arr, new_load, mask | (1 << i), new_cost, path)
                    path.pop()  # Backtrack

        # --- 3. RUN ---
        solve(start_node, start_time, initial_load, 0, 0.0, [])

        if best_cost == float("inf"):
            return Infeasible(reason="DFS found no solution")

        return FeasibleRoute(
            route=Route(
                stops=tuple(stops[i] for i in best_path),
                total_travel_time=0.0,  # Not tracking strictly, but best_cost approx
                total_travel_distance=0.0,
                total_delay_cost=best_cost,
            ),
            cost=best_cost,
        )

    def _build_travel_time_cache(self, nodes: set[int]) -> dict[tuple[int, int], float]:
        """Pre-calculates all-pairs travel times for a small set of nodes."""
        # assume all nodes pares are primed in the cache
        return self._network.get_travel_times_l1_cache_subset(nodes)

    # @staticmethod
    # def _build_stops(request_a: Request, request_b: Request) -> tuple[Stop, ...]:
    #     return (
    #         Stop(node=request_a.origin_node, is_pickup=True, request_id=request_a.id),
    #         Stop(
    #             node=request_a.destination_node,
    #             is_pickup=False,
    #             request_id=request_a.id,
    #         ),
    #         Stop(node=request_b.origin_node, is_pickup=True, request_id=request_b.id),
    #         Stop(
    #             node=request_b.destination_node,
    #             is_pickup=False,
    #             request_id=request_b.id,
    #         ),
    #     )


if __name__ == "__main__":
    from src.demos.route_planner_demo import run_demo

    run_demo()
