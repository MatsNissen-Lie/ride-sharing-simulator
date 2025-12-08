import numpy as np
from dataclasses import dataclass, field

from config import get_logger
from config.config import (
    OMEGA,
    DELTA,
)
from src.interfaces import (
    VehicleAvailability,
    Request,
    FeasibleRoute,
    Route,
)
from src.interfaces.data_models import FeasibilityResult, Infeasible, Stop as DMStop
from src.interfaces.services import INetworkService

logger = get_logger(__name__)
INF = float("inf")


@dataclass
class TripGraph:
    """
    A lightweight, array-centric container for a specific combination of Requests.
    It pre-computes the static constraints between requests, but checks vehicles dynamically.
    """

    # --- Static Request Data (Pre-computed once per Trip) ---
    req_ids: list[str]  # IDs of the new requests
    nodes: list[int]  # Node IDs for [R1_Pick, R1_Drop, R2_Pick, ...]
    is_pickup: list[bool]  # Type mask
    req_times: list[float]  # Original request times (for delay calc)
    hard_latest: list[float]  # Constraint: Latest arrival

    # Adjacency matrix for the Request stops ONLY.
    # Shape: [N_req_stops, N_req_stops]
    static_adj: np.ndarray

    # Precedence bitmask (Dropoff `i` requires Pickup `j` visited)
    # 0 for Pickups, (1 << pickup_idx) for Dropoffs
    prereqs: list[int]

    # Mapping to help reconstruction
    request_obj_map: dict = field(repr=False)

    @classmethod
    def from_requests(
        cls, requests: set[Request], network: INetworkService
    ) -> "TripGraph":
        """
        Builds the static graph for a set of requests (The Trip).
        """
        nodes = []
        is_pickup = []
        req_times = []
        hard_earliest = []
        hard_latest = []
        prereqs = []
        req_ids_list = []

        # Build raw arrays
        # Pattern: [R1_P, R1_D, R2_P, R2_D, ...]
        current_idx = 0
        pickup_indices = {}  # Map ReqID -> Index of pickup stop

        for r in requests:
            req_ids_list.append(r.id)

            # 1. Pickup
            nodes.append(r.origin_node)
            is_pickup.append(True)
            req_times.append(r.request_time)
            hard_earliest.append(r.request_time)
            hard_latest.append(r.request_time + OMEGA)
            prereqs.append(0)  # Pickups have no internal prereqs in this local view
            pickup_indices[r.id] = current_idx
            current_idx += 1

            # 2. Dropoff
            nodes.append(r.destination_node)
            is_pickup.append(False)
            req_times.append(r.request_time)
            hard_earliest.append(r.request_time)  # Can't drop before pick

            limit = r.earliest_dropoff_time
            hard_latest.append((limit + DELTA) if limit else INF)

            # Dropoff depends on the pickup we just added
            prereqs.append(1 << pickup_indices[r.id])
            current_idx += 1

        # Pre-compute internal travel times for these requests
        # This is the "Trip Centric" optimization: done once, reused for many vehicles
        static_adj = network.get_travel_time_matrix(nodes, nodes)

        return cls(
            req_ids=req_ids_list,
            nodes=nodes,
            is_pickup=is_pickup,
            req_times=req_times,
            hard_latest=hard_latest,
            static_adj=static_adj,
            prereqs=prereqs,
            request_obj_map={r.id: r for r in requests},
        )

    def solve(
        self, vehicle: VehicleAvailability, network: INetworkService
    ) -> FeasibilityResult:
        """
        Solves the TSP/VRP for this specific Trip + This specific Vehicle.
        Uses cached travel times and bitwise masks for maximum speed.
        """

        # --- 1. SETUP DYNAMIC CONTEXT ---
        # We need to merge Vehicle State (Onboard passengers) + Static Trip Stops

        v_loc = vehicle.available_at_node
        v_start_time = vehicle.available_from_time
        v_cap = vehicle.capacity
        initial_load = len(vehicle.passengers_onboard)

        # Collect Onboard Passenger Stops (Drop-offs only)
        onboard_stops_nodes = []
        onboard_req_times = []
        onboard_latest = []
        onboard_ids = []

        for p in vehicle.passengers_onboard:
            req = p.original_request
            onboard_stops_nodes.append(req.destination_node)
            onboard_req_times.append(req.request_time)
            onboard_ids.append(req.id)
            limit = req.earliest_dropoff_time
            onboard_latest.append((limit + DELTA) if limit else INF)

        num_onboard = len(onboard_stops_nodes)
        num_static = len(self.nodes)
        total_stops = num_onboard + num_static

        # Full Arrays for DFS
        # Order: [Onboard_Drops..., Static_Stops...]

        # Nodes
        all_nodes = onboard_stops_nodes + self.nodes

        # Constraints
        # Onboard passengers have no 'earliest' constraint other than 'now'
        all_latest = onboard_latest + self.hard_latest

        # Load Delta (-1 for onboard drops, variable for static)
        # Note: Static logic -> Pickup (+1), Dropoff (-1)
        static_deltas = [1 if is_p else -1 for is_p in self.is_pickup]
        all_deltas = [-1] * num_onboard + static_deltas

        # Prereqs
        # Onboard drops have no prerequisites (they are already picked up)
        # Static stops: We need to shift the bitmask bits because indices shifted by num_onboard
        # BUT: The bitmask refers to the *relative* index in the full list.
        # The static `prereqs` refer to indices 0..N. We must shift them by `num_onboard`.
        full_prereqs = [0] * num_onboard
        for p_mask in self.prereqs:
            # Shift the mask: if bit 0 was set, now bit (0+num_onboard) is set
            if p_mask > 0:
                full_prereqs.append(p_mask << num_onboard)
            else:
                full_prereqs.append(0)

        # Delay Calculation
        # We need request times to calc delay cost (Arrival - ReqTime)
        # Only relevant for Dropoffs.
        all_req_times = onboard_req_times + self.req_times
        is_dropoff_mask = [True] * num_onboard + [not p for p in self.is_pickup]

        # --- 2. TRAVEL TIME MATRIX MATERIALIZATION ---
        # Pre-calculate full N x N matrix to avoid function calls in DFS loop.

        # Initialize full matrix
        full_adj = np.zeros((total_stops, total_stops))

        # A. Fill Static-to-Static (Bottom-Right) - Block Copy using Trip Graph Cache
        if num_static > 0:
            full_adj[num_onboard:, num_onboard:] = self.static_adj

        # B. Fill Dynamic Rows (Vehicle + Onboard -> All)
        # We query: [V_Loc, Onboard_1, ... Onboard_N] -> [All Nodes]
        sources = [v_loc] + onboard_stops_nodes
        targets = all_nodes

        dyn_matrix = network.get_travel_time_matrix(sources, targets)

        # Row 0 is V_Start distances. Separate to avoid special -1 index logic in matrix
        dist_from_start = dyn_matrix[0]  # Shape [total_stops]

        # Rows 1..N are Onboard -> All. Copy to Top-Rows of full_adj
        if num_onboard > 0:
            full_adj[:num_onboard, :] = dyn_matrix[1:]

        # C. Fill Static -> Onboard (Cross terms)
        # These are rare (pickup new pax then drop old pax?), but valid topological paths.
        if num_onboard > 0 and num_static > 0:
            cross_matrix = network.get_travel_time_matrix(
                self.nodes, onboard_stops_nodes
            )
            full_adj[num_onboard:, :num_onboard] = cross_matrix

        # --- 3. DFS SOLVER ---

        best_cost = INF
        best_path = []
        full_mask = (1 << total_stops) - 1

        def _dfs(
            u_idx: int,
            current_time: float,
            current_load: int,
            visited_mask: int,
            current_cost: float,
            path: list[int],
        ):
            nonlocal best_cost, best_path

            # Pruning 1: Cost
            if current_cost >= best_cost:
                return

            # Base Case: All visited
            if visited_mask == full_mask:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = list(path)
                return

            # Branching
            for v in range(total_stops):
                # 1. Already Visited?
                if (visited_mask >> v) & 1:
                    continue

                # 2. Precedence Check (Fast Bitwise)
                # If v has prereqs, ensure they are in visited_mask
                if (full_prereqs[v] & visited_mask) != full_prereqs[v]:
                    continue

                # 3. Capacity Check
                new_load = current_load + all_deltas[v]
                if new_load > v_cap:
                    continue

                # 4. Time Check
                # FAST LOOKUP via Array
                if u_idx == -1:
                    dist = dist_from_start[v]
                else:
                    dist = full_adj[u_idx, v]

                arrival = current_time + dist

                # Wait if early (Standard VRP logic: max(arrival, earliest))
                act_arr = arrival

                if act_arr > all_latest[v]:
                    continue

                # 5. Cost Calculation (Passenger Delay)
                # We add cost ONLY if this is a dropoff
                added_cost = 0.0
                if is_dropoff_mask[v]:
                    added_cost = act_arr - all_req_times[v]

                # Recurse
                path.append(v)
                _dfs(
                    v,
                    act_arr,
                    new_load,
                    visited_mask | (1 << v),
                    current_cost + added_cost,
                    path,
                )
                path.pop()

        # --- 4. EXECUTE ---
        # Start from "Node -1" (Vehicle Start), mask 0
        _dfs(-1, v_start_time, initial_load, 0, 0.0, [])

        if best_cost == INF:
            return Infeasible(
                reason="No feasible route found for this vehicle and trip."
            )

        # --- 5. RECONSTRUCT ---
        final_stops_dm = []

        # Map back to interface objects
        # We need to handle: Onboard Passengers vs New Requests

        # Map for Onboard IDs
        onboard_p_objs = list(vehicle.passengers_onboard)

        for idx in best_path:
            node_id = all_nodes[idx]

            if idx < num_onboard:
                # It was an existing passenger dropoff
                p = onboard_p_objs[idx]
                final_stops_dm.append(
                    DMStop(
                        node=node_id, is_pickup=False, request_id=p.original_request.id
                    )
                )
            else:
                # It was a new request stop
                static_idx = idx - num_onboard
                r_id = self.req_ids[static_idx // 2]  # Map back to ID
                is_p = self.is_pickup[static_idx]
                final_stops_dm.append(
                    DMStop(node=node_id, is_pickup=is_p, request_id=r_id)
                )

        return FeasibleRoute(
            route=Route(tuple(final_stops_dm), best_cost, 0.0, 0.0), cost=best_cost
        )
