from collections.abc import Iterable
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from config import get_logger
from config.config import OMEGA, DELTA
from src.interfaces import (
    VehicleAvailability,
    Request,
    FeasibleRoute,
    Route,
    Stop as DMStop,
)
from src.interfaces.data_models import FeasibilityResult, Infeasible

logger = get_logger(__name__)
INF = float("inf")

# --- Types ---
StopId = Tuple[bool, str]  # (is_pickup, request_id)

MEMO_TYPE = Dict[Tuple[int, int], "MemoEntry"]


@dataclass
class MemoEntry:
    __slots__ = (
        "cost",
        "path_canon_bits",
        "latest_start_time",
        "required_capacity_surge",
    )
    cost: float
    path_canon_bits: Tuple[int, ...]  # Store canonical bit integers, not StopId objects
    latest_start_time: float
    required_capacity_surge: float


class TripSolver:

    # Heuristic: Only cache if we have at least 4 stops remaining
    # This prevents caching trivial leaf calculations
    MIN_CACHE_DEPTH = 4

    @staticmethod
    def solve(
        requests: Iterable[Request],
        vehicle: VehicleAvailability,
        time_matrix: np.ndarray,
        memo: Optional[Dict[Tuple[int, int], MemoEntry]] = None,
    ) -> FeasibilityResult:

        if memo is None:
            memo = {}

        # --- 1. SORT REQUESTS FOR STABILITY ---
        # Essential for Integer Caching: Ensure R1 always maps to Bit 0, R2 to Bit 1, etc.
        # This makes the cache vehicle-independent.
        sorted_requests = sorted(list(requests), key=lambda r: r.id)

        # --- 2. SETUP LOCAL CONTEXT ---
        v_node = vehicle.available_at_node
        v_start = vehicle.available_from_time
        v_cap = vehicle.capacity
        initial_load = len(vehicle.passengers_onboard)

        node_indices = []
        is_pickup = []
        hard_latest = []
        req_times = []
        req_ids_ordered = []

        # New Array: Maps local loop index -> Canonical Bit (1 << K)
        # If 0, it means "Not Cacheable" (e.g. onboard passenger)
        local_to_canon_bit = []

        # A. Onboard (Vehicle Specific - Not Canonical)
        for p in vehicle.passengers_onboard:
            r = p.original_request
            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(r.request_time)
            req_ids_ordered.append(r.id)
            local_to_canon_bit.append(0)  # 0 = Ignore in cache mask

        n_onboard = len(node_indices)
        onboard_mask = (1 << n_onboard) - 1

        # B. New Requests (Canonical)
        # Assign bit positions: 0, 1, 2... for the sorted requests
        current_bit_pos = 0
        total_canon_mask = 0

        # We also need Prereqs relative to local indices
        prereqs = [0] * n_onboard

        # Helper to find pickup index
        pk_indices_map = {}

        for i, r in enumerate(sorted_requests):
            # 1. Pickup
            node_indices.append(r.origin_node)
            is_pickup.append(True)
            hard_latest.append(r.latest_pickup_time)
            req_times.append(r.request_time)
            req_ids_ordered.append(r.id)

            # Assign Canonical Bit
            pk_bit = 1 << current_bit_pos
            local_to_canon_bit.append(pk_bit)
            total_canon_mask |= pk_bit
            current_bit_pos += 1

            pk_local_idx = len(node_indices) - 1
            pk_indices_map[r.id] = pk_local_idx
            prereqs.append(0)

            # 2. Dropoff
            node_indices.append(r.destination_node)
            is_pickup.append(False)
            hard_latest.append(r.latest_dropoff_time)
            req_times.append(r.request_time)
            req_ids_ordered.append(r.id)

            # Assign Canonical Bit
            dp_bit = 1 << current_bit_pos
            local_to_canon_bit.append(dp_bit)
            total_canon_mask |= dp_bit
            current_bit_pos += 1

            # Prereq is the pickup
            prereqs.append(1 << pk_local_idx)

        total_stops = len(node_indices)
        full_mask = (1 << total_stops) - 1

        # Optimization: Pre-compute reverse map for cache reconstruction
        # Map: Canonical Bit -> Local Index
        # Since bits are powers of 2, we can use a dict or array
        canon_bit_to_local = {b: i for i, b in enumerate(local_to_canon_bit) if b > 0}

        # Arrays for loop speed
        nodes_arr = np.array(node_indices, dtype=np.int32)
        load_deltas = [1 if p else -1 for p in is_pickup]
        cost_mask = [not p for p in is_pickup]

        # --- FUNCTIONAL DFS ---

        def _dfs(
            u_local_idx: int,
            current_time: float,
            current_load: int,
            visited_mask: int,
            canon_visited_mask: int,  # <--- The Second Mask (Integer)
        ) -> Tuple[float, Tuple[int, ...], float, float]:

            # --- A. CHECK CACHE ---
            # Fast integer checks only
            remaining_depth = total_stops - visited_mask.bit_count()

            cache_key = None

            if remaining_depth >= TripSolver.MIN_CACHE_DEPTH:
                # Must be past onboard stops
                if (
                    u_local_idx >= n_onboard
                    and (visited_mask & onboard_mask) == onboard_mask
                ):

                    # Integer Key Construction: O(1)
                    # Key = (Who am I, What is LEFT to do)
                    u_canon = local_to_canon_bit[u_local_idx]
                    remaining_canon = (
                        total_canon_mask ^ canon_visited_mask
                    )  # XOR to find remaining

                    cache_key = (u_canon, remaining_canon)

                    if cache_key in memo:
                        entry = memo[cache_key]
                        if current_time <= entry.latest_start_time:
                            if current_load + entry.required_capacity_surge <= v_cap:
                                # Reconstruction: Map Canonical Bits -> Local Indices
                                # This is small loop O(k)
                                path = []
                                for bit in entry.path_canon_bits:
                                    path.append(canon_bit_to_local[bit])

                                return (
                                    entry.cost,
                                    tuple(path),
                                    entry.latest_start_time,
                                    entry.required_capacity_surge,
                                )

            # --- B. BASE CASE ---
            if visited_mask == full_mask:
                return 0.0, (), INF, 0.0

            # --- C. RECURSIVE STEP ---
            best_local_cost = INF
            best_result = (INF, (), -INF, INF)

            u_global_idx = nodes_arr[u_local_idx] if u_local_idx >= 0 else v_node

            # Loop
            for v in range(total_stops):
                if (visited_mask >> v) & 1:
                    continue
                if (prereqs[v] & visited_mask) != prereqs[v]:
                    continue

                delta = load_deltas[v]
                if current_load + delta > v_cap:
                    continue

                dist = time_matrix[u_global_idx, nodes_arr[v]]
                arrival = current_time + dist

                if arrival > hard_latest[v]:
                    continue

                step_cost = 0.0
                if cost_mask[v]:
                    step_cost = arrival - req_times[v]

                # UPDATE DUAL MASKS
                # 1. visited_mask | (1 << v) -> Standard
                # 2. canon_visited_mask | local_to_canon_bit[v] -> Integer Update

                res_cost, res_path, res_latest, res_surge = _dfs(
                    v,
                    arrival,
                    current_load + delta,
                    visited_mask | (1 << v),
                    canon_visited_mask | local_to_canon_bit[v],
                )

                if res_cost == INF:
                    continue

                total_cost = step_cost + res_cost
                if total_cost < best_local_cost:
                    best_local_cost = total_cost

                    limit_u = hard_latest[u_local_idx] if u_local_idx >= 0 else INF
                    my_latest = min(limit_u, res_latest - dist)
                    my_surge = max(0.0, float(delta), delta + res_surge)

                    best_result = (total_cost, (v,) + res_path, my_latest, my_surge)

            # --- D. WRITE CACHE ---
            if cache_key is not None and best_local_cost != INF:
                # Convert Local Indices -> Canonical Bits for storage
                path_bits = tuple(local_to_canon_bit[i] for i in best_result[1])

                memo[cache_key] = MemoEntry(
                    cost=best_local_cost,
                    path_canon_bits=path_bits,
                    latest_start_time=best_result[2],
                    required_capacity_surge=best_result[3],
                )

            return best_result

        # --- EXECUTE ---
        # canon_visited_mask starts at 0
        cost, path, _, _ = _dfs(-1, v_start, initial_load, 0, 0)

        if cost == INF:
            return Infeasible(reason="No feasible route found")

        final_stops = []
        for idx in path:
            final_stops.append(
                DMStop(
                    node=node_indices[idx],
                    is_pickup=is_pickup[idx],
                    request_id=req_ids_ordered[idx],
                )
            )

        return FeasibleRoute(route=Route(tuple(final_stops), cost, 0.0, 0.0), cost=cost)
