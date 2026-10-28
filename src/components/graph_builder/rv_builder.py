"""
rv_graph_builder
=================

This module provides an implementation of the pairwise request-vehicle (RV)
graph described in the supplemental material of Alonso-Mora et al. (2017).
The RV-graph captures which pairs of requests may be served together by a
single vehicle and which individual requests may be served by a given
vehicle. Each edge in the graph is annotated with the cost returned from
the single vehicle route planner, allowing subsequent steps (such as trip
construction and assignment) to reason about the feasibility and cost of
possible matches.

The implementation here does not impose any particular limit on the number
of edges per node; however, users may choose to prune the graph to speed up
subsequent computations. See the paper for details on how such pruning can
trade off optimality for runtime.

The central class exported by this module is :class:`RVGraphBuilder`, which
implements the :class:`IRVGraphBuilder` interface from ``src.interfaces``.
Occupied vehicles are checked with the exact trip solver; empty and
one-passenger vehicles and request pairs use vectorised closed forms. The resulting graph is returned as an instance of
:class:`RVGraph` defined below.
"""

import itertools
import time
from functools import cache
from typing import cast

import numpy as np

from config import get_logger
from config.config import DELTA, OMEGA


from src.interfaces import (
    VehicleAvailability,
    IRVGraphBuilder,
    ITripSolver,
    Request,
    Stop,
    Route,
    FeasibleRoute,
)
from src.interfaces.graphs import RVGraph
from src.interfaces.services import INetworkService
from src.utils.timing import record, timed


logger = get_logger("rv_graph_builder")


@cache
def _stop_orders(n_onboard: int, capacity: int) -> np.ndarray:
    """Orders of n_onboard drop-offs (stops 0..n-1), a pickup (n) and its drop-off
    (n + 1): pickup before its drop-off, and a free seat when picking up."""
    pickup, dropoff = n_onboard, n_onboard + 1
    orders = [
        order
        for order in itertools.permutations(range(n_onboard + 2))
        if order.index(pickup) < order.index(dropoff)
        and n_onboard - order.index(pickup) + 1 <= capacity  # drops before pickup free seats
    ]
    return np.array(orders, dtype=np.int64).reshape(len(orders), n_onboard + 2)

# Stop order of each pair permutation (see _build_route_from_perm), as
# positions in [o1, d1, o2, d2].
_PERM_ORDER = np.array(
    [[0, 2, 1, 3], [0, 2, 3, 1], [2, 0, 1, 3], [2, 0, 3, 1], [0, 1, 2, 3], [2, 3, 0, 1]]
)

# silence the logger if this is not the main process
if __name__ != "__main__":
    logger.setLevel("WARNING")


class RVGraphBuilder(IRVGraphBuilder):
    def __init__(
        self,
        network: INetworkService,
        trip_solver: ITripSolver,
        omega: float = OMEGA,
        delta: float = DELTA,
    ) -> None:
        self._network = network
        self._trip_solver = trip_solver
        # Vectorised edges for empty and one-passenger vehicles are only valid
        # when they reproduce the trip solver; otherwise use it for every edge.
        self._closed_forms = getattr(trip_solver, "matches_rv_closed_forms", False)
        self._omega = omega
        self._delta = delta

    def build_graph(
        self, requests: set[Request], vehicles: set[VehicleAvailability]
    ) -> RVGraph:
        graph = RVGraph()

        # Convert to lists (stable indexing)
        req_list = list(requests)
        veh_list = list(vehicles)

        n_req = len(req_list)
        n_veh = len(veh_list)

        if n_req == 0:
            return graph
        t0 = time.perf_counter()

        # --- 1. RR PIPELINE (Prune + Solve Integrated) ---
        # A. Full Matrix Fetch (N x N)
        # We fetch this once. We will use it for BOTH pruning AND the solver.
        req_origins = [r.origin_node for r in req_list]
        rr_matrix = self._network.get_travel_time_matrix(req_origins, req_origins)

        # --- PREPARATION: DATA ARRAYS ---
        # Create numpy views for fast access
        req_origins = np.array(req_origins)
        req_dests = np.array([r.destination_node for r in req_list])
        req_times = np.array([r.request_time for r in req_list])

        # B. Vectorized Pruning
        # Necessary condition: whichever origin is visited first, the second
        # pickup is no earlier than t_first + tau(o_first, o_second), and must
        # happen by t_second + Omega. Upper triangle only (pairs are unordered).
        t_gap = req_times[None, :] - req_times[:, None]  # t_j - t_i
        rr_mask = np.triu(
            (rr_matrix <= self._omega + t_gap) | (rr_matrix.T <= self._omega - t_gap),
            k=1,
        )

        rr_idx_i, rr_idx_j = np.where(rr_mask)
        record("rr_prune", t0)

        # C. Vectorized Solving
        if len(rr_idx_i) > 0:
            logger.info(f"RR Pruning: evaluating {len(rr_idx_i)} pairs.")

            # OPTIMIZATION: Extract the O->O times we already have!
            # We need both directions: i->j and j->i
            # rr_matrix[i, j] gives time(i->j)
            # rr_matrix[j, i] gives time(j->i) -> This is equivalent to rr_matrix.T[i, j]

            t_o1_o2 = rr_matrix[rr_idx_i, rr_idx_j]
            t_o2_o1 = rr_matrix.T[rr_idx_i, rr_idx_j]

            with timed("rr_pairs"):
                self._solve_rr_matrix_fast(
                    rr_idx_i,
                    rr_idx_j,
                    req_list,
                    req_times,
                    req_origins,
                    req_dests,
                    t_o1_o2,
                    t_o2_o1,  # Pass pre-calculated times
                    graph,
                )

        # --- 2. RV PIPELINE (Split Execution) ---

        if n_veh > 0:
            t0 = time.perf_counter()
            veh_locs = [v.available_at_node for v in veh_list]

            # Broadcasting shapes
            v_times_arr = np.array([v.available_from_time for v in veh_list])[
                :, None
            ]  # (V, 1)

            # A. Matrix Fetch & Pruning
            rv_matrix = self._network.get_travel_time_matrix(
                veh_locs, req_origins.tolist()
            )  # (V, R)

            # Pickup deadline (request_time + Omega, or the promised pickup time)
            arrival = v_times_arr + rv_matrix
            r_latest_pickup = np.array([r.latest_pickup_time for r in req_list])
            rv_mask = arrival <= r_latest_pickup[None, :]

            # Indices of ALL feasible pairs
            v_idx, r_idx = np.where(rv_mask)

            if len(v_idx) > 0:
                # B. Split Execution based on Occupancy

                # 1. Compute Occupancy for ALL candidates once
                occupancy = np.array(
                    [len(v.passengers_onboard) for v in veh_list]
                )[v_idx]
                record("rv_prune", t0)

                # --- CASE 0: EMPTY (m=0) ---
                mask_0 = (occupancy == 0) & self._closed_forms
                if np.any(mask_0):
                    with timed("rv_empty"):
                        self._solve_rv_matrix_empty(
                            v_idx[mask_0],
                            r_idx[mask_0],
                            veh_list,
                            req_list,
                            # delay_vals
                        arrival[v_idx[mask_0], r_idx[mask_0]] - req_times[r_idx[mask_0]],
                            graph,
                        )

                # --- CASE 1: 1 PASSENGER (m=1) ---
                mask_1 = (occupancy == 1) & self._closed_forms
                if np.any(mask_1):
                    with timed("rv_one_passenger"):
                        self._solve_rv_matrix_m1(
                            v_idx[mask_1], r_idx[mask_1], veh_list, req_list, graph
                        )

                # --- CASE 2+: COMPLEX (m >= 2) ---
                mask_high = ~(mask_0 | mask_1)
                if np.any(mask_high):
                    with timed("rv_occupied"):
                        self._solve_rv_occupied(
                            v_idx[mask_high], r_idx[mask_high], veh_list, req_list, graph
                        )

        return graph

    def _solve_rr_matrix_fast(
        self,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
        req_list: list[Request],
        req_times: np.ndarray,
        req_origins: np.ndarray,
        req_dests: np.ndarray,
        t_o1_o2: np.ndarray,
        t_o2_o1: np.ndarray,  # Pre-calculated
        graph: RVGraph,
    ):
        """
        Solved RR matrix using pre-fetched O-O times.
        Only fetches O-D and D-D times from cache.
        """
        # 1. Gather remaining data
        # Origins are not needed, we have t_o1_o2
        d1, d2 = req_dests[idx_i], req_dests[idx_j]
        # Origins ARE needed for O->D lookup
        o1, o2 = req_origins[idx_i], req_origins[idx_j]
        t1, t2 = req_times[idx_i], req_times[idx_j]

        # 2. Fetch only missing travel times (O->D, D->D)
        mat = self._network.get_l1_cache()

        def get_v(u, v):
            return mat[u, v]

        t_o1_d1 = get_v(o1, d1)
        t_o1_d2 = get_v(o1, d2)
        t_o2_d1 = get_v(o2, d1)
        t_o2_d2 = get_v(o2, d2)
        t_d1_d2 = get_v(d1, d2)
        t_d2_d1 = get_v(d2, d1)
        t_d1_o2 = get_v(d1, o2)
        t_d2_o1 = get_v(d2, o1)

        # 3. Compute Costs (Same Matrix Logic as before)
        costs = np.full((6, len(idx_i)), np.inf)

        # Helper to apply mask and cost
        def check(perm, p1, p2, d_arr1, d_arr2):
            # Delay = (Arrival - Req) - Direct_Travel
            del1 = (d_arr1 - t1) - t_o1_d1
            del2 = (d_arr2 - t2) - t_o2_d2

            # Also check Pickup delays
            p_del1 = p1 - t1
            p_del2 = p2 - t2

            mask = (
                (del1 <= self._delta)
                & (del2 <= self._delta)
                & (p_del1 <= self._omega)
                & (p_del2 <= self._omega)
            )
            costs[perm, mask] = del1[mask] + del2[mask]

        # Perm 0: o1->o2->d1->d2
        p1_0, p2_0 = t1, np.maximum(t1 + t_o1_o2, t2)
        d1_0 = p2_0 + t_o2_d1
        d2_0 = d1_0 + t_d1_d2
        check(0, p1_0, p2_0, d1_0, d2_0)

        # Perm 1: o1->o2->d2->d1
        p1_1, p2_1 = t1, np.maximum(t1 + t_o1_o2, t2)
        d2_1 = p2_1 + t_o2_d2
        d1_1 = d2_1 + t_d2_d1
        check(1, p1_1, p2_1, d1_1, d2_1)

        # Perm 2: o2->o1->d1->d2
        p2_2, p1_2 = t2, np.maximum(t2 + t_o2_o1, t1)
        d1_2 = p1_2 + t_o1_d1
        d2_2 = d1_2 + t_d1_d2
        check(2, p1_2, p2_2, d1_2, d2_2)

        # Perm 3: o2->o1->d2->d1
        p2_3, p1_3 = t2, np.maximum(t2 + t_o2_o1, t1)
        d2_3 = p1_3 + t_o1_d2
        d1_3 = d2_3 + t_d2_d1
        check(3, p1_3, p2_3, d1_3, d2_3)

        # Perm 4: o1->d1->o2->d2
        p1_4 = t1
        d1_4 = p1_4 + t_o1_d1
        p2_4 = np.maximum(d1_4 + t_d1_o2, t2)
        d2_4 = p2_4 + t_o2_d2
        check(4, p1_4, p2_4, d1_4, d2_4)

        # Perm 5: o2->d2->o1->d1
        p2_5 = t2
        d2_5 = p2_5 + t_o2_d2
        p1_5 = np.maximum(d2_5 + t_d2_o1, t1)
        d1_5 = p1_5 + t_o1_d1
        check(5, p1_5, p2_5, d1_5, d2_5)

        # 4. Reduce & Add
        best_costs = np.min(costs, axis=0)
        best_perms = np.argmin(costs, axis=0)
        valid = np.isfinite(best_costs)

        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            logger.info(f"RR Matrix: Found {len(valid_idx)} feasible edges.")
            # Travel time and distance of the best order, leg by leg as in
            # get_shortest_path (legs without a path add nothing).
            ends = np.stack([o1, d1, o2, d2])[:, valid_idx].T
            seq = np.take_along_axis(ends, _PERM_ORDER[best_perms[valid_idx]], axis=1)
            leg_t = mat[seq[:, :-1], seq[:, 1:]]
            leg_d = self._network.get_distance_matrix()[seq[:, :-1], seq[:, 1:]]
            has_path = np.isfinite(leg_t) & (leg_t <= 1e9) & (leg_t >= 0)
            leg_t = np.where(has_path, leg_t, 0.0)
            leg_d = np.where(has_path, leg_d, 0.0)
            total_t = (leg_t[:, 0] + leg_t[:, 1]) + leg_t[:, 2]
            total_d = (leg_d[:, 0] + leg_d[:, 1]) + leg_d[:, 2]
            for n, k in enumerate(valid_idx):
                r1 = req_list[idx_i[k]]
                r2 = req_list[idx_j[k]]
                route = self._build_route_from_perm(
                    best_perms[k], r1, r2, best_costs[k], float(total_t[n]), float(total_d[n])
                )
                graph.add_rr_edge(r1.id, r2.id, route)

    def _solve_rv_matrix_empty(
        self,
        v_idxs: np.ndarray,
        r_idxs: np.ndarray,
        veh_list: list[VehicleAvailability],
        req_list: list[Request],
        delay_vals: np.ndarray,
        graph: RVGraph,
    ):
        """Edges for empty vehicles: drive to the pickup, then straight to the drop-off.

        Cost is the rider's delay, which for a direct ride equals the pickup wait.
        """
        if len(v_idxs) == 0:
            return

        logger.info(f"RV Matrix (Empty): Processing {len(v_idxs)} edges.")

        # We need the Pickup->Dropoff time to complete the route info
        # We can fetch this in bulk
        r_origins = np.array([req_list[i].origin_node for i in r_idxs])
        r_dests = np.array([req_list[i].destination_node for i in r_idxs])

        mat = self._network.get_l1_cache()
        t_pd = mat[r_origins, r_dests]

        # Calculate positive delay (if arrived early, delay is 0)
        costs = np.maximum(delay_vals, 0)

        # Insert into graph
        # This loop is fast because it's only doing dict insertions
        for k, (v_i, r_i) in enumerate(zip(v_idxs, r_idxs)):
            v = veh_list[v_i]
            r = req_list[r_i]

            route = FeasibleRoute(
                route=Route(
                    stops=(
                        Stop(r.origin_node, True, r.id),
                        Stop(r.destination_node, False, r.id),
                    ),
                    total_travel_time=float(t_pd[k]),
                    total_travel_distance=0.0,
                    total_delay_cost=float(costs[k]),
                ),
                cost=float(costs[k]),
            )
            graph.add_rv_edge(r.id, v.vehicle_id, route)

    def _solve_rv_occupied(
        self,
        v_idxs: np.ndarray,
        r_idxs: np.ndarray,
        veh_list: list[VehicleAvailability],
        req_list: list[Request],
        graph: RVGraph,
    ):
        """Exact trip solver for the edges without a closed form."""
        if self._closed_forms and len(v_idxs):
            keep = self._may_be_feasible(v_idxs, r_idxs, veh_list, req_list)
            v_idxs, r_idxs = v_idxs[keep], r_idxs[keep]
        for v, r in zip(v_idxs, r_idxs):
            vehicle, req = veh_list[v], req_list[r]
            result = self._trip_solver.solve([req], vehicle)
            if result.is_feasible:
                graph.add_rv_edge(req.id, vehicle.vehicle_id, cast(FeasibleRoute, result))

    def _may_be_feasible(
        self,
        v_idxs: np.ndarray,
        r_idxs: np.ndarray,
        veh_list: list[VehicleAvailability],
        req_list: list[Request],
    ) -> np.ndarray:
        """False where no stop order can serve the request.

        Extends partial stop orders (onboard drop-offs, the pickup, then its
        drop-off; a free seat when picking up) for all pairs at once, keeping
        only those that meet every deadline so far. Travel times are added in
        the order the trip solver adds them (hard deadlines, no waiting), so a
        route the solver finds survives and only pairs without any route are
        skipped; on real batches that is 95% of them.
        """
        mat = self._network.get_l1_cache()
        n_pairs = len(v_idxs)
        k_max = max(len(veh_list[i].passengers_onboard) for i in v_idxs)
        n_stops = k_max + 2
        # Stops 0..k_max-1: onboard drop-offs (unused ones count as visited);
        # k_max: the pickup; k_max + 1: its drop-off.
        rows: dict[int, tuple[list[int], list[float], int]] = {}
        nodes, deadlines, done = [], [], []
        for vi, ri in zip(v_idxs, r_idxs):
            if vi not in rows:
                onboard = [p.original_request for p in veh_list[vi].passengers_onboard]
                pad = k_max - len(onboard)
                rows[vi] = (
                    [q.destination_node for q in onboard] + [0] * pad,
                    [q.latest_dropoff_time for q in onboard] + [0.0] * pad,
                    ((1 << k_max) - 1) ^ ((1 << len(onboard)) - 1),
                )
            dests, latest, padded = rows[vi]
            r = req_list[ri]
            nodes.append(dests + [r.origin_node, r.destination_node])
            deadlines.append(latest + [r.latest_pickup_time, r.latest_dropoff_time])
            done.append(padded)
        nodes, deadlines, done = np.array(nodes), np.array(deadlines), np.array(done)
        free = np.array([veh_list[i].capacity - len(veh_list[i].passengers_onboard) for i in v_idxs])

        # Frontier of partial orders: pair, position, time, visited stops, free seats.
        pair = np.arange(n_pairs)
        here = np.array([veh_list[i].available_at_node for i in v_idxs])
        t = np.array([veh_list[i].available_from_time for i in v_idxs])
        seats = free
        full = (1 << n_stops) - 1
        feasible = np.zeros(n_pairs, dtype=bool)
        while len(pair):
            new = [], [], [], [], [], []
            for stop in range(n_stops):
                bit = 1 << stop
                allowed = (done & bit) == 0
                if stop == k_max:
                    allowed &= seats > 0
                elif stop == k_max + 1:
                    allowed &= (done & (1 << k_max)) != 0
                c = np.flatnonzero(allowed)
                p = pair[c]
                arrival = t[c] + mat[here[c], nodes[p, stop]]
                ok = arrival <= deadlines[p, stop]
                c, p = c[ok], p[ok]
                new[0].append(p)
                new[1].append(nodes[p, stop])
                new[2].append(arrival[ok])
                new[3].append(done[c] | bit)
                new[4].append(seats[c] + (-1 if stop == k_max else 1))
                new[5].append(np.full(len(c), stop))
            pair, here, t, done, seats, last = (np.concatenate(a) for a in new)
            finished = done == full
            feasible[pair[finished]] = True
            alive = ~finished & ~feasible[pair]  # one route per pair is enough
            # Of partial orders with the same pair, visited stops and last stop,
            # the earliest can finish whatever a later one can: keep only it.
            key = (pair * (full + 1) + done) * n_stops + last
            first = np.lexsort((t, key))
            first = first[alive[first]]
            if len(first):
                first = first[np.r_[True, key[first][1:] != key[first][:-1]]]
            pair, here, t, done, seats = pair[first], here[first], t[first], done[first], seats[first]
        return feasible

    def _solve_rv_matrix_m1(
        self,
        v_idxs: np.ndarray,
        r_idxs: np.ndarray,
        veh_list: list[VehicleAvailability],
        req_list: list[Request],
        graph: RVGraph,
    ):
        """Vectorized solver for Vehicles with 1 existing passenger (3 Permutations)."""
        logger.info(f"RV Matrix (m=1): Processing {len(v_idxs)} edges.")

        # 1. GATHER DATA
        v_data = []
        for i in v_idxs:
            v = veh_list[i]
            pax = list(v.passengers_onboard)[0]  # The only passenger (set -> list)
            d1_dead = pax.original_request.latest_dropoff_time
            v_data.append(
                (
                    v.available_at_node,
                    v.available_from_time,
                    pax.original_request.destination_node,
                    d1_dead,
                )
            )

        v_arr = np.array(v_data)
        v_loc = v_arr[:, 0].astype(int)
        v_time = v_arr[:, 1]
        d1_loc = v_arr[:, 2].astype(int)
        d1_dead = v_arr[:, 3]

        # Pull request data
        r_origins = np.array([req_list[i].origin_node for i in r_idxs])
        r_dests = np.array([req_list[i].destination_node for i in r_idxs])
        r_times = np.array([req_list[i].request_time for i in r_idxs])
        r_dead = np.array([req_list[i].latest_dropoff_time for i in r_idxs])
        r_lp = np.array([req_list[i].latest_pickup_time for i in r_idxs])

        # 2. FETCH TRAVEL TIMES
        mat = self._network.get_l1_cache()

        def get_v(u, v):
            return mat[u, v]

        # Edges
        t_v_p = get_v(v_loc, r_origins)
        t_v_d1 = get_v(v_loc, d1_loc)

        t_p_d = get_v(r_origins, r_dests)
        t_p_d1 = get_v(r_origins, d1_loc)

        t_d1_p = get_v(d1_loc, r_origins)
        t_d1_d = get_v(d1_loc, r_dests)
        t_d_d1 = get_v(r_dests, d1_loc)

        # 3. PERMUTATIONS (3 Valid)
        # Cost = new rider's delay (a_d - t*) + passenger's extra delay versus
        # driving straight to d1 (a_d1 - (v_time + t_v_d1)).
        r_star = r_times + t_p_d
        d1_direct = v_time + t_v_d1
        costs = np.full((3, len(v_idxs)), np.inf)

        # -- Perm 0: P -> D -> D1 --
        a_p = np.maximum(v_time + t_v_p, r_times)
        a_d = a_p + t_p_d
        a_d1 = a_d + t_d_d1
        mask0 = (a_p <= r_lp) & (a_d1 <= d1_dead) & (a_d <= r_dead)
        costs[0, mask0] = ((a_d - r_star) + (a_d1 - d1_direct))[mask0]

        # -- Perm 1: P -> D1 -> D --
        a_p = np.maximum(v_time + t_v_p, r_times)
        a_d1 = a_p + t_p_d1
        a_d = a_d1 + t_d1_d
        mask1 = (a_p <= r_lp) & (a_d1 <= d1_dead) & (a_d <= r_dead)
        costs[1, mask1] = ((a_d - r_star) + (a_d1 - d1_direct))[mask1]

        # -- Perm 2: D1 -> P -> D --
        a_d1 = v_time + t_v_d1
        a_p = np.maximum(a_d1 + t_d1_p, r_times)
        a_d = a_p + t_p_d
        mask2 = (a_d1 <= d1_dead) & (a_p <= r_lp) & (a_d <= r_dead)
        costs[2, mask2] = ((a_d - r_star) + (a_d1 - d1_direct))[mask2]

        # 4. REDUCE
        best_costs = np.min(costs, axis=0)
        best_perms = np.argmin(costs, axis=0)

        # 5. BUILD
        valid = np.isfinite(best_costs)
        valid_idx = np.where(valid)[0]

        for k in valid_idx:
            self._add_m1_edge(
                graph,
                int(v_idxs[k]),
                int(r_idxs[k]),
                veh_list,
                req_list,
                int(best_perms[k]),
                float(best_costs[k]),
            )

    def _add_m1_edge(
        self,
        graph: RVGraph,
        v_idx: int,
        r_idx: int,
        v_list: list[VehicleAvailability],
        r_list: list[Request],
        perm: int,
        cost: float,
    ):
        v = v_list[v_idx]
        r = r_list[r_idx]
        # Vehicle has 1 passenger (set->list->[0])
        d1_req = list(v.passengers_onboard)[0].original_request

        s_p = Stop(r.origin_node, True, r.id)
        s_d = Stop(r.destination_node, False, r.id)
        s_d1 = Stop(d1_req.destination_node, False, d1_req.id)

        if perm == 0:
            stops = (s_p, s_d, s_d1)
        elif perm == 1:
            stops = (s_p, s_d1, s_d)
        else:
            stops = (s_d1, s_p, s_d)

        route = FeasibleRoute(
            route=Route(
                stops=stops,
                total_travel_time=0.0,
                total_travel_distance=0.0,
                total_delay_cost=float(cost),
            ),
            cost=float(cost),
        )
        graph.add_rv_edge(r.id, v.vehicle_id, route)

    def _build_route_from_perm(
        self,
        perm_id: int,
        r1: Request,
        r2: Request,
        cost: float,
        total_time: float,
        total_dist: float,
    ) -> FeasibleRoute:
        """Helper to reconstruct the Stop sequence for the chosen best permutation."""
        # Sequence definitions matching the matrix logic above
        # 0: o1 -> o2 -> d1 -> d2
        # 1: o1 -> o2 -> d2 -> d1
        # 2: o2 -> o1 -> d1 -> d2
        # 3: o2 -> o1 -> d2 -> d1
        # 4: o1 -> d1 -> o2 -> d2
        # 5: o2 -> d2 -> o1 -> d1

        # Build Stops
        s_o1 = Stop(r1.origin_node, True, r1.id)
        s_d1 = Stop(r1.destination_node, False, r1.id)
        s_o2 = Stop(r2.origin_node, True, r2.id)
        s_d2 = Stop(r2.destination_node, False, r2.id)

        if perm_id == 0:
            seq = [s_o1, s_o2, s_d1, s_d2]
        elif perm_id == 1:
            seq = [s_o1, s_o2, s_d2, s_d1]
        elif perm_id == 2:
            seq = [s_o2, s_o1, s_d1, s_d2]
        elif perm_id == 3:
            seq = [s_o2, s_o1, s_d2, s_d1]
        elif perm_id == 4:
            seq = [s_o1, s_d1, s_o2, s_d2]
        elif perm_id == 5:
            seq = [s_o2, s_d2, s_o1, s_d1]
        else:
            raise ValueError(f"Invalid perm_id {perm_id}")

        return FeasibleRoute(
            route=Route(
                stops=tuple(seq),
                total_travel_time=total_time,
                total_travel_distance=total_dist,
                total_delay_cost=cost,
            ),
            cost=cost,
        )
