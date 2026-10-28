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

# Pickup budgets are at most Omega; the near-node index reaches this much further
# (seconds), so float rounding never drops a reachable pair.
NEAR_MARGIN = 1.0


def _expand(
    starts: np.ndarray, lengths: np.ndarray, labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """For each k, the positions starts[k] .. starts[k] + lengths[k] - 1, with
    the label of k (default k) repeated alongside."""
    total = int(lengths.sum())
    labels = np.arange(len(starts)) if labels is None else labels
    offsets = np.cumsum(lengths) - lengths
    positions = np.arange(total) - np.repeat(offsets - starts, lengths)
    return np.repeat(labels, lengths), positions


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
        # Nodes within Omega + NEAR_MARGIN of each node, built on first use.
        self._near: tuple[np.ndarray, np.ndarray] | None = None
        self._rr_routes: dict[tuple[str, str], FeasibleRoute] = {}  # last batch's
        # Shareable-pair test results across batches (see _shareable_pairs):
        # requests numbered on first sight, and the pairs (a < b) that passed.
        self._rr_num: dict[str, int] = {}
        self._rr_next = 0
        self._rr_a = np.zeros(0, dtype=np.int64)
        self._rr_b = np.zeros(0, dtype=np.int64)
        # (best cost, stop order) with request a first, and with b first
        self._rr_ab = np.zeros((0, 2))
        self._rr_ba = np.zeros((0, 2))

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
        # --- PREPARATION: DATA ARRAYS ---
        req_origins = np.array([r.origin_node for r in req_list])
        req_dests = np.array([r.destination_node for r in req_list])
        req_times = np.array([r.request_time for r in req_list])

        # B. Vectorized Pruning and solving: shareable pairs with their best cost
        # and stop order (only pairs with a request new to this batch are
        # computed; see _shareable_pairs)
        rr_idx_i, rr_idx_j, rr_costs, rr_perms = self._shareable_pairs(
            req_list, req_origins, req_dests, req_times
        )
        record("rr_prune", t0)

        # C. Routes of the shareable pairs
        if len(rr_idx_i) > 0:
            with timed("rr_pairs"):
                self._add_rr_edges(
                    rr_idx_i, rr_idx_j, rr_costs, rr_perms, req_list, req_origins, req_dests, graph
                )

        # --- 2. RV PIPELINE (Split Execution) ---

        if n_veh > 0:
            t0 = time.perf_counter()
            veh_locs = [v.available_at_node for v in veh_list]

            v_times = np.array([v.available_from_time for v in veh_list])

            # A. Pruning: pairs where the vehicle reaches the pickup by its
            # deadline (request_time + Omega, or the promised pickup time)
            r_latest_pickup = np.array([r.latest_pickup_time for r in req_list])
            v_idx, r_idx, arrival = self._reachable_pairs(
                veh_locs, v_times, req_origins, r_latest_pickup
            )

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
                            arrival[mask_0] - req_times[r_idx[mask_0]],  # delay_vals
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

    def _shareable_pairs(
        self,
        req_list: list[Request],
        req_origins: np.ndarray,
        req_dests: np.ndarray,
        req_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Shareable request pairs (i, j), i < j, as np.where on the upper
        triangle orders them, with the best cost and stop order of serving
        req_list[i] and req_list[j] together (_pair_costs).

        Pruning (necessary condition): whichever origin is visited first, the
        second pickup is no earlier than t_first + tau(o_first, o_second), and
        must happen by t_second + Omega. The test and the costs depend only on
        the two requests, and the test is symmetric (the same floats either way
        round), so a pair is tested and solved (both ways round) once, in the
        first batch holding both, and kept while both stay in the pool if it can
        be served. Request ids are unique within a run.
        """
        mat = self._network.get_l1_cache()
        first_new = self._rr_next
        num = np.array([self._rr_num.get(r.id, -1) for r in req_list])
        is_new = num < 0
        num[is_new] = first_new + np.arange(int(is_new.sum()))
        self._rr_next = first_new + int(is_new.sum())
        self._rr_num = {r.id: int(k) for r, k in zip(req_list, num)}
        pos = np.full(self._rr_next, -1)
        pos[num] = np.arange(len(num))

        # Pairs from earlier batches whose requests are both still waiting.
        alive = (pos[self._rr_a] >= 0) & (pos[self._rr_b] >= 0)
        a, b, ab, ba = self._rr_a[alive], self._rr_b[alive], self._rr_ab[alive], self._rr_ba[alive]
        # Pairs with a new request: test it against every request.
        new = np.flatnonzero(is_new)
        if len(new):
            gap = req_times[None, :] - req_times[new, None]  # t_j - t_new
            ok = (mat[req_origins[new, None], req_origins[None, :]] <= self._omega + gap) | (
                mat[req_origins[None, :], req_origins[new, None]] <= self._omega - gap
            )
            k, j = np.nonzero(ok)
            k = new[k]
            once = (k != j) & (~is_new[j] | (k < j))  # a pair of two new ones once
            k, j = k[once], j[once]
            first, second = np.where(num[k] < num[j], k, j), np.where(num[k] < num[j], j, k)
            new_ab = np.stack(self._pair_costs(first, second, req_origins, req_dests, req_times))
            new_ba = np.stack(self._pair_costs(second, first, req_origins, req_dests, req_times))
            served = np.isfinite(new_ab[0]) | np.isfinite(new_ba[0])
            a = np.concatenate([a, num[first[served]]])
            b = np.concatenate([b, num[second[served]]])
            ab = np.concatenate([ab, new_ab[:, served].T])
            ba = np.concatenate([ba, new_ba[:, served].T])
        self._rr_a, self._rr_b, self._rr_ab, self._rr_ba = a, b, ab, ba

        a_first = pos[a] < pos[b]
        i, j = np.where(a_first, pos[a], pos[b]), np.where(a_first, pos[b], pos[a])
        cost_perm = np.where(a_first[:, None], ab, ba)
        valid = np.isfinite(cost_perm[:, 0])
        i, j, cost_perm = i[valid], j[valid], cost_perm[valid]
        order = np.lexsort((j, i))
        return i[order], j[order], cost_perm[order, 0], cost_perm[order, 1].astype(np.int64)

    def _reachable_pairs(
        self,
        veh_locs: list[int],
        v_times: np.ndarray,
        req_origins: np.ndarray,
        latest_pickup: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(vehicle, request) pairs with arrival <= latest pickup, as np.where
        orders them (by vehicle, then request), and their arrival times.

        Only vehicles at nodes within Omega + NEAR_MARGIN of a pickup can make it when
        every budget (latest pickup - vehicle time) is below the radius, which
        holds since budgets are at most Omega; only those pairs are checked
        (about 1% of them). Otherwise every pair is checked.
        """
        budget = latest_pickup.max() - v_times.min()
        if budget + 1.0 > self._omega + NEAR_MARGIN:
            arrival = v_times[:, None] + self._network.get_travel_time_matrix(
                veh_locs, req_origins.tolist()
            )
            v_idx, r_idx = np.where(arrival <= latest_pickup[None, :])
            return v_idx, r_idx, arrival[v_idx, r_idx]

        mat = self._network.get_l1_cache()
        if self._near is None:
            # near_nodes[ptr[o]:ptr[o + 1]]: nodes u with tau(u, o) <= radius
            origin, node = np.nonzero(mat.T <= self._omega + NEAR_MARGIN)
            ptr = np.zeros(len(mat) + 1, dtype=np.int64)
            np.cumsum(np.bincount(origin, minlength=len(mat)), out=ptr[1:])
            self._near = ptr, node
        ptr, near_nodes = self._near

        # Requests x nearby nodes, then x vehicles at those nodes.
        locs = np.asarray(veh_locs)
        by_node = np.argsort(locs, kind="stable")
        count = np.bincount(locs, minlength=len(mat))
        first = np.cumsum(count) - count
        r_idx, node = _expand(ptr[req_origins], ptr[req_origins + 1] - ptr[req_origins])
        node = near_nodes[node]
        r_idx, slot = _expand(first[node], count[node], r_idx)
        v_idx = by_node[slot]

        arrival = v_times[v_idx] + mat[locs[v_idx], req_origins[r_idx]]
        ok = arrival <= latest_pickup[r_idx]
        order = np.lexsort((r_idx[ok], v_idx[ok]))
        return v_idx[ok][order], r_idx[ok][order], arrival[ok][order]

    def _pair_costs(
        self,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
        req_origins: np.ndarray,
        req_dests: np.ndarray,
        req_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Best cost (total delay) and stop order (row of _PERM_ORDER) of serving
        request idx_i (as r1) and idx_j (as r2) together; inf if no order meets
        both requests' Omega and Delta."""
        # 1. Gather data
        d1, d2 = req_dests[idx_i], req_dests[idx_j]
        o1, o2 = req_origins[idx_i], req_origins[idx_j]
        t1, t2 = req_times[idx_i], req_times[idx_j]

        # 2. Fetch only missing travel times (O->D, D->D)
        mat = self._network.get_l1_cache()

        def get_v(u, v):
            return mat[u, v]

        t_o1_o2 = get_v(o1, o2)
        t_o2_o1 = get_v(o2, o1)
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

        # 4. Reduce
        return np.min(costs, axis=0), np.argmin(costs, axis=0)

    def _add_rr_edges(
        self,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
        best_costs: np.ndarray,
        best_perms: np.ndarray,
        req_list: list[Request],
        req_origins: np.ndarray,
        req_dests: np.ndarray,
        graph: RVGraph,
    ) -> None:
        """Add the shareable pairs to the graph with their routes, in order."""
        # A pair's route depends only on the two requests' times and nodes,
        # which never change, so last batch's routes are reused.
        previous, self._rr_routes = self._rr_routes, {}
        keys = [(req_list[i].id, req_list[j].id) for i, j in zip(idx_i, idx_j)]
        missing = np.array([k not in previous for k in keys], dtype=bool)
        if missing.any():
            # Travel time and distance of the best order, leg by leg as in
            # get_shortest_path (legs without a path add nothing).
            mat = self._network.get_l1_cache()
            mi, mj = idx_i[missing], idx_j[missing]
            ends = np.stack([req_origins[mi], req_dests[mi], req_origins[mj], req_dests[mj]]).T
            seq = np.take_along_axis(ends, _PERM_ORDER[best_perms[missing]], axis=1)
            leg_t = mat[seq[:, :-1], seq[:, 1:]]
            leg_d = self._network.get_distance_matrix()[seq[:, :-1], seq[:, 1:]]
            has_path = np.isfinite(leg_t) & (leg_t <= 1e9) & (leg_t >= 0)
            leg_t = np.where(has_path, leg_t, 0.0)
            leg_d = np.where(has_path, leg_d, 0.0)
            totals = iter(
                zip(
                    ((leg_t[:, 0] + leg_t[:, 1]) + leg_t[:, 2]).tolist(),
                    ((leg_d[:, 0] + leg_d[:, 1]) + leg_d[:, 2]).tolist(),
                )
            )
        for k, key in enumerate(keys):
            route = previous.get(key)
            if route is None:
                total_t, total_d = next(totals)
                route = self._build_route_from_perm(
                    best_perms[k], req_list[idx_i[k]], req_list[idx_j[k]], best_costs[k],
                    total_t, total_d,
                )
            self._rr_routes[key] = route
            graph.add_rr_edge(*key, route)

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

        After two cheap lower bounds, extends partial stop orders (onboard
        drop-offs, the pickup, then its drop-off; a free seat when picking up)
        for all pairs at once, keeping only those that meet every deadline so
        far. Travel times are added in the order the trip solver adds them (hard
        deadlines, no waiting), so a route the solver finds survives and only
        pairs without any route are skipped; on real batches that is 95% of them.
        """
        mat = self._network.get_l1_cache()
        n_pairs = len(v_idxs)
        # Per vehicle (once), then per pair by indexing.
        vehicles, v_pos = np.unique(v_idxs, return_inverse=True)
        riders = [[p.original_request for p in veh_list[i].passengers_onboard] for i in vehicles]
        k_max = max(len(q) for q in riders)
        n_stops = k_max + 2
        # Stops 0..k_max-1: onboard drop-offs (unused ones count as visited);
        # k_max: the pickup; k_max + 1: its drop-off.
        v_nodes = np.array([[q.destination_node for q in r] + [0] * (k_max - len(r)) for r in riders])
        v_deadlines = np.array(
            [[q.latest_dropoff_time for q in r] + [0.0] * (k_max - len(r)) for r in riders]
        )
        v_padded = np.array([((1 << k_max) - 1) ^ ((1 << len(r)) - 1) for r in riders])
        v_free = np.array([veh_list[i].capacity - len(veh_list[i].passengers_onboard) for i in vehicles])
        v_at = np.array([veh_list[i].available_at_node for i in vehicles])
        v_t0 = np.array([veh_list[i].available_from_time for i in vehicles])
        requests, r_pos = np.unique(r_idxs, return_inverse=True)
        reqs = [req_list[i] for i in requests]
        r_nodes = np.array([(r.origin_node, r.destination_node) for r in reqs])
        r_deadlines = np.array([(r.latest_pickup_time, r.latest_dropoff_time) for r in reqs])
        nodes = np.concatenate([v_nodes[v_pos], r_nodes[r_pos]], axis=1)
        deadlines = np.concatenate([v_deadlines[v_pos], r_deadlines[r_pos]], axis=1)
        done, free = v_padded[v_pos], v_free[v_pos]
        here, t = v_at[v_pos], v_t0[v_pos]

        # Cheap bounds first (about half the pairs fail them): a route picks the
        # rider up first (a free seat, every onboard drop-off still reachable from
        # the pickup) or drops some passenger j first (pickup via j's destination).
        # The tolerance keeps float rounding of the triangle inequality from
        # rejecting a feasible pair.
        tol = 1e-6
        onboard = (done[:, None] & (1 << np.arange(k_max))) == 0
        origin, dest = nodes[:, k_max], nodes[:, :k_max]
        pickup = t + mat[here, origin]
        late = onboard & (pickup[:, None] + mat[origin[:, None], dest] > deadlines[:, :k_max] + tol)
        via = t[:, None] + mat[here[:, None], dest] + mat[dest, origin[:, None]]
        maybe = ((free > 0) & ~late.any(axis=1)) | (
            onboard & (via <= deadlines[:, k_max, None] + tol)
        ).any(axis=1)

        # Frontier of partial orders: pair, position, time, visited stops, free seats.
        pair = np.flatnonzero(maybe)
        here, t, done, seats = here[pair], t[pair], done[pair], free[pair]
        full = (1 << n_stops) - 1
        bits = 1 << np.arange(n_stops)
        feasible = np.zeros(n_pairs, dtype=bool)
        depth = 0
        while len(pair):
            depth += 1
            # Every next stop of every partial order at once.
            visited = (done[:, None] & bits) != 0
            allowed = ~visited
            allowed[:, k_max] &= seats > 0
            allowed[:, k_max + 1] &= visited[:, k_max]
            arrival = t[:, None] + mat[here[:, None], nodes[pair]]
            state, last = np.nonzero(allowed & (arrival <= deadlines[pair]))
            pair, t = pair[state], arrival[state, last]
            here = nodes[pair, last]
            done = done[state] | bits[last]
            seats = seats[state] + np.where(last == k_max, -1, 1)
            finished = done == full
            feasible[pair[finished]] = True
            alive = ~finished & ~feasible[pair]  # one route per pair is enough
            # Of partial orders with the same pair, visited stops and last stop,
            # the earliest can finish whatever a later one can: keep only it
            # (two orders can only meet like this after three stops).
            if depth < 3:
                first = np.flatnonzero(alive)
            else:
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
