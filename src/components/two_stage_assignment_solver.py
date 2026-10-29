"""Joint assignment as a two-stage stochastic program over sampled futures (sample average approximation)."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from config import get_logger
from src.components.joint_assignment_solver import JointAssignmentSolver
from src.interfaces import (
    AssignmentResult,
    INetworkService,
    IObjective,
    IRequestSampler,
    IVehicleValue,
    IZoneIndex,
    Request,
    Route,
    Stop,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph, TripId
from src.utils.timing import timed

logger = get_logger(__name__)


class TwoStageAssignmentSolver(JointAssignmentSolver):
    """
    First stage (decided now, integer): the joint ILP's options, i.e. a real
    trip, a move of a free vehicle to a zone, or keeping the current plan.
    Second stage (one per sampled future s = 1..S, continuous): every vehicle,
    in the state its first-stage option leaves it (node and time it is free),
    may serve one rider of future s that it reaches by the rider's latest
    pickup, starting once q has requested (a vehicle can wait where it is but
    cannot drive towards a rider before the rider exists; so pre-positioning
    pays only through being near where riders appear). Leaving a future rider
    q unserved costs w/S * chi_q, so serving it saves

        (w / S) * (chi_q - move_cost(state -> o_q) - wait_q)

    with constraints, per future s:
    - each rider of s is served at most once;
    - a state serves at most x of its option (for keeping the plan,
      1 - sum of the vehicle's other options).

    Given the first stage, each future is a bipartite matching, so the
    continuous second stage is exact. Because a future rider can be claimed
    once, the fifth vehicle sent to a zone gains nothing from riders the
    first four already cover: values fall with the number of vehicles, which
    the per-vehicle values of design C1 lacked.

    - Futures: S fresh samples of the requests in [t_now, t_now + horizon).
    - Moves: each free vehicle may move to the `n_moves` reachable zones
      (within `max_move` s) with the most sampled riders it could still catch
      there; a move is worth only what its state gains in the second stage.
    - Only the `n_recourse` best riders per state and future become columns.
    - The joint ILP's vehicle value V still prices the first stage (`zero`
      for pure two-stage), so the design combines with C3's values.

    With S = 0 (or no sampled riders) moves never pay and this is the joint ILP.
    """

    def __init__(
        self,
        objective: IObjective,
        network: INetworkService,
        zones: IZoneIndex,
        value: IVehicleValue,
        sampler: IRequestSampler,
        n_scenarios: int = 5,
        horizon: float = 600.0,
        weight: float = 0.5,
        n_recourse: int = 3,
        max_move: float = 900.0,
        n_moves: int = 5,
    ):
        super().__init__(objective, network, zones, value, max_move, n_moves)
        self._zones = zones
        self._sampler = sampler
        self._n_scenarios = n_scenarios
        self._horizon = horizon
        self._weight = weight
        self._n_recourse = n_recourse
        self._near_radius: float | None = None
        self._near_pairs = csr_matrix((0, 0))
        self.scenarios: list[list[Request]] = []
        self._rtv = RTVGraph()
        self._vehicles: dict[str, VehicleAvailability] = {}

    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,
    ) -> AssignmentResult:
        with timed("sample"):
            self.scenarios = [
                self._sampler.sample(current_time, current_time + self._horizon) for _ in range(self._n_scenarios)
            ]
        self._rtv = rtv_graph
        self._vehicles = {v.vehicle_id: v for v in vehicles}
        try:
            return super().solve(rtv_graph, requests, vehicles, solver_timeout, current_time)
        except RuntimeError as err:
            # No solution in time (e.g. the first batch, every vehicle idle): decide without the second stage.
            logger.warning(f"Two-stage: {err}; this batch without the second stage.")
            self.scenarios = []
            return super().solve(rtv_graph, requests, vehicles, solver_timeout, current_time)

    @staticmethod
    def _free(vehicles) -> list[VehicleAvailability]:
        return sorted(
            (
                v
                for v in vehicles
                if not v.passengers_onboard and all(s.is_rebalancing for s in v.remaining_stops_to_serve)
            ),
            key=lambda v: v.vehicle_id,
        )

    def _moves(self, vehicles, current_time) -> list[tuple[str, float, Route]]:
        """
        Moves to the `n_moves` reachable zones with the most sampled riders the
        vehicle could still catch there (deadline after its arrival; ties:
        nearer first). Counting all riders of a zone would favour far busy
        zones the vehicle reaches only after their riders are gone.
        """
        free = self._free(vehicles)
        riders = [q for s in self.scenarios for q in s]
        if not free or not riders:
            return []
        col = {z: k for k, z in enumerate(self._zone_ids)}
        zone = np.array([col[int(z)] for z in self._zones.zones_of(np.array([q.origin_node for q in riders]))])
        deadline = np.array([q.latest_pickup_time for q in riders])
        origins = [v.available_at_node for v in free]
        travel = self._network.get_travel_time_matrix(origins, self._centres)
        dist = self._network.get_distance_matrix()[np.ix_(origins, self._centres)]
        arrival = np.array([v.available_from_time for v in free])[:, None] + travel
        catchable = np.zeros(arrival.shape)
        for j in range(len(self._zone_ids)):
            d = np.sort(deadline[zone == j])
            catchable[:, j] = len(d) - np.searchsorted(d, arrival[:, j], side="right")
        ok = (travel > 0) & (travel <= self._max_move) & (catchable > 0)
        order = np.lexsort((travel, -np.where(ok, catchable, -1)), axis=1)[:, : self._n_moves]
        rows = np.repeat(np.arange(len(free)), order.shape[1])
        cols = order.ravel()
        keep = ok[rows, cols]
        rows, cols = rows[keep], cols[keep]
        if not len(rows):
            return []
        value = self._value.values(self._centres[cols], arrival[rows, cols], current_time)
        cost = (
            self._objective.rebalancing_cost(travel[rows, cols], dist[rows, cols])
            - value
            + np.array([self._plan_value[v.vehicle_id] for v in free])[rows]
        )
        moves = []
        for i, j, c in zip(rows.tolist(), cols.tolist(), cost.tolist()):
            stop = Stop.zone_move(int(self._centres[j]), self._zone_ids[j])
            moves.append((free[i].vehicle_id, c, Route((stop,), float(travel[i, j]), float(dist[i, j]), 0.0)))
        return moves

    def _states(self, edges: list[tuple[TripId, str]], moves) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Node and time at which each first-stage option leaves its vehicle free,
        and the option's link: its column (trips, then moves) or, for keeping
        the plan, the vehicle's list of option columns.
        """
        n_x = len(edges)
        own: dict[str, list[int]] = {v_id: [] for v_id in self._vehicles}
        ends, links = [], []
        for e, (trip, v_id) in enumerate(edges):
            ends.append(self._end_state(self._vehicles[v_id], self._rtv.trip_vehicle_details[(trip, v_id)].route.stops))
            links.append(e)
            own[v_id].append(e)
        for k, (v_id, _, route) in enumerate(moves):
            ends.append(self._end_state(self._vehicles[v_id], route.stops))
            links.append(n_x + k)
            own[v_id].append(n_x + k)
        for v_id, v in self._vehicles.items():
            ends.append(self._end_state(v, v.remaining_stops_to_serve))
            links.append(own[v_id])
        nodes, times = (np.array(a) for a in zip(*ends)) if ends else (np.array([], int), np.array([]))
        return nodes.astype(np.int64), times.astype(float), links

    def _near(self, radius: float) -> csr_matrix:
        """Node x node travel time + 1 for pairs within `radius` seconds (the 1 keeps zero times stored)."""
        if self._near_radius != radius:
            times = self._network.get_l1_cache()
            r, c = np.nonzero(times <= radius)
            self._near_pairs = csr_matrix((times[r, c] + 1.0, (r, c)), shape=times.shape)
            self._near_radius = radius
        return self._near_pairs

    def _arcs(self, nodes, times, origin, t_req, deadline, reward) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (source, rider, cost) for the `n_recourse` cheapest riders each source
        (free at `nodes` at `times`) reaches in time. A source serves q only
        within q's patience of its origin, so only those pairs are computed.
        """
        near = self._near(float((deadline - t_req).max()))
        sources = np.flatnonzero(times < deadline.max())
        pairs = near[nodes[sources]].tocsc()[:, origin].tocoo()
        rows, c, travel = sources[pairs.row], pairs.col, pairs.data - 1.0
        arrival = np.maximum(times[rows], t_req[c]) + travel
        dist = self._network.get_distance_matrix()[nodes[rows], origin[c]]
        cost = self._objective.rebalancing_cost(travel, dist) + (arrival - t_req[c]) - reward[c]
        ok = (arrival <= deadline[c]) & (cost < 0)
        rows, c, cost = rows[ok], c[ok], cost[ok]
        # Per source, the `n_recourse` cheapest riders (ties: rider order).
        order = np.lexsort((c, cost, rows))
        rows, c, cost = rows[order], c[order], cost[order]
        best = np.arange(len(rows)) - np.searchsorted(rows, rows) < self._n_recourse
        return rows[best], c[best], cost[best]

    def _chain_arcs(self, future, origin, t_req, deadline, reward):
        """(rider, next rider, cost): riders a vehicle may serve after another one. None here."""
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([])

    def _continuous_columns(self, edges, moves, n_unserved):
        if not any(self.scenarios):
            return None
        nodes, times, links = self._states(edges, moves)
        n_first = len(edges) + len(moves) + n_unserved
        n_s = len(self.scenarios)
        scale = self._weight / n_s

        # Arcs into riders, from vehicle states or from riders; riders are numbered across futures.
        st_from, st_to, st_cost, ch_from, ch_to, ch_cost = [], [], [], [], [], []
        base = 0
        for future in self.scenarios:
            if future:
                origin = np.array([q.origin_node for q in future])
                t_req = np.array([q.request_time for q in future])
                deadline = np.array([q.latest_pickup_time for q in future])
                reward = np.array([self._objective.unserved_penalty(q, q.request_time) for q in future])
                a, b, c = self._arcs(nodes, times, origin, t_req, deadline, reward)
                st_from.append(a), st_to.append(base + b), st_cost.append(scale * c)
                a, b, c = self._chain_arcs(future, origin, t_req, deadline, reward)
                ch_from.append(base + a), ch_to.append(base + b), ch_cost.append(scale * c)
            base += len(future)
        st_from, st_to, ch_from, ch_to = (np.concatenate(x) if x else np.array([], dtype=np.int64)
                                          for x in (st_from, st_to, ch_from, ch_to))
        if not len(st_from):
            return None
        n_st, n_ch = len(st_from), len(ch_from)
        e_cost = np.concatenate(st_cost + ch_cost)
        cols = n_first + np.arange(n_st + n_ch)
        future_of = np.repeat(np.arange(n_s), [len(f) for f in self.scenarios])

        # Rows: each rider served at most once (arcs in <= 1); a rider passes a
        # vehicle on at most once it is served (arcs out - arcs in <= 0); each
        # (state, future) serves at most its option.
        riders, into = np.unique(np.concatenate([st_to, ch_to]), return_inverse=True)
        out_riders, out_row = np.unique(ch_from, return_inverse=True)
        links_used, link_row = np.unique(st_from * n_s + future_of[st_to], return_inverse=True)
        n_r, n_o, n_l = len(riders), len(out_riders), len(links_used)
        r_idx, c_idx, vals = [into], [cols], [np.ones(n_st + n_ch)]
        if n_o:
            # out: +1 on the rider's outgoing arcs, -1 on its incoming arcs.
            r_idx.append(n_r + out_row), c_idx.append(cols[n_st:]), vals.append(np.ones(n_ch))
            pos = np.searchsorted(out_riders, np.concatenate([st_to, ch_to]))
            has = (pos < n_o) & (out_riders[np.minimum(pos, n_o - 1)] == np.concatenate([st_to, ch_to]))
            r_idx.append(n_r + pos[has]), c_idx.append(cols[has]), vals.append(-np.ones(has.sum()))
        r_idx.append(n_r + n_o + link_row), c_idx.append(cols[:n_st]), vals.append(np.ones(n_st))
        upper = np.concatenate([np.ones(n_r), np.zeros(n_o), np.zeros(n_l)])
        for row, key in enumerate(links_used):
            link = links[key // n_s]
            if isinstance(link, list):  # keeping the plan: y + sum of the vehicle's options <= 1
                r_idx.append(np.full(len(link), n_r + n_o + row))
                c_idx.append(np.array(link, dtype=np.int64))
                vals.append(np.ones(len(link)))
                upper[n_r + n_o + row] = 1.0
            else:  # an option: y - x <= 0
                r_idx.append(np.array([n_r + n_o + row]))
                c_idx.append(np.array([link]))
                vals.append(np.array([-1.0]))
        n_rows = n_r + n_o + n_l
        a = coo_matrix(
            (np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
            shape=(n_rows, n_first + n_st + n_ch),
        ).tocsr()
        logger.info(
            f"Two-stage: {n_s} futures, {base} sampled riders, {n_st} state and {n_ch} chain columns, "
            f"{n_r} rider rows, {n_o} chain rows, {n_l} state rows."
        )
        return e_cost, a, np.full(n_rows, -np.inf), upper


class ChainedTwoStageAssignmentSolver(TwoStageAssignmentSolver):
    """
    Two-stage as above, but in each future a vehicle may serve riders one
    after another: after rider q it is free at q's destination at
    t_q + travel(o_q, d_q) (pickup on request, an optimistic bound) and may
    take one of the `n_recourse` cheapest riders it then reaches in time.
    Each future becomes a min-cost flow from vehicle states through riders,
    still exact as an LP. A state is then worth the riders it can chain, not
    at most one, which matters when future riders outnumber vehicles
    everywhere (then every state finds one rider and none looks better).
    One rider at a time: no pooling in the recourse.
    """

    def _chain_arcs(self, future, origin, t_req, deadline, reward):
        dest = np.array([q.destination_node for q in future], dtype=np.int64)
        free = t_req + self._network.get_l1_cache()[origin, dest]
        a, b, c = self._arcs(dest, free, origin, t_req, deadline, reward)
        keep = a != b
        return a[keep], b[keep], c[keep]
