"""Forecast-based rebalancing with the fewest vehicle-seconds moved."""

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix

from src.components.demand import integer_targets, zone_vector
from src.interfaces import (
    IDemandModel,
    INetworkService,
    IRebalancingSolver,
    IZoneIndex,
    Request,
    VehicleAvailability,
)
from src.interfaces.data_models import Route, Stop

STOP = Route((), 0.0, 0.0, 0.0)


class TransportRebalancingSolver(IRebalancingSolver):
    """
    1. Waiting riders a free vehicle can still reach before their pickup
       deadline get one, matched by minimum total travel time. Riders no free
       vehicle can reach are not chased.
    2. The other free vehicles are spread like the expected requests over the
       next `horizon` seconds: target_z = N * share_z (largest remainder),
       where N counts free vehicles, vehicles sent in step 1 and vehicles
       already moving to a zone (counted where they head). Which vehicles move
       where is a transportation problem: fill as many target places as
       possible, then minimise total travel time, with moves of at most
       `max_travel` seconds. Its LP is a bipartite b-matching, so the simplex
       solution is integral.

    Free vehicles are idle ones and ones still driving to a rider; zone moves
    persist until arrival (they are counted, not re-planned).
    """

    def __init__(
        self,
        network: INetworkService,
        zones: IZoneIndex,
        demand: IDemandModel,
        horizon: float = 900.0,
        max_travel: float = 900.0,
    ):
        self._network = network
        self._zones = zones
        self._demand = demand
        self._horizon = horizon
        self._max_travel = max_travel
        self._zone_ids = [int(z) for z in zones.zone_ids]
        self._col = {z: k for k, z in enumerate(self._zone_ids)}
        self._centres = [zones.center_node[z] for z in self._zone_ids]

    def solve(
        self,
        idle_vehicles: list[VehicleAvailability],
        unassigned_requests: list[Request],
        t_now: float,
        en_route: list[VehicleAvailability] = (),
    ) -> dict[str, Route]:
        zone_move = [v for v in en_route if v.remaining_stops_to_serve[-1].is_zone_move]
        free = sorted(
            [*idle_vehicles, *(v for v in en_route if v not in zone_move)],
            key=lambda v: v.vehicle_id,
        )
        routes = {v.vehicle_id: STOP for v in en_route if v not in zone_move}

        routes.update(self._serve_reachable(free, unassigned_requests))
        free = [v for v in free if routes.get(v.vehicle_id, STOP) is STOP]

        expected = zone_vector(
            self._demand.expected_requests(t_now, t_now + self._horizon), self._zone_ids
        )
        if not free or expected.sum() <= 0:
            return routes

        heading = [r.stops[-1].node for r in routes.values() if r.stops]
        heading += [v.remaining_stops_to_serve[-1].node for v in zone_move]
        n_total = len(free) + len(heading)
        target = integer_targets(expected / expected.sum() * n_total)
        for z in self._zones.zones_of(heading):
            target[self._col[int(z)]] -= 1
        routes.update(self._transport(free, np.maximum(target, 0)))
        return routes

    def _serve_reachable(self, free, riders) -> dict[str, Route]:
        if not free or not riders:
            return {}
        travel = self._network.get_travel_time_matrix(
            [v.available_at_node for v in free], [r.origin_node for r in riders]
        )
        arrival = np.array([v.available_from_time for v in free])[:, None] + travel
        deadline = np.array([r.latest_pickup_time for r in riders])[None, :]
        cost = np.where(arrival <= deadline, travel, 1e9)
        out = {}
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e9:
                stop = Stop.rebalancing(riders[j].origin_node, riders[j].id)
                out[free[i].vehicle_id] = Route((stop,), float(travel[i, j]), 0.0, 0.0)
        return out

    def _transport(self, free, slots: np.ndarray) -> dict[str, Route]:
        """Vehicle i to zone z (staying counts for its own zone at no cost)."""
        home = [self._col[int(z)] for z in self._zones.zones_of([v.available_at_node for v in free])]
        travel = self._network.get_travel_time_matrix(
            [v.available_at_node for v in free], self._centres
        )
        reward = 2.0 * self._max_travel + 1.0  # filling a place beats any allowed move
        rows, cols = [], []
        for i in range(len(free)):
            for z in range(len(self._zone_ids)):
                if slots[z] > 0 and (z == home[i] or travel[i, z] <= self._max_travel):
                    rows.append(i)
                    cols.append(z)
        if not rows:
            return {}
        rows, cols = np.array(rows), np.array(cols)
        cost = np.where(cols == np.array(home)[rows], 0.0, travel[rows, cols]) - reward
        n, m = len(free), len(self._zone_ids)
        k = np.arange(len(rows))
        a = coo_matrix(
            (np.ones(2 * len(rows)), (np.concatenate([rows, n + cols]), np.concatenate([k, k]))),
            shape=(n + m, len(rows)),
        )
        result = linprog(
            cost,
            A_ub=a.tocsr(),
            b_ub=np.concatenate([np.ones(n), slots]),
            bounds=(0, 1),
            method="highs-ds",
        )
        if result.x is None:
            raise RuntimeError(f"Rebalancing transport LP failed: {result.message}")

        out = {}
        for e in np.flatnonzero(result.x > 0.5):
            i, z = rows[e], cols[e]
            if z != home[i]:
                stop = Stop.zone_move(self._centres[z], self._zone_ids[z])
                out[free[i].vehicle_id] = Route((stop,), float(travel[i, z]), 0.0, 0.0)
        return out
