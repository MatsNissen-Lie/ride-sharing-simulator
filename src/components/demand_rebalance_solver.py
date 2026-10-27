"""Forecast-based rebalancing: spread idle vehicles like the expected demand."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.components.demand import integer_targets, zone_vector
from src.components.rebalance_solver import ReactiveRebalancingSolver
from src.interfaces import (
    IDemandModel,
    INetworkService,
    IRebalancingSolver,
    IZoneIndex,
    Request,
    VehicleAvailability,
)
from src.interfaces.data_models import Route, Stop


class DemandRebalancingSolver(IRebalancingSolver):
    """
    1. Unassigned riders first: idle vehicles are matched to them as in the
       reactive rebalancer (Alonso-Mora et al., Algorithm 4).
    2. The remaining idle vehicles, plus vehicles already heading somewhere,
       are spread over zones in proportion to the expected requests in the next
       `horizon` seconds: target_z = N * share_z, rounded so targets sum to N.
    3. Surplus idle vehicles move to deficit zones (the zone's representative
       node), matched to minimise total travel time; moves longer than
       `max_travel` seconds are not made.

    In the spirit of Wallar et al. (2018); vehicles carrying riders are not
    counted (only idle and rebalancing supply is balanced).
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
        self._reactive = ReactiveRebalancingSolver(network)

    def solve(
        self,
        idle_vehicles: list[VehicleAvailability],
        unassigned_requests: list[Request],
        t_now: float,
        en_route: list[VehicleAvailability] = (),
    ) -> dict[str, Route]:
        # Vehicles chasing a rider are re-planned every batch; zone moves persist.
        zone_move = [v for v in en_route if v.remaining_stops_to_serve[-1].is_zone_move]
        chasing = [v for v in en_route if v not in zone_move]
        routes = self._reactive.solve(idle_vehicles, unassigned_requests, t_now, chasing)
        free = [
            v for v in [*idle_vehicles, *chasing]
            if v.vehicle_id not in routes or not routes[v.vehicle_id].stops
        ]

        zone_ids = self._zones.zone_ids
        col = {int(z): k for k, z in enumerate(zone_ids)}
        expected = zone_vector(self._demand.expected_requests(t_now, t_now + self._horizon), zone_ids)
        if not free or expected.sum() <= 0:
            return routes

        # Current supply per zone: free vehicles where they are, others where they head.
        heading = [r.stops[-1].node for r in routes.values() if r.stops]
        heading += [v.remaining_stops_to_serve[-1].node for v in zone_move]
        supply = np.zeros(len(zone_ids))
        for z in self._zones.zones_of([v.available_at_node for v in free] + heading):
            supply[col[int(z)]] += 1
        target = integer_targets(expected / expected.sum() * supply.sum())

        # Movers: free vehicles in surplus zones (as many as the surplus).
        movers: list[VehicleAvailability] = []
        free_by_zone: dict[int, list[VehicleAvailability]] = {}
        for v in sorted(free, key=lambda v: v.vehicle_id):
            free_by_zone.setdefault(self._zones.zone(v.available_at_node), []).append(v)
        for z, vehicles in free_by_zone.items():
            surplus = int(supply[col[z]] - target[col[z]])
            movers += vehicles[: max(surplus, 0)]
        slots = [
            int(zone_ids[k])
            for k in range(len(zone_ids))
            for _ in range(max(int(target[k] - supply[k]), 0))
        ]
        if not movers or not slots:
            return routes

        dest = [self._zones.center_node[z] for z in slots]
        travel = self._network.get_travel_time_matrix([v.available_at_node for v in movers], dest)
        cost = np.where(travel <= self._max_travel, travel, 1e9)
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e9 and travel[i, j] > 0:
                stop = Stop.zone_move(dest[j], slots[j])
                routes[movers[i].vehicle_id] = Route((stop,), float(travel[i, j]), 0.0, 0.0)
        return routes
