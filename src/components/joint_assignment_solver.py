"""Assignment and rebalancing in one ILP, weighing where each vehicle ends up."""

import numpy as np

from src.components.assignment_solver import HighsAssignmentSolver
from src.components.zones import ZoneIndex
from src.interfaces import (
    AssignmentResult,
    INetworkService,
    IObjective,
    IVehicleValue,
    Request,
    Route,
    Stop,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph, TripId


class ZeroValue(IVehicleValue):
    """No look-ahead: the joint ILP reduces to the paper's assignment."""

    def values(self, nodes, times, t_now):
        return np.zeros(len(nodes))


class JointAssignmentSolver(HighsAssignmentSolver):
    """
    Each vehicle takes at most one option: a trip from the RTV graph, a move
    to a nearby zone (only if it carries nobody), or neither (it keeps its
    current plan). An option is scored by its objective cost minus the value
    V(node, time) of the vehicle where and when the option leaves it free,
    relative to keeping the current plan:

        c'_{T,v} = c_{T,v}           - V(end of T)      + U_v
        c'_{z,v} = move_cost(v -> z) - V(zone z centre)  + U_v
        U_v      = V(end of current plan) - cost of the move it is still making

    Only the `n_moves` best moves per vehicle with a positive gain become
    columns; moves longer than `max_move` seconds are not considered. With
    V = 0 no move has a gain and trip costs are unchanged (for vehicles not
    already rebalancing), so the ILP is the paper's.
    """

    def __init__(
        self,
        objective: IObjective,
        network: INetworkService,
        zones: ZoneIndex,
        value: IVehicleValue,
        max_move: float = 900.0,
        n_moves: int = 5,
    ):
        super().__init__(objective)
        self._network = network
        self._value = value
        self._max_move = max_move
        self._n_moves = n_moves
        self._zone_ids = [int(z) for z in zones.zone_ids]
        self._centres = np.array([zones.center_node[z] for z in self._zone_ids])
        self._t_now = 0.0
        self._plan_value: dict[str, float] = {}

    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,
    ) -> AssignmentResult:
        self._t_now = current_time
        self._value.prepare(vehicles, requests, current_time)
        self._plan_value = self._current_plan_values(vehicles)
        return super().solve(rtv_graph, requests, vehicles, solver_timeout, current_time)

    def _end_state(self, vehicle: VehicleAvailability, stops) -> tuple[int, float]:
        """Node and time at which `vehicle` is free after driving `stops`."""
        times = self._network.get_l1_cache()
        node, t = vehicle.available_at_node, vehicle.available_from_time
        for stop in stops:
            t += times[node, stop.node]
            node = stop.node
        return node, t

    def _current_plan_values(self, vehicles: set[VehicleAvailability]) -> dict[str, float]:
        ids, nodes, times, sunk = [], [], [], []
        dist = self._network.get_distance_matrix()
        for v in vehicles:
            stops = v.remaining_stops_to_serve
            node, t = self._end_state(v, stops)
            ids.append(v.vehicle_id)
            nodes.append(node)
            times.append(t)
            moving = bool(stops) and all(s.is_rebalancing for s in stops)
            sunk.append(
                self._objective.rebalancing_cost(
                    t - v.available_from_time, dist[v.available_at_node, node]
                )
                if moving
                else 0.0
            )
        value = self._value.values(np.array(nodes, dtype=np.int64), np.array(times), self._t_now)
        return {i: float(val - c) for i, val, c in zip(ids, value, sunk)}

    def _edge_costs(self, rtv_graph, requests, vehicles) -> dict[tuple[TripId, str], float]:
        costs = super()._edge_costs(rtv_graph, requests, vehicles)
        veh_map = {v.vehicle_id: v for v in vehicles}
        keys = list(costs)
        ends = [
            self._end_state(veh_map[v_id], rtv_graph.trip_vehicle_details[(trip, v_id)].route.stops)
            for trip, v_id in keys
        ]
        if not ends:
            return costs
        nodes, times = zip(*ends)
        value = self._value.values(np.array(nodes, dtype=np.int64), np.array(times), self._t_now)
        return {
            key: costs[key] - float(val) + self._plan_value[key[1]]
            for key, val in zip(keys, value)
        }

    def _moves(self, vehicles, current_time) -> list[tuple[str, float, Route]]:
        free = sorted(
            (
                v
                for v in vehicles
                if not v.passengers_onboard
                and all(s.is_rebalancing for s in v.remaining_stops_to_serve)
            ),
            key=lambda v: v.vehicle_id,
        )
        if not free:
            return []
        origins = [v.available_at_node for v in free]
        travel = self._network.get_travel_time_matrix(origins, self._centres)
        dist = self._network.get_distance_matrix()[np.ix_(origins, self._centres)]
        arrival = np.array([v.available_from_time for v in free])[:, None] + travel
        reachable = (travel > 0) & (travel <= self._max_move)
        rows, cols = np.nonzero(reachable)
        value = self._value.values(self._centres[cols], arrival[rows, cols], current_time)

        moves: list[tuple[str, float, Route]] = []
        by_vehicle: dict[int, list[tuple[float, int]]] = {}
        for i, j, val in zip(rows, cols, value):
            cost = (
                self._objective.rebalancing_cost(float(travel[i, j]), float(dist[i, j]))
                - float(val)
                + self._plan_value[free[i].vehicle_id]
            )
            if cost < 0:
                by_vehicle.setdefault(int(i), []).append((cost, int(j)))
        for i, options in by_vehicle.items():
            for cost, j in sorted(options)[: self._n_moves]:
                stop = Stop.rebalancing(int(self._centres[j]), f"zone{self._zone_ids[j]}")
                route = Route((stop,), float(travel[i, j]), float(dist[i, j]), 0.0)
                moves.append((free[i].vehicle_id, cost, route))
        return moves
