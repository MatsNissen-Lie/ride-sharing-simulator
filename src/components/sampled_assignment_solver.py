"""Joint assignment with sampled future requests (scenario look-ahead)."""

import numpy as np

from src.components.joint_assignment_solver import JointAssignmentSolver
from src.components.zones import ZoneIndex
from src.interfaces import (
    AssignmentResult,
    INetworkService,
    IObjective,
    IRequestSampler,
    IVehicleValue,
    Request,
    Route,
    Stop,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph


class SampledRequestsAssignmentSolver(JointAssignmentSolver):
    """
    The joint ILP with virtual future requests as extra columns, after
    Alonso-Mora, Wallar & Rus (2017, IROS).

    - Scenario: virtual requests with request times in (t_now, t_now + horizon].
      The scenario rolls: requests sampled earlier are kept until their request
      time, and each batch only the newly visible end of the horizon is sampled,
      so a vehicle sent to a virtual rider sees the same rider next batch.
    - Columns: a free vehicle v (nobody on board, at most a move in progress)
      may take virtual rider q if it reaches q's origin by q's latest pickup:

          c_{q,v} = move_cost(v -> o_q) + w * wait_q - w * chi_q + U_v

      with wait_q = max(0, arrival - t_q), chi_q the objective's unserved
      penalty of a fresh request and w the virtual weight. Each virtual rider
      is taken at most once; only the `n_virtual` cheapest columns with a
      negative cost per vehicle are kept.
    - Real riders first: under the delay objective chi = 1e5, so with w <= 0.1
      a virtual rider is worth at most 1e4, far less than the 1e5 of leaving a
      real rider unserved (a trip's delay cost is at most a few thousand).
    - A vehicle that takes a virtual rider drives to its origin: a rebalancing
      move, never a pickup. Moves persist until arrival unless redirected.

    With no virtual requests this is the joint ILP (the paper's with V = 0).
    """

    def __init__(
        self,
        objective: IObjective,
        network: INetworkService,
        zones: ZoneIndex,
        value: IVehicleValue,
        sampler: IRequestSampler,
        horizon: float = 300.0,
        weight: float = 0.05,
        n_virtual: int = 10,
        max_move: float = 900.0,
        n_moves: int = 5,
    ):
        super().__init__(objective, network, zones, value, max_move, n_moves)
        self._sampler = sampler
        self._horizon = horizon
        self._weight = weight
        self._n_virtual = n_virtual
        self.virtual: list[Request] = []
        self._sampled_until: float | None = None
        self._virtual_of_move: dict[int, int] = {}  # move index -> index in self.virtual

    def solve(
        self,
        rtv_graph: RTVGraph,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
        solver_timeout: float,
        current_time: float,
    ) -> AssignmentResult:
        start = current_time if self._sampled_until is None else max(self._sampled_until, current_time)
        end = current_time + self._horizon
        if end > start:
            self.virtual += self._sampler.sample(start, end)
            self._sampled_until = end
        self.virtual = [q for q in self.virtual if q.request_time > current_time]
        return super().solve(rtv_graph, requests, vehicles, solver_timeout, current_time)

    def _moves(self, vehicles, current_time) -> list[tuple[str, float, Route]]:
        moves = super()._moves(vehicles, current_time)
        self._virtual_of_move = {}
        free = sorted(
            (
                v
                for v in vehicles
                if not v.passengers_onboard
                and all(s.is_rebalancing for s in v.remaining_stops_to_serve)
            ),
            key=lambda v: v.vehicle_id,
        )
        if not free or not self.virtual:
            return moves
        origins = [v.available_at_node for v in free]
        pickups = [q.origin_node for q in self.virtual]
        travel = self._network.get_travel_time_matrix(origins, pickups)
        dist = self._network.get_distance_matrix()[np.ix_(origins, pickups)]
        arrival = np.array([v.available_from_time for v in free])[:, None] + travel
        t_request = np.array([q.request_time for q in self.virtual])
        deadline = np.array([q.latest_pickup_time for q in self.virtual])
        reward = self._weight * np.array(
            [self._objective.unserved_penalty(q, q.request_time) for q in self.virtual]
        )
        plan = np.array([self._plan_value[v.vehicle_id] for v in free])
        cost = (
            self._objective.rebalancing_cost(travel, dist)
            + self._weight * np.maximum(arrival - t_request, 0.0)
            - reward
            + plan[:, None]
        )
        cost[~(arrival <= deadline)] = np.inf

        for i, v in enumerate(free):
            best = np.argsort(cost[i])[: self._n_virtual]
            for j in best[cost[i, best] < 0]:
                q = self.virtual[j]
                stops = (Stop.rebalancing(q.origin_node, q.id),) if travel[i, j] > 0 else ()
                self._virtual_of_move[len(moves)] = int(j)
                moves.append((v.vehicle_id, float(cost[i, j]), Route(stops, float(travel[i, j]), float(dist[i, j]), 0.0)))
        return moves

    def _move_groups(self, moves) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for k, j in self._virtual_of_move.items():
            groups.setdefault(j, []).append(k)
        return list(groups.values())
