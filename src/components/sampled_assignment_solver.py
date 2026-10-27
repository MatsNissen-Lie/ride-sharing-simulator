"""Joint assignment with sampled future requests (scenario look-ahead)."""

import numpy as np

from config import get_logger
from src.components.joint_assignment_solver import JointAssignmentSolver, ZeroValue
from src.components.request_sampler import VIRTUAL_PREFIX
from src.components.zones import ZoneIndex
from src.interfaces import (
    AssignmentResult,
    FeasibleRoute,
    INetworkService,
    IObjective,
    IRequestSampler,
    Request,
    Route,
    Stop,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph, TripId

logger = get_logger(__name__)


def is_virtual(request_id: str) -> bool:
    return request_id.startswith(VIRTUAL_PREFIX)


class VirtualWeightedObjective(IObjective):
    """An objective whose penalty for leaving a virtual rider unserved is scaled by `weight`."""

    def __init__(self, inner: IObjective, weight: float):
        self._inner = inner
        self._weight = weight

    def trip_cost(self, route, vehicle, requests):
        return self._inner.trip_cost(route, vehicle, requests)

    def unserved_penalty(self, request, current_time):
        if is_virtual(request.id):  # a fresh request: no waiting time yet
            return self._weight * self._inner.unserved_penalty(request, request.request_time)
        return self._inner.unserved_penalty(request, current_time)

    def rebalancing_cost(self, travel_time, distance_m):
        return self._inner.rebalancing_cost(travel_time, distance_m)


class SampledRequestsAssignmentSolver(JointAssignmentSolver):
    """
    The paper's ILP with virtual future requests, after Alonso-Mora, Wallar &
    Rus (2017, IROS). Virtual riders are rows of the ILP like real ones, but
    leaving one unserved costs only w * chi (w = `weight`, chi the objective's
    penalty for a fresh request).

    - Scenario: virtual requests with request times in (t_now, t_now + horizon].
      It rolls: virtual requests are kept until their request time and each
      batch only the newly visible end of the horizon is sampled, so a vehicle
      sent to a virtual rider still sees that rider in the next batch.
    - Options: a free vehicle v (nobody on board, at most a move in progress)
      may take a virtual rider q if it reaches q's origin by q's latest pickup,
      at cost

          c_{q,v} = move_cost(v -> o_q) + w * wait_q + U_v

      with wait_q = max(0, arrival - t_q) and U_v the joint ILP's offset for a
      move still in progress. Only the `n_virtual` cheapest options with a
      gain (c < w * chi) per vehicle are kept, and only virtual riders with an
      option get a row.
    - Commit: real trips as usual; a vehicle that takes only a virtual rider
      drives to its origin (a rebalancing move, never a pickup). Moves persist
      until arrival unless the ILP redirects the vehicle.
    - Real riders first: under the delay objective chi = 1e5, so with w <= 0.1
      a virtual rider is worth at most 1e4, a tenth of a real rider.

    With no virtual requests this is the paper's ILP.
    """

    def __init__(
        self,
        objective: IObjective,
        network: INetworkService,
        zones: ZoneIndex,
        sampler: IRequestSampler,
        horizon: float = 300.0,
        weight: float = 0.05,
        n_virtual: int = 10,
    ):
        super().__init__(VirtualWeightedObjective(objective, weight), network, zones, ZeroValue())
        self._sampler = sampler
        self._horizon = horizon
        self._weight = weight
        self._n_virtual = n_virtual
        self.virtual: list[Request] = []
        self._sampled_until: float | None = None
        self._real_graph = RTVGraph()
        self._virtual_costs: dict[tuple[TripId, str], float] = {}
        self._free: set[str] = set()

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

        self._t_now = current_time
        self._plan_value = self._current_plan_values(vehicles)
        options = self._virtual_options(rtv_graph, vehicles)
        self._real_graph = rtv_graph
        self._virtual_costs = {key: cost for key, (_, cost) in options.items()}
        graph = RTVGraph(
            trip_vehicle_details=rtv_graph.trip_vehicle_details
            | {key: route for key, (route, _) in options.items()}
        )
        used = {r for trip, _ in options for r in trip}
        virtual = {q for q in self.virtual if q.id in used}
        result = super().solve(graph, requests | virtual, vehicles, solver_timeout, current_time)
        logger.info(
            f"Virtual riders: {len(self.virtual)} sampled, {len(virtual)} with options, "
            f"{len(options)} options; {len(result.rebalancing)} vehicles sent to one."
        )
        return AssignmentResult(
            assignments=result.assignments,
            unassigned_requests={r for r in result.unassigned_requests if not is_virtual(r.id)},
            total_cost=result.total_cost,
            is_optimal=result.is_optimal,
            rebalancing=result.rebalancing,
        )

    def _virtual_options(
        self, rtv_graph: RTVGraph, vehicles: set[VehicleAvailability]
    ) -> dict[tuple[TripId, str], tuple[FeasibleRoute, float]]:
        """
        Columns that serve virtual riders: (trip, v) -> (route, cost on top of
        the trip's real part). Here: ({q}, v) for free vehicles.
        """
        free = sorted(
            (
                v
                for v in vehicles
                if not v.passengers_onboard
                and all(s.is_rebalancing for s in v.remaining_stops_to_serve)
            ),
            key=lambda v: v.vehicle_id,
        )
        self._free = {v.vehicle_id for v in free}
        if not free or not self.virtual:
            return {}
        origins = [v.available_at_node for v in free]
        pickups = [q.origin_node for q in self.virtual]
        travel = self._network.get_travel_time_matrix(origins, pickups)
        dist = self._network.get_distance_matrix()[np.ix_(origins, pickups)]
        arrival = np.array([v.available_from_time for v in free])[:, None] + travel
        cost = (
            self._objective.rebalancing_cost(travel, dist)
            + self._weight * self._wait(arrival)
            + np.array([self._plan_value[v.vehicle_id] for v in free])[:, None]
        )
        gain = self._reward()[None, :] - cost
        gain[~(arrival <= self._deadlines()[None, :])] = -np.inf

        options = {}
        for i, v in enumerate(free):
            best = np.argsort(-gain[i])[: self._n_virtual]
            for j in best[gain[i, best] > 0]:
                q = self.virtual[j]
                route = Route((Stop(q.origin_node, True, q.id),), float(travel[i, j]), float(dist[i, j]), 0.0)
                options[(frozenset({q.id}), v.vehicle_id)] = (FeasibleRoute(route, 0.0), float(cost[i, j]))
        return options

    def _wait(self, arrival: np.ndarray) -> np.ndarray:
        """Wait of each virtual rider (columns) for a vehicle arriving at `arrival`."""
        return np.maximum(arrival - np.array([q.request_time for q in self.virtual])[None, :], 0.0)

    def _deadlines(self) -> np.ndarray:
        return np.array([q.latest_pickup_time for q in self.virtual])

    def _reward(self) -> np.ndarray:
        """w * chi of each virtual rider: what serving it saves in the ILP."""
        return np.array([self._objective.unserved_penalty(q, self._t_now) for q in self.virtual])

    def _edge_costs(self, rtv_graph, requests, vehicles):
        """Real trips as in the joint ILP; a virtual option adds its cost to that of its real part."""
        real = {r for r in requests if not is_virtual(r.id)}
        costs = super()._edge_costs(self._real_graph, real, vehicles)
        for (trip, v_id), extra in self._virtual_costs.items():
            real_part = frozenset(r for r in trip if not is_virtual(r))
            costs[(trip, v_id)] = (costs[(real_part, v_id)] if real_part else 0.0) + extra
        return costs

    def _result(self, rtv_graph, requests, chosen, total_cost, is_optimal, rebalancing=None):
        """Commit real trips; a free vehicle chosen for a virtual rider moves to its origin."""
        real, moves = [], dict(rebalancing or {})
        for trip, v_id in chosen:
            if not any(is_virtual(r) for r in trip):
                real.append((trip, v_id))
                continue
            route = rtv_graph.get_route(trip, v_id).route
            real_part = frozenset(r for r in trip if not is_virtual(r))
            if real_part:
                real.append((real_part, v_id))
            elif v_id in self._free:
                moves[v_id] = self._move_to_virtual(route)
        return super()._result(self._real_graph, requests, real, total_cost, is_optimal, moves)

    @staticmethod
    def _move_to_virtual(route: Route) -> Route:
        if route.total_travel_time <= 0:
            return Route((), 0.0, 0.0, 0.0)  # already there: stop
        first = next(s for s in route.stops if is_virtual(s.request_id))
        return Route((Stop.rebalancing(first.node, first.request_id),), route.total_travel_time,
                     route.total_travel_distance, 0.0)

    def _moves(self, vehicles, current_time):
        """No zone moves: with V = 0 they would only cut moves to virtual riders short."""
        return []


class SampledChainAssignmentSolver(SampledRequestsAssignmentSolver):
    """
    Also lets every option claim a virtual rider after it: after the trip T of
    an RTV edge, or after its current plan for a vehicle carrying riders. The
    chain says where the option leaves the vehicle; (T + q, v) costs

        c'_{T,v} + w * (move_cost(end of T -> o_q) + wait_q)

    (the move and the wait happen only if q turns up, so both get weight w).
    Only the `n_chain` cheapest virtual riders per option are kept. A chosen
    chain commits only its real part, so a vehicle that claims a virtual rider
    after its current plan just keeps that plan.
    """

    def __init__(self, *args, n_chain: int = 3, chunk: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_chain = n_chain
        self._chunk = chunk

    def _virtual_options(self, rtv_graph, vehicles):
        options = super()._virtual_options(rtv_graph, vehicles)
        veh = {v.vehicle_id: v for v in vehicles}
        bases = [(trip, v_id, fr.route.stops) for (trip, v_id), fr in rtv_graph.trip_vehicle_details.items()]
        bases += [(frozenset(), v.vehicle_id, v.remaining_stops_to_serve) for v in vehicles if v.passengers_onboard]
        if not bases or not self.virtual:
            return options
        ends = [self._end_state(veh[v_id], stops) for _, v_id, stops in bases]
        nodes, times = [n for n, _ in ends], np.array([t for _, t in ends])
        pickups = [q.origin_node for q in self.virtual]
        deadline = self._deadlines()
        k = min(self._n_chain, len(pickups))
        for lo in range(0, len(bases), self._chunk):
            rows = slice(lo, lo + self._chunk)
            travel = self._network.get_travel_time_matrix(nodes[rows], pickups)
            dist = self._network.get_distance_matrix()[np.ix_(nodes[rows], pickups)]
            arrival = times[rows, None] + travel
            extra = self._weight * (self._objective.rebalancing_cost(travel, dist) + self._wait(arrival))
            extra[~(arrival <= deadline[None, :])] = np.inf
            for i, best in enumerate(np.argpartition(extra, k - 1, axis=1)[:, :k]):
                trip, v_id, stops = bases[lo + i]
                for j in best[np.isfinite(extra[i, best])]:
                    q = self.virtual[j]
                    route = Route((*stops, Stop(q.origin_node, True, q.id)), 0.0, 0.0, 0.0)
                    options[(trip | {q.id}, v_id)] = (FeasibleRoute(route, 0.0), float(extra[i, j]))
        return options
