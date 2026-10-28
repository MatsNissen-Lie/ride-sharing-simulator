"""Joint assignment with sampled future requests (scenario look-ahead)."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import get_logger
from src.components.joint_assignment_solver import JointAssignmentSolver
from src.interfaces import IZoneIndex
from src.interfaces import (
    AssignmentResult,
    FeasibleRoute,
    INetworkService,
    IObjective,
    IRequestSampler,
    IVehicleValue,
    Request,
    Route,
    Stop,
    VehicleAvailability,
)
from src.interfaces.graphs import RTVGraph, TripId
from src.utils.timing import timed

logger = get_logger(__name__)


class VirtualWeightedObjective(IObjective):
    """An objective whose penalty for leaving a virtual rider (id in `virtual_ids`) unserved is scaled by `weight`."""

    def __init__(self, inner: IObjective, weight: float, virtual_ids: set[str]):
        self._inner = inner
        self._weight = weight
        self._virtual_ids = virtual_ids

    def trip_cost(self, route, vehicle, requests):
        return self._inner.trip_cost(route, vehicle, requests)

    def unserved_penalty(self, request, current_time):
        if request.id in self._virtual_ids:  # a fresh request: no waiting time yet
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

    - Scenario: virtual requests with request times up to t_now + horizon.
      It rolls: each batch only the newly visible end of the horizon is
      sampled, and a virtual rider stays until its latest pickup, like a real
      one waiting in the pool. A vehicle sent to a virtual rider therefore
      still sees it in the next batches, until it could have picked it up.
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

    With no virtual requests this is the paper's ILP. Costs of virtual
    options ignore the vehicle value V (use `zero`); V still prices real trips.
    """

    def __init__(
        self,
        objective: IObjective,
        network: INetworkService,
        zones: IZoneIndex,
        value: IVehicleValue,
        sampler: IRequestSampler,
        horizon: float = 300.0,
        weight: float = 0.05,
        n_virtual: int = 20,
    ):
        self._virtual_ids: set[str] = set()
        self._costs = VirtualWeightedObjective(objective, weight, self._virtual_ids)
        self._net = network
        super().__init__(self._costs, network, zones, value)
        self._sampler = sampler
        self._horizon = horizon
        self._weight = weight
        self._n_virtual = n_virtual
        self.virtual: list[Request] = []
        self.active: list[Request] = []  # the virtual riders offered to vehicles this batch
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
        with timed("sample"):
            if end > start:
                self.virtual += self._sampler.sample(start, end)
                self._sampled_until = end
            self.virtual = [q for q in self.virtual if q.latest_pickup_time > current_time]
            self._virtual_ids.clear()
            self._virtual_ids.update(q.id for q in self.virtual)
            self.active = self._active(vehicles, current_time)

        with timed("virtual_options"):
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
            f"Virtual riders: {len(self.virtual)} sampled, {len(self.active)} active, {len(virtual)} with options, "
            f"{len(options)} options; {len(self._free)} free vehicles, "
            f"{len({v for _, v in options} & self._free)} with options, {len(result.rebalancing)} sent to one."
        )
        return AssignmentResult(
            assignments=result.assignments,
            unassigned_requests={r for r in result.unassigned_requests if r.id not in self._virtual_ids},
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
        if not free or not self.active:
            return {}
        origins = [v.available_at_node for v in free]
        pickups = [q.origin_node for q in self.active]
        travel = self._net.get_travel_time_matrix(origins, pickups)
        arrival = np.array([v.available_from_time for v in free])[:, None] + travel
        # Costs only for the pairs in time (most are not); the rest gain -inf.
        rows, cols = np.nonzero(arrival <= self._deadlines()[None, :])
        travel, arrival = travel[rows, cols], arrival[rows, cols]
        dist = self._net.get_distance_matrix()[np.array(origins)[rows], np.array(pickups)[cols]]
        cost = (
            self._costs.rebalancing_cost(travel, dist)
            + self._weight * self._wait(arrival, cols)
            - np.array([self._move_left(v) for v in free])[rows]
        )
        gain = np.full((len(free), len(self.active)), -np.inf)
        gain[rows, cols] = self._reward()[cols] - cost

        with timed("candidates"):
            keep = self._candidates(gain)
        options = {}
        # Columns are a subset of the pairs in time, in the same (row-major) order.
        pick = np.flatnonzero(keep[rows, cols])
        for i, j, t, d, c in zip(
            rows[pick].tolist(), cols[pick].tolist(), travel[pick].tolist(), dist[pick].tolist(), cost[pick].tolist()
        ):
            v, q = free[i], self.active[j]
            route = Route((Stop(q.origin_node, True, q.id),), t, d, 0.0)
            options[(frozenset({q.id}), v.vehicle_id)] = (FeasibleRoute(route, 0.0), c)
        return options

    def _candidates(self, gain: np.ndarray) -> np.ndarray:
        """Which (free vehicle, virtual rider) pairs become columns: the `n_virtual` best per vehicle."""
        keep = np.zeros(gain.shape, dtype=bool)
        k = min(self._n_virtual, gain.shape[1])
        rows = np.arange(gain.shape[0])[:, None]
        keep[rows, np.argpartition(-gain, k - 1, axis=1)[:, :k]] = True
        return keep & (gain > 0)

    def _active(self, vehicles: set[VehicleAvailability], current_time: float) -> list[Request]:
        """Virtual riders offered to vehicles this batch: all of them."""
        return self.virtual

    def _wait(self, arrival: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Wait of virtual riders `cols` (indices into `active`) for vehicles arriving at `arrival`."""
        return np.maximum(arrival - np.array([q.request_time for q in self.active])[cols], 0.0)

    def _deadlines(self) -> np.ndarray:
        return np.array([q.latest_pickup_time for q in self.active])

    def _reward(self) -> np.ndarray:
        """w * chi of each virtual rider: what serving it saves in the ILP."""
        return np.array([self._costs.unserved_penalty(q, q.request_time) for q in self.active])

    def _free_at(self, vehicle: VehicleAvailability, stops) -> tuple[int, float]:
        """Node and time at which `vehicle` is free after driving `stops`."""
        node, t = vehicle.available_at_node, vehicle.available_from_time
        for stop in stops:
            t += self._net.get_travel_time(node, stop.node)
            node = stop.node
        return node, t

    def _move_left(self, vehicle: VehicleAvailability) -> float:
        """Cost of the rest of a free vehicle's move in progress (the joint ILP's -U_v under V = 0)."""
        if not vehicle.remaining_stops_to_serve:
            return 0.0
        node, t = self._free_at(vehicle, vehicle.remaining_stops_to_serve)
        dist = self._net.get_distance_matrix()[vehicle.available_at_node, node]
        return float(self._costs.rebalancing_cost(t - vehicle.available_from_time, dist))

    def _edge_costs(self, rtv_graph, requests, vehicles):
        """Real trips as in the joint ILP; a virtual option adds its cost to that of its real part."""
        real = {r for r in requests if r.id not in self._virtual_ids}
        costs = super()._edge_costs(self._real_graph, real, vehicles)
        for (trip, v_id), extra in self._virtual_costs.items():
            real_part = trip - self._virtual_ids
            costs[(trip, v_id)] = (costs[(real_part, v_id)] if real_part else 0.0) + extra
        return costs

    def _result(self, rtv_graph, requests, chosen, total_cost, is_optimal, rebalancing=None):
        """Commit real trips; a free vehicle chosen for a virtual rider moves to its origin."""
        real, moves = [], dict(rebalancing or {})
        for trip, v_id in chosen:
            real_part = trip - self._virtual_ids
            if real_part == trip:
                real.append((trip, v_id))
                continue
            route = rtv_graph.get_route(trip, v_id).route
            if real_part:
                real.append((real_part, v_id))
            elif v_id in self._free:
                moves[v_id] = self._move_to_virtual(route)
        return super()._result(self._real_graph, requests, real, total_cost, is_optimal, moves)

    def _move_to_virtual(self, route: Route) -> Route:
        if route.total_travel_time <= 0:
            return Route((), 0.0, 0.0, 0.0)  # already there: stop
        first = next(s for s in route.stops if s.request_id in self._virtual_ids)
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
        if not bases or not self.active:
            return options
        ends = [self._free_at(veh[v_id], stops) for _, v_id, stops in bases]
        nodes, times = np.array([n for n, _ in ends]), np.array([t for _, t in ends])
        pickups = np.array([q.origin_node for q in self.active])
        t_request = np.array([q.request_time for q in self.active])
        deadline = self._deadlines()
        # Most trips end after every virtual deadline: skip them, and give each
        # chunk of options (sorted by end time) only the riders it could reach.
        by_end = np.argsort(times)
        by_end = by_end[times[by_end] < deadline.max()]
        by_deadline = np.argsort(deadline)
        for lo in range(0, len(by_end), self._chunk):
            rows = by_end[lo : lo + self._chunk]
            cols = by_deadline[np.searchsorted(deadline[by_deadline], times[rows[0]], side="right") :]
            travel = self._net.get_travel_time_matrix(nodes[rows], pickups[cols])
            dist = self._net.get_distance_matrix()[np.ix_(nodes[rows], pickups[cols])]
            arrival = times[rows, None] + travel
            wait = np.maximum(arrival - t_request[cols][None, :], 0.0)
            extra = self._weight * (self._costs.rebalancing_cost(travel, dist) + wait)
            extra[~(arrival <= deadline[cols][None, :])] = np.inf
            k = min(self._n_chain, len(cols))
            for i, best in enumerate(np.argpartition(extra, k - 1, axis=1)[:, :k]):
                trip, v_id, stops = bases[rows[i]]
                for j in best[np.isfinite(extra[i, best])]:
                    q = self.active[cols[j]]
                    route = Route((*stops, Stop(q.origin_node, True, q.id)), 0.0, 0.0, 0.0)
                    options[(trip | {q.id}, v_id)] = (FeasibleRoute(route, 0.0), float(extra[i, j]))
        return options


class SupplyMatched:
    """
    Mixin for the sampled solvers: offer only about `virtual_per_free` virtual
    riders per free vehicle. In the scarce regime future riders outnumber free
    vehicles several times everywhere, so with every virtual rider offered each
    free vehicle finds one next to it and hardly anyone moves. Matching free
    vehicles to a demand-distributed sample of about their own number spreads
    them like the demand instead (the vehicle-level counterpart of
    demand-proportional zone targets).

    Each virtual rider draws a fixed mark u ~ U(0, 1) when sampled; riders with
    u < min(1, virtual_per_free * free vehicles / virtual riders) are active.
    The active set is a random thinning of the scenario that changes little
    between batches, so moves stay stable.
    """

    def __init__(self, *args, virtual_per_free: float = 1.0, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._per_free = virtual_per_free
        self._marks_rng = np.random.default_rng(seed)
        self._marks: dict[str, float] = {}

    def _active(self, vehicles, current_time):
        self._marks = {
            q.id: self._marks[q.id] if q.id in self._marks else self._marks_rng.random() for q in self.virtual
        }
        share = min(1.0, self._per_free * self._supply(vehicles, current_time) / max(len(self.virtual), 1))
        return [q for q in self.virtual if self._marks[q.id] < share]

    def _supply(self, vehicles, current_time) -> int:
        """Vehicles that can take a virtual rider: the free ones."""
        return sum(
            1 for v in vehicles
            if not v.passengers_onboard and all(s.is_rebalancing for s in v.remaining_stops_to_serve)
        )


class Bidirectional(SupplyMatched):
    """
    SupplyMatched, and every active virtual rider is also offered to its
    `n_virtual` best free vehicles (not only every vehicle its best riders),
    so that free vehicles clustered in a quiet zone are not all left with the
    same few nearby riders.
    """

    def _candidates(self, gain):
        keep = super()._candidates(gain)
        k = min(self._n_virtual, gain.shape[0])
        cols = np.arange(gain.shape[1])[None, :]
        keep[np.argpartition(-gain, k - 1, axis=0)[:k, :], cols] = True
        return keep & (gain > 0)


class SupplyMatchedSampledSolver(SupplyMatched, SampledRequestsAssignmentSolver):
    """`sampled` with about `virtual_per_free` virtual riders per free vehicle."""


class SupplyMatchedChainSolver(SupplyMatched, SampledChainAssignmentSolver):
    """`sampled-chain` with about `virtual_per_free` virtual riders per free vehicle."""


class FullMatching(SupplyMatched):
    """
    SupplyMatched, and the columns always contain a complete min-cost matching
    of free vehicles to active virtual riders (Hungarian method on the full
    gain matrix) besides each vehicle's `n_virtual` best riders. With top-k
    columns alone, vehicles clustered at the same few nodes share the same
    nearest riders, so the ILP can move only a fraction of them.
    """

    def _candidates(self, gain):
        keep = super()._candidates(gain)
        rows, cols = linear_sum_assignment(np.where(gain > 0, -gain, 0.0))
        keep[rows, cols] = True
        return keep & (gain > 0)


class BidirectionalSampledSolver(Bidirectional, SampledRequestsAssignmentSolver):
    """`sampled-matched` with columns chosen from both sides."""


class SoonFreeSupply:
    """Mixin for matched chain solvers: supply also counts busy vehicles free within the horizon."""

    def _supply(self, vehicles, current_time):
        soon = sum(
            1 for v in vehicles
            if v.passengers_onboard
            and self._free_at(v, v.remaining_stops_to_serve)[1] <= current_time + self._horizon
        )
        return super()._supply(vehicles, current_time) + soon


class BidirectionalChainSolver(SoonFreeSupply, Bidirectional, SampledChainAssignmentSolver):
    """`sampled-chain-matched` with columns from both sides; supply includes vehicles free within the horizon."""


class FullMatchingSampledSolver(FullMatching, SampledRequestsAssignmentSolver):
    """`sampled-matched` whose columns contain a full matching of free vehicles to active riders."""


class FullMatchingChainSolver(SoonFreeSupply, FullMatching, SampledChainAssignmentSolver):
    """`sampled-chain-matched` whose columns contain a full matching; supply includes vehicles free within the horizon."""
