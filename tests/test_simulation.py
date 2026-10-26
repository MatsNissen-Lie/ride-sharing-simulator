"""End to end on the synthetic network: every committed plan must hold up in replay."""

import random
from datetime import datetime

import pytest

from src.components.assignment_solver import GurobiAssignmentSolver, HighsAssignmentSolver
from src.components.batch_controller import BatchController
from src.components.graph_builder.rtv_trip_centric import TripCentricRTVGraphBuilder
from src.components.graph_builder.rtv_vehicle_centric import (
    VehicleCentricRTVGraphBuilder,
)
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.components.rebalance_solver import ReactiveRebalancingSolver
from src.interfaces import Vehicle
from src.interfaces.data_models import MetaRequest
from src.components.trip_solver import AStarTripSolver
from src.simulator import Simulator
from src.utils.sim_clock import SimulationClock


class Provider:
    def get_simulation_clock(self):
        return SimulationClock(start_time=datetime(2025, 1, 1))


class CheckedController(BatchController):
    """Fails if a committed route misses a pickup or drop-off deadline."""

    def commit_assignment(self, t_now, result):
        for a in result.assignments:
            v = self._id_to_vehicle[a.vehicle_id]
            reqs = {p.original_request.id: p.original_request for p in v.passengers_onboard}
            node, t = v.last_visited_node, v.last_visited_time
            for s in a.route.stops:
                r = reqs.get(s.request_id) or self._request_index[s.request_id]
                t += self._network.get_travel_time(node, s.node)
                node = s.node
                deadline = r.latest_pickup_time if s.is_pickup else r.latest_dropoff_time
                assert t <= deadline + 1e-6, (t_now, a.vehicle_id, s)
        super().commit_assignment(t_now, result)


def run(network, rtv_builder, solver, seed=0, n_vehicles=5, n_requests=150, horizon=1800):
    rng = random.Random(seed)
    fleet = [
        Vehicle(f"v{k}", 4, set(), rng.randrange(network.N), 0.0) for k in range(n_vehicles)
    ]
    times = sorted(rng.uniform(0, horizon) for _ in range(n_requests))
    metas = [
        MetaRequest(f"r{k}", *rng.sample(range(network.N), 2), t) for k, t in enumerate(times)
    ]
    controller = CheckedController(network, fleet)
    trip_solver = AStarTripSolver(network.get_l1_cache())
    sim = Simulator(
        provider=Provider(),  # type: ignore[arg-type]
        controller=controller,
        rv_builder=RVGraphBuilder(network, trip_solver),
        rtv_builder=rtv_builder(trip_solver),
        assignment_solver=solver(),
        rebalancing_solver=ReactiveRebalancingSolver(network),
        network_service=network,
        db_cache=None,
        read_only=True,
    )
    batches = list(
        sim.run(iter(metas), min_start_time=0.0, batch_duration=10, warmup_duration=0,
                end_time=horizon + 900)
    )
    return batches, controller


def test_simulation_is_consistent(network):
    batches, controller = run(network, TripCentricRTVGraphBuilder, HighsAssignmentSolver)
    delivered = sum(b.requests_delivered for b in batches)
    expired = sum(b.requests_expired for b in batches)
    assert delivered > 50
    assert delivered + expired + len(controller._request_index) <= 150
    assert max(b.assigned_trips_size_2 for b in batches) > 0  # rides were shared


class CrossCheckedBuilder(TripCentricRTVGraphBuilder):
    """Trip-centric graph, asserted identical to the vehicle-centric one."""

    def build_graph(self, rv_graph, vehicles, requests):
        trip = super().build_graph(rv_graph, vehicles, requests)
        veh = VehicleCentricRTVGraphBuilder(self._trip_solver).build_graph(
            rv_graph, vehicles, requests
        )
        assert trip.trip_vehicle_details.keys() == veh.trip_vehicle_details.keys()
        for key, route in trip.trip_vehicle_details.items():
            assert route.cost == pytest.approx(veh.trip_vehicle_details[key].cost)
        return trip


class CrossCheckedSolver(HighsAssignmentSolver):
    """HiGHS solution, asserted to reach the same optimum as Gurobi."""

    def solve(self, rtv_graph, requests, vehicles, solver_timeout, current_time):
        highs = super().solve(rtv_graph, requests, vehicles, solver_timeout, current_time)
        gurobi = GurobiAssignmentSolver().solve(
            rtv_graph, requests, vehicles, solver_timeout, current_time
        )
        assert highs.total_cost == pytest.approx(gurobi.total_cost, abs=1e-6)
        return highs


@pytest.mark.parametrize("seed", range(2))
def test_builders_and_solvers_agree_every_batch(network, seed):
    """Whole trajectories can differ through ties, so compare on identical inputs."""
    batches, _ = run(network, CrossCheckedBuilder, CrossCheckedSolver, seed=seed)
    assert sum(b.requests_delivered for b in batches) > 50
