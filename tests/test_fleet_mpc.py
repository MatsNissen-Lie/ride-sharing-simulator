"""Zone-level fleet MPC: LP duals as vehicle values."""

import random
import time
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from src.components.assignment_solver import HighsAssignmentSolver
from src.components.fleet_mpc import FleetMpcValue, fleet_lp_values
from src.components.mpc_rebalance_solver import MpcRebalancingSolver
from src.components.zone_trips import ZoneTrips
from src.components.joint_assignment_solver import JointAssignmentSolver
from src.components.objectives import DelayObjective
from src.interfaces import IDemandModel, Stop, VehicleAvailability
from tests.conftest import random_requests, random_vehicle
from tests.test_joint_assignment import instance

INF = np.inf


def lp(supply, demand, trip, move=None, move_cost=None, reward=1.0, pooling=1.0, shares=None):
    supply, demand = np.atleast_2d(supply).astype(float), np.atleast_2d(demand).astype(float)
    n_zones, n_steps = supply.shape
    if shares is None:
        shares = np.eye(n_zones)
    return fleet_lp_values(
        supply,
        demand,
        np.repeat(np.asarray(shares, dtype=float)[None], n_steps, axis=0),
        np.asarray(trip, dtype=float).reshape(n_zones, n_zones),
        np.full((n_zones, n_zones), INF) if move is None else np.asarray(move, dtype=float),
        np.zeros((n_zones, n_zones)) if move_cost is None else np.asarray(move_cost, dtype=float),
        reward,
        pooling,
    )[0]


def test_one_zone_value_counts_trips_left_in_the_horizon():
    """Trips take 2 steps: a vehicle free at step 0 or 1 serves twice, at 2 or 3 once."""
    values = lp(supply=[[1, 0, 0, 0]], demand=[[10, 10, 10, 10]], trip=[2])
    assert values[0] == pytest.approx([2, 2, 1, 1], abs=1e-6)


def test_fractional_trip_time_makes_values_linear_in_time():
    """A trip of 1.5 steps returns half a vehicle after 1 step and half after 2."""
    values = lp(supply=[[1, 0, 0]], demand=[[10, 10, 10]], trip=[1.5])
    # V2 = 1; V1 = 1 + 0.5 V2 = 1.5; V0 = 1 + 0.5 V1 + 0.5 V2 = 2.25
    assert values[0] == pytest.approx([2.25, 1.5, 1.0], abs=1e-6)


def test_surplus_vehicles_are_worth_nothing():
    assert lp(supply=[[5]], demand=[[1]], trip=[1])[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_zero_demand_gives_zero_values():
    values = lp(supply=[[3, 1], [0, 2]], demand=np.zeros((2, 2)), trip=np.ones((2, 2)),
                move=[[INF, 1], [1, INF]], move_cost=[[0, 5], [5, 0]])
    assert values == pytest.approx(np.zeros((2, 2)), abs=1e-9)


def test_pooling_lets_one_vehicle_serve_several_riders():
    assert lp(supply=[[1]], demand=[[3]], trip=[1], pooling=2.0)[0, 0] == pytest.approx(2.0)


def test_a_vehicle_far_from_demand_is_worth_the_move_less_its_cost():
    """Demand only in B; moving A -> B takes one step and costs 0.1."""
    values = lp(
        supply=[[1, 0, 0], [1, 1, 1]],
        demand=[[0, 0, 0], [5, 5, 5]],
        trip=[[10, 10], [10, 10]],  # every trip ends after the horizon
        move=[[INF, 1], [1, INF]],
        move_cost=[[0, 0.1], [0.1, 0]],
    )
    assert values[1] == pytest.approx([1, 1, 1], abs=1e-6)
    assert values[0] == pytest.approx([0.9, 0.9, 0.0], abs=1e-6)


def test_planned_moves_of_the_first_step():
    """Two vehicles in A, riders only in B: both move at once (B trips take one step)."""
    _, moves = fleet_lp_values(
        np.array([[2.0, 0, 0], [0, 0, 0]]), np.array([[0.0, 0, 0], [5, 5, 5]]),
        np.repeat(np.eye(2)[None], 3, axis=0), np.ones((2, 2)),
        np.array([[INF, 1], [1, INF]]), np.zeros((2, 2)), 1.0,
    )
    assert moves == pytest.approx(np.array([[0, 2], [0, 0]]), abs=1e-3)


@pytest.mark.parametrize("vehicles, value_now", [(1, 2.0), (3, 1.0)])
def test_value_is_that_of_the_marginal_trip_class(vehicles, value_now):
    """
    Riders from A go back to A (worth 1 + V_A1 = 2) or to B, where nothing is
    left to do (worth 1), 2.5 trips each in step 0. One vehicle takes an A->A
    trip; with three, A->A is full and the marginal vehicle goes to B.
    """
    values = lp(supply=[[vehicles, 0], [5, 5]], demand=[[5, 5], [0, 0]], trip=np.ones((2, 2)),
                shares=[[0.5, 0.5], [0.0, 1.0]])
    assert values[0] == pytest.approx([value_now, 1.0], abs=1e-6)
    assert values[1] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_runtime_on_a_city_sized_instance():
    """65 zones x 12 steps of 5 min, all 4225 pairs with riders, moves up to 15 min."""
    rng = np.random.default_rng(0)
    n_zones, n_steps = 65, 12
    xy = rng.uniform(0, 1, (n_zones, 2))
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    trip_steps = 1 + 6 * dist
    move_steps = np.where((dist > 0) & (dist < 0.4), np.maximum(6 * dist, 1), INF)
    shares = rng.dirichlet(np.ones(n_zones), size=(n_steps, n_zones))
    supply = rng.poisson(5, (n_zones, n_steps)).astype(float)
    demand = rng.poisson(8, (n_zones, n_steps)).astype(float)
    start = time.perf_counter()
    values, _ = fleet_lp_values(supply, demand, shares, trip_steps, move_steps, 300 * move_steps,
                                1e5, 1.2)
    assert time.perf_counter() - start < 5.0  # solved once a minute; batches are 10 s
    assert values.shape == (n_zones, n_steps) and (values >= -1e-6).all()


# --- FleetMpcValue on the synthetic street grid -------------------------------------------------

T_NOW = 10_000.0


class Zones:
    """Left half of the 6x6 grid is zone 1, right half zone 2."""

    zone_ids = np.array([1, 2])
    center_node = {1: 14, 2: 21}

    def zones_of(self, nodes):
        return np.array([1 if n % 6 < 3 else 2 for n in nodes], dtype=np.int64)


class FixedDemand(IDemandModel):
    """Expected requests per zone and second."""

    def __init__(self, per_second):
        self.per_second = per_second

    def expected_requests(self, t_start, t_end):
        return {z: r * (t_end - t_start) for z, r in self.per_second.items()}


def trips_to_same_zone(trip_time):
    shares = {(kind, h): np.eye(2) for kind in ("weekday", "weekend") for h in range(24)}
    return ZoneTrips(shares, np.full((2, 2), trip_time), datetime(2025, 1, 16))


def mpc(network, demand, trip_time=600.0, **kwargs):
    kwargs = {"step": 60.0, "n_steps": 10, "smoothing": 1.0, **kwargs}
    return FleetMpcValue(network, Zones(), FixedDemand(demand), trips_to_same_zone(trip_time),
                         DelayObjective(), 1e5, **kwargs)


def idle(vid, node):
    return VehicleAvailability(vid, 4, node, T_NOW, frozenset(), ())


def test_busy_vehicles_arrive_where_and_when_their_plan_ends(network):
    """Node n sits in grid column n % 6: columns 0-2 are zone 1, 3-5 zone 2."""
    value = mpc(network, {})
    busy = VehicleAvailability("b", 4, 5, T_NOW + 20, frozenset(), (Stop.rebalancing(12, "x"),))
    k = (20 + network.get_l1_cache()[5, 12]) / 60
    expected = np.zeros((2, 10))
    expected[1, 0] = 1  # idle at node 3: zone 2, now
    expected[0, int(k)] = 1 - (k - int(k))  # busy until it reaches node 12: zone 1, after k steps
    expected[0, int(k) + 1] = k - int(k)
    assert value.supply_grid({idle("a", 3), busy}, T_NOW) == pytest.approx(expected)


def test_waiting_riders_add_to_the_first_step(network):
    value = mpc(network, {1: 0.01})
    rng = random.Random(0)
    riders = random_requests(network, rng, 3, T_NOW)
    grid = value.demand_grid(set(riders), T_NOW)
    left = sum(Zones().zones_of([r.origin_node])[0] == 1 for r in riders)
    assert grid[0] == pytest.approx([0.6 + left] + [0.6] * 9)
    assert grid[1] == pytest.approx([3 - left] + [0.0] * 9)


def test_values_favour_the_zone_with_demand_and_fall_with_time(network):
    """Plenty of riders on the right only; a trip takes 3 steps of 1 min."""
    value = mpc(network, {2: 0.1}, trip_time=180.0)
    value.prepare({idle("a", 0)}, set(), T_NOW)
    left, right = 0, 5
    v = value.values(np.array([left, right, right, right]),
                     np.array([T_NOW, T_NOW, T_NOW + 240, T_NOW + 3600]), T_NOW)
    # right now: trips at steps 0, 3, 6, 9; after 4 min: 4, 7; left: move 1 step, then 1, 4, 7
    assert v[1] == pytest.approx(4e5, rel=1e-3) and v[2] == pytest.approx(2e5, rel=1e-3)
    assert 2e5 < v[0] < 3e5
    assert v[3] == pytest.approx(0.0)


def test_zero_forecast_gives_zero_values(network):
    rng = random.Random(1)
    vehicles = {random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 1])) for k in range(4)}
    value = mpc(network, {})
    value.prepare(vehicles, set(), T_NOW)
    assert value.values(np.arange(36), np.full(36, T_NOW), T_NOW) == pytest.approx(np.zeros(36))


def test_zero_weight_is_the_papers_ilp(network):
    rng = random.Random(2)
    vehicles = {random_vehicle(network, rng, f"v{k}", T_NOW, rng.choice([0, 1])) for k in range(4)}
    requests = set(random_requests(network, rng, 6, T_NOW))
    rtv = instance(network, vehicles, requests)
    plain = HighsAssignmentSolver().solve(rtv, requests, vehicles, 30, T_NOW)
    value = mpc(network, {1: 0.1, 2: 0.1}, weight=0.0)
    combined = JointAssignmentSolver(DelayObjective(), network, Zones(), value).solve(
        rtv, requests, vehicles, 30, T_NOW
    )
    assert combined.total_cost == pytest.approx(plain.total_cost)
    assert combined.rebalancing == {}


def test_values_are_reused_between_solves_and_smoothed(network):
    value = mpc(network, {2: 0.01}, smoothing=0.5, resolve_every=60.0)
    value.prepare({idle("a", 5)}, set(), T_NOW)
    first = value.values(np.array([5]), np.array([T_NOW + 60]), T_NOW)
    value.prepare(set(), set(), T_NOW + 10)  # within resolve_every: no new solve
    assert value.values(np.array([5]), np.array([T_NOW + 60]), T_NOW + 10) == pytest.approx(first)
    value.prepare(set(), set(), T_NOW + 60)  # no vehicles at all: LP values are higher
    fresh = mpc(network, {2: 0.01})
    fresh.prepare(set(), set(), T_NOW + 60)
    lp_only = fresh.values(np.array([5]), np.array([T_NOW + 60]), T_NOW + 60)
    smoothed = value.values(np.array([5]), np.array([T_NOW + 60]), T_NOW + 60)
    assert smoothed == pytest.approx(0.5 * lp_only + 0.5 * first)


def test_zone_trips_fit_shares_and_trip_times():
    class NodeZones:
        zone_ids = np.array([1, 2])
        center_node = {1: 0, 2: 1}

        def zones_of(self, nodes):
            return np.asarray(nodes) + 1

    data = pl.DataFrame(
        {
            "pickup_datetime": [datetime(2025, 1, 15, 18, 5), datetime(2025, 1, 15, 18, 40),
                                datetime(2025, 1, 15, 18, 50), datetime(2025, 1, 16, 18, 5)],
            "orig_node": [0, 0, 0, 0],
            "dest_node": [0, 1, 1, 0],  # the last trip is on the excluded day
        }
    )
    times = np.array([[100.0, 400.0], [500.0, 50.0]])
    trips = ZoneTrips.fit(data, NodeZones(), times, datetime(2025, 1, 16), exclude={date(2025, 1, 16)},
                          pickup_time=60.0)
    assert trips.shares_at(18 * 3600 + 60)[0] == pytest.approx([1 / 3, 2 / 3])
    assert trips.trip_time == pytest.approx(np.array([[160.0, 460.0], [560.0, 110.0]]))


def test_rebalancer_carries_out_the_first_moves_after_a_solve(network):
    value = mpc(network, {2: 0.1}, trip_time=120.0)  # moving now beats moving later
    vehicles = [idle("a", 0), idle("b", 1), idle("c", 6)]  # all in zone 1
    rebalancer = MpcRebalancingSolver(network, Zones(), value)
    value.prepare(set(vehicles), set(), T_NOW)
    assert value.moves[0, 1] == pytest.approx(3, abs=1e-2)
    routes = rebalancer.solve(vehicles, [], T_NOW)
    assert set(routes) == {"a", "b", "c"}
    assert all(r.stops[0].node == 21 and r.stops[0].is_rebalancing for r in routes.values())
    assert rebalancer.solve(vehicles, [], T_NOW + 10) == {}  # no new solve, nothing to do


def test_provider_reads_trips_with_destinations(tmp_path, monkeypatch):
    """The hook ZoneTrips needs: request_trips() concatenates the rider files."""
    import networkx as nx

    from src.components import simulation_data_provider as sdp

    monkeypatch.setattr(sdp, "load_graph", lambda: nx.MultiDiGraph())
    for k in range(2):
        pl.DataFrame({"pickup_datetime": [datetime(2025, 1, 15, 8, k)], "orig_node": [k],
                      "dest_node": [k + 10], "other": [0]}).write_parquet(tmp_path / f"trips_{k}.parquet")
    trips = sdp.SimulationDataProvider(rider_dir=tmp_path, rider_pattern="trips_*.parquet").request_trips()
    assert trips.columns == ["pickup_datetime", "orig_node", "dest_node"]
    assert trips["dest_node"].to_list() == [10, 11]
