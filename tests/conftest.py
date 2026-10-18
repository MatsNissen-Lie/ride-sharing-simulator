"""Shared fixtures: a small random street network behind the real NetworkXService."""

import itertools
import random

import networkx as nx
import numpy as np
import pytest

from config.config import DELTA, OMEGA
from src.components.network_service import NetworkXService
from src.interfaces import Passenger, Request, VehicleAvailability


class InMemoryPathCache:
    """Stands in for DuckDBPathCache with exact all-pairs travel times."""

    read_only = True

    def __init__(self, graph: nx.MultiDiGraph):
        n = max(graph.nodes) + 1
        self.matrix = np.full((n, n), np.inf)
        for o, times in nx.all_pairs_dijkstra_path_length(graph, weight="travel_time"):
            for d, t in times.items():
                self.matrix[o, d] = t

    def get_travel_time_matrix(self, origins, destinations):
        return self.matrix[np.ix_(origins, destinations)]

    def get_simple(self, o, d):
        return float(self.matrix[o, d]), 0.0

    def connect(self, read_only=None):
        pass

    def close(self):
        pass


def make_street_graph(seed: int, side: int = 6) -> nx.MultiDiGraph:
    """Strongly connected grid with random travel times (8-30 s per edge)."""
    rng = random.Random(seed)
    grid = nx.grid_2d_graph(side, side)
    index = {node: k for k, node in enumerate(grid.nodes)}
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(index.values())
    for u, v in grid.edges:
        for a, b in ((u, v), (v, u)):
            t = rng.uniform(8, 30)
            graph.add_edge(index[a], index[b], travel_time=t, length=t * 10)
    return graph


@pytest.fixture
def network() -> NetworkXService:
    graph = make_street_graph(seed=7)
    return NetworkXService(graph, db=InMemoryPathCache(graph))  # type: ignore[arg-type]


def make_request(network, rid, origin, dest, t_request, latest_pickup=None) -> Request:
    direct = network.get_travel_time(origin, dest)
    return Request(
        id=rid,
        origin_node=origin,
        destination_node=dest,
        request_time=t_request,
        latest_pickup_time=latest_pickup if latest_pickup is not None else t_request + OMEGA,
        latest_dropoff_time=t_request + direct + DELTA,
    )


def random_requests(network, rng, n, t_now, prefix="r") -> list[Request]:
    nodes = list(range(network.N))
    out = []
    for k in range(n):
        o, d = rng.sample(nodes, 2)
        out.append(make_request(network, f"{prefix}{k}", o, d, t_now - rng.uniform(0, 60)))
    return out


def random_vehicle(network, rng, vid, t_now, n_onboard, capacity=4) -> VehicleAvailability:
    """A vehicle with `n_onboard` passengers whose drop-off deadlines are reachable."""
    nodes = list(range(network.N))
    node = rng.choice(nodes)
    passengers = set()
    for k in range(n_onboard):
        for _ in range(100):
            o, d = rng.sample(nodes, 2)
            req = make_request(network, f"{vid}p{k}", o, d, t_now - rng.uniform(60, 240))
            if t_now + network.get_travel_time(node, d) <= req.latest_dropoff_time - 60:
                break
        passengers.add(Passenger(original_request=req, pickup_time=t_now - 30))
    return VehicleAvailability(
        vehicle_id=vid,
        capacity=capacity,
        available_at_node=node,
        available_from_time=t_now + rng.choice([0.0, 0.0, rng.uniform(0, 40)]),
        passengers_onboard=frozenset(passengers),
        remaining_stops_to_serve=(),
    )


def brute_force_route(requests, vehicle, matrix):
    """Exhaustive travel(v, T): (feasible, marginal delay, stop order)."""

    def best(new_reqs):
        stops = [("drop", p.original_request) for p in vehicle.passengers_onboard]
        for r in new_reqs:
            stops += [("pick", r), ("drop", r)]
        best_cost, best_order = np.inf, None
        for order in itertools.permutations(range(len(stops))):
            seen, load = set(), len(vehicle.passengers_onboard)
            node, t, cost, ok = vehicle.available_at_node, vehicle.available_from_time, 0.0, True
            for i in order:
                kind, r = stops[i]
                if kind == "drop" and r in new_reqs and ("pick", r.id) not in seen:
                    ok = False
                    break
                stop_node = r.origin_node if kind == "pick" else r.destination_node
                t += matrix[node, stop_node]
                node = stop_node
                load += 1 if kind == "pick" else -1
                deadline = r.latest_pickup_time if kind == "pick" else r.latest_dropoff_time
                if load > vehicle.capacity or t > deadline:
                    ok = False
                    break
                if kind == "drop":
                    cost += t - (r.request_time + matrix[r.origin_node, r.destination_node])
                seen.add((kind, r.id))
            if ok and cost < best_cost:
                best_cost, best_order = cost, [stops[i] for i in order]
        return best_cost, best_order

    baseline, _ = best([])
    cost, order = best(list(requests))
    if not np.isfinite(cost):
        return False, None, None
    return True, cost - (baseline if np.isfinite(baseline) else 0.0), order
