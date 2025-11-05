"""Minimal end-to-end rider -> vehicle matching.

This module demonstrates how to load the cached Manhattan street network,
read the enriched rider table, sample vehicle positions, and pair the first
``n`` riders with their nearest reachable vehicles. For each assignment we
return two route segments: the dispatch leg (vehicle -> pickup) and the
en-route leg (pickup -> drop-off). See :mod:`src.state.utils` for generic
plotting helpers that visualise these segments.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Optional

import math

import networkx as nx
import polars as pl

from config import BASE_DIR, get_logger
from src.state_representation.manhatten_fast import ensure_travel_times, load_graph
from src.state_representation.vehicle_sampling import sample_vehicle_positions
from src.state.utils import plot_assignment


logger = get_logger(__name__)


RIDER_DATA_PATH = (
    BASE_DIR / "data" / "riders" / "fhv_tripdata_2025-01_enriched.parquet"
)
"""Enriched rider dataset with pre-computed origin/destination nodes."""

RIDER_COLUMNS: list[str] = [
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_zone",
    "dropoff_zone",
    "orig_node",
    "dest_node",
]


@dataclass(slots=True)
class RouteSegment:
    """Lightweight representation of a graph route."""

    nodes: list[int]
    distance_m: float
    travel_time_s: float


@dataclass(slots=True)
class RiderVehicleMatch:
    """Match between a rider request and the selected vehicle."""

    rider_index: int
    pickup_node: int
    dropoff_node: int
    vehicle_id: int
    vehicle_node: int
    dispatch: RouteSegment
    enroute: RouteSegment


def load_minimal_graph() -> nx.MultiDiGraph:
    """Return the cached Manhattan graph with travel-time weights."""

    logger.info("Loading Manhattan drive network (cached if available)...")
    graph = ensure_travel_times(load_graph())
    logger.info(
        "Graph ready: %d nodes / %d edges", graph.number_of_nodes(), graph.number_of_edges()
    )
    return graph


def load_riders(n: int) -> pl.DataFrame:
    """Load the first *n* riders with valid origin/destination nodes."""

    if n <= 0:
        raise ValueError("n must be positive")

    logger.info("Loading %d rider(s) from %s", n, RIDER_DATA_PATH)
    riders = pl.read_parquet(RIDER_DATA_PATH, columns=RIDER_COLUMNS, n_rows=n)
    riders = riders.drop_nulls(subset=["orig_node", "dest_node"]).with_columns(
        pl.col("orig_node").cast(pl.Int64),
        pl.col("dest_node").cast(pl.Int64),
    )
    logger.info("Loaded %d rider(s) with valid graph nodes", riders.height)
    return riders


def _select_edge_data(
    edges: Mapping[Hashable, Mapping[str, Any]],
    weight_key: str,
) -> Mapping[str, Any]:
    """Return the edge data entry that minimises *weight_key*."""

    best_weight = math.inf
    best_data: Optional[Mapping[str, Any]] = None

    for data in edges.values():
        weight = data.get(weight_key)
        if weight is None:
            continue
        weight_value = float(weight)
        if weight_value < best_weight:
            best_weight = weight_value
            best_data = data

    if best_data is None:
        raise ValueError(f"No edge carried the '{weight_key}' attribute")

    return best_data


def _summarise_route(graph: nx.MultiDiGraph, path: list[int]) -> RouteSegment:
    """Compute aggregate metrics for *path* and return a :class:`RouteSegment`."""

    if not path:
        raise ValueError("Path cannot be empty")

    total_length = 0.0
    total_travel_time = 0.0

    for u, v in pairwise(path):
        edge_bundle = graph.get_edge_data(u, v)
        if not edge_bundle:
            raise ValueError(f"Edge data missing for hop {u} -> {v}")
        chosen = _select_edge_data(edge_bundle, "travel_time")
        total_length += float(chosen.get("length", 0.0) or 0.0)
        total_travel_time += float(chosen.get("travel_time", 0.0) or 0.0)

    return RouteSegment(nodes=path, distance_m=total_length, travel_time_s=total_travel_time)


def _shortest_route(
    graph: nx.MultiDiGraph,
    source: int,
    target: int,
) -> RouteSegment:
    """Compute the fastest route between *source* and *target*."""

    path = nx.shortest_path(graph, source, target, weight="travel_time")
    return _summarise_route(graph, path)


def _choose_vehicle(
    graph: nx.MultiDiGraph,
    pickup_node: int,
    vehicles: pl.DataFrame,
) -> tuple[int, int, RouteSegment]:
    """Return the vehicle assignment minimising dispatch travel-time."""

    best: Optional[tuple[float, int, int, RouteSegment]] = None

    for vehicle in vehicles.iter_rows(named=True):
        vehicle_id = int(vehicle["vehicle_id"])
        vehicle_node = int(vehicle["node_id"])
        try:
            dispatch_route = _shortest_route(graph, vehicle_node, pickup_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        travel_time = dispatch_route.travel_time_s
        if best is None or travel_time < best[0]:
            best = (travel_time, vehicle_id, vehicle_node, dispatch_route)

    if best is None:
        msg = "No vehicle with a reachable dispatch route"
        raise ValueError(msg)

    _, vehicle_id, vehicle_node, dispatch_route = best
    return vehicle_id, vehicle_node, dispatch_route


def match_riders_to_nearest_vehicles(
    rider_count: int = 1,
    *,
    vehicle_count: int = 50,
    seed: Optional[int] = None,
    graph: Optional[nx.MultiDiGraph] = None,
) -> list[RiderVehicleMatch]:
    """Assign the first *rider_count* riders to sampled vehicles.

    Parameters
    ----------
    rider_count:
        Number of riders to match starting from the top of the enriched table.
    vehicle_count:
        Number of candidate vehicle positions sampled from the road network.
    seed:
        Optional RNG seed forwarded to :func:`sample_vehicle_positions`.
    graph:
        Pre-loaded street graph; when omitted it is loaded on demand.
    """

    if vehicle_count <= 0:
        raise ValueError("vehicle_count must be positive")

    graph = graph or load_minimal_graph()
    riders = load_riders(rider_count)
    vehicles = sample_vehicle_positions(graph, vehicle_count, seed=seed)

    assignments: list[RiderVehicleMatch] = []

    for idx, rider in enumerate(riders.iter_rows(named=True)):
        pickup_node = int(rider["orig_node"])
        dropoff_node = int(rider["dest_node"])

        vehicle_id, vehicle_node, dispatch = _choose_vehicle(graph, pickup_node, vehicles)
        try:
            enroute = _shortest_route(graph, pickup_node, dropoff_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
            msg = f"No path from pickup node {pickup_node} to dropoff node {dropoff_node}"
            raise ValueError(msg) from error

        assignments.append(
            RiderVehicleMatch(
                rider_index=idx,
                pickup_node=pickup_node,
                dropoff_node=dropoff_node,
                vehicle_id=vehicle_id,
                vehicle_node=vehicle_node,
                dispatch=dispatch,
                enroute=enroute,
            )
        )

    return assignments


def describe_match(match: RiderVehicleMatch) -> str:
    """Return a concise textual summary of a rider -> vehicle assignment."""

    dispatch = match.dispatch
    enroute = match.enroute
    dispatch_summary = (
        f"dispatch {dispatch.travel_time_s:.0f}s over {dispatch.distance_m/1000:.2f} km"
    )
    enroute_summary = (
        f"en-route {enroute.travel_time_s:.0f}s over {enroute.distance_m/1000:.2f} km"
    )
    return (
        f"rider #{match.rider_index}: vehicle {match.vehicle_id} "
        f"(node {match.vehicle_node}) -> pickup {match.pickup_node} -> dropoff {match.dropoff_node}; "
        f"{dispatch_summary}, {enroute_summary}"
    )


__all__ = [
    "RouteSegment",
    "RiderVehicleMatch",
    "describe_match",
    "load_minimal_graph",
    "load_riders",
    "match_riders_to_nearest_vehicles",
]


def main() -> None:
    """Quick smoke test when executed as a script."""

    graph = load_minimal_graph()
    matches = match_riders_to_nearest_vehicles(graph=graph)
    for match in matches:
        print(describe_match(match))
        plot_assignment(graph, match)


if __name__ == "__main__":
    main()
