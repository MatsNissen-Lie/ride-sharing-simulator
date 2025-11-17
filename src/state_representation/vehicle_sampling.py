"""Utilities for sampling vehicle start locations on a street graph."""

from dataclasses import dataclass
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import osmnx as ox
import polars as pl
from pyproj import Transformer
from shapely.geometry import LineString

from config import VEHICLE_SAMPLE_PATH

__all__ = ["VehicleSample", "sample_vehicle_positions"]


@dataclass(frozen=True)
class VehicleSample:
    """Represents a sampled vehicle location."""

    node_id: int
    x: float
    y: float
    lon: Optional[float]
    lat: Optional[float]
    edge_u: int
    edge_v: int
    edge_key: int


def _ensure_multidigraph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if not isinstance(graph, nx.MultiDiGraph):
        msg = "Expected a networkx.MultiDiGraph"
        raise TypeError(msg)
    return graph


def _build_transformer(graph: nx.MultiDiGraph) -> Optional[Transformer]:
    crs = graph.graph.get("crs")
    if not crs:
        return None
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def _length_weighted_edges(graph: nx.MultiDiGraph) -> pl.DataFrame:
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)[["geometry", "length"]]
    edges = edges_gdf.reset_index()
    valid = edges["length"].to_numpy(dtype=float) > 0.0
    edges = edges.loc[valid]
    if edges.empty:
        raise ValueError("Graph does not contain edges with positive length")

    data = {
        "u": edges["u"].to_numpy(dtype=int, copy=False),
        "v": edges["v"].to_numpy(dtype=int, copy=False),
        "key": edges["key"].to_numpy(dtype=int, copy=False),
        "length": edges["length"].to_numpy(dtype=float, copy=False),
        "geometry": edges["geometry"].to_numpy(copy=False),
    }
    return pl.DataFrame(data)


def _interpolate_points(
    edges: pl.DataFrame,
    indices: np.ndarray,
    fractions: np.ndarray,
) -> Iterable[VehicleSample]:
    for idx, frac in zip(indices, fractions):
        row = edges.row(int(idx), named=True)
        line = row["geometry"]
        if not isinstance(line, LineString):
            line = LineString(line)
        point = line.interpolate(float(frac), normalized=True)
        u = int(row["u"])
        v = int(row["v"])
        key = int(row["key"])
        yield VehicleSample(
            node_id=-1,
            x=float(point.x),
            y=float(point.y),
            lon=None,
            lat=None,
            edge_u=u,
            edge_v=v,
            edge_key=key,
        )


def _snap_to_nodes(
    graph: nx.MultiDiGraph,
    samples: list[VehicleSample],
) -> None:
    nodes = graph.nodes
    for idx, sample in enumerate(samples):
        u_data = nodes[sample.edge_u]
        v_data = nodes[sample.edge_v]
        ux, uy = float(u_data["x"]), float(u_data["y"])
        vx, vy = float(v_data["x"]), float(v_data["y"])
        dx_u = (ux - sample.x) ** 2 + (uy - sample.y) ** 2
        dx_v = (vx - sample.x) ** 2 + (vy - sample.y) ** 2
        node_id = sample.edge_u if dx_u <= dx_v else sample.edge_v
        samples[idx] = VehicleSample(
            node_id=node_id,
            x=sample.x,
            y=sample.y,
            lon=sample.lon,
            lat=sample.lat,
            edge_u=sample.edge_u,
            edge_v=sample.edge_v,
            edge_key=sample.edge_key,
        )


def _add_lon_lat(
    transformer: Optional[Transformer],
    samples: list[VehicleSample],
) -> None:
    if transformer is None:
        return
    xs = [s.x for s in samples]
    ys = [s.y for s in samples]
    lon, lat = transformer.transform(xs, ys)
    for idx, sample in enumerate(samples):
        samples[idx] = VehicleSample(
            node_id=sample.node_id,
            x=sample.x,
            y=sample.y,
            lon=float(lon[idx]),
            lat=float(lat[idx]),
            edge_u=sample.edge_u,
            edge_v=sample.edge_v,
            edge_key=sample.edge_key,
        )


def sample_vehicle_positions(
    graph: nx.MultiDiGraph,
    count: int,
    *,
    seed: Optional[int] = None,
    include_lonlat: bool = True,
) -> pl.DataFrame:
    """Return uniformly sampled vehicle locations on *graph*.

    Parameters
    ----------
    graph:
        Projected street network (``MultiDiGraph``) with ``x``/``y`` node attributes.
    count:
        Number of vehicle positions to generate.
    seed:
        Optional RNG seed for reproducible sampling.
    include_lonlat:
        If ``True`` (default) and the graph carries CRS metadata, additional
        ``lon``/``lat`` columns are computed.

    Returns
    -------
    polars.DataFrame
        DataFrame containing ``vehicle_id``, ``node_id``, projected ``x``/``y``
        coordinates, optional geographic coordinates, and the edge identifiers
        used during sampling.

    Notes
    -----
    The sampler draws points along edges proportional to edge length, then snaps
    each point to the nearer endpoint node. As a result, every returned position
    lies on the street network.
    """

    # TODO: make sure the vehicles are sampled from connected components only...
    if VEHICLE_SAMPLE_PATH.exists():
        return pl.read_parquet(VEHICLE_SAMPLE_PATH)

    if count <= 0:
        raise ValueError("count must be positive")

    graph = _ensure_multidigraph(graph)
    edges = _length_weighted_edges(graph)

    rng = np.random.default_rng(seed)
    lengths = edges["length"].to_numpy()
    total_length = lengths.sum()
    if not np.isfinite(total_length) or total_length <= 0:
        raise ValueError("Edge lengths are not suitable for sampling")
    probabilities = lengths / total_length

    if len(edges) == 1:
        indices = np.zeros(count, dtype=int)
    else:
        indices = rng.choice(len(edges), size=count, p=probabilities)
    fractions = rng.random(count)

    samples = list(_interpolate_points(edges, indices, fractions))
    _snap_to_nodes(graph, samples)

    transformer = _build_transformer(graph) if include_lonlat else None
    _add_lon_lat(transformer, samples)

    data = {
        "vehicle_id": np.arange(count, dtype=np.int64),
        "node_id": [sample.node_id for sample in samples],
        "x": [sample.x for sample in samples],
        "y": [sample.y for sample in samples],
        "edge_u": [sample.edge_u for sample in samples],
        "edge_v": [sample.edge_v for sample in samples],
        "edge_key": [sample.edge_key for sample in samples],
    }
    if include_lonlat and transformer is not None:
        data["lon"] = [sample.lon for sample in samples]
        data["lat"] = [sample.lat for sample in samples]
    else:
        data["lon"] = [None] * count
        data["lat"] = [None] * count

    vehicles = pl.DataFrame(data)
    VEHICLE_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    vehicles.write_parquet(VEHICLE_SAMPLE_PATH)
    return vehicles


if __name__ == "__main__":
    from src.state_representation.manhatten_fast import load_graph

    street_graph = load_graph()
    df = sample_vehicle_positions(street_graph, 10, seed=42)
    print(f"Saved {df.height} vehicle samples to {VEHICLE_SAMPLE_PATH}")
