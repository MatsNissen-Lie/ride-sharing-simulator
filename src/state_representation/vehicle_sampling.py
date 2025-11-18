"""Utilities for sampling vehicle start locations on a street graph."""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import osmnx as ox
import polars as pl
from pyproj import Transformer
from shapely.geometry import LineString

from config import RIDER_DIR, RIDER_PATTERN, VEHICLE_SAMPLE_PATH, get_logger

__all__ = ["VehicleSample", "sample_vehicle_positions"]


logger = get_logger(__name__)


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


def _compute_node2scc(graph: nx.MultiDiGraph) -> dict[int, int]:
    node2scc: dict[int, int] = {}
    for cid, component in enumerate(nx.strongly_connected_components(graph)):
        for node in component:
            node2scc[int(node)] = cid
    return node2scc


def _load_rider_demand_by_scc(node2scc: dict[int, int]) -> Counter[int]:
    rider_dir = RIDER_DIR
    if not rider_dir.exists():
        return Counter()
    files = sorted(rider_dir.glob(RIDER_PATTERN))
    demand: Counter[int] = Counter()
    if not files:
        logger.info("No rider files found in %s matching %s", rider_dir, RIDER_PATTERN)
        return demand

    files_processed = 0
    total_rows = 0
    matched_rows = 0
    for path in files:
        try:
            df = pl.read_parquet(path, columns=["orig_node"])
        except FileNotFoundError:
            continue
        if "orig_node" not in df.columns:
            continue
        files_processed += 1
        series = df.get_column("orig_node").drop_nulls()
        if series.is_empty():
            continue
        nodes = series.to_numpy()
        total_rows += len(nodes)
        scc_array = np.fromiter(
            (node2scc.get(int(node), -1) for node in nodes),
            dtype=np.int64,
            count=len(nodes),
        )
        mask = scc_array >= 0
        if not np.any(mask):
            continue
        matched_rows += int(np.sum(mask))
        unique, counts = np.unique(scc_array[mask], return_counts=True)
        demand.update({int(scc): int(count) for scc, count in zip(unique, counts)})

    total_demand = sum(demand.values())
    if demand:
        logger.info(
            "Rider demand spans %d SCCs (files=%d, rows=%d, matched=%d)",
            len(demand),
            files_processed,
            total_rows,
            matched_rows,
        )
        _log_distribution(
            "Rider demand share",
            {k: float(v) for k, v in demand.items()},
            float(total_demand),
        )
    else:
        logger.info(
            "No rider demand detected among %d files using pattern %s",
            files_processed,
            RIDER_PATTERN,
        )
    return demand


def _filter_edges_for_demand(
    edges: pl.DataFrame,
    node2scc: dict[int, int],
    demand: Counter[int],
) -> tuple[pl.DataFrame, np.ndarray]:
    if not demand:
        return edges, np.array([], dtype=np.int64)

    u_nodes = edges["u"].to_numpy()
    v_nodes = edges["v"].to_numpy()
    scc_u = np.fromiter(
        (node2scc.get(int(node), -1) for node in u_nodes),
        dtype=np.int64,
        count=len(u_nodes),
    )
    scc_v = np.fromiter(
        (node2scc.get(int(node), -1) for node in v_nodes),
        dtype=np.int64,
        count=len(v_nodes),
    )

    demand_ids = np.fromiter(
        (int(key) for key in demand.keys()), dtype=np.int64, count=len(demand)
    )
    same_comp = (scc_u == scc_v) & np.isin(scc_u, demand_ids) & (scc_u >= 0)
    if not np.any(same_comp):
        msg = "No graph edges overlap SCCs with rider demand"
        raise ValueError(msg)

    edges = (
        edges.with_columns(pl.Series("_mask", same_comp))
        .filter(pl.col("_mask"))
        .drop("_mask")
    )
    return edges, scc_u[same_comp]


def _log_distribution(prefix: str, values: dict[int, float], total: float) -> None:
    if not values or total <= 0:
        return
    items = sorted(values.items(), key=lambda item: item[1], reverse=True)[:5]
    summary = ", ".join(f"{scc}:{value / total:.1%}" for scc, value in items)
    logger.info("%s across %d SCCs: %s", prefix, len(values), summary)


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

    # if VEHICLE_SAMPLE_PATH.exists():
    #     return pl.read_parquet(VEHICLE_SAMPLE_PATH)

    if count <= 0:
        raise ValueError("count must be positive")

    graph = _ensure_multidigraph(graph)
    edges = _length_weighted_edges(graph)

    node2scc = _compute_node2scc(graph)
    rider_demand = _load_rider_demand_by_scc(node2scc)
    if rider_demand:
        edges, edge_scc_ids = _filter_edges_for_demand(edges, node2scc, rider_demand)
        lengths = edges["length"].to_numpy()
        demand_weights = np.fromiter(
            (float(rider_demand[int(scc)]) for scc in edge_scc_ids),
            dtype=float,
            count=len(edge_scc_ids),
        )
        weights = lengths * demand_weights
        weight_by_scc: dict[int, float] = {}
        for scc, weight in zip(edge_scc_ids, weights):
            key = int(scc)
            weight_by_scc[key] = weight_by_scc.get(key, 0.0) + float(weight)
        weight_total = float(sum(weight_by_scc.values()))
        _log_distribution(
            "Vehicle sampling weight share",
            weight_by_scc,
            weight_total,
        )
    else:
        logger.info(
            "No rider demand data available; sampling vehicles purely by edge length"
        )
        lengths = edges["length"].to_numpy()
        weights = lengths
        weight_total = float(weights.sum())

    rng = np.random.default_rng(seed)
    total_weight = weight_total
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Edge weights are not suitable for sampling")
    probabilities = weights / total_weight

    if len(edges) == 1:
        indices = np.zeros(count, dtype=int)
    else:
        indices = rng.choice(len(edges), size=count, p=probabilities)
    fractions = rng.random(count)

    samples = list(_interpolate_points(edges, indices, fractions))
    _snap_to_nodes(graph, samples)

    transformer = _build_transformer(graph) if include_lonlat else None
    _add_lon_lat(transformer, samples)

    vehicle_scc_counts: Counter[int] = Counter()
    unassigned = 0
    for sample in samples:
        scc_id = node2scc.get(sample.node_id, -1)
        if scc_id >= 0:
            vehicle_scc_counts[int(scc_id)] += 1
        else:
            unassigned += 1
    _log_distribution(
        "Sampled vehicle share",
        {k: float(v) for k, v in vehicle_scc_counts.items()},
        float(len(samples)),
    )
    if unassigned:
        logger.warning("Sampled %d vehicle positions outside known SCCs", unassigned)

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
    df = sample_vehicle_positions(street_graph, 1000, seed=42)
    print(f"Saved {df.height} vehicle samples to {VEHICLE_SAMPLE_PATH}")
