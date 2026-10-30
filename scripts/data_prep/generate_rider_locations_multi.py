"""Batch enrich FHV trip parquet files with network snaps and coordinates."""

from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, cast

import geopandas as gpd
import numpy as np
import osmnx as ox
import polars as pl
import networkx as nx
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from config import format_log_block, get_logger, RIDER_DIR
from config.paths import TAXI_ZONES_PATH
from src.components.network.network_service import NetworkXService
from src.inputs.manhattan_graph import load_graph

logger = get_logger(__name__)
logger.setLevel("DEBUG")


@dataclass(frozen=True)
class TripStats:
    total_before: int
    candidate_count: int
    pickup_missing: int
    dropoff_missing: int
    datetime_missing: int
    after_zone: int
    removed_after_filter: int
    written: int
    dropped_snapping: int
    path_resample_success: int
    path_resample_fail: int


INPUT_DIR = RIDER_DIR
OUTPUT_DIR = RIDER_DIR
FILE_PATTERNS = [
    # "fhv_tripdata_2025-*.parquet",
    "fhvhv_tripdata_2025-*.parquet",
    "yellow_tripdata_2025-*.parquet",
]

TMP_DIR = OUTPUT_DIR / "_tmp_enriched"

MERGED_KEEP_COLUMNS = [
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_zone",
    "dropoff_zone",
    "orig_node",
    "dest_node",
    "pu_x",
    "pu_y",
    "do_x",
    "do_y",
    "pu_lon",
    "pu_lat",
    "do_lon",
    "do_lat",
]

SAMPLE_BATCH_SIZE = 20_000
PICKUP_SEED = 7
DROPOFF_SEED = 17
MAX_RESAMPLE_ATTEMPTS = 10
MONTHLY_OUTPUT_PARTS = 5


class ZoneSampler:
    """Cache per-zone edge subsets and provide reproducible sampling."""

    def __init__(self, edges: gpd.GeoDataFrame) -> None:
        if "length" not in edges.columns:
            msg = "Expected edge GeoDataFrame to include a 'length' column"
            raise ValueError(msg)
        self._edges = edges[["geometry", "length"]].copy()
        self._cache: Dict[int, Optional[gpd.GeoDataFrame]] = {}

    def _edges_for_zone(
        self, zone_id: int, polygon: Optional[BaseGeometry]
    ) -> Optional[gpd.GeoDataFrame]:
        if polygon is None or polygon.is_empty:
            self._cache[zone_id] = None
            return None

        mask = self._edges.geometry.intersects(polygon)
        if not bool(mask.any()):
            self._cache[zone_id] = None
            return None

        edges = self._edges.loc[mask]
        if edges.empty or float(edges["length"].sum()) <= 0:
            self._cache[zone_id] = None
            return None

        self._cache[zone_id] = edges
        return edges

    def sample_points(
        self,
        zone_id: int,
        polygon: Optional[BaseGeometry],
        n: int,
        rng: np.random.Generator,
    ) -> Optional[gpd.GeoSeries]:
        if n <= 0:
            return None

        edges = self._cache.get(zone_id)
        if edges is None:
            edges = self._edges_for_zone(zone_id, polygon)
        if edges is None:
            return None

        weights = edges["length"].to_numpy(dtype=float)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            return None
        weights = weights / total

        selection = rng.choice(len(edges), size=n, p=weights)
        chosen = edges.iloc[selection]
        positions = rng.random(n)
        geoms = [
            geom.interpolate(float(pos), normalized=True)
            for geom, pos in zip(chosen["geometry"], positions)
        ]
        return gpd.GeoSeries(geoms, crs=edges.crs)


def load_trips(path: Path) -> pl.DataFrame:
    logger.info("Loading rider trips from %s", path)
    df = pl.read_parquet(path)
    cols = set(df.columns)

    fhv_schema = {
        "pickup_datetime",
        "dropOff_datetime",
        "PUlocationID",
        "DOlocationID",
    }
    fhvhv_schema = {
        "pickup_datetime",
        "dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    }
    yellow_schema = {
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
    }

    if fhv_schema <= cols:
        trips = df.select(
            [
                pl.col("pickup_datetime"),
                pl.col("dropOff_datetime").alias("dropoff_datetime"),
                pl.col("PUlocationID").alias("pickup_zone"),
                pl.col("DOlocationID").alias("dropoff_zone"),
            ]
        )
    elif fhvhv_schema <= cols:
        trips = df.select(
            [
                pl.col("pickup_datetime"),
                pl.col("dropoff_datetime"),
                pl.col("PULocationID").alias("pickup_zone"),
                pl.col("DOLocationID").alias("dropoff_zone"),
            ]
        )
    elif yellow_schema <= cols:
        trips = df.select(
            [
                pl.col("tpep_pickup_datetime").alias("pickup_datetime"),
                pl.col("tpep_dropoff_datetime").alias("dropoff_datetime"),
                pl.col("PULocationID").alias("pickup_zone"),
                pl.col("DOLocationID").alias("dropoff_zone"),
            ]
        )
    else:
        msg = f"Unrecognized trip schema for file {path}"
        raise ValueError(msg)

    trips = trips.with_columns(
        pl.col("pickup_zone").cast(pl.Int32),
        pl.col("dropoff_zone").cast(pl.Int32),
        pl.col("pickup_datetime").cast(pl.Datetime(time_unit="us")),
        pl.col("dropoff_datetime").cast(pl.Datetime(time_unit="us")),
    )
    logger.info("Trips loaded from %s: %d records", path.name, trips.height)
    return trips


def load_zones(path: Path, target_crs: str) -> gpd.GeoDataFrame:
    logger.info("Loading taxi zones from %s", path)
    zones = gpd.read_file(path)
    zones = zones[["LocationID", "zone", "borough", "geometry"]]
    zones = zones.to_crs(target_crs)
    logger.info("Taxi zones loaded: %d polygons", len(zones))
    return zones


def uniform_points_in_geometry(
    geom: Optional[BaseGeometry],
    count: int,
    rng: np.random.Generator,
) -> list[Point]:
    if geom is None or geom.is_empty or count <= 0:
        return []

    if isinstance(geom, GeometryCollection):
        geom = geom.buffer(0)

    if isinstance(geom, Polygon):
        polygons: list[Polygon] = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = [
            poly
            for poly in geom.geoms
            if isinstance(poly, Polygon) and not poly.is_empty
        ]
    else:
        polygons = [cast(Polygon, geom.convex_hull)]

    polygons = [poly for poly in polygons if poly.area > 0]
    if not polygons:
        return [geom.representative_point() for _ in range(count)]

    areas = np.array([poly.area for poly in polygons], dtype=float)
    weights = areas / areas.sum()
    allocations = rng.multinomial(count, weights)

    points: list[Point] = []
    for poly, quota in zip(polygons, allocations):
        if quota == 0:
            continue
        minx, miny, maxx, maxy = poly.bounds
        generated = 0
        attempts = 0
        max_attempts = max(100, quota * 25)
        while generated < quota and attempts < max_attempts:
            xs = rng.uniform(minx, maxx, size=quota - generated)
            ys = rng.uniform(miny, maxy, size=quota - generated)
            for x, y in zip(xs, ys):
                candidate = Point(float(x), float(y))
                if poly.contains(candidate):
                    points.append(candidate)
                    generated += 1
                attempts += 1
                if generated >= quota:
                    break
        while generated < quota:
            points.append(poly.representative_point())
            generated += 1
    return points


def gen_coords_for_zones(
    zone_ids: np.ndarray,
    sampler: ZoneSampler,
    zone_geoms: Mapping[int, BaseGeometry],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = np.full(len(zone_ids), np.nan, dtype=float)
    ys = np.full(len(zone_ids), np.nan, dtype=float)

    for start in range(0, len(zone_ids), SAMPLE_BATCH_SIZE):
        stop = min(start + SAMPLE_BATCH_SIZE, len(zone_ids))
        batch_indices = np.arange(start, stop)
        batch_zones = zone_ids[start:stop]
        unique_zones = np.unique(batch_zones)
        for zone_id in unique_zones:
            mask = batch_zones == zone_id
            indices = batch_indices[mask]
            n_points = indices.size
            polygon = zone_geoms.get(int(zone_id))
            sampled = sampler.sample_points(int(zone_id), polygon, n_points, rng)
            if sampled is not None and len(sampled) == n_points:
                xs[indices] = sampled.x.to_numpy(dtype=float, copy=False)
                ys[indices] = sampled.y.to_numpy(dtype=float, copy=False)
            else:
                fallback = uniform_points_in_geometry(polygon, n_points, rng)
                if fallback:
                    xs[indices] = np.fromiter(
                        (pt.x for pt in fallback), float, count=n_points
                    )
                    ys[indices] = np.fromiter(
                        (pt.y for pt in fallback), float, count=n_points
                    )
                else:
                    xs[indices] = np.nan
                    ys[indices] = np.nan
    return xs, ys


def snap_points_to_nodes(
    graph: nx.MultiDiGraph,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """
    Snap points to their nearest graph nodes. Uses edge midpoints to decide
    which node to snap to.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    nodes = np.full(xs.shape, -1, dtype=np.int64)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.any():
        edges, _ = ox.distance.nearest_edges(
            graph, xs[valid], ys[valid], return_dist=True  # type: ignore[arg-type]
        )
        chosen = []
        for (u, v, _), x, y in zip(edges, xs[valid], ys[valid]):
            ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
            vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
            if (ux - x) ** 2 + (uy - y) <= (vx - x) ** 2 + (vy - y):
                chosen.append(int(u))
            else:
                chosen.append(int(v))
        nodes[valid] = np.array(chosen, dtype=np.int64)
    return nodes


def to_lonlat(
    xs: np.ndarray,
    ys: np.ndarray,
    transformer: Transformer,
) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.full(xs.shape, np.nan, dtype=float)
    lat = np.full(ys.shape, np.nan, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.any():
        lon_valid, lat_valid = transformer.transform(xs[valid], ys[valid])
        lon[valid] = lon_valid
        lat[valid] = lat_valid
    return lon, lat


def mask_points_inside_polygon(
    xs: np.ndarray,
    ys: np.ndarray,
    polygon: BaseGeometry,
) -> np.ndarray:
    mask = np.zeros_like(xs, dtype=bool)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        return mask
    points = (Point(float(x), float(y)) for x, y in zip(xs[finite], ys[finite]))
    inside = np.fromiter((polygon.contains(pt) for pt in points), dtype=bool)
    mask[finite] = inside
    return mask


# --- helpers: filtering / loading -------------------------------------------------


def _filter_candidates(
    trips: pl.DataFrame, valid_zone_ids: set[int]
) -> tuple[pl.DataFrame, dict]:
    valid_list = sorted(valid_zone_ids)

    pickup_valid_or_nan = (
        pl.col("pickup_zone").is_in(valid_list) | pl.col("pickup_zone").is_null()
    )
    dropoff_valid_or_nan = (
        pl.col("dropoff_zone").is_in(valid_list) | pl.col("dropoff_zone").is_null()
    )
    both_nan = pl.col("pickup_zone").is_null() & pl.col("dropoff_zone").is_null()

    candidates = trips.filter(pickup_valid_or_nan & dropoff_valid_or_nan & ~both_nan)

    pickup_missing = candidates.filter(pl.col("pickup_zone").is_null()).height
    dropoff_missing = candidates.filter(pl.col("dropoff_zone").is_null()).height
    datetime_missing = candidates.filter(pl.col("pickup_datetime").is_null()).height

    datetime_valid_mask = pl.col("pickup_datetime").is_not_null()
    pickup_valid_mask = pl.col("pickup_zone").is_in(valid_list)
    dropoff_valid_mask = pl.col("dropoff_zone").is_in(valid_list)

    trimmed = candidates.filter(
        pickup_valid_mask & dropoff_valid_mask & datetime_valid_mask
    ).with_row_index(name="row_id")

    stats = dict(
        candidate_count=candidates.height,
        pickup_missing=pickup_missing,
        dropoff_missing=dropoff_missing,
        datetime_missing=datetime_missing,
        removed_after_candidates=candidates.height - trimmed.height,
    )
    return trimmed, stats


# --- helpers: geometry / snapping -------------------------------------------------


def _gen_and_snap(
    pickup_zones: np.ndarray,
    dropoff_zones: np.ndarray,
    sampler: ZoneSampler,
    zone_geoms: Mapping[int, BaseGeometry],
    graph_hull: BaseGeometry,
    graph_proj: nx.MultiDiGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pu_x, pu_y = gen_coords_for_zones(pickup_zones, sampler, zone_geoms, PICKUP_SEED)
    do_x, do_y = gen_coords_for_zones(dropoff_zones, sampler, zone_geoms, DROPOFF_SEED)

    pu_inside = mask_points_inside_polygon(pu_x, pu_y, graph_hull)
    do_inside = mask_points_inside_polygon(do_x, do_y, graph_hull)

    pu_x = np.where(pu_inside, pu_x, np.nan)
    pu_y = np.where(pu_inside, pu_y, np.nan)
    do_x = np.where(do_inside, do_x, np.nan)
    do_y = np.where(do_inside, do_y, np.nan)

    pu_nodes = snap_points_to_nodes(graph_proj, pu_x, pu_y)
    do_nodes = snap_points_to_nodes(graph_proj, do_x, do_y)
    return pu_x, pu_y, do_x, do_y, pu_nodes, do_nodes


# --- helpers: SCC / reachability --------------------------------------------------


def _get_node2scc(
    graph_proj: nx.MultiDiGraph, network_service: NetworkXService
) -> dict[int, int]:
    node2scc_attr = getattr(network_service, "node2scc", None)
    if isinstance(node2scc_attr, dict):
        return cast(dict[int, int], node2scc_attr)
    node2scc: dict[int, int] = {}
    for cid, comp in enumerate(nx.strongly_connected_components(graph_proj)):
        for n in comp:
            node2scc[n] = cid
    return node2scc


def _pairs_need_resample(
    pu_nodes: np.ndarray, do_nodes: np.ndarray, node2scc: dict[int, int]
) -> np.ndarray:
    invalid_nodes = (pu_nodes == -1) | (do_nodes == -1)
    need = invalid_nodes.copy()
    idxs = np.where(~invalid_nodes)[0]
    if idxs.size:

        def eq(u, v):
            cu = node2scc.get(int(u))
            cv = node2scc.get(int(v))
            return (cu is not None) and (cv is not None) and (cu == cv) and (u != v)

        mismatch = np.fromiter(
            (not eq(pu_nodes[i], do_nodes[i]) for i in idxs), bool, count=idxs.size
        )
        need[idxs] |= mismatch
    return need


# --- helpers: batched resampling --------------------------------------------------


def _batch_resample_indices(
    idxs: np.ndarray,
    pickup_zones: np.ndarray,
    dropoff_zones: np.ndarray,
    sampler: ZoneSampler,
    zone_geoms: Mapping[int, BaseGeometry],
    graph_hull: BaseGeometry,
    graph_proj: nx.MultiDiGraph,
    node2scc: dict[int, int],
    pu_x: np.ndarray,
    pu_y: np.ndarray,
    do_x: np.ndarray,
    do_y: np.ndarray,
    pu_nodes: np.ndarray,
    do_nodes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Resample PU/DO for provided indices in one pass; return updated fail_mask for those indices and count fixed."""
    if idxs.size == 0:
        return np.zeros(0, dtype=bool), 0

    rng = np.random.default_rng(PICKUP_SEED + DROPOFF_SEED)
    seed1 = int(rng.integers(0, 2**32 - 1))
    seed2 = seed1 ^ 0x9E3779B9

    new_pu_x, new_pu_y = gen_coords_for_zones(
        pickup_zones[idxs], sampler, zone_geoms, seed1
    )
    new_do_x, new_do_y = gen_coords_for_zones(
        dropoff_zones[idxs], sampler, zone_geoms, seed2
    )

    inside_pu = mask_points_inside_polygon(new_pu_x, new_pu_y, graph_hull)
    inside_do = mask_points_inside_polygon(new_do_x, new_do_y, graph_hull)

    # snap
    snapped_pu = snap_points_to_nodes(
        graph_proj,
        np.where(inside_pu, new_pu_x, np.nan),
        np.where(inside_pu, new_pu_y, np.nan),
    )
    snapped_do = snap_points_to_nodes(
        graph_proj,
        np.where(inside_do, new_do_x, np.nan),
        np.where(inside_do, new_do_y, np.nan),
    )

    # write back where both snaps succeeded
    ok = (snapped_pu != -1) & (snapped_do != -1)
    if np.any(ok):
        ok_idx = idxs[ok]
        pu_x[ok_idx] = new_pu_x[ok]
        pu_y[ok_idx] = new_pu_y[ok]
        do_x[ok_idx] = new_do_x[ok]
        do_y[ok_idx] = new_do_y[ok]
        pu_nodes[ok_idx] = snapped_pu[ok]
        do_nodes[ok_idx] = snapped_do[ok]

    # recompute failures for these indices
    still_invalid = (pu_nodes[idxs] == -1) | (do_nodes[idxs] == -1)
    still_mismatch = np.zeros(idxs.size, dtype=bool)
    ok_nodes = ~still_invalid
    if np.any(ok_nodes):
        sub = idxs[ok_nodes]

        def eq(u, v):
            cu = node2scc.get(int(u))
            cv = node2scc.get(int(v))
            return (cu is not None) and (cv is not None) and (cu == cv) and (u != v)

        still_mismatch[ok_nodes] = np.fromiter(
            (not eq(pu_nodes[i], do_nodes[i]) for i in sub), bool, count=sub.size
        )

    fail_mask_sub = still_invalid | still_mismatch
    fixed_count = int(np.sum(~fail_mask_sub))
    return fail_mask_sub, fixed_count


# --- helpers: attach outputs / write --------------------------------------------


def _attach_outputs(
    trips: pl.DataFrame,
    pu_x: np.ndarray,
    pu_y: np.ndarray,
    do_x: np.ndarray,
    do_y: np.ndarray,
    pu_nodes: np.ndarray,
    do_nodes: np.ndarray,
    graph_proj: nx.MultiDiGraph,
) -> tuple[pl.DataFrame, int]:
    transformer = Transformer.from_crs(
        graph_proj.graph["crs"], "EPSG:4326", always_xy=True
    )
    pu_lon, pu_lat = to_lonlat(pu_x, pu_y, transformer)
    do_lon, do_lat = to_lonlat(do_x, do_y, transformer)

    out = trips.with_columns(
        pl.Series("pu_x", pu_x, dtype=pl.Float64),
        pl.Series("pu_y", pu_y, dtype=pl.Float64),
        pl.Series("do_x", do_x, dtype=pl.Float64),
        pl.Series("do_y", do_y, dtype=pl.Float64),
        pl.Series("orig_node", pu_nodes).replace(-1, None).cast(pl.Int64),
        pl.Series("dest_node", do_nodes).replace(-1, None).cast(pl.Int64),
        pl.Series("pu_lon", pu_lon, dtype=pl.Float64),
        pl.Series("pu_lat", pu_lat, dtype=pl.Float64),
        pl.Series("do_lon", do_lon, dtype=pl.Float64),
        pl.Series("do_lat", do_lat, dtype=pl.Float64),
    )

    invalid_mask = out["orig_node"].is_null() | out["dest_node"].is_null()
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        logger.info(
            "Dropping %d trips that could not be snapped / connected", invalid_count
        )
        out = out.filter(~invalid_mask)
    return out, invalid_count


# --- helpers: monthly merge -----------------------------------------------------


def merge_month_temp_files(
    month_key: str,
    temp_paths: list[Path],
    output_path: Path,
) -> tuple[int, list[Path]]:
    """Merge temp parquet files for a month, write five chunked outputs, then delete temps."""
    dfs: list[pl.DataFrame] = []
    existing_paths: list[Path] = []
    for path in temp_paths:
        if not path.exists():
            logger.warning("Temp file %s missing before merge", path)
            continue
        dfs.append(pl.read_parquet(path))
        existing_paths.append(path)

    if not dfs:
        logger.info("No temp dataframes to merge for month %s", month_key)
        return 0, []

    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)

    if MERGED_KEEP_COLUMNS:
        common_cols &= set(MERGED_KEEP_COLUMNS)

    if not common_cols:
        files = ", ".join(path.name for path in existing_paths)
        msg = f"No common columns to merge for month {month_key} (files: {files})"
        raise ValueError(msg)

    cols = sorted(common_cols)
    merged = pl.concat([df.select(cols) for df in dfs], how="vertical")
    if "pickup_datetime" in merged.columns:
        merged = merged.sort("pickup_datetime")

    total_rows = merged.height
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove legacy single-file outputs and stale part files to avoid stale artifacts.
    output_path.unlink(missing_ok=True)
    stale_pattern = f"{output_path.stem}_part"
    for stale in output_path.parent.glob(f"{stale_pattern}*{output_path.suffix}"):
        try:
            stale.unlink(missing_ok=True)
        except FileNotFoundError:
            continue

    if total_rows == 0:
        logger.info(
            "Merged %d temp files for %s but no rows remain", len(dfs), month_key
        )
        return 0, []

    parts: list[Path] = []
    chunk_size = max(1, ceil(total_rows / MONTHLY_OUTPUT_PARTS))
    for part_idx in range(MONTHLY_OUTPUT_PARTS):
        start = part_idx * chunk_size
        if start >= total_rows:
            break
        stop = min(start + chunk_size, total_rows)
        chunk = merged.slice(start, stop - start)
        part_path = output_path.with_name(
            f"{output_path.stem}_part{part_idx + 1}{output_path.suffix}"
        )
        chunk.write_parquet(part_path)
        parts.append(part_path)
        logger.info(
            "Merged %d temp files for %s -> %s (%d rows, chunk %d/%d)",
            len(dfs),
            month_key,
            part_path.name,
            chunk.height,
            part_idx + 1,
            MONTHLY_OUTPUT_PARTS,
        )

    for path in existing_paths:
        try:
            path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
    for parent in {path.parent for path in existing_paths}:
        try:
            parent.rmdir()
        except OSError:
            pass

    return total_rows, parts


# --- orchestration: main function ------------------------------------------------


def process_trip_file(
    input_path: Path,
    graph_proj: nx.MultiDiGraph,
    network_service: NetworkXService,
    sampler: ZoneSampler,
    zone_geoms: Mapping[int, BaseGeometry],
    valid_zone_ids: set[int],
    graph_hull: BaseGeometry,
) -> tuple[pl.DataFrame, TripStats]:
    logger.info("Processing %s", input_path.name)
    trips = load_trips(input_path)
    total_before = trips.height
    if total_before == 0:
        logger.info("Skipping %s (no trips)", input_path.name)
        empty_stats = TripStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return pl.DataFrame(), empty_stats

    trips, pre_stats = _filter_candidates(trips, valid_zone_ids)
    logger.info(
        "Trips w/ Manhattan zone (%s): total=%d, pickup_missing=%d, dropoff_missing=%d, "
        "missing_datetime=%d, removed_after_filter=%d",
        input_path.name,
        pre_stats["candidate_count"],
        pre_stats["pickup_missing"],
        pre_stats["dropoff_missing"],
        pre_stats["datetime_missing"],
        pre_stats["removed_after_candidates"],
    )
    after_zone = trips.height
    logger.info("Trips remaining after all filters: %d", after_zone)
    if after_zone == 0:
        stats = TripStats(
            total_before=total_before,
            candidate_count=pre_stats["candidate_count"],
            pickup_missing=pre_stats["pickup_missing"],
            dropoff_missing=pre_stats["dropoff_missing"],
            datetime_missing=pre_stats["datetime_missing"],
            after_zone=0,
            removed_after_filter=pre_stats["removed_after_candidates"],
            written=0,
            dropped_snapping=0,
            path_resample_success=0,
            path_resample_fail=0,
        )
        return pl.DataFrame(), stats

    pickup_zones = trips["pickup_zone"].to_numpy()
    dropoff_zones = trips["dropoff_zone"].to_numpy()

    pu_x, pu_y, do_x, do_y, pu_nodes, do_nodes = _gen_and_snap(
        pickup_zones, dropoff_zones, sampler, zone_geoms, graph_hull, graph_proj
    )

    node2scc = _get_node2scc(graph_proj, network_service)
    need_resample = _pairs_need_resample(pu_nodes, do_nodes, node2scc)

    path_resample_success = 0
    path_resample_fail = 0

    # Batched attempts
    fail_mask = need_resample.copy()
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        idxs = np.where(fail_mask)[0]
        if idxs.size == 0:
            break
        fail_mask_sub, fixed = _batch_resample_indices(
            idxs,
            pickup_zones,
            dropoff_zones,
            sampler,
            zone_geoms,
            graph_hull,
            graph_proj,
            node2scc,
            pu_x,
            pu_y,
            do_x,
            do_y,
            pu_nodes,
            do_nodes,
        )
        path_resample_success += fixed
        # merge submask back
        fail_mask[idxs] = fail_mask_sub

    if np.any(fail_mask):
        path_resample_fail += int(np.sum(fail_mask))
        pu_nodes[fail_mask] = -1
        do_nodes[fail_mask] = -1

    # Attach outputs, drop invalids
    trips, invalid_count = _attach_outputs(
        trips, pu_x, pu_y, do_x, do_y, pu_nodes, do_nodes, graph_proj
    )

    final_count = trips.height
    trips = trips.sort("pickup_datetime")
    if "row_id" in trips.columns:
        trips = trips.drop("row_id")

    summary = format_log_block(
        "\n".join(
            [
                f"{input_path.name} summary:",
                f"  total trips: {total_before}",
                f"  candidates with Manhattan zone: {pre_stats['candidate_count']}",
                f"  pickup missing: {pre_stats['pickup_missing']}",
                f"  dropoff missing: {pre_stats['dropoff_missing']}",
                f"  datetime missing: {pre_stats['datetime_missing']}",
                f"  after zone filter: {after_zone}",
                f"  dropped during snapping: {invalid_count}",
                f"  path resample success: {path_resample_success}",
                f"  path resample failures: {path_resample_fail}",
                f"  written (in-memory): {final_count}",
            ]
        )
    )
    logger.info(summary)

    stats = TripStats(
        total_before=total_before,
        candidate_count=pre_stats["candidate_count"],
        pickup_missing=pre_stats["pickup_missing"],
        dropoff_missing=pre_stats["dropoff_missing"],
        datetime_missing=pre_stats["datetime_missing"],
        after_zone=after_zone,
        removed_after_filter=pre_stats["removed_after_candidates"],
        written=final_count,
        dropped_snapping=invalid_count,
        path_resample_success=path_resample_success,
        path_resample_fail=path_resample_fail,
    )
    return trips, stats


def main() -> None:
    logger.info("Preparing Manhattan graph context")
    graph = load_graph()
    graph_proj = ox.project_graph(graph)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)
    logger.info(
        "Graph projected: %d nodes, %d edges",
        graph_proj.number_of_nodes(),
        graph_proj.number_of_edges(),
    )

    zones = load_zones(TAXI_ZONES_PATH, graph_proj.graph["crs"])
    zone_geoms: Dict[int, BaseGeometry] = {}
    for _, row in zones.iterrows():
        geometry = cast(BaseGeometry, row["geometry"])
        zone_geoms[int(row["LocationID"])] = geometry

    graph_hull = nodes_gdf.geometry.union_all().convex_hull
    valid_zone_ids = {
        zone_id
        for zone_id, geom in zone_geoms.items()
        if geom is not None and geom.intersects(graph_hull)
    }

    edges_sindex = edges_gdf.sindex
    zones_with_edges: set[int] = set()
    for zone_id in list(valid_zone_ids):
        polygon = zone_geoms.get(zone_id)
        if polygon is None or polygon.is_empty:
            continue
        if edges_sindex is None:
            has_edges = bool(edges_gdf.geometry.intersects(polygon).any())
        else:
            hits = edges_sindex.query(polygon, predicate="intersects")
            has_edges = len(hits) > 0
        if has_edges:
            zones_with_edges.add(zone_id)

    zones_without_edges = len(valid_zone_ids - zones_with_edges)
    valid_zone_ids &= zones_with_edges

    logger.info(
        "Zones intersecting the Manhattan graph: %d of %d",
        len(valid_zone_ids),
        len(zone_geoms),
    )
    if zones_without_edges:
        logger.info("Discarded %d zones without drivable edges", zones_without_edges)

    all_files: list[Path] = []
    for pattern in FILE_PATTERNS:
        all_files.extend(
            path
            for path in INPUT_DIR.glob(pattern)
            if path.is_file() and not path.name.endswith("_enriched.parquet")
        )

    if not all_files:
        logger.warning(
            "No input files matching any of %s in %s", FILE_PATTERNS, INPUT_DIR
        )
        return

    files_by_month: dict[str, list[Path]] = defaultdict(list)
    for path in all_files:
        stem = path.stem
        parts = stem.split("_")
        month_key = parts[-1] if len(parts) >= 2 else stem
        files_by_month[month_key].append(path)

    network_service = NetworkXService(graph_proj)
    sampler = ZoneSampler(edges_gdf)

    totals: list[TripStats] = []
    for month_key in sorted(files_by_month.keys()):
        month_files = sorted(files_by_month[month_key])
        logger.info(
            "Processing month %s (%d files): %s",
            month_key,
            len(month_files),
            ", ".join(path.name for path in month_files),
        )

        month_tmp_dir = TMP_DIR / month_key
        month_tmp_dir.mkdir(parents=True, exist_ok=True)
        month_tmp_paths: list[Path] = []
        for input_path in month_files:
            df_trips, stats = process_trip_file(
                input_path,
                graph_proj,
                network_service,
                sampler,
                zone_geoms,
                set(valid_zone_ids),
                graph_hull,
            )
            totals.append(stats)
            if df_trips.height > 0:
                tmp_path = month_tmp_dir / f"{input_path.stem}_tmp.parquet"
                df_trips.write_parquet(tmp_path)
                month_tmp_paths.append(tmp_path)

        output_path = OUTPUT_DIR / f"trips_{month_key}_enriched.parquet"
        merged_rows, part_paths = merge_month_temp_files(
            month_key, month_tmp_paths, output_path
        )
        if merged_rows == 0:
            logger.info("Month %s yielded no merged trips; skipping write", month_key)
            continue

        logger.info(
            "Wrote %d monthly parquet parts for %s (%d rows total): %s",
            len(part_paths),
            month_key,
            merged_rows,
            ", ".join(path.name for path in part_paths),
        )

    total_files = len(all_files)
    total_before = sum(item.total_before for item in totals)
    total_candidates = sum(item.candidate_count for item in totals)
    total_pickup_missing = sum(item.pickup_missing for item in totals)
    total_dropoff_missing = sum(item.dropoff_missing for item in totals)
    total_datetime_missing = sum(item.datetime_missing for item in totals)
    total_after_zone = sum(item.after_zone for item in totals)
    total_removed_after_filter = sum(item.removed_after_filter for item in totals)
    total_written = sum(item.written for item in totals)
    total_dropped = sum(item.dropped_snapping for item in totals)
    total_path_resample_success = sum(item.path_resample_success for item in totals)
    total_path_resample_fail = sum(item.path_resample_fail for item in totals)

    summary = format_log_block(
        "\n".join(
            [
                "Batch rider enrichment complete:",
                f"Processed raw files: {total_files}",
                f"Trips total: {total_before}",
                f"Trips with Manhattan zone: {total_candidates}",
                f"Pickup missing among candidates: {total_pickup_missing}",
                f"Dropoff missing among candidates: {total_dropoff_missing}",
                f"Datetime missing among candidates: {total_datetime_missing}",
                f"Trips after strict zone filter: {total_after_zone}",
                f"Removed after zone filter: {total_removed_after_filter}",
                f"Trips dropped during snapping: {total_dropped}",
                f"Path resample success: {total_path_resample_success}",
                f"Path resample failures: {total_path_resample_fail}",
                f"Trips written: {total_written}",
            ]
        )
    )
    logger.info(summary)


if __name__ == "__main__":
    main()
