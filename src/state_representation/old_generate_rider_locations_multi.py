"""Batch enrich FHV trip parquet files with network snaps and coordinates."""

from dataclasses import dataclass
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

from config import format_log_block, get_logger, RIDER_DIR
from src.components.network_service import NetworkXService

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
FILE_PATTERN = "fhv_tripdata_2025-*.parquet"

SAMPLE_BATCH_SIZE = 20_000
PICKUP_SEED = 7
DROPOFF_SEED = 17
MAX_RESAMPLE_ATTEMPTS = 10


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
    trips = pl.read_parquet(
        path,
        columns=[
            "pickup_datetime",
            "dropOff_datetime",
            "PUlocationID",
            "DOlocationID",
        ],
    )
    trips = trips.rename(
        {
            "PUlocationID": "pickup_zone",
            "DOlocationID": "dropoff_zone",
            "dropOff_datetime": "dropoff_datetime",
        }
    )
    trips = trips.with_columns(
        pl.col("pickup_zone").cast(pl.Int32),
        pl.col("dropoff_zone").cast(pl.Int32),
        pl.col("pickup_datetime").cast(pl.Datetime(time_unit="us")),
        pl.col("dropoff_datetime").cast(pl.Datetime(time_unit="us")),
    )
    logger.info("Trips loaded: %d records", trips.height)
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


def process_trip_file_old(
    input_path: Path,
    output_path: Path,
    graph_proj: nx.MultiDiGraph,
    network_service: NetworkXService,
    sampler: ZoneSampler,
    zone_geoms: Mapping[int, BaseGeometry],
    valid_zone_ids: set[int],
    graph_hull: BaseGeometry,
) -> TripStats:
    logger.info("Processing %s", input_path.name)
    trips = load_trips(input_path)
    total_before = trips.height
    if total_before == 0:
        logger.info("Skipping %s (no trips)", input_path.name)
        return TripStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    valid_list = sorted(valid_zone_ids)

    pickup_valid_or_nan = (
        pl.col("pickup_zone").is_in(valid_list) | pl.col("pickup_zone").is_null()
    )
    dropoff_valid_or_nan = (
        pl.col("dropoff_zone").is_in(valid_list) | pl.col("dropoff_zone").is_null()
    )
    both_nan = pl.col("pickup_zone").is_null() & pl.col("dropoff_zone").is_null()

    candidates = trips.filter(pickup_valid_or_nan & dropoff_valid_or_nan & ~both_nan)
    candidate_count = candidates.height
    if candidate_count == 0:
        logger.info(
            "Skipping %s (no trips have Manhattan pickup OR dropoff zone)",
            input_path.name,
        )
        return TripStats(total_before, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    pickup_missing = candidates.filter(pl.col("pickup_zone").is_null()).height
    dropoff_missing = candidates.filter(pl.col("dropoff_zone").is_null()).height
    datetime_missing = candidates.filter(pl.col("pickup_datetime").is_null()).height

    datetime_valid_mask = pl.col("pickup_datetime").is_not_null()
    pickup_valid_mask = pl.col("pickup_zone").is_in(valid_list)
    dropoff_valid_mask = pl.col("dropoff_zone").is_in(valid_list)

    trips = candidates.filter(
        pickup_valid_mask & dropoff_valid_mask & datetime_valid_mask
    ).with_row_index(name="row_id")

    removed_after_candidates = candidate_count - trips.height
    logger.info(
        "Trips w/ Manhattan zone (%s): total=%d, pickup_missing=%d, dropoff_missing=%d, "
        "missing_datetime=%d, removed_after_filter=%d",
        input_path.name,
        candidate_count,
        pickup_missing,
        dropoff_missing,
        datetime_missing,
        removed_after_candidates,
    )
    logger.info("Trips remaining after all filters: %d", trips.height)

    after_zone = trips.height
    if after_zone == 0:
        logger.info("Skipping %s (no trips remain after zone filter)", input_path.name)
        return TripStats(
            total_before=total_before,
            candidate_count=candidate_count,
            pickup_missing=pickup_missing,
            dropoff_missing=dropoff_missing,
            datetime_missing=datetime_missing,
            after_zone=0,
            removed_after_filter=removed_after_candidates,
            written=0,
            dropped_snapping=0,
            path_resample_success=0,
            path_resample_fail=0,
        )

    # --- Generate pickup / dropoff coordinates in batch ---
    pickup_zones = trips["pickup_zone"].to_numpy()
    dropoff_zones = trips["dropoff_zone"].to_numpy()

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

    # --- Strongly connected components precheck (O(1) reachability) ---

    logger.info("Building SCC map for reachability precheck...")
    node2scc = {}
    for cid, comp in enumerate(nx.strongly_connected_components(graph_proj)):
        for n in comp:
            node2scc[n] = cid

    def scc_equal(u: int, v: int) -> bool:
        cu = node2scc.get(int(u))
        cv = node2scc.get(int(v))
        return (cu is not None) and (cv is not None) and (cu == cv)

    # mark initial invalids (snapping failed)
    invalid_nodes = (pu_nodes == -1) | (do_nodes == -1)

    # vectorized SCC mismatch mask
    scc_mismatch = np.zeros(len(pu_nodes), dtype=bool)
    valid_pair_mask = ~invalid_nodes
    if np.any(valid_pair_mask):
        idxs = np.where(valid_pair_mask)[0]
        idx_list = [int(i) for i in idxs.tolist()]
        # vectorize scc check
        scc_mismatch[idxs] = np.fromiter(
            (not scc_equal(int(pu_nodes[i]), int(do_nodes[i])) for i in idx_list),
            dtype=bool,
            count=len(idx_list),
        )

    # we need resampling where either nodes invalid OR SCC mismatch
    need_resample = invalid_nodes | scc_mismatch

    # --- Batched resampling loop ---
    path_resample_success = 0
    path_resample_fail = 0

    # Helper to (re)generate coords & nodes for a set of indices in one go
    def batch_resample(idxs: np.ndarray) -> None:
        nonlocal pu_x, pu_y, do_x, do_y, pu_nodes, do_nodes
        if idxs.size == 0:
            return
        # Resample using one seed per attempt (keeps results stable per pass)
        seed = int(
            np.random.default_rng(PICKUP_SEED + DROPOFF_SEED).integers(0, 2**32 - 1)
        )
        # Generate new coords for those zones in one call each
        new_pu_x, new_pu_y = gen_coords_for_zones(
            pickup_zones[idxs], sampler, zone_geoms, seed
        )
        new_do_x, new_do_y = gen_coords_for_zones(
            dropoff_zones[idxs], sampler, zone_geoms, seed ^ 0x9E3779B9
        )

        # clip to hull
        inside_pu = mask_points_inside_polygon(new_pu_x, new_pu_y, graph_hull)
        inside_do = mask_points_inside_polygon(new_do_x, new_do_y, graph_hull)

        # snap (invalid → -1)
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

    # iterate limited resample attempts
    fail_mask = need_resample.copy()
    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        idxs = np.where(fail_mask)[0]
        if idxs.size == 0:
            break
        batch_resample(idxs)

        # recompute invalid and scc mismatch only for these idxs
        still_invalid_nodes = (pu_nodes[idxs] == -1) | (do_nodes[idxs] == -1)
        still_scc_mismatch = np.zeros(idxs.size, dtype=bool)
        ok_nodes = ~still_invalid_nodes
        if np.any(ok_nodes):
            sub = idxs[ok_nodes]
            sub_list = [int(i) for i in sub.tolist()]
            still_scc_mismatch[ok_nodes] = np.fromiter(
                (not scc_equal(int(pu_nodes[i]), int(do_nodes[i])) for i in sub_list),
                dtype=bool,
                count=len(sub_list),
            )

        # trips fixed this round:
        fixed = ~(still_invalid_nodes | still_scc_mismatch)
        n_fixed = int(np.sum(fixed))
        if n_fixed > 0:
            path_resample_success += n_fixed
            fail_mask[idxs[fixed]] = False  # mark solved

        # remaining stay True and will attempt again

    # mark permanent failures
    if np.any(fail_mask):
        path_resample_fail += int(np.sum(fail_mask))
        pu_nodes[fail_mask] = -1
        do_nodes[fail_mask] = -1

    # --- output columns ---
    transformer = Transformer.from_crs(
        graph_proj.graph["crs"], "EPSG:4326", always_xy=True
    )
    pu_lon, pu_lat = to_lonlat(pu_x, pu_y, transformer)
    do_lon, do_lat = to_lonlat(do_x, do_y, transformer)

    trips = trips.with_columns(
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

    invalid_mask = trips["orig_node"].is_null() | trips["dest_node"].is_null()
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        logger.info(
            "Dropping %d trips in %s that could not be connected",
            invalid_count,
            input_path.name,
        )
        trips = trips.filter(~invalid_mask)

    final_count = trips.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trips = trips.sort("pickup_datetime")
    trips.drop("row_id").write_parquet(output_path)

    summary = format_log_block(
        "\n".join(
            [
                f"{input_path.name} summary:",
                f"  total trips: {total_before}",
                f"  candidates with Manhattan zone: {candidate_count}",
                f"  pickup missing: {pickup_missing}",
                f"  dropoff missing: {dropoff_missing}",
                f"  datetime missing: {datetime_missing}",
                f"  after zone filter: {after_zone}",
                f"  dropped during snapping: {invalid_count}",
                f"  path resample success: {path_resample_success}",
                f"  path resample failures: {path_resample_fail}",
                f"  written: {final_count} → {output_path.name}",
            ]
        )
    )
    logger.info(summary)
    return TripStats(
        total_before=total_before,
        candidate_count=candidate_count,
        pickup_missing=pickup_missing,
        dropoff_missing=dropoff_missing,
        datetime_missing=datetime_missing,
        after_zone=after_zone,
        removed_after_filter=removed_after_candidates,
        written=final_count,
        dropped_snapping=invalid_count,
        path_resample_success=path_resample_success,
        path_resample_fail=path_resample_fail,
    )
