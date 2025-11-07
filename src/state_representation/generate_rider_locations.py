"""Generate graph-constrained pickup and dropoff nodes for NYC FHV trips."""

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

from config import BASE_DIR, format_log_block, get_logger
from src.state_representation.manhatten_fast import ensure_travel_times, load_graph

logger = get_logger(__name__)

RIDER_DATA_PATH = BASE_DIR / "data" / "riders" / "fhv_tripdata_2025-01.parquet"
OUTPUT_DATA_PATH = (
    BASE_DIR / "data" / "riders" / "fhv_tripdata_2025-01_enriched.parquet"
)
TAXI_ZONES_PATH = BASE_DIR / "data" / "zones" / "taxi_zones.shp"

SAMPLE_BATCH_SIZE = 20_000
PICKUP_SEED = 7
DROPOFF_SEED = 17


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

    trips = trips.drop_nulls(
        subset=["PUlocationID", "DOlocationID", "pickup_datetime"]
    ).rename(
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


def main() -> pl.DataFrame:
    logger.info("Preparing Manhattan graph context")
    G = ensure_travel_times(load_graph())
    G_proj = ox.project_graph(G)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)
    logger.info(
        "Graph projected: %d nodes, %d edges",
        G_proj.number_of_nodes(),
        G_proj.number_of_edges(),
    )

    zones = load_zones(TAXI_ZONES_PATH, G_proj.graph["crs"])
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

    trips = load_trips(RIDER_DATA_PATH)
    before_filter = trips.height
    valid_list = sorted(valid_zone_ids)
    trips = trips.filter(
        pl.col("pickup_zone").is_in(valid_list)
        & pl.col("dropoff_zone").is_in(valid_list)
    ).with_row_index(name="row_id")
    logger.info(
        "Trips restricted to Manhattan zones: %d → %d",
        before_filter,
        trips.height,
    )

    sampler = ZoneSampler(edges_gdf)

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

    pu_nodes = snap_points_to_nodes(G_proj, pu_x, pu_y)
    do_nodes = snap_points_to_nodes(G_proj, do_x, do_y)

    transformer = Transformer.from_crs(G_proj.graph["crs"], "EPSG:4326", always_xy=True)
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
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.info(
            "Dropping %d trips that could not be snapped to the graph",
            int(invalid_count),
        )
        trips = trips.filter(~invalid_mask)

    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    trips.drop("row_id").write_parquet(OUTPUT_DATA_PATH)

    summary = format_log_block(
        "\n".join(
            [
                "Rider location generation complete:",
                f"Saved {trips.height} trips to {OUTPUT_DATA_PATH}",
                f"Same-zones trips: {trips.filter(pl.col('pickup_zone') == pl.col('dropoff_zone')).height}",
                f"Columns available: {', '.join(trips.columns)}",
            ]
        )
    )
    logger.info(summary)
    return trips


if __name__ == "__main__":
    main()
