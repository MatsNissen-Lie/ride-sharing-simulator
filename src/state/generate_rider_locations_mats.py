import osmnx as ox
import geopandas as gpd
import numpy as np
from config import BASE_DIR
from src.state_representation.manhatten_fast import load_graph

# RIDER SAMPLE

RIDER_DATA_PATH = BASE_DIR / "data/riders/fhv_tripdata_2025-01.parquet"

G = load_graph()

# 0) Inputs you already have:
#   G : Manhattan drive graph (MultiDiGraph) with 'travel_time' added
#   trips : DataFrame with PULocationID, DOLocationID, and timestamps
#   zones : GeoDataFrame of TLC Taxi Zones (columns: LocationID, geometry)

# 1) Prepare per-zone subgraphs for length-weighted street sampling
#    (project + undirect for accurate length and uniform sampling along edges)
import osmnx as ox
import geopandas as gpd
import numpy as np

# Assumptions:
#   G     : projected Manhattan graph OR unprojected; we project below
#   trips : DataFrame with PULocationID, DOLocationID
#   zones : GeoDataFrame with columns [LocationID, geometry]

# 1) Project + undirect once
G_proj = ox.project_graph(G)
G_undir = ox.utils_graph.get_undirected(G_proj)

# 2) Align zones CRS with graph CRS
zones = zones.to_crs(G_undir.graph["crs"])
zone_geoms = dict(zip(zones.LocationID, zones.geometry))

# 3) Precompute nodes/edges GeoDataFrames ONCE (docs: ox.convert.graph_to_gdfs)
nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(G_undir)


def sample_points_in_zone(zone_id: int, n: int):
    """Sample n points along the road network inside a Taxi Zone (length-weighted)."""
    poly = zone_geoms.get(zone_id)
    if poly is None or poly.is_empty:
        return None

    # Filter edges that intersect the zone polygon (fast spatial filter)
    sel = edges_gdf.intersects(poly)  # polygon already in same CRS
    if not sel.any():
        return None

    edges_sub = edges_gdf.loc[sel]
    node_ids = set(edges_sub.index.get_level_values(0)) | set(
        edges_sub.index.get_level_values(1)
    )
    nodes_sub = nodes_gdf.loc[nodes_gdf.index.intersection(node_ids)]

    # Rebuild a tiny subgraph from GDFs (docs: ox.convert.graph_from_gdfs)
    sub = ox.convert.graph_from_gdfs(nodes_sub, edges_sub)

    # Length-weighted sampling along edges (docs: utils_geo.sample_points)
    return ox.utils_geo.sample_points(sub, n)  # GeoSeries in projected coords


def gen_coords_for_trips(trips, col, batch=10000, seed=7):
    rng = np.random.default_rng(seed)
    out_x, out_y = np.empty(len(trips)), np.empty(len(trips))
    i = 0
    while i < len(trips):
        j = min(i + batch, len(trips))
        chunk = trips.iloc[i:j]
        for zid, idx in chunk.groupby(col).groups.items():
            n = len(idx)
            pts = sample_points_in_zone(zid, n)
            if pts is None or len(pts) < n:
                poly = zone_geoms.get(zid)
                if poly is None or poly.is_empty:
                    out_x[idx] = np.nan
                    out_y[idx] = np.nan
                    continue
                # Fallback: random points in polygon (projected CRS)
                xs, ys = [], []
                for _ in range(n):
                    # uniform-ish interior sample via representative_point jitter on exterior as simple fallback
                    p = poly.representative_point()
                    xs.append(p.x)
                    ys.append(p.y)
            else:
                xs = pts.geometry.x.values
                ys = pts.geometry.y.values
            out_x[idx] = np.asarray(xs)
            out_y[idx] = np.asarray(ys)
        i = j
    return out_x, out_y


# Generate pickup/dropoff projected coordinates
trips = trips.copy()
trips["pu_x"], trips["pu_y"] = gen_coords_for_trips(trips, "PULocationID")
trips["do_x"], trips["do_y"] = gen_coords_for_trips(trips, "DOLocationID")

# 4) Snap directly to NEAREST NODES (vectorized; docs: ox.distance.nearest_nodes)
#    (this avoids the "always pick u" shortcut)
trips["orig"] = ox.distance.nearest_nodes(
    G_undir, X=trips["pu_x"].values, Y=trips["pu_y"].values
)
trips["dest"] = ox.distance.nearest_nodes(
    G_undir, X=trips["do_x"].values, Y=trips["do_y"].values
)
