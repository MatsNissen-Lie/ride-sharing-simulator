"""Faster Manhattan routing leveraging caching and precomputed coordinates.

This script follows the recommendations in `src/state_representation/efficiency.md`
by:
    * enabling the OSMnx HTTP cache
    * persisting the downloaded street graph to GraphML for instant reloads
    * caching geocoded landmark coordinates to avoid repeated API calls
"""

import json
from datetime import datetime
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from config import BASE_DIR, format_log_block, get_logger


PLACE_NAME = "Manhattan, New York City, New York, USA"

GRAPH_DIR = BASE_DIR / "data" / "graphs"
PLOT_DIR = BASE_DIR / "data" / "plots"
GEOCODE_DIR = BASE_DIR / "data" / "geocodes"
GRAPH_PATH = GRAPH_DIR / "manhattan_drive.graphml"
ROUTE_PLOT_PATH = PLOT_DIR / "manhattan_route.png"
GEOCODE_PATH = GEOCODE_DIR / "manhattan_landmarks.json"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
GEOCODE_DIR.mkdir(parents=True, exist_ok=True)

ADDRESS_BOOK = {
    "origin": "Times Square, Manhattan, NY, USA",
    # "destination": "Wall Street, Manhattan, NY, USA",
    "destination": "Trump Tower, Manhattan, NY, USA",
}

FALLBACK_SPEED_KPH = 20.0
FALLBACK_SPEED_MPS = FALLBACK_SPEED_KPH * 1000 / 3600


ox.settings.use_cache = True


logger = get_logger(__name__)


def load_graph() -> nx.MultiDiGraph:
    start = datetime.now()
    if GRAPH_PATH.exists() and True:
        logger.info("Loading cached street graph from %s", GRAPH_PATH)
        G = ox.load_graphml(GRAPH_PATH)
    else:
        logger.info("Downloading Manhattan drive network (first run will be slower)...")
        G = ox.graph_from_place(PLACE_NAME, network_type="drive", simplify=True)
        # ensure travel times are present
        G = ensure_travel_times(G)
        ox.save_graphml(G, GRAPH_PATH)
        logger.info("Saved graph to %s", GRAPH_PATH)
    elapsed = (datetime.now() - start).total_seconds()
    logger.info("Graph ready in %.2f seconds", elapsed)
    return G


def ensure_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Guarantee every edge exposes `speed_kph` and `travel_time` attributes."""

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    #     osmnx.routing.add_edge_travel_times(G)
    # Add edge travel time (seconds) to graph as new travel_time edge attributes.

    # Calculates free-flow travel time along each edge, based on length and speed_kph attributes. Note: run add_edge_speeds first to generate the speed_kph attribute. All edges must have length and speed_kph attributes and all their values must be non-null.

    # Parameters:
    # G (MultiDiGraph) – Input graph.

    # Returns:
    # G – Graph with travel_time attributes on all edges.

    # Return type:
    # networkx.MultiDiGraph
    G = ox.routing.add_edge_travel_times(G)

    patched_edges = 0
    for _, _, data in G.edges(data=True):
        length = data.get("length")
        travel_time = data.get("travel_time")
        if length and (travel_time is None or travel_time <= 0):
            data["travel_time"] = length / FALLBACK_SPEED_MPS
            data.setdefault("speed_kph", FALLBACK_SPEED_KPH)
            patched_edges += 1

    logger.info("Applied fallback travel time to %d edges", patched_edges)

    return G


def _load_geocode_store() -> Dict[str, Dict[str, float]]:
    if not GEOCODE_PATH.exists():
        return {}

    raw: Dict[str, Dict[str, float]] = json.loads(GEOCODE_PATH.read_text())

    # Migrate any legacy label-based entries ("origin" -> actual address).
    migrated = False
    for label, address in ADDRESS_BOOK.items():
        if label in raw and address not in raw:
            raw[address] = raw[label]
            migrated = True
        if label in raw:
            raw.pop(label)

    if migrated:
        GEOCODE_PATH.write_text(json.dumps(raw, indent=2, sort_keys=True))

    return {
        key: value
        for key, value in raw.items()
        if isinstance(value, dict) and "lat" in value and "lon" in value
    }


def load_landmark_coordinates(
    addresses: Iterable[str],
) -> Dict[str, Tuple[float, float]]:
    store = _load_geocode_store()
    updated = False
    coordinates: Dict[str, Tuple[float, float]] = {}

    for address in addresses:
        entry = store.get(address)
        if not entry or "lat" not in entry or "lon" not in entry:
            logger.info("Geocoding address: %s", address)
            lat, lon = ox.geocode(address)
            entry = {"lat": lat, "lon": lon}
            store[address] = entry
            updated = True

        coordinates[address] = (entry["lat"], entry["lon"])

    if updated:
        GEOCODE_PATH.write_text(json.dumps(store, indent=2, sort_keys=True))

    return coordinates


def describe_route(G: nx.MultiDiGraph) -> None:
    coordinates = load_landmark_coordinates(ADDRESS_BOOK.values())
    origin_address = ADDRESS_BOOK["origin"]
    dest_address = ADDRESS_BOOK["destination"]
    origin_lat, origin_lon = coordinates[origin_address]
    dest_lat, dest_lon = coordinates[dest_address]

    # log the time to find nearest nodes
    start = datetime.now()
    origin_node = ox.nearest_nodes(G, origin_lon, origin_lat)
    dest_node = ox.nearest_nodes(G, dest_lon, dest_lat)
    elapsed_milliseconds = (datetime.now() - start).microseconds / 1000
    logger.info(
        "Found nearest graph nodes to origin/destination in %.2f ms",
        elapsed_milliseconds,
    )

    # log each time we compute a route
    start = datetime.now()
    route_nodes = nx.shortest_path(G, origin_node, dest_node, weight="travel_time")
    logger.info(
        "Computed shortest path from origin to destination in %.2f ms",
        (datetime.now() - start).microseconds / 1000,
    )

    start = datetime.now()
    route_time_seconds = nx.shortest_path_length(
        G, origin_node, dest_node, weight="travel_time"
    )
    logger.info(
        "Computed route travel time in %.2f ms",
        (datetime.now() - start).microseconds / 1000,
    )

    start = datetime.now()
    route_distance_meters = nx.shortest_path_length(
        G, origin_node, dest_node, weight="length"
    )
    logger.info(
        "Computed route distance in %.2f ms",
        (datetime.now() - start).microseconds / 1000,
    )

    if route_time_seconds > 0:
        avg_speed_kph = (route_distance_meters / 1000) / (route_time_seconds / 3600)
    else:
        avg_speed_kph = 0.0

    summary = "\n".join(
        [
            "Route overview:",
            f"Origin ({origin_address}) → graph node {origin_node}",
            f"Destination ({dest_address}) → graph node {dest_node}",
            f"Shortest route has {len(route_nodes) - 1} road segments and covers {route_distance_meters:.1f} meters",
            f"Estimated travel time: {route_time_seconds:.1f} seconds ({route_time_seconds/60:.1f} minutes)",
            f"Average speed along route: {avg_speed_kph:.1f} km/h",
        ]
    )
    logger.info(format_log_block(summary))
    logger.info("Route map saved to %s", ROUTE_PLOT_PATH)


def main() -> None:
    start = datetime.now()
    G = load_graph()
    G = ensure_travel_times(G)
    describe_route(G)
    total_elapsed = (datetime.now() - start).total_seconds()
    logger.info("Total script runtime: %.2f seconds", total_elapsed)


if __name__ == "__main__":
    main()
