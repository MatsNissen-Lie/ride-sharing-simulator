"""Faster Manhattan routing leveraging caching and precomputed coordinates.

This script follows the recommendations in `src/state_representation/efficiency.md`
by:
    * enabling the OSMnx HTTP cache
    * persisting the downloaded street graph to GraphML for instant reloads
    * caching geocoded landmark coordinates to avoid repeated API calls
"""

from datetime import datetime
import networkx as nx
import osmnx as ox

from config import BASE_DIR, get_logger


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


def main() -> None:
    G = load_graph()
    G = ensure_travel_times(G)


if __name__ == "__main__":
    main()
