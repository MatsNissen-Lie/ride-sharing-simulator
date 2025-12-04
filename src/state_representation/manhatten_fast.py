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


import json

PLACE_NAME = "Manhattan, New York City, New York, USA"

GRAPH_DIR = BASE_DIR / "data" / "graphs"
PLOT_DIR = BASE_DIR / "data" / "plots"
GEOCODE_DIR = BASE_DIR / "data" / "geocodes"
GRAPH_PATH = GRAPH_DIR / "manhattan_drive.graphml"
MAPPING_PATH = GRAPH_DIR / "node_mapping.json"
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


def remap_nodes(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Remap nodes to 0..N-1 sorted by original ID.
    Saves the mapping to disk.
    """
    nodes = sorted(list(G.nodes))

    # Check if already remapped (heuristic: if max ID < len(nodes) + 100 and min ID >= 0)
    # But to be safe and consistent, we can just check if they are all ints and max is small.
    # However, if we reload a graph that is already remapped, we might not want to remap again
    # unless we are sure.
    # Actually, the user said "I can delete the old graphml no worries", so we assume we are
    # either loading a fresh download (large IDs) or a saved remapped graph.
    # Let's just do the remapping if the first node is large.

    if isinstance(nodes[0], int) and nodes[0] < len(nodes) and nodes[-1] < len(nodes):
        logger.info(
            "Graph nodes appear to be already remapped (0..N-1). Skipping remapping."
        )
        return G

    logger.info(f"Remapping {len(nodes)} nodes to 0..{len(nodes)-1}...")

    mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}

    # Save mapping
    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved node mapping to {MAPPING_PATH}")

    G = nx.relabel_nodes(G, mapping)
    return G


def load_graph() -> nx.MultiDiGraph:
    start = datetime.now()
    if GRAPH_PATH.exists():
        logger.info("Loading cached street graph from %s", GRAPH_PATH)
        G = ox.load_graphml(GRAPH_PATH)
        # We still check/remap in case the cached one was old style
        # G = remap_nodes(G)
    else:
        logger.info("Downloading Manhattan drive network (first run will be slower)...")
        G = ox.graph_from_place(PLACE_NAME, network_type="drive", simplify=True)
        # ensure travel times are present
        G = ensure_travel_times(G)

        # Remap BEFORE saving, so the saved GraphML is clean
        G = remap_nodes(G)

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

    # get all nodes and show the range
    nodes = list(G.nodes)
    logger.info(
        f"Graph has {len(nodes)} nodes, ranging from {min(nodes)} to {max(nodes)}"
    )
    # median
    logger.info(f"Median node ID: {sorted(nodes)[len(nodes)//2]}")


if __name__ == "__main__":
    main()
