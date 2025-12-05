import sys
from pathlib import Path
import networkx as nx
import numpy as np

# add workspace to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.simple_db import DuckDBPathCache
from src.state_representation.manhatten_fast import load_graph
from config import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Debugging Pair 3530 -> 2202...")
    
    o, d = 3530, 2202
    
    # 1. Check Database
    logger.info("Checking Database...")
    db = DuckDBPathCache(read_only=True)
    
    # Simple lookup
    hit = db.get_simple(o, d)
    if hit:
        logger.info(f"DB get_simple({o}, {d}): time={hit[0]}, dist={hit[1]}")
    else:
        logger.warning(f"DB get_simple({o}, {d}): Not found")
        
    # Matrix lookup
    matrix = db.get_travel_time_matrix([o], [d])
    val = matrix[0, 0]
    logger.info(f"DB get_travel_time_matrix: {val}")
    
    # 2. Check NetworkX
    logger.info("Checking NetworkX (OSM Graph)...")
    try:
        G = load_graph()
        logger.info(f"Graph loaded. Nodes: {G.number_of_nodes()}")
        
        if o not in G:
            logger.error(f"Origin {o} not in Graph!")
        if d not in G:
            logger.error(f"Destination {d} not in Graph!")
            
        if o in G and d in G:
            try:
                nx_time = nx.shortest_path_length(G, o, d, weight="travel_time")
                logger.info(f"NetworkX shortest_path_length: {nx_time}")
            except nx.NetworkXNoPath:
                logger.warning("NetworkX: No path found.")
                
            # Check distance too if possible
            try:
                # This is an approximation if we just sum lengths of the time-shortest path
                path = nx.shortest_path(G, o, d, weight="travel_time")
                dist = 0.0
                for u, v in zip(path, path[1:]):
                    edge_data = min(G[u][v].values(), key=lambda x: x.get("travel_time", float("inf")))
                    dist += float(edge_data.get("length", 0.0))
                logger.info(f"NetworkX path distance (sum of lengths): {dist}")
            except nx.NetworkXNoPath:
                pass

    except Exception as e:
        logger.exception(f"Error loading graph or computing path: {e}")

if __name__ == "__main__":
    main()
