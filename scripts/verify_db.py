import random
import networkx as nx
import sys
from pathlib import Path
from typing import Tuple

# Add workspace to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import DATA_DIR
from src.database.simple_db import DuckDBPathCache
from src.state_representation.manhatten_fast import load_graph
from src.components.network.network_service import NetworkXService
from config import get_logger

logger = get_logger(__name__)

def verify_db_vs_nx():
    logger.info("Loading graph...")
    G = load_graph()
    
    # Initialize Service and Cache
    service = NetworkXService(G)
    cache = DuckDBPathCache()
    con = cache.con
    
    # Get all pairs currently in the DB
    logger.info("Fetching sample pairs from DB...")
    # We fetch a random sample of 100 pairs that have a valid path (time_s >= 0)
    rows = con.execute("""
        SELECT o, d, time_s, dist_m 
        FROM time_simple 
        WHERE time_s >= 0 
        USING SAMPLE 100
    """).fetchall()
    
    if not rows:
        logger.warning("Database is empty or has no valid paths!")
        return

    logger.info(f"Verifying {len(rows)} pairs...")
    
    mismatches = 0
    errors = 0
    
    for row in rows:
        o, d, db_time, db_dist = row
        o, d = int(o), int(d)
        
        # 1. Compute using NetworkX directly (bypassing cache logic inside service if we wanted, 
        #    but service.get_shortest_path uses cache. 
        #    So we must compute MANUALLY using NX to be sure we are comparing against truth.)
        
        try:
            # We use the same logic as the service's fallback:
            # nx.shortest_path uses Dijkstra for weighted graphs
            path_nodes = nx.shortest_path(G, o, d, weight="travel_time")
            
            # Re-calculate time/dist from the path
            nx_time = 0.0
            nx_dist = 0.0
            for u, v in zip(path_nodes, path_nodes[1:]):
                # Handle MultiDiGraph: pick fastest edge
                edge_data = min(G[u][v].values(), key=lambda x: x["travel_time"])
                nx_time += float(edge_data["travel_time"])
                nx_dist += float(edge_data.get("length", 0.0))
                
        except nx.NetworkXNoPath:
            # DB says there is a path, but NX says no path?
            logger.error(f"MISMATCH: DB has path for {o}->{d}, but NX found none!")
            mismatches += 1
            continue
        except Exception as e:
            logger.error(f"Error computing {o}->{d}: {e}")
            errors += 1
            continue

        # Compare (allowing small float tolerance)
        # Tolerance: 1e-4 seconds, 1e-2 meters
        time_diff = abs(db_time - nx_time)
        dist_diff = abs(db_dist - nx_dist)
        
        if time_diff > 1e-4 or dist_diff > 1e-2:
            logger.error(
                f"MISMATCH {o}->{d}: "
                f"DB(t={db_time:.4f}, d={db_dist:.2f}) vs "
                f"NX(t={nx_time:.4f}, d={nx_dist:.2f})"
            )
            mismatches += 1
        else:
            # logger.info(f"OK {o}->{d}")
            pass

    if mismatches == 0 and errors == 0:
        logger.info("SUCCESS: All sampled pairs match NetworkX output!")
    else:
        logger.error(f"FAILED: Found {mismatches} mismatches and {errors} errors.")

if __name__ == "__main__":
    verify_db_vs_nx()
