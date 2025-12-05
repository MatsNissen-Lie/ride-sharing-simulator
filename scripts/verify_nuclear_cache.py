import sys
from pathlib import Path
import numpy as np
import networkx as nx

# add workspace to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.components.network_service import NetworkXService
from src.database.simple_db import DuckDBPathCache
from config import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Verifying Nuclear Cache...")

    # Mock Graph
    G = nx.MultiDiGraph()

    # Mock DB
    class MockDB:
        def get_travel_time_matrix(self, sources, targets):
            # Return a matrix of 10.0s
            arr = np.full((len(sources), len(targets)), 10.0, dtype=np.float32)
            np.fill_diagonal(arr, 0.0)
            return arr

        def get_simple(self, o, d):
            # Simulate DB hit for (0, 1) if not prefetched
            if o == 0 and d == 1:
                return (20.0, 100.0)  # time, dist
            return None

    service = NetworkXService(G, max_node_id=100)
    service._db = MockDB()

    nodes = [0, 1, 2]

    # Test 1: Access before prefetch -> Should use L2 fallback
    logger.info("Test 1: Accessing (0, 1) before prefetch (L2 Fallback)...")
    path = service.get_shortest_path(0, 1)
    if path and path.travel_time == 20.0:
        logger.info("Test 1 PASSED: L2 Fallback worked (20.0s).")
    else:
        logger.error(f"Test 1 FAILED: Got {path}")

    # Test 2: Prefetch
    logger.info("Prefetching nodes [0, 1, 2]...")
    service.prefetch_all_pairs(nodes)

    # Test 3: Access after prefetch -> Should use cached value (10.0s from matrix)
    # Note: prefetch overwrites whatever was there.
    path = service.get_shortest_path(0, 1)
    if path and path.travel_time == 10.0:
        logger.info("Test 3 PASSED: Prefetch worked (10.0s).")
    else:
        logger.error(f"Test 3 FAILED: Got {path}")

    # Test 4: Subset extraction
    subset = service.get_travel_times_l1_cache_subset(nodes)
    if subset.get((0, 1)) == 10.0 and subset.get((0, 0)) == 0.0:
        logger.info("Test 4 PASSED: Subset extraction correct.")
    else:
        logger.error(f"Test 4 FAILED: Subset {subset}")

    # Test 5: Clear cache
    service.clear_l1_cache()
    # After clear, (0, 1) should fallback to L2 again (20.0s)
    path = service.get_shortest_path(0, 1)
    if path and path.travel_time == 20.0:
        logger.info("Test 5 PASSED: Cache cleared, fallback worked.")
    else:
        logger.error(f"Test 5 FAILED: Got {path}")


if __name__ == "__main__":
    main()
