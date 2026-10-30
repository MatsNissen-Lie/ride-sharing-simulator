import time
import multiprocessing
import networkx as nx
import sys
from pathlib import Path
import polars as pl  # Using Polars for fast DuckDB ingestion

# Add workspace to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import DATA_DIR
from src.database.simple_db import DuckDBPathCache
from src.inputs.manhattan_graph import load_graph
from config import get_logger

logger = get_logger(__name__)

# Global variable for workers
H_local: nx.DiGraph = None  # type: ignore


def init_worker(graph_data: nx.DiGraph):
    """Initialize worker with the SCC subgraph."""
    global H_local
    H_local = graph_data


def compute_one_source(o: int) -> dict[str, list]:
    """
    Returns columnar data: {'o': [...], 'd': [...], 'time_s': [...], 'dist_m': [...]}
    """
    # 1. Dijkstra on the Subgraph (Much faster than full graph)
    try:
        preds, times = nx.dijkstra_predecessor_and_distance(
            H_local, o, weight="travel_time"
        )
    except Exception:
        # Handle isolated nodes or errors gracefully
        return {}

    # 2. Topological sort for distance reconstruction
    # Sorting by time ensures we process parents before children
    sorted_nodes = sorted(times.keys(), key=lambda k: times[k])

    dists: dict[int, float] = {o: 0.0}

    # Columnar storage for DataFrame creation
    out_o, out_d, out_t, out_dist = [], [], [], []

    # Add self-loop explicitly
    out_o.append(o)
    out_d.append(o)
    out_t.append(0.0)
    out_dist.append(0.0)

    for v in sorted_nodes:
        if v == o:
            continue

        t = times[v]
        # Retrieve parent. preds[v] is a list of all parents with equal cost.
        # We take the first one.
        if not preds[v]:
            continue

        u = preds[v][0]

        # Lookup edge length.
        # Since H_local is a subgraph, these edges exist.
        d_u_v = H_local[u][v]["length"]
        d_v = dists[u] + d_u_v
        dists[v] = d_v

        out_o.append(o)
        out_d.append(v)
        out_t.append(t)
        out_dist.append(d_v)

    return {"o": out_o, "d": out_d, "time_s": out_t, "dist_m": out_dist}


def project_to_minweight_digraph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Keep only the fastest edge per (u,v)."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for u, v, data in G.edges(keys=False, data=True):
        t = data.get("travel_time")
        if t is None:
            continue
        d = float(data.get("length", 0.0))
        t = float(t)

        cur = H.get_edge_data(u, v)
        if cur is None or t < cur["travel_time"]:
            H.add_edge(u, v, travel_time=t, length=d)
    return H


def main():
    # ==== CONFIG ====
    TARGET_SCC_ID = 32
    BATCH_SIZE_INSERT = 200_000  # Increased for Polars speed
    WORKERS = max(1, multiprocessing.cpu_count() - 1)
    # ================

    # Force Fork on Linux/Mac to enable zero-copy shared memory of the graph
    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("fork")
        except RuntimeError:
            pass  # Context already set

    (DATA_DIR / "duckDB").mkdir(parents=True, exist_ok=True)

    logger.info("Loading graph...")
    G = load_graph()

    logger.info("Projecting graph...")
    H = project_to_minweight_digraph(G)

    logger.info(f"Extracting SCC {TARGET_SCC_ID}...")

    # Identify SCC nodes
    # We use the same logic as before: enumerate and find the target ID
    # Note: SCC order in NX is arbitrary unless sorted, but if we assume
    # the ID comes from a previous deterministic run or we just want the largest one...
    # For now, we stick to the previous logic of "ID 32" from the enumeration.
    # WARNING: If the graph changed, ID 32 might be different.

    node2scc = {}
    for cid, comp in enumerate(nx.strongly_connected_components(H)):
        for n in comp:
            node2scc[n] = cid

    scc_nodes = [n for n, cid in node2scc.items() if cid == TARGET_SCC_ID]

    if not scc_nodes:
        logger.error(f"No nodes in SCC {TARGET_SCC_ID}")
        return

    # === OPTIMIZATION 1: SUBGRAPH ===
    # Only pass the relevant component to workers.
    # copy() is important so it's a contiguous separate object in memory
    H_scc = H.subgraph(scc_nodes).copy()

    logger.info(
        f"SCC {TARGET_SCC_ID}: {len(scc_nodes)} nodes, {H_scc.number_of_edges()} edges."
    )

    cache = DuckDBPathCache(read_only=False)
    con = cache.con

    # Check existing work using GROUP BY
    logger.info("Checking existing sources...")

    # Fetch all sources that have ANY entries.
    # Ideally we check count, but for speed we can just check existence if we assume atomic commits.
    # Or we check count >= len(scc_nodes) - 1

    # Using Polars to read the check is also fast
    # But standard SQL is fine here.
    rows = con.execute("SELECT o, COUNT(*) FROM time_simple GROUP BY o").fetchall()
    existing_counts = {r[0]: r[1] for r in rows}

    expected = len(scc_nodes)  # - 1  # Rough check

    sources_to_process = []
    for n in scc_nodes:
        if existing_counts.get(n, 0) < expected:
            sources_to_process.append(n)

    logger.info(f"Sources to compute: {len(sources_to_process)}")

    if not sources_to_process:
        logger.info("All sources completed.")
        return

    # Start Processing
    t0 = time.time()
    total_inserted = 0

    # Buffers for columnar storage
    col_o, col_d, col_t, col_dist = [], [], [], []

    with multiprocessing.Pool(
        processes=WORKERS, initializer=init_worker, initargs=(H_scc,)
    ) as pool:
        iterator = pool.imap_unordered(
            compute_one_source, sources_to_process, chunksize=20
        )

        for res in iterator:
            if not res:
                continue

            # Extend columns directly (fast)
            col_o.extend(res["o"])
            col_d.extend(res["d"])
            col_t.extend(res["time_s"])
            col_dist.extend(res["dist_m"])

            if len(col_o) >= BATCH_SIZE_INSERT:
                # === OPTIMIZATION 2: POLARS INSERT ===
                df_batch = pl.DataFrame(
                    {
                        "o": col_o,
                        "d": col_d,
                        "time_s": col_t,
                        "dist_m": col_dist,
                    }
                )

                # DuckDB can query Polars DF directly
                con.register("df_batch_view", df_batch)
                con.execute(
                    "INSERT OR REPLACE INTO time_simple SELECT * FROM df_batch_view"
                )
                con.unregister("df_batch_view")

                total_inserted += len(col_o)
                col_o.clear()
                col_d.clear()
                col_t.clear()
                col_dist.clear()

                elapsed = time.time() - t0
                rate = total_inserted / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Inserted {total_inserted:,} rows. Rate: {rate:.0f} rows/s"
                )

    # Final flush
    if col_o:
        df_batch = pl.DataFrame(
            {"o": col_o, "d": col_d, "time_s": col_t, "dist_m": col_dist}
        )
        con.execute("INSERT OR REPLACE INTO time_simple SELECT * FROM df_batch")
        total_inserted += len(col_o)

    logger.info(f"Finished. Total: {total_inserted:,} in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
