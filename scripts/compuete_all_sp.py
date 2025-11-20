# scripts/precompute_scc32_resume.py

from typing import Dict, Iterable, List, Tuple, Set, cast
import time
import random
import networkx as nx

# add workspace to sys.path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import DATA_DIR
from src.database.simple_db import DuckDBPathCache  # << NEW DB
from src.state_representation.manhatten_fast import load_graph
from config import get_logger

logger = get_logger(__name__)

# ---- helpers -----------------------------------------------------------------


def compute_node2scc(graph: nx.MultiDiGraph) -> Dict[int, int]:
    node2scc: Dict[int, int] = {}
    for cid, comp in enumerate(nx.strongly_connected_components(graph)):
        for n in comp:
            node2scc[int(n)] = cid
    return node2scc


def project_to_minweight_digraph(
    G: nx.MultiDiGraph,
    *,
    time_attr: str = "travel_time",
    dist_attr: str = "length",
) -> nx.DiGraph:
    """Keep only the fastest edge per (u,v) for speedy bidirectional Dijkstra."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for u, v, data in G.edges(keys=False, data=True):
        t = data.get(time_attr)
        if t is None:
            continue
        d = float(data.get(dist_attr, 0.0))
        t = float(t)
        cur = H.get_edge_data(u, v)
        if cur is None or t < cur["travel_time"]:
            H.add_edge(u, v, travel_time=t, length=d)
    return H


def accumulate_on_path(
    H: nx.DiGraph, nodes: Iterable[int]
) -> Tuple[float, float, List[float]]:
    total_t = 0.0
    total_d = 0.0
    edge_times: List[float] = []
    it = iter(nodes)
    try:
        u = next(it)
    except StopIteration:
        return 0.0, 0.0, edge_times
    for v in it:
        ed = H[u][v]
        t = float(ed["travel_time"])
        total_t += t
        total_d += float(ed.get("length", 0.0))
        edge_times.append(t)
        u = v
    return total_t, total_d, edge_times


# ---- core --------------------------------------------------------------------


def precompute_scc_with_bidij_resumable(
    G: nx.MultiDiGraph,
    target_scc_id: int,
    cache: DuckDBPathCache,
    *,
    write_extended: bool = False,
    log_every_sources: int = 50,
    batch_size_sources: int = 200,  # number of sources per logged batch
) -> None:
    """
    Resumable APSP for a given SCC using bidirectional Dijkstra on a projected DiGraph.
    - Skips (o,d) pairs already present in DuckDB (time_simple)
    - SKIPS WHOLE BATCHES that are fully complete
    - Logs per-batch timing + overall progress (done / total, %)
    - Optionally also ensures path_extended is filled

    Requires edge attrs: 'length' (meters) and 'travel_time' (seconds).
    """

    # --- sanity: required edge attributes ---
    any_edge = (
        next(iter(G.edges(data=True, keys=True)))[3] if G.number_of_edges() else {}
    )
    if "length" not in any_edge or "travel_time" not in any_edge:
        raise ValueError("Graph edges must have 'length' and 'travel_time' attributes.")

    # --- project MultiDiGraph -> DiGraph with min travel_time per (u,v) ---
    H = project_to_minweight_digraph(G, time_attr="travel_time", dist_attr="length")

    # --- determine SCC nodes ---
    n2c = compute_node2scc(G)
    scc_nodes = [n for n, cid in n2c.items() if cid == target_scc_id]
    if not scc_nodes:
        logger.warning(f"No nodes found in SCC {target_scc_id}")
        return

    N = len(scc_nodes)
    expected_per_source = N - 1
    total_pairs = N * (N - 1)  # ordered OD pairs inside the SCC
    scc_set: Set[int] = set(scc_nodes)

    logger.info(
        f"SCC {target_scc_id}: {N} nodes (ordered pairs total = {total_pairs:,})"
    )

    con = cache.con
    total_inserted = 0

    # ---- progress helpers (NEW DB: no graph_hash) ----
    def existing_time(o: int) -> Set[int]:
        return cache.existing_dests_time(o) & scc_set

    def count_existing_time(o: int) -> int:
        # Optimization: COUNT(*) is much faster than fetching all IDs for checking completion
        row = con.execute("SELECT COUNT(*) FROM time_simple WHERE o=?", [o]).fetchone()
        return int(row[0]) if row else 0

    def existing_ext(o: int) -> Set[int]:
        # path_extended exists but cache class doesn't expose a helper — query here:
        rows = con.execute("SELECT d FROM path_extended WHERE o=?", [o]).fetchall()
        return {int(r[0]) for r in rows} & scc_set

    def count_existing_ext(o: int) -> int:
        row = con.execute("SELECT COUNT(*) FROM path_extended WHERE o=?", [o]).fetchone()
        return int(row[0]) if row else 0

    def compute_progress():
        # Bulk query for total progress
        # This is much faster than iterating 20k nodes
        row = con.execute("SELECT COUNT(*) FROM time_simple").fetchone()
        simple_done = int(row[0]) if row else 0
        return simple_done, total_pairs

    def compute_progress_extended():
        row_s = con.execute("SELECT COUNT(*) FROM time_simple").fetchone()
        simple_done = int(row_s[0]) if row_s else 0
        
        row_x = con.execute("SELECT COUNT(*) FROM path_extended").fetchone()
        ext_done = int(row_x[0]) if row_x else 0
        
        return simple_done, total_pairs, ext_done, total_pairs

    # initial progress
    if write_extended:
        s_done, s_tot, x_done, x_tot = compute_progress_extended()
        logger.info(
            f"Initial progress (simple): {s_done:,}/{s_tot:,} ({(100*s_done/s_tot if s_tot else 0):.2f}%)"
        )
        logger.info(
            f"Initial progress (extended): {x_done:,}/{x_tot:,} ({(100*x_done/x_tot if x_tot else 0):.2f}%)"
        )
    else:
        s_done, s_tot = compute_progress()
        logger.info(
            f"Initial progress: {s_done:,}/{s_tot:,} ({(100*s_done/s_tot if s_tot else 0):.2f}%)"
        )

    # --- iterate sources in batches (by index) ---
    for batch_start in range(0, N, batch_size_sources):
        batch_end = min(batch_start + batch_size_sources, N)
        batch_sources = scc_nodes[batch_start:batch_end]
        batch_idx = (batch_start // batch_size_sources) + 1
        batch_count = len(batch_sources)

        # ---- fast pre-check: is this batch already complete? ----
        # Heuristic: check first, last, and ~5 random sources to speed up skipping
        # OPTIMIZATION: Fetch counts for ALL check indices in ONE query
        indices_to_check = {0, len(batch_sources) - 1}
        if len(batch_sources) > 2:
            population = range(1, len(batch_sources) - 1)
            k = min(5, len(population))
            indices_to_check.update(random.sample(population, k))
        
        check_nodes = [batch_sources[i] for i in indices_to_check]
        
        # Check simple counts
        placeholders = ",".join(["?"] * len(check_nodes))
        rows = con.execute(
            f"SELECT o, COUNT(*) FROM time_simple WHERE o IN ({placeholders}) GROUP BY o",
            check_nodes
        ).fetchall()
        counts_map = {r[0]: r[1] for r in rows}
        
        batch_complete = True
        for o in check_nodes:
            if counts_map.get(o, 0) != expected_per_source:
                batch_complete = False
                break
        
        if batch_complete and write_extended:
            # Check extended counts if needed
            rows_ext = con.execute(
                f"SELECT o, COUNT(*) FROM path_extended WHERE o IN ({placeholders}) GROUP BY o",
                check_nodes
            ).fetchall()
            counts_ext_map = {r[0]: r[1] for r in rows_ext}
            for o in check_nodes:
                if counts_ext_map.get(o, 0) != expected_per_source:
                    batch_complete = False
                    break

        if batch_complete:
            # ... (logging remains same)
            if write_extended:
                s_done, s_tot, x_done, x_tot = compute_progress_extended()
                logger.info(
                    f"Batch {batch_idx} already complete; skipping "
                    f"(sources [{batch_start}..{batch_end}) size={batch_count}) | "
                    f"progress simple {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%), "
                    f"extended {x_done:,}/{x_tot:,} ({(100*x_done/x_tot):.2f}%)"
                )
            else:
                s_done, s_tot = compute_progress()
                logger.info(
                    f"Batch {batch_idx} already complete; skipping "
                    f"(sources [{batch_start}..{batch_end}) size={batch_count}) | "
                    f"progress {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%)"
                )
            continue

        # ---- start batch ----
        logger.info(
            f"Batch {batch_idx} start: sources [{batch_start}..{batch_end}) size={batch_count}"
        )
        t0 = time.perf_counter()
        batch_inserted = 0

        # ---- process sources in the batch ----
        for i, o in enumerate(batch_sources, start=batch_start + 1):
            # Check count first to avoid fetching set if done
            cnt_simple = count_existing_time(o)
            if cnt_simple == expected_per_source:
                 if not write_extended:
                     continue
                 if count_existing_ext(o) == expected_per_source:
                     continue

            # Only fetch the set if we actually need to compute something
            done_simple = existing_time(o)
            done_ext = existing_ext(o) if write_extended else set()

            def needs(d: int) -> bool:
                if d == o:
                    return False
                if write_extended:
                    return (d not in done_simple) or (d not in done_ext)
                return d not in done_simple

            try:
                con.execute("BEGIN")
                # OPTIMIZATION: Compute ALL paths from 'o' in one go
                # Explicitly pass target=None to help IDE type inference (returns dicts, not scalars)
                # Pyright still struggles with the overload, so we cast the result.
                all_lengths, all_paths = nx.single_source_dijkstra(
                    H, o, target=None, weight="travel_time"
                )
                all_paths = cast(Dict[int, List[int]], all_paths)

                for d, path_nodes in all_paths.items():
                    # Filter: only process if 'd' is in our target SCC and needs update
                    if d not in scc_set or not needs(d):
                        continue

                    # We have the path and time, just need to sum up the distance (meters)
                    # (accumulate_on_path is still fast enough for just summing attributes)
                    time_s, dist_m, edge_times = accumulate_on_path(H, path_nodes)

                    # Upsert simple (new schema)
                    if d not in done_simple:
                        cache.put_simple(o, d, time_s, dist_m)

                    # Upsert extended if requested
                    if write_extended and d not in done_ext:
                        cache.put_extended(o, d, path_nodes, edge_times)

                    batch_inserted += 1
                    total_inserted += 1

                con.execute("COMMIT")

            except KeyboardInterrupt:
                logger.warning(
                    f"Interrupted on source {o} — rolling back current source."
                )
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                raise

            except RuntimeError as e:
                msg = str(e).lower()
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                if "interrupted" in msg:
                    logger.warning(
                        "DuckDB reported 'interrupted'. Reopening connection and continuing…"
                    )
                    cache.reconnect()
                    con = cache.con
                    continue
                else:
                    raise

            # periodic progress log (global)
            if (i % log_every_sources) == 0:
                if write_extended:
                    s_done, s_tot, x_done, x_tot = compute_progress_extended()
                    logger.info(
                        f"Progress: processed {i}/{N} sources; "
                        f"simple {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%), "
                        f"extended {x_done:,}/{x_tot:,} ({(100*x_done/x_tot):.2f}%)"
                    )
                else:
                    s_done, s_tot = compute_progress()
                    logger.info(
                        f"Progress: processed {i}/{N} sources; "
                        f"{s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%)"
                    )

        # ---- end batch: timing + verification + progress ----
        elapsed = time.perf_counter() - t0
        if write_extended:
            s_done, s_tot, x_done, x_tot = compute_progress_extended()
            logger.info(
                f"Batch {batch_idx} done in {elapsed:.2f}s: size={batch_count}, new rows ≈ {batch_inserted:,} | "
                f"progress simple {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%), "
                f"extended {x_done:,}/{x_tot:,} ({(100*x_done/x_tot):.2f}%)"
            )
        else:
            s_done, s_tot = compute_progress()
            logger.info(
                f"Batch {batch_idx} done in {elapsed:.2f}s: size={batch_count}, new rows ≈ {batch_inserted:,} | "
                f"progress {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%)"
            )

    # final progress
    if write_extended:
        s_done, s_tot, x_done, x_tot = compute_progress_extended()
        logger.info(
            f"Completed pass over SCC {target_scc_id}. "
            f"simple {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%), "
            f"extended {x_done:,}/{x_tot:,} ({(100*x_done/x_tot):.2f}%)"
        )
    else:
        s_done, s_tot = compute_progress()
        logger.info(
            f"Completed pass over SCC {target_scc_id}. "
            f"{s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%)"
        )


# ---- main (edit constants only) ---------------------------------------------

if __name__ == "__main__":
    # ==== EDIT THESE CONSTANTS ONLY ====
    TARGET_SCC_ID = 32
    WRITE_EXTENDED = False  # True to also populate path_extended
    LOG_EVERY = 50  # log progress every N sources (global)
    BATCH_SIZE_SOURCES = 10  # number of sources per logged batch
    # ===================================

    (DATA_DIR / "duckDB").mkdir(parents=True, exist_ok=True)

    G = load_graph()  # must provide 'length' and 'travel_time'
    cache = DuckDBPathCache()  # NEW DB cache

    precompute_scc_with_bidij_resumable(
        G,
        target_scc_id=TARGET_SCC_ID,
        cache=cache,
        write_extended=WRITE_EXTENDED,
        log_every_sources=LOG_EVERY,
        batch_size_sources=BATCH_SIZE_SOURCES,
    )
