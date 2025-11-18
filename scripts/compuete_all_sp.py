# scripts/precompute_scc32_resume.py

from typing import Dict, Iterable, List, Tuple, cast
import time
import networkx as nx

from config.paths import DATA_DIR
from src.components.network_service import NetworkXService
from src.database.shortest_path_db import DuckDBPathCache
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


def graph_hash(G: nx.MultiDiGraph, weight_attr: str = "travel_time") -> str:
    return NetworkXService.graph_hash(G, weight_attr)


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
    - Skips (o,d) pairs already present in DuckDB
    - SKIPS WHOLE BATCHES that are fully complete
    - Logs per-batch timing
    - Logs overall progress: done / total (percent) for simple (and extended if enabled)

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
    scc_set = set(scc_nodes)

    logger.info(
        f"SCC {target_scc_id}: {N} nodes (ordered pairs total = {total_pairs:,})"
    )

    # --- cache + graph versioning ---
    gh = graph_hash(G, "travel_time")
    con = cache.con
    total_inserted = 0

    # helper: compute overall progress from DB (simple + extended)
    def compute_progress_x(
        write_extended=write_extended,
    ) -> tuple[int, int] | tuple[int, int, int, int]:
        simple_done = 0
        ext_done = 0
        for o in scc_nodes:
            dset = cache.existing_dests_simple(o, gh)
            simple_done += len(dset & scc_set)
            if write_extended:
                dsetx = cache.existing_dests_extended(o, gh)
                ext_done += len(dsetx & scc_set)
        if write_extended:
            return simple_done, total_pairs, ext_done, total_pairs
        else:
            return simple_done, total_pairs

    def compute_progress_extended() -> tuple[int, int, int, int]:
        res = compute_progress_x(write_extended=True)
        return cast(Tuple[int, int, int, int], res)

    def compute_progress() -> tuple[int, int]:
        res = compute_progress_x(write_extended=False)
        return cast(Tuple[int, int], res)

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
        batch_complete = True
        for o in batch_sources:
            if len(cache.existing_dests_simple(o, gh) & scc_set) != expected_per_source:
                batch_complete = False
                break
            if write_extended:
                if (
                    len(cache.existing_dests_extended(o, gh) & scc_set)
                    != expected_per_source
                ):
                    batch_complete = False
                    break

        if batch_complete:
            # still log that we’re skipping a finished batch
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
            # Query what's already present for this origin (and keep only SCC destinations)
            done_simple = cache.existing_dests_simple(o, gh) & scc_set
            done_ext = (
                (cache.existing_dests_extended(o, gh) & scc_set)
                if write_extended
                else set()
            )

            # Quick skip if the whole source is already complete
            if (len(done_simple) == expected_per_source) and (
                (not write_extended) or (len(done_ext) == expected_per_source)
            ):
                continue

            def needs(d: int) -> bool:
                if d == o:
                    return False
                if write_extended:
                    return (d not in done_simple) or (d not in done_ext)
                return d not in done_simple

            try:
                con.execute("BEGIN")
                for d in scc_nodes:
                    if not needs(d):
                        continue
                    try:
                        length, path_nodes = nx.bidirectional_dijkstra(
                            H, o, d, weight="travel_time"
                        )
                    except nx.NetworkXNoPath:
                        continue

                    time_s, dist_m, edge_times = accumulate_on_path(H, path_nodes)

                    # Upsert simple
                    if d not in done_simple:
                        cache.put_simple(o, d, gh, time_s, dist_m)

                    # Upsert extended if requested
                    if write_extended and d not in done_ext:
                        cache.put_extended(
                            o, d, gh, path_nodes, edge_times, time_s, dist_m
                        )

                    batch_inserted += 1
                    total_inserted += 1

                con.execute("COMMIT")

            except KeyboardInterrupt:
                # Clean rollback so this source recomputes on next run
                logger.warning(
                    f"Interrupted on source {o} — rolling back current source."
                )
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                raise

            except RuntimeError as e:
                # DuckDB can remain 'interrupted' after Ctrl-C; reconnect & continue
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
                    # Skip this source for now; it will be retried on the next run
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
                f"Batch {batch_idx} done: size={batch_count}, new rows ≈ {batch_inserted:,}, elapsed={elapsed:.2f}s | "
                f"progress simple {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%), "
                f"extended {x_done:,}/{x_tot:,} ({(100*x_done/x_tot):.2f}%)"
            )
        else:
            s_done, s_tot = compute_progress()
            logger.info(
                f"Batch {batch_idx} done: size={batch_count}, new rows ≈ {batch_inserted:,}, elapsed={elapsed:.2f}s | "
                f"progress {s_done:,}/{s_tot:,} ({(100*s_done/s_tot):.2f}%)"
            )

        # Post-batch verification (ensures batch completion)
        missing_sources = 0
        for o in batch_sources:
            simple_count = len(cache.existing_dests_simple(o, gh) & scc_set)
            miss_simple = expected_per_source - simple_count
            if write_extended:
                ext_count = len(cache.existing_dests_extended(o, gh) & scc_set)
                miss_ext = expected_per_source - ext_count
            else:
                miss_ext = 0
            if miss_simple > 0 or miss_ext > 0:
                missing_sources += 1
                if write_extended:
                    logger.warning(
                        f"Verify batch {batch_idx}: source {o} missing "
                        f"{miss_simple} simple and {miss_ext} extended rows"
                    )
                else:
                    logger.warning(
                        f"Verify batch {batch_idx}: source {o} missing {miss_simple} simple rows"
                    )

        if missing_sources == 0:
            logger.info(f"Batch {batch_idx} verification OK (all sources complete).")
        else:
            logger.warning(
                f"Batch {batch_idx} verification: {missing_sources} source(s) incomplete."
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
    WRITE_EXTENDED = True
    LOG_EVERY = 50  # log progress every N sources (global)
    BATCH_SIZE_SOURCES = 10  # number of sources per logged batch
    # ===================================

    (DATA_DIR / "duckDB").mkdir(parents=True, exist_ok=True)

    G = load_graph()  # must provide 'length' and 'travel_time'
    cache = DuckDBPathCache()

    precompute_scc_with_bidij_resumable(
        G,
        target_scc_id=TARGET_SCC_ID,
        cache=cache,
        write_extended=WRITE_EXTENDED,
        log_every_sources=LOG_EVERY,
        batch_size_sources=BATCH_SIZE_SOURCES,
    )
