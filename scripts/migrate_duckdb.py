# pyright: reportAssignmentType=false
# ruff: noqa: E501
# ignore: file

from pathlib import Path
import duckdb
from config.paths import DATA_DIR


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}ZB"


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
        [name],
    ).fetchone()
    return bool(row)


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def migrate(
    src_db: Path,
    dst_db: Path,
    *,
    cluster_after: bool = True,
) -> None:
    src_db.parent.mkdir(parents=True, exist_ok=True)
    dst_db.parent.mkdir(parents=True, exist_ok=True)

    # Use one connection; ATTACH the destination as a separate schema
    con = duckdb.connect(str(src_db))
    con.execute("PRAGMA threads=8")

    # Create destination by attaching as schema 'newdb'
    con.execute(f"ATTACH '{dst_db}' AS newdb")

    # Quick report
    size_before_src = file_size(src_db)
    size_before_dst = file_size(dst_db)
    print(f"[info] src: {src_db} ({human_bytes(size_before_src)})")
    print(f"[info] dst: {dst_db} ({human_bytes(size_before_dst)})")

    # Check source tables
    has_simple = table_exists(con, "path_simple")
    has_extended = table_exists(con, "path_extended")

    if not has_simple and not has_extended:
        raise RuntimeError(
            "Neither 'path_simple' nor 'path_extended' found in the source DB."
        )

    # Warn if multiple graph hashes exist (you said there’s only one)
    if has_simple:
        gh_stats_row = con.execute(
            "SELECT COUNT(DISTINCT graph_hash) FROM path_simple"
        ).fetchone()
        if gh_stats_row:
            gh_stats = gh_stats_row[0]
            if gh_stats > 1:
                print(
                    f"[warn] Multiple graph_hash values detected in source ({gh_stats}). "
                    "The migration will drop graph_hash and keep last-write-wins per (o,d)."
                )

    # Create destination schema (no graph_hash)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS newdb.time_simple(
          o BIGINT,
          d BIGINT,
          time_s DOUBLE,
          dist_m DOUBLE
        );
    """
    )
    con.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_time_simple_od
        ON newdb.time_simple(o, d);
    """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS newdb.path_extended(
          o BIGINT,
          d BIGINT,
          nodes BIGINT[],
          edge_times DOUBLE[]
        );
    """
    )
    con.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_path_extended_od
        ON newdb.path_extended(o, d);
    """
    )

    # Copy data
    # Note: we do not GROUP; if there are duplicate (o,d) across graph_hash,
    # INSERT OR REPLACE will keep a deterministic last one within this run.
    con.execute("BEGIN")
    try:
        if has_simple:
            con.execute(
                """
                INSERT OR REPLACE INTO newdb.time_simple(o, d, time_s, dist_m)
                SELECT o, d, time_s, dist_m
                FROM path_simple;
            """
            )
        if has_extended:
            con.execute(
                """
                INSERT OR REPLACE INTO newdb.path_extended(o, d, nodes, edge_times)
                SELECT o, d, nodes, edge_times
                FROM path_extended;
            """
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    # Cluster/compact (optional but recommended for locality)
    if cluster_after:
        con.execute(
            """
            CREATE TABLE newdb.time_simple_clustered AS
            SELECT * FROM newdb.time_simple ORDER BY o, d;
        """
        )
        con.execute("DROP TABLE newdb.time_simple")
        con.execute("ALTER TABLE newdb.time_simple_clustered RENAME TO time_simple")
        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_time_simple_od
            ON newdb.time_simple(o, d);
        """
        )

        con.execute(
            """
            CREATE TABLE newdb.path_extended_clustered AS
            SELECT * FROM newdb.path_extended ORDER BY o, d;
        """
        )
        con.execute("DROP TABLE newdb.path_extended")
        con.execute("ALTER TABLE newdb.path_extended_clustered RENAME TO path_extended")
        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_path_extended_od
            ON newdb.path_extended(o, d);
        """
        )

    # Checkpoint both DBs
    con.execute("CHECKPOINT")  # src
    con.execute("CHECKPOINT newdb")  # dst

    # Report
    t_cnt_row = con.execute("SELECT COUNT(*) FROM newdb.time_simple").fetchone()
    t_cnt = t_cnt_row[0] if t_cnt_row else 0
    x_cnt_row = con.execute("SELECT COUNT(*) FROM newdb.path_extended").fetchone()
    x_cnt = x_cnt_row[0] if x_cnt_row else 0
    size_after_dst = file_size(dst_db)
    print(f"[info] migrated counts: time_simple={t_cnt:,}, path_extended={x_cnt:,}")
    print(f"[info] dst size after:  {human_bytes(size_after_dst)}")
    con.close()


if __name__ == "__main__":
    # ====== EDIT THESE IF NEEDED ======
    SRC_DB = DATA_DIR / "duckDB" / "route_cache.duckdb"  # OLD schema
    DST_DB = DATA_DIR / "duckDB" / "route_db.duckdb"  # NEW schema
    CLUSTER_AFTER = True
    # =================================

    migrate(SRC_DB, DST_DB, cluster_after=CLUSTER_AFTER)
