from typing import Optional, Sequence
import duckdb

from config.paths import DATA_DIR


class DuckDBPathCache:
    def __init__(self):
        db_path = DATA_DIR / "duckDB" / "route_cache.duckdb"
        # make filepath if not exists
        # db_path.parent.mkdir(parents=True, exist_ok=True)

        self.con = duckdb.connect(db_path)
        # Small, fixed-width table for hot path (no lists)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS path_simple(
              o BIGINT,
              d BIGINT,
              graph_hash TEXT,
              time_s DOUBLE,
              dist_m DOUBLE,
              PRIMARY KEY (o, d, graph_hash)
            );
        """
        )
        self.con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_path_simple
            ON path_simple(o, d, graph_hash);
        """
        )

        # Extended table (denormalized for speed; duplicates nodes)
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS path_extended(
              o BIGINT,
              d BIGINT,
              graph_hash TEXT,
              nodes BIGINT[],
              edge_times DOUBLE[],
              time_s DOUBLE,
              dist_m DOUBLE,
              PRIMARY KEY (o, d, graph_hash)
            );
        """
        )
        self.con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_path_ext
            ON path_extended(o, d, graph_hash);
        """
        )

    def reconnect(self):
        # close and reopen the connection (same file path)
        db_path = DATA_DIR / "duckDB" / "route_cache.duckdb"
        try:
            self.con.close()
        except Exception:
            pass
        self.con = duckdb.connect(db_path)

    # ---------- Simple (hot) ----------
    def get_simple(self, o: int, d: int, h: str) -> Optional[tuple[float, float]]:
        row = self.con.execute(
            "SELECT time_s, dist_m FROM path_simple WHERE o=? AND d=? AND graph_hash=?",
            [o, d, h],
        ).fetchone()
        return (float(row[0]), float(row[1])) if row else None

    def put_simple(self, o: int, d: int, h: str, time_s: float, dist_m: float) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO path_simple VALUES (?,?,?,?,?)",
            [o, d, h, float(time_s), float(dist_m)],
        )

    # ---------- Extended ----------
    def get_extended(
        self, o: int, d: int, h: str
    ) -> Optional[tuple[Sequence[int], Sequence[float], float, float]]:
        row = self.con.execute(
            "SELECT nodes, edge_times, time_s, dist_m "
            "FROM path_extended WHERE o=? AND d=? AND graph_hash=?",
            [o, d, h],
        ).fetchone()
        if not row:
            return None
        nodes, edge_times, time_s, dist_m = row
        return nodes, edge_times, float(time_s), float(dist_m)

    def put_extended(
        self,
        o: int,
        d: int,
        h: str,
        nodes: Sequence[int],
        edge_times: Sequence[float],
        time_s: float,
        dist_m: float,
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO path_extended VALUES (?,?,?,?,?,?,?)",
            [o, d, h, list(nodes), list(edge_times), float(time_s), float(dist_m)],
        )

    # ---------- Resumability helpers ----------
    def existing_dests_simple(self, o: int, h: str) -> set[int]:
        rows = self.con.execute(
            "SELECT d FROM path_simple WHERE o=? AND graph_hash=?", [o, h]
        ).fetchall()
        return {int(r[0]) for r in rows}

    def existing_dests_extended(self, o: int, h: str) -> set[int]:
        rows = self.con.execute(
            "SELECT d FROM path_extended WHERE o=? AND graph_hash=?", [o, h]
        ).fetchall()
        return {int(r[0]) for r in rows}

    # ---------- Maintenance / Utilities ----------
    def counts(self) -> tuple[int, int]:
        """Return (rows_in_simple, rows_in_extended)."""
        simple = self.con.execute("SELECT COUNT(*) FROM path_simple").fetchone()
        simple = simple[0] if simple else 0
        try:
            extended = self.con.execute("SELECT COUNT(*) FROM path_extended").fetchone()
            extended = extended[0] if extended else 0
        except duckdb.CatalogException:
            extended = 0
        return int(simple), int(extended)

    def clear_graph(self, graph_hash: str, *, force: bool = False) -> None:
        """
        Delete only entries for a specific graph_hash (safe for multi-graph caches).
        Prompts unless force=True.
        """
        if not force:
            ans = input(
                f"This will delete all cached paths for graph_hash '{graph_hash}'. "
                "Type YES to proceed: "
            ).strip()
            if ans != "YES":
                print("Aborted.")
                return

        self.con.execute("BEGIN")
        try:
            self.con.execute(
                "DELETE FROM path_simple WHERE graph_hash = ?", [graph_hash]
            )
            self.con.execute(
                "DELETE FROM path_extended WHERE graph_hash = ?", [graph_hash]
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        # Persist changes to disk
        self.con.execute("CHECKPOINT")
        print(f"Cleared cache for graph_hash={graph_hash}")

    def clear_all(self, *, force: bool = False) -> None:
        """
        Wipe both tables. Prompts 'are you sure?' unless force=True.
        """
        if not force:
            ans = input(
                "This will PERMANENTLY delete ALL cached paths (simple + extended). "
                "Type YES to proceed: "
            ).strip()
            if ans != "YES":
                print("Aborted.")
                return

        self.con.execute("BEGIN")
        try:
            self.con.execute("DELETE FROM path_simple")
            self.con.execute("DELETE FROM path_extended")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("CHECKPOINT")
        print("Cleared all cached paths.")


if __name__ == "__main__":
    cache = DuckDBPathCache()
