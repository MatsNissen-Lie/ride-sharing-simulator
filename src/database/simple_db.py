from __future__ import annotations
from typing import Optional, Sequence, Set
import duckdb
from config.paths import DATA_DIR


class DuckDBPathCache:
    """
    Minimal DuckDB cache for shortest paths.

    Tables:
      - time_simple(o BIGINT, d BIGINT, time_s DOUBLE, dist_m DOUBLE)
        * tiny hot path table
        * time_s = -1.0, dist_m = -1.0 indicates NO PATH exists
      - path_extended(o BIGINT, d BIGINT, nodes BIGINT[], edge_times DOUBLE[])
        * optional, only populated when extended results are requested
    """

    def __init__(self):
        db_path = DATA_DIR / "duckDB" / "route_db.duckdb"
        (db_path.parent).mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self.con = duckdb.connect(self._db_path)
        # small point lookups; threads mostly help on index builds
        self.con.execute("PRAGMA threads=8")

        # Hot, minimal table for 20M+ pairs
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS time_simple(
              o BIGINT,
              d BIGINT,
              time_s DOUBLE,
              dist_m DOUBLE
            );
        """
        )
        self.con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_time_simple_od
            ON time_simple(o, d);
        """
        )

        # Extended: only for pairs where you explicitly need nodes/edge_times
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS path_extended(
              o BIGINT,
              d BIGINT,
              nodes BIGINT[],
              edge_times DOUBLE[]
            );
        """
        )
        self.con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_path_extended_od
            ON path_extended(o, d);
        """
        )

    # ---------- connection mgmt ----------
    def reconnect(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass
        self.con = duckdb.connect(self._db_path)

    # ---------- simple (hot) ----------
    def get_simple(self, o: int, d: int) -> Optional[tuple[float, float]]:
        row = self.con.execute(
            "SELECT time_s, dist_m FROM time_simple WHERE o=? AND d=?", [o, d]
        ).fetchone()
        return (float(row[0]), float(row[1])) if row else None

    def put_simple(self, o: int, d: int, time_s: float, dist_m: float) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO time_simple(o,d,time_s,dist_m) VALUES (?,?,?,?)",
            [int(o), int(d), float(time_s), float(dist_m)],
        )

    def put_no_path(self, o: int, d: int) -> None:
        """Explicitly store that there is NO path between o and d (using -1.0)."""
        self.put_simple(o, d, -1.0, -1.0)

    # ---------- extended (optional) ----------
    def get_extended(
        self, o: int, d: int
    ) -> Optional[tuple[Sequence[int], Sequence[float]]]:
        row = self.con.execute(
            "SELECT nodes, edge_times FROM path_extended WHERE o=? AND d=?", [o, d]
        ).fetchone()
        if not row:
            return None
        nodes, edge_times = row
        return nodes, edge_times

    def put_extended(
        self,
        o: int,
        d: int,
        nodes: Sequence[int],
        edge_times: Sequence[float],
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO path_extended(o,d,nodes,edge_times) VALUES (?,?,?,?)",
            [int(o), int(d), list(nodes), list(edge_times)],
        )

    def get_both(
        self, o: int, d: int
    ) -> tuple[float | None, float | None, list[int] | None, list[float] | None]:
        """
        Single round-trip: returns (time_s, dist_m, nodes, edge_times).
        Any field may be None if not present.
        """
        row = self.con.execute(
            """
            SELECT t.time_s, t.dist_m, x.nodes, x.edge_times
            FROM time_simple t
            FULL OUTER JOIN path_extended x
              ON t.o = x.o AND t.d = x.d
            WHERE COALESCE(t.o, x.o) = ? AND COALESCE(t.d, x.d) = ?
            """,
            [o, d],
        ).fetchone()
        if not row:
            return None, None, None, None
        time_s, dist_m, nodes, edge_times = row
        return (
            float(time_s) if time_s is not None else None,
            float(dist_m) if dist_m is not None else None,
            list(nodes) if nodes is not None else None,
            list(edge_times) if edge_times is not None else None,
        )

    # ---------- helpers ----------
    def existing_dests_time(self, o: int) -> Set[int]:
        rows = self.con.execute("SELECT d FROM time_simple WHERE o=?", [o]).fetchall()
        return {int(r[0]) for r in rows}

    def counts(self) -> tuple[int, int]:
        t = self.con.execute("SELECT COUNT(*) FROM time_simple").fetchone()
        x = self.con.execute("SELECT COUNT(*) FROM path_extended").fetchone()
        return int(t[0] if t else 0), int(x[0] if x else 0)

    # Maintenance
    def clear_all(self, *, force: bool = False) -> None:
        if not force:
            ans = input(
                "This will delete ALL cached rows. Type YES to proceed: "
            ).strip()
            if ans != "YES":
                print("Aborted.")
                return
        self.con.execute("BEGIN")
        try:
            self.con.execute("DELETE FROM time_simple")
            self.con.execute("DELETE FROM path_extended")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        self.con.execute("CHECKPOINT")
        print("Cleared all cached rows.")

    def cluster(self) -> None:
        # Optional compaction after large loads
        self.con.execute(
            """
            CREATE TABLE time_simple_clustered AS
            SELECT * FROM time_simple ORDER BY o, d;
        """
        )
        self.con.execute("DROP TABLE time_simple")
        self.con.execute("ALTER TABLE time_simple_clustered RENAME TO time_simple")
        self.con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_time_simple_od
            ON time_simple(o, d);
        """
        )
        self.con.execute(
            """
            CREATE TABLE path_extended_clustered AS
            SELECT * FROM path_extended ORDER BY o, d;
        """
        )
        self.con.execute("DROP TABLE path_extended")
        self.con.execute("ALTER TABLE path_extended_clustered RENAME TO path_extended")
        self.con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_path_extended_od
            ON path_extended(o, d);
        """
        )
        self.con.execute("CHECKPOINT")
