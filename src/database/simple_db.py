import os
from typing import Optional, Sequence, Set
import numpy as np
import duckdb
from pathlib import Path
from config.paths import DATA_DIR
from config import get_logger

logger = get_logger(__name__)


class DuckDBPathCache:
    """
    Minimal DuckDB cache for shortest paths.
    Supports Read-Write (Main) and Read-Only (Worker) modes.
    """

    def __init__(self, read_only: bool = True):
        self.read_only = read_only

        db_path = DATA_DIR / "duckDB" / "route_db.duckdb"
        self._db_path = str(db_path)

        # Only the writer should try to create directories
        if not self.read_only:
            (db_path.parent).mkdir(parents=True, exist_ok=True)

        self.connect(read_only=self.read_only)

    # ---------- connection mgmt ----------
    def connect(self, read_only: bool | None = None) -> None:
        """Establishes connection based on mode."""
        # Update internal state if provided
        if read_only is not None:
            self.read_only = read_only

        config: dict[str, str | bool | int | float | list[str]] = {}
        if self.read_only is None or self.read_only:
            # CRITICAL: This allows multiple workers to open the file simultaneously
            config = {"access_mode": "READ_ONLY"}

        try:
            self.con = duckdb.connect(self._db_path, config=config)

            # Threads help with index builds/sorts, but less relevant for point lookups.
            # Keep low for workers to avoid over-subscribing CPU.
            self.con.execute("PRAGMA threads=2")

            # Only Writer creates schema. Readers assume it exists.
            if not self.read_only:
                self._init_schema()

        except duckdb.IOException as e:
            if self.read_only:
                logger.error(
                    f"Worker failed to open DB. Is the Main process holding the lock? {e}"
                )
            raise e

    def close(self) -> None:
        """Manually close connection to release the file lock."""
        try:
            self.con.close()
        except Exception:
            pass

    def reconnect(self) -> None:
        self.close()
        self.connect(read_only=self.read_only)

    def _init_schema(self):
        """Idempotent schema creation."""
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

    # ---------- simple (hot) ----------
    def get_simple(self, o: int, d: int) -> Optional[tuple[float, float]]:
        # Works in RO mode
        row = self.con.execute(
            "SELECT time_s, dist_m FROM time_simple WHERE o=? AND d=?", [o, d]
        ).fetchone()
        return (float(row[0]), float(row[1])) if row else None

    def put_simple(self, o: int, d: int, time_s: float, dist_m: float) -> None:
        # GUARD: Workers cannot write to the shared file
        if self.read_only:
            return

        self.con.execute(
            "INSERT OR REPLACE INTO time_simple(o,d,time_s,dist_m) VALUES (?,?,?,?)",
            [int(o), int(d), float(time_s), float(dist_m)],
        )

    def put_no_path(self, o: int, d: int) -> None:
        if self.read_only:
            return
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
        if self.read_only:
            return

        self.con.execute(
            "INSERT OR REPLACE INTO path_extended(o,d,nodes,edge_times) VALUES (?,?,?,?)",
            [int(o), int(d), list(nodes), list(edge_times)],
        )

    def get_both(
        self, o: int, d: int
    ) -> tuple[float | None, float | None, list[int] | None, list[float] | None]:
        # Works in RO mode
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

    def get_travel_time_matrix(
        self, sources: list[int], targets: list[int], fill_value: float = float("inf")
    ) -> np.ndarray:
        """Free-flow travel times in seconds."""
        return self._pair_matrix(sources, targets, "time_s", fill_value)

    def get_distance_matrix(
        self, sources: list[int], targets: list[int], fill_value: float = float("inf")
    ) -> np.ndarray:
        """Shortest-path (by time) distances in metres."""
        return self._pair_matrix(sources, targets, "dist_m", fill_value)

    def _pair_matrix(
        self, sources: list[int], targets: list[int], column: str, fill_value: float
    ) -> np.ndarray:
        if not sources or not targets:
            return np.array([[]])

        # The all-pairs matrix (nodes 0..n-1 both ways) is kept as a .npy next to
        # the database: loading it takes ~0.1 s instead of ~5 s per run. It is
        # rebuilt whenever the database file is newer.
        cache = None
        if sources == targets and sources == list(range(len(sources))):
            cache = Path(self._db_path).with_name(
                f"{Path(self._db_path).stem}.{column}_{len(sources)}.npy"
            )
        if cache is not None and cache.exists() and (
            cache.stat().st_mtime >= Path(self._db_path).stat().st_mtime
        ):
            matrix = np.load(cache)
        else:
            matrix = self._query_pair_matrix(sources, targets, column)
            if cache is not None:
                tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npy")
                np.save(tmp, matrix)
                os.replace(tmp, cache)  # atomic, so parallel runs never read half a file

        matrix[np.isnan(matrix)] = fill_value
        matrix[matrix < 0] = fill_value

        return matrix

    def _query_pair_matrix(
        self, sources: list[int], targets: list[int], column: str
    ) -> np.ndarray:
        """Values for every (source, target), NaN where the table has no row."""
        # Works in RO mode because TEMPORARY tables are process-local (in RAM)
        n_rows = len(sources)
        n_cols = len(targets)

        self.con.execute(
            """
            CREATE OR REPLACE TEMPORARY TABLE v_sources AS 
            SELECT unnest(?) AS node_id, generate_subscripts(?, 1) as idx
            """,
            [sources, sources],
        )

        self.con.execute(
            """
            CREATE OR REPLACE TEMPORARY TABLE v_targets AS 
            SELECT unnest(?) AS node_id, generate_subscripts(?, 1) as idx
            """,
            [targets, targets],
        )

        # Fetch the pairs that exist as columns and scatter them; pairs without
        # a row stay NaN, like the NULLs of a left join.
        query = f"""
            SELECT s.idx AS row_idx, t.idx AS col_idx, ts.{column} AS value
            FROM v_sources s
            JOIN time_simple ts ON s.node_id = ts.o
            JOIN v_targets t ON t.node_id = ts.d
        """
        result = self.con.execute(query).fetchnumpy()
        matrix = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        matrix[result["row_idx"] - 1, result["col_idx"] - 1] = np.ma.filled(
            result["value"].astype(np.float64), np.nan
        )
        return matrix

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
