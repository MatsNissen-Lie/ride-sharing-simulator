"""Central access to graph, vehicle, and rider request data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import networkx as nx
import numpy as np
import polars as pl

from config import NUMBER_OF_VEHICLES, get_logger, RIDER_DIR, RIDER_PATTERN
from config.config import VEHICLE_CAPACITY
from src.interfaces.data_models import MetaRequest, Passenger, Vehicle
from src.utils.sim_clock import SimulationClock
from src.state_representation.manhatten_fast import load_graph
from src.state_representation.vehicle_sampling import sample_vehicle_positions

logger = get_logger(__name__)

RIDER_COLUMNS: list[str] = [
    "pickup_datetime",  # NOTE: We are assuming the pickup time is the the order time in our system.
    # "dropoff_datetime",
    # "pickup_zone",
    # "dropoff_zone",
    "orig_node",
    "dest_node",
]


class SimulationDataProvider:
    """Loads the Manhattan graph and yields vehicle/request dataclasses."""

    def __init__(
        self,
        *,
        rider_dir: Path = RIDER_DIR,
        rider_pattern: str = RIDER_PATTERN,
        vehicle_count: int = NUMBER_OF_VEHICLES,
        vehicle_capacity: int = VEHICLE_CAPACITY,
    ) -> None:
        self._graph: nx.MultiDiGraph = load_graph()
        self._rider_dir = rider_dir
        self._rider_pattern = rider_pattern
        self._vehicle_count = vehicle_count
        self._vehicle_capacity = vehicle_capacity
        self._request_counter = 0
        self._clock: SimulationClock | None = None

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    # ------------------------------------------------------------------
    # Riders
    # ------------------------------------------------------------------

    def iter_request_files(self) -> Iterable[Path]:
        return sorted(self._rider_dir.glob(self._rider_pattern))

    def iter_requests(self, start: float | None = None) -> Iterator[MetaRequest]:
        """Requests in time order. With `start` (simulation seconds), requests
        before it are skipped without being built; ids stay those of a full pass."""
        last_timestamp = None
        paths = list(self.iter_request_files())
        for idx, path in enumerate(paths):
            logger.info("Loading rider file %s", path.name)
            df = pl.read_parquet(path, columns=RIDER_COLUMNS)
            # set up clock on first file and set dates for date column
            if self._clock is None:
                first_date_value = self._ensure_datetime(df["pickup_datetime"][0])
                self._clock = SimulationClock(start_time=first_date_value)
                last_timestamp = self._ensure_datetime(df["pickup_datetime"][-1])

            elif last_timestamp is not None:
                first_date_value = self._ensure_datetime(df["pickup_datetime"][0])
                if first_date_value < last_timestamp:
                    raise AssertionError(
                        f"Rider file {path.name} has pickup datetimes earlier than previous file {paths[idx-1].name}"
                    )
                last_timestamp = self._ensure_datetime(df["pickup_datetime"][-1])

            df = df.with_columns(
                pl.col("pickup_datetime")
                .cast(pl.Datetime)
                .dt.replace_time_zone("America/New_York")
            )
            series = df["pickup_datetime"]
            if not series.is_sorted():
                raise AssertionError(
                    f"Pickup datetimes in {path.name} are not sorted ascending"
                )
            yield from self._requests_from_frame(df, start)

    def _requests_from_frame(
        self, df: pl.DataFrame, start: float | None = None
    ) -> Iterator[MetaRequest]:
        """`_request_from_row` for every row from `start` on (rows are sorted by
        time), with the conversions done per column."""
        times = self._clock.to_relative_seconds_array(  # type: ignore[union-attr]
            df["pickup_datetime"].dt.epoch("us").to_numpy()
        )
        skip = 0 if start is None else int(np.searchsorted(times, start, side="left"))
        first = self._request_counter + 1 + skip
        self._request_counter += df.height
        df, times = df[skip:], times[skip:]
        ids = df.select(
            pl.col("pickup_datetime").dt.strftime("%m%d")
            + "_"
            + pl.int_range(first, first + pl.len()).cast(pl.Utf8)
        ).to_series()
        for request_id, origin, destination, t in zip(
            ids.to_list(),
            df["orig_node"].cast(pl.Int64).to_list(),
            df["dest_node"].cast(pl.Int64).to_list(),
            times.tolist(),
        ):
            yield MetaRequest(request_id, origin, destination, t)

    def _request_from_row(self, row: dict) -> MetaRequest:
        pickup_dt = self._ensure_datetime(row["pickup_datetime"])
        timestamp = self._clock.to_relative_seconds(pickup_dt)  # type: ignore
        month_day = "0000"
        if hasattr(pickup_dt, "strftime"):
            month_day = pickup_dt.strftime("%m%d")
        self._request_counter += 1
        request_id = f"{month_day}_{self._request_counter}"
        return MetaRequest(
            id=request_id,
            origin_node=int(row["orig_node"]),
            destination_node=int(row["dest_node"]),
            request_time=float(timestamp),
        )

    @staticmethod
    def _ensure_datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        raise TypeError("pickup_datetime must be a datetime value")

    def first_request_time(self) -> datetime:
        """Pickup time of the first request; simulation time 0."""
        path = next(iter(self.iter_request_files()))
        df = pl.read_parquet(path, columns=["pickup_datetime"], n_rows=1)
        return self._ensure_datetime(df["pickup_datetime"][0])

    def request_table(self) -> pl.DataFrame:
        """All requests in the rider files: pickup_datetime and orig_node."""
        return pl.concat(
            pl.read_parquet(path, columns=["pickup_datetime", "orig_node"])
            for path in self.iter_request_files()
        )

    def request_trips(self) -> pl.DataFrame:
        """All requests in the rider files: pickup_datetime, orig_node and dest_node."""
        return pl.concat(
            pl.read_parquet(path, columns=["pickup_datetime", "orig_node", "dest_node"])
            for path in self.iter_request_files()
        )

    def request_origins_between(self, start: float, end: float) -> pl.DataFrame:
        """Requests with start <= time < end (simulation seconds): columns t, origin."""
        t0 = self.first_request_time()
        lo = t0 + timedelta(seconds=start)
        hi = t0 + timedelta(seconds=end)
        frames = [
            pl.scan_parquet(path)
            .filter(pl.col("pickup_datetime").is_between(lo, hi, closed="left"))
            .select(
                ((pl.col("pickup_datetime") - t0).dt.total_microseconds() / 1e6).alias("t"),
                pl.col("orig_node").cast(pl.Int64).alias("origin"),
            )
            for path in self.iter_request_files()
        ]
        return pl.concat(frames).collect().sort("t")

    # ------------------------------------------------------------------
    # Vehicles
    # ------------------------------------------------------------------

    def iter_vehicles(self, vehicle_count: int) -> Iterator[Vehicle]:
        logger.info("Sampling %d vehicle positions from graph", vehicle_count)
        df = sample_vehicle_positions(self._graph, vehicle_count, seed=42)
        for row in df.to_dicts():
            yield self._vehicle_from_row(row)

    def _vehicle_from_row(self, row: dict) -> Vehicle:
        passengers: set[Passenger] = set()
        node_id = int(row.get("current_node", row["node_id"]))
        return Vehicle(
            id=str(row["vehicle_id"]),
            capacity=self._vehicle_capacity,
            passengers_onboard=passengers,
            last_visited_node=node_id,
            last_visited_time=0.0,
            current_route=None,
            route_assignment_time=None,
        )

    def get_simulation_clock(self) -> SimulationClock:
        if self._clock is None:
            raise RuntimeError(
                "Simulation clock is not initialized yet before requests are loaded. Iterate requests first."
            )
        return self._clock


__all__ = ["SimulationDataProvider"]


if __name__ == "__main__":
    provider = SimulationDataProvider()
    print(f"Graph nodes: {provider.graph.number_of_nodes()}")

    requests = list(provider.iter_requests())[:5]
    print(f"Sample requests: {[req.id for req in requests]}")

    vehicles = list(provider.iter_vehicles(5))
    print(f"Sample vehicles: {[veh.id for veh in vehicles]}")
