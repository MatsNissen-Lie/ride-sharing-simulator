"""Central access to graph, vehicle, and rider request data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional

import networkx as nx
import polars as pl

from config import get_logger, RIDER_DIR, RIDER_PATTERN
from src.interfaces.data_models import Passenger, Request, Vehicle
from src.state_representation.manhatten_fast import load_graph

logger = get_logger(__name__)


class SimulationDataProvider:
    """Loads the Manhattan graph and yields vehicle/request dataclasses."""

    def __init__(
        self,
        *,
        rider_dir: Path = RIDER_DIR,
        rider_pattern: str = RIDER_PATTERN,
        vehicle_path: Optional[Path] = None,
    ) -> None:
        self._graph: nx.MultiDiGraph = load_graph()
        self._rider_dir = rider_dir
        self._rider_pattern = rider_pattern
        self._vehicle_path = vehicle_path
        self._request_counter = 0

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

    def iter_requests(self) -> Iterator[Request]:
        for path in self.iter_request_files():
            logger.info("Loading rider file %s", path.name)
            df = pl.read_parquet(
                path,
                columns=[
                    "pickup_datetime",
                    "pickup_zone",
                    "dropoff_zone",
                ],
            )
            series = df["pickup_datetime"]
            if not series.is_sorted():
                raise AssertionError(
                    f"Pickup datetimes in {path.name} are not sorted ascending"
                )
            for row in df.to_dicts():
                yield self._request_from_row(path.stem, row)

    def _request_from_row(self, source_id: str, row: dict) -> Request:
        pickup_dt = row["pickup_datetime"]
        if hasattr(pickup_dt, "strftime"):
            month_day = pickup_dt.strftime("%m%d")
            timestamp = float(pickup_dt.timestamp())
        else:
            month_day = "0000"
            timestamp = float(pickup_dt)
        self._request_counter += 1
        request_id = f"{month_day}_{self._request_counter}"
        return Request(
            id=request_id,
            origin_node=int(row["pickup_zone"]),
            destination_node=int(row["dropoff_zone"]),
            request_time=float(timestamp),
            earliest_dropoff_time=float(timestamp),
        )

    # ------------------------------------------------------------------
    # Vehicles
    # ------------------------------------------------------------------

    def iter_vehicles(self) -> Iterator[Vehicle]:
        if self._vehicle_path is None:
            return iter(())  # type: ignore[return-value]
        logger.info("Loading vehicle file %s", self._vehicle_path)
        df = pl.read_parquet(self._vehicle_path)
        for row in df.to_dicts():
            yield self._vehicle_from_row(row)

    def _vehicle_from_row(self, row: dict) -> Vehicle:
        passengers: set[Passenger] = set()
        return Vehicle(
            id=str(row["vehicle_id"]),
            current_node=int(row["current_node"]),
            capacity=int(row["capacity"]),
            passengers_onboard=passengers,
            current_route=None,
        )


__all__ = ["SimulationDataProvider"]


if __name__ == "__main__":
    provider = SimulationDataProvider()
    print(f"Graph nodes: {provider.graph.number_of_nodes()}")

    requests = list(provider.iter_requests())[:5]
    print(f"Sample requests: {[req.id for req in requests]}")

    vehicles = list(provider.iter_vehicles())[:5]
    print(f"Sample vehicles: {[veh.id for veh in vehicles]}")
