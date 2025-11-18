from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Iterator, Tuple, Set, Optional

from src.interfaces import (
    AssignmentResult,
    MetaRequest,
    Request,
    Vehicle,
    VehicleAvailability,
)
from src.interfaces.services import INetworkService


class IBatchController(ABC):
    """
    Owns persistent fleet + promise state.
    Provides:
      1) A pure projection step -> projected vehicles + promised requests.
      2) A convenience 'prepare_batch' that also converts MetaRequest -> Request.
      3) A commit step to apply the ILP result and update promises.
    """

    # ---- Wiring ----
    def __init__(self, network: INetworkService, vehicles: list[Vehicle]) -> None:
        self._network = network
        self._fleet: list[Vehicle] = vehicles
        self._request_index: dict[str, Request] = {}
        super().__init__()

    # ---- Core: Batch Yielder ----
    @abstractmethod
    def iter_batches(
        self,
        meta_stream: Iterable[MetaRequest],
        batch_window: float,
    ) -> Iterator[Tuple[float, Set[Request], Set[VehicleAvailability]]]:
        """
        Yields successive batches of (t_window_end, Requests, VehicleAvailability)
        covering the incoming meta_stream in windows of size batch_window.
        """
        ...

    @abstractmethod
    def project_fleet_and_update_requests(
        self,
        t_now: float,
    ) -> Set[VehicleAvailability]:
        """
        Project current vehicles to (available_at_node, available_from_time),
        trim served stops, and return:
          - projected VehicleAvailability (drop-offs only for onboard riders)
          - promised-but-unpicked requests rehydrated as immutable Requests
            with promised_pickup_time set and earliest_dropoff_time precomputed.
        """
        ...

    # ---- Core: Meta-Request -> Request Yielder ----
    @abstractmethod
    def convert_meta_requests(
        self,
        new_meta: Iterable[MetaRequest],
    ) -> Set[Request]:
        """
        Converts incoming MetaRequest ->
        Request (computing earliest_dropoff_time = request_time + τ(o,d)),
        merges with promised requests, and returns the full set of Requests.
        """
        ...

    # ---- Convenience: full batch assembly ----
    @abstractmethod
    def prepare_batch(
        self,
        t_now: float,
        new_meta: Iterable[MetaRequest],
    ) -> Tuple[Set[Request], Set[VehicleAvailability]]:
        """
        Calls project_fleet_and_promises(t_now), then converts incoming MetaRequest ->
        Request (computing earliest_dropoff_time = request_time + τ(o,d)),
        merges with promised requests, and returns the batch pools.
        """
        ...

    # ---- Apply solver result ----
    @abstractmethod
    def commit_assignment(
        self,
        t_now: float,
        result: AssignmentResult,
    ) -> None:
        """
        - Set vehicles' current_route and route_start_time = t_now.
        - Move picked riders to onboard; remove completed.
        - For assigned-but-unpicked requests, update persistent promise:
            promised[id] = min(old_promise, scheduled_pickup_time).
        """
        ...
