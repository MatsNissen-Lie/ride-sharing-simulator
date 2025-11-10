from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Set, Optional

from src.interfaces.data_models import (
    Request,
    MetaRequest,
    VehicleAvailability,
    AssignmentResult,
)
from src.interfaces.services import INetworkService


# ---------- Pure fleet projector (no side effects) ----------


class IFleetProjector(ABC):
    @abstractmethod
    def project_fleet_and_promises(
        self,
        t_now: float,
        vehicles_state: Iterable[VehicleAvailability],
    ) -> Tuple[Set[VehicleAvailability], Set[Request]]:
        """
        Input: current fleet state (may include en-route info and pending assignments).
        Output:
          - projected_vehicles: per-vehicle (available_at_node, available_from_time),
            remaining_stops contain ONLY drop-offs for already-onboard riders.
          - promised_requests: unpicked riders from previous plans rehydrated as
            immutable Requests with promised_pickup_time set.
        Must not mutate input state.
        """
        ...


# ---------- Request factory (keeps Request frozen) ----------


class IRequestFactory(ABC):
    @abstractmethod
    def make_request(
        self,
        meta: MetaRequest,
        net: INetworkService,
        promised_pickup_time: Optional[float] = None,
    ) -> Request:
        """
        Computes earliest_dropoff_time = meta.request_time + τ(meta.o, meta.d)
        and returns a frozen Request, optionally with promised_pickup_time.
        """
        ...


# ---------- Batch controller (orchestrates per-batch flow) ----------


class IBatchController(ABC):
    @abstractmethod
    def prepare_batch(
        self,
        t_now: float,
        new_meta: Iterable[MetaRequest],
        vehicles_state: Iterable[VehicleAvailability],
    ) -> Tuple[Set[Request], Set[VehicleAvailability]]:
        """
        - Uses IFleetProjector.project_fleet_and_promises(...) to get:
            projected_vehicles, promised_requests
        - Converts new_meta -> Requests via IRequestFactory (computing t* once)
        - Returns:
            requests = promised_requests ∪ new_requests
            vehicles = projected_vehicles
        """
        ...

    @abstractmethod
    def commit_assignment(
        self,
        t_now: float,
        result: AssignmentResult,
    ) -> None:
        """
        Applies the chosen plan:
          - set vehicles' current_route + route_start_time = t_now
          - mark picked riders onboard; remove completed
          - update promise registry:
              promised_pickup_time[id] = min(old, scheduled_pickup_time)
          (Promised-but-unpicked stay out of vehicles until actually picked)
        """
        ...

    # Optional: tiny promise registry accessors (persistent across batches)
    @abstractmethod
    def get_promised_pickup_time(self, request_id: str) -> Optional[float]: ...

    @abstractmethod
    def set_promised_pickup_time(self, request_id: str, t_pickup: float) -> None: ...
