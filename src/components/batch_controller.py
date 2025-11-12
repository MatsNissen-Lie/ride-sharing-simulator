"""Batch controller implementation stub with meta request conversion."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Set, Tuple

from src.interfaces.batch_control import IBatchController
from src.interfaces.data_models import (
    AssignmentResult,
    MetaRequest,
    Request,
    VehicleAvailability,
)
from src.interfaces.services import INetworkService


class BatchController(IBatchController):
    """Minimal batch controller focusing on MetaRequest conversion."""

    def __init__(self, network: INetworkService) -> None:
        super().__init__(network)

    def project_fleet_and_promises(
        self, t_now: float
    ) -> Tuple[Set[VehicleAvailability], Set[Request]]:
        raise NotImplementedError("Fleet projection not implemented yet")

    def convert_meta_requests(self, new_meta: Iterable[MetaRequest]) -> Set[Request]:
        converted: Set[Request] = set()
        for meta in new_meta:
            travel_time = self._network.get_shortest_path(
                meta.origin_node, meta.destination_node
            ).travel_time
            converted.add(
                Request(
                    id=meta.id,
                    origin_node=meta.origin_node,
                    destination_node=meta.destination_node,
                    request_time=meta.request_time,
                    earliest_dropoff_time=meta.request_time + travel_time,
                )
            )
        return converted

    def prepare_batch(
        self, t_now: float, new_meta: Iterable[MetaRequest]
    ) -> Tuple[Set[Request], Set[VehicleAvailability]]:
        raise NotImplementedError("Batch preparation not implemented yet")

    def commit_assignment(self, t_now: float, result: AssignmentResult) -> None:
        raise NotImplementedError("Assignment commit not implemented yet")

    def get_promised_pickup_time(self, request_id: str) -> Optional[float]:
        raise NotImplementedError("Promise storage not implemented")

    def set_promised_pickup_time(self, request_id: str, t_pickup: float) -> None:
        raise NotImplementedError("Promise storage not implemented")

    def _create_promised_request(
        self, request: Request, promised_pickup_time: float
    ) -> Request:
        """Placeholder for deriving promised requests from base requests."""
        raise NotImplementedError("Promised request creation not implemented")


__all__ = ["BatchController"]
