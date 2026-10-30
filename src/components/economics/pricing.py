"""Fare models: what a rider pays."""

from config.config import FARE_BASE, FARE_PER_KM, FARE_PER_MIN, POOLING_DISCOUNT
from src.interfaces import IFareModel, INetworkService


class LinearFareModel(IFareModel):
    """
    Posted solo price base + per_km * km + per_min * min on the simulated shortest
    path (time includes the travel-time factor), minus an upfront pooling discount
    that every rider receives because all riders accept pooling.

    Coefficients and discount: scripts/analysis/fare_model.py.
    """

    def __init__(
        self,
        network: INetworkService,
        base: float = FARE_BASE,
        per_km: float = FARE_PER_KM,
        per_min: float = FARE_PER_MIN,
        pooling_discount: float = POOLING_DISCOUNT,
    ):
        self._network = network
        self._base = base
        self._per_km = per_km
        self._per_min = per_min
        self._factor = 1.0 - pooling_discount

    def price(self, origin: int, destination: int) -> float:
        km = self._network.get_distance_matrix()[origin, destination] / 1000
        minutes = self._network.get_travel_time(origin, destination) / 60
        return self._factor * (self._base + self._per_km * km + self._per_min * minutes)
