"""Simulation clock utilities for converting datetimes to relative seconds."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


class SimulationClock:
    """Tracks the first timestamp and converts values to/from relative seconds."""

    def __init__(self, start_time: datetime) -> None:
        self._start_epoch: float = self._to_epoch_seconds(start_time)

    def to_relative_seconds(self, value: datetime) -> float:
        epoch = self._to_epoch_seconds(value)
        return epoch - self._start_epoch

    def from_relative_seconds(self, offset: float) -> datetime:
        if self._start_epoch is None:
            raise ValueError("Simulation clock has not been initialized")
        # Create datetime in NYC time
        nyc_zone = ZoneInfo("America/New_York")
        return datetime.fromtimestamp(self._start_epoch + offset, tz=nyc_zone)

    @staticmethod
    def _to_epoch_seconds(value: datetime) -> float:
        if value.tzinfo is None:
            # Assume naive datetimes are in NYC time
            nyc_zone = ZoneInfo("America/New_York")
            value = value.replace(tzinfo=nyc_zone)
        return value.timestamp()


__all__ = ["SimulationClock"]
