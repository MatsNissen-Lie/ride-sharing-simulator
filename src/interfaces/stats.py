"""Dataclasses for simulation statistics."""

from dataclasses import dataclass


@dataclass
class ControllerStats:
    """Aggregated statistics from the BatchController for a single batch."""

    delivered_count: int = 0
    expired_count: int = 0
    new_requests_count: int = 0

    # Sums for averaging
    sum_wait_time: float = 0.0
    sum_total_delay: float = 0.0
    sum_expired_wait: float = 0.0

    @property
    def avg_wait_time(self) -> float:
        return (
            self.sum_wait_time / self.delivered_count
            if self.delivered_count > 0
            else 0.0
        )

    @property
    def avg_total_delay(self) -> float:
        return (
            self.sum_total_delay / self.delivered_count
            if self.delivered_count > 0
            else 0.0
        )

    @property
    def avg_expired_wait(self) -> float:
        return (
            self.sum_expired_wait / self.expired_count
            if self.expired_count > 0
            else 0.0
        )


@dataclass
class BatchStatistics:
    """Complete statistics for a single simulation batch (CSV Row)."""

    # Context
    batch_id: int
    sim_time: float
    is_warmup: bool

    # Demand
    requests_new: int
    requests_active: int
    requests_delivered: int
    requests_expired: int

    # Fleet
    vehicles_active: int
    vehicles_idle: int

    # Quality of Service (Realized)
    avg_wait_time: float
    avg_total_delay: float
    avg_expired_wait: float

    # Computational Performance
    time_rv: float
    time_rtv: float
    time_assign: float
    time_rebal: float
    time_total_batch: float

    # Algorithm Stats
    rv_edges: int
    trips_count: int
    objective_value: float

    # Breakdown (flat for CSV)
    # Candidate Trips (from RTV Graph)
    candidate_trips_size_1: int = 0
    candidate_trips_size_2: int = 0
    candidate_trips_size_3: int = 0
    candidate_trips_size_4: int = 0
    candidate_trips_size_5: int = 0
    candidate_trips_size_6: int = 0

    # Assigned Trips (from Solver)
    assigned_trips_size_1: int = 0
    assigned_trips_size_2: int = 0
    assigned_trips_size_3: int = 0
    assigned_trips_size_4: int = 0
    assigned_trips_size_5: int = 0
    assigned_trips_size_6: int = 0

    def to_dict(self) -> dict:
        return self.__dict__
