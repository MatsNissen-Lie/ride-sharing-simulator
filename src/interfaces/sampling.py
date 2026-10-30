from abc import ABC, abstractmethod

from src.interfaces.data_models import Request


class IRequestSampler(ABC):
    """Virtual future requests: one possible realisation of the demand to come."""

    @abstractmethod
    def sample(self, t_start: float, t_end: float, t_now: float | None = None) -> list[Request]:
        """Virtual requests with request times in [t_start, t_end) (simulation seconds); `t_now` as for IDemandModel."""
