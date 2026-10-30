"""A forecast corrected by how today's demand has compared with it so far."""

import numpy as np

from src.interfaces import IDemandModel


class NowcastDemandModel(IDemandModel):
    """
    The base forecast scaled by today's ratio of observed to forecast requests
    over the last `window` seconds before `t_now`:

        ratio = (observed + prior) / (forecast + prior),  clipped to [lo, hi]

    city-wide (per zone the counts are too small). `prior` requests of
    pseudo-count keep the ratio near 1 when little is known. Only requests
    before `t_now` are used; without `t_now` the base forecast is returned.

    `times` and `zones` are the requests of the simulated days as they will
    arrive (simulation seconds, zone of origin).
    """

    def __init__(
        self,
        base: IDemandModel,
        times: np.ndarray,
        window: float = 3600.0,
        prior: float = 50.0,
        lo: float = 0.5,
        hi: float = 2.0,
    ):
        self._base = base
        self._times = np.sort(np.asarray(times, dtype=float))
        self._window = window
        self._prior = prior
        self._lo, self._hi = lo, hi

    def ratio(self, t_now: float) -> float:
        start = t_now - self._window
        observed = np.searchsorted(self._times, t_now) - np.searchsorted(self._times, start)
        forecast = sum(self._base.expected_requests(start, t_now).values())
        return float(np.clip((observed + self._prior) / (forecast + self._prior), self._lo, self._hi))

    def expected_requests(self, t_start: float, t_end: float, t_now: float | None = None) -> dict[int, float]:
        expected = self._base.expected_requests(t_start, t_end, t_now)
        if t_now is None:
            return expected
        r = self.ratio(t_now)
        return {zone: r * v for zone, v in expected.items()}
