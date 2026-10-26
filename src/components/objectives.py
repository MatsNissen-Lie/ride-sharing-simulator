"""Assignment objectives: what a trip assignment and an unserved request cost."""

from src.interfaces import FeasibleRoute, IObjective, Request, VehicleAvailability


class DelayObjective(IObjective):
    """
    Alonso-Mora et al. (2017): minimise total passenger delay.

    A trip costs the marginal delay it adds (the trip solver's route cost).
    Leaving a request unserved costs a large base penalty that grows with its
    waiting time, doubled once it has been promised a pickup (SI Appendix,
    Section IV: the cost of ignoring a matched request is increased).
    """

    def __init__(self, penalty_base: float = 1e5, penalty_growth_rate: float = 10.0):
        self._penalty_base = penalty_base
        self._penalty_growth_rate = penalty_growth_rate

    def trip_cost(
        self,
        route: FeasibleRoute,
        vehicle: VehicleAvailability,
        requests: list[Request],
    ) -> float:
        return route.cost

    def unserved_penalty(self, request: Request, current_time: float) -> float:
        wait_time = current_time - request.request_time
        penalty = self._penalty_base + wait_time * self._penalty_growth_rate
        if request.promised_pickup_time is not None:
            penalty *= 2.0
        return penalty
