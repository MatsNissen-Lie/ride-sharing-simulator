"""Assignment objectives: what a trip assignment and an unserved request cost."""

from src.interfaces import (
    FeasibleRoute,
    INetworkService,
    IObjective,
    Request,
    VehicleAvailability,
)


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


class ProfitObjective(IObjective):
    """
    Maximise operator profit: fares of served riders minus variable cost per vehicle-km.

    A trip costs cost_per_km * (extra vehicle-km) - (fares of its new riders),
    so profitable trips have negative cost. Extra km = the route's km from the
    vehicle's position minus the km of the drop-offs it would drive anyway, in
    their current order. Leaving a rider unserved costs nothing, except that
    breaking a promised pickup costs `promise_penalty` (kept large so promises
    are honoured, as under the delay objective).
    """

    def __init__(
        self,
        network: INetworkService,
        cost_per_km: float,
        promise_penalty: float = 1e4,
    ):
        self._network = network
        self._cost_per_km = cost_per_km
        self._promise_penalty = promise_penalty

    def _route_km(self, start: int, nodes: list[int]) -> float:
        dist = self._network.get_distance_matrix()
        metres, here = 0.0, start
        for node in nodes:
            metres += dist[here, node]
            here = node
        return metres / 1000

    def trip_cost(
        self,
        route: FeasibleRoute,
        vehicle: VehicleAvailability,
        requests: list[Request],
    ) -> float:
        start = vehicle.available_at_node
        km = self._route_km(start, [s.node for s in route.route.stops])
        km_anyway = self._route_km(start, [s.node for s in vehicle.remaining_stops_to_serve])
        return self._cost_per_km * (km - km_anyway) - sum(r.fare for r in requests)

    def unserved_penalty(self, request: Request, current_time: float) -> float:
        return self._promise_penalty if request.promised_pickup_time is not None else 0.0
