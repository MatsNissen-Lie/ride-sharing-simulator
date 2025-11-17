from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast


from config import get_logger

from src.interfaces import (
    FeasibleRoute,
    Infeasible,
    IRVGraphBuilder,
    ISingleVehicleRoutePlanner,
    Request,
    VehicleAvailability,
)


logger = get_logger(__name__)


@dataclass
class RVGraph:
    """Simple container for the pairwise request-vehicle graph.

    The graph consists of two kinds of edges:

    * **Request-Request edges**: an undirected edge exists between two
      requests if a virtual empty vehicle can pick up and drop off both
      requests within the maximum waiting and delay constraints. The edge
      is stored in :attr:`request_request` as a bidirectional relation
      between request identifiers. The associated feasibility and cost
      returned from the route planner are stored in :attr:`rr_costs` and
      :attr:`rr_routes`.

    * **Request-Vehicle edges**: a directed edge exists from a request to
      a vehicle if the request can be served by that vehicle on its own
      given the vehicle's current state (location, passengers and
      remaining route). These edges are stored in
      :attr:`request_vehicle`. The associated feasibility and cost are
      stored in :attr:`rv_costs` and :attr:`rv_routes`.

    The graph does not encode any particular semantic beyond storing
    these relations; subsequent processing (such as building the
    request-trip-vehicle graph) should interpret the data accordingly.
    """

    request_request: dict[str, set[str]] = field(default_factory=dict)
    request_vehicle: dict[str, set[str]] = field(default_factory=dict)
    rr_costs: dict[tuple[str, str], float] = field(default_factory=dict)
    rr_routes: dict[tuple[str, str], FeasibleRoute | None] = field(default_factory=dict)
    rv_costs: dict[tuple[str, str], float] = field(default_factory=dict)
    rv_routes: dict[tuple[str, str], FeasibleRoute | None] = field(default_factory=dict)

    def add_rr_edge(
        self,
        r1_id: str,
        r2_id: str,
        route: FeasibleRoute | None,
        cost: float,
    ) -> None:
        """Add an undirected request-request edge to the graph.

        This method updates both directions of the relation in
        :attr:`request_request` and stores the cost and route for
        reference. The edge is assumed to be symmetric; the same cost is
        recorded for (r1, r2) and (r2, r1).
        """

        self.request_request.setdefault(r1_id, set()).add(r2_id)
        self.request_request.setdefault(r2_id, set()).add(r1_id)
        self.rr_costs[(r1_id, r2_id)] = cost
        self.rr_costs[(r2_id, r1_id)] = cost
        self.rr_routes[(r1_id, r2_id)] = route
        self.rr_routes[(r2_id, r1_id)] = route

    def add_rv_edge(
        self,
        r_id: str,
        v_id: str,
        route: FeasibleRoute | None,
        cost: float,
    ) -> None:
        """Add a request-vehicle edge to the graph.

        This method records that request ``r_id`` can be served by
        vehicle ``v_id`` on its own. The cost and route information are
        stored separately for possible use when constructing trips and
        pricing.
        """

        self.request_vehicle.setdefault(r_id, set()).add(v_id)
        self.rv_costs[(r_id, v_id)] = cost
        self.rv_routes[(r_id, v_id)] = route


class RVGraphBuilder(IRVGraphBuilder):
    """Concrete implementation of :class:`IRVGraphBuilder`.

    This builder constructs the pairwise request-vehicle graph by
    querying a provided single vehicle route planner. For each pair of
    requests, a virtual vehicle with sufficient capacity is tested to
    determine whether both requests can be served together. For each
    request and each actual vehicle, the planner is asked to evaluate
    whether the request can be served individually given the vehicle's
    current state. The resulting feasibility information and associated
    cost are stored in an :class:`RVGraph`.

    Parameters
    ----------
    route_planner:
        Instance of a single vehicle route planner used to test
        feasibility. It must implement the
        :func:`~ISingleVehicleRoutePlanner.check_feasibility` method.

    max_virtual_capacity:
        Optional override for the capacity of the virtual vehicle used
        when testing pairs of requests. If not provided, the builder
        derives a capacity by taking the maximum capacity among the
        vehicles supplied to :meth:`build_graph`, but at least two. A
        capacity lower than two would prevent any pair of requests from
        being combined.
    """

    def __init__(
        self,
        route_planner: ISingleVehicleRoutePlanner,
        *,
        max_virtual_capacity: int | None = None,
    ) -> None:
        super().__init__(route_planner)
        self._max_virtual_capacity = max_virtual_capacity

    def _make_virtual_vehicle(
        self, request_a: Request, request_b: Request, capacity: int
    ) -> VehicleAvailability:
        """Construct a virtual vehicle for testing a pair of requests.

        The virtual vehicle is initialised so as not to violate the batch
        window check performed by the route planner. In particular, the
        vehicle's availability time is set to the earliest request time
        among the two requests and its starting node is one of the
        request's origin nodes. No passengers are on board, and there are
        no remaining stops to serve.
        """

        earliest_time = min(request_a.request_time, request_b.request_time)
        start_node = request_a.origin_node
        return VehicleAvailability(
            vehicle_id="virtual",
            capacity=capacity,
            available_at_node=start_node,
            available_from_time=earliest_time,
            passengers_onboard=frozenset(),
            remaining_stops_to_serve=tuple(),
        )

    def build_graph(
        self,
        requests: set[Request],
        vehicles: set[VehicleAvailability],
    ) -> RVGraph:
        """Construct the request-vehicle graph for the given requests and vehicles.

        Parameters
        ----------
        requests:
            Set of all ride requests awaiting assignment. Each request
            must provide a unique identifier via ``id`` as well as the
            attributes required by the route planner (e.g.
            ``origin_node`` and ``request_time``).

        vehicles:
            Set of vehicles currently available (potentially with
            passengers on board) for servicing requests. Each vehicle
            must expose an identifier via ``vehicle_id`` (or ``id``) and
            the attributes required by the route planner (e.g.
            ``capacity``, ``available_at_node``, ``available_from_time``,
            ``passengers_onboard`` and ``remaining_stops_to_serve``).

        Returns
        -------
        RVGraph
            The constructed graph encoding pairwise feasible
            request-request and request-vehicle combinations along with
            their associated costs and routes.
        """

        graph = RVGraph()

        for req in requests:
            graph.request_request.setdefault(req.id, set())
            graph.request_vehicle.setdefault(req.id, set())

        virtual_capacity = self._resolve_virtual_capacity(vehicles)

        req_list = list(requests)
        for i in range(len(req_list)):
            for j in range(i + 1, len(req_list)):
                r1 = req_list[i]
                r2 = req_list[j]
                try:
                    vehicle = self._make_virtual_vehicle(r1, r2, virtual_capacity)
                    result = self._route_planner.check_feasibility(
                        {r1, r2},
                        vehicle,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Failed RR feasibility check for %s/%s: %s",
                        r1.id,
                        r2.id,
                        exc,
                    )
                    raise RuntimeError(
                        "Unexpected error during RR feasibility check"
                    ) from exc
                    # result = Infeasible(
                    #     reason=f"exception during RR feasibility check: {exc}"
                    # )

                if getattr(result, "is_feasible", False):

                    feasible = cast(FeasibleRoute, result)  # type: ignore[assignment]
                    cost = getattr(feasible, "cost", float("inf"))
                    graph.add_rr_edge(r1.id, r2.id, feasible, cost)

        for req in requests:
            for vehicle in vehicles:
                if self._request_already_onboard(req, vehicle.passengers_onboard):
                    raise ValueError(
                        f"Request {req.id} is already onboard vehicle {vehicle.vehicle_id}"
                    )
                try:
                    result = self._route_planner.check_feasibility(
                        {req},
                        vehicle,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Failed RV feasibility check for %s/%s: %s",
                        req.id,
                        getattr(vehicle, "vehicle_id", "unknown"),
                        exc,
                    )
                    raise RuntimeError(
                        "Unexpected error during RV feasibility check"
                    ) from exc

                if getattr(result, "is_feasible", False):
                    feasible = cast(FeasibleRoute, result)  # type: ignore[assignment]
                    cost = getattr(feasible, "cost", float("inf"))
                    vehicle_id = self._vehicle_identifier(vehicle)
                    graph.add_rv_edge(req.id, vehicle_id, feasible, cost)

        return graph

    def _resolve_virtual_capacity(self, vehicles: set[VehicleAvailability]) -> int:
        if self._max_virtual_capacity is not None:
            return max(2, self._max_virtual_capacity)
        if not vehicles:
            return 2
        max_capacity = max(getattr(vehicle, "capacity", 2) for vehicle in vehicles)
        return max(2, max_capacity)

    @staticmethod
    def _request_already_onboard(
        request: Request, passengers: Iterable[object]
    ) -> bool:
        for passenger in passengers or []:
            original = getattr(passenger, "original_request", None)
            if original and getattr(original, "id", None) == request.id:
                return True
        return False

    @staticmethod
    def _vehicle_identifier(vehicle: VehicleAvailability) -> str:
        identifier = getattr(vehicle, "vehicle_id", None)
        if identifier is None:
            identifier = getattr(vehicle, "id", None)
        if identifier is None:
            raise AttributeError("VehicleAvailability lacks an identifier field")
        return str(identifier)


__all__ = ["RVGraph", "RVGraphBuilder"]


if __name__ == "__main__":
    """Ad-hoc demonstration for the RV-graph builder.

    Loads a handful of requests and vehicles using the simulation data
    provider, then instantiates the concrete route planner to evaluate
    feasibility edges. The output is a simple textual summary of the
    resulting RV-graph so developers can verify integration quickly.
    """

    from itertools import islice

    from config import BATCH_DURATION
    from src.components.batch_controller import BatchController
    from src.components.network_service import NetworkXService
    from src.components.route_planner import RoutePlanner
    from src.components.simulation_data_provider import SimulationDataProvider

    provider = SimulationDataProvider()
    network_service = NetworkXService(provider.graph)
    planner = RoutePlanner(network_service=network_service)
    rv_builder = RVGraphBuilder(planner)
    controller = BatchController(network_service)

    meta_requests = list(islice(provider.iter_requests(), 200))
    if len(meta_requests) < 2:
        raise SystemExit("Need at least two meta requests for the RV-graph demo")

    converted = sorted(
        controller.convert_meta_requests(meta_requests),
        key=lambda req: req.request_time,
    )

    demo_requests: list[Request] = []
    for seed in converted:
        window = [
            req
            for req in converted
            if 0 <= req.request_time - seed.request_time <= BATCH_DURATION
        ]
        if len(window) >= 10:
            demo_requests = window
            break

    if not demo_requests:
        raise SystemExit("Unable to find two requests within one batch window")

    request_pool = set(demo_requests)
    sim_time = max(req.request_time for req in request_pool)

    vehicle_snapshots: set[VehicleAvailability] = set()
    for vehicle in islice(provider.iter_vehicles(10), 10):
        try:
            snapshot = VehicleAvailability.from_vehicle(vehicle, sim_time)
        except NotImplementedError:
            continue
        vehicle_snapshots.add(snapshot)
        if len(vehicle_snapshots) >= 3:
            break

    if not vehicle_snapshots:
        raise SystemExit("No idle vehicles available for RV-graph demo")

    graph = rv_builder.build_graph(request_pool, vehicle_snapshots)

    total_rr_edges = (
        sum(len(neighbors) for neighbors in graph.request_request.values()) // 2
    )
    total_rv_edges = sum(len(vehicles) for vehicles in graph.request_vehicle.values())

    print("RV-graph demo summary")
    print(f"Requests considered: {len(request_pool)}")
    print(f"Vehicles considered: {len(vehicle_snapshots)}")
    print(f"Request-request edges: {total_rr_edges}")
    print(f"Request-vehicle edges: {total_rv_edges}")

    if total_rr_edges:
        sample_req = next(iter(graph.request_request))
        partners = graph.request_request[sample_req]
        print(f"Sample R-R partners for {sample_req}: {sorted(partners)}")

    for req_id, vehicles in graph.request_vehicle.items():
        if vehicles:
            print(f"Request {req_id} has feasible vehicles: {sorted(vehicles)}")
            break
