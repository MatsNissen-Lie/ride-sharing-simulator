"""Rebalancing that carries out the fleet MPC's planned empty moves."""

import numpy as np

from src.components.fleet_mpc import FleetMpcValue
from src.components.zones import ZoneIndex
from src.interfaces import INetworkService, IRebalancingSolver, Route, Stop


class MpcRebalancingSolver(IRebalancingSolver):
    """
    Carries out the fleet LP's empty moves (Zhang & Pavone's MPC: apply the
    first step of the plan). The joint ILP prices moves with the LP's duals,
    but where the LP moves vehicles the duals make the move worth exactly its
    cost, so the ILP is indifferent and the number of moves is left to
    chance. This step sends round(m_ij) idle vehicles from zone i to the
    centre of zone j, those nearest to it, in the batch right after each LP
    solve. Vehicles already moving keep going.
    """

    def __init__(self, network: INetworkService, zones: ZoneIndex, value: FleetMpcValue):
        self._network = network
        self._zones = zones
        self._value = value
        self._zone_ids = np.asarray(zones.zone_ids)
        self._centres = np.array([zones.center_node[int(z)] for z in self._zone_ids])

    def solve(self, idle_vehicles, unassigned_requests, t_now, en_route=()):
        if self._value.t_solved != t_now or not idle_vehicles:
            return {}
        idle = sorted(idle_vehicles, key=lambda v: v.vehicle_id)
        nodes = [v.available_at_node for v in idle]
        zone_of = np.searchsorted(self._zone_ids, self._zones.zones_of(nodes))
        travel = self._network.get_travel_time_matrix(nodes, list(self._centres))
        dist = self._network.get_distance_matrix()
        routes: dict[str, Route] = {}
        moves = np.round(self._value.moves)
        for i, j in zip(*np.nonzero(moves)):
            free = [k for k in np.nonzero(zone_of == i)[0] if idle[k].vehicle_id not in routes]
            for k in sorted(free, key=lambda k: travel[k, j])[: int(moves[i, j])]:
                stop = Stop.rebalancing(int(self._centres[j]), f"zone{int(self._zone_ids[j])}")
                routes[idle[k].vehicle_id] = Route(
                    (stop,), float(travel[k, j]), float(dist[nodes[k], self._centres[j]]), 0.0
                )
        return routes
