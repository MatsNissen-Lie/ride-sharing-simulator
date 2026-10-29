"""
Zone-level fleet MPC whose duals price the assignment (stage C, option C).

Every minute a fluid LP plans the fleet over zones x time steps (Zhang, Rossi
& Pavone 2016; Iglesias et al. 2018). The dual of a zone-step supply
constraint is what one more free vehicle there and then is worth; the joint
assignment ILP uses it as V(node, time). Reasoning and results:
research/stage-c/fleet-mpc.md.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from config.logging import get_logger
from src.components.zone_trips import ZoneTrips
from src.interfaces import (
    IDemandModel,
    INetworkService,
    IFleetPlan,
    IObjective,
    IZoneIndex,
    Request,
    VehicleAvailability,
)
from src.utils.timing import timed

logger = get_logger(__name__)


def _split(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A duration of d steps lands (1 - f) at floor(d) and f at floor(d) + 1, f = d - floor(d)."""
    lo = np.floor(steps)
    return lo.astype(np.int64), steps - lo


def fleet_lp_model(
    supply: np.ndarray,
    demand: np.ndarray,
    shares: np.ndarray,
    trip_steps: np.ndarray,
    move_steps: np.ndarray,
    move_cost: np.ndarray,
    reward: float,
    pooling: float = 1.0,
    eps_supply: float = 1e-3,
) -> tuple[np.ndarray, csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The LP of `fleet_lp_values` as (cost, A_eq, b_eq, bounds, step-0 move
    columns, their origin and destination zones). Demand enters only the
    upper bounds of the trip columns.
    """
    n_zones, n_steps = supply.shape
    row = lambda i, k: k * n_zones + i  # noqa: E731
    rows, cols, vals = [], [], []

    def arrive(col, dest, start, dur, weight):
        """Column `col` delivers `weight` vehicles to zone `dest` `dur` steps after step `start`."""
        lo, f = _split(dur)
        for k, share in ((start + lo, weight * (1 - f)), (start + lo + 1, weight * f)):
            keep = (k < n_steps) & (share > 0)
            rows.append(row(dest[keep], k[keep]))
            cols.append(col[keep])
            vals.append(-share[keep])

    zk = np.arange(n_zones * n_steps)
    z_of, k_of = zk % n_zones, zk // n_zones  # row order: k-major

    # x: one column per step and pair of zones with riders.
    xk, xi, xj = np.nonzero(shares > 0)
    x_col = np.arange(xk.size)
    rows.append(row(xi, xk)), cols.append(x_col), vals.append(np.ones(xk.size))
    arrive(x_col, xj, xk, trip_steps[xi, xj], np.ones(xk.size))

    # w: carries a vehicle to the next step.
    w_col = xk.size + zk
    rows.append(zk), cols.append(w_col), vals.append(np.ones(zk.size))
    nxt = k_of + 1 < n_steps
    rows.append(row(z_of[nxt], k_of[nxt] + 1)), cols.append(w_col[nxt]), vals.append(-np.ones(nxt.sum()))

    # m: one column per allowed pair and step.
    pi, pj = np.nonzero(np.isfinite(move_steps))
    pk = np.repeat(np.arange(n_steps), pi.size)
    mi, mj = np.tile(pi, n_steps), np.tile(pj, n_steps)
    m_col = xk.size + zk.size + np.arange(mi.size)
    rows.append(row(mi, pk)), cols.append(m_col), vals.append(np.ones(mi.size))
    arrive(m_col, mj, pk, move_steps[mi, mj], np.ones(mi.size))

    n_cols = xk.size + zk.size + mi.size
    a_eq = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(zk.size, n_cols),
    ).tocsr()
    b_eq = supply.T.ravel() + eps_supply  # k-major, like the rows
    cost = np.concatenate([
        np.full(xk.size, -pooling * reward),
        np.zeros(zk.size),
        move_cost[mi, mj],
    ])
    upper = np.concatenate([
        demand[xi, xk] * shares[xk, xi, xj] / pooling,
        np.full(zk.size + mi.size, np.inf),
    ])
    bounds = np.column_stack([np.zeros(n_cols), upper])
    first = pk == 0
    return cost, a_eq, b_eq, bounds, m_col[first], mi[first], mj[first]


def fleet_lp_values(
    supply: np.ndarray,
    demand: np.ndarray,
    shares: np.ndarray,
    trip_steps: np.ndarray,
    move_steps: np.ndarray,
    move_cost: np.ndarray,
    reward: float,
    pooling: float = 1.0,
    eps_supply: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fluid fleet LP over Z zones and K steps. Returns the dual value (Z, K) of
    one more free vehicle in zone i at the start of step k, as a saving, and
    the planned empty moves (Z, Z) from zone i to zone j in step 0.

        max  sum_ijk pooling * reward * x_ijk - sum_ijk move_cost_ij * m_ijk
        s.t. sum_j x_ijk + sum_j m_ijk + w_ik - w_i,k-1
               - sum_j,k' x_jik' [arrives at i in step k]
               - sum_j,k' m_jik' [arrives at i in step k]  = supply_ik
             0 <= x_ijk <= demand_ik * shares_kij / pooling,   m, w >= 0

    x_ijk: vehicle trips from zone i to zone j starting in step k, each
    carrying `pooling` riders on average; they free the vehicle in j
    trip_steps_ij steps later. Trips are per destination, so where vehicles
    are scarce the LP serves the most valuable trips first and the dual is
    the value of the marginal (worst served) trip class, not of an average
    rider. m_ijk: empty moves, allowed where move_steps is finite. w_ik:
    vehicles waiting into step k+1. A duration d that is not a whole number
    of steps is split, (1 - f) at floor(d) and f at floor(d) + 1, so the
    duals are linear in time between step starts. Arrivals after the last
    step are worth nothing (see FleetMpcValue for the tail).

    Degeneracy: a zone-step without vehicles has no basic variable to pin its
    dual. `eps_supply` extra vehicles in every zone-step make the LP route
    them to their best use, so every dual is the value of one more vehicle
    (the right derivative), not an arbitrary point of the dual face.

    supply, demand: (Z, K); shares: (K, Z, Z), rows sum to 1;
    trip_steps, move_steps, move_cost: (Z, Z); durations must be >= 1 step.
    """
    cost, a_eq, b_eq, bounds, m0, mi, mj = fleet_lp_model(
        supply, demand, shares, trip_steps, move_steps, move_cost, reward, pooling, eps_supply
    )
    n_zones, n_steps = supply.shape
    with timed("fleet_lp"):
        # Interior point is fast but can stop on a numerical error; the simplex
        # method then solves the same LP (same optimum, slower).
        for method in ("highs-ipm", "highs-ds"):
            res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method=method)
            if res.status == 0:
                break
            logger.warning("Fleet LP: %s failed (%s)", method, res.message)
    if res.status != 0:
        raise RuntimeError(f"Fleet LP failed: {res.message}")
    moves = np.zeros((n_zones, n_zones))
    moves[mi, mj] = res.x[m0]
    return -res.eqlin.marginals.reshape(n_steps, n_zones).T, moves


def free_at(vehicle: VehicleAvailability, times: np.ndarray) -> tuple[int, float]:
    """Node and time at which `vehicle` has driven all its remaining stops."""
    node, t = vehicle.available_at_node, vehicle.available_from_time
    for stop in vehicle.remaining_stops_to_serve:
        t += times[node, stop.node]
        node = stop.node
    return node, t


class FleetMpcValue(IFleetPlan):
    """
    V(node, t) from the fluid fleet LP (`fleet_lp_values`), re-solved every
    `resolve_every` seconds over `n_steps` steps of `step` seconds:

    - supply: idle vehicles now, plus every busy or moving vehicle in the zone
      and step where its current plan ends (known future arrivals);
    - demand: the forecast per zone and step, plus the riders in the pool now
      (step 0; promised riders too, as their pickups are re-planned every batch);
    - served riders free their vehicle at the destination zone (OD shares of
      training days) after the mean trip time; empty moves between zones up to
      `max_move` seconds apart cost IObjective.rebalancing_cost.

    Values are per rider-equivalent: the dual divided by `pooling` (riders per
    vehicle trip in the LP), so a trip of average length breaks even where the
    fleet is scarce. They are scaled by `weight`, smoothed between solves
    (new = smoothing * LP + (1 - smoothing) * previous, on absolute time) and
    interpolated linearly in time between step starts. After the last step
    they fall at the mean slope of the horizon, down to zero.
    """

    def __init__(
        self,
        network: INetworkService,
        zones: IZoneIndex,
        demand: IDemandModel,
        trips: ZoneTrips,
        objective: IObjective,
        rider_reward: float,
        step: float = 300.0,
        n_steps: int = 12,
        pooling: float = 1.0,
        weight: float = 1.0,
        smoothing: float = 0.5,
        resolve_every: float = 60.0,
        max_move: float = 900.0,
    ):
        self._network = network
        self._zones = zones
        self._demand = demand
        self._trips = trips
        self._reward = rider_reward
        self._step = step
        self._n_steps = n_steps
        self._pooling = pooling
        self._weight = weight
        self._smoothing = smoothing
        self._resolve_every = resolve_every

        self._zone_ids = np.asarray(zones.zone_ids)
        centres = [zones.center_node[int(z)] for z in self._zone_ids]
        travel = network.get_travel_time_matrix(centres, centres)
        dist = network.get_distance_matrix()[np.ix_(centres, centres)]
        allowed = (travel > 0) & (travel <= max_move)
        self._move_steps = np.where(allowed, np.maximum(travel / step, 1.0), np.inf)
        self._move_cost = np.vectorize(objective.rebalancing_cost)(travel, dist)
        self._trip_steps = np.maximum(trips.trip_time / step, 1.0)

        self._t_solved: float | None = None
        self._moves = np.zeros((self._zone_ids.size, self._zone_ids.size))  # step-0 moves of the last solve
        self._grid = np.zeros((self._zone_ids.size, n_steps))  # values at t_solved + k * step
        self._slope = 0.0

    def prepare(self, vehicles, requests, t_now):
        if self._t_solved is not None and t_now < self._t_solved + self._resolve_every:
            return
        lp, self._moves = self._plan(
            self.supply_grid(vehicles, t_now),
            self.demand_grid(requests, t_now),
            np.stack([self._trips.shares_at(t_now + k * self._step) for k in range(self._n_steps)]),
        )
        lp = lp * (self._weight / self._pooling)
        if self._t_solved is not None:
            times = t_now + self._step * np.arange(self._n_steps)
            previous = self._interpolate(np.repeat(np.arange(self._zone_ids.size), self._n_steps),
                                         np.tile(times, self._zone_ids.size))
            lp = self._smoothing * lp + (1 - self._smoothing) * previous.reshape(lp.shape)
        self._t_solved, self._grid = t_now, lp
        mean = lp.mean(axis=0)
        self._slope = max((mean[0] - mean[-1]) / (self._step * (self._n_steps - 1)), 0.0)

    def _plan(self, supply: np.ndarray, demand: np.ndarray, shares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Duals (Z, K) and step-0 moves (Z, Z) of the fleet LP for these grids."""
        return fleet_lp_values(
            supply, demand, shares, self._trip_steps, self._move_steps, self._move_cost, self._reward, self._pooling
        )

    @property
    def t_solved(self) -> float | None:
        return self._t_solved

    @property
    def moves(self) -> np.ndarray:
        return self._moves

    def values(self, nodes, times, t_now):
        if self._t_solved is None:
            return np.zeros(len(nodes))
        cols = np.searchsorted(self._zone_ids, self._zones.zones_of(nodes))
        return self._interpolate(cols, np.asarray(times, dtype=float))

    def _interpolate(self, cols: np.ndarray, times: np.ndarray) -> np.ndarray:
        last = self._n_steps - 1
        f = np.clip((times - self._t_solved) / self._step, 0.0, last)
        lo = np.minimum(np.floor(f).astype(np.int64), last - 1) if last else np.zeros(f.size, dtype=np.int64)
        frac = f - lo
        hi = np.minimum(lo + 1, last)
        inside = (1 - frac) * self._grid[cols, lo] + frac * self._grid[cols, hi]
        beyond = np.maximum(times - (self._t_solved + last * self._step), 0.0)
        return np.maximum(inside - self._slope * beyond, 0.0)

    def supply_grid(self, vehicles: set[VehicleAvailability], t_now: float) -> np.ndarray:
        times = self._network.get_l1_cache()
        ends = [free_at(v, times) for v in vehicles]
        supply = np.zeros((self._zone_ids.size, self._n_steps))
        if not ends:
            return supply
        nodes, t = zip(*ends)
        cols = np.searchsorted(self._zone_ids, self._zones.zones_of(nodes))
        lo, f = _split(np.maximum(np.array(t) - t_now, 0.0) / self._step)
        for k, w in ((lo, 1 - f), (lo + 1, f)):
            keep = k < self._n_steps
            np.add.at(supply, (cols[keep], k[keep]), w[keep])
        return supply

    def demand_grid(self, requests: set[Request], t_now: float) -> np.ndarray:
        grid = np.zeros((self._zone_ids.size, self._n_steps))
        for k in range(self._n_steps):
            start = t_now + k * self._step
            for zone, rate in self._demand.expected_requests(start, start + self._step).items():
                col = np.searchsorted(self._zone_ids, zone)
                if col < self._zone_ids.size and self._zone_ids[col] == zone:
                    grid[col, k] += rate
        waiting = [r.origin_node for r in requests]
        if waiting:
            np.add.at(grid[:, 0], np.searchsorted(self._zone_ids, self._zones.zones_of(waiting)), 1.0)
        return grid
