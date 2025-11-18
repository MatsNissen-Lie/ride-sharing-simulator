# utils/plot_rtv_graph.py

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Hashable

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as osmnx

from src.interfaces import Request, VehicleAvailability
from src.interfaces import RTVGraph  # adjust import to your layout

REQ_COLOR = "black"
VEH_COLOR = "green"


def plot_rtv_graph_on_manhattan(
    rtv_graph: RTVGraph,
    requests: Iterable[Request],
    vehicles: Iterable[VehicleAvailability],
    road_graph: nx.MultiDiGraph,
    drop_singleton_trips: bool = True,
    dotted_vehicle_edges: bool = False,
):
    """
    Plot the RTV-graph on top of the Manhattan road network.

    - Requests are plotted at their origin_node.
    - Vehicles are plotted at their available_at_node.
    - For each (Trip T, Vehicle v) pair:
      * All R-R edges (between requests in T) are drawn as solid lines
        in a single color unique to that (T, v).
      * Exactly one R-V edge is drawn: from v to the first pickup request
        in the optimal route for (T, v), in the same color.

    Overlapping edges (same R-R or R-V pair used in multiple (T, v))
    are slightly offset so that all colors are visible.
    """
    # Base plot of the Manhattan network
    fig, ax = osmnx.plot_graph(  # type: ignore[call-arg]
        road_graph,
        show=False,
        close=False,
        bgcolor="white",
        node_size=0,
        edge_linewidth=0.5,
        dpi=300 * 2,
    )

    # Build lookup maps
    requests_by_id: Dict[str, Request] = {r.id: r for r in requests}
    vehicles_by_id: Dict[str, VehicleAvailability] = {v.vehicle_id: v for v in vehicles}

    # Helper to get coordinates for a road node
    def node_xy(node_id: int) -> Tuple[float, float]:
        data = road_graph.nodes[node_id]
        return float(data["x"]), float(data["y"])

    # --- Plot request nodes ---
    req_xs, req_ys = [], []
    for r_id, req in requests_by_id.items():
        try:
            x, y = node_xy(req.origin_node)
        except KeyError:
            continue
        req_xs.append(x)
        req_ys.append(y)

    ax.scatter(
        req_xs,
        req_ys,
        s=2,
        marker="o",
        zorder=1,
        label="Requests",
        color=REQ_COLOR,
    )

    # --- Plot vehicle nodes ---
    veh_xs, veh_ys = [], []
    for v in vehicles_by_id.values():
        try:
            x, y = node_xy(v.available_at_node)
        except KeyError:
            continue
        veh_xs.append(x)
        veh_ys.append(y)

    ax.scatter(
        veh_xs,
        veh_ys,
        s=2,
        marker="s",
        zorder=1,
        label="Vehicles",
        color=VEH_COLOR,
    )

    # --- Build edge records grouped by logical pair ---

    class EdgeRecord:
        __slots__ = ("x1", "y1", "x2", "y2", "color", "is_rr")

        def __init__(self, x1, y1, x2, y2, color, is_rr: bool):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.color = color
            self.is_rr = is_rr

    edges_by_key: Dict[Tuple[Hashable, ...], list[EdgeRecord]] = {}

    cmap = plt.get_cmap("tab20")
    trips = list(rtv_graph.trip_to_vehicles.keys())

    pair_idx = 0  # color index for (trip, vehicle) pairs

    for trip in trips:
        trip_req_ids = list(trip)
        if drop_singleton_trips and len(trip_req_ids) <= 1:
            continue

        vehicle_ids = rtv_graph.trip_to_vehicles.get(trip, set())
        for v_id in vehicle_ids:
            v = vehicles_by_id.get(v_id)
            if v is None:
                continue
                # 1) require that we can plot the vehicle
            # try:
            #     xv, yv = node_xy(v.available_at_node)
            # except KeyError:
            #     continue

            # # 2) require that we have a route for (trip, v)
            # route = rtv_graph.get_route(trip, v_id)
            # if route is None:
            #     continue

            # # 3) find first pickup in this trip
            # first_req_id = None
            # for stop in route.route.stops:
            #     if stop.is_pickup and stop.request_id in trip:
            #         first_req_id = stop.request_id
            #         break
            # if first_req_id is None:
            #     continue

            # Unique color per (trip, vehicle) pair
            color = cmap(pair_idx % cmap.N)
            pair_idx += 1

            # ---- R-R edges for this (trip, v) pair ----
            for i in range(len(trip_req_ids)):
                r1_id = trip_req_ids[i]
                r1 = requests_by_id.get(r1_id)
                if r1 is None:
                    continue
                try:
                    x1, y1 = node_xy(r1.origin_node)
                except KeyError:
                    continue

                for j in range(i + 1, len(trip_req_ids)):
                    r2_id = trip_req_ids[j]
                    r2 = requests_by_id.get(r2_id)
                    if r2 is None:
                        continue
                    try:
                        x2, y2 = node_xy(r2.origin_node)
                    except KeyError:
                        continue

                    s_r1, s_r2 = sorted((r1_id, r2_id))
                    key = ("rr", s_r1, s_r2)
                    rec = EdgeRecord(x1, y1, x2, y2, color, is_rr=True)
                    edges_by_key.setdefault(key, []).append(rec)

            # ---- R-V edge: only to first pickup in the route ----
            try:
                xv, yv = node_xy(v.available_at_node)
            except KeyError:
                continue

            route = rtv_graph.get_route(trip, v_id)
            if route is None:
                continue

            # Find first pickup in the route that belongs to this trip
            first_req_id: str | None = None
            for stop in route.route.stops:
                if stop.is_pickup and stop.request_id in trip:
                    first_req_id = stop.request_id
                    break

            if first_req_id is None:
                continue

            req = requests_by_id.get(first_req_id)
            if req is None:
                continue

            try:
                xr, yr = node_xy(req.origin_node)
            except KeyError:
                continue

            key = ("rv", first_req_id, v_id)
            rec = EdgeRecord(xr, yr, xv, yv, color, is_rr=False)
            edges_by_key.setdefault(key, []).append(rec)

    # --- Draw edges with offsets for overlapping pairs ---

    import math

    offset_base = 0.0003  # adjust based on map scale

    for key, recs in edges_by_key.items():
        n = len(recs)
        if n == 0:
            continue

        for idx, rec in enumerate(recs):
            dx = rec.x2 - rec.x1
            dy = rec.y2 - rec.y1
            length = math.hypot(dx, dy)
            if length == 0:
                ux, uy = 0.0, 0.0
            else:
                ux = -dy / length
                uy = dx / length

            # spread overlapping edges around base line
            k = idx - (n - 1) / 2
            offset = offset_base * k

            ox = ux * offset
            oy = uy * offset

            x1o = rec.x1 + ox
            y1o = rec.y1 + oy
            x2o = rec.x2 + ox
            y2o = rec.y2 + oy

            if rec.is_rr:
                linestyle = "-"
            else:
                linestyle = ":" if dotted_vehicle_edges else "-"

            ax.plot(
                [x1o, x2o],
                [y1o, y2o],
                linestyle=linestyle,
                linewidth=0.5,
                alpha=0.9,
                zorder=4 if rec.is_rr else 3,
                color=rec.color,
            )

    ax.legend(loc="upper right")
    return fig, ax


def plot_rtv_graph(
    rtv_graph: RTVGraph,
    vehicle_avail_pool: Iterable[VehicleAvailability],
    all_requests_in_pool: Iterable[Request],
    manhattan_graph: nx.MultiDiGraph,
    title: str = "RTV Graph on Manhattan Network",
    datetime: datetime | None = None,
    output_dir: Path | None = None,
    drop_singleton_trips: bool = True,
    dotted_vehicle_edges: bool = False,
):
    """
    High-level helper: plot RTV graph and optionally save as SVG.

    Parameters
    ----------
    drop_singleton_trips : bool
        If True (default), trips of length 1 (|T| == 1) are not drawn.
    dotted_vehicle_edges : bool
        If True, R-V edges are drawn dotted; otherwise they are solid.
    """
    fig, ax = plot_rtv_graph_on_manhattan(
        rtv_graph=rtv_graph,
        requests=all_requests_in_pool,
        vehicles=vehicle_avail_pool,
        road_graph=manhattan_graph,
        drop_singleton_trips=drop_singleton_trips,
        dotted_vehicle_edges=dotted_vehicle_edges,
    )

    if output_dir is not None:
        datetime_str = datetime.strftime("%Y_%m_%d_%H:%M:%S") if datetime else ""
        file_name = title.replace(" ", "_") + f"_{datetime_str}.svg"
        output_path = output_dir / file_name
        fig.savefig(output_path, format="svg", bbox_inches="tight")

    plt.show(block=True)
    return fig, ax
