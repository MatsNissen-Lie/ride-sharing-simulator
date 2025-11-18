# utils/plot_rv_graph.py

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from src.interfaces import Request, VehicleAvailability
from src.interfaces import RVGraph  # adjust import to your layout

# Consistent colors
RR_COLOR = "tab:blue"  # R-R edges
RV_COLOR = "tab:orange"  # R-V edges
REQ_COLOR = "black"  # request nodes (was "black")
VEH_COLOR = "green"  # vehicle nodes


def plot_rv_graph_on_manhattan(
    rv_graph: RVGraph,
    requests: Iterable[Request],
    vehicles: Iterable[VehicleAvailability],
    road_graph: nx.MultiDiGraph,
):
    """
    Plot the RV-graph on top of the Manhattan road network.

    - Requests are plotted at their origin_node.
    - Vehicles are plotted at their available_at_node.
    - R-R edges are solid lines between request nodes.
    - R-V edges are dotted lines between request and vehicle nodes.
    """
    # Base plot of the Manhattan network
    fig, ax = ox.plot_graph(  # type: ignore[call-arg]
        road_graph,
        show=False,
        close=False,
        bgcolor="white",
        node_size=0,
        edge_linewidth=0.5,
        # dpi=300 * 2,
    )

    # Build lookup maps
    requests_by_id: Dict[str, Request] = {r.id: r for r in requests}
    vehicles_by_id: Dict[str, VehicleAvailability] = {v.vehicle_id: v for v in vehicles}

    # Helper to get coordinates for a road node
    def node_xy(node_id: int) -> Tuple[float, float]:
        data = road_graph.nodes[node_id]
        return float(data["x"]), float(data["y"])

    # --- Plot request nodes ---
    req_xs = []
    req_ys = []
    for r_id in rv_graph.get_all_request_nodes():
        req = requests_by_id.get(r_id)
        if req is None:
            continue
        try:
            x, y = node_xy(req.origin_node)
        except KeyError:
            continue
        req_xs.append(x)
        req_ys.append(y)

    ax.scatter(
        req_xs,
        req_ys,
        s=3,
        marker="o",
        zorder=5,
        label="Requests",
        color=REQ_COLOR,
    )

    # --- Plot vehicle nodes ---
    veh_xs = []
    veh_ys = []
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
        s=3,
        marker="s",
        zorder=6,
        label="Vehicles",
        color=VEH_COLOR,
    )

    # --- Plot R-R edges (solid, one color) ---
    drawn_rr = set[tuple[str, str]]()
    for r1_id, neighbors in rv_graph.request_request.items():
        r1 = requests_by_id.get(r1_id)
        if r1 is None:
            continue
        try:
            x1, y1 = node_xy(r1.origin_node)
        except KeyError:
            continue

        for r2_id in neighbors:
            # Avoid drawing each undirected edge twice
            s_r1, s_r2 = sorted((r1_id, r2_id))
            key: tuple[str, str] = (s_r1, s_r2)
            if key in drawn_rr:
                continue
            drawn_rr.add(key)

            r2 = requests_by_id.get(r2_id)
            if r2 is None:
                continue
            try:
                x2, y2 = node_xy(r2.origin_node)
            except KeyError:
                continue

            ax.plot(
                [x1, x2],
                [y1, y2],
                linestyle="-",  # solid
                linewidth=0.5,
                alpha=0.9,
                zorder=4,
                color=RR_COLOR,
            )

    # --- Plot R-V edges (dotted, one color) ---
    for r_id, vehicle_ids in rv_graph.request_vehicle.items():
        req = requests_by_id.get(r_id)
        if req is None:
            continue
        try:
            xr, yr = node_xy(req.origin_node)
        except KeyError:
            continue

        for v_id in vehicle_ids:
            v = vehicles_by_id.get(v_id)
            if v is None:
                continue
            try:
                xv, yv = node_xy(v.available_at_node)
            except KeyError:
                continue

            ax.plot(
                [xr, xv],
                [yr, yv],
                linestyle=":",  # dotted
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
                color=RV_COLOR,
            )

    ax.legend(loc="upper right")
    return fig, ax


def plot_rv_graph(
    rv_graph: RVGraph,
    vehicle_avail_pool: Iterable[VehicleAvailability],
    all_requests_in_pool: Iterable[Request],
    manhattan_graph: nx.MultiDiGraph,
    title: str = "RV Graph on Manhattan Network",
    datetime: datetime | None = None,
    output_dir: Path | None = None,
):
    """
    High-level helper: plot RV graph and optionally save as SVG.
    """
    fig, ax = plot_rv_graph_on_manhattan(
        rv_graph=rv_graph,
        requests=all_requests_in_pool,
        vehicles=vehicle_avail_pool,
        road_graph=manhattan_graph,
    )

    if output_dir is not None:
        # Format time like 2025_06_15_15:30:30
        datetime_str = datetime.strftime("%Y_%m_%d_%H:%M:%S") if datetime else ""
        file_name = title.replace(" ", "_") + f"_{datetime_str}.svg"
        output_path = output_dir / file_name
        fig.savefig(output_path, format="svg", bbox_inches="tight")

    # You can keep block=False if you want non-blocking in scripts
    plt.show(block=False)

    return fig, ax
