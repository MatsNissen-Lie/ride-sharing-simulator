# utils/rtv_debug.py  (or wherever you keep debugging utilities)

from collections.abc import Iterable
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from src.interfaces import Request, VehicleAvailability
from src.interfaces.graphs import RTVGraph


# from src.components.rtv_graph import RTVGraph, TripId  # adapt to your actual import
# Here I assume:
# - RTVGraph.request_to_trips: dict[str, set[frozenset[str]]]
# - RTVGraph.trip_to_vehicles: dict[frozenset[str], set[str]]

NodeId = Tuple[str, str]  # e.g. ("r", "req-123"), ("v", "veh-7"), ("t", "r1,r2")


def _build_rtv_nx_graph_on_manhattan(
    rtv_graph: "RTVGraph",
    requests: Iterable[Request],
    vehicles: Iterable[VehicleAvailability],
    road_graph: nx.MultiDiGraph,
) -> tuple[nx.Graph, dict[NodeId, tuple[float, float]]]:
    """
    Build a NetworkX graph representing the RTV graph, with positions
    derived from the Manhattan road graph.

    Nodes:
      - ("r", r_id) for each request.
      - ("t", "r1|r2|...") for each trip (frozenset of request IDs).
      - ("v", v_id) for each vehicle.

    Edges:
      - ("r", r_id) -- ("t", trip_id)  if r_id ∈ trip.
      - ("t", trip_id) -- ("v", v_id)  if vehicle can serve that trip.

    Positions (pos):
      - Requests at their origin_node coordinates.
      - Vehicles at their available_at_node coordinates.
      - Trips at centroid of their requests' origin positions.

    Returns
    -------
    G : nx.Graph
        The logical RTV graph for debug/visualization.
    pos : dict[NodeId, (x, y)]
        Positions in the same coordinate system as `road_graph`.
    """
    G: nx.Graph = nx.Graph()
    pos: dict[NodeId, tuple[float, float]] = {}

    # Build id -> Request / Vehicle maps for quick lookup
    requests_by_id: Dict[str, Request] = {r.id: r for r in requests}
    vehicles_by_id: Dict[str, VehicleAvailability] = {v.vehicle_id: v for v in vehicles}

    # Helper to get (x, y) for a road node
    def node_xy(node_id: int) -> tuple[float, float]:
        data = road_graph.nodes[node_id]
        return float(data["x"]), float(data["y"])

    # --- Add request nodes ---
    for r_id in rtv_graph.request_to_trips.keys():
        req_node: NodeId = ("r", r_id)
        G.add_node(req_node, kind="request", request_id=r_id)

        req = requests_by_id.get(r_id)
        if req is not None:
            x, y = node_xy(req.origin_node)
            pos[req_node] = (x, y)

    # --- Add vehicle nodes ---
    for v_id, v_avail in vehicles_by_id.items():
        veh_node: NodeId = ("v", v_id)
        G.add_node(veh_node, kind="vehicle", vehicle_id=v_id)

        try:
            x, y = node_xy(v_avail.available_at_node)
            pos[veh_node] = (x, y)
        except KeyError:
            # If node missing from road_graph, just skip position
            pass

    # --- Add trip nodes and edges ---
    for trip, vehicle_ids in rtv_graph.trip_to_vehicles.items():
        # trip is a frozenset[str] of request IDs
        # Create a string label for the node
        trip_label = "|".join(sorted(trip))
        trip_node: NodeId = ("t", trip_label)

        if trip_node not in G:
            G.add_node(trip_node, kind="trip", trip=list(trip))

            # Compute centroid of request origins for position
            xs: list[float] = []
            ys: list[float] = []
            for r_id in trip:
                req = requests_by_id.get(r_id)
                if req is None:
                    continue
                try:
                    x, y = node_xy(req.origin_node)
                    xs.append(x)
                    ys.append(y)
                except KeyError:
                    continue

            if xs and ys:
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                pos[trip_node] = (cx, cy)

        # Connect trip to its member requests
        for r_id in trip:
            req_node: NodeId = ("r", r_id)
            if req_node in G:
                G.add_edge(req_node, trip_node)

        # Connect trip to vehicles that can serve it
        for v_id in vehicle_ids:
            veh_node: NodeId = ("v", v_id)
            if veh_node in G:
                G.add_edge(trip_node, veh_node)

    return G, pos


def plot_rtv_graph(
    rtv_graph: "RTVGraph",
    vehicle_avail_pool: set[VehicleAvailability],
    all_requests_in_pool: set[Request],
    road_graph: nx.MultiDiGraph,
):
    G, pos = _build_rtv_nx_graph_on_manhattan(
        rtv_graph=rtv_graph,
        requests=all_requests_in_pool,
        vehicles=vehicle_avail_pool,
        road_graph=road_graph,
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- 1) Draw the Manhattan road network as background -------------------
    # Use the 'x', 'y' node attributes from the road graph
    road_pos = {
        n: (float(d["x"]), float(d["y"])) for n, d in road_graph.nodes(data=True)
    }

    # Only edges; we don't really need to show road nodes
    nx.draw_networkx_edges(
        road_graph,
        pos=road_pos,
        ax=ax,
        width=0.5,
        edge_color="lightgray",
        alpha=0.7,
    )

    # --- 2) Draw RTV nodes, with different shapes/colors by kind ------------
    req_nodes = [n for n, k in G.nodes(data="kind") if k == "request"]
    trip_nodes = [n for n, k in G.nodes(data="kind") if k == "trip"]
    veh_nodes = [n for n, k in G.nodes(data="kind") if k == "vehicle"]

    # Requests: small circles
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=req_nodes,
        node_size=30,
        node_color="tab:blue",
        node_shape="o",
        label="requests",
        ax=ax,
    )

    # Trips: squares
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=trip_nodes,
        node_size=40,
        node_color="tab:orange",
        node_shape="s",
        label="trips",
        ax=ax,
    )

    # Vehicles: triangles
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=veh_nodes,
        node_size=40,
        node_color="tab:green",
        node_shape="^",
        label="vehicles",
        ax=ax,
    )

    # --- 3) Draw RTV edges with different styles ----------------------------
    # Interpret your request as:
    #   - solid lines between riders (requests) and trips
    #   - dotted (dashed) lines between vehicles and trips

    rt_edges = []  # request <-> trip
    tv_edges = []  # trip <-> vehicle

    for u, v in G.edges():
        ku = G.nodes[u]["kind"]
        kv = G.nodes[v]["kind"]
        kinds = {ku, kv}

        if kinds == {"request", "trip"}:
            rt_edges.append((u, v))
        elif kinds == {"trip", "vehicle"}:
            tv_edges.append((u, v))

    # Solid edges: requests <-> trips
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=rt_edges,
        ax=ax,
        width=1.0,
        edge_color="black",
        style="solid",
    )

    # Dotted edges: trips <-> vehicles
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=tv_edges,
        ax=ax,
        width=1.0,
        edge_color="black",
        style="dotted",  # or "dashed" if you prefer
    )

    # Optional: nicer axis behavior
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_rtv_graph_old(
    rtv_graph: "RTVGraph",
    vehicle_avail_pool: set[VehicleAvailability],
    all_requests_in_pool: set[Request],
    road_graph: nx.MultiDiGraph,
):

    G, pos = _build_rtv_nx_graph_on_manhattan(
        rtv_graph=rtv_graph,
        requests=all_requests_in_pool,
        vehicles=vehicle_avail_pool,
        road_graph=road_graph,
    )

    # Quick-and-dirty plot:
    plt.figure(figsize=(10, 10))

    # Draw only the RTV nodes/edges (not the whole road network for now)
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_size=30,
    )

    plt.show()
