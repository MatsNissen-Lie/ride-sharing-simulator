"""Utilities for visualising rider/vehicle assignments."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from matplotlib.lines import Line2D

from config import BASE_DIR, get_logger

if TYPE_CHECKING:
    from src.state.minimal import RiderVehicleMatch


logger = get_logger(__name__)


DEFAULT_PLOT_DIR = BASE_DIR / "data" / "plots"
"""Fallback directory for generated state figures."""

DEFAULT_ROUTE_COLORS = ["#2563eb", "#f97316", "#059669", "#d946ef"]
"""Palette cycled when plotting generic routes."""


def plot_routes(
    graph: nx.MultiDiGraph,
    routes: list[list[int]],
    *,
    filepath: Path | None = None,
    colors: list[str] | None = None,
    route_linewidths: list[float] | None = None,
    orig_dest_size: int = 8,
    orig_dest_color: list[str] | str | None = None,
    show: bool = True,
    close: bool = False,
    bgcolor: str = "white",
    node_size: float = 0,
    title: str | None = None,
) -> Path | None:
    """Plot one or more *routes* (lists of nodes) on *graph* and save the figure."""

    if not routes:
        raise ValueError("'routes' must contain at least one list of nodes")
    if any(len(route) == 0 for route in routes):
        raise ValueError("Each route must contain at least one node")

    if colors is None:
        colors = [
            DEFAULT_ROUTE_COLORS[idx % len(DEFAULT_ROUTE_COLORS)]
            for idx in range(len(routes))
        ]
    if route_linewidths is None:
        route_linewidths = [2.0] * len(routes)
    if orig_dest_color is None:
        orig_dest_color = colors

    fig, ax = ox.plot_graph_routes(  # type: ignore[call-arg]
        graph,
        routes=routes,
        route_colors=colors,
        route_linewidths=route_linewidths,
        orig_dest_size=orig_dest_size,
        # orig_dest_color=orig_dest_color,
        show=False,
        close=False,
        bgcolor=bgcolor,
        node_size=node_size,
    )

    if title:
        ax.set_title(title, fontsize=12)

    if filepath is not None:
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        logger.info("Saved routes plot to %s", filepath)

    if show:
        plt.show(block=False)
    if close:
        plt.close(fig)

    return filepath


def plot_nodes(
    graph: nx.MultiDiGraph,
    nodes: list[int],
    *,
    filepath: Path | None = None,
    show: bool = True,
    close: bool = False,
    title: str | None = None,
) -> None:
    """Plot the provided *nodes* on *graph* and save the figure."""

    if not nodes:
        raise ValueError("'nodes' must contain at least one node")

    missing = [node for node in nodes if node not in graph]
    if missing:
        missing_str = ", ".join(str(node) for node in missing)
        raise ValueError(f"Nodes not present in graph: {missing_str}")

    colors = [
        DEFAULT_ROUTE_COLORS[idx % len(DEFAULT_ROUTE_COLORS)]
        for idx in range(len(nodes))
    ]
    if filepath is None:
        filepath = DEFAULT_PLOT_DIR / f"{title}.png"
        filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = ox.plot_graph(  # type: ignore[call-arg]
        graph,
        node_size=0,
        node_alpha=0.0,
        edge_color="#94a3b8",
        edge_linewidth=0.5,
        bgcolor="white",
        show=False,
        close=False,
    )

    xs = [graph.nodes[node]["x"] for node in nodes]
    ys = [graph.nodes[node]["y"] for node in nodes]
    ax.scatter(
        xs,
        ys,
        c=colors,
        s=40,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    if not show:
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        logger.info("Saved nodes plot to %s", filepath)

    if title:
        ax.set_title(title, fontsize=12)

    if show:
        plt.show()
    if close:
        plt.close(fig)


def plot_assignment(
    graph: nx.MultiDiGraph,
    match: "RiderVehicleMatch",
    *,
    filepath: Path | None = None,
    show: bool = False,
    close: bool = True,
) -> Path:
    """Render the dispatch and en-route legs for *match* and return the image path."""

    if not match.dispatch.nodes or not match.enroute.nodes:
        msg = "Both dispatch and en-route segments must contain at least one node"
        raise ValueError(msg)

    if filepath is None:
        filename = f"rider_{match.rider_index}_vehicle_{match.vehicle_id}.png"
        filepath = DEFAULT_PLOT_DIR / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)

    routes = [match.dispatch.nodes, match.enroute.nodes]
    colors = ["#f97316", "#2563eb"]
    fig, ax = ox.plot_graph_routes(  # type: ignore[call-arg]
        graph,
        routes=routes,
        route_colors=colors,
        route_linewidths=[2.0, 2.0],
        orig_dest_size=8,
        # orig_dest_color=colors, #NOTE: pretty sure this worked before.
        show=False,
        close=False,
        bgcolor="white",
        node_size=0,
    )

    legend_items = [
        Line2D([0], [0], color=colors[0], lw=3.0, label="dispatch"),
        Line2D([0], [0], color=colors[1], lw=3.0, label="en route"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[0],
            markeredgecolor="none",
            markersize=6,
            label="pickup",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[1],
            markeredgecolor="none",
            markersize=6,
            label="dropoff",
        ),
    ]
    ax.legend(handles=legend_items, loc="lower left")

    ax.set_title(
        f"Rider {match.rider_index} / Vehicle {match.vehicle_id}",
        fontsize=12,
    )
    fig.savefig(filepath, dpi=200, bbox_inches="tight")

    logger.info("Saved assignment plot to %s", filepath)

    if show:
        plt.show(block=False)
    if close:
        plt.close(fig)

    return filepath


__all__ = ["plot_assignment", "plot_routes", "plot_nodes"]
