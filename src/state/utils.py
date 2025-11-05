"""Shared helpers for visualising state assignments."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from config import BASE_DIR, get_logger

if TYPE_CHECKING:
    from src.state.minimal import RiderVehicleMatch


logger = get_logger(__name__)


DEFAULT_PLOT_DIR = BASE_DIR / "data" / "plots"
"""Fallback directory for generated state figures."""


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
        route_linewidths=[3.0, 3.0],
        show=False,
        close=False,
        bgcolor="white",
        node_size=0,
    )

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


__all__ = ["plot_assignment"]

