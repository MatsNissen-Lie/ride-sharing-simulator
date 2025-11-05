"""Plot sampled pickup locations on top of the Manhattan graph."""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import osmnx as ox
import polars as pl

from config import BASE_DIR, format_log_block, get_logger
from src.state_representation.manhatten_fast import ensure_travel_times, load_graph
from src.state_representation.vehicle_sampling import sample_vehicle_positions

logger = get_logger(__name__)

RIDER_DATA_PATH = BASE_DIR / "data" / "riders" / "fhv_tripdata_2025-01_enriched.parquet"
DEFAULT_OUTPUT = BASE_DIR / "data" / "plots" / "pickup_sample.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise sampled pickup locations or vehicle positions"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of pickup samples to plot (default: 100)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Inclusive start timestamp (e.g. 2025-01-02T00:00:00)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Inclusive end timestamp (e.g. 2025-01-03T12:00:00)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to save the figure (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--vehicles",
        action="store_true",
        help="Plot sampled vehicle positions instead of rider pickups",
    )
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=100,
        help="Number of vehicles to sample when --vehicles is set (default: 100)",
    )
    return parser


def parse_time(value: Optional[str]) -> Optional[pl.Datetime]:
    if value is None:
        return None
    try:
        ts = pl.Series([value]).str.strptime(pl.Datetime, strict=False).item()
    except pl.exceptions.ComputeError as exc:  # pragma: no cover - runtime validation
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return ts


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    start_ts = parse_time(args.start)
    end_ts = parse_time(args.end)

    if args.vehicles:
        sample = None
    else:
        logger.info("Loading enriched rider data from %s", RIDER_DATA_PATH)
        trips = pl.read_parquet(RIDER_DATA_PATH)
        if trips.is_empty():
            logger.warning("No trips available; nothing to plot")
            return

        trips = trips.with_columns(
            pl.col("pickup_datetime").cast(pl.Datetime(time_unit="us"))
        )

        if start_ts is not None:
            trips = trips.filter(pl.col("pickup_datetime") >= pl.lit(start_ts))
        if end_ts is not None:
            trips = trips.filter(pl.col("pickup_datetime") <= pl.lit(end_ts))

        if trips.is_empty():
            logger.warning("No trips left after applying the time window")
            return

        sample_size = max(1, min(args.count, trips.height))
        sample = trips.sample(n=sample_size, seed=args.seed).to_pandas()

    logger.info("Loading Manhattan graph for plotting")
    graph = ensure_travel_times(load_graph())
    graph_proj = ox.project_graph(graph)

    graph = ensure_travel_times(load_graph())
    graph_proj = ox.project_graph(graph)

    if args.vehicles:
        vehicle_count = max(1, args.vehicle_count)
        vehicles = sample_vehicle_positions(graph_proj, vehicle_count, seed=args.seed)
        subtitle = f"Vehicles: {vehicle_count} | Seed: {args.seed}"

        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        fig.suptitle(subtitle)
        ox.plot_graph(
            graph_proj,
            node_size=0,
            edge_color="#cccccc",
            bgcolor="white",
            ax=ax,
            show=False,
            close=False,
        )

        ax.scatter(
            vehicles["x"].to_numpy(),
            vehicles["y"].to_numpy(),
            s=25,
            c="#7570b3",
            edgecolors="black",
            linewidths=0.3,
            alpha=0.9,
            label="Vehicle start",
        )
        ax.legend(loc="upper right")

        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        summary = [
            f"Vehicle plot saved to {output_path}",
            f"Vehicles visualised: {vehicle_count}",
            f"Seed: {args.seed}",
        ]
        logger.info(format_log_block("\n".join(summary)))
        return

    assert sample is not None
    sample_size_display = sample.shape[0]

    def fmt_ts(value: Optional[pl.Datetime]) -> str:
        if value is None:
            return "-∞"
        return pl.Series([value]).dt.strftime("%Y-%m-%d %H:%M:%S").item()

    start_repr = fmt_ts(start_ts)
    end_repr = fmt_ts(end_ts)
    subtitle = f"Count: {sample_size_display} | Start: {start_repr} | End: {end_repr}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(subtitle)
    panel_titles = ("Raw pickup samples", "Snapped pickup nodes")
    for ax, title in zip(axes, panel_titles):
        ox.plot_graph(
            graph_proj,
            node_size=0,
            edge_color="#cccccc",
            bgcolor="white",
            ax=ax,
            show=False,
            close=False,
        )
        ax.set_title(title)

    axes[0].scatter(
        sample["pu_x"],
        sample["pu_y"],
        s=25,
        c="#1b9e77",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        label="Raw pickup points",
    )
    axes[0].legend(loc="upper right")

    node_positions = {
        node: (data["x"], data["y"]) for node, data in graph_proj.nodes(data=True)
    }
    sample_nodes = sample["orig_node"].dropna().astype(int)
    xs: list[float] = []
    ys: list[float] = []
    for node in sample_nodes:
        coords = node_positions.get(node)
        if coords is None:
            continue
        xs.append(coords[0])
        ys.append(coords[1])

    axes[1].scatter(
        xs,
        ys,
        s=25,
        c="#d95f02",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        label="Snapped pickup nodes",
    )
    axes[1].legend(loc="upper right")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    summary = [
        f"Sample plot saved to {output_path}",
        f"Trips visualised: {sample_size_display}",
    ]
    if start_ts is not None or end_ts is not None:
        summary.append(f"Applied time window: {start_repr} → {end_repr}")
    summary.append(f"Seed: {args.seed}")
    logger.info(format_log_block("\n".join(summary)))


if __name__ == "__main__":
    main()
