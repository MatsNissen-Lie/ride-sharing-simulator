from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def generate_plots(csv_path):
    print(f"Loading stats from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 0. Preprocessing: Datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Fix: Strip timezone info to ensure plots show local time (e.g. 16:11) instead of UTC (21:11)
        # This keeps the "wall clock" time from the string rather than converting to UTC.
        if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # type: ignore
        x_col = "timestamp"
        x_label = "Time of Day"

        # Helper to format X-axis as HH:MM
        def format_xaxis(ax):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            # Note: fig is defined later in each plot block, so this call might fail if fig is not in scope.
            # For now, assuming fig is accessible or will be handled by the context where format_xaxis is called.
            # The original code had this structure, so maintaining it.
            if "fig" in locals() or "fig" in globals():  # Check if fig is defined
                fig.autofmt_xdate()  # Rotation

    else:
        x_col = "sim_time"
        x_label = "Simulation Time (s)"

        def format_xaxis(ax):
            pass

    # Filter out warmup if needed (or just plot everything and mark it)
    # df = df[df['is_warmup'] == False]

    # Define report plots directory - USER REQUEST: Save alongside stats/metadata
    output_dir = os.path.dirname(csv_path)
    # output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(csv_path))), "report", "plots")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Runtime Breakdown (Stacked Area)
    # Columns: time_rv, time_rtv, time_assign, time_rebal

    columns_and_labels = {
        "time_rv": "RV Graph",
        "time_assign": "ILP Assignment",
        "time_rtv": "RTV Graph",
    }
    runtime_cols = list(columns_and_labels.keys())
    labels = list(columns_and_labels.values())

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis: Stacked Runtime (Default Colors)
    ax1.stackplot(df[x_col], df[runtime_cols].T, labels=labels)
    ax1.set_xlabel(x_label)
    format_xaxis(ax1)
    ax1.set_ylabel("Runtime (s)")
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.7, linestyle="--")

    # Secondary Axis: Incoming Requests
    ax2 = ax1.twinx()
    req_color = "black"
    ax2.set_ylabel("Incoming Requests (New)", color=req_color)
    ax2.plot(
        df[x_col],
        df["requests_new"],
        color=req_color,
        linestyle="-",
        linewidth=1.5,
        label="Incoming Requests",
        alpha=0.7,
    )
    ax2.tick_params(axis="y", labelcolor=req_color)

    plt.title("Computational Runtime Breakdown")

    # Combined Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_breakdown.png"), dpi=300)
    plt.close()
    print("Generated runtime_breakdown.png")

    # 2. Graph Size / Scalability (Dual Axis)
    # Columns: rv_edges, trips_count
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel(x_label)
    format_xaxis(ax1)
    ax1.set_ylabel("RV Edges", color=color)
    ax1.plot(df[x_col], df["rv_edges"], color=color, label="RV Graph Edges")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:orange"
    ax2.set_ylabel("Feasible Trips (RTV Nodes)", color=color)
    ax2.plot(df[x_col], df["trips_count"], color=color, label="Potential Trips")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Constraint Graph Scalability")
    fig.tight_layout()  # otherwise the right y-label may be slightly clipped
    plt.savefig(os.path.join(output_dir, "graph_size_complexity.png"))
    plt.close()
    print("Generated graph_size_complexity.png")

    # 3. Average Delay Over Time
    plt.figure(figsize=(10, 6))
    if "avg_total_delay" in df.columns:
        # Filter: Only plot batches where requests were actually served
        df_served = df[df["requests_delivered"] > 0]
        if not df_served.empty:
            plt.plot(
                df_served[x_col],
                df_served["avg_total_delay"],
                marker="o",
                label="Avg Delay",
            )
        else:
            plt.plot([], [], label="No Data")

        plt.title("Average Delay per Batch")
        plt.ylabel("Delay (s)")
    else:
        plt.plot([], [], label="Missing avg_total_delay")

    plt.xlabel(x_label)
    if "timestamp" in df.columns:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()

    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "average_delay_over_time.png"))
    plt.close()
    print("Generated average_delay_over_time.png")

    # 4. Fleet Utilization
    plt.figure(figsize=(10, 6))

    # Calculate total fleet size from the first row (assuming fixed fleet size)
    total_vehicles = df["vehicles_active"].iloc[0] + df["vehicles_idle"].iloc[0]

    plt.plot(df[x_col], df["vehicles_active"], label="Active Vehicles", color="green")
    # plt.plot(df[x_col], df["vehicles_idle"], label="Idle Vehicles", color="red") # removed as requested

    plt.xlabel(x_label)
    if "timestamp" in df.columns:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()

    plt.ylabel("Vehicle Count")
    plt.title(f"Fleet Utilization Over Time (Total Fleet: {total_vehicles})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fleet_utilization.png"))
    plt.close()
    print("Generated fleet_utilization.png")

    # 5. Service Throughput (Requests per Batch)
    plt.figure(figsize=(10, 6))
    
    # Calculate final completion metric for title (Served / (Served + Expired))
    total_delivered = df["requests_delivered"].sum()
    total_expired = df["requests_expired"].sum()
    total_finished = total_delivered + total_expired
    final_rate = (total_delivered / total_finished * 100) if total_finished > 0 else 0.0

    # Plot Incoming vs Served (Per Batch)
    plt.plot(df[x_col], df["requests_new"], label="Incoming Requests", color="black", linestyle="--", alpha=0.5)
    plt.plot(df[x_col], df["requests_delivered"], label="Served Requests", color="purple", linewidth=1.5, marker=".")
    
    plt.xlabel(x_label)
    if "timestamp" in df.columns:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()
    
    plt.ylabel("Requests per Batch")
    plt.title(f"Service Throughput (Completion Rate: {final_rate:.1f}%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "service_throughput.png"))
    plt.close()
    print("Generated service_throughput.png")

    # 6. Runtime vs Trip Sizes (Coupled Subplots)
    # Top: Runtime Breakdown
    # Bottom: Candidate Trip Sizes (Lines)

    trip_size_cols = [c for c in df.columns if "candidate_trips_size_" in c]
    # Filter to only sizes that actually have data > 0 at some point
    active_size_cols = [c for c in trip_size_cols if df[c].sum() > 0]

    if active_size_cols:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Top: Runtime
        # Use same styling as the standalone plot
        ax1.stackplot(df[x_col], df[runtime_cols].T, labels=labels)
        ax1.set_ylabel("Runtime (s)")
        ax1.tick_params(axis="y")
        ax1.grid(True, alpha=0.7, linestyle="--")
        ax1.set_title("Runtime vs Candidate Trip Sizes")
        ax1.legend(loc="upper left")
        # ShareX so format applied to bottom axis will affect top too technically if shared,
        # but let's apply to both or just bottom if standard sharex behavior

        # Bottom: Trip Sizes
        # Plot each size as a line
        for col in active_size_cols:
            rank = col.replace("candidate_trips_size_", "")
            ax2.plot(df[x_col], df[col], label=f"Size {rank} Trips")

        ax2.set_ylabel("Count (Candidate Trips)")
        ax2.set_xlabel(x_label)
        format_xaxis(ax2)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "runtime_vs_trip_sizes.png"), dpi=300)
        plt.close()
        print("Generated runtime_vs_trip_sizes.png")


if __name__ == "__main__":

    PATH = Path("reports/herorun1")
    csv_file = PATH / "simulation_stats.csv"

    if csv_file.exists():
        generate_plots(csv_file)
    else:
        print(f"File not found: {csv_file}")
