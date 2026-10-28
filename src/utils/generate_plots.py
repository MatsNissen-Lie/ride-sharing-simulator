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
        "time_prepare": "Fleet Replay & Requests",
        "time_rv": "RV Graph",
        "time_assign": "ILP Assignment",
        "time_rtv": "RTV Graph",
        "time_rebal": "Rebalancing",
    }
    columns_and_labels = {k: v for k, v in columns_and_labels.items() if k in df.columns}

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

    # Combined Legend
    ax1.legend(loc="upper left")
    
    # Calculate Mean Runtime
    if "time_total_batch" in df.columns:
        mean_runtime = df["time_total_batch"].mean()
        if "time_prepare" in df.columns:
            mean_runtime += df["time_prepare"].mean()
    else:
        # Fallback if total column missing
        mean_runtime = df[runtime_cols].sum(axis=1).mean()
        
    ax1.set_title(f"Runtime Breakdown | Mean: {mean_runtime:.4f}s")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_breakdown.png"), dpi=300)
    plt.close()
    print("Generated runtime_breakdown.png")

    # 1.5. Incoming and Active Requests (Standalone)
    plt.figure(figsize=(10, 6))
    plt.plot(
        df[x_col],
        df["requests_new"],
        label="Incoming Requests (New)",
        color="black",
        linestyle="-",
        linewidth=0.5,
    )
    plt.plot(
        df[x_col],
        df["requests_active"],
        label="Active Requests (Queue)",
        color="tab:orange",
        linestyle="-",
        linewidth=0.5,
    )

    plt.xlabel(x_label)
    if "timestamp" in df.columns:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()

    plt.ylabel("Request Count")
    plt.title("Request Volume")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "request_volume.png"))
    plt.close()
    print("Generated request_volume.png")

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

    # 3. Average Delay Breakdown (Stacked Area)
    # Wait Time + Detour Delay = Total Delay
    plt.figure(figsize=(10, 6))
    if "avg_total_delay" in df.columns and "avg_wait_time" in df.columns:
        # Filter: Only plot batches where requests were actually served
        df_served = df[df["requests_delivered"] > 0].copy()

        if not df_served.empty:
            # Derived metric: Detour = Total - Wait
            df_served["avg_detour_delay"] = (
                df_served["avg_total_delay"] - df_served["avg_wait_time"]
            )
            # Ensure no negative values due to floating point or data quirks
            df_served["avg_detour_delay"] = df_served["avg_detour_delay"].clip(lower=0)  # type: ignore

            # Calculate Global Average Delay (Weighted by request count)
            total_weighted_delay = (
                df_served["avg_total_delay"] * df_served["requests_delivered"]
            ).sum()
            total_served = df_served["requests_delivered"].sum()
            global_avg_delay = (
                total_weighted_delay / total_served if total_served > 0 else 0.0
            )

            # Stacked Area Plot
            plt.stackplot(
                df_served[x_col],
                df_served["avg_wait_time"],
                df_served["avg_detour_delay"],
                labels=["Avg Wait Time", "Avg Detour Delay"],
                colors=["#3498db", "#e74c3c"],
                alpha=0.7,
            )

            # Optional: Plot Total Delay as a line on top for clarity
            # plt.plot(
            #     df_served[x_col],
            #     df_served["avg_total_delay"],
            #     label="Total Delay",
            #     color="#c92d2d",
            #     linewidth=0.3,
            #     linestyle="-",
            # )

            plt.legend(loc="upper left")
            plt.title(
                f"Average Delay Breakdown (Wait + Detour) | Mean: {global_avg_delay:.2f}s"
            )
        else:
            plt.plot([], [], label="No Data")
            plt.legend()
            plt.title("Average Delay Breakdown (Wait + Detour)")
        plt.ylabel("Time (s)")
    else:
        plt.plot([], [], label="Missing delay columns")
        plt.legend()

    plt.xlabel(x_label)
    if "timestamp" in df.columns:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()

    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(output_dir, "average_delay_over_time.png"))
    plt.close()
    print("Generated average_delay_over_time.png (Stacked)")

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
    plt.plot(
        df[x_col],
        df["requests_new"],
        label="Incoming Requests",
        color="black",
        # linestyle="--",
        alpha=0.5,
    )
    plt.plot(
        df[x_col],
        df["requests_delivered"],
        label="Served Requests",
        color="purple",
        # linewidth=1.,
        # marker=".",
    )

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
        
        # Calculate Mean Runtime (Local to this scope if needed, or largely same as global)
        if "time_total_batch" in df.columns:
            mean_runtime = df["time_total_batch"].mean()
        else:
            mean_runtime = df[runtime_cols].sum(axis=1).mean()

        ax1.set_title(f"Runtime vs Candidate Trip Sizes | Mean Runtime: {mean_runtime:.4f}s")
        ax1.legend(loc="upper left")
        # ShareX so format applied to bottom axis will affect top too technically if shared,
        # but let's apply to both or just bottom if standard sharex behavior

        # Bottom: Trip Sizes (Stacked Area)
        # Prepare data for stackplot
        trip_data = [df[col] for col in active_size_cols]
        trip_labels = [
            col.replace("candidate_trips_size_", "Size ") for col in active_size_cols
        ]

        # Use a distinct colormap for trip sizes (Set2) to contrast with runtime (standard)
        trip_colors = plt.get_cmap("Set2")(range(len(active_size_cols)))

        ax2.stackplot(
            df[x_col], trip_data, labels=trip_labels, colors=trip_colors, alpha=1
        )

        ax2.set_ylabel("Count (Candidate Trips)")
        ax2.set_xlabel(x_label)
        format_xaxis(ax2)
        ax2.grid(True, alpha=0.7)
        ax2.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "runtime_vs_trip_sizes.png"), dpi=300)
        plt.close()
        print("Generated runtime_vs_trip_sizes.png")

    # 7. Assigned Trip Sizes
    assigned_trip_cols = [c for c in df.columns if "assigned_trips_size_" in c]
    # Filter to only sizes that actually have data > 0 at some point
    active_assigned_cols = [c for c in assigned_trip_cols if df[c].sum() > 0]

    if active_assigned_cols:
        plt.figure(figsize=(10, 6))
        for col in active_assigned_cols:
            rank = col.replace("assigned_trips_size_", "")
            plt.plot(df[x_col], df[col], label=f"Size {rank} Trips", linewidth=0.5)

        plt.xlabel(x_label)
        if "timestamp" in df.columns:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.gcf().autofmt_xdate()

        plt.ylabel("Count (Assigned Trips)")
        plt.title("Assigned Trip Sizes Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "assigned_trip_sizes.png"))
        plt.close()
        print("Generated assigned_trip_sizes.png")

        # 8. Assigned Trip Sizes Distribution (Pie Chart)
        plt.figure(figsize=(10, 6))
        sums = [df[col].sum() for col in active_assigned_cols]
        total_assigned = sum(sums)

        # Create labels with percentages for the legend
        legend_labels = []
        for col, s in zip(active_assigned_cols, sums):
            rank = col.replace("assigned_trips_size_", "")
            pct = (s / total_assigned * 100) if total_assigned > 0 else 0
            legend_labels.append(f"Size {rank} ({pct:.2f}%)")

        pie_result = plt.pie(sums, labels=None, startangle=140)
        wedges = pie_result[0]

        plt.legend(
            wedges,
            legend_labels,
            title="Trip Sizes",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
        plt.title("Distribution of Assigned Trip Sizes")
        plt.axis("equal")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "assigned_trip_sizes_pie.png"))
        plt.close()
        print("Generated assigned_trip_sizes_pie.png")

    # 9. Candidate Trip Sizes Distribution (Pie Chart)
    # Using active_size_cols which was computed in section 6
    if active_size_cols:
        plt.figure(figsize=(10, 6))
        sums = [df[col].sum() for col in active_size_cols]
        total_candidate = sum(sums)

        # Create labels with percentages for the legend
        legend_labels = []
        for col, s in zip(active_size_cols, sums):
            rank = col.replace("candidate_trips_size_", "")
            pct = (s / total_candidate * 100) if total_candidate > 0 else 0
            legend_labels.append(f"Size {rank} ({pct:.2f}%)")

        pie_result = plt.pie(sums, labels=None, startangle=140)
        wedges = pie_result[0]

        plt.legend(
            wedges,
            legend_labels,
            title="Trip Sizes",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )
        plt.title("Distribution of Candidate Trip Sizes")
        plt.axis("equal")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "candidate_trip_sizes_pie.png"))
        plt.close()
        print("Generated candidate_trip_sizes_pie.png")

    # 10. Combined Trip Sizes Distribution (Side-by-Side Pie Charts)
    # Only if both have data
    if active_assigned_cols and active_size_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # --- Subplot 1: Assigned Trips ---
        sums1 = [df[col].sum() for col in active_assigned_cols]
        total1 = sum(sums1)
        legend_labels1 = []
        for col, s in zip(active_assigned_cols, sums1):
            rank = col.replace("assigned_trips_size_", "")
            pct = (s / total1 * 100) if total1 > 0 else 0
            legend_labels1.append(f"Size {rank} ({pct:.2f}%)")

        pie_result1 = ax1.pie(sums1, labels=None, startangle=140)
        wedges1 = pie_result1[0]
        ax1.legend(
            wedges1,
            legend_labels1,
            title="Sizes",
            loc="center left",
            bbox_to_anchor=(0.8, 0, 0.5, 1),
        )
        ax1.set_title("Assigned Trip Sizes")
        ax1.axis("equal")

        # --- Subplot 2: Candidate Trips ---
        sums2 = [df[col].sum() for col in active_size_cols]
        total2 = sum(sums2)
        legend_labels2 = []
        for col, s in zip(active_size_cols, sums2):
            rank = col.replace("candidate_trips_size_", "")
            pct = (s / total2 * 100) if total2 > 0 else 0
            legend_labels2.append(f"Size {rank} ({pct:.2f}%)")

        pie_result2 = ax2.pie(sums2, labels=None, startangle=140)
        wedges2 = pie_result2[0]
        ax2.legend(
            wedges2,
            legend_labels2,
            title="Sizes",
            loc="center left",
            bbox_to_anchor=(0.8, 0, 0.5, 1),
        )
        ax2.set_title("Candidate Trip Sizes")
        ax2.axis("equal")

        plt.suptitle("Comparison of Trip Size Distributions")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_trip_sizes_pie.png"))
        plt.close()
        print("Generated combined_trip_sizes_pie.png")


if __name__ == "__main__":

    PATH = Path("reports/herorun20")
    csv_file = PATH / "simulation_stats.csv"

    if csv_file.exists():
        generate_plots(csv_file)
    else:
        print(f"File not found: {csv_file}")
