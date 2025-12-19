import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_comparisons(results_map: dict[str, str], output_dir: str):
    """
    Generates comparative plots for multiple simulation runs.
    results_map: {scenario_name: csv_path}
    """
    os.makedirs(output_dir, exist_ok=True)

    dfs = {}
    for name, path in results_map.items():
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
        else:
            print(f"Warning: CSV not found for {name} at {path}")

    if not dfs:
        print("No data found for comparison.")
        return

    # Prepare Summary Data
    summary_data = []
    
    for name, df in dfs.items():
        # Even if there are multiple batches, we sum/mean them for a high-level comparison
        total_runtime = df["time_total_batch"].sum()
        total_delivered = df["requests_delivered"].sum()
        
        # Weighted average delay (if multiple batches) or just the value
        # Delay is usually "avg_total_delay" per batch.
        # To get global avg delay: sum(avg_delay * delivered) / sum(delivered)
        mask = df["requests_delivered"] > 0
        if mask.any():
            avg_delay = (df.loc[mask, "avg_total_delay"] * df.loc[mask, "requests_delivered"]).sum() / df.loc[mask, "requests_delivered"].sum()
        else:
            avg_delay = 0

        summary_data.append({
            "Scenario": name,
            "Runtime (s)": total_runtime,
            "Throughput": total_delivered,
            "Avg Delay (s)": avg_delay
        })

    if not summary_data:
        return

    summary_df = pd.DataFrame(summary_data).set_index("Scenario")

    # 1. Runtime Comparison (Bar Chart) - REQUESTED
    plt.figure(figsize=(10, 6))
    summary_df["Runtime (s)"].plot(kind="bar", color="skyblue")
    plt.ylabel("Total Runtime (s)")
    plt.title("Comparison: Simulation Runtime")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_runtime_bar.png"))
    plt.close()

    # 2. Throughput Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    summary_df["Throughput"].plot(kind="bar", color="lightgreen")
    plt.ylabel("Total Requests Served")
    plt.title("Comparison: Throughput")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_throughput_bar.png"))
    plt.close()

    # 3. Delay Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    summary_df["Avg Delay (s)"].plot(kind="bar", color="salmon")
    plt.ylabel("Average Delay (s)")
    plt.title("Comparison: Average Trip Delay")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_delay_bar.png"))
    plt.close()

    # 4. Occupancy Distribution (Grouped Bar Chart)
    comparison_data = []
    size_cols = [f"assigned_trips_size_{i}" for i in range(1, 4)]
    
    for name, df in dfs.items():
        total_assigned = df[size_cols].sum().sum()
        if total_assigned > 0:
            distribution = (df[size_cols].sum() / total_assigned) * 100
            distribution.index = ["1 Pax", "2 Pax", "3+ Pax"]
            distribution.name = name
            comparison_data.append(distribution)

    if comparison_data:
        comp_df = pd.concat(comparison_data, axis=1).T
        # Plot grouped bar chart
        comp_df.plot(kind="bar", figsize=(10, 6), width=0.8)
        plt.ylabel("Share of Trips (%)")
        plt.title("Comparison: Occupancy Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Trip Size")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "compare_occupancy_bar.png"))
        plt.close()

    print(f"Comparative plots generated in {output_dir}")
