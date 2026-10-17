import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import Any
from matplotlib.patches import Patch


def plot_stress_test_comparison(
    results: list[dict[str, Any]],
    output_dir: str,
    title: str = "RTV Build Time Comparison",
):
    """
    Plots a comparison of RTV build times specifically.
    Grouped by scenario, comparing the two builders.
    """
    if not results:
        print("No results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    # Convert results into a more usable format for plotting RTV times specifically
    # We look for 'breakdown' -> 'RTV Graph'
    if "breakdown" in df.columns:
        # Extract RTV time safely
        def get_rtv_time(breakdown_val):
            if isinstance(breakdown_val, dict):
                return breakdown_val.get("RTV Graph", 0.0)
            return 0.0

        df["rtv_time"] = df["breakdown"].apply(get_rtv_time)
    else:
        # Fallback if no breakdown, use total time but warn?
        # Or just assume total is what we want if breakdown is missing (unlikely given previous context)
        df["rtv_time"] = df["total_time"]

    plt.figure(figsize=(12, 7))

    # Determine colors and labels if 'builder' info is present
    if "builder" in df.columns and "scenario" in df.columns:
        # Grouped Bar Chart Logic

        # Get unique scenarios in order of appearance
        scenarios = df["scenario"].unique()
        builders = df["builder"].unique()

        # Setup positions
        x = np.arange(len(scenarios))
        width = 0.35  # width of each bar

        # Define colors
        builder_colors = {
            "RTVGraphBuilder": "#3498db",  # Blue
            "TripCentricRTVGraphBuilder": "#e67e22",  # Orange
        }

        # Plot bars for each builder type found
        for i, builder in enumerate(builders):
            # Filter data
            builder_data = []
            for s in scenarios:
                row = df[(df["scenario"] == s) & (df["builder"] == builder)]
                if not row.empty:
                    builder_data.append(row.iloc[0]["rtv_time"])
                else:
                    builder_data.append(0)

            # Calculate offset
            offset = (i - (len(builders) - 1) / 2) * width

            color = builder_colors.get(builder, "gray")
            rects = plt.bar(x + offset, builder_data, width, label=builder, color=color)

            # Add labels
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    plt.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        height,
                        f"{height:.2f}s",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        # Cast to list to satisfy type checker
        plt.xticks(x, list(scenarios), rotation=0)

        # Custom Legend
        legend_elements = [
            Patch(
                facecolor=builder_colors["RTVGraphBuilder"],
                label="Standard (Vehicle-Centric)",
            ),
            Patch(
                facecolor=builder_colors["TripCentricRTVGraphBuilder"],
                label="Trip-Centric",
            ),
        ]
        plt.legend(handles=legend_elements)

    else:
        # Legacy/Fallback behavior
        x_pos = range(len(df))
        bars = plt.bar(x_pos, df["rtv_time"], color="skyblue")
        plt.xticks(x_pos, df["name"].tolist(), rotation=45, ha="right")

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
            )

    plt.ylabel("RTV Build Time (s)")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "rtv_build_time_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved RTV build time comparison plot to {output_path}")


if __name__ == "__main__":
    import glob
    import json

    # Update this path to point to your specific results folder
    # Example: "reports/rtv_comparison_2023-12-20_13-10-05"
    path_to_results_folder = "reports/rtv_comparison_best"

    if not os.path.exists(path_to_results_folder):
        print(
            f"Warning: Directory {path_to_results_folder} does not exist. Please set 'path_to_results_folder' to a valid directory."
        )
    else:
        # Look for comparison_results.json
        json_files = glob.glob(os.path.join(path_to_results_folder, "*.json"))

        all_results = []
        for jf in json_files:
            if "metadata" in jf:
                continue
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_results.extend(data)
                    elif isinstance(data, dict):
                        # If individual results are saved separately
                        all_results.append(data)
            except Exception as e:
                print(f"Error loading {jf}: {e}")

        if all_results:
            print(f"Loaded {len(all_results)} result entries.")
            plot_stress_test_comparison(
                all_results, path_to_results_folder, title="Re-generated RTV Comparison"
            )
        else:
            print("No valid JSON result files found.")
