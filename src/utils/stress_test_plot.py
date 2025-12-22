import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict, Any


def plot_stress_test_comparison(
    results: list[Dict[str, Any]],
    output_dir: str,
    title: str = "Simulation Runtime Comparison",
):
    """
    Plots a comparison of runtimes for different scenarios.

    Args:
        results: List of dictionaries. Each dict must have:
                 - 'name': str (scenario name)
                 - 'total_time': float (total batch duration)
                 - 'breakdown': dict (optional, breakdown of times like {'RV': 1.2, 'RTV': 3.4 ...})
        output_dir: Directory to save the plot.
    """
    if not results:
        print("No results to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # 1. Total Runtime Comparison (Bar Chart)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        df["name"], df["total_time"], color=["#3498db", "#e74c3c", "#2ecc71"]
    )

    plt.ylabel("Total Batch Runtime (s)")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "runtime_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved runtime comparison plot to {output_path}")

    # 2. Stacked Breakdown (if available)
    # Check if 'breakdown' column exists and has data
    if "breakdown" in df.columns and bool(df["breakdown"].notna().all()):
        # Expand the breakdown dict into columns
        breakdown_df = df["breakdown"].apply(pd.Series)

        # Filter out rebal time if present
        cols_to_drop = [c for c in breakdown_df.columns if "rebal" in str(c).lower()]
        if cols_to_drop:
            breakdown_df.drop(columns=cols_to_drop, inplace=True)

        # Combine with name
        plot_df = pd.concat([df["name"], breakdown_df], axis=1)
        plot_df.set_index("name", inplace=True)

        plot_df.plot(kind="bar", stacked=True, figsize=(10, 6))
        plt.ylabel("Runtime (s)")
        plt.title(f"{title} - Breakdown")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Component")

        # lets add height for the bars only at the top
        for i, total in enumerate(plot_df.sum(axis=1)):
            plt.text(
                i,
                total,
                f"{total:.2f}s",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        breakdown_path = os.path.join(output_dir, "runtime_breakdown_comparison.png")
        plt.savefig(breakdown_path)
        plt.close()
        print(f"✅ Saved breakdown comparison plot to {breakdown_path}")


if __name__ == "__main__":
    import glob
    import json
    import sys

    # You can change this manually or pass as arg
    # Example usage: python src/utils/stress_test_plot.py reports/batch_test_2023-12-20_...
    if len(sys.argv) > 1:
        path_to_results_folder = sys.argv[1]
    else:
        path_to_results_folder = "reports/constraint_test"  # Default fallback

    print(f"Looking for results in: {path_to_results_folder}")

    if not os.path.exists(path_to_results_folder):
        print(f"❌ Directory {path_to_results_folder} does not exist.")
    else:
        # Look for comparison_results.json first
        json_files = glob.glob(
            os.path.join(path_to_results_folder, "comparison_results.json")
        )
        # If not found, look for any json
        if not json_files:
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
                        all_results.append(data)
            except Exception as e:
                print(f"Error loading {jf}: {e}")

        if all_results:
            print(f"Loaded {len(all_results)} result entries.")
            plot_stress_test_comparison(
                all_results,
                path_to_results_folder,
                title="Runtime Comparison",
            )
        else:
            print("No valid JSON result files found.")
