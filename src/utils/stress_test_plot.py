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
    if "breakdown" in df.columns and df["breakdown"].notna().all():
        # Expand the breakdown dict into columns
        breakdown_df = df["breakdown"].apply(pd.Series)
        # Combine with name
        plot_df = pd.concat([df["name"], breakdown_df], axis=1)
        plot_df.set_index("name", inplace=True)

        plot_df.plot(kind="bar", stacked=True, figsize=(10, 6))
        plt.ylabel("Runtime (s)")
        plt.title(f"{title} - Breakdown")
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Component")

        plt.tight_layout()
        breakdown_path = os.path.join(output_dir, "runtime_breakdown_comparison.png")
        plt.savefig(breakdown_path)
        plt.close()
        print(f"✅ Saved breakdown comparison plot to {breakdown_path}")
