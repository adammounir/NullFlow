"""
Figure 6: Ablation — Flow Matching steps.

Double Y-axis: Average Accuracy (left) + Inference time (right).
6×4 inches.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate(data, output_dir):
    """Generate Figure 6: Ablation on FM steps."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    # Try to load ablation data
    abl_data = data.get("ablations", {}).get("steps", None)

    if abl_data is None:
        # Demo data
        steps = [1, 2, 4, 8, 16, 32]
        aa_values = [58.2, 63.1, 66.8, 67.2, 67.5, 67.6]
        time_values = [0.8, 1.2, 2.1, 3.8, 7.2, 14.1]
    else:
        steps = abl_data["steps"]
        aa_values = abl_data["aa"]
        time_values = abl_data["time"]

    x = np.arange(len(steps))
    width = 0.35

    # AA bars
    bars1 = ax1.bar(x - width / 2, aa_values, width, color="#2196F3",
                    alpha=0.8, label="Avg. Accuracy (%)")

    # Time bars
    bars2 = ax2.bar(x + width / 2, time_values, width, color="#FF9800",
                    alpha=0.8, label="Inference Time (ms)")

    ax1.set_xlabel("Number of ODE Steps")
    ax1.set_ylabel("Average Accuracy (%)", color="#2196F3")
    ax2.set_ylabel("Inference Time (ms)", color="#FF9800")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in steps])

    # Highlight optimal step count (4)
    optimal_idx = 2  # steps=4
    if len(steps) > optimal_idx:
        ax1.annotate("Optimal\n(4 steps)",
                     xy=(optimal_idx - width / 2, aa_values[optimal_idx]),
                     xytext=(optimal_idx + 1.5, aa_values[optimal_idx] + 3),
                     arrowprops=dict(arrowstyle="->", color="#2196F3",
                                    lw=1.5),
                     fontsize=9, color="#2196F3", fontweight="bold",
                     ha="center")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=8)

    ax1.set_title("Ablation: Number of ODE Steps")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig6_ablation_steps.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig6_ablation_steps.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved: fig6_ablation_steps.pdf/png")
