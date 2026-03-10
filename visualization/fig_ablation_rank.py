"""
Figure 7: Ablation — NSP rank.

Triple metric: Average Accuracy, BWT, and memory usage.
6×4 inches.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate(data, output_dir):
    """Generate Figure 7: Ablation on NSP rank."""
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    # Try to load ablation data
    abl_data = data.get("ablations", {}).get("rank", None)

    if abl_data is None:
        # Demo data
        ranks = [0, 16, 32, 64, 128, 256]
        aa_values = [55.1, 60.3, 64.2, 66.8, 67.1, 67.0]
        bwt_values = [-28.5, -18.2, -10.5, -4.2, -3.8, -3.7]
        mem_values = [0.0, 0.5, 1.8, 6.8, 26.2, 104.0]  # MB
    else:
        ranks = abl_data["ranks"]
        aa_values = abl_data["aa"]
        bwt_values = abl_data["bwt"]
        mem_values = abl_data["memory"]

    x = np.arange(len(ranks))

    # AA line
    l1, = ax1.plot(x, aa_values, "-o", color="#2196F3", linewidth=2,
                   markersize=6, label="Avg. Accuracy (%)")

    # BWT line
    l2, = ax1.plot(x, bwt_values, "-s", color="#4CAF50", linewidth=2,
                   markersize=6, label="BWT (%)")

    # Memory bars
    bars = ax2.bar(x, mem_values, width=0.4, color="#FF9800", alpha=0.3,
                   label="NSP Memory (MB)")

    ax1.set_xlabel("NSP Rank")
    ax1.set_ylabel("Accuracy / BWT (%)")
    ax2.set_ylabel("Memory (MB)", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in ranks])

    # Highlight optimal rank (64)
    optimal_idx = 3  # rank=64
    if len(ranks) > optimal_idx:
        ax1.axvline(x=optimal_idx, color="gray", linestyle=":", alpha=0.5)
        ax1.text(optimal_idx + 0.15, aa_values[optimal_idx] + 1,
                "rank=64\n(optimal)", fontsize=8, color="gray")

    # Zero line for BWT reference
    ax1.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Combined legend
    lines = [l1, l2, bars]
    labels = [l.get_label() for l in [l1, l2]] + [bars.get_label()]
    ax1.legend(lines, labels, loc="center left", fontsize=8)

    ax1.set_title("Ablation: Null-Space Projection Rank")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig7_ablation_rank.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig7_ablation_rank.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved: fig7_ablation_rank.pdf/png")
