"""
Figure 1: Accuracy curves — Average Accuracy after each task for all methods.

X = number of tasks learned (1 to T)
Y = Average Accuracy (%)
One curve per method with error bands.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "nullflow": "#2196F3", "ddpm_replay": "#F44336", "ewc": "#4CAF50",
    "der++": "#FF9800", "gdumb": "#9C27B0", "fine_tune": "#607D8B",
    "joint": "#000000", "latent_replay": "#795548",
}

METHOD_LABELS = {
    "nullflow": "NullFlow (Ours)", "ddpm_replay": "DDPM Replay",
    "ewc": "EWC", "der++": "DER++", "gdumb": "GDumb",
    "fine_tune": "Fine-tuning", "joint": "Joint Training",
    "latent_replay": "Latent Replay",
}


def compute_aa_curve(R):
    """Compute AA after each task from accuracy matrix R."""
    R = np.array(R)
    T = R.shape[0]
    aa_curve = []
    for i in range(T):
        aa = np.mean(R[i, :i+1])
        aa_curve.append(aa)
    return aa_curve


def generate(data, output_dir):
    """Generate Figure 1: Accuracy curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot order: fine_tune first, joint last, NullFlow highlighted
    plot_order = ["fine_tune", "ewc", "gdumb", "latent_replay",
                  "der++", "ddpm_replay", "nullflow", "joint"]

    for method in plot_order:
        R = None
        if method == "nullflow" and "nullflow" in data:
            R = np.array(data["nullflow"]["accuracy_matrix"])
        elif "baselines" in data and method in data["baselines"]:
            R = np.array(data["baselines"][method]["accuracy_matrix"])

        if R is None or R.size == 0:
            continue

        aa_curve = compute_aa_curve(R)
        T = len(aa_curve)
        x = np.arange(1, T + 1)

        color = COLORS.get(method, "#999999")
        label = METHOD_LABELS.get(method, method)

        if method == "joint":
            ax.plot(x, aa_curve, "--", color=color, label=label,
                    linewidth=1.5, alpha=0.8)
        elif method == "fine_tune":
            ax.plot(x, aa_curve, "--", color=color, label=label,
                    linewidth=1.5, alpha=0.6)
        elif method == "nullflow":
            ax.plot(x, aa_curve, "-o", color=color, label=label,
                    linewidth=2.5, markersize=6, zorder=10)
        else:
            ax.plot(x, aa_curve, "-s", color=color, label=label,
                    linewidth=1.5, markersize=4)

    ax.set_xlabel("Number of Tasks Learned")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Average Accuracy Throughout Continual Learning")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(0, 100)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig1_accuracy_curves.pdf"))
    fig.savefig(os.path.join(output_dir, "fig1_accuracy_curves.png"))
    plt.close(fig)
    print("  Saved: fig1_accuracy_curves.pdf/png")
