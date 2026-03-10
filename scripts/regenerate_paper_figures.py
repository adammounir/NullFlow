#!/usr/bin/env python3
"""
Regenerate paper figures using corrected baseline results.
Outputs to paper/figures/cifar100/ and paper/figures/tinyimagenet/.
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "NullFlow": "#2196F3",
    "NCM": "#E91E63",
    "iCaRL": "#4CAF50",
    "Latent Replay": "#795548",
    "DER++": "#FF9800",
    "EWC": "#9C27B0",
    "GDumb": "#00BCD4",
    "Fine-tune": "#607D8B",
    "Joint": "#000000",
}

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PAPER_FIG_DIR = os.path.join(BASE_DIR, "paper", "figures")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, subdir, name):
    out_dir = os.path.join(PAPER_FIG_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved: {subdir}/{name}.pdf")


# ============================================================================
# CIFAR-100 Figures
# ============================================================================

def gen_cifar100_baseline_comparison():
    """Bar chart comparing all methods on CIFAR-100 (mean ± std over 3 seeds)."""
    # Corrected results
    methods = ["Fine-tune", "GDumb", "EWC", "DER++", "Latent Replay",
               "iCaRL", "NCM", "Joint", "NullFlow"]
    aa_mean = [5.17, 4.90, 5.17, 5.53, 6.87, 10.47, 11.43, 16.97, 30.21]
    aa_std  = [0.15, 0.26, 0.40, 0.21, 0.47, 0.12, 0.29, 0.78, 0.36]
    colors = [COLORS.get(m, "#999") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(methods)), aa_mean, yerr=aa_std, capsize=3,
                  color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Split-CIFAR-100 — Baseline Comparison")

    # Add value labels
    for bar, val, std in zip(bars, aa_mean, aa_std):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylim(0, 38)
    fig.tight_layout()
    save_fig(fig, "cifar100", "baseline_comparison")


def gen_cifar100_task_evolution():
    """Task accuracy evolution showing how accuracy drops as tasks are added."""
    # Load NullFlow accuracy matrix
    nf_data = load_json(os.path.join(RESULTS_DIR, "split_cifar100",
                                     "ablation_nullflow_v5c", "results.json"))
    R_nf = np.array(nf_data["accuracy_matrix"])

    # Load corrected baselines
    bl_data = load_json(os.path.join(RESULTS_DIR, "split_cifar100",
                                     "baselines_fixed", "baseline_results_seed42.json"))

    T = R_nf.shape[0]
    tasks = np.arange(1, T + 1)

    # Compute average accuracy after each task
    nf_aa = [np.mean([R_nf[i, j] for j in range(i+1)]) for i in range(T)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tasks, nf_aa, "o-", color=COLORS["NullFlow"], linewidth=2.5,
            markersize=7, label="NullFlow (Ours)", zorder=5)

    for name, label in [("ncm", "NCM"), ("icarl", "iCaRL"),
                        ("latent_replay", "Latent Replay"),
                        ("joint", "Joint"), ("fine_tune", "Fine-tune")]:
        if name in bl_data:
            R = np.array(bl_data[name]["accuracy_matrix"])
            aa_curve = [np.mean([R[i, j] for j in range(i+1)]) for i in range(T)]
            style = "--" if name == "joint" else "-"
            ax.plot(tasks, aa_curve, style, color=COLORS.get(label, "#999"),
                    linewidth=1.5, markersize=4, marker="s", label=label, alpha=0.8)

    ax.set_xlabel("Number of Tasks Learned")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Split-CIFAR-100 — Accuracy Evolution")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(tasks)
    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(0, 60)
    fig.tight_layout()
    save_fig(fig, "cifar100", "task_evolution")


def gen_cifar100_accuracy_matrix():
    """Heatmap of the accuracy matrix for NullFlow."""
    nf_data = load_json(os.path.join(RESULTS_DIR, "split_cifar100",
                                     "ablation_nullflow_v5c", "results.json"))
    R = np.array(nf_data["accuracy_matrix"])
    T = R.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xlabel("Task $j$ (evaluated)")
    ax.set_ylabel("Task $i$ (after training)")
    ax.set_title("NullFlow — Split-CIFAR-100")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(range(1, T+1))
    ax.set_yticklabels(range(1, T+1))

    # Add text annotations
    for i in range(T):
        for j in range(T):
            val = R[i, j]
            if val > 0:
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    save_fig(fig, "cifar100", "accuracy_matrix")


def gen_cifar100_forgetting():
    """Per-task forgetting analysis."""
    nf_data = load_json(os.path.join(RESULTS_DIR, "split_cifar100",
                                     "ablation_nullflow_v5c", "results.json"))
    R_nf = np.array(nf_data["accuracy_matrix"])

    bl_data = load_json(os.path.join(RESULTS_DIR, "split_cifar100",
                                     "baselines_fixed", "baseline_results_seed42.json"))

    T = R_nf.shape[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # NullFlow per-task forgetting
    nf_forg = [max(R_nf[k, j] for k in range(j, T)) - R_nf[T-1, j] for j in range(T-1)]
    tasks_forg = np.arange(1, T)
    ax.bar(tasks_forg - 0.2, nf_forg, width=0.35, color=COLORS["NullFlow"],
           label="NullFlow", alpha=0.9, edgecolor="white")

    # Best baseline (NCM) per-task forgetting
    if "ncm" in bl_data:
        R_ncm = np.array(bl_data["ncm"]["accuracy_matrix"])
        ncm_forg = [max(R_ncm[k, j] for k in range(j, T)) - R_ncm[T-1, j] for j in range(T-1)]
        ax.bar(tasks_forg + 0.2, ncm_forg, width=0.35, color=COLORS["NCM"],
               label="NCM", alpha=0.9, edgecolor="white")

    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting (%)")
    ax.set_title("Split-CIFAR-100 — Per-Task Forgetting")
    ax.legend()
    ax.set_xticks(tasks_forg)
    fig.tight_layout()
    save_fig(fig, "cifar100", "forgetting_analysis")


def gen_cifar100_multiseed():
    """Multi-seed comparison showing consistency."""
    seeds_data = {}
    for seed, suffix in [(42, "ablation_nullflow_v5c"),
                         (123, "ablation_nullflow_v5c_s123"),
                         (456, "ablation_nullflow_v5c_s456")]:
        path = os.path.join(RESULTS_DIR, "split_cifar100", suffix, "results.json")
        if os.path.exists(path):
            seeds_data[seed] = load_json(path)

    if len(seeds_data) < 3:
        print("  WARNING: Not all 3 seeds found for multi-seed comparison")
        return

    T = 10
    tasks = np.arange(1, T + 1)

    all_curves = []
    for seed, data in seeds_data.items():
        R = np.array(data["accuracy_matrix"])
        curve = [np.mean([R[i, j] for j in range(i+1)]) for i in range(T)]
        all_curves.append(curve)

    all_curves = np.array(all_curves)
    mean_curve = all_curves.mean(axis=0)
    std_curve = all_curves.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tasks, mean_curve, "o-", color=COLORS["NullFlow"], linewidth=2.5,
            markersize=7, label="NullFlow (mean)")
    ax.fill_between(tasks, mean_curve - std_curve, mean_curve + std_curve,
                    color=COLORS["NullFlow"], alpha=0.2, label="±1 std")

    for i, (seed, curve) in enumerate(zip(seeds_data.keys(), all_curves)):
        ax.plot(tasks, curve, "--", color=COLORS["NullFlow"], alpha=0.4,
                linewidth=1, label=f"Seed {seed}")

    ax.set_xlabel("Number of Tasks Learned")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Split-CIFAR-100 — Multi-Seed Stability")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(tasks)
    ax.set_xlim(0.5, T + 0.5)
    fig.tight_layout()
    save_fig(fig, "cifar100", "multiseed_comparison")


def gen_cifar100_ablation():
    """Ablation study bar chart."""
    variants = ["Full NullFlow (v5c)", "w/o FM aug (v5b)", "w/o NSP (FM only v2)",
                "w/o FM & NSP (baseline)", "NSP only", "FM only"]
    aa = [30.62, 24.82, 27.47, 25.61, 24.11, 22.16]
    fr = [27.68, 24.14, 24.93, 46.46, 46.78, 43.63]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(variants))

    # AA
    bars1 = ax1.barh(x, aa, color=COLORS["NullFlow"], alpha=0.85, edgecolor="white")
    bars1[0].set_color("#1565C0")  # Darker for full model
    ax1.set_yticks(x)
    ax1.set_yticklabels(variants, fontsize=9)
    ax1.set_xlabel("Average Accuracy (%)")
    ax1.set_title("Accuracy")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, aa):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=9, fontweight="bold")

    # FR
    bars2 = ax2.barh(x, fr, color="#F44336", alpha=0.85, edgecolor="white")
    bars2[0].set_color("#C62828")
    ax2.set_yticks(x)
    ax2.set_yticklabels(variants, fontsize=9)
    ax2.set_xlabel("Forgetting Rate (%)")
    ax2.set_title("Forgetting")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, fr):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=9, fontweight="bold")

    fig.suptitle("Split-CIFAR-100 — Ablation Study", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "cifar100", "ablation_study")


# ============================================================================
# TinyImageNet Figures
# ============================================================================

def gen_tinyimagenet_baseline_comparison():
    """Bar chart comparing all methods on TinyImageNet."""
    methods = ["Fine-tune", "GDumb", "EWC", "DER++", "Latent Replay",
               "iCaRL", "NCM", "Joint", "NullFlow"]
    aa = [4.80, 3.70, 4.60, 5.00, 6.40, 12.70, 17.90, 19.40, 60.77]
    colors = [COLORS.get(m, "#999") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(methods)), aa, color=colors, edgecolor="white",
                  linewidth=0.5, alpha=0.9)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Split-TinyImageNet — Baseline Comparison")

    for bar, val in zip(bars, aa):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylim(0, 70)
    fig.tight_layout()
    save_fig(fig, "tinyimagenet", "baseline_comparison")


def gen_tinyimagenet_accuracy_matrix():
    """Accuracy matrix for TinyImageNet."""
    path = os.path.join(RESULTS_DIR, "split_tinyimagenet", "nullflow_v5c", "results.json")
    data = load_json(path)
    R = np.array(data["accuracy_matrix"])
    T = R.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xlabel("Task $j$ (evaluated)")
    ax.set_ylabel("Task $i$ (after training)")
    ax.set_title("NullFlow — Split-TinyImageNet")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(range(1, T+1))
    ax.set_yticklabels(range(1, T+1))

    for i in range(T):
        for j in range(T):
            val = R[i, j]
            if val > 0:
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    fig.tight_layout()
    save_fig(fig, "tinyimagenet", "accuracy_matrix")


def gen_tinyimagenet_task_evolution():
    """Task evolution for TinyImageNet."""
    nf_data = load_json(os.path.join(RESULTS_DIR, "split_tinyimagenet",
                                     "nullflow_v5c", "results.json"))
    R_nf = np.array(nf_data["accuracy_matrix"])

    bl_data = load_json(os.path.join(RESULTS_DIR, "split_tinyimagenet",
                                     "baselines_fixed", "baseline_results_seed42.json"))

    T = R_nf.shape[0]
    tasks = np.arange(1, T + 1)

    nf_aa = [np.mean([R_nf[i, j] for j in range(i+1)]) for i in range(T)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tasks, nf_aa, "o-", color=COLORS["NullFlow"], linewidth=2.5,
            markersize=7, label="NullFlow (Ours)", zorder=5)

    for name, label in [("ncm", "NCM"), ("icarl", "iCaRL"),
                        ("joint", "Joint"), ("fine_tune", "Fine-tune")]:
        if name in bl_data:
            R = np.array(bl_data[name]["accuracy_matrix"])
            aa_curve = [np.mean([R[i, j] for j in range(i+1)]) for i in range(T)]
            style = "--" if name == "joint" else "-"
            ax.plot(tasks, aa_curve, style, color=COLORS.get(label, "#999"),
                    linewidth=1.5, markersize=4, marker="s", label=label, alpha=0.8)

    ax.set_xlabel("Number of Tasks Learned")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Split-TinyImageNet — Accuracy Evolution")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(tasks)
    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(0, 80)
    fig.tight_layout()
    save_fig(fig, "tinyimagenet", "task_evolution")


def gen_tinyimagenet_forgetting():
    """Forgetting analysis for TinyImageNet."""
    nf_data = load_json(os.path.join(RESULTS_DIR, "split_tinyimagenet",
                                     "nullflow_v5c", "results.json"))
    R_nf = np.array(nf_data["accuracy_matrix"])

    bl_data = load_json(os.path.join(RESULTS_DIR, "split_tinyimagenet",
                                     "baselines_fixed", "baseline_results_seed42.json"))

    T = R_nf.shape[0]
    tasks_forg = np.arange(1, T)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    nf_forg = [max(R_nf[k, j] for k in range(j, T)) - R_nf[T-1, j] for j in range(T-1)]
    ax.bar(tasks_forg - 0.2, nf_forg, width=0.35, color=COLORS["NullFlow"],
           label="NullFlow", alpha=0.9, edgecolor="white")

    if "ncm" in bl_data:
        R_ncm = np.array(bl_data["ncm"]["accuracy_matrix"])
        ncm_forg = [max(R_ncm[k, j] for k in range(j, T)) - R_ncm[T-1, j] for j in range(T-1)]
        ax.bar(tasks_forg + 0.2, ncm_forg, width=0.35, color=COLORS["NCM"],
               label="NCM", alpha=0.9, edgecolor="white")

    ax.set_xlabel("Task")
    ax.set_ylabel("Forgetting (%)")
    ax.set_title("Split-TinyImageNet — Per-Task Forgetting")
    ax.legend()
    ax.set_xticks(tasks_forg)
    fig.tight_layout()
    save_fig(fig, "tinyimagenet", "forgetting_analysis")


def gen_tinyimagenet_multiseed():
    """Placeholder multiseed for TinyImageNet (only seed 42 available)."""
    path = os.path.join(RESULTS_DIR, "split_tinyimagenet", "nullflow_v5c", "results.json")
    data = load_json(path)
    R = np.array(data["accuracy_matrix"])
    T = R.shape[0]
    tasks = np.arange(1, T + 1)

    curve = [np.mean([R[i, j] for j in range(i+1)]) for i in range(T)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tasks, curve, "o-", color=COLORS["NullFlow"], linewidth=2.5,
            markersize=7, label="NullFlow (seed 42)")
    ax.set_xlabel("Number of Tasks Learned")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Split-TinyImageNet — Accuracy Evolution")
    ax.legend(loc="upper right")
    ax.set_xticks(tasks)
    ax.set_xlim(0.5, T + 0.5)
    fig.tight_layout()
    save_fig(fig, "tinyimagenet", "multiseed_comparison")


if __name__ == "__main__":
    print("=" * 60)
    print("Regenerating Paper Figures (corrected baselines)")
    print("=" * 60)

    print("\n--- CIFAR-100 ---")
    gen_cifar100_baseline_comparison()
    gen_cifar100_task_evolution()
    gen_cifar100_accuracy_matrix()
    gen_cifar100_forgetting()
    gen_cifar100_multiseed()
    gen_cifar100_ablation()

    print("\n--- TinyImageNet ---")
    gen_tinyimagenet_baseline_comparison()
    gen_tinyimagenet_accuracy_matrix()
    gen_tinyimagenet_task_evolution()
    gen_tinyimagenet_forgetting()
    gen_tinyimagenet_multiseed()

    print("\n" + "=" * 60)
    print(f"All figures saved to {PAPER_FIG_DIR}/")
    print("=" * 60)
