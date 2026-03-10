#!/usr/bin/env python3
"""
NullFlow — Publication Figure Generator.

Generates publication-quality figures from training results and baselines:
    1. Accuracy matrix heatmap
    2. Task accuracy evolution curves
    3. Baseline comparison bar chart
    4. Forgetting analysis per task

Usage:
    python scripts/generate_figures.py --results_dir results/split_cifar100/
"""

import argparse
import json
import os
import sys
import numpy as np

# Use non-interactive backend for server/CI environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Publication-quality settings
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Consistent color palette
COLORS = {
    "NullFlow v5c": "#2196F3",
    "NullFlow v2": "#64B5F6",
    "NullFlow": "#2196F3",
    "Baseline (ER)": "#BDBDBD",
    "Fine-tuning": "#9E9E9E",
    "Joint": "#4CAF50",
    "EWC": "#FF9800",
    "DER++": "#E91E63",
    "GDumb": "#9C27B0",
    "Latent Replay": "#00BCD4",
    "iCaRL": "#FF5722",
    "NCM": "#607D8B",
    "DDPM Replay": "#795548",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate NullFlow figures")
    parser.add_argument("--results_dir", type=str,
                       default="results/split_cifar100/")
    parser.add_argument("--output_dir", type=str,
                       default="paper_assets/figures/")
    return parser.parse_args()


def load_nullflow_results(results_dir):
    """Load NullFlow training results."""
    path = os.path.join(results_dir, "results.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def load_baseline_results(results_dir):
    """Load baseline comparison results (try multiple filenames)."""
    candidates = [
        os.path.join(results_dir, "baselines", "baseline_results.json"),
        os.path.join(results_dir, "baselines", "baseline_results_seed42.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    print(f"WARNING: No baseline results found in {results_dir}/baselines/")
    return None


# =========================================================================
# Figure 1: Accuracy Matrix Heatmap
# =========================================================================

def plot_accuracy_matrix(results, output_dir):
    """Plot accuracy matrix as heatmap (Figure 1)."""
    R = np.array(results["accuracy_matrix"])
    T = R.shape[0]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Mask upper triangle (tasks not yet seen)
    mask = np.triu(np.ones_like(R, dtype=bool), k=1)
    R_masked = np.ma.array(R, mask=mask)

    im = ax.imshow(R_masked, cmap="YlOrRd", vmin=0, vmax=80, aspect="equal")

    # Annotate cells
    for i in range(T):
        for j in range(T):
            if not mask[i, j]:
                color = "white" if R[i, j] < 30 else "black"
                ax.text(j, i, f"{R[i,j]:.0f}", ha="center", va="center",
                       fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("Task $j$ (evaluated on)")
    ax.set_ylabel("After learning task $i$")
    ax.set_title("NullFlow — Accuracy Matrix $R_{i,j}$ (%)")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"T{i+1}" for i in range(T)])
    ax.set_yticklabels([f"T{i+1}" for i in range(T)])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")

    path = os.path.join(output_dir, "accuracy_matrix.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# Figure 2: Task Accuracy Evolution
# =========================================================================

def plot_task_evolution(results, output_dir):
    """Plot how each task's accuracy evolves as new tasks are learned."""
    R = np.array(results["accuracy_matrix"])
    T = R.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, T))

    for j in range(T):
        # Task j accuracy after each subsequent task
        accs = [R[i, j] for i in range(j, T)]
        steps = list(range(j + 1, T + 1))
        ax.plot(steps, accs, "o-", color=cmap[j], markersize=4,
               label=f"Task {j+1}", linewidth=1.5)

    # Average accuracy line
    aa_per_step = []
    for i in range(T):
        row_accs = [R[i, j] for j in range(i + 1)]
        aa_per_step.append(np.mean(row_accs))
    ax.plot(range(1, T + 1), aa_per_step, "k--", linewidth=2.5,
           label="Avg. Accuracy", markersize=0)

    ax.set_xlabel("After learning task $t$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Task Accuracy Evolution")
    ax.set_xticks(range(1, T + 1))
    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(0, 80)
    ax.legend(ncol=3, loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "task_evolution.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# Figure 3: Baseline Comparison Bar Chart
# =========================================================================

def plot_baseline_comparison(nullflow_results, baseline_results, output_dir):
    """Plot bar chart comparing NullFlow against baselines."""
    methods = {}

    # NullFlow
    if nullflow_results is not None:
        methods["NullFlow"] = nullflow_results["metrics"]

    # Baselines
    name_map = {
        "fine_tune": "Fine-tuning",
        "joint": "Joint",
        "ewc": "EWC",
        "der++": "DER++",
        "gdumb": "GDumb",
        "latent_replay": "Latent Replay",
        "ddpm_replay": "DDPM Replay",
        "icarl": "iCaRL",
        "ncm": "NCM",
    }

    if baseline_results is not None:
        for key, data in baseline_results.items():
            display_name = name_map.get(key, key)
            methods[display_name] = data["metrics"]

    # Sort by AA (ascending for visual clarity)
    sorted_methods = sorted(methods.items(), key=lambda x: x[1]["AA"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, (metric, title) in enumerate([
        ("AA", "Average Accuracy ↑"),
        ("FR", "Forgetting Rate ↓"),
        ("BWT", "Backward Transfer ↑"),
    ]):
        ax = axes[idx]
        names = [m[0] for m in sorted_methods]
        values = [m[1][metric] for m in sorted_methods]
        colors = [COLORS.get(n, "#607D8B") for n in names]

        bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)

        # Add value labels
        for bar, val in zip(bars, values):
            offset = 1 if val >= 0 else -1
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                   f"{val:.1f}%", va="center", fontsize=8, fontweight="bold")

        ax.set_xlabel(f"{metric} (%)")
        ax.set_title(title)
        ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "baseline_comparison.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# Figure 4: Per-task Forgetting Analysis
# =========================================================================

def plot_forgetting_analysis(results, output_dir):
    """Plot how much each task forgets over time."""
    R = np.array(results["accuracy_matrix"])
    T = R.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Compute forgetting per task: max accuracy - final accuracy
    forgetting = []
    initial_accs = []
    final_accs = []

    for j in range(T - 1):  # Last task has no forgetting
        initial = R[j, j]  # Accuracy right after learning
        final = R[T - 1, j]  # Accuracy after all tasks
        forgetting.append(initial - final)
        initial_accs.append(initial)
        final_accs.append(final)

    x = np.arange(1, T)
    width = 0.35

    bars1 = ax.bar(x - width / 2, initial_accs, width, label="After learning",
                   color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x + width / 2, final_accs, width, label="After all tasks",
                   color="#F44336", alpha=0.8)

    # Add forgetting arrows
    for i in range(len(forgetting)):
        ax.annotate("", xy=(x[i] + width / 2, final_accs[i]),
                   xytext=(x[i] - width / 2, initial_accs[i]),
                   arrowprops=dict(arrowstyle="->", color="black",
                                 lw=1.5, connectionstyle="arc3,rad=0.2"))
        ax.text(x[i], max(initial_accs[i], final_accs[i]) + 2,
               f"−{forgetting[i]:.0f}", ha="center", fontsize=7,
               color="black", fontweight="bold")

    ax.set_xlabel("Task")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Task Forgetting Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in x])
    ax.legend()
    ax.set_ylim(0, max(initial_accs) + 15)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(output_dir, "forgetting_analysis.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# Figure 5: Multi-Seed Comparison with Error Bars
# =========================================================================

def plot_multiseed_comparison(all_results, output_dir):
    """Plot bar chart with error bars from multi-seed experiments.

    Args:
        all_results: dict of {method_name: {"seeds": {seed: metrics_dict}}}
    """
    if not all_results:
        print("  No multi-seed results to plot.")
        return

    # Compute mean ± std for each method
    method_stats = {}
    for method, data in all_results.items():
        seeds_data = data["seeds"]
        aa_vals = [s["AA"] for s in seeds_data.values()]
        fr_vals = [s["FR"] for s in seeds_data.values()]
        bwt_vals = [s["BWT"] for s in seeds_data.values()]
        method_stats[method] = {
            "AA_mean": np.mean(aa_vals), "AA_std": np.std(aa_vals),
            "FR_mean": np.mean(fr_vals), "FR_std": np.std(fr_vals),
            "BWT_mean": np.mean(bwt_vals), "BWT_std": np.std(bwt_vals),
            "n_seeds": len(aa_vals),
        }

    # Sort by AA (ascending)
    sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]["AA_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # --- AA chart ---
    ax = axes[0]
    names = [m[0] for m in sorted_methods]
    aa_means = [m[1]["AA_mean"] for m in sorted_methods]
    aa_stds = [m[1]["AA_std"] for m in sorted_methods]
    colors = [COLORS.get(n, "#607D8B") for n in names]

    bars = ax.barh(names, aa_means, xerr=aa_stds, color=colors,
                   edgecolor="white", height=0.6, capsize=3)
    for bar, mean, std in zip(bars, aa_means, aa_stds):
        ax.text(mean + std + 0.5, bar.get_y() + bar.get_height() / 2,
               f"{mean:.1f}±{std:.1f}", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Average Accuracy (%)")
    ax.set_title("Average Accuracy ↑ (mean ± σ)")
    ax.grid(axis="x", alpha=0.3)

    # --- FR chart ---
    ax = axes[1]
    fr_means = [m[1]["FR_mean"] for m in sorted_methods]
    fr_stds = [m[1]["FR_std"] for m in sorted_methods]
    # Sort by FR ascending (lower is better)
    sorted_fr = sorted(zip(names, fr_means, fr_stds, colors), key=lambda x: x[1])
    fr_names = [s[0] for s in sorted_fr]
    fr_means_sorted = [s[1] for s in sorted_fr]
    fr_stds_sorted = [s[2] for s in sorted_fr]
    fr_colors = [s[3] for s in sorted_fr]

    bars = ax.barh(fr_names, fr_means_sorted, xerr=fr_stds_sorted,
                   color=fr_colors, edgecolor="white", height=0.6, capsize=3)
    for bar, mean, std in zip(bars, fr_means_sorted, fr_stds_sorted):
        ax.text(mean + std + 0.5, bar.get_y() + bar.get_height() / 2,
               f"{mean:.1f}±{std:.1f}", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Forgetting Rate (%)")
    ax.set_title("Forgetting Rate ↓ (mean ± σ)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "multiseed_comparison.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print table
    print("\n  Multi-Seed Results Summary:")
    print(f"  {'Method':<20} {'AA (mean±σ)':>14} {'FR (mean±σ)':>14} {'n':>4}")
    print("  " + "-" * 56)
    for name, stats in sorted(method_stats.items(), key=lambda x: -x[1]["AA_mean"]):
        print(f"  {name:<20} {stats['AA_mean']:>6.2f}±{stats['AA_std']:<5.2f} "
              f"{stats['FR_mean']:>6.2f}±{stats['FR_std']:<5.2f} {stats['n_seeds']:>4}")


# =========================================================================
# Figure 6: Ablation Bar Chart
# =========================================================================

def plot_ablation(results_dir, output_dir):
    """Plot NullFlow ablation results (FM, NSP, Herding contributions)."""
    ablation_configs = {
        "Baseline (ER)": "ablation_baseline",
        "FM only": "ablation_fm_v2",
        "NSP only": "ablation_nsp_only",
        "FM + NSP (v2)": "ablation_nullflow_v2",
        "FM + NSP + Herding (v5c)": "ablation_nullflow_v5c",
    }

    methods = {}
    for display_name, dirname in ablation_configs.items():
        path = os.path.join(results_dir, dirname, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            methods[display_name] = data["metrics"]

    if not methods:
        print("  No ablation results found.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(methods.keys())
    aa_vals = [methods[n]["AA"] for n in names]
    colors = ["#BDBDBD", "#FFA726", "#7E57C2", "#42A5F5", "#2196F3"][:len(names)]

    bars = ax.bar(names, aa_vals, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, aa_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
               f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("NullFlow Ablation Study — Component Contributions")
    ax.set_ylim(0, max(aa_vals) + 5)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "ablation_study.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# Collect multi-seed results
# =========================================================================

def collect_multiseed_nullflow(results_dir):
    """Collect NullFlow results across seeds from directory naming convention."""
    # NullFlow multi-seed dirs: ablation_nullflow_v5c, ..._s123, ..._s456
    # Also check nullflow_v5c dir for TinyImageNet
    configs = {
        "NullFlow v5c": [
            ("ablation_nullflow_v5c", 42),
            ("ablation_nullflow_v5c_s123", 123),
            ("ablation_nullflow_v5c_s456", 456),
            ("nullflow_v5c", 42),
            ("nullflow_v5c_s123", 123),
            ("nullflow_v5c_s456", 456),
        ],
        "NullFlow v2": [
            ("ablation_nullflow_v2", 42),
            ("ablation_nullflow_v2_s123", 123),
            ("ablation_nullflow_v2_s456", 456),
        ],
        "Baseline (ER)": [
            ("ablation_baseline", 42),
            ("ablation_baseline_s123", 123),
            ("ablation_baseline_s456", 456),
        ],
    }

    results = {}
    for method, runs in configs.items():
        seeds = {}
        for dirname, seed in runs:
            path = os.path.join(results_dir, dirname, "results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                seeds[seed] = data["metrics"]
        if seeds:
            results[method] = {"seeds": seeds}
    return results


def collect_multiseed_baselines(results_dir, seeds=(42, 123, 456)):
    """Collect baseline results across seeds."""
    baselines_dir = os.path.join(results_dir, "baselines")
    if not os.path.isdir(baselines_dir):
        return {}

    name_map = {
        "fine_tune": "Fine-tuning",
        "joint": "Joint",
        "ewc": "EWC",
        "der++": "DER++",
        "gdumb": "GDumb",
        "latent_replay": "Latent Replay",
        "icarl": "iCaRL",
        "ncm": "NCM",
    }

    # Collect per-seed files
    baseline_data = {}  # {method_key: {seed: metrics}}
    for seed in seeds:
        path = os.path.join(baselines_dir, f"baseline_results_seed{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for key, val in data.items():
                if key not in baseline_data:
                    baseline_data[key] = {}
                baseline_data[key][seed] = val["metrics"]

    results = {}
    for key, seeds_data in baseline_data.items():
        display_name = name_map.get(key, key)
        results[display_name] = {"seeds": seeds_data}
    return results


# =========================================================================
# Main
# =========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)

    # Load single-seed NullFlow results (for matrix/evolution plots)
    # Try multiple directory names
    nullflow = None
    for dirname in ["ablation_nullflow_v5c", "nullflow_v5c"]:
        nullflow = load_nullflow_results(
            os.path.join(args.results_dir, dirname)
        )
        if nullflow is not None:
            break
    baselines = load_baseline_results(args.results_dir)

    if nullflow:
        print("\n[1/6] Accuracy Matrix Heatmap")
        plot_accuracy_matrix(nullflow, args.output_dir)

        print("\n[2/6] Task Accuracy Evolution")
        plot_task_evolution(nullflow, args.output_dir)

        print("\n[3/6] Forgetting Analysis")
        plot_forgetting_analysis(nullflow, args.output_dir)

    if nullflow and baselines:
        print("\n[4/6] Single-Seed Baseline Comparison")
        plot_baseline_comparison(nullflow, baselines, args.output_dir)

    # Multi-seed comparison
    print("\n[5/6] Multi-Seed Comparison (with error bars)")
    nullflow_ms = collect_multiseed_nullflow(args.results_dir)
    baseline_ms = collect_multiseed_baselines(args.results_dir)
    all_ms = {**nullflow_ms, **baseline_ms}
    plot_multiseed_comparison(all_ms, args.output_dir)

    # Ablation
    print("\n[6/6] Ablation Study")
    plot_ablation(args.results_dir, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
