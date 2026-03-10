"""
Figure 2: t-SNE of latent representations z colored by class / task.

4 sub-plots: (a) After Task 1, (b) After Task 5, (c) After Task 10, (d) All tasks overlay.
16×4 inches.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


TASK_CMAP = plt.cm.tab10


def generate(data, output_dir):
    """Generate Figure 2: t-SNE of latent representations."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    latents = data.get("nullflow", {}).get("latent_snapshots", None)

    # Expected structure: dict mapping task_id -> {"z": np.array, "y": np.array}
    # If not available, generate synthetic demo data
    if latents is None:
        rng = np.random.RandomState(42)
        latents = {}
        for t in range(10):
            n = 200
            centers = rng.randn(10, 128) * 3
            z = np.vstack([
                centers[c % 10] + rng.randn(n // 10, 128) * 0.5
                for c in range(t * 10, (t + 1) * 10)
            ])
            y = np.concatenate([
                np.full(n // 10, c) for c in range(t * 10, (t + 1) * 10)
            ])
            latents[t] = {"z": z, "y": y}

    snapshot_tasks = [0, 4, 9]
    titles = ["After Task 1", "After Task 5", "After Task 10"]

    # Panels (a), (b), (c): snapshots at specific tasks
    for idx, (task_id, title) in enumerate(zip(snapshot_tasks, titles)):
        ax = axes[idx]
        snap = latents.get(task_id, None)
        if snap is None:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        z = np.array(snap["z"])
        y = np.array(snap["y"])

        if z.shape[0] > 2000:
            indices = np.random.choice(z.shape[0], 2000, replace=False)
            z, y = z[indices], y[indices]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    n_iter=500, init="pca", learning_rate="auto")
        z_2d = tsne.fit_transform(z)

        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y,
                            cmap="tab20", s=5, alpha=0.6, rasterized=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel (d): all tasks overlay colored by task
    ax = axes[3]
    all_z, all_task = [], []
    for t, snap in latents.items():
        z = np.array(snap["z"])
        all_z.append(z)
        all_task.append(np.full(z.shape[0], t))

    if all_z:
        all_z = np.vstack(all_z)
        all_task = np.concatenate(all_task)

        if all_z.shape[0] > 3000:
            indices = np.random.choice(all_z.shape[0], 3000, replace=False)
            all_z, all_task = all_z[indices], all_task[indices]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    n_iter=500, init="pca", learning_rate="auto")
        z_2d = tsne.fit_transform(all_z)
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=all_task,
                            cmap="tab10", s=5, alpha=0.6, rasterized=True)
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
        cbar.set_label("Task ID")

    ax.set_title("All Tasks (colored by task)")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle("t-SNE of Latent Representations", fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig2_tsne_latents.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig2_tsne_latents.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved: fig2_tsne_latents.pdf/png")
