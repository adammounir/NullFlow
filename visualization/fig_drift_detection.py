"""
Figure 9: Drift detection visualization.

Loss + Page-Hinkley statistic + drift detection lines.
10×3 inches.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate(data, output_dir):
    """Generate Figure 9: Drift detection visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3 * 2),
                                    sharex=True, height_ratios=[1, 1])

    # Try to load drift data
    drift_data = data.get("nullflow", {}).get("drift_detection", None)

    if drift_data is None:
        # Generate demo drift data
        rng = np.random.RandomState(42)
        n_steps = 500
        # Simulate loss with task boundaries at steps 100, 200, 300, 400
        loss = np.zeros(n_steps)
        task_boundaries = [100, 200, 300, 400]

        for i in range(n_steps):
            # Find current segment
            base = 0.3
            for tb in task_boundaries:
                if i >= tb and i < tb + 30:
                    base = 1.5 - (i - tb) * 0.04  # spike then decay
                    break
            else:
                base = 0.3 + rng.exponential(0.02)
            loss[i] = max(0.1, base + rng.randn() * 0.05)

        # Page-Hinkley statistic
        delta = 0.005
        threshold = 50
        ph_stat = np.zeros(n_steps)
        m_t = 0
        M_t = 0
        detected = []
        warmup = 100

        for i in range(n_steps):
            m_t += loss[i] - delta
            M_t = min(M_t, m_t)
            ph_stat[i] = m_t - M_t

            if i >= warmup and ph_stat[i] > threshold:
                detected.append(i)
                m_t = 0
                M_t = 0
    else:
        loss = np.array(drift_data["loss"])
        ph_stat = np.array(drift_data["ph_statistic"])
        detected = drift_data.get("detected_drifts", [])
        task_boundaries = drift_data.get("true_boundaries", [])
        threshold = drift_data.get("threshold", 50)
        n_steps = len(loss)

    steps = np.arange(n_steps)

    # Panel 1: Loss curve
    ax1.plot(steps, loss, color="#2196F3", linewidth=0.8, alpha=0.8,
             label="Streaming Loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("Task-Free Drift Detection (Page-Hinkley Test)")

    # Mark true boundaries
    for tb in task_boundaries:
        ax1.axvline(x=tb, color="#4CAF50", linestyle="--", alpha=0.5,
                    linewidth=1.5)
    # Add one for legend
    if task_boundaries:
        ax1.axvline(x=task_boundaries[0], color="#4CAF50", linestyle="--",
                    alpha=0.5, linewidth=1.5, label="True Task Boundary")

    # Mark detected drifts
    for d in detected:
        ax1.axvline(x=d, color="#F44336", linestyle=":", alpha=0.7,
                    linewidth=1.5)
    if detected:
        ax1.axvline(x=detected[0], color="#F44336", linestyle=":",
                    alpha=0.7, linewidth=1.5, label="Detected Drift")

    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Page-Hinkley statistic
    ax2.plot(steps, ph_stat, color="#9C27B0", linewidth=1.0,
             label="PH Statistic")
    ax2.axhline(y=threshold, color="#F44336", linestyle="-", linewidth=1.5,
                alpha=0.6, label=f"Threshold = {threshold}")
    ax2.fill_between(steps, 0, ph_stat, alpha=0.1, color="#9C27B0")

    for d in detected:
        ax2.axvline(x=d, color="#F44336", linestyle=":", alpha=0.5,
                    linewidth=1.0)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("PH Statistic")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "fig9_drift_detection.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig9_drift_detection.png"),
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved: fig9_drift_detection.pdf/png")
