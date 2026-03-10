"""
Continual Learning metrics computation.

Implements standard CL evaluation metrics:
    - Average Accuracy (AA)
    - Backward Transfer (BWT)
    - Forward Transfer (FWT)
    - Forgetting Rate (FR)
    - FID (Fréchet Inception Distance) for generative quality
    - Inference time measurement

All metrics operate on the accuracy matrix R ∈ ℝ^{T×T} where
R[i][j] = accuracy on task j after learning tasks 0..i.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from scipy import linalg


def average_accuracy(R: np.ndarray) -> float:
    """
    Average Accuracy after all tasks.

    AA = (1/T) * Σ_{j=0}^{T-1} R[T-1, j]

    Args:
        R: Accuracy matrix, shape (T, T). R[i][j] = acc on task j after task i.

    Returns:
        AA in percentage.
    """
    T = R.shape[0]
    return float(np.mean(R[T - 1, :T]))


def backward_transfer(R: np.ndarray) -> float:
    """
    Backward Transfer: measures forgetting (negative) or positive backward transfer.

    BWT = (1/(T-1)) * Σ_{j=0}^{T-2} (R[T-1, j] - R[j, j])

    Negative BWT indicates forgetting.

    Args:
        R: Accuracy matrix, shape (T, T).

    Returns:
        BWT in percentage points.
    """
    T = R.shape[0]
    if T <= 1:
        return 0.0
    bwt = 0.0
    for j in range(T - 1):
        bwt += R[T - 1, j] - R[j, j]
    return float(bwt / (T - 1))


def forward_transfer(R: np.ndarray, baseline_acc: Optional[np.ndarray] = None) -> float:
    """
    Forward Transfer: measures zero-shot performance on future tasks.

    FWT = (1/(T-1)) * Σ_{j=1}^{T-1} (R[j-1, j] - baseline_j)

    If baseline_acc is None, uses 1/num_classes_per_task as random baseline.

    Args:
        R: Accuracy matrix, shape (T, T).
        baseline_acc: Baseline accuracy for each task, shape (T,).
                      If None, defaults to 0.

    Returns:
        FWT in percentage points.
    """
    T = R.shape[0]
    if T <= 1:
        return 0.0

    if baseline_acc is None:
        baseline_acc = np.zeros(T)

    fwt = 0.0
    for j in range(1, T):
        fwt += R[j - 1, j] - baseline_acc[j]
    return float(fwt / (T - 1))


def forgetting_rate(R: np.ndarray) -> float:
    """
    Forgetting Rate: maximum accuracy drop for each task.

    FR = (1/(T-1)) * Σ_{j=0}^{T-2} max_{k∈{0..T-2}} (R[k, j] - R[T-1, j])

    Args:
        R: Accuracy matrix, shape (T, T).

    Returns:
        FR in percentage points (higher = more forgetting).
    """
    T = R.shape[0]
    if T <= 1:
        return 0.0

    fr = 0.0
    for j in range(T - 1):
        max_prev = max(R[k, j] for k in range(T - 1))
        fr += max(0, max_prev - R[T - 1, j])
    return float(fr / (T - 1))


def compute_all_metrics(
    R: np.ndarray,
    baseline_acc: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all standard CL metrics from the accuracy matrix.

    Args:
        R: Accuracy matrix, shape (T, T).
        baseline_acc: Optional baseline for FWT computation.

    Returns:
        Dictionary with keys: 'AA', 'BWT', 'FWT', 'FR'.
    """
    return {
        "AA": average_accuracy(R),
        "BWT": backward_transfer(R),
        "FWT": forward_transfer(R, baseline_acc),
        "FR": forgetting_rate(R),
    }


def compute_fid(
    real_features: np.ndarray,
    generated_features: np.ndarray,
) -> float:
    """
    Compute Fréchet Inception Distance between real and generated features.

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2*(Σ_r * Σ_g)^{1/2})

    Args:
        real_features: Features from real data, shape (N, D).
        generated_features: Features from generated data, shape (M, D).

    Returns:
        FID score (lower is better).
    """
    # Compute statistics
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    # Compute FID
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fid


def compute_latent_fid(
    real_latents: torch.Tensor,
    generated_latents: torch.Tensor,
) -> float:
    """
    Compute FID directly in latent space (without Inception network).

    Useful for quick evaluation during training.

    Args:
        real_latents: Real latent vectors, shape (N, D).
        generated_latents: Generated latent vectors, shape (M, D).

    Returns:
        Latent FID score.
    """
    real_np = real_latents.detach().cpu().numpy()
    gen_np = generated_latents.detach().cpu().numpy()
    return compute_fid(real_np, gen_np)


def measure_inference_time(
    flow_model,
    n_samples: int = 1000,
    num_steps: int = 4,
    latent_dim: int = 128,
    device: str = "cuda",
    num_warmup: int = 5,
    num_trials: int = 10,
) -> float:
    """
    Measure average time to generate n_samples replay samples.

    Args:
        flow_model: Flow matching model.
        n_samples: Number of samples to generate.
        num_steps: Number of Euler ODE steps.
        latent_dim: Latent space dimension.
        device: Device.
        num_warmup: Number of warmup iterations.
        num_trials: Number of timed iterations.

    Returns:
        Average generation time in seconds.
    """
    flow_model.eval()

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            flow_model.sample(n=n_samples, class_label=0,
                            num_steps=num_steps, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    # Timed trials
    times = []
    for _ in range(num_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        start = time.time()
        with torch.no_grad():
            flow_model.sample(n=n_samples, class_label=0,
                            num_steps=num_steps, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        end = time.time()
        times.append(end - start)

    return float(np.mean(times))


class CLMetricsTracker:
    """
    Tracks accuracy matrix and metrics throughout continual learning.

    Records R[i][j] = accuracy on task j after learning tasks 0..i.
    """

    def __init__(self):
        self.accuracy_matrix: Dict[int, Dict[int, float]] = {}
        self._num_tasks = 0

    def update(self, current_task: int, task_accuracies: Dict[int, float]):
        """
        Record accuracies after learning a new task.

        Args:
            current_task: Index of the task just learned (0-based).
            task_accuracies: Dict mapping task_id → accuracy (%).
        """
        self.accuracy_matrix[current_task] = task_accuracies
        self._num_tasks = max(self._num_tasks, current_task + 1)

    def get_accuracy_matrix(self) -> List[List[float]]:
        """
        Get the accuracy matrix as a 2D list.

        Returns:
            R as list of lists, shape conceptually (T, T).
        """
        T = self._num_tasks
        R = []
        for i in range(T):
            row = []
            for j in range(T):
                acc = self.accuracy_matrix.get(i, {}).get(j, 0.0)
                row.append(acc)
            R.append(row)
        return R

    def get_accuracy_matrix_np(self) -> np.ndarray:
        """Get the accuracy matrix as a numpy array."""
        return np.array(self.get_accuracy_matrix())

    def get_current_aa(self) -> float:
        """Get current Average Accuracy."""
        R = self.get_accuracy_matrix_np()
        if R.size == 0:
            return 0.0
        return average_accuracy(R)

    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute all final metrics."""
        R = self.get_accuracy_matrix_np()
        if R.size == 0:
            return {"AA": 0.0, "BWT": 0.0, "FWT": 0.0, "FR": 0.0}
        return compute_all_metrics(R)
