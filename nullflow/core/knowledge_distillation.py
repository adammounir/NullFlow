"""
Knowledge Distillation utilities for continual learning.

Implements the KL-divergence based distillation loss used during the
Sleep-REM phase to transfer knowledge from the old classifier to the
new one via generated replay samples.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    temperature: float = 2.0,
    seen_classes: list = None,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss (KL divergence with temperature).

    KD_loss = KL(softmax(logits_student / T) || softmax(logits_teacher / T)) * T²

    The T² scaling ensures that the magnitudes of the gradients produced by
    the soft targets are roughly the same as those produced by the hard targets.

    Args:
        logits_student: Logits from the student (new) classifier, shape (B, C).
        logits_teacher: Logits from the teacher (old) classifier, shape (B, C).
        temperature: Softmax temperature (default 2.0). Higher T → softer distributions.
        seen_classes: List of class indices to include in KD. If None, use all.
            When provided, logits are sliced to only these classes, focusing
            KD on meaningful signal instead of noise from unseen class logits.

    Returns:
        Scalar KD loss.
    """
    assert logits_student.shape == logits_teacher.shape, \
        f"Shape mismatch: student {logits_student.shape} vs teacher {logits_teacher.shape}"
    assert temperature > 0, f"Temperature must be positive, got {temperature}"

    # Mask KD to seen classes only — avoids diluting the distillation signal
    # with noise from unseen-class logits (which are near-random).
    if seen_classes is not None and len(seen_classes) < logits_student.shape[1]:
        idx = torch.tensor(sorted(seen_classes), device=logits_student.device, dtype=torch.long)
        logits_student = logits_student[:, idx]
        logits_teacher = logits_teacher[:, idx]

    # Soft distributions
    p_student = F.log_softmax(logits_student / temperature, dim=1)
    p_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # KL divergence: KL(student || teacher)
    loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (temperature ** 2)

    return loss


def feature_distillation_loss(
    features_student: torch.Tensor,
    features_teacher: torch.Tensor,
) -> torch.Tensor:
    """
    Feature-level distillation loss (MSE between intermediate features).

    Args:
        features_student: Features from student model, shape (B, D).
        features_teacher: Features from teacher model, shape (B, D).

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(features_student, features_teacher)
