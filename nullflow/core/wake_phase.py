"""
Wake Phase — online learning with experience replay for stability.

During the Wake phase, the model learns from incoming data in a standard
supervised fashion (cross-entropy loss on the classifier). To prevent
catastrophic forgetting of old tasks, experience replay from the latent
buffer is interleaved with new-task learning.

No null-space projection is applied during wake — the classifier needs
full gradient freedom to learn new task classes. Protection comes from:
  1. Experience replay (mixing old buffer data during wake)
  2. Knowledge distillation on replay samples (optional)
  3. NSP applied later during REM phase
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..models.nullflow_model import NullFlowModel
from ..models.classifier import LatentClassifier
from .null_space import NullSpaceProjector
from .drift_detector import PageHinkleyDetector
from .knowledge_distillation import distillation_loss


def wake_step(
    model: NullFlowModel,
    null_projector: NullSpaceProjector,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    drift_detector: PageHinkleyDetector = None,
    seen_classes: Optional[list] = None,
) -> Tuple[float, bool]:
    """
    Execute one Wake phase training step on new-task data only.

    Args:
        model: NullFlowModel (encoder is frozen, classifier is trainable).
        null_projector: NullSpaceProjector (unused during wake, kept for API).
        batch: Tuple of (images, labels), both tensors.
        optimizer: Optimizer for classifier parameters.
        drift_detector: Optional PageHinkleyDetector for task-free mode.
        seen_classes: List of all seen class IDs (for output masking).

    Returns:
        loss_value: Scalar loss for this step.
        drift_detected: True if drift was detected (False if no detector).
    """
    model.classifier.train()
    x, y = batch
    device = next(model.classifier.parameters()).device
    x, y = x.to(device), y.to(device)

    # Encode to latent space (frozen, no grad)
    z = model.encode(x)

    # Classify and compute loss
    logits = model.classify(z)

    # Output masking: set logits of unseen classes to -inf
    if seen_classes is not None:
        mask = torch.full_like(logits, float('-inf'))
        for c in seen_classes:
            mask[:, c] = 0.0
        logits = logits + mask

    loss = F.cross_entropy(logits, y)

    # Backward + update (no NSP during wake)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Drift detection (optional)
    drift_detected = False
    if drift_detector is not None:
        drift_detected = drift_detector.update(loss.item())

    return loss.item(), drift_detected


def wake_step_with_replay(
    model: NullFlowModel,
    batch: Tuple[torch.Tensor, torch.Tensor],
    replay_z: torch.Tensor,
    replay_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    replay_weight: float = 0.5,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 0.0,
    old_classifier: Optional[LatentClassifier] = None,
    wake_kd_weight: float = 0.0,
    kd_temperature: float = 3.0,
    seen_classes: Optional[list] = None,
    latent_noise_std: float = 0.0,
    null_projector: Optional[NullSpaceProjector] = None,
) -> float:
    """
    Wake step with interleaved experience replay from the latent buffer.

    Trains on BOTH new-task data AND old-task buffer exemplars in the
    same gradient step. This prevents catastrophic forgetting during wake
    while still allowing the classifier to learn new classes.

    The total loss is:
        L = CE(new_task) + replay_weight * CE(buffer_replay) + wake_kd_weight * KD(replay)

    Args:
        model: NullFlowModel (encoder is frozen, classifier is trainable).
        batch: (images, labels) for the current task.
        replay_z: Latent vectors from the buffer, shape (M, d).
        replay_y: Labels from the buffer, shape (M,).
        optimizer: Optimizer for classifier parameters.
        replay_weight: Weight for the replay loss (default 0.5).
        label_smoothing: Label smoothing for CE loss (default 0.0).
        grad_clip_norm: Max gradient norm for clipping (0=off).
        old_classifier: Old classifier snapshot for KD (optional).
        wake_kd_weight: Weight for KD loss during wake (0=off).
        kd_temperature: Temperature for KD softmax.
        seen_classes: List of all class IDs seen so far (for output masking).
        latent_noise_std: Std of Gaussian noise added to latent vectors (0=off).

    Returns:
        loss_value: Total scalar loss for this step.
    """
    model.classifier.train()
    x, y = batch
    device = next(model.classifier.parameters()).device
    x, y = x.to(device), y.to(device)
    replay_z = replay_z.to(device)
    replay_y = replay_y.to(device)

    # Encode new data to latent space
    z_new = model.encode(x)

    # Latent-space augmentation: add noise ONLY to replay vectors,
    # NOT to new-task data. New-task latents should be clean for
    # maximum plasticity. Replay noise prevents memorization of
    # stored buffer vectors.
    if latent_noise_std > 0:
        replay_z = replay_z + torch.randn_like(replay_z) * latent_noise_std

    # Concatenate new + replay latents and labels
    z_all = torch.cat([z_new, replay_z], dim=0)
    y_all = torch.cat([y, replay_y], dim=0)

    # Forward pass on combined batch
    logits = model.classify(z_all)

    # Output masking: set logits of unseen classes to -inf
    # This prevents random untrained class outputs from competing
    if seen_classes is not None:
        mask = torch.full_like(logits, float('-inf'))
        for c in seen_classes:
            mask[:, c] = 0.0
        logits_masked = logits + mask
    else:
        logits_masked = logits

    # Separate losses for weighting
    n_new = z_new.shape[0]
    loss_new = F.cross_entropy(
        logits_masked[:n_new], y_all[:n_new], label_smoothing=label_smoothing,
    )
    loss_replay = F.cross_entropy(
        logits_masked[n_new:], y_all[n_new:], label_smoothing=label_smoothing,
    )

    loss = loss_new + replay_weight * loss_replay

    # KD on replay samples (if old_classifier provided and weight > 0)
    if old_classifier is not None and wake_kd_weight > 0:
        with torch.no_grad():
            old_logits = old_classifier(replay_z)
        loss_kd = distillation_loss(
            logits[n_new:], old_logits,
            temperature=kd_temperature,
            seen_classes=seen_classes,  # Focus KD on seen classes only
        )
        loss = loss + wake_kd_weight * loss_kd

    # Backward + update (NSP optional during wake)
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping for stability
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            model.classifier.parameters(), max_norm=grad_clip_norm,
        )

    # Null-space projection during wake (optional — protects old task directions)
    if null_projector is not None:
        null_projector.project_gradients(model.classifier)

    optimizer.step()

    return loss.item()


def wake_epoch(
    model: NullFlowModel,
    null_projector: NullSpaceProjector,
    data_loader,
    optimizer: torch.optim.Optimizer,
    drift_detector: PageHinkleyDetector = None,
    verbose: bool = False,
    seen_classes: Optional[list] = None,
) -> Tuple[float, bool, list]:
    """
    Execute one full epoch of Wake phase training (without replay).

    Args:
        model: NullFlowModel.
        null_projector: NullSpaceProjector (unused, kept for API compat).
        data_loader: DataLoader yielding (images, labels) batches.
        optimizer: Optimizer for classifier parameters.
        drift_detector: Optional drift detector.
        verbose: Whether to print progress.
        seen_classes: List of all seen class IDs (for output masking).

    Returns:
        avg_loss: Average loss over the epoch.
        any_drift: True if any drift was detected during the epoch.
        losses: List of per-batch losses.
    """
    from tqdm import tqdm

    total_loss = 0.0
    num_batches = 0
    any_drift = False
    losses = []

    pbar = tqdm(data_loader, desc="Wake", disable=not verbose)
    for batch in pbar:
        loss_val, drift = wake_step(
            model, null_projector, batch, optimizer, drift_detector,
            seen_classes=seen_classes,
        )
        total_loss += loss_val
        num_batches += 1
        losses.append(loss_val)
        if drift:
            any_drift = True
        pbar.set_postfix(loss=f"{loss_val:.4f}", drift=drift)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, any_drift, losses


def wake_epoch_with_replay(
    model: NullFlowModel,
    data_loader,
    buffer_data: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    replay_weight: float = 0.5,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 0.0,
    verbose: bool = False,
    old_classifier: Optional[LatentClassifier] = None,
    wake_kd_weight: float = 0.0,
    kd_temperature: float = 3.0,
    seen_classes: Optional[list] = None,
    latent_noise_std: float = 0.0,
    replay_ratio: float = 3.0,
    null_projector: Optional[NullSpaceProjector] = None,
) -> Tuple[float, list]:
    """
    Execute one epoch of Wake training WITH experience replay.

    For each batch of new-task data, samples a matching batch from the
    buffer and trains on both simultaneously. This prevents the classifier
    from catastrophically forgetting old tasks while learning new ones.

    Args:
        model: NullFlowModel.
        data_loader: DataLoader for current task (images, labels).
        buffer_data: (z_buffer, y_buffer) from latent replay buffer.
        optimizer: Optimizer for classifier parameters.
        replay_weight: Weight for replay loss vs. new-task loss.
        label_smoothing: Label smoothing for CE loss.
        grad_clip_norm: Max gradient norm for clipping (0=off).
        verbose: Whether to print progress.
        old_classifier: Old classifier snapshot for KD (optional).
        wake_kd_weight: Weight for KD loss during wake (0=off).
        kd_temperature: Temperature for KD softmax.

    Returns:
        avg_loss: Average loss over the epoch.
        losses: List of per-batch losses.
    """
    from tqdm import tqdm

    buf_z, buf_y = buffer_data
    n_buf = buf_z.shape[0]

    total_loss = 0.0
    num_batches = 0
    losses = []

    pbar = tqdm(data_loader, desc="Wake+Replay", disable=not verbose)
    for batch in pbar:
        # Sample replay batch — configurable ratio (replay:new).
        batch_size = batch[0].shape[0]
        replay_size = min(int(batch_size * replay_ratio), n_buf)
        
        # Class-balanced sampling: equal samples per old class
        unique_classes = torch.unique(buf_y)
        n_classes = len(unique_classes)
        per_cls = max(1, replay_size // n_classes)
        bal_indices = []
        for c in unique_classes:
            cls_mask = (buf_y == c).nonzero(as_tuple=True)[0]
            k = min(per_cls, len(cls_mask))
            perm = torch.randperm(len(cls_mask))[:k]
            bal_indices.append(cls_mask[perm])
        idx = torch.cat(bal_indices)
        # Shuffle the balanced batch
        idx = idx[torch.randperm(len(idx))]
        replay_z = buf_z[idx]
        replay_y = buf_y[idx]

        loss_val = wake_step_with_replay(
            model, batch, replay_z, replay_y, optimizer, replay_weight,
            label_smoothing=label_smoothing, grad_clip_norm=grad_clip_norm,
            old_classifier=old_classifier, wake_kd_weight=wake_kd_weight,
            kd_temperature=kd_temperature,
            seen_classes=seen_classes, latent_noise_std=latent_noise_std,
            null_projector=null_projector,
        )
        total_loss += loss_val
        num_batches += 1
        losses.append(loss_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, losses
