"""
Sleep-REM Phase — replay-based consolidation + knowledge distillation.

During the REM phase:
    1. Replay ALL seen classes using high-quality buffer exemplars
    2. For past-class replay: KD loss + CE loss
    3. For current-task replay: CE loss only (no KD)
    4. Optionally: null-space projection on classifier gradients

Two replay modes:
    - buffer_replay_data: use REAL buffer exemplars (best quality)
    - FM-generated replay: generate via Flow Matching (fallback)

Using real buffer exemplars gives significantly better consolidation
because the FM may not produce high-fidelity latent vectors.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple

from ..models.flow_matching import ConditionalFlowMatching
from ..models.classifier import LatentClassifier
from .null_space import NullSpaceProjector
from .knowledge_distillation import distillation_loss

logger = logging.getLogger(__name__)


def sleep_rem(
    flow_model: ConditionalFlowMatching,
    classifier: LatentClassifier,
    old_classifier: LatentClassifier,
    null_projector: NullSpaceProjector,
    seen_classes: List[int],
    new_classes: List[int],
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 20,
    replay_per_class: int = 50,
    kd_temperature: float = 2.0,
    kd_weight: float = 1.0,
    ce_replay_weight: float = 0.1,
    num_ode_steps: int = 4,
    ode_solver: str = "euler",
    batch_size: int = 128,
    device: str = "cuda",
    verbose: bool = False,
    current_task_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    buffer_replay_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 0.0,
    scheduler = None,
    latent_noise_std: float = 0.0,
    use_train_masking: bool = False,
) -> List[float]:
    """
    Execute the Sleep-REM phase: replay-based consolidation.

    KEY DESIGN: Two modes —
      Mode A (buffer_replay_data provided): Use REAL buffer exemplars for
        ALL classes. Much higher quality than FM-generated replay.
      Mode B (fallback): Generate FM replay for past + real for current.

    For each epoch:
        1. Prepare replay data (buffer or FM-generated)
        2. For each batch:
            a. CE loss on ALL replay labels (with label smoothing)
            b. KD loss ONLY on past-class portion
            c. Gradient clipping + NSP + update

    Args:
        flow_model: Trained FM model (after NREM phase).
        classifier: Current classifier h_ψ (to be updated).
        old_classifier: Copy of classifier BEFORE current task (for KD).
        null_projector: NullSpaceProjector.
        seen_classes: All classes seen so far (including current task).
        new_classes: Classes from the current task.
        optimizer: Optimizer for classifier parameters.
        n_epochs: Number of REM epochs.
        replay_per_class: Samples per class (FM mode only).
        kd_temperature: Temperature for KD softmax.
        kd_weight: Weight for KD loss.
        ce_replay_weight: Weight for CE loss on replay.
        num_ode_steps: ODE steps for FM sampling.
        ode_solver: ODE solver — 'euler' or 'heun'.
        batch_size: Batch size for REM training.
        device: Device string.
        verbose: Whether to print progress.
        current_task_data: (z, y) real exemplars for current task (FM mode).
        buffer_replay_data: (z, y) from full latent buffer (preferred mode).
        label_smoothing: Label smoothing factor for CE loss (0=off).
        grad_clip_norm: Max gradient norm for clipping (0=off).

    Returns:
        epoch_losses: List of average losses per epoch.
    """
    from tqdm import tqdm

    new_classes_set = set(new_classes)
    past_classes = [c for c in seen_classes if c not in new_classes_set]
    has_past = len(past_classes) > 0

    classifier.train()
    old_classifier.eval()
    flow_model.eval()

    epoch_losses = []

    # Pre-compute buffer data if provided (used as anchor data each epoch)
    buf_z_anchor, buf_y_anchor, buf_past_anchor = None, None, None
    if buffer_replay_data is not None:
        buf_z_anchor = buffer_replay_data[0].to(device)
        buf_y_anchor = buffer_replay_data[1].to(device)
        buf_past_anchor = torch.tensor(
            [int(y.item()) not in new_classes_set for y in buf_y_anchor],
            dtype=torch.bool, device=device,
        )

    for epoch in range(n_epochs):
        # ---- Prepare replay data EACH EPOCH (fresh FM samples for diversity) ----
        replay_z_list, replay_y_list, is_past_list = [], [], []

        # Add buffer data as anchors (if available)
        if buf_z_anchor is not None:
            replay_z_list.append(buf_z_anchor)
            replay_y_list.append(buf_y_anchor)
            is_past_list.append(buf_past_anchor)

        # Add FM-generated data for past classes (fresh each epoch)
        if has_past and replay_per_class > 0:
            for c in past_classes:
                z_samples = flow_model.sample(
                    n=replay_per_class, class_label=c,
                    num_steps=num_ode_steps, device=device,
                    solver=ode_solver,
                )
                replay_z_list.append(z_samples)
                replay_y_list.append(
                    torch.full((replay_per_class,), c, dtype=torch.long, device=device)
                )
                is_past_list.append(
                    torch.ones(replay_per_class, dtype=torch.bool, device=device)
                )

        # Add current task data (real exemplars)
        if current_task_data is not None and replay_per_class > 0:
            cur_z, cur_y = current_task_data
            cur_z, cur_y = cur_z.to(device), cur_y.to(device)
            for c in new_classes:
                mask = cur_y == c
                z_cls = cur_z[mask]
                if len(z_cls) == 0:
                    continue
                n_take = min(replay_per_class, len(z_cls))
                idx = torch.randperm(len(z_cls), device=device)[:n_take]
                replay_z_list.append(z_cls[idx])
                replay_y_list.append(
                    torch.full((n_take,), c, dtype=torch.long, device=device)
                )
                is_past_list.append(
                    torch.zeros(n_take, dtype=torch.bool, device=device)
                )

        if len(replay_z_list) == 0:
            return []

        all_z = torch.cat(replay_z_list, dim=0)
        all_y = torch.cat(replay_y_list, dim=0)
        is_past = torch.cat(is_past_list, dim=0)

        # Create DataLoader for this epoch
        dataset = TensorDataset(all_z, all_y, is_past.long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # ---- Train on replay data ----
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"REM [{epoch+1}/{n_epochs}]", disable=not verbose)
        for z_replay, y_replay, past_mask in pbar:
            past_mask = past_mask.bool()

            # Latent noise augmentation to prevent overfitting to buffer vectors
            if latent_noise_std > 0:
                z_replay = z_replay + torch.randn_like(z_replay) * latent_noise_std

            logits_new = classifier(z_replay)

            # Output masking: constrain CE to seen classes only
            if use_train_masking:
                mask = torch.full_like(logits_new, float('-inf'))
                for c in seen_classes:
                    mask[:, c] = 0.0
                logits_masked = logits_new + mask
            else:
                logits_masked = logits_new

            # CE Loss on ALL replay labels (masked to seen classes)
            loss_ce = F.cross_entropy(
                logits_masked, y_replay,
                label_smoothing=label_smoothing,
            ) * ce_replay_weight

            # KD Loss ONLY on past-class samples (masked to seen classes)
            loss_kd = torch.tensor(0.0, device=device)
            if has_past and past_mask.any():
                with torch.no_grad():
                    logits_old = old_classifier(z_replay[past_mask])
                kd_seen = list(seen_classes) if use_train_masking else None
                loss_kd = distillation_loss(
                    logits_new[past_mask], logits_old,
                    temperature=kd_temperature,
                    seen_classes=kd_seen,
                ) * kd_weight

            loss = loss_kd + loss_ce

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (before NSP)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    classifier.parameters(), max_norm=grad_clip_norm,
                )

            # Null-space projection
            null_projector.project_gradients(classifier)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        if verbose:
            logger.info(f"  REM Epoch {epoch+1}/{n_epochs} \u2014 Avg Loss: {avg_loss:.4f}")

    return epoch_losses
