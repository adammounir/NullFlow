"""
NullFlow Strategy — orchestrates the complete Wake-Sleep cycle.

This is the main entry point for running NullFlow on a continual learning
benchmark. It manages:
    1. Encoder pre-training / calibration
    2. Task-aware or task-free training loop
    3. Wake → NREM → REM cycle per task/drift
    4. Latent replay buffer for FM consolidation
    5. Evaluation and metric tracking
"""

import copy
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple

from ..models.nullflow_model import NullFlowModel
from ..core.null_space import NullSpaceProjector
from ..core.drift_detector import PageHinkleyDetector
from ..core.wake_phase import wake_step, wake_epoch, wake_epoch_with_replay
from ..core.sleep_nrem import sleep_nrem, encode_task_data
from ..core.sleep_rem import sleep_rem
from ..metrics.cl_metrics import CLMetricsTracker
from ..data.benchmarks import CLBenchmark
from ..data.latent_buffer import LatentReplayBuffer
from ..utils.logging_utils import get_logger


class NullFlowStrategy:
    """
    Orchestrates the NullFlow Wake-Sleep cycle for continual learning.

    Supports both task-aware and task-free modes:
        - Task-aware: explicit task boundaries trigger Sleep phases
        - Task-free: Page-Hinkley drift detection triggers Sleep phases
    """

    @staticmethod
    @torch.no_grad()
    def weight_align(
        classifier: nn.Module,
        old_classes: list,
        new_classes: list,
        gamma_clamp: tuple = (0.1, 10.0),
    ) -> float:
        """
        Weight Aligning (WA) bias correction from Zhao et al. (2020).

        After learning new classes, their weight rows in the final Linear
        layer tend to have larger norms (recency bias). WA rescales new-class
        rows to match the mean norm of old classes.

        Applied AFTER training, BEFORE evaluation. Zero extra cost.

        Args:
            classifier: LatentClassifier with classifier.classifier[-1] as Linear.
            old_classes: Class indices from previous tasks.
            new_classes: Class indices from the current task.
            gamma_clamp: (min, max) clamp for correction factor.

        Returns:
            gamma: Correction factor applied.
        """
        if len(old_classes) == 0 or len(new_classes) == 0:
            return 1.0

        output_layer = classifier.classifier[-1]
        W = output_layer.weight
        norms = W.norm(dim=1)

        mu_old = norms[old_classes].mean()
        mu_new = norms[new_classes].mean()

        if mu_new < 1e-8:
            return 1.0

        gamma = (mu_old / mu_new).clamp(*gamma_clamp).item()

        W[new_classes] *= gamma
        if output_layer.bias is not None:
            output_layer.bias.data[new_classes] *= gamma

        return gamma

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary (from YAML).
        """
        self.config = config
        self.device = config.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.logger = get_logger("NullFlowStrategy")

        # Encoder type
        encoder_type = config.get("encoder_type", "resnet")

        # Initialize model
        self.model = NullFlowModel(
            latent_dim=config.get("latent_dim", 128),
            flow_hidden_dim=config.get("flow_hidden_dim", 512),
            flow_num_layers=config.get("flow_num_layers", 3),
            classifier_hidden_dim=config.get("classifier_hidden_dim", 256),
            num_classes_max=config.get("num_classes_max", 200),
            in_channels=3,
            image_size=config.get("image_size", 32),
            device=self.device,
            encoder_type=encoder_type,
            use_raw_features=config.get("use_raw_features", False),
            use_cosine_classifier=config.get("use_cosine_classifier", False),
            cosine_temperature=config.get("cosine_temperature", 0.1),
            classifier_type=config.get("classifier_type", "mlp"),
        ).to(self.device)

        # Null-Space Projector
        self.null_projector = NullSpaceProjector(
            rank=config.get("nsp_rank", 64),
            device=self.device,
            alpha=config.get("nsp_alpha", 0.5),
        )

        # Drift Detector (for task-free mode)
        self.drift_detector = PageHinkleyDetector(
            delta=config.get("drift_delta", 0.005),
            threshold=config.get("drift_threshold", 50),
            warmup=config.get("drift_warmup", 100),
        )

        # Latent Replay Buffer — prevents FM forgetting
        self.latent_buffer = LatentReplayBuffer(
            max_per_class=config.get("buffer_per_class", 100),
            max_total=config.get("buffer_max_total", 10000),
            selection=config.get("buffer_selection", "fifo"),
        )

        # Optimizers
        self.classifier_optimizer = torch.optim.Adam(
            self.model.classifier.parameters(),
            lr=config.get("wake_lr", 1e-3),
        )
        self.fm_optimizer = torch.optim.Adam(
            self.model.flow_model.parameters(),
            lr=config.get("fm_lr", 5e-4),
        )

        # State tracking
        self.seen_classes: List[int] = []
        self.task_count: int = 0
        self.metrics_tracker = CLMetricsTracker()

        # Logging
        self.logger.info(self.model.summary())

    def pretrain_encoder(self, benchmark: CLBenchmark, epochs: int = 30):
        """
        Pre-train / calibrate the encoder before CL begins.

        - **ResNet encoder**: calibrate the projection head on the
          first task's data only (unsupervised, no data leakage).
        - **VAE encoder**: pre-train on the first task's data only
          (or load from disk if already done).

        Args:
            benchmark: CLBenchmark providing training experiences.
            epochs: Number of pre-training / calibration epochs.
        """
        self.logger.info("=" * 60)

        enc_path = os.path.join(
            self.config.get("output_dir", "results/"), "pretrained_encoder.pt"
        )

        if os.path.exists(enc_path):
            self.logger.info(f"Loading pre-trained encoder from {enc_path}")
            self.model.encoder.load_pretrained(enc_path)
            self.model.encoder.to(self.device)
            self.model.encoder.freeze()
            return

        # Use first task's data only — no data leakage
        first_exp = benchmark.get_train_stream()[0]
        loader = first_exp.get_dataloader(
            batch_size=128,
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
        )

        encoder_type = self.config.get("encoder_type", "resnet")

        if encoder_type == "resnet":
            self.logger.info(
                "Calibrating ResNet projection head on first task data..."
            )
            self.model.encoder.to(self.device)
            self.model.encoder.calibrate(
                loader,
                epochs=min(epochs, 10),
                lr=1e-3,
                verbose=True,
                num_classes=benchmark.num_classes,
            )
        else:
            self.logger.info(
                "Pre-training VAE on first task data..."
            )
            self.model.encoder.to(self.device)
            self.model.encoder.pretrain(
                loader,
                epochs=epochs,
                lr=1e-3,
                kl_weight=1e-3,
                verbose=True,
            )

        self.model.encoder.freeze()

        # Save for reuse
        os.makedirs(os.path.dirname(enc_path) or ".", exist_ok=True)
        self.model.encoder.save_pretrained(enc_path)
        self.logger.info(f"Saved pre-trained encoder to {enc_path}")

    # Keep backward compatibility
    def pretrain_vae(self, benchmark: CLBenchmark, epochs: int = 30):
        """Backward-compatible alias for pretrain_encoder()."""
        self.pretrain_encoder(benchmark, epochs)

    def train_task_aware(self, benchmark: CLBenchmark) -> Dict:
        """
        Train with explicit task boundaries (task-aware mode).

        For each task:
            1. Wake phase: train classifier on new task data
            2. Sleep-NREM: train FM on new task latents
            3. Sleep-REM: replay + distillation
            4. Evaluate on all tasks

        Args:
            benchmark: CLBenchmark instance.

        Returns:
            results: Dictionary with metrics and accuracy matrix.
        """
        train_stream = benchmark.get_train_stream()
        test_stream = benchmark.get_test_stream()

        self.logger.info("=" * 60)
        self.logger.info(f"Task-Aware Training: {len(train_stream)} tasks")
        self.logger.info("=" * 60)

        for task_id, experience in enumerate(train_stream):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Task {task_id + 1}/{len(train_stream)} — "
                           f"Classes: {experience.classes}")
            self.logger.info(f"{'='*60}")

            new_classes = experience.classes
            self.seen_classes.extend(new_classes)

            train_loader = experience.get_dataloader(
                batch_size=self.config.get("wake_batch_size", 64),
                shuffle=True,
                num_workers=self.config.get("num_workers", 4),
            )

            # Save old classifier for KD (snapshot BEFORE learning this task)
            old_classifier = copy.deepcopy(self.model.classifier)
            old_classifier.eval()

            # Config shortcuts
            label_smoothing = self.config.get("label_smoothing", 0.0)
            grad_clip_norm = self.config.get("grad_clip_norm", 0.0)

            # ===== ENCODE TASK DATA (FIRST — encoder is frozen, deterministic) =====
            latent_z, latent_y = encode_task_data(
                self.model, train_loader, self.device,
            )

            # ===== WAKE PHASE (with experience replay from buffer) =====
            self.logger.info("Phase: WAKE")

            # Prototype Initialization: set new class output weights to class means.
            # For cosine classifier: use L2-normalised latent means (NCM-optimal).
            # For MLP classifier: use penultimate feature means (scaled to match old norms).
            if task_id > 0 and self.config.get("use_proto_init", False):
                use_cosine = self.config.get("use_cosine_classifier", False)
                cls_type = self.config.get("classifier_type", "mlp")
                with torch.no_grad():
                    self.model.classifier.eval()
                    out_layer = self.model.classifier.classifier[-1]   # Last Linear layer
                    z_all = latent_z.to(self.device)

                    if cls_type == "linear":
                        # Linear mode: weight = class mean in latent space (norm-matched)
                        old_cls = [c for c in self.seen_classes if c not in new_classes]
                        old_weight_norm = None
                        if old_cls:
                            old_weight_norm = out_layer.weight.data[old_cls].norm(dim=1).mean().item()
                        for c in new_classes:
                            cls_mask = (latent_y.to(self.device) == c)
                            if cls_mask.sum() > 0:
                                z_mean = z_all[cls_mask].mean(0)
                                if old_weight_norm is not None and old_weight_norm > 0:
                                    z_norm = z_mean.norm().item() + 1e-8
                                    z_mean = z_mean * (old_weight_norm / z_norm)
                                out_layer.weight.data[c] = z_mean
                                if out_layer.bias is not None:
                                    out_layer.bias.data[c] = 0.0
                        self.logger.info("  Proto-init (linear): weights = latent means.")
                    elif use_cosine:
                        # Cosine mode: set weight = L2-normalised latent class mean
                        for c in new_classes:
                            cls_mask = (latent_y.to(self.device) == c)
                            if cls_mask.sum() > 0:
                                z_mean = z_all[cls_mask].mean(0)
                                out_layer.weight.data[c] = F.normalize(z_mean.unsqueeze(0), dim=1).squeeze(0)
                        self.logger.info("  Proto-init (cosine): weights = normalised latent means.")
                    else:
                        # MLP mode: set weight = penultimate feature mean, norm-matched
                        feat_extractor = self.model.classifier.classifier[:-1]
                        old_cls = [c for c in self.seen_classes if c not in new_classes]
                        old_weight_norm = None
                        if old_cls:
                            old_weight_norm = out_layer.weight.data[old_cls].norm(dim=1).mean().item()
                        h_all = feat_extractor(z_all)
                        for c in new_classes:
                            cls_mask = (latent_y.to(self.device) == c)
                            if cls_mask.sum() > 0:
                                h_mean = h_all[cls_mask].mean(0)
                                if old_weight_norm is not None and old_weight_norm > 0:
                                    h_norm = h_mean.norm().item() + 1e-8
                                    h_mean = h_mean * (old_weight_norm / h_norm)
                                out_layer.weight.data[c] = h_mean
                                if out_layer.bias is not None:
                                    out_layer.bias.data[c] = 0.0
                        self.logger.info("  Proto-init (MLP): weights = penultimate means.")
                    self.model.classifier.train()

            # Reset optimizer for each task (fresh Adam momentum)
            wake_lr = self.config.get("wake_lr", 1e-3)
            wake_wd = self.config.get("wake_weight_decay", 0.0)
            OptimizerCls = torch.optim.AdamW if wake_wd > 0 else torch.optim.Adam
            self.classifier_optimizer = OptimizerCls(
                self.model.classifier.parameters(),
                lr=wake_lr,
                weight_decay=wake_wd,
            )

            wake_epochs = self.config.get("wake_epochs_per_task", 5)

            # Cosine annealing LR scheduler (warm start → decay)
            wake_scheduler = None
            if self.config.get("use_cosine_schedule", False):
                wake_scheduler = CosineAnnealingLR(
                    self.classifier_optimizer,
                    T_max=wake_epochs,
                    eta_min=wake_lr * 0.01,  # Decay to 1% of initial LR
                )

            # Balanced replay sampling if enabled
            use_balanced = self.config.get("use_balanced_replay", False)
            if use_balanced and len(self.latent_buffer) > 0:
                # Sample balanced: ~3 samples per class per batch
                n_classes = len(self.latent_buffer.get_classes())
                per_class = max(3, 192 // max(n_classes, 1))
                replay_data = self.latent_buffer.sample_balanced(per_class)
            else:
                replay_data = self.latent_buffer.get_all()
            has_replay = replay_data is not None

            # ===== FM-AUGMENTED REPLAY (small buffer → compensate with FM samples) =====
            fm_augment_wake = self.config.get("fm_augment_wake", False)
            fm_samples_per_class = self.config.get("fm_samples_per_class", 100)
            if fm_augment_wake and has_replay and task_id > 0:
                # Generate FM samples for ALL old classes to complement buffer
                old_classes = [c for c in self.seen_classes if c not in new_classes]
                if old_classes:
                    self.logger.info(
                        f"  FM augmentation: generating {fm_samples_per_class}/class "
                        f"for {len(old_classes)} old classes"
                    )
                    fm_z_list, fm_y_list = [], []
                    self.model.flow_model.eval()
                    with torch.no_grad():
                        for c in old_classes:
                            z_gen = self.model.flow_model.sample(
                                n=fm_samples_per_class, class_label=c,
                                num_steps=self.config.get("num_ode_steps", 6),
                                device=self.device,
                                solver=self.config.get("ode_solver", "heun"),
                            )
                            fm_z_list.append(z_gen.cpu())
                            fm_y_list.append(
                                torch.full((fm_samples_per_class,), c, dtype=torch.long)
                            )
                    fm_z = torch.cat(fm_z_list, dim=0)
                    fm_y = torch.cat(fm_y_list, dim=0)
                    # Concatenate buffer + FM-generated
                    buf_z, buf_y = replay_data
                    replay_data = (
                        torch.cat([buf_z, fm_z], dim=0),
                        torch.cat([buf_y, fm_y], dim=0),
                    )
                    self.logger.info(
                        f"  Augmented replay: {len(replay_data[0])} total "
                        f"({len(buf_z)} buffer + {len(fm_z)} FM-generated)"
                    )

            # KD during wake (optional — protects old knowledge while learning new)
            wake_kd_weight = self.config.get("wake_kd_weight", 0.0)
            kd_temperature = self.config.get("kd_temperature", 2.0)

            # Training output masking: constrain CE/KD to seen classes only.
            # This prevents the classifier from wasting capacity on unseen-class
            # logits and focuses the KD signal on meaningful class relationships.
            use_train_masking = self.config.get("use_train_masking", False)
            train_seen = list(self.seen_classes) if use_train_masking else None

            # NSP during wake (optional — protects old-task gradient directions)
            use_nsp_wake = self.config.get("use_nsp_wake", False)
            wake_nsp = self.null_projector if (use_nsp_wake and task_id > 0 and self.null_projector.V_r is not None) else None

            for epoch in range(wake_epochs):
                if has_replay:
                    avg_loss, _ = wake_epoch_with_replay(
                        self.model, train_loader, replay_data,
                        self.classifier_optimizer,
                        replay_weight=self.config.get("wake_replay_weight", 0.5),
                        label_smoothing=label_smoothing,
                        grad_clip_norm=grad_clip_norm,
                        old_classifier=old_classifier if wake_kd_weight > 0 else None,
                        wake_kd_weight=wake_kd_weight,
                        kd_temperature=kd_temperature,
                        seen_classes=train_seen,
                        latent_noise_std=self.config.get("latent_noise_std", 0.0),
                        replay_ratio=self.config.get("wake_replay_ratio", 3.0),
                        null_projector=wake_nsp,
                        verbose=False,
                    )
                else:
                    avg_loss, _, _ = wake_epoch(
                        self.model, self.null_projector, train_loader,
                        self.classifier_optimizer, verbose=False,
                        seen_classes=train_seen,
                    )
                self.logger.info(f"  Wake Epoch {epoch+1}/{wake_epochs} — "
                               f"Loss: {avg_loss:.4f}"
                               f"{' (w/ replay)' if has_replay else ''}")
                # Step the cosine scheduler
                if wake_scheduler is not None:
                    wake_scheduler.step()
            # ===== SLEEP-NREM PHASE =====
            self.logger.info("Phase: SLEEP-NREM")

            # Update NSP BEFORE NREM/REM so projections protect during sleep
            nsp_alpha = self.config.get("nsp_alpha", 0.5)
            latent_dataset = torch.utils.data.TensorDataset(
                latent_z.to(self.device), latent_y.to(self.device),
            )
            latent_loader = DataLoader(
                latent_dataset, batch_size=self.config.get("fm_batch_size", 128),
                shuffle=False,
            )
            if nsp_alpha > 0:
                self.null_projector.update(
                    self.model.classifier, latent_loader,
                    num_samples=self.config.get("nsp_num_samples", 500),
                )
                if self.null_projector.V_r is not None:
                    self.logger.info(f"  NSP updated — V_r shape: {self.null_projector.V_r.shape}")

            all_buffer_data = self.latent_buffer.get_all()
            if all_buffer_data is not None:
                self.logger.info(
                    f"  Buffer replay: {len(all_buffer_data[0])} exemplars "
                    f"from {len(self.latent_buffer.get_classes())} classes"
                )

            nrem_losses = sleep_nrem(
                flow_model=self.model.flow_model,
                latent_data=(latent_z, latent_y),
                null_projector=self.null_projector,
                optimizer=self.fm_optimizer,
                n_epochs=self.config.get("fm_epochs_per_task", 50),
                batch_size=self.config.get("fm_batch_size", 128),
                device=self.device,
                verbose=False,
                replay_data=all_buffer_data,
            )
            if nrem_losses:
                self.logger.info(f"  NREM Final Loss: {nrem_losses[-1]:.4f}")

            # ===== ADD TO BUFFER (after NREM, before REM) =====
            self.latent_buffer.add(latent_z, latent_y)
            self.logger.info(f"  Buffer: {self.latent_buffer}")

            # ===== SLEEP-REM PHASE (or BFT alternative) =====
            use_bft = self.config.get("use_bft", False)

            if use_bft and task_id > 0:
                # ===== BALANCED FINE-TUNING (BFT) =====
                # Replace REM with simple balanced CE + KD on full buffer.
                # Proven approach from LUCIR/BiC/WA literature.
                self.logger.info("Phase: BALANCED FINE-TUNING (BFT)")

                bft_epochs = self.config.get("bft_epochs", 30)
                bft_lr = self.config.get("rem_lr", 0.0004)
                bft_wd = self.config.get("wake_weight_decay", 0.0)
                BFT_Opt = torch.optim.AdamW if bft_wd > 0 else torch.optim.Adam
                bft_optimizer = BFT_Opt(
                    self.model.classifier.parameters(),
                    lr=bft_lr, weight_decay=bft_wd,
                )
                bft_scheduler = CosineAnnealingLR(
                    bft_optimizer, T_max=bft_epochs, eta_min=bft_lr * 0.01,
                )

                buf_data = self.latent_buffer.get_all()
                buf_z, buf_y = buf_data[0].to(self.device), buf_data[1].to(self.device)
                noise_std = self.config.get("latent_noise_std", 0.0)
                kd_temp = self.config.get("kd_temperature", 3.0)
                bft_kd_weight = self.config.get("bft_kd_weight", 0.0)

                from ..core.knowledge_distillation import distillation_loss as bft_kd_loss

                # Precompute class indices for balanced sampling
                unique_cls = torch.unique(buf_y)
                n_cls = len(unique_cls)
                per_cls = max(1, 256 // n_cls)  # ~256 samples per batch
                cls_indices = {}
                for c in unique_cls:
                    cls_indices[c.item()] = (buf_y == c).nonzero(as_tuple=True)[0]

                n_iters = max(1, len(buf_z) // 256)

                for epoch in range(bft_epochs):
                    total_loss = 0.0
                    self.model.classifier.train()

                    for _ in range(n_iters):
                        bal_idx = []
                        for c_val, c_idx in cls_indices.items():
                            k = min(per_cls, len(c_idx))
                            perm = torch.randperm(len(c_idx), device=self.device)[:k]
                            bal_idx.append(c_idx[perm])
                        idx = torch.cat(bal_idx)
                        idx = idx[torch.randperm(len(idx), device=self.device)]

                        zb = buf_z[idx]
                        yb = buf_y[idx]
                        if noise_std > 0:
                            zb = zb + torch.randn_like(zb) * noise_std

                        logits = self.model.classifier(zb)
                        loss_ce = F.cross_entropy(logits, yb)

                        if bft_kd_weight > 0:
                            with torch.no_grad():
                                old_logits = old_classifier(zb)
                            loss_kd = bft_kd_loss(logits, old_logits, temperature=kd_temp)
                            loss = loss_ce + bft_kd_weight * loss_kd
                        else:
                            loss = loss_ce

                        bft_optimizer.zero_grad()
                        loss.backward()
                        bft_optimizer.step()
                        total_loss += loss.item()

                    bft_scheduler.step()
                    avg = total_loss / n_iters
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        self.logger.info(f"  BFT Epoch {epoch+1}/{bft_epochs} — Loss: {avg:.4f}")
            else:
                # ===== STANDARD SLEEP-REM PHASE =====
                self.logger.info("Phase: SLEEP-REM")

                # Separate REM optimizer with configurable LR
                rem_lr = self.config.get("rem_lr", self.config.get("wake_lr", 1e-3))
                rem_wd = self.config.get("wake_weight_decay", 0.0)
                RemOptimizerCls = torch.optim.AdamW if rem_wd > 0 else torch.optim.Adam
                rem_optimizer = RemOptimizerCls(
                    self.model.classifier.parameters(),
                    lr=rem_lr,
                    weight_decay=rem_wd,
                )

                # Cosine annealing for REM
                rem_epochs_cfg = self.config.get("rem_epochs", 20)
                rem_scheduler = None
                if self.config.get("use_cosine_schedule", False):
                    rem_scheduler = CosineAnnealingLR(
                        rem_optimizer,
                        T_max=rem_epochs_cfg,
                        eta_min=rem_lr * 0.01,
                    )

                # Optionally pass buffer data to REM for anchor-based consolidation
                rem_buffer_data = None
                if self.config.get("rem_use_buffer", False):
                    rem_buffer_data = self.latent_buffer.get_all()

                rem_losses = sleep_rem(
                    flow_model=self.model.flow_model,
                    classifier=self.model.classifier,
                    old_classifier=old_classifier,
                    null_projector=self.null_projector,
                    seen_classes=self.seen_classes,
                    new_classes=new_classes,
                    optimizer=rem_optimizer,
                    n_epochs=self.config.get("rem_epochs", 20),
                    replay_per_class=self.config.get("replay_per_class", 50),
                    kd_temperature=self.config.get("kd_temperature", 2.0),
                    kd_weight=self.config.get("kd_weight", 1.0),
                    ce_replay_weight=self.config.get("ce_replay_weight", 0.1),
                    num_ode_steps=self.config.get("num_ode_steps", 4),
                    ode_solver=self.config.get("ode_solver", "euler"),
                    batch_size=self.config.get("fm_batch_size", 128),
                    device=self.device,
                    verbose=False,
                    current_task_data=(latent_z, latent_y),
                    buffer_replay_data=rem_buffer_data,
                    label_smoothing=label_smoothing,
                    grad_clip_norm=grad_clip_norm,
                    scheduler=rem_scheduler,
                    latent_noise_std=self.config.get("latent_noise_std", 0.0),
                    use_train_masking=self.config.get("use_train_masking", False),
                )
                if rem_losses:
                    self.logger.info(f"  REM Final Loss: {rem_losses[-1]:.4f}")

            # ===== WEIGHT ALIGNING (WA) — Bias Correction =====
            if task_id > 0 and self.config.get("use_weight_align", False):
                old_cls = [c for c in self.seen_classes if c not in new_classes]
                gamma = self.weight_align(
                    self.model.classifier, old_cls, list(new_classes),
                )
                self.logger.info(f"  WA correction: γ={gamma:.3f}")

            # ===== EVALUATION =====
            self.logger.info("Evaluating on all tasks...")
            task_accuracies = self._evaluate_all_tasks(test_stream, task_id)
            self.metrics_tracker.update(task_id, task_accuracies)

            # Log results
            self.logger.info(f"  Task Accuracies: {task_accuracies}")
            aa = self.metrics_tracker.get_current_aa()
            self.logger.info(f"  Average Accuracy (AA): {aa:.2f}%")

            self.task_count += 1

        # ===== FINAL CONSOLIDATION (optional) =====
        # After all tasks: retrain classifier on FULL buffer with KD anchor.
        # Pushes AA toward the offline upper bound.
        final_consol_epochs = self.config.get("final_consolidation_epochs", 0)
        if final_consol_epochs > 0 and len(self.latent_buffer) > 0:
            self.logger.info(f"Phase: FINAL CONSOLIDATION ({final_consol_epochs} epochs)")
            buf_data = self.latent_buffer.get_all()
            buf_z, buf_y = buf_data
            buf_z, buf_y = buf_z.to(self.device), buf_y.to(self.device)

            # Snapshot the CL-trained classifier as KD teacher
            old_classifier = copy.deepcopy(self.model.classifier)
            old_classifier.eval()

            fc_lr = self.config.get("final_consol_lr", 0.0003)
            fc_wd = self.config.get("wake_weight_decay", 0.0)
            FC_Opt = torch.optim.AdamW if fc_wd > 0 else torch.optim.Adam
            fc_optimizer = FC_Opt(
                self.model.classifier.parameters(),
                lr=fc_lr, weight_decay=fc_wd,
            )
            fc_scheduler = CosineAnnealingLR(
                fc_optimizer, T_max=final_consol_epochs, eta_min=fc_lr * 0.01,
            )

            dataset = torch.utils.data.TensorDataset(buf_z, buf_y)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=256, shuffle=True, drop_last=False,
            )

            kd_temp = self.config.get("kd_temperature", 3.0)
            fc_kd_weight = self.config.get("final_consol_kd_weight", 1.0)
            noise_std = self.config.get("latent_noise_std", 0.0)

            from ..core.knowledge_distillation import distillation_loss as fc_kd_loss

            for epoch in range(final_consol_epochs):
                self.model.classifier.train()
                total_loss = 0.0
                n_batches = 0
                for zb, yb in loader:
                    if noise_std > 0:
                        zb = zb + torch.randn_like(zb) * noise_std
                    logits = self.model.classifier(zb)
                    loss_ce = F.cross_entropy(logits, yb)
                    with torch.no_grad():
                        old_logits = old_classifier(zb)
                    loss_kd = fc_kd_loss(logits, old_logits, temperature=kd_temp)
                    loss = loss_ce + fc_kd_weight * loss_kd
                    fc_optimizer.zero_grad()
                    loss.backward()
                    fc_optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1
                fc_scheduler.step()
                avg = total_loss / max(n_batches, 1)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self.logger.info(f"  Consolidation Epoch {epoch+1}/{final_consol_epochs} — Loss: {avg:.4f}")

            # Re-evaluate after consolidation
            self.logger.info("Re-evaluating after final consolidation...")
            task_accuracies = self._evaluate_all_tasks(test_stream, self.task_count - 1)
            self.metrics_tracker.update(self.task_count - 1, task_accuracies)
            aa = self.metrics_tracker.get_current_aa()
            self.logger.info(f"  Post-consolidation AA: {aa:.2f}%")

        # Final summary
        results = self._compile_results()
        self._log_final_summary(results)
        return results

    def train_joint_retrain(self, benchmark: CLBenchmark) -> Dict:
        """
        Joint-retrain mode: after each task, retrain classifier on ALL
        buffer data with standard CE loss. No KD, no FM, no NSP.

        This serves as an upper-bound baseline to measure the ceiling
        achievable with frozen features and full data retention.

        For each task:
            1. Encode new task data → add to buffer
            2. Re-initialise classifier optimizer
            3. Train classifier on ALL buffer data for N epochs
            4. Evaluate on all tasks

        Args:
            benchmark: CLBenchmark instance.

        Returns:
            results: Dictionary with metrics and accuracy matrix.
        """
        from torch.utils.data import DataLoader, TensorDataset

        train_stream = benchmark.get_train_stream()
        test_stream = benchmark.get_test_stream()

        retrain_epochs = self.config.get("retrain_epochs", 30)
        retrain_lr = self.config.get("retrain_lr", 0.001)
        retrain_batch_size = self.config.get("retrain_batch_size", 256)
        label_smoothing = self.config.get("label_smoothing", 0.0)
        latent_noise = self.config.get("retrain_noise_std", 0.1)
        weight_decay = self.config.get("retrain_weight_decay", 1e-4)

        self.logger.info("=" * 60)
        self.logger.info(f"Joint-Retrain Mode: {len(train_stream)} tasks")
        self.logger.info(f"  Epochs per task: {retrain_epochs}, LR: {retrain_lr}"
                       f", Noise: {latent_noise}, WD: {weight_decay}")
        self.logger.info("=" * 60)

        for task_id, experience in enumerate(train_stream):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Task {task_id + 1}/{len(train_stream)} — "
                           f"Classes: {experience.classes}")
            self.logger.info(f"{'='*60}")

            new_classes = experience.classes
            self.seen_classes.extend(new_classes)

            train_loader = experience.get_dataloader(
                batch_size=128, shuffle=True,
                num_workers=self.config.get("num_workers", 4),
            )

            # ===== ENCODE TASK DATA =====
            latent_z, latent_y = encode_task_data(
                self.model, train_loader, self.device,
            )

            # ===== ADD TO BUFFER =====
            self.latent_buffer.add(latent_z, latent_y)
            self.logger.info(f"  Buffer: {self.latent_buffer}")

            # ===== JOINT RETRAIN: train classifier on ALL buffer data =====
            self.logger.info("Phase: JOINT RETRAIN")

            # Get all buffer data
            buf_z, buf_y = self.latent_buffer.get_all()
            buf_z = buf_z.to(self.device)
            buf_y = buf_y.to(self.device)

            self.logger.info(f"  Training on {len(buf_z)} exemplars from "
                           f"{len(torch.unique(buf_y))} classes")

            # Fresh optimizer each task (with weight decay for regularization)
            optimizer = torch.optim.AdamW(
                self.model.classifier.parameters(),
                lr=retrain_lr,
                weight_decay=weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=retrain_epochs, eta_min=retrain_lr * 0.01,
            )

            dataset = TensorDataset(buf_z, buf_y)
            loader = DataLoader(
                dataset, batch_size=retrain_batch_size, shuffle=True,
                drop_last=False,
            )

            self.model.classifier.train()
            for epoch in range(retrain_epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                for z_batch, y_batch in loader:
                    # Latent noise augmentation to prevent overfitting
                    if latent_noise > 0:
                        z_batch = z_batch + torch.randn_like(z_batch) * latent_noise

                    logits = self.model.classifier(z_batch)

                    # Mask unseen classes during training
                    mask = torch.full_like(logits, float('-inf'))
                    for c in self.seen_classes:
                        mask[:, c] = 0.0
                    logits_masked = logits + mask

                    loss = F.cross_entropy(
                        logits_masked, y_batch,
                        label_smoothing=label_smoothing,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    correct += (logits_masked.argmax(1) == y_batch).sum().item()
                    total += y_batch.size(0)

                scheduler.step()
                acc = 100.0 * correct / max(total, 1)
                if epoch % 10 == 0 or epoch == retrain_epochs - 1:
                    self.logger.info(
                        f"  Retrain Epoch {epoch+1}/{retrain_epochs} — "
                        f"Loss: {total_loss / max(len(loader), 1):.4f}, "
                        f"Train Acc: {acc:.1f}%"
                    )

            # ===== EVALUATION =====
            self.logger.info("Evaluating on all tasks...")
            task_accuracies = self._evaluate_all_tasks(test_stream, task_id)
            self.metrics_tracker.update(task_id, task_accuracies)

            self.logger.info(f"  Task Accuracies: {task_accuracies}")
            aa = self.metrics_tracker.get_current_aa()
            self.logger.info(f"  Average Accuracy (AA): {aa:.2f}%")

            self.task_count += 1

        results = self._compile_results()
        self._log_final_summary(results)
        return results

    def train_task_free(self, benchmark: CLBenchmark) -> Dict:
        """
        Train without task boundaries (task-free mode).

        Uses Page-Hinkley drift detection to identify task transitions.

        Args:
            benchmark: CLBenchmark instance.

        Returns:
            results: Dictionary with metrics.
        """
        test_stream = benchmark.get_test_stream()
        stream = benchmark.get_task_free_stream(
            batch_size=self.config.get("wake_batch_size", 64),
            num_workers=self.config.get("num_workers", 4),
        )
        true_boundaries = stream.task_boundaries

        self.logger.info("=" * 60)
        self.logger.info("Task-Free Training with Drift Detection")
        self.logger.info(f"  True boundaries: {true_boundaries}")
        self.logger.info("=" * 60)

        batch_count = 0
        current_task_latents_z = []
        current_task_latents_y = []

        # We need to track all classes seen
        # In task-free mode, we discover classes as they come
        classes_this_segment = set()

        for batch in stream:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            # Track discovered classes
            new_cls = set(y.unique().tolist())
            classes_this_segment.update(new_cls)
            for c in new_cls:
                if c not in self.seen_classes:
                    self.seen_classes.append(c)

            # Encode and store latents for NREM
            with torch.no_grad():
                z = self.model.encode(x)
            current_task_latents_z.append(z.cpu())
            current_task_latents_y.append(y.cpu())

            # Wake step
            loss_val, drift = wake_step(
                self.model, self.null_projector, batch,
                self.classifier_optimizer, self.drift_detector,
            )

            batch_count += 1

            if drift:
                self.logger.info(f"\n  DRIFT DETECTED at batch {batch_count}!")

                # Save old classifier
                old_classifier = copy.deepcopy(self.model.classifier)
                old_classifier.eval()

                # Prepare latent data
                all_z = torch.cat(current_task_latents_z, dim=0)
                all_y = torch.cat(current_task_latents_y, dim=0)
                new_classes = list(classes_this_segment)

                # Update NSP with classifier's Jacobian on this segment's data
                latent_dataset = torch.utils.data.TensorDataset(
                    all_z.to(self.device), all_y.to(self.device),
                )
                latent_loader = DataLoader(
                    latent_dataset,
                    batch_size=self.config.get("fm_batch_size", 128),
                    shuffle=False,
                )
                self.null_projector.update(
                    self.model.classifier, latent_loader,
                    num_samples=self.config.get("nsp_num_samples", 500),
                )

                # NREM with buffer replay
                replay_data = self.latent_buffer.get_all()
                sleep_nrem(
                    flow_model=self.model.flow_model,
                    latent_data=(all_z, all_y),
                    null_projector=self.null_projector,
                    optimizer=self.fm_optimizer,
                    n_epochs=self.config.get("fm_epochs_per_task", 50),
                    batch_size=self.config.get("fm_batch_size", 128),
                    device=self.device,
                    replay_data=replay_data,
                )

                # Add to buffer after FM training
                self.latent_buffer.add(all_z, all_y)

                # REM
                sleep_rem(
                    flow_model=self.model.flow_model,
                    classifier=self.model.classifier,
                    old_classifier=old_classifier,
                    null_projector=self.null_projector,
                    seen_classes=self.seen_classes,
                    new_classes=new_classes,
                    optimizer=self.classifier_optimizer,
                    n_epochs=self.config.get("rem_epochs", 20),
                    replay_per_class=self.config.get("replay_per_class", 50),
                    kd_temperature=self.config.get("kd_temperature", 2.0),
                    num_ode_steps=self.config.get("num_ode_steps", 4),
                    device=self.device,
                )

                # Evaluate
                task_accuracies = self._evaluate_all_tasks(
                    test_stream, self.task_count
                )
                self.metrics_tracker.update(self.task_count, task_accuracies)
                aa = self.metrics_tracker.get_current_aa()
                self.logger.info(f"  AA after drift {self.task_count+1}: {aa:.2f}%")

                self.task_count += 1

                # Reset segment tracking
                current_task_latents_z = []
                current_task_latents_y = []
                classes_this_segment = set()

        # Final consolidation for last segment
        if current_task_latents_z:
            old_classifier = copy.deepcopy(self.model.classifier)
            old_classifier.eval()
            all_z = torch.cat(current_task_latents_z, dim=0)
            all_y = torch.cat(current_task_latents_y, dim=0)
            new_classes = list(classes_this_segment)

            # Update NSP with classifier's Jacobian on final segment
            latent_dataset = torch.utils.data.TensorDataset(
                all_z.to(self.device), all_y.to(self.device),
            )
            latent_loader = DataLoader(
                latent_dataset,
                batch_size=self.config.get("fm_batch_size", 128),
                shuffle=False,
            )
            self.null_projector.update(
                self.model.classifier, latent_loader,
                num_samples=self.config.get("nsp_num_samples", 500),
            )

            replay_data = self.latent_buffer.get_all()
            sleep_nrem(
                flow_model=self.model.flow_model,
                latent_data=(all_z, all_y),
                null_projector=self.null_projector,
                optimizer=self.fm_optimizer,
                n_epochs=self.config.get("fm_epochs_per_task", 50),
                device=self.device,
                replay_data=replay_data,
            )
            self.latent_buffer.add(all_z, all_y)
            sleep_rem(
                flow_model=self.model.flow_model,
                classifier=self.model.classifier,
                old_classifier=old_classifier,
                null_projector=self.null_projector,
                seen_classes=self.seen_classes,
                new_classes=new_classes,
                optimizer=self.classifier_optimizer,
                n_epochs=self.config.get("rem_epochs", 20),
                replay_per_class=self.config.get("replay_per_class", 50),
                device=self.device,
            )
            task_accuracies = self._evaluate_all_tasks(test_stream, self.task_count)
            self.metrics_tracker.update(self.task_count, task_accuracies)
            self.task_count += 1

        results = self._compile_results()
        results["drift_history"] = self.drift_detector.get_history()
        results["true_boundaries"] = true_boundaries
        self._log_final_summary(results)
        return results

    def _evaluate_all_tasks(
        self, test_stream, current_task_id: int,
    ) -> Dict[int, float]:
        """
        Evaluate classifier accuracy on all tasks seen so far.

        Args:
            test_stream: List of test experiences.
            current_task_id: Current task index (0-based).

        Returns:
            Dictionary mapping task_id → accuracy (%).
        """
        self.model.classifier.eval()
        task_accuracies = {}

        for task_id in range(min(current_task_id + 1, len(test_stream))):
            exp = test_stream[task_id]
            loader = exp.get_dataloader(
                batch_size=256, shuffle=False,
                num_workers=self.config.get("num_workers", 4),
            )

            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    z = self.model.encode(x)
                    logits = self.model.classifier(z)
                    # Output masking at eval: only consider seen classes
                    if len(self.seen_classes) > 0:
                        mask = torch.full_like(logits, float('-inf'))
                        for c in self.seen_classes:
                            mask[:, c] = 0.0
                        logits = logits + mask
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            accuracy = 100.0 * correct / max(total, 1)
            task_accuracies[task_id] = accuracy

        self.model.classifier.train()
        return task_accuracies

    def _compile_results(self) -> Dict:
        """Compile all results into a dictionary."""
        metrics = self.metrics_tracker.compute_final_metrics()
        return {
            "accuracy_matrix": self.metrics_tracker.get_accuracy_matrix(),
            "metrics": metrics,
            "seen_classes": self.seen_classes,
            "num_tasks": self.task_count,
            "config": self.config,
        }

    def _log_final_summary(self, results: Dict):
        """Print final summary of results."""
        metrics = results["metrics"]
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"  Average Accuracy (AA): {metrics.get('AA', 0):.2f}%")
        self.logger.info(f"  Backward Transfer (BWT): {metrics.get('BWT', 0):.2f}%")
        self.logger.info(f"  Forward Transfer (FWT): {metrics.get('FWT', 0):.2f}%")
        self.logger.info(f"  Forgetting Rate (FR): {metrics.get('FR', 0):.2f}%")
        self.logger.info("=" * 60)

    def save_results(self, results: Dict, output_dir: str = None):
        """Save results to disk."""
        if output_dir is None:
            output_dir = self.config.get("output_dir", "results/")
        os.makedirs(output_dir, exist_ok=True)

        # Save accuracy matrix
        matrix_path = os.path.join(output_dir, "accuracy_matrix.json")
        with open(matrix_path, "w") as f:
            json.dump(results["accuracy_matrix"], f, indent=2)

        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2)

        # Save model
        model_path = os.path.join(output_dir, "model_final.pt")
        torch.save({
            "classifier": self.model.classifier.state_dict(),
            "flow_model": self.model.flow_model.state_dict(),
            "seen_classes": self.seen_classes,
            "task_count": self.task_count,
        }, model_path)

        self.logger.info(f"Results saved to {output_dir}")

    def save_state(self, path: str):
        """Save full strategy state for resumption."""
        torch.save({
            "model_classifier": self.model.classifier.state_dict(),
            "model_flow": self.model.flow_model.state_dict(),
            "classifier_optimizer": self.classifier_optimizer.state_dict(),
            "fm_optimizer": self.fm_optimizer.state_dict(),
            "seen_classes": self.seen_classes,
            "task_count": self.task_count,
            "null_projector_V_r": self.null_projector.V_r,
        }, path)

    def load_state(self, path: str):
        """Load strategy state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.classifier.load_state_dict(checkpoint["model_classifier"])
        self.model.flow_model.load_state_dict(checkpoint["model_flow"])
        self.classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer"])
        self.fm_optimizer.load_state_dict(checkpoint["fm_optimizer"])
        self.seen_classes = checkpoint["seen_classes"]
        self.task_count = checkpoint["task_count"]
        self.null_projector.V_r = checkpoint["null_projector_V_r"]
