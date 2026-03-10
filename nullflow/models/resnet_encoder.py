"""
Pre-trained ResNet-18 encoder — eliminates data leakage.

Uses ImageNet-pretrained features as a frozen backbone, providing a
universal feature extractor that is completely independent of the
target CL benchmark.  This resolves the methodological concern of
pre-training a VAE on the full target dataset (which sees future-task
data before CL begins).

For small images (≤64px, e.g. CIFAR-32), we replace the first 7×7 conv
with a 3×3 conv and remove maxpool, following standard practice in
self-supervised learning on small images (SimCLR, SupCon, etc.).

The trainable projection head (512 → latent_dim) is calibrated on the
first task's data only using VICReg-style variance-covariance
regularization (no labels required), then frozen for all subsequent tasks.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

logger = logging.getLogger(__name__)


class ResNetEncoder(nn.Module):
    """
    Frozen ResNet-18 encoder with trainable projection head.

    Architecture:
        Input image → ResNet-18 backbone (frozen ImageNet weights)
        → 512-d features → Linear(512, latent_dim) → LayerNorm → z

    The projection head is calibrated on the first task's data
    (unsupervised), then frozen for all subsequent tasks.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 3,
        image_size: int = 32,
        device: str = "cuda",
        use_raw_features: bool = False,
    ):
        """
        Args:
            latent_dim: Dimensionality of output latent vectors.
            in_channels: Number of input image channels (3 for RGB).
            image_size: Spatial size of input images (32 for CIFAR).
            device: Torch device string.
            use_raw_features: If True, skip the learned projection head and
                use raw backbone features (512-d) with normalization only.
                latent_dim MUST equal 512 when this is True.
        """
        super().__init__()
        self.image_size = image_size
        self.device = device
        self._is_calibrated = False
        self.use_raw_features = use_raw_features

        # Load ImageNet-pretrained ResNet-18
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        except Exception:
            resnet = models.resnet18(pretrained=True)

        # NOTE: We keep the original ImageNet conv1 (7×7, stride=2) and
        # maxpool even for small images. Replacing conv1 with a random 3×3
        # destroys the pretrained features — catastrophic for a frozen
        # backbone.  AdaptiveAvgPool at the end handles any spatial dims.

        # Extract backbone (drop the final FC)
        self.feature_dim = resnet.fc.in_features  # 512
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Freeze backbone entirely
        for param in self.backbone.parameters():
            param.requires_grad = False

        if use_raw_features:
            # Raw mode: no learned projection, just pass-through.
            # latent_dim is forced to feature_dim (512).
            self.latent_dim = self.feature_dim
            if latent_dim != self.feature_dim:
                logger.warning(
                    f"use_raw_features=True forces latent_dim={self.feature_dim} "
                    f"(was {latent_dim})"
                )
            self.projection = nn.Identity()
        else:
            self.latent_dim = latent_dim
            # Trainable projection head
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )

        # Latent normalisation statistics (computed during calibration)
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_std", torch.ones(self.latent_dim))

    # -----------------------------------------------------------------
    # Core interface (matches LatentEncoder API)
    # -----------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.

        Args:
            x: Images, shape (B, C, H, W).

        Returns:
            Latent vectors, shape (B, latent_dim).
        """
        with torch.no_grad():
            features = self.backbone(x)  # (B, 512)
        z = self.projection(features)    # (B, latent_dim)
        # Normalise if calibrated
        if self._is_calibrated:
            z = (z - self.latent_mean) / (self.latent_std + 1e-8)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Not supported for ResNet encoder (no decoder)."""
        raise NotImplementedError(
            "ResNet encoder does not support decoding. "
            "Use the VAE encoder for reconstruction / visualisation."
        )

    def freeze(self):
        """Freeze all parameters (backbone + projection)."""
        for param in self.parameters():
            param.requires_grad = False

    # -----------------------------------------------------------------
    # Calibration
    # -----------------------------------------------------------------

    def calibrate(
        self,
        data_loader,
        epochs: int = 10,
        lr: float = 1e-3,
        verbose: bool = True,
        num_classes: int = 100,
    ):
        """
        Calibrate the projection head using supervised + decorrelation loss.

        A temporary linear classifier is trained jointly with the projection
        head using cross-entropy on the first task's data.  Decorrelation
        regularisation prevents dimension collapse.  After calibration the
        temporary classifier is discarded and only the projection head is
        kept.

        When use_raw_features=True, only computes normalisation statistics
        (no projection training — the projection is an identity).

        This should be called ONCE on the first task's data, then the
        encoder is frozen for all subsequent tasks.

        Args:
            data_loader: DataLoader yielding (images, labels).
            epochs: Calibration epochs (default 10).
            lr: Learning rate for projection head.
            verbose: Print progress.
            num_classes: Total number of classes in the dataset.
        """
        if self.use_raw_features:
            # Raw features mode: no projection training, just normalisation.
            if verbose:
                logger.info("  Raw features mode — computing normalisation "
                           "statistics only (no projection training).")
            self._compute_latent_stats(data_loader)
            self._is_calibrated = True
            if verbose:
                logger.info("  Raw features calibrated (normalisation only).")
            return

        # Temporary supervised head (discarded after calibration)
        temp_head = nn.Linear(self.latent_dim, num_classes).to(self.device)
        nn.init.kaiming_normal_(temp_head.weight, nonlinearity="linear")
        nn.init.zeros_(temp_head.bias)

        params = list(self.projection.parameters()) + list(temp_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        self.backbone.eval()
        self.projection.train()

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            n_batches = 0
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    features = self.backbone(x)
                z = self.projection(features)
                logits = temp_head(z)

                # Supervised cross-entropy loss
                ce_loss = F.cross_entropy(logits, y)

                # Decorrelation regularisation (prevents dimension collapse)
                z_c = z - z.mean(dim=0)
                cov = (z_c.T @ z_c) / max(z.shape[0] - 1, 1)
                cov_loss = cov.fill_diagonal_(0).pow(2).sum() / self.latent_dim

                loss = ce_loss + 0.01 * cov_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)
                n_batches += 1

            acc = 100.0 * correct / max(total, 1)
            if verbose:
                logger.info(
                    f"  Calibration Epoch {epoch + 1}/{epochs} — "
                    f"Loss: {total_loss / max(n_batches, 1):.4f}, "
                    f"Acc: {acc:.1f}%"
                )

        # Discard temporary classifier
        del temp_head

        # Compute latent normalisation statistics
        self._compute_latent_stats(data_loader)
        self._is_calibrated = True

        if verbose:
            logger.info("  Projection head calibrated.")

    def _compute_latent_stats(self, data_loader):
        """Compute mean and std of latent representations."""
        all_z = []
        self.eval()
        with torch.no_grad():
            for x, _y in data_loader:
                x = x.to(self.device)
                features = self.backbone(x)
                z = self.projection(features)
                all_z.append(z)
        all_z = torch.cat(all_z, dim=0)
        self.latent_mean.copy_(all_z.mean(dim=0))
        self.latent_std.copy_(all_z.std(dim=0).clamp(min=1e-6))

    # This serves as the equivalent of VAE's pretrain() method
    def pretrain(self, loader, epochs=5, lr=1e-3, kl_weight=None, verbose=True,
                 num_classes=100):
        """Alias for calibrate() — matches LatentEncoder.pretrain() API."""
        self.calibrate(loader, epochs=epochs, lr=lr, verbose=verbose,
                       num_classes=num_classes)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save_pretrained(self, path: str):
        """Save projection head and normalisation stats."""
        state = {
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "is_calibrated": self._is_calibrated,
            "use_raw_features": self.use_raw_features,
        }
        if not self.use_raw_features:
            state["projection"] = self.projection.state_dict()
        torch.save(state, path)

    def load_pretrained(self, path: str):
        """Load projection head and normalisation stats."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        if not self.use_raw_features and "projection" in data:
            self.projection.load_state_dict(data["projection"])
        self.latent_mean.copy_(data["latent_mean"])
        self.latent_std.copy_(data["latent_std"])
        self._is_calibrated = data.get("is_calibrated", True)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary."""
        backbone_p = sum(p.numel() for p in self.backbone.parameters())
        proj_p = sum(p.numel() for p in self.projection.parameters())
        mode = "raw" if self.use_raw_features else "learned"
        return (
            f"ResNetEncoder: backbone={backbone_p:,} (frozen ImageNet), "
            f"projection={proj_p:,} ({mode}), latent_dim={self.latent_dim}"
        )

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
