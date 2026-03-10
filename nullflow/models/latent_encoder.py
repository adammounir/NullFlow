"""
Latent Encoder — Frozen VAE for projecting images into a compact latent space.

The encoder is a lightweight Convolutional VAE that maps input images x ∈ ℝ^{C×H×W}
to latent vectors z ∈ ℝ^d (d=128 for CIFAR, d=256 for TinyImageNet).
All parameters are FROZEN after initialization — no gradients flow through the encoder.

The encoder also provides a decoder for reconstructing images from latents (used for
visualization of generated samples).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    """Convolutional encoder: images → latent distribution (mu, logvar)."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, image_size: int = 32):
        """
        Args:
            in_channels: Number of input image channels (3 for RGB).
            latent_dim: Dimensionality of the latent space.
            image_size: Spatial size of input images (assumes square).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Determine architecture based on image size
        if image_size <= 32:
            # Conv-4 encoder for CIFAR-sized images (32×32)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1),   # -> 16×16
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, 2, 1),             # -> 8×8
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, 2, 1),            # -> 4×4
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, 2, 1),           # -> 2×2
                nn.BatchNorm2d(256),
                nn.SiLU(),
            )
            self.flat_dim = 256 * 2 * 2
        elif image_size <= 64:
            # Conv-5 encoder for TinyImageNet (64×64)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1),   # -> 32×32
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, 2, 1),             # -> 16×16
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, 2, 1),            # -> 8×8
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, 2, 1),           # -> 4×4
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.Conv2d(256, 512, 4, 2, 1),           # -> 2×2
                nn.BatchNorm2d(512),
                nn.SiLU(),
            )
            self.flat_dim = 512 * 2 * 2
        else:
            # Conv-6 encoder for larger images (128×128+)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 4, 2, 1),   # -> H/2
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, 2, 1),             # -> H/4
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, 2, 1),            # -> H/8
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, 2, 1),           # -> H/16
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.Conv2d(256, 512, 4, 2, 1),           # -> H/32
                nn.BatchNorm2d(512),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(2),                 # -> 2×2
            )
            self.flat_dim = 512 * 2 * 2

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Encode images to latent distribution parameters.

        Args:
            x: Input images, shape (B, C, H, W).

        Returns:
            mu: Mean of latent distribution, shape (B, latent_dim).
            logvar: Log-variance of latent distribution, shape (B, latent_dim).
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvDecoder(nn.Module):
    """Convolutional decoder: latent vectors → reconstructed images."""

    def __init__(self, latent_dim: int = 128, out_channels: int = 3, image_size: int = 32):
        """
        Args:
            latent_dim: Dimensionality of the latent space.
            out_channels: Number of output image channels.
            image_size: Spatial size of output images.
        """
        super().__init__()
        self.image_size = image_size

        if image_size <= 32:
            self.flat_dim = 256 * 2 * 2
            self.fc = nn.Linear(latent_dim, self.flat_dim)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),   # -> 4×4
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),    # -> 8×8
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),     # -> 16×16
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # -> 32×32
                nn.Sigmoid(),
            )
            self.reshape = (256, 2, 2)
        elif image_size <= 64:
            self.flat_dim = 512 * 2 * 2
            self.fc = nn.Linear(latent_dim, self.flat_dim)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),   # -> 4×4
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),   # -> 8×8
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),    # -> 16×16
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),     # -> 32×32
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # -> 64×64
                nn.Sigmoid(),
            )
            self.reshape = (512, 2, 2)
        else:
            self.flat_dim = 512 * 2 * 2
            self.fc = nn.Linear(latent_dim, self.flat_dim)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),   # -> 4×4
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),   # -> 8×8
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),    # -> 16×16
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),     # -> 32×32
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),     # -> 64×64
                nn.BatchNorm2d(16),
                nn.SiLU(),
                nn.ConvTranspose2d(16, out_channels, 4, 2, 1),  # -> 128×128
                nn.Sigmoid(),
            )
            self.reshape = (512, 2, 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            z: Latent vectors, shape (B, latent_dim).

        Returns:
            Reconstructed images, shape (B, C, H, W).
        """
        h = self.fc(z)
        h = h.view(h.size(0), *self.reshape)
        return self.decoder(h)


class LatentEncoder(nn.Module):
    """
    Frozen VAE encoder/decoder for continual learning in latent space.

    The VAE is pre-trained on the full dataset (before continual learning begins)
    and then frozen. All parameters have requires_grad=False.

    Provides:
        - encode(x) → z: Encode images to normalized latent vectors.
        - decode(z) → x̂: Decode latent vectors back to images.
        - pretrain(dataloader, epochs): Pre-train the VAE on the full dataset.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 3,
        image_size: int = 32,
        device: str = "cuda",
    ):
        """
        Args:
            latent_dim: Dimensionality of the latent space (128 for CIFAR, 256 for TinyImageNet).
            in_channels: Number of input channels (3 for RGB).
            image_size: Spatial size of input images.
            device: Device to place the model on.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.device = device

        self.encoder = ConvEncoder(in_channels, latent_dim, image_size)
        self.decoder = ConvDecoder(latent_dim, in_channels, image_size)

        # Latent normalization statistics (updated during pretraining)
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))
        self._is_pretrained = False

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * exp(0.5 * logvar).

        Args:
            mu: Mean, shape (B, d).
            logvar: Log-variance, shape (B, d).

        Returns:
            Sampled latent vector, shape (B, d).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean at eval time

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to normalized latent vectors.

        Args:
            x: Input images, shape (B, C, H, W).

        Returns:
            z: Normalized latent vectors, shape (B, latent_dim).
                Approximately mean=0, std=1 after normalization.
        """
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            # Normalize latents to have approx. mean=0, std=1
            z = (z - self.latent_mean.to(z.device)) / (self.latent_std.to(z.device) + 1e-8)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            z: Normalized latent vectors, shape (B, latent_dim).

        Returns:
            Reconstructed images, shape (B, C, H, W).
        """
        with torch.no_grad():
            # Un-normalize latents
            z_unnorm = z * (self.latent_std.to(z.device) + 1e-8) + self.latent_mean.to(z.device)
            return self.decoder(z_unnorm)

    def _vae_loss(self, x: torch.Tensor, x_recon: torch.Tensor,
                  mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 1e-3) -> torch.Tensor:
        """
        VAE loss = reconstruction (MSE) + KL divergence.

        Args:
            x: Original images, shape (B, C, H, W).
            x_recon: Reconstructed images, shape (B, C, H, W).
            mu: Latent mean, shape (B, d).
            logvar: Latent log-variance, shape (B, d).
            kl_weight: Weight for KL term (beta-VAE style).

        Returns:
            Total loss (scalar).
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss

    def pretrain(self, dataloader, epochs: int = 30, lr: float = 1e-3,
                 kl_weight: float = 1e-3, verbose: bool = True):
        """
        Pre-train the VAE on the full dataset before continual learning.

        After pretraining, computes latent normalization statistics and freezes
        all parameters.

        Args:
            dataloader: DataLoader yielding (images, labels) batches.
            epochs: Number of pretraining epochs.
            lr: Learning rate for Adam optimizer.
            kl_weight: Weight for KL divergence term.
            verbose: Whether to print training progress.
        """
        from tqdm import tqdm

        self.train()
        for p in self.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            pbar = tqdm(dataloader, desc=f"VAE Pretrain [{epoch+1}/{epochs}]",
                        disable=not verbose)
            for x, _ in pbar:
                x = x.to(self.device)
                mu, logvar = self.encoder(x)
                z = self.reparameterize(mu, logvar)
                x_recon = self.decoder(z)
                loss = self._vae_loss(x, x_recon, mu, logvar, kl_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(num_batches, 1)
            if verbose:
                logger.info(f"  Epoch {epoch+1}/{epochs} \u2014 Avg Loss: {avg_loss:.4f}")

        # Compute latent normalization statistics
        self._compute_latent_stats(dataloader)

        # Freeze all parameters
        self.freeze()
        self._is_pretrained = True

    def _compute_latent_stats(self, dataloader):
        """
        Compute mean and std of latent representations across the dataset.

        Args:
            dataloader: DataLoader yielding (images, labels) batches.
        """
        self.eval()
        all_mu = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                mu, _ = self.encoder(x)
                all_mu.append(mu)
        all_mu = torch.cat(all_mu, dim=0)
        self.latent_mean = all_mu.mean(dim=0)
        self.latent_std = all_mu.std(dim=0).clamp(min=1e-8)

    def freeze(self):
        """Freeze all parameters (no gradient computation)."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @property
    def is_pretrained(self) -> bool:
        return self._is_pretrained

    def save_pretrained(self, path: str):
        """Save pretrained VAE state dict."""
        torch.save({
            "state_dict": self.state_dict(),
            "latent_dim": self.latent_dim,
            "image_size": self.image_size,
            "is_pretrained": self._is_pretrained,
        }, path)

    def load_pretrained(self, path: str):
        """Load pretrained VAE state dict and freeze."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        self._is_pretrained = checkpoint.get("is_pretrained", True)
        self.freeze()
