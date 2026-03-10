"""
Conditional Flow Matching in latent space.

Implements a class-conditional flow matching model that learns the velocity field
v_θ(t, z, c) mapping noise z₀ ~ N(0,I) to data latents z₁ ~ q(z|c).

The optimal transport conditional flow matching (OT-CFM) loss is:
    L_FM = E_{t~U[0,1], z₀~N(0,I), z₁~q(z)} [ ||v_θ(t, z_t, c) - (z₁ - z₀)||² ]
    with z_t = (1-t)*z₀ + t*z₁ (linear interpolation = OT path)

Sampling uses Euler integration of the ODE dz/dt = v_θ(t, z, c) from t=0 to t=1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VelocityNetwork(nn.Module):
    """
    MLP velocity network with skip connections for flow matching.

    Architecture: [z; t; c_emb] → 512 → 512 → 512 → [d]
    where z is the latent, t is the time, and c_emb is the class embedding.

    Features:
        - SiLU (Swish) activations
        - Layer Normalization after each hidden layer
        - Skip connection: output = f(input) + linear_skip(input)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_classes_max: int = 200,
        class_emb_dim: int = 128,
    ):
        """
        Args:
            latent_dim: Dimensionality of the latent space.
            hidden_dim: Width of hidden layers.
            num_layers: Number of hidden layers (default 3).
            num_classes_max: Maximum number of classes (for embedding table).
            class_emb_dim: Dimensionality of class embeddings.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes_max = num_classes_max

        # Class embedding: c → c_emb
        self.class_embedding = nn.Embedding(num_classes_max, class_emb_dim)

        # Input: [z; t; c_emb] → total input dim
        input_dim = latent_dim + 1 + class_emb_dim

        # Build hidden layers
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.SiLU())
        self.trunk = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Skip connection: input → output dimension
        self.skip_proj = nn.Linear(input_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize class embeddings
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity v_θ(t, z, c).

        Args:
            t: Time values, shape (B,) or (B, 1), in [0, 1].
            z: Latent vectors, shape (B, d).
            c: Class labels, shape (B,), dtype long. If None, uses class 0.

        Returns:
            Predicted velocity, shape (B, d).
        """
        B = z.shape[0]

        # Ensure t has shape (B, 1)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B).unsqueeze(1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        assert t.shape == (B, 1), f"Expected t shape ({B}, 1), got {t.shape}"

        # Get class embedding
        if c is None:
            c = torch.zeros(B, dtype=torch.long, device=z.device)
        c_emb = self.class_embedding(c)  # (B, class_emb_dim)

        # Concatenate input: [z; t; c_emb]
        x = torch.cat([z, t, c_emb], dim=1)

        # Forward through trunk + skip connection
        h = self.trunk(x)
        velocity = self.output_proj(h) + self.skip_proj(x)

        return velocity


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching model for latent-space generative replay.

    Learns a velocity field v_θ(t, z, c) such that integrating the ODE
    dz/dt = v_θ from t=0 (noise) to t=1 (data) generates latent vectors
    from the learned distribution, conditioned on class label c.

    Key features:
        - Class-conditional generation for targeted replay per class
        - OT-CFM loss for stable training
        - Fast Euler sampling (as few as 4 steps vs. 1000 for DDPM)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_classes_max: int = 200,
    ):
        """
        Args:
            latent_dim: Dimensionality of the latent space.
            hidden_dim: Width of velocity network hidden layers.
            num_layers: Number of hidden layers in velocity network.
            num_classes_max: Maximum number of classes supported.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.velocity_net = VelocityNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes_max=num_classes_max,
        )

    def compute_loss(
        self,
        z1: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the OT-CFM training loss.

        L_FM = E_{t, z₀, z₁} [ ||v_θ(t, z_t, c) - (z₁ - z₀)||² ]
        where z_t = (1-t)*z₀ + t*z₁

        Args:
            z1: Target latent vectors (data), shape (B, d).
            c: Class labels, shape (B,), dtype long.

        Returns:
            Scalar loss value.
        """
        B, d = z1.shape
        assert d == self.latent_dim, f"Expected latent dim {self.latent_dim}, got {d}"

        # Sample source noise z₀ ~ N(0, I)
        z0 = torch.randn_like(z1)

        # Sample random time t ~ U[0, 1]
        t = torch.rand(B, device=z1.device)

        # Interpolate: z_t = (1-t)*z₀ + t*z₁
        t_expand = t.unsqueeze(1)  # (B, 1)
        z_t = (1.0 - t_expand) * z0 + t_expand * z1

        # Target velocity: dz/dt = z₁ - z₀ (the OT path)
        target_velocity = z1 - z0

        # Predicted velocity
        pred_velocity = self.velocity_net(t, z_t, c)

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample(
        self,
        n: int,
        class_label: Optional[int] = None,
        num_steps: int = 4,
        device: str = "cuda",
        solver: str = "euler",
    ) -> torch.Tensor:
        """
        Generate latent samples by integrating the learned ODE.

        Supports two solvers:
          - 'euler': 1st-order, 1 function eval/step  (fast)
          - 'heun':  2nd-order, 2 function evals/step (better quality)

        Args:
            n: Number of samples to generate.
            class_label: If provided, generate samples for this class.
                         If None, uses class 0.
            num_steps: Number of integration steps (default 4).
            device: Device for generation.
            solver: ODE solver — 'euler' or 'heun'.

        Returns:
            Generated latent vectors z₁, shape (n, latent_dim).
        """
        self.eval()
        dt = 1.0 / num_steps

        # Initial noise z₀ ~ N(0, I)
        z = torch.randn(n, self.latent_dim, device=device)

        # Class conditioning
        if class_label is not None:
            c = torch.full((n,), class_label, dtype=torch.long, device=device)
        else:
            c = torch.zeros(n, dtype=torch.long, device=device)

        for step in range(num_steps):
            t_i = torch.full((n,), step * dt, device=device)
            v1 = self.velocity_net(t_i, z, c)

            if solver == "heun" and step < num_steps - 1:
                # Heun's method: predictor-corrector
                z_pred = z + dt * v1
                t_next = torch.full((n,), (step + 1) * dt, device=device)
                v2 = self.velocity_net(t_next, z_pred, c)
                z = z + 0.5 * dt * (v1 + v2)
            else:
                # Euler step
                z = z + dt * v1

        return z

    @torch.no_grad()
    def sample_batch(
        self,
        class_labels: torch.Tensor,
        num_steps: int = 4,
        solver: str = "euler",
    ) -> torch.Tensor:
        """
        Generate latent samples for a batch of class labels.

        Args:
            class_labels: Class labels, shape (B,), dtype long.
            num_steps: Number of integration steps.
            solver: ODE solver — 'euler' or 'heun'.

        Returns:
            Generated latent vectors, shape (B, latent_dim).
        """
        self.eval()
        B = class_labels.shape[0]
        device = class_labels.device
        dt = 1.0 / num_steps

        z = torch.randn(B, self.latent_dim, device=device)

        for step in range(num_steps):
            t_i = torch.full((B,), step * dt, device=device)
            v1 = self.velocity_net(t_i, z, class_labels)

            if solver == "heun" and step < num_steps - 1:
                z_pred = z + dt * v1
                t_next = torch.full((B,), (step + 1) * dt, device=device)
                v2 = self.velocity_net(t_next, z_pred, class_labels)
                z = z + 0.5 * dt * (v1 + v2)
            else:
                z = z + dt * v1

        return z

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
