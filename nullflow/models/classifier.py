"""
Latent Classifier — classification head operating in the latent space.

A lightweight MLP classifier that takes latent vectors z ∈ ℝ^d from the frozen
VAE encoder and produces class logits. This is the component that learns
incrementally during the Wake phase.

Supports two modes:
  - MLP mode (default): z → hidden → hidden → logits
  - Cosine mode: logits = cos(z, W) / τ  (no hidden layers)
    Cosine mode is particularly suited for class-incremental learning:
    normalised weights remove recency bias and enable NCM-style proto-init.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentClassifier(nn.Module):
    """
    Classifier operating in the latent space.

    MLP mode:    z → hidden → hidden → logits  (SiLU + LN + Dropout)
    Cosine mode: logits = cos_sim(z, W) / temperature  (single Linear, no bias)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 200,
        dropout: float = 0.1,
        use_cosine: bool = False,
        cosine_temperature: float = 0.1,
        classifier_type: str = "mlp",
    ):
        """
        Args:
            latent_dim: Dimensionality of input latent vectors.
            hidden_dim: Width of hidden layers (ignored in cosine/linear mode).
            num_classes: Maximum number of output classes.
            dropout: Dropout probability (ignored in cosine/linear mode).
            use_cosine: If True, use cosine similarity classifier.
            cosine_temperature: Temperature scaling for cosine logits.
            classifier_type: 'mlp' (default) or 'linear' (single layer, no hidden).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self._use_cosine = use_cosine
        self.cosine_temperature = cosine_temperature
        self._classifier_type = classifier_type

        if classifier_type == "linear":
            # Single linear layer: latent_dim → num_classes
            # Much better for low-data regimes (e.g., 20 exemplars/class)
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, num_classes),
            )
        elif use_cosine:
            # Same MLP hidden layers, but last Linear has no bias.
            # Forward applies L2 normalisation to features & weights.
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes, bias=False),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Classify latent vectors.

        In cosine mode, computes:  logit_c = cos(z, W_c) / temperature
        In MLP mode:               logits = MLP(z)

        Args:
            z: Latent vectors, shape (B, latent_dim).

        Returns:
            Logits, shape (B, num_classes).
        """
        assert z.shape[-1] == self.latent_dim, \
            f"Expected latent dim {self.latent_dim}, got {z.shape[-1]}"

        if self._classifier_type == "linear":
            return self.classifier(z)

        if self._use_cosine:
            # Pass through hidden layers (everything except last Linear)
            h = self.classifier[:-1](z)                    # (B, hidden_dim)
            h_norm = F.normalize(h, p=2, dim=1)
            w = self.classifier[-1].weight                 # (num_classes, hidden_dim)
            w_norm = F.normalize(w, p=2, dim=1)
            return (h_norm @ w_norm.T) / self.cosine_temperature

        return self.classifier(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels from latent vectors.

        Args:
            z: Latent vectors, shape (B, latent_dim).

        Returns:
            Predicted class labels, shape (B,).
        """
        logits = self.forward(z)
        return logits.argmax(dim=1)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
