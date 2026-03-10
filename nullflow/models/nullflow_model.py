"""
NullFlow Model — unified model combining all components.

Brings together:
    - Encoder (frozen ResNet-18 **or** VAE — selectable)
    - ConditionalFlowMatching (generative replay)
    - LatentClassifier (classification head)

Provides a clean interface for encoding, classifying, and generating.
"""

import torch
import torch.nn as nn
from typing import Optional

from .latent_encoder import LatentEncoder
from .resnet_encoder import ResNetEncoder
from .flow_matching import ConditionalFlowMatching
from .classifier import LatentClassifier


class NullFlowModel(nn.Module):
    """
    Unified NullFlow model combining encoder, flow matching, and classifier.

    The encoder is always frozen. Only the flow matching model and classifier
    are updated during continual learning.

    Supports two encoder types:
        - 'resnet': ImageNet-pretrained ResNet-18 (no data leakage)
        - 'vae':    Convolutional VAE (needs pre-training)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        flow_hidden_dim: int = 512,
        flow_num_layers: int = 3,
        classifier_hidden_dim: int = 256,
        num_classes_max: int = 200,
        in_channels: int = 3,
        image_size: int = 32,
        device: str = "cuda",
        encoder_type: str = "resnet",
        use_raw_features: bool = False,
        use_cosine_classifier: bool = False,
        cosine_temperature: float = 0.1,
        classifier_type: str = "mlp",
    ):
        """
        Args:
            latent_dim: Latent space dimensionality.
            flow_hidden_dim: Hidden dim of FM velocity network.
            flow_num_layers: Number of hidden layers in FM velocity network.
            classifier_hidden_dim: Hidden dim of classifier.
            num_classes_max: Maximum number of classes.
            in_channels: Image channels (3 for RGB).
            image_size: Input image spatial size.
            device: Device.
            encoder_type: 'resnet' (ImageNet, no leakage) or 'vae'.
            use_raw_features: If True, skip projection (use raw 512-d features).
            use_cosine_classifier: If True, use cosine similarity classifier.
            cosine_temperature: Temperature for cosine classifier logits.
            classifier_type: 'mlp' or 'linear' (single layer).
        """
        super().__init__()
        self.device = device
        self.encoder_type = encoder_type

        # Encoder — selectable
        if encoder_type == "resnet":
            self.encoder = ResNetEncoder(
                latent_dim=latent_dim,
                in_channels=in_channels,
                image_size=image_size,
                device=device,
                use_raw_features=use_raw_features,
            )
        else:
            self.encoder = LatentEncoder(
                latent_dim=latent_dim,
                in_channels=in_channels,
                image_size=image_size,
                device=device,
            )

        # Use the encoder's actual latent_dim (may differ in raw mode)
        actual_latent_dim = self.encoder.latent_dim
        self.latent_dim = actual_latent_dim

        # Trainable Flow Matching model
        self.flow_model = ConditionalFlowMatching(
            latent_dim=actual_latent_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
            num_classes_max=num_classes_max,
        )

        # Trainable classifier
        self.classifier = LatentClassifier(
            latent_dim=actual_latent_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes_max,
            use_cosine=use_cosine_classifier,
            cosine_temperature=cosine_temperature,
            classifier_type=classifier_type,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to normalized latent vectors (no gradients).

        Args:
            x: Images, shape (B, C, H, W).

        Returns:
            Latent vectors, shape (B, latent_dim).
        """
        return self.encoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images (VAE encoder only).

        Args:
            z: Latent vectors, shape (B, latent_dim).

        Returns:
            Reconstructed images, shape (B, C, H, W).

        Raises:
            NotImplementedError: If using ResNet encoder (no decoder).
        """
        return self.encoder.decode(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Classify latent vectors.

        Args:
            z: Latent vectors, shape (B, latent_dim).

        Returns:
            Logits, shape (B, num_classes_max).
        """
        return self.classifier(z)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end prediction: images → class labels.

        Args:
            x: Images, shape (B, C, H, W).

        Returns:
            Predicted class labels, shape (B,).
        """
        z = self.encode(x)
        return self.classifier.predict(z)

    def generate_replay(
        self,
        n: int,
        class_label: int,
        num_steps: int = 4,
    ) -> torch.Tensor:
        """
        Generate replay latent vectors for a given class.

        Args:
            n: Number of samples.
            class_label: Class to generate for.
            num_steps: Number of Euler ODE steps.

        Returns:
            Generated latent vectors, shape (n, latent_dim).
        """
        return self.flow_model.sample(
            n=n, class_label=class_label,
            num_steps=num_steps, device=self.device,
        )

    def generate_replay_images(
        self,
        n: int,
        class_label: int,
        num_steps: int = 4,
    ) -> torch.Tensor:
        """
        Generate replay images for a given class (latents → decoder → images).

        Args:
            n: Number of samples.
            class_label: Class to generate for.
            num_steps: Number of Euler ODE steps.

        Returns:
            Generated images, shape (n, C, H, W).
        """
        z = self.generate_replay(n, class_label, num_steps)
        return self.decode(z)

    def to_device(self, device: str):
        """Move all sub-modules to device."""
        self.device = device
        self.to(device)
        self.encoder.device = device
        return self

    def get_trainable_params(self):
        """Return only trainable parameters (excludes frozen encoder)."""
        params = []
        params.extend(self.flow_model.parameters())
        params.extend(self.classifier.parameters())
        return params

    def summary(self) -> str:
        """Return a summary of model components and parameter counts."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        fm_params = self.flow_model.get_num_params()
        cls_params = self.classifier.get_num_params()
        total = enc_params + fm_params + cls_params
        trainable = fm_params + cls_params

        enc_label = (
            f"Encoder [{self.encoder_type}] (frozen)"
            if hasattr(self, "encoder_type")
            else "Encoder (frozen)"
        )

        return (
            f"NullFlow Model Summary:\n"
            f"  {enc_label}: {enc_params:,} params\n"
            f"  Flow Matching:    {fm_params:,} params\n"
            f"  Classifier:       {cls_params:,} params\n"
            f"  Total:            {total:,} params\n"
            f"  Trainable:        {trainable:,} params"
        )
