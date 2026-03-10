"""
Tests for new NullFlow components:
    - ResNetEncoder
    - LatentReplayBuffer
    - Heun solver
    - Encoder selection in NullFlowModel
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.models.resnet_encoder import ResNetEncoder
from nullflow.data.latent_buffer import LatentReplayBuffer
from nullflow.models.flow_matching import ConditionalFlowMatching
from nullflow.models.nullflow_model import NullFlowModel


class TestResNetEncoder(unittest.TestCase):
    """Tests for the ImageNet-pretrained ResNet-18 encoder."""

    def setUp(self):
        self.encoder = ResNetEncoder(
            latent_dim=64, in_channels=3, image_size=32, device="cpu",
        ).to("cpu")

    def test_encode_shape(self):
        x = torch.randn(4, 3, 32, 32)
        z = self.encoder.encode(x)
        self.assertEqual(z.shape, (4, 64))

    def test_encode_no_nan(self):
        x = torch.randn(2, 3, 32, 32)
        z = self.encoder.encode(x)
        self.assertFalse(torch.isnan(z).any())

    def test_backbone_frozen(self):
        for p in self.encoder.backbone.parameters():
            self.assertFalse(p.requires_grad)

    def test_projection_trainable_before_freeze(self):
        trainable = sum(p.requires_grad for p in self.encoder.projection.parameters())
        self.assertGreater(trainable, 0)

    def test_freeze(self):
        self.encoder.freeze()
        trainable = sum(
            p.requires_grad for p in self.encoder.parameters()
        )
        self.assertEqual(trainable, 0)

    def test_calibrate(self):
        dataset = TensorDataset(torch.randn(16, 3, 32, 32), torch.zeros(16).long())
        loader = DataLoader(dataset, batch_size=8)
        self.encoder.calibrate(loader, epochs=1, verbose=False)
        self.assertTrue(self.encoder._is_calibrated)

    def test_save_load_roundtrip(self):
        import tempfile
        self.encoder._is_calibrated = True
        self.encoder.latent_mean.fill_(1.0)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self.encoder.save_pretrained(path)
            encoder2 = ResNetEncoder(latent_dim=64, device="cpu").to("cpu")
            encoder2.load_pretrained(path)
            self.assertTrue(encoder2._is_calibrated)
            self.assertTrue(torch.allclose(encoder2.latent_mean, torch.ones(64)))
        finally:
            os.unlink(path)


class TestLatentReplayBuffer(unittest.TestCase):
    """Tests for the latent replay buffer."""

    def test_add_and_get(self):
        buf = LatentReplayBuffer(max_per_class=10, max_total=100)
        z = torch.randn(5, 16)
        y = torch.tensor([0, 0, 1, 1, 2])
        buf.add(z, y)
        self.assertEqual(len(buf), 5)
        self.assertSetEqual(set(buf.get_classes()), {0, 1, 2})

    def test_ring_buffer_overflow(self):
        buf = LatentReplayBuffer(max_per_class=3, max_total=100)
        z = torch.randn(10, 8)
        y = torch.zeros(10).long()
        buf.add(z, y)
        self.assertEqual(len(buf), 3)  # capped at max_per_class

    def test_get_all(self):
        buf = LatentReplayBuffer(max_per_class=5, max_total=100)
        z1 = torch.randn(3, 8)
        y1 = torch.zeros(3).long()
        z2 = torch.randn(4, 8)
        y2 = torch.ones(4).long()
        buf.add(z1, y1)
        buf.add(z2, y2)
        all_z, all_y = buf.get_all()
        self.assertEqual(all_z.shape[0], 7)
        self.assertEqual(all_y.shape[0], 7)

    def test_sample(self):
        buf = LatentReplayBuffer(max_per_class=50, max_total=1000)
        buf.add(torch.randn(20, 8), torch.zeros(20).long())
        buf.add(torch.randn(20, 8), torch.ones(20).long())
        z, y = buf.sample(10)
        self.assertEqual(z.shape[0], 10)

    def test_sample_balanced(self):
        buf = LatentReplayBuffer(max_per_class=50, max_total=1000)
        buf.add(torch.randn(20, 8), torch.zeros(20).long())
        buf.add(torch.randn(20, 8), torch.ones(20).long())
        z, y = buf.sample_balanced(per_class=5)
        self.assertEqual(z.shape[0], 10)  # 5 per class × 2 classes

    def test_empty_buffer(self):
        buf = LatentReplayBuffer()
        self.assertIsNone(buf.get_all())
        self.assertIsNone(buf.sample(5))

    def test_global_cap(self):
        buf = LatentReplayBuffer(max_per_class=10, max_total=15)
        buf.add(torch.randn(10, 8), torch.zeros(10).long())
        buf.add(torch.randn(10, 8), torch.ones(10).long())
        self.assertLessEqual(len(buf), 15)


class TestHeunSolver(unittest.TestCase):
    """Test the Heun ODE solver in Flow Matching."""

    def setUp(self):
        self.fm = ConditionalFlowMatching(
            latent_dim=16, hidden_dim=32, num_layers=2, num_classes_max=10,
        )

    def test_euler_sample_shape(self):
        z = self.fm.sample(8, class_label=0, num_steps=4, device="cpu", solver="euler")
        self.assertEqual(z.shape, (8, 16))

    def test_heun_sample_shape(self):
        z = self.fm.sample(8, class_label=0, num_steps=4, device="cpu", solver="heun")
        self.assertEqual(z.shape, (8, 16))

    def test_heun_different_from_euler(self):
        torch.manual_seed(42)
        z_euler = self.fm.sample(4, class_label=0, num_steps=4, device="cpu", solver="euler")
        torch.manual_seed(42)
        z_heun = self.fm.sample(4, class_label=0, num_steps=4, device="cpu", solver="heun")
        # They use same init but different integration → should differ
        self.assertFalse(torch.allclose(z_euler, z_heun, atol=1e-4))

    def test_heun_sample_batch(self):
        labels = torch.tensor([0, 1, 2, 3])
        z = self.fm.sample_batch(labels, num_steps=4, solver="heun")
        self.assertEqual(z.shape, (4, 16))


class TestEncoderSelection(unittest.TestCase):
    """Test that NullFlowModel correctly selects encoder type."""

    def test_resnet_encoder(self):
        model = NullFlowModel(
            latent_dim=64, num_classes_max=10, image_size=32,
            device="cpu", encoder_type="resnet",
        )
        self.assertIsInstance(model.encoder, ResNetEncoder)
        x = torch.randn(2, 3, 32, 32)
        z = model.encode(x)
        self.assertEqual(z.shape, (2, 64))

    def test_vae_encoder_default_backward_compat(self):
        """Passing encoder_type='vae' should use LatentEncoder."""
        from nullflow.models.latent_encoder import LatentEncoder
        model = NullFlowModel(
            latent_dim=64, num_classes_max=10, image_size=32,
            device="cpu", encoder_type="vae",
        )
        self.assertIsInstance(model.encoder, LatentEncoder)

    def test_summary_contains_encoder_type(self):
        model = NullFlowModel(
            latent_dim=64, num_classes_max=10, image_size=32,
            device="cpu", encoder_type="resnet",
        )
        summary = model.summary()
        self.assertIn("resnet", summary)


if __name__ == "__main__":
    unittest.main()
