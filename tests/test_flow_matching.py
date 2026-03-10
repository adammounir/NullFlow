"""
Tests for ConditionalFlowMatching model.
"""

import sys
import os
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.models.flow_matching import VelocityNetwork, ConditionalFlowMatching


class TestVelocityNetwork(unittest.TestCase):
    """Test the velocity network architecture."""

    def setUp(self):
        self.latent_dim = 64
        self.hidden_dim = 128
        self.num_classes_max = 10
        self.net = VelocityNetwork(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_classes_max=self.num_classes_max,
        )

    def test_output_shape(self):
        """Output shape must match latent_dim."""
        batch = 16
        z = torch.randn(batch, self.latent_dim)
        t = torch.rand(batch)
        y = torch.randint(0, self.num_classes_max, (batch,))
        out = self.net(t, z, y)
        self.assertEqual(out.shape, (batch, self.latent_dim))

    def test_output_no_nan(self):
        """Output should not contain NaN."""
        z = torch.randn(4, self.latent_dim)
        t = torch.rand(4)
        y = torch.randint(0, self.num_classes_max, (4,))
        out = self.net(t, z, y)
        self.assertFalse(torch.isnan(out).any())

    def test_different_timesteps(self):
        """Different timesteps should produce different outputs."""
        z = torch.randn(1, self.latent_dim)
        y = torch.tensor([0])
        out_0 = self.net(torch.tensor([0.0]), z, y)
        out_1 = self.net(torch.tensor([1.0]), z, y)
        self.assertFalse(torch.allclose(out_0, out_1, atol=1e-5))


class TestConditionalFlowMatching(unittest.TestCase):
    """Test the conditional flow matching model."""

    def setUp(self):
        self.latent_dim = 64
        self.num_classes_max = 10
        self.fm = ConditionalFlowMatching(
            latent_dim=self.latent_dim,
            hidden_dim=128,
            num_classes_max=self.num_classes_max,
        )

    def test_compute_loss(self):
        """Loss should be a scalar > 0."""
        z = torch.randn(8, self.latent_dim)
        y = torch.randint(0, self.num_classes_max, (8,))
        loss = self.fm.compute_loss(z, y)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.item() > 0)

    def test_loss_backward(self):
        """Loss should be differentiable."""
        z = torch.randn(8, self.latent_dim)
        y = torch.randint(0, self.num_classes_max, (8,))
        loss = self.fm.compute_loss(z, y)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.fm.velocity_net.parameters()
        )
        self.assertTrue(has_grad)

    def test_sample_shape(self):
        """Sampled latents must have correct shape."""
        n = 12
        samples = self.fm.sample(n, class_label=3, device="cpu")
        self.assertEqual(samples.shape, (n, self.latent_dim))

    def test_sample_no_nan(self):
        """Samples should not contain NaN."""
        samples = self.fm.sample(4, class_label=0, device="cpu")
        self.assertFalse(torch.isnan(samples).any())

    def test_sample_different_classes(self):
        """Different class labels should produce different samples."""
        torch.manual_seed(0)
        s0 = self.fm.sample(8, class_label=0, device="cpu")
        torch.manual_seed(0)
        s1 = self.fm.sample(8, class_label=1, device="cpu")
        self.assertFalse(torch.allclose(s0, s1, atol=1e-4))

    def test_sample_batch(self):
        """sample_batch should return correct shape."""
        labels = torch.arange(self.num_classes_max).repeat_interleave(5)
        samples = self.fm.sample_batch(labels)
        self.assertEqual(samples.shape[0], self.num_classes_max * 5)
        self.assertEqual(samples.shape[1], self.latent_dim)


class TestFlowMatchingTraining(unittest.TestCase):
    """Test a short training loop to verify convergence signal."""

    def test_loss_decreases(self):
        """Loss should decrease over a few steps on simple data."""
        fm = ConditionalFlowMatching(
            latent_dim=16, hidden_dim=64, num_classes_max=2
        )
        optimizer = torch.optim.Adam(fm.velocity_net.parameters(), lr=1e-3)

        torch.manual_seed(42)
        z0 = torch.randn(32, 16) - 2
        z1 = torch.randn(32, 16) + 2
        z = torch.cat([z0, z1])
        y = torch.cat([torch.zeros(32), torch.ones(32)]).long()

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = fm.compute_loss(z, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
