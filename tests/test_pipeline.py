"""
Integration tests for the full NullFlow pipeline.
"""

import sys
import os
import unittest
import tempfile
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.models.latent_encoder import LatentEncoder
from nullflow.models.flow_matching import ConditionalFlowMatching
from nullflow.models.classifier import LatentClassifier
from nullflow.models.nullflow_model import NullFlowModel
from nullflow.core.null_space import NullSpaceProjector
from nullflow.core.knowledge_distillation import distillation_loss
from nullflow.metrics.cl_metrics import (
    average_accuracy, backward_transfer, forgetting_rate,
    compute_all_metrics, CLMetricsTracker
)
from nullflow.utils.reproducibility import set_seed


class TestLatentEncoder(unittest.TestCase):
    """Test VAE encoder."""

    def test_encode_decode_32x32(self):
        """Encode and decode 32x32 images."""
        enc = LatentEncoder(latent_dim=32, in_channels=3, image_size=32,
                            device="cpu")
        x = torch.randn(4, 3, 32, 32)
        z = enc.encode(x)
        self.assertEqual(z.shape, (4, 32))
        x_recon = enc.decode(z)
        self.assertEqual(x_recon.shape, (4, 3, 32, 32))

    def test_freeze(self):
        """Freeze should disable gradients."""
        enc = LatentEncoder(latent_dim=32, in_channels=3, image_size=32,
                            device="cpu")
        enc.freeze()
        for p in enc.parameters():
            self.assertFalse(p.requires_grad)

    def test_save_load(self):
        """Save and load encoder state."""
        enc = LatentEncoder(latent_dim=32, in_channels=3, image_size=32,
                            device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "encoder.pt")
            enc.save_pretrained(path)
            enc2 = LatentEncoder(latent_dim=32, in_channels=3, image_size=32,
                                 device="cpu")
            enc2.load_pretrained(path)
            for p1, p2 in zip(enc.parameters(), enc2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


class TestNullFlowModel(unittest.TestCase):
    """Test the unified NullFlow model."""

    def setUp(self):
        set_seed(42)
        self.model = NullFlowModel(
            latent_dim=32,
            flow_hidden_dim=64,
            num_classes_max=10,
            in_channels=3,
            image_size=32,
            device="cpu",
        )

    def test_encode(self):
        """Encode images to latents."""
        x = torch.randn(4, 3, 32, 32)
        z = self.model.encode(x)
        self.assertEqual(z.shape, (4, 32))

    def test_classify(self):
        """Classify latent vectors."""
        z = torch.randn(4, 32)
        logits = self.model.classify(z)
        self.assertEqual(logits.shape, (4, 10))

    def test_predict(self):
        """End-to-end prediction."""
        x = torch.randn(4, 3, 32, 32)
        preds = self.model.predict(x)
        self.assertEqual(preds.shape, (4,))

    def test_generate_replay(self):
        """Generate replay latents."""
        z = self.model.generate_replay(8, class_label=3)
        self.assertEqual(z.shape, (8, 32))

    def test_summary(self):
        """Model summary should return a string."""
        summary = self.model.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Total", summary)


class TestKnowledgeDistillation(unittest.TestCase):
    """Test KD loss functions."""

    def test_distillation_loss_shape(self):
        """KD loss should be a scalar."""
        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)
        loss = distillation_loss(student_logits, teacher_logits, temperature=2.0)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(loss.item() >= 0)

    def test_same_logits_zero_loss(self):
        """Identical logits should yield zero (or near-zero) KD loss."""
        logits = torch.randn(8, 10)
        loss = distillation_loss(logits, logits, temperature=2.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)


class TestCLMetrics(unittest.TestCase):
    """Test CL metric calculations."""

    def test_average_accuracy(self):
        """AA should be the mean of the last row."""
        R = np.array([
            [90.0, 0.0],
            [80.0, 85.0],
        ])
        aa = average_accuracy(R)
        self.assertAlmostEqual(aa, 82.5)

    def test_backward_transfer(self):
        """BWT should measure forgetting."""
        R = np.array([
            [90.0, 0.0],
            [70.0, 85.0],
        ])
        bwt = backward_transfer(R)
        # BWT = (R[1][0] - R[0][0]) = -20
        self.assertAlmostEqual(bwt, -20.0)

    def test_forgetting_rate(self):
        """Forgetting should measure max drop."""
        R = np.array([
            [95.0, 0.0, 0.0],
            [80.0, 90.0, 0.0],
            [70.0, 85.0, 88.0],
        ])
        forg = forgetting_rate(R)
        self.assertGreater(forg, 0)

    def test_compute_all_metrics(self):
        """compute_all_metrics should return a dict with known keys."""
        R = np.array([
            [90.0, 0.0, 0.0],
            [80.0, 85.0, 0.0],
            [75.0, 80.0, 82.0],
        ])
        metrics = compute_all_metrics(R)
        self.assertIn("AA", metrics)
        self.assertIn("BWT", metrics)
        self.assertIn("FR", metrics)

    def test_tracker(self):
        """CLMetricsTracker should record and report."""
        tracker = CLMetricsTracker()
        tracker.update(current_task=0, task_accuracies={0: 90.0})
        tracker.update(current_task=1, task_accuracies={0: 80.0, 1: 85.0})
        metrics = tracker.compute_final_metrics()
        self.assertIn("AA", metrics)
        self.assertAlmostEqual(metrics["AA"], 82.5)


class TestEndToEnd(unittest.TestCase):
    """End-to-end smoke test of a mini training loop."""

    def test_mini_training_loop(self):
        """Run a minimal 2-task continual learning loop."""
        set_seed(42)
        device = torch.device("cpu")

        model = NullFlowModel(
            latent_dim=16, flow_hidden_dim=32, num_classes_max=4,
            in_channels=3, image_size=32, device="cpu",
        )

        nsp_cls = NullSpaceProjector(rank=4, device="cpu")
        tracker = CLMetricsTracker()

        optimizer_cls = torch.optim.Adam(
            model.classifier.parameters(), lr=1e-3
        )
        optimizer_fm = torch.optim.Adam(
            model.flow_model.velocity_net.parameters(), lr=1e-3
        )

        for task_id in range(2):
            # Generate synthetic task data
            x = torch.randn(32, 3, 32, 32)
            y = torch.randint(task_id * 2, (task_id + 1) * 2, (32,))

            # Wake phase (simplified)
            z = model.encode(x)
            logits = model.classify(z)
            loss = nn.CrossEntropyLoss()(logits, y)
            optimizer_cls.zero_grad()
            loss.backward()
            if task_id > 0:
                nsp_cls.project_gradients(model.classifier)
            optimizer_cls.step()

            # NREM phase (simplified)
            z = model.encode(x)
            fm_loss = model.flow_model.compute_loss(z.detach(), y)
            optimizer_fm.zero_grad()
            fm_loss.backward()
            optimizer_fm.step()

            # Update NSPs with DataLoader of latent vectors
            z_all = model.encode(x).detach()
            ds = TensorDataset(z_all, y)
            loader = DataLoader(ds, batch_size=8)
            nsp_cls.update(model.classifier, loader, num_samples=16)

            # Evaluate
            task_accs = {}
            with torch.no_grad():
                for eval_task in range(task_id + 1):
                    x_eval = torch.randn(16, 3, 32, 32)
                    y_eval = torch.randint(eval_task * 2, (eval_task + 1) * 2,
                                           (16,))
                    preds = model.predict(x_eval)
                    acc = (preds == y_eval).float().mean().item() * 100
                    task_accs[eval_task] = acc

            tracker.update(current_task=task_id, task_accuracies=task_accs)

        metrics = tracker.compute_final_metrics()
        self.assertIn("AA", metrics)
        self.assertIsInstance(metrics["AA"], float)


if __name__ == "__main__":
    unittest.main()
