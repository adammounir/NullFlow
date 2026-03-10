"""
Tests for Null-Space Projection.
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.core.null_space import NullSpaceProjector, flatten_params, unflatten_params


class SimpleModel(nn.Module):
    """Small model for testing NSP."""
    def __init__(self, in_dim=16, hidden=32, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def make_loader(in_dim=16, n=32, num_classes=4, batch_size=8):
    """Create a simple DataLoader for testing."""
    x = torch.randn(n, in_dim)
    y = torch.randint(0, num_classes, (n,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


class TestFlattenUnflatten(unittest.TestCase):
    """Test flatten/unflatten utility functions."""

    def test_roundtrip(self):
        """Flatten and unflatten should be inverse operations."""
        tensors = [torch.randn(3, 4), torch.randn(5), torch.randn(2, 2, 2)]
        shapes = [t.shape for t in tensors]
        flat = flatten_params(tensors)

        total = sum(t.numel() for t in tensors)
        self.assertEqual(flat.shape, (total,))

        restored = unflatten_params(flat, shapes)
        for orig, rest in zip(tensors, restored):
            self.assertTrue(torch.allclose(orig, rest))


class TestNullSpaceProjector(unittest.TestCase):
    """Test NSP functionality."""

    def setUp(self):
        self.model = SimpleModel()
        self.nsp = NullSpaceProjector(rank=8, device="cpu")

    def test_compute_jacobian_basis(self):
        """Jacobian basis should have correct shape."""
        loader = make_loader()
        basis = self.nsp.compute_jacobian_basis(
            self.model, loader, num_samples=16
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(basis.shape[0], n_params)
        self.assertLessEqual(basis.shape[1], self.nsp.rank)

    def test_update_stores_basis(self):
        """After update, projector should have a basis V_r."""
        loader = make_loader()
        self.nsp.update(self.model, loader, num_samples=16)
        self.assertIsNotNone(self.nsp.V_r)
        self.assertGreater(self.nsp.V_r.shape[1], 0)

    def test_project_reduces_component(self):
        """Projection should reduce the component along the basis."""
        loader = make_loader()
        self.nsp.update(self.model, loader, num_samples=16)

        n_params = sum(p.numel() for p in self.model.parameters())
        g = torch.randn(n_params)
        g_proj = self.nsp.project(g)

        # The projected gradient should have a smaller component along V_r
        V_r = self.nsp.V_r
        comp_before = torch.norm(V_r.T @ g)
        comp_after = torch.norm(V_r.T @ g_proj)
        self.assertLessEqual(comp_after.item(), comp_before.item() + 1e-5)

    def test_project_gradients_inplace(self):
        """project_gradients should modify model gradients in-place."""
        loader = make_loader()
        self.nsp.update(self.model, loader, num_samples=16)

        # Compute gradients
        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))
        out = self.model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()

        # Store original grads
        orig_grads = [p.grad.clone() for p in self.model.parameters()
                      if p.grad is not None]

        # Project
        self.nsp.project_gradients(self.model)

        # Check grads changed
        new_grads = [p.grad.clone() for p in self.model.parameters()
                     if p.grad is not None]

        any_changed = False
        for og, ng in zip(orig_grads, new_grads):
            if not torch.allclose(og, ng, atol=1e-6):
                any_changed = True
                break
        self.assertTrue(any_changed)

    def test_incremental_update(self):
        """Multiple updates should accumulate knowledge."""
        loader1 = make_loader()
        self.nsp.update(self.model, loader1, num_samples=16)
        rank_after_1 = self.nsp.V_r.shape[1]

        loader2 = make_loader()
        self.nsp.update(self.model, loader2, num_samples=16)
        rank_after_2 = self.nsp.V_r.shape[1]

        # Rank should stay at max (rank truncation)
        self.assertLessEqual(rank_after_2, self.nsp.rank)

    def test_no_basis_project_identity(self):
        """With no basis, projection should return gradient unchanged."""
        g = torch.randn(100)
        g_proj = self.nsp.project(g)
        self.assertTrue(torch.allclose(g, g_proj))

    def test_orthonormality(self):
        """V_r should be approximately orthonormal."""
        loader = make_loader()
        self.nsp.update(self.model, loader, num_samples=16)

        V = self.nsp.V_r
        VtV = V.T @ V
        I = torch.eye(VtV.shape[0])
        self.assertTrue(torch.allclose(VtV, I, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
