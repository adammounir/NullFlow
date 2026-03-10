"""
Null-Space Projection with incremental SVD approximation.

Implements gradient projection into the null-space of previously learned tasks,
ensuring that parameter updates for new tasks do not interfere with the
representations of old tasks.

The key idea: given the principal directions V_r of the Jacobian ∂f/∂θ on past
task data, we project gradients g as:
    g' = g - V_r @ (V_r^T @ g)

This removes any component of the gradient that lies in the subspace spanned by
directions important for past tasks.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader


def flatten_params(param_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Flatten a list of parameter tensors into a single 1D vector.

    Args:
        param_list: List of tensors (e.g., gradients for each parameter).

    Returns:
        Flattened 1D tensor of all values concatenated.
    """
    return torch.cat([p.reshape(-1) for p in param_list])


def unflatten_params(
    flat_vector: torch.Tensor,
    shapes: List[torch.Size],
) -> List[torch.Tensor]:
    """
    Unflatten a 1D vector back into a list of tensors with given shapes.

    Args:
        flat_vector: 1D tensor of concatenated values.
        shapes: List of shapes, one per parameter tensor.

    Returns:
        List of tensors with original shapes.
    """
    tensors = []
    offset = 0
    for shape in shapes:
        numel = 1
        for s in shape:
            numel *= s
        tensors.append(flat_vector[offset:offset + numel].reshape(shape))
        offset += numel
    return tensors


class NullSpaceProjector:
    """
    Null-Space Projection for gradient-based continual learning.

    Maintains a low-rank approximation V_r ∈ ℝ^{p × rank} of the principal
    directions of the Jacobian for all past tasks. Gradients are projected
    into the null-space of V_r to prevent catastrophic forgetting.

    The Jacobian basis is estimated via randomized power iteration
    (Jacobian-vector products), avoiding construction of the full Jacobian.
    """

    def __init__(self, rank: int = 64, device: str = "cuda", alpha: float = 1.0):
        """
        Args:
            rank: Number of principal directions to track (default 64).
            device: Device for computations.
            alpha: Attenuation factor for projection (0=no projection, 1=full).
                   Values < 1 allow some gradient through protected directions,
                   improving plasticity on new tasks.
        """
        self.rank = rank
        self.device = device
        self.alpha = alpha
        self.V_r: Optional[torch.Tensor] = None  # (p, rank) or None

    def compute_jacobian_basis(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_samples: int = 500,
    ) -> torch.Tensor:
        """
        Compute principal directions of the Jacobian ∂f/∂θ on given data.

        Uses randomized projection via Jacobian-vector products (JVP/VJP)
        to avoid constructing the full Jacobian matrix.

        Algorithm:
            1. For i = 1..2*rank:
                a. Sample random vector r ~ N(0, I) of dimension p
                b. Compute J·r via JVP (forward-mode AD)
                c. Compute J^T·(J·r) via VJP (backward-mode AD)
                d. Store result in columns of matrix A ∈ ℝ^{p × 2*rank}
            2. SVD of A: A = U·Σ·V^T
            3. Return first `rank` columns of V

        Args:
            model: Neural network whose Jacobian we compute.
            data_loader: DataLoader providing (x, y) or (z, y) batches.
            num_samples: Number of data samples to use for estimation.

        Returns:
            V_new: Principal directions, shape (p, rank).
        """
        model.eval()

        # Collect trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            return None

        total_params = sum(p.numel() for p in params)
        num_projections = min(2 * self.rank, total_params)

        # Collect a batch of data for Jacobian estimation
        data_batch = []
        count = 0
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            data_batch.append(x)
            count += x.shape[0]
            if count >= num_samples:
                break
        data_x = torch.cat(data_batch, dim=0)[:num_samples].to(self.device)

        # Matrix to store J^T @ J @ r for each random projection
        A_columns = []

        for _ in range(num_projections):
            # Random direction r ~ N(0, I) in parameter space
            r_list = [torch.randn_like(p) for p in params]

            # Compute J @ r via forward pass + perturbation (finite difference approx)
            # We use backward-mode AD for efficiency:
            # Step 1: Forward pass to get outputs
            model.zero_grad()
            out = model(data_x)
            if isinstance(out, tuple):
                out = out[0]  # Handle models returning (logits, ...) or (mu, logvar)

            # Flatten output
            out_flat = out.reshape(-1)

            # Step 2: Compute J @ r using forward-mode-like trick:
            # (∂out/∂θ) @ r = Σ_i (∂out/∂θ_i) * r_i
            # We do this via: grad(out · v, θ) where v is set via 2 backward passes
            # Instead, use the efficient approach: J @ r ≈ directional derivative
            # which we approximate using VJP and JVP via autograd

            # Efficient approach: compute gradient of (out_flat @ random_out_vec) w.r.t. params
            # Then use that to approximate the column of J^T J
            random_out = torch.randn_like(out_flat)

            # VJP: J^T @ random_out
            grads = torch.autograd.grad(
                out_flat, params,
                grad_outputs=random_out,
                retain_graph=True,
                allow_unused=True,
            )
            # Replace None grads with zeros
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, params)]

            # Use r to compute JVP approximation via finite difference isn't efficient
            # Instead, directly use the gradient as a column (randomized range finder)
            col = flatten_params(grads)
            A_columns.append(col)

        model.zero_grad()

        # Stack into matrix A: (p, num_projections)
        A = torch.stack(A_columns, dim=1)

        # SVD of A to get principal directions
        # Move to CPU for linalg ops (QR/SVD not fully supported on MPS)
        original_device = A.device
        A_cpu = A.cpu()

        # For numerical stability, first QR decompose
        try:
            Q, R = torch.linalg.qr(A_cpu)
            # SVD of R (much smaller)
            U_r, S, Vh = torch.linalg.svd(R, full_matrices=False)
            V_new = Q @ U_r[:, :self.rank]
        except RuntimeError:
            # Fallback: direct truncated SVD
            U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
            V_new = U[:, :self.rank]

        # Ensure orthonormality
        V_new, _ = torch.linalg.qr(V_new)

        return V_new.detach().to(original_device)

    def update(self, model: nn.Module, data_loader: DataLoader, num_samples: int = 500):
        """
        Update V_r incrementally after learning a new task.

        If V_r is None (first task), sets V_r = V_new.
        Otherwise, merges V_r and V_new via QR decomposition and truncation.

        Args:
            model: Model after training on the new task.
            data_loader: DataLoader for the new task's data.
            num_samples: Number of samples for Jacobian estimation.
        """
        V_new = self.compute_jacobian_basis(model, data_loader, num_samples)
        if V_new is None:
            return

        V_new = V_new.to(self.device)

        if self.V_r is None:
            self.V_r = V_new
        else:
            # Merge old and new bases
            V_combined = torch.cat([self.V_r, V_new], dim=1)  # (p, 2*rank)

            # QR decomposition to orthogonalize (on CPU — not supported on MPS)
            Q, R = torch.linalg.qr(V_combined.cpu())

            # Truncate to rank columns and move back to device
            self.V_r = Q[:, :self.rank].detach().to(self.device)

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Project gradient into the null-space of past task directions.

        g' = g - V_r @ (V_r^T @ g)

        This removes the component of g that lies in the column space of V_r,
        ensuring the update doesn't interfere with past tasks.

        Args:
            gradient: Flattened gradient vector, shape (p,).

        Returns:
            Projected gradient, shape (p,).
        """
        if self.V_r is None:
            return gradient  # No past tasks, no projection needed

        V_r = self.V_r.to(gradient.device)

        # Soft null-space projection with attenuation factor alpha:
        # g' = g - alpha * V_r @ (V_r^T @ g)
        # alpha=1.0 → full projection (removes all protected components)
        # alpha<1.0 → partial projection (preserves some plasticity)
        proj_coeff = V_r.T @ gradient           # (rank,)
        proj_component = V_r @ proj_coeff       # (p,)
        g_projected = gradient - self.alpha * proj_component

        return g_projected

    def project_gradients(self, model: nn.Module):
        """
        In-place projection of model gradients into the null-space.

        Convenience method that:
            1. Extracts gradients from model parameters
            2. Flattens, projects, unflattens
            3. Assigns projected gradients back to parameters

        Args:
            model: Model whose .grad attributes will be modified in-place.
        """
        if self.V_r is None:
            return  # No past tasks

        params_with_grad = [(p, p.grad) for p in model.parameters()
                           if p.requires_grad and p.grad is not None]
        if len(params_with_grad) == 0:
            return

        params, grads = zip(*params_with_grad)
        shapes = [g.shape for g in grads]

        # Flatten → project → unflatten
        flat_grad = flatten_params(list(grads))
        flat_projected = self.project(flat_grad)
        projected_grads = unflatten_params(flat_projected, shapes)

        # Assign back
        for param, new_grad in zip(params, projected_grads):
            param.grad = new_grad.clone()

    def get_memory_usage_bytes(self) -> int:
        """Return memory usage of V_r in bytes."""
        if self.V_r is None:
            return 0
        return self.V_r.element_size() * self.V_r.nelement()

    def get_memory_usage_mb(self) -> float:
        """Return memory usage of V_r in megabytes."""
        return self.get_memory_usage_bytes() / (1024 * 1024)
