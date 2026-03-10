#!/usr/bin/env python3
"""
NullFlow — Baseline comparison script (optimized with pre-encoding).

Runs all baselines on the same benchmark for fair comparison:
    1. Fine-tuning (Lower Bound)
    2. Joint Training (Upper Bound)
    3. EWC (Elastic Weight Consolidation)
    4. DER++ (Dark Experience Replay)
    5. GDumb (Greedy Sampler)
    6. Latent Replay (replay in latent space, no generative model)

All baselines operate in the SAME frozen latent space as NullFlow for
fair comparison. Data is pre-encoded once to avoid redundant encoder
forward passes.

Usage:
    python scripts/run_baselines.py --config configs/split_cifar100.yaml
"""

import argparse
import copy
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.utils.config import load_config
from nullflow.utils.reproducibility import setup_reproducibility, get_device
from nullflow.data.benchmarks import get_benchmark
from nullflow.models.resnet_encoder import ResNetEncoder
from nullflow.models.latent_encoder import LatentEncoder
from nullflow.models.classifier import LatentClassifier
from nullflow.metrics.cl_metrics import compute_all_metrics, CLMetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Run CL Baselines")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--baselines", type=str, nargs="+",
                       default=["fine_tune", "joint", "ewc", "der++",
                                "gdumb", "latent_replay", "icarl", "ncm"],
                       help="Which baselines to run")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (overrides config)")
    return parser.parse_args()


# =============================================================================
# Pre-encoding: encode ALL data once through frozen encoder
# =============================================================================

def get_encoder(config, device):
    """Load pretrained encoder."""
    encoder_type = config.get("encoder_type", "resnet")

    # Search for pretrained encoder in multiple locations
    base_dir = config.get("output_dir", "results/")
    enc_candidates = [
        os.path.join(base_dir, "pretrained_encoder.pt"),
    ]
    # Also try results/<benchmark>/<config_name>/
    results_root = "results/"
    benchmark = config.get("benchmark", "split_cifar100")
    bench_dir = os.path.join(results_root, benchmark)
    if os.path.isdir(bench_dir):
        for d in sorted(os.listdir(bench_dir)):
            p = os.path.join(bench_dir, d, "pretrained_encoder.pt")
            if os.path.exists(p):
                enc_candidates.append(p)

    if encoder_type == "resnet":
        encoder = ResNetEncoder(
            latent_dim=config.get("latent_dim", 512),
            in_channels=3,
            image_size=config.get("image_size", 32),
            device=device,
            use_raw_features=config.get("use_raw_features", False),
        ).to(device)
    else:
        encoder = LatentEncoder(
            latent_dim=config.get("latent_dim", 512),
            image_size=config.get("image_size", 32),
            device=device,
        ).to(device)

    # Load from first existing checkpoint
    enc_path = None
    for c in enc_candidates:
        if os.path.exists(c):
            enc_path = c
            break

    if enc_path is not None:
        encoder.load_pretrained(enc_path)
        print(f"  Loaded pretrained encoder from {enc_path}")
    else:
        print("  WARNING: No pretrained encoder found. Using raw ImageNet features.")

    encoder.freeze()
    return encoder


def pre_encode_all(encoder, benchmark, device):
    """Pre-encode ALL train/test data to latent space (one-time cost)."""
    print("  Pre-encoding all data to latent space...")
    train_latents, test_latents = {}, {}

    for tid, exp in enumerate(benchmark.get_train_stream()):
        loader = exp.get_dataloader(batch_size=256, shuffle=False, num_workers=0)
        all_z, all_y = [], []
        with torch.no_grad():
            for x, y in loader:
                z = encoder.encode(x.to(device))
                all_z.append(z.cpu())
                all_y.append(y)
        train_latents[tid] = (torch.cat(all_z), torch.cat(all_y))

    for tid, exp in enumerate(benchmark.get_test_stream()):
        loader = exp.get_dataloader(batch_size=256, shuffle=False, num_workers=0)
        all_z, all_y = [], []
        with torch.no_grad():
            for x, y in loader:
                z = encoder.encode(x.to(device))
                all_z.append(z.cpu())
                all_y.append(y)
        test_latents[tid] = (torch.cat(all_z), torch.cat(all_y))

    print(f"  Encoded {sum(len(v[0]) for v in train_latents.values())} train "
          f"+ {sum(len(v[0]) for v in test_latents.values())} test samples.")
    return train_latents, test_latents


def evaluate_latent(classifier, test_latents, device, num_tasks,
                    seen_classes=None):
    """Evaluate classifier on pre-encoded test data.

    Args:
        classifier: Trained LatentClassifier.
        test_latents: Dict mapping task_id → (z, y) tensors.
        device: Torch device.
        num_tasks: Number of tasks to evaluate (0..num_tasks-1).
        seen_classes: Optional list of class indices seen so far.
            If provided, logits for unseen classes are masked to -inf
            (consistent with NullFlow's evaluation protocol).
    """
    classifier.eval()
    accs = {}
    for tid in range(num_tasks):
        z, y = test_latents[tid]
        z, y = z.to(device), y.to(device)
        with torch.no_grad():
            logits = classifier(z)
            if seen_classes is not None:
                mask = torch.full_like(logits, float('-inf'))
                for c in seen_classes:
                    mask[:, c] = 0.0
                logits = logits + mask
            pred = logits.argmax(dim=1)
        correct = (pred == y).sum().item()
        accs[tid] = 100.0 * correct / max(len(y), 1)
    classifier.train()
    return accs


def make_classifier(config, device):
    """Create a fresh classifier with config-specified dimensions."""
    return LatentClassifier(
        latent_dim=config.get("latent_dim", 512),
        hidden_dim=config.get("classifier_hidden_dim", 512),
        num_classes=config.get("num_classes", 100),
        classifier_type=config.get("classifier_type", "mlp"),
    ).to(device)


# =============================================================================
# Baseline 1: Fine-tuning (Lower Bound)
# =============================================================================

def run_fine_tuning(config, device, train_latents, test_latents):
    """Sequential fine-tuning without any CL protection."""
    print("\n" + "="*60)
    print("BASELINE: Fine-tuning")
    print("="*60)

    classifier = make_classifier(config, device)
    optimizer = torch.optim.Adam(classifier.parameters(),
                                lr=config.get("wake_lr", 1e-3))

    tracker = CLMetricsTracker()
    n_tasks = len(train_latents)
    seen_classes = []
    # Match NullFlow's total budget: wake + REM epochs
    epochs = config.get("wake_epochs_per_task", 15) + config.get("rem_epochs", 25)

    for tid in range(n_tasks):
        z_train, y_train = train_latents[tid]
        seen_classes.extend(sorted(y_train.unique().tolist()))
        ds = TensorDataset(z_train, y_train)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        for epoch in range(epochs):
            for z, y in loader:
                z, y = z.to(device), y.to(device)
                loss = F.cross_entropy(classifier(z), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        accs = evaluate_latent(classifier, test_latents, device, tid + 1,
                               seen_classes=seen_classes)
        tracker.update(tid, accs)
        print(f"  Task {tid+1}: AA={tracker.get_current_aa():.1f}%")

    return tracker.get_accuracy_matrix_np()

def run_joint_training(config, device, train_latents, test_latents):
    """Train on all tasks simultaneously (upper bound)."""
    print("\n" + "="*60)
    print("BASELINE: Joint Training (Upper Bound)")
    print("="*60)

    classifier = make_classifier(config, device)
    lr = config.get("wake_lr", 1e-3)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr,
                                 weight_decay=1e-4)

    # Combine all training data
    all_z = torch.cat([train_latents[t][0] for t in train_latents])
    all_y = torch.cat([train_latents[t][1] for t in train_latents])
    ds = TensorDataset(all_z, all_y)
    loader = DataLoader(ds, batch_size=128, shuffle=True)

    # Fixed training schedule — no model selection on test data
    total_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(total_epochs):
        for z, y in loader:
            z, y = z.to(device), y.to(device)
            loss = F.cross_entropy(classifier(z), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                accs = evaluate_latent(classifier, test_latents, device,
                                       len(train_latents))
                aa = np.mean(list(accs.values()))
            print(f"  Epoch {epoch+1}/{total_epochs}: AA={aa:.1f}%")

    # Evaluate final model (no best-model selection to avoid test snooping)
    T = len(train_latents)
    R = np.zeros((T, T))
    accs = evaluate_latent(classifier, test_latents, device, T)
    for i in range(T):
        for j in range(T):
            R[i][j] = accs.get(j, 0)

    print(f"  Joint AA: {np.mean(list(accs.values())):.1f}%")
    return R


# =============================================================================
# Baseline 3: EWC (Elastic Weight Consolidation)
# =============================================================================

def run_ewc(config, device, train_latents, test_latents, ewc_lambda=5000):
    """EWC: regularize with Fisher Information Matrix."""
    print("\n" + "="*60)
    print("BASELINE: EWC")
    print("="*60)

    classifier = make_classifier(config, device)
    optimizer = torch.optim.Adam(classifier.parameters(),
                                lr=config.get("wake_lr", 1e-3))

    tracker = CLMetricsTracker()
    n_tasks = len(train_latents)
    fisher_params = []
    seen_classes = []

    for tid in range(n_tasks):
        z_train, y_train = train_latents[tid]
        seen_classes.extend(sorted(y_train.unique().tolist()))
        ds = TensorDataset(z_train, y_train)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        n_epochs = config.get("wake_epochs_per_task", 15) + config.get("rem_epochs", 25)
        for epoch in range(n_epochs):
            for z, y in loader:
                z, y = z.to(device), y.to(device)
                loss = F.cross_entropy(classifier(z), y)

                # EWC penalty
                ewc_loss = 0.0
                for fisher_d, params_d in fisher_params:
                    for name, param in classifier.named_parameters():
                        if name in fisher_d:
                            ewc_loss += (fisher_d[name] * (param - params_d[name])**2).sum()
                loss += (ewc_lambda / 2) * ewc_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute Fisher for this task
        fisher = {}
        params_copy = {}
        classifier.eval()
        for name, param in classifier.named_parameters():
            fisher[name] = torch.zeros_like(param)
            params_copy[name] = param.data.clone()

        for z, y in loader:
            z, y = z.to(device), y.to(device)
            logits = classifier(z)
            loss = F.cross_entropy(logits, y)
            classifier.zero_grad()
            loss.backward()
            for name, param in classifier.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        for name in fisher:
            fisher[name] /= len(loader)

        fisher_params.append((fisher, params_copy))
        classifier.train()

        accs = evaluate_latent(classifier, test_latents, device, tid + 1,
                               seen_classes=seen_classes)
        tracker.update(tid, accs)
        print(f"  Task {tid+1}: AA={tracker.get_current_aa():.1f}%")

    return tracker.get_accuracy_matrix_np()


# =============================================================================
# Baseline 4: DER++ (Dark Experience Replay)
# =============================================================================

def run_der_plus_plus(config, device, train_latents, test_latents,
                      buffer_size=None, alpha=0.5, beta=0.5):
    """DER++: buffer replay with logit distillation."""
    if buffer_size is None:
        buffer_size = config.get("buffer_per_class", 20) * config.get("num_classes", 100)
    print("\n" + "="*60)
    print(f"BASELINE: DER++ (buffer={buffer_size})")
    print("="*60)

    classifier = make_classifier(config, device)
    optimizer = torch.optim.Adam(classifier.parameters(),
                                lr=config.get("wake_lr", 1e-3))

    tracker = CLMetricsTracker()
    n_tasks = len(train_latents)
    seen_classes = []
    buffer_z, buffer_y, buffer_logits = [], [], []

    for tid in range(n_tasks):
        z_train, y_train = train_latents[tid]
        seen_classes.extend(sorted(y_train.unique().tolist()))
        ds = TensorDataset(z_train, y_train)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        n_epochs = config.get("wake_epochs_per_task", 15) + config.get("rem_epochs", 25)
        for epoch in range(n_epochs):
            for z, y in loader:
                z, y = z.to(device), y.to(device)
                logits = classifier(z)
                loss = F.cross_entropy(logits, y)

                if len(buffer_z) > 0:
                    buf_idx = np.random.choice(len(buffer_z),
                                              min(64, len(buffer_z)), replace=False)
                    bz = torch.stack([buffer_z[i] for i in buf_idx]).to(device)
                    by = torch.stack([buffer_y[i] for i in buf_idx]).to(device)
                    bl = torch.stack([buffer_logits[i] for i in buf_idx]).to(device)
                    buf_out = classifier(bz)
                    loss += alpha * F.cross_entropy(buf_out, by)
                    loss += beta * F.mse_loss(buf_out, bl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Reservoir sampling for buffer
                with torch.no_grad():
                    for i in range(z.size(0)):
                        if len(buffer_z) < buffer_size:
                            buffer_z.append(z[i].cpu())
                            buffer_y.append(y[i].cpu())
                            buffer_logits.append(logits[i].detach().cpu())
                        else:
                            j = np.random.randint(0, tid * len(ds) + i + 1)
                            if j < buffer_size:
                                buffer_z[j] = z[i].cpu()
                                buffer_y[j] = y[i].cpu()
                                buffer_logits[j] = logits[i].detach().cpu()

        accs = evaluate_latent(classifier, test_latents, device, tid + 1,
                               seen_classes=seen_classes)
        tracker.update(tid, accs)
        print(f"  Task {tid+1}: AA={tracker.get_current_aa():.1f}%")

    return tracker.get_accuracy_matrix_np()


# =============================================================================
# Baseline 5: GDumb
# =============================================================================

def run_gdumb(config, device, train_latents, test_latents, buffer_size=None):
    """GDumb: store samples greedily, retrain from scratch each time."""
    if buffer_size is None:
        buffer_size = config.get("buffer_per_class", 20) * config.get("num_classes", 100)
    print("\n" + "="*60)
    print(f"BASELINE: GDumb (buffer={buffer_size})")
    print("="*60)

    tracker = CLMetricsTracker()
    n_tasks = len(train_latents)
    buffer_z, buffer_y = [], []
    seen_classes = []

    for tid in range(n_tasks):
        z_train, y_train = train_latents[tid]
        seen_classes.extend(sorted(y_train.unique().tolist()))

        # Add to buffer (balanced greedy)
        for i in range(len(z_train)):
            if len(buffer_z) < buffer_size:
                buffer_z.append(z_train[i])
                buffer_y.append(y_train[i])
            else:
                j = np.random.randint(0, len(buffer_z))
                buffer_z[j] = z_train[i]
                buffer_y[j] = y_train[i]

        # Retrain from scratch
        classifier = make_classifier(config, device)
        opt = torch.optim.Adam(classifier.parameters(),
                              lr=config.get("wake_lr", 1e-3))

        buf_ds = TensorDataset(torch.stack(buffer_z), torch.stack(buffer_y))
        buf_loader = DataLoader(buf_ds, batch_size=64, shuffle=True)

        epochs = config.get("wake_epochs_per_task", 15) * 3
        for _ in range(epochs):
            for bz, by in buf_loader:
                bz, by = bz.to(device), by.to(device)
                loss = F.cross_entropy(classifier(bz), by)
                opt.zero_grad()
                loss.backward()
                opt.step()

        accs = evaluate_latent(classifier, test_latents, device, tid + 1,
                               seen_classes=seen_classes)
        tracker.update(tid, accs)
        print(f"  Task {tid+1}: AA={tracker.get_current_aa():.1f}%")

    return tracker.get_accuracy_matrix_np()


# =============================================================================
# Baseline 6: Latent Replay (buffer in latent space, no generative model)
# =============================================================================

def run_latent_replay(config, device, train_latents, test_latents,
                      buffer_size=None):
    """Latent replay with fixed-size buffer, no generative model."""
    if buffer_size is None:
        buffer_size = config.get("buffer_per_class", 20) * config.get("num_classes", 100)
    print("\n" + "="*60)
    print(f"BASELINE: Latent Replay (buffer={buffer_size})")
    print("="*60)

    classifier = make_classifier(config, device)
    optimizer = torch.optim.Adam(classifier.parameters(),
                                lr=config.get("wake_lr", 1e-3))

    tracker = CLMetricsTracker()
    n_tasks = len(train_latents)
    seen_classes = []
    buffer_z, buffer_y = [], []

    for tid in range(n_tasks):
        z_train, y_train = train_latents[tid]
        seen_classes.extend(sorted(y_train.unique().tolist()))
        ds = TensorDataset(z_train, y_train)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        n_epochs = config.get("wake_epochs_per_task", 15) + config.get("rem_epochs", 25)
        for epoch in range(n_epochs):
            for z, y in loader:
                z, y = z.to(device), y.to(device)
                loss = F.cross_entropy(classifier(z), y)

                if len(buffer_z) > 0:
                    idx = np.random.choice(len(buffer_z),
                                          min(64, len(buffer_z)), replace=False)
                    bz = torch.stack([buffer_z[i] for i in idx]).to(device)
                    by = torch.stack([buffer_y[i] for i in idx]).to(device)
                    loss += F.cross_entropy(classifier(bz), by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update buffer (reservoir sampling)
        for i in range(len(z_train)):
            if len(buffer_z) < buffer_size:
                buffer_z.append(z_train[i])
                buffer_y.append(y_train[i])
            else:
                j = np.random.randint(0, len(buffer_z))
                buffer_z[j] = z_train[i]
                buffer_y[j] = y_train[i]

        accs = evaluate_latent(classifier, test_latents, device, tid + 1,
                               seen_classes=seen_classes)
        tracker.update(tid, accs)
        print(f"  Task {tid+1}: AA={tracker.get_current_aa():.1f}%")

    return tracker.get_accuracy_matrix_np()


# =============================================================================
# Baseline 7: iCaRL (Incremental Classifier and Representation Learning)
# =============================================================================

def run_icarl(config, device, train_latents, test_latents):
    """iCaRL: CE + sigmoid KD training, herding exemplars, NCM eval."""
    print("\n" + "="*60)
    print("BASELINE: iCaRL (NCM + Herding + KD)")
    print("="*60)

    buffer_per_class = config.get("buffer_per_class", 20)
    classifier = make_classifier(config, device)
    optimizer = torch.optim.Adam(classifier.parameters(),
                                lr=config.get("wake_lr", 1e-3))

    tracker = CLMetricsTracker()
    n_tasks = len(train_latents)
    latent_dim = config.get("latent_dim", 512)

    class_exemplars = {}  # {class_id: Tensor[K, latent_dim]}
    seen_classes = []

    for tid in range(n_tasks):
        z_train, y_train = train_latents[tid]
        new_classes = sorted(y_train.unique().tolist())

        # Save old model for KD
        old_classifier = copy.deepcopy(classifier) if tid > 0 else None

        # Build training set: new data + exemplar buffer
        all_z = [z_train]
        all_y = [y_train]
        for cls_id, exs in class_exemplars.items():
            all_z.append(exs)
            all_y.append(torch.full((len(exs),), cls_id, dtype=torch.long))

        combined_z = torch.cat(all_z)
        combined_y = torch.cat(all_y)
        ds = TensorDataset(combined_z, combined_y)
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        n_epochs = config.get("wake_epochs_per_task", 15) + config.get("rem_epochs", 25)
        for epoch in range(n_epochs):
            for z, y in loader:
                z, y = z.to(device), y.to(device)
                logits = classifier(z)
                loss = F.cross_entropy(logits, y)

                # Sigmoid-based KD on old classes (iCaRL-style)
                if old_classifier is not None and len(seen_classes) > 0:
                    with torch.no_grad():
                        old_logits = old_classifier(z)
                    old_probs = torch.sigmoid(old_logits[:, seen_classes])
                    new_probs = torch.sigmoid(logits[:, seen_classes])
                    kd_loss = F.binary_cross_entropy(new_probs, old_probs)
                    loss += kd_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update seen classes
        seen_classes.extend(new_classes)

        # Herding: select exemplars for new classes
        for cls_id in new_classes:
            cls_mask = y_train == cls_id
            cls_z = z_train[cls_mask]
            K = min(buffer_per_class, len(cls_z))

            cls_mean = cls_z.mean(dim=0)
            selected_indices = []
            running_sum = torch.zeros(latent_dim)

            remaining = list(range(len(cls_z)))
            for _ in range(K):
                best_idx = -1
                best_dist = float('inf')
                for r in remaining:
                    candidate_mean = (running_sum + cls_z[r]) / (len(selected_indices) + 1)
                    dist = (cls_mean - candidate_mean).pow(2).sum().item()
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = r
                selected_indices.append(best_idx)
                running_sum += cls_z[best_idx]
                remaining.remove(best_idx)

            class_exemplars[cls_id] = cls_z[selected_indices]

        # Evaluate using NCM (Nearest Class Mean)
        class_means = {}
        for cls_id, exs in class_exemplars.items():
            class_means[cls_id] = exs.mean(dim=0).to(device)

        accs = {}
        for eval_tid in range(tid + 1):
            z_test, y_test = test_latents[eval_tid]
            z_test, y_test = z_test.to(device), y_test.to(device)

            class_ids = sorted(class_means.keys())
            mean_stack = torch.stack([class_means[c] for c in class_ids])

            # L2 distances to class means
            dists = torch.cdist(z_test, mean_stack)
            pred_idx = dists.argmin(dim=1)
            pred = torch.tensor([class_ids[i] for i in pred_idx], device=device)

            correct = (pred == y_test).sum().item()
            accs[eval_tid] = 100.0 * correct / max(len(y_test), 1)

        tracker.update(tid, accs)
        print(f"  Task {tid+1}: AA={tracker.get_current_aa():.1f}%")

    return tracker.get_accuracy_matrix_np()


# =============================================================================
# Baseline 8: NCM / SimpleCIL (Nearest Class Mean — no buffer limit)
# =============================================================================

def run_ncm(config, device, train_latents, test_latents):
    """NCM/SimpleCIL: classify by nearest class mean using ALL training data.
    No training needed, no buffer limit. Strong frozen-feature baseline."""
    print("\n" + "="*60)
    print("BASELINE: NCM / SimpleCIL (all data, no buffer)")
    print("="*60)

    tracker = CLMetricsTracker()
    n_tasks = len(train_latents)

    class_means = {}  # {class_id: mean_vector}

    for tid in range(n_tasks):
        z_train, y_train = train_latents[tid]

        # Compute mean for each new class (using ALL training data)
        for cls_id in y_train.unique().tolist():
            cls_mask = y_train == cls_id
            class_means[cls_id] = z_train[cls_mask].mean(dim=0).to(device)

        # Evaluate using NCM
        accs = {}
        class_ids = sorted(class_means.keys())
        mean_stack = torch.stack([class_means[c] for c in class_ids])

        for eval_tid in range(tid + 1):
            z_test, y_test = test_latents[eval_tid]
            z_test, y_test = z_test.to(device), y_test.to(device)

            dists = torch.cdist(z_test, mean_stack)
            pred_idx = dists.argmin(dim=1)
            pred = torch.tensor([class_ids[i] for i in pred_idx], device=device)

            correct = (pred == y_test).sum().item()
            accs[eval_tid] = 100.0 * correct / max(len(y_test), 1)

        tracker.update(tid, accs)
        print(f"  Task {tid+1}: AA={tracker.get_current_aa():.1f}%")

    return tracker.get_accuracy_matrix_np()


# =============================================================================
# Baseline registry
# =============================================================================

BASELINE_RUNNERS = {
    "fine_tune": run_fine_tuning,
    "joint": run_joint_training,
    "ewc": run_ewc,
    "der++": run_der_plus_plus,
    "gdumb": run_gdumb,
    "latent_replay": run_latent_replay,
    "icarl": run_icarl,
    "ncm": run_ncm,
}


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    device = get_device(config.get("device", "cuda"))
    config["device"] = device
    seed = config.get("seed", 42)
    setup_reproducibility(seed)

    output_dir = args.output_dir or os.path.join(
        config.get("output_dir", "results/"), "baselines"
    )
    os.makedirs(output_dir, exist_ok=True)

    benchmark = get_benchmark(config)
    config["num_classes"] = benchmark.num_classes

    print(f"\nBenchmark: {config.get('benchmark', 'unknown')}")
    print(f"  Classes: {benchmark.num_classes}, Tasks: {benchmark.num_tasks}")
    print(f"  Buffer: {config.get('buffer_per_class', 20)}/class = "
          f"{config.get('buffer_per_class', 20) * benchmark.num_classes} total")
    print(f"  Seed: {seed}")

    # Pre-encode ALL data once (massive speedup)
    encoder = get_encoder(config, device)
    train_latents, test_latents = pre_encode_all(encoder, benchmark, device)
    del encoder  # Free encoder memory

    all_results = {}

    for baseline_name in args.baselines:
        if baseline_name not in BASELINE_RUNNERS:
            print(f"Unknown baseline: {baseline_name}, skipping.")
            continue

        setup_reproducibility(config.get("seed", 42))
        runner = BASELINE_RUNNERS[baseline_name]

        start = time.time()
        R = runner(config, device, train_latents, test_latents)
        elapsed = time.time() - start

        metrics = compute_all_metrics(R)
        metrics["time_seconds"] = elapsed

        all_results[baseline_name] = {
            "accuracy_matrix": R.tolist(),
            "metrics": metrics,
        }

        print(f"\n  {baseline_name}: AA={metrics['AA']:.1f}%, "
              f"BWT={metrics['BWT']:.1f}%, FR={metrics['FR']:.1f}%, "
              f"Time={elapsed:.0f}s")

    # Save all results (merge with existing if any)
    result_file = os.path.join(output_dir, f"baseline_results_seed{seed}.json")
    existing = {}
    if os.path.exists(result_file):
        with open(result_file) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(result_file, "w") as f:
        json.dump(existing, f, indent=2)
    all_results = existing

    print("\n" + "="*60)
    print(f"ALL BASELINE RESULTS (seed={seed})")
    print("="*60)
    print(f"{'Method':<20} {'AA':>8} {'BWT':>8} {'FR':>8} {'Time':>8}")
    print("-" * 52)
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"{name:<20} {m['AA']:>7.1f}% {m['BWT']:>7.1f}% "
              f"{m['FR']:>7.1f}% {m['time_seconds']:>7.0f}s")
    print("="*60)
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
