#!/usr/bin/env python3
"""Full pipeline smoke test with ResNet encoder + Buffer + Heun solver."""

import torch
import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.models.nullflow_model import NullFlowModel
from nullflow.core.null_space import NullSpaceProjector
from nullflow.core.wake_phase import wake_epoch
from nullflow.core.sleep_nrem import sleep_nrem, encode_task_data
from nullflow.core.sleep_rem import sleep_rem
from nullflow.data.latent_buffer import LatentReplayBuffer
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = "cpu"
LATENT = 64
N_CLS = 10
IMG_SIZE = 32
N_SAMPLES = 80

print("=" * 60)
print("SMOKE TEST: ResNet encoder + Buffer + Heun")
print("=" * 60)

# 1. Create model with ResNet encoder
print("[1] Creating NullFlowModel (encoder_type=resnet)...")
model = NullFlowModel(
    latent_dim=LATENT, flow_hidden_dim=128, flow_num_layers=2,
    classifier_hidden_dim=64, num_classes_max=N_CLS,
    in_channels=3, image_size=IMG_SIZE, device=device,
    encoder_type="resnet",
)
print(model.summary())

nsp = NullSpaceProjector(rank=32, device=device)
buffer = LatentReplayBuffer(max_per_class=20, max_total=200)
clf_opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
fm_opt = torch.optim.Adam(model.flow_model.parameters(), lr=5e-4)

# 2. Calibrate encoder
print("[2] Calibrating ResNet encoder on Task-0...")
x0 = torch.randn(N_SAMPLES, 3, IMG_SIZE, IMG_SIZE)
y0 = torch.randint(0, 5, (N_SAMPLES,))
ds0 = TensorDataset(x0, y0)
loader0 = DataLoader(ds0, batch_size=16, shuffle=True)
model.encoder.calibrate(loader0, epochs=2, verbose=False)
model.encoder.freeze()
print("  Calibrated + frozen.")

# ========== TASK 0 ==========
print()
print("=" * 50)
print("TASK 0  (classes 0-4)")
print("=" * 50)

print("[3] Wake phase...")
for ep in range(2):
    loss, _, _ = wake_epoch(model, nsp, loader0, clf_opt, verbose=False)
print(f"  Wake loss: {loss:.4f}")

print("[4] Updating NSP...")
z0, y0_enc = encode_task_data(model, loader0, device)
lat_ds0 = TensorDataset(z0.to(device), y0_enc.to(device))
lat_loader0 = DataLoader(lat_ds0, batch_size=32)
nsp.update(model.classifier, lat_loader0, num_samples=50)
print(f"  V_r shape: {nsp.V_r.shape}")

print("[5] NREM (FM training)...")
losses = sleep_nrem(
    model.flow_model, (z0, y0_enc), nsp, fm_opt,
    n_epochs=3, batch_size=32, device=device, verbose=False,
    replay_data=buffer.get_all(),
)
print(f"  NREM final loss: {losses[-1]:.4f}")
buffer.add(z0, y0_enc)
print(f"  Buffer: {buffer}")

print("[6] REM (skip -- no past classes for Task 0)...")
old_clf = copy.deepcopy(model.classifier)
old_clf.eval()
rem_losses = sleep_rem(
    model.flow_model, model.classifier, old_clf, nsp,
    seen_classes=list(range(5)), new_classes=list(range(5)),
    optimizer=clf_opt, n_epochs=2, replay_per_class=10,
    num_ode_steps=4, ode_solver="heun", device=device, verbose=True,
)
print(f"  REM epochs: {len(rem_losses)} (expected 0 for first task)")

# ========== TASK 1 ==========
print()
print("=" * 50)
print("TASK 1  (classes 5-9)")
print("=" * 50)

x1 = torch.randn(N_SAMPLES, 3, IMG_SIZE, IMG_SIZE)
y1 = torch.randint(5, 10, (N_SAMPLES,))
ds1 = TensorDataset(x1, y1)
loader1 = DataLoader(ds1, batch_size=16, shuffle=True)
old_clf = copy.deepcopy(model.classifier)
old_clf.eval()

print("[7] Wake phase...")
for ep in range(2):
    loss, _, _ = wake_epoch(model, nsp, loader1, clf_opt, verbose=False)
print(f"  Wake loss: {loss:.4f}")

print("[8] Updating NSP...")
z1, y1_enc = encode_task_data(model, loader1, device)
lat_ds1 = TensorDataset(z1.to(device), y1_enc.to(device))
lat_loader1 = DataLoader(lat_ds1, batch_size=32)
nsp.update(model.classifier, lat_loader1, num_samples=50)
print(f"  V_r shape: {nsp.V_r.shape}")

print("[9] NREM (FM training WITH buffer replay)...")
replay_data = buffer.get_all()
print(f"  Buffer replay: {replay_data[0].shape[0]} exemplars")
losses = sleep_nrem(
    model.flow_model, (z1, y1_enc), nsp, fm_opt,
    n_epochs=3, batch_size=32, device=device, verbose=False,
    replay_data=replay_data,
)
print(f"  NREM final loss: {losses[-1]:.4f}")
buffer.add(z1, y1_enc)
print(f"  Buffer: {buffer}")

print("[10] REM (replay + KD, Heun solver)...")
rem_losses = sleep_rem(
    model.flow_model, model.classifier, old_clf, nsp,
    seen_classes=list(range(10)), new_classes=list(range(5, 10)),
    optimizer=clf_opt, n_epochs=2, replay_per_class=10,
    num_ode_steps=4, ode_solver="heun", device=device, verbose=False,
)
print(f"  REM final loss: {rem_losses[-1]:.4f}")

# ========== TASK 2 (extra task to test growing buffer) ==========
print()
print("=" * 50)
print("TASK 2 -- verifying growing buffer")
print("=" * 50)

x2 = torch.randn(N_SAMPLES, 3, IMG_SIZE, IMG_SIZE)
y2 = torch.randint(0, 5, (N_SAMPLES,))
ds2 = TensorDataset(x2, y2)
loader2 = DataLoader(ds2, batch_size=16, shuffle=True)
old_clf = copy.deepcopy(model.classifier)
old_clf.eval()

for ep in range(2):
    loss, _, _ = wake_epoch(model, nsp, loader2, clf_opt, verbose=False)
z2, y2_enc = encode_task_data(model, loader2, device)

replay_data = buffer.get_all()
print(f"  Buffer has {replay_data[0].shape[0]} exemplars from {len(buffer.get_classes())} classes")
losses = sleep_nrem(
    model.flow_model, (z2, y2_enc), nsp, fm_opt,
    n_epochs=2, batch_size=32, device=device, verbose=False,
    replay_data=replay_data,
)
print(f"  NREM final loss: {losses[-1]:.4f}")
buffer.add(z2, y2_enc)

rem_losses = sleep_rem(
    model.flow_model, model.classifier, old_clf, nsp,
    seen_classes=list(range(10)), new_classes=list(range(5)),
    optimizer=clf_opt, n_epochs=2, replay_per_class=10,
    num_ode_steps=4, ode_solver="heun", device=device, verbose=False,
)
print(f"  REM final loss: {rem_losses[-1]:.4f}")

# ========== EVALUATION ==========
print()
print("=" * 50)
print("EVALUATION")
print("=" * 50)
model.eval()
for task_id, ds in enumerate([ds0, ds1, ds2]):
    loader = DataLoader(ds, batch_size=32)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model.predict(x.to(device))
            correct += (pred == y.to(device)).sum().item()
            total += y.size(0)
    print(f"  Task {task_id} acc: {100*correct/total:.1f}% (random data)")

# Test generate_replay
print()
print("[11] Testing generate_replay (Heun)...")
gen_z_all = []
gen_y_all = []
for c in range(10):
    z_c = model.generate_replay(n=5, class_label=c, num_steps=4)
    gen_z_all.append(z_c)
    gen_y_all.append(torch.full((5,), c, dtype=torch.long))
gen_z = torch.cat(gen_z_all)
gen_y = torch.cat(gen_y_all)
print(f"  Generated: z={gen_z.shape}, y={gen_y.shape}")
assert gen_z.shape == (50, LATENT), f"Expected (50,{LATENT}), got {gen_z.shape}"
assert gen_y.shape == (50,)

print()
print("=" * 60)
print("ALL CHECKS PASSED -- Full pipeline works end-to-end!")
print("=" * 60)
