#!/usr/bin/env python3
"""
NullFlow — Post-training Evaluation Script.

Usage:
    python scripts/evaluate.py --results_dir results/split_cifar100/ --config configs/split_cifar100.yaml

Evaluates a trained NullFlow model and computes all metrics.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.utils.config import load_config
from nullflow.utils.reproducibility import setup_reproducibility, get_device
from nullflow.metrics.cl_metrics import (
    compute_all_metrics,
    compute_latent_fid,
    measure_inference_time,
)
from nullflow.data.benchmarks import get_benchmark
from nullflow.models.nullflow_model import NullFlowModel


def parse_args():
    parser = argparse.ArgumentParser(description="NullFlow Evaluation")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing trained model and results")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config.get("device", "cuda"))
    config["device"] = device
    setup_reproducibility(config.get("seed", 42))

    # Load accuracy matrix
    matrix_path = os.path.join(args.results_dir, "accuracy_matrix.json")
    if os.path.exists(matrix_path):
        with open(matrix_path, "r") as f:
            R_list = json.load(f)
        R = np.array(R_list)

        print("=" * 60)
        print("ACCURACY MATRIX R[i][j]")
        print("=" * 60)
        print("(Row i = after learning task i, Col j = accuracy on task j)")
        print()
        T = R.shape[0]
        header = "      " + "".join([f"Task {j:2d}  " for j in range(T)])
        print(header)
        for i in range(T):
            row = f"T={i:2d}  " + "".join([f"{R[i,j]:6.1f}  " for j in range(T)])
            print(row)
        print()

        # Compute metrics
        metrics = compute_all_metrics(R)
        print("=" * 60)
        print("METRICS")
        print("=" * 60)
        print(f"  Average Accuracy (AA):    {metrics['AA']:.2f}%")
        print(f"  Backward Transfer (BWT):  {metrics['BWT']:.2f}%")
        print(f"  Forward Transfer (FWT):   {metrics['FWT']:.2f}%")
        print(f"  Forgetting Rate (FR):     {metrics['FR']:.2f}%")
        print("=" * 60)
    else:
        print(f"No accuracy matrix found at {matrix_path}")

    # Load model and measure inference time
    model_path = os.path.join(args.results_dir, "model_final.pt")
    if os.path.exists(model_path):
        print("\nMeasuring inference time...")
        model = NullFlowModel(
            latent_dim=config.get("latent_dim", 128),
            flow_hidden_dim=config.get("flow_hidden_dim", 512),
            flow_num_layers=config.get("flow_num_layers", 3),
            num_classes_max=config.get("num_classes_max", 200),
            image_size=config.get("image_size", 32),
            device=device,
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.flow_model.load_state_dict(checkpoint["flow_model"])

        for steps in [1, 2, 4, 8, 16, 32]:
            t = measure_inference_time(
                model.flow_model, n_samples=1000, num_steps=steps,
                latent_dim=config.get("latent_dim", 128), device=device,
            )
            print(f"  FM ({steps} steps): {t*1000:.1f}ms for 1000 samples")

    # Load and display overall results
    results_path = os.path.join(args.results_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        print(f"\nTraining time: {results.get('training_time_seconds', 0):.1f}s")
        print(f"Mode: {results.get('mode', 'unknown')}")


if __name__ == "__main__":
    main()
