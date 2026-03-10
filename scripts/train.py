#!/usr/bin/env python3
"""
NullFlow — Main Training Script.

Usage:
    python scripts/train.py --config configs/split_cifar100.yaml --mode task_aware
    python scripts/train.py --config configs/split_cifar100.yaml --mode task_free
    python scripts/train.py --config configs/split_tinyimagenet.yaml --mode task_aware

The script:
    1. Loads the YAML configuration
    2. Fixes all random seeds for reproducibility
    3. Initializes the benchmark and NullFlow strategy
    4. Pre-trains / calibrates the encoder on the first task only
    5. Runs training (task-aware or task-free)
    6. Evaluates and saves results (metrics, accuracy matrix, model)
    7. Logs to CSV and TensorBoard
"""

import argparse
import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nullflow.utils.config import load_config, save_config, config_to_str
from nullflow.utils.reproducibility import setup_reproducibility, get_device
from nullflow.utils.logging_utils import setup_logger, CSVLogger
from nullflow.data.benchmarks import get_benchmark
from nullflow.strategies.nullflow_strategy import NullFlowStrategy


def parse_args():
    parser = argparse.ArgumentParser(
        description="NullFlow: Task-Free Continual Learning"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/split_cifar100.yaml)",
    )
    parser.add_argument(
        "--mode", type=str, default="task_aware",
        choices=["task_aware", "task_free", "joint_retrain"],
        help="Training mode: 'task_aware', 'task_free', or 'joint_retrain'",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--encoder_epochs", type=int, default=30,
        help="Number of encoder pretraining / calibration epochs",
    )
    parser.add_argument(
        "--no_encoder_pretrain", action="store_true",
        help="Skip encoder pretraining (load from checkpoint)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ===== 1. Load Configuration =====
    config = load_config(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    # Ensure output directory exists
    output_dir = config.get("output_dir", "results/")
    os.makedirs(output_dir, exist_ok=True)

    # ===== 2. Setup Reproducibility =====
    seed = config.get("seed", 42)
    setup_reproducibility(seed, deterministic=True)

    # ===== 3. Setup Device =====
    device = get_device(config.get("device", "cuda"))
    config["device"] = device

    # macOS + MPS: force num_workers=0 to avoid fork() deadlocks
    import platform
    if platform.system() == "Darwin" or device == "mps":
        config["num_workers"] = 0

    # ===== 4. Setup Logging =====
    logger = setup_logger(
        "nullflow",
        log_file=os.path.join(output_dir, "train.log"),
    )
    logger.info("=" * 70)
    logger.info("NullFlow Training")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info("-" * 70)
    logger.info("Configuration:")
    logger.info(config_to_str(config))
    logger.info("-" * 70)

    # Save config for reproducibility
    save_config(config, os.path.join(output_dir, "config.yaml"))

    # ===== 5. Initialize Benchmark =====
    logger.info("Loading benchmark...")
    benchmark = get_benchmark(config)
    logger.info(f"  Benchmark: {config.get('benchmark', 'unknown')}")
    logger.info(f"  Tasks: {benchmark.num_tasks}")
    logger.info(f"  Classes per task: {benchmark.classes_per_task}")
    logger.info(f"  Total classes: {benchmark.num_classes}")
    logger.info(f"  Class order: {benchmark.class_order[:20]}...")

    # ===== 6. Initialize Strategy =====
    logger.info("Initializing NullFlow strategy...")
    strategy = NullFlowStrategy(config)

    # ===== 7. Pre-train / Calibrate Encoder =====
    if not args.no_encoder_pretrain:
        strategy.pretrain_encoder(benchmark, epochs=args.encoder_epochs)

    # ===== 8. Train =====
    logger.info(f"\nStarting {args.mode} training...")
    start_time = time.time()

    if args.mode == "task_aware":
        results = strategy.train_task_aware(benchmark)
    elif args.mode == "joint_retrain":
        results = strategy.train_joint_retrain(benchmark)
    else:
        results = strategy.train_task_free(benchmark)

    elapsed = time.time() - start_time
    logger.info(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # ===== 9. Save Results =====
    results["training_time_seconds"] = elapsed
    results["mode"] = args.mode
    strategy.save_results(results, output_dir)

    # Save full results JSON
    results_json = {
        "metrics": results["metrics"],
        "accuracy_matrix": results["accuracy_matrix"],
        "training_time_seconds": elapsed,
        "mode": args.mode,
        "config_file": args.config,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # ===== 10. Final Summary =====
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    metrics = results["metrics"]
    print(f"  Average Accuracy (AA):    {metrics.get('AA', 0):.2f}%")
    print(f"  Backward Transfer (BWT):  {metrics.get('BWT', 0):.2f}%")
    print(f"  Forward Transfer (FWT):   {metrics.get('FWT', 0):.2f}%")
    print(f"  Forgetting Rate (FR):     {metrics.get('FR', 0):.2f}%")
    print(f"  Total Time:               {elapsed:.1f}s")
    print(f"  Results saved to:         {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
