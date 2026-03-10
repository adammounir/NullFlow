#!/usr/bin/env python3
"""
NullFlow — Multi-Seed Runner for Statistical Reporting.

Runs NullFlow with multiple seeds and computes mean ± std.

Usage:
    python scripts/run_multiseed.py --config configs/split_cifar100.yaml
    python scripts/run_multiseed.py --config configs/split_cifar100.yaml --seeds 42 123 456
"""

import argparse
import json
import os
import sys
import subprocess
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed NullFlow runner")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--mode", type=str, default="task_aware")
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.config), "..", "results", "split_cifar100", "multiseed"
    )
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    all_aa, all_fr, all_bwt = [], [], []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        # Run training
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # Delete pretrained encoder to force fresh calibration per seed
        enc_path = os.path.join(seed_dir, "pretrained_encoder.pt")
        if os.path.exists(enc_path):
            os.remove(enc_path)

        cmd = [
            sys.executable, "scripts/train.py",
            "--config", args.config,
            "--mode", args.mode,
            "--seed", str(seed),
        ]

        env = os.environ.copy()
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        env["PYTHONUNBUFFERED"] = "1"

        log_path = os.path.join(seed_dir, f"train_seed{seed}.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 env=env, cwd=os.path.dirname(args.config) + "/..")
            if proc.returncode != 0:
                print(f"  ERROR: seed {seed} failed (returncode={proc.returncode})")
                continue

        # Read results
        results_path = os.path.join("results", "split_cifar100", "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                res = json.load(f)

            # Save per-seed results
            with open(os.path.join(seed_dir, "results.json"), "w") as f:
                json.dump(res, f, indent=2)

            metrics = res["metrics"]
            all_aa.append(metrics["AA"])
            all_fr.append(metrics["FR"])
            all_bwt.append(metrics["BWT"])
            all_results[seed] = metrics

            print(f"  Seed {seed}: AA={metrics['AA']:.2f}%, "
                  f"FR={metrics['FR']:.2f}%, BWT={metrics['BWT']:.2f}%")
        else:
            print(f"  WARNING: No results found for seed {seed}")

    # Summary
    if len(all_aa) > 0:
        summary = {
            "seeds": args.seeds[:len(all_aa)],
            "n_seeds": len(all_aa),
            "AA_mean": float(np.mean(all_aa)),
            "AA_std": float(np.std(all_aa)),
            "FR_mean": float(np.mean(all_fr)),
            "FR_std": float(np.std(all_fr)),
            "BWT_mean": float(np.mean(all_bwt)),
            "BWT_std": float(np.std(all_bwt)),
            "per_seed": all_results,
        }

        with open(os.path.join(output_dir, "multiseed_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("MULTI-SEED RESULTS")
        print(f"{'='*60}")
        print(f"  Seeds: {args.seeds[:len(all_aa)]}")
        print(f"  AA: {np.mean(all_aa):.2f} ± {np.std(all_aa):.2f}%")
        print(f"  FR: {np.mean(all_fr):.2f} ± {np.std(all_fr):.2f}%")
        print(f"  BWT: {np.mean(all_bwt):.2f} ± {np.std(all_bwt):.2f}%")
        print(f"{'='*60}")
        print(f"Results saved to {output_dir}/multiseed_summary.json")


if __name__ == "__main__":
    main()
