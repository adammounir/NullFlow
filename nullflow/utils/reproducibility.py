"""
Reproducibility utilities.

Ensures deterministic behavior across runs by fixing all random seeds
and configuring PyTorch for deterministic operations.
"""

import os
import logging
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.

    Fixes seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and all CUDA devices)
        - CUDA CuDNN

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Environment variable for some operations
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_deterministic(warn: bool = True):
    """
    Configure PyTorch for fully deterministic behavior.

    Note: This may reduce performance.

    Args:
        warn: Whether to print a warning about performance impact.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    if warn:
        logger.info("Deterministic mode enabled. This may reduce performance.")


def setup_reproducibility(seed: int = 42, deterministic: bool = True):
    """
    Full reproducibility setup: seeds + deterministic mode.

    Args:
        seed: Random seed.
        deterministic: Whether to enable deterministic algorithms.
    """
    set_seed(seed)
    if deterministic:
        ensure_deterministic(warn=False)


def get_device(preferred: str = "cuda") -> str:
    """
    Get the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        preferred: Preferred device ('cuda', 'mps', or 'cpu').

    Returns:
        Device string ('cuda', 'mps', or 'cpu').
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif preferred in ("cuda", "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device
