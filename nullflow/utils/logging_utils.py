"""
Logging utilities for structured experiment tracking.

Provides:
    - Python logger setup with file and console handlers
    - CSV logger for metric curves
    - TensorBoard integration
"""

import os
import csv
import logging
import sys
from typing import Dict, Any, Optional


def setup_logger(
    name: str = "nullflow",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Path to log file (optional).
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "nullflow") -> logging.Logger:
    """
    Get or create a logger by name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class CSVLogger:
    """
    Simple CSV logger for recording metrics over time.

    Usage:
        logger = CSVLogger("results/metrics.csv", ["epoch", "loss", "accuracy"])
        logger.log({"epoch": 1, "loss": 0.5, "accuracy": 80.0})
        logger.close()
    """

    def __init__(self, filepath: str, fieldnames: list):
        """
        Args:
            filepath: Path to the CSV file.
            fieldnames: List of column names.
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: Dict[str, Any]):
        """
        Write a row of metrics.

        Args:
            row: Dictionary mapping column names to values.
        """
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the CSV file."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TensorBoardLogger:
    """
    Wrapper around TensorBoard SummaryWriter for experiment logging.
    """

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory for TensorBoard logs.
        """
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under one main tag."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image, step: int):
        """Log an image."""
        self.writer.add_image(tag, image, step)

    def close(self):
        """Close the writer."""
        self.writer.close()
