"""Utility functions: seeding, logging, device helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Configure logging to console and a file in output_dir.

    Returns the root logger for the geniesae package.
    """
    logger = logging.getLogger("geniesae")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_device_info(device: str) -> dict:
    """Return device information. For CUDA devices, includes GPU name, memory, and CUDA version."""
    info: dict = {"device": device}

    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            idx = int(device.split(":")[1])
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["total_memory_gb"] = round(
            torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2
        )
        info["cuda_version"] = torch.version.cuda or "N/A"
    else:
        info["gpu_name"] = "N/A"
        info["total_memory_gb"] = 0
        info["cuda_version"] = "N/A"

    return info
