"""Experiment configuration dataclass with YAML I/O and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


_DEVICE_PATTERN = re.compile(r"^(cpu|cuda(:\d+)?)$")


@dataclass
class ExperimentConfig:
    """Full experiment configuration for the SAE pipeline."""

    # Model checkpoint
    model_checkpoint_path: str = ""

    # GENIE model architecture parameters
    model_arch: str = "s2s_CAT"  # "transformer" or "s2s_CAT"
    in_channel: int = 128
    model_channels: int = 128
    out_channel: int = 128
    vocab_size: int = 30522
    config_name: str = "bert-base-uncased"
    logits_mode: int = 1
    init_pretrained: bool = False  # False when loading from checkpoint
    token_emb_type: str = "random"  # "random" when loading from checkpoint
    learn_sigma: bool = False
    fix_encoder: bool = False

    # Dataset
    dataset_name: str = "xsum"
    max_samples: int = 10000

    # Device
    device: str = "cuda:0"

    # Diffusion process
    diffusion_steps: int = 2000  # Total diffusion steps in the noise schedule
    noise_schedule: str = "sqrt"  # Beta schedule name ("sqrt", "linear", "cosine")

    # Activation collection
    diffusion_timesteps: list[int] = field(default_factory=lambda: [100, 200, 300, 400, 500])
    collection_batch_size: int = 16

    # SAE
    sae_dictionary_size: int = 16384
    sae_top_k: int = 32
    sae_learning_rate: float = 3e-4
    sae_training_epochs: int = 5
    sae_batch_size: int = 256
    force_retrain: bool = False

    # Output
    output_dir: str = "./experiments/run_001"
    random_seed: int = 42
    log_interval: int = 100

    @staticmethod
    def from_yaml(path: str) -> ExperimentConfig:
        """Load config from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(p) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return ExperimentConfig(**data)

    def to_yaml(self, path: str) -> None:
        """Serialize config to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> None:
        """Validate all required fields and constraints.

        Raises ValueError with details about any invalid or missing fields.
        """
        errors: list[str] = []

        # Check string fields are non-empty
        for fname in ("model_checkpoint_path", "dataset_name", "output_dir"):
            val = getattr(self, fname)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"'{fname}' must be a non-empty string")

        # Check model_arch
        if self.model_arch not in ("transformer", "s2s_CAT"):
            errors.append(
                f"'model_arch' must be 'transformer' or 's2s_CAT', got {self.model_arch!r}"
            )

        # Check positive integers
        for fname in (
            "in_channel",
            "model_channels",
            "out_channel",
            "vocab_size",
            "max_samples",
            "collection_batch_size",
            "sae_dictionary_size",
            "sae_top_k",
            "sae_training_epochs",
            "sae_batch_size",
            "log_interval",
        ):
            val = getattr(self, fname)
            if not isinstance(val, int) or val <= 0:
                errors.append(f"'{fname}' must be a positive integer, got {val!r}")

        # Check positive float
        if not isinstance(self.sae_learning_rate, (int, float)) or self.sae_learning_rate <= 0:
            errors.append(
                f"'sae_learning_rate' must be a positive number, got {self.sae_learning_rate!r}"
            )

        # Check random_seed is an integer
        if not isinstance(self.random_seed, int):
            errors.append(f"'random_seed' must be an integer, got {self.random_seed!r}")

        # Check device pattern
        if not isinstance(self.device, str) or not _DEVICE_PATTERN.match(self.device):
            errors.append(
                f"'device' must match 'cpu', 'cuda', or 'cuda:N', got {self.device!r}"
            )

        # Check diffusion process params
        if not isinstance(self.diffusion_steps, int) or self.diffusion_steps <= 0:
            errors.append(f"'diffusion_steps' must be a positive integer, got {self.diffusion_steps!r}")
        if self.noise_schedule not in ("sqrt", "linear", "cosine"):
            errors.append(
                f"'noise_schedule' must be 'sqrt', 'linear', or 'cosine', got {self.noise_schedule!r}"
            )

        # Check timesteps
        if (
            not isinstance(self.diffusion_timesteps, list)
            or len(self.diffusion_timesteps) == 0
        ):
            errors.append("'diffusion_timesteps' must be a non-empty list of integers")
        elif not all(isinstance(t, int) and t > 0 for t in self.diffusion_timesteps):
            errors.append("'diffusion_timesteps' must contain only positive integers")

        # Check booleans
        for fname in ("force_retrain", "init_pretrained", "learn_sigma", "fix_encoder"):
            val = getattr(self, fname)
            if not isinstance(val, bool):
                errors.append(f"'{fname}' must be a boolean, got {val!r}")

        if errors:
            raise ValueError(
                "Invalid ExperimentConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )
