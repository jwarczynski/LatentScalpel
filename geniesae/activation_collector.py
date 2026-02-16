"""NNsight-based activation extraction from GENIE decoder layers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from geniesae.config import ExperimentConfig
from geniesae.genie_model import DiffusionHelper

logger = logging.getLogger("geniesae.activation_collector")


class _LayerDataset(Dataset):
    """PyTorch Dataset backed by .pt files in a single layer directory."""

    def __init__(self, layer_dir: Path) -> None:
        self._files = sorted(layer_dir.glob("*.pt"))
        if not self._files:
            raise FileNotFoundError(f"No .pt files found in {layer_dir}")
        # Pre-load shapes so __len__ is cheap: each file is (N, dim)
        self._cumulative: list[int] = []
        total = 0
        for f in self._files:
            t = torch.load(f, map_location="cpu", weights_only=True)
            total += t.shape[0]
            self._cumulative.append(total)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._total}")
        lo, hi = 0, len(self._cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        file_idx = lo
        offset = idx - (self._cumulative[file_idx - 1] if file_idx > 0 else 0)
        tensor = torch.load(self._files[file_idx], map_location="cpu", weights_only=True)
        return tensor[offset]


class ActivationStore:
    """Disk-backed activation dataset that provides DataLoader-compatible access.

    Storage layout::

        {base_dir}/
        ├── layer_00/
        │   ├── timestep_100_batch_000.pt
        │   └── ...
        ├── layer_01/
        │   └── ...
        └── metadata.json

    Each ``.pt`` file contains a tensor of shape ``(N, activation_dim)``.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)
        self._metadata: dict | None = None
        meta_path = self._base_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

    # ------------------------------------------------------------------
    # Writing API
    # ------------------------------------------------------------------

    def save_activations(
        self,
        layer_idx: int,
        timestep: int,
        batch_idx: int,
        activations: torch.Tensor,
    ) -> None:
        layer_dir = self._base_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        filename = f"timestep_{timestep:04d}_batch_{batch_idx:03d}.pt"
        torch.save(activations.cpu(), layer_dir / filename)

    def save_metadata(self, metadata: dict) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._base_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self._metadata = metadata

    # ------------------------------------------------------------------
    # Reading API
    # ------------------------------------------------------------------

    def get_layer_dataset(self, layer_idx: int) -> Dataset:
        layer_dir = self._base_dir / f"layer_{layer_idx:02d}"
        if not layer_dir.exists():
            raise FileNotFoundError(f"No activation directory for layer {layer_idx}: {layer_dir}")
        return _LayerDataset(layer_dir)

    @property
    def num_layers(self) -> int:
        return int(self.metadata["num_layers"])

    @property
    def activation_dim(self) -> int:
        return int(self.metadata["activation_dim"])

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            meta_path = self._base_dir / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"metadata.json not found in {self._base_dir}. "
                    "Run activation collection first."
                )
            with open(meta_path) as f:
                self._metadata = json.load(f)
        return self._metadata


class ActivationCollector:
    """Collects residual stream activations from GENIE decoder layers using NNsight.

    Handles the GENIE-specific forward pass: converts token IDs to embeddings,
    adds diffusion noise at each timestep, and passes the proper arguments
    ``(noised_x, timesteps, src_input_ids, src_attention_mask)`` through the
    model.

    Args:
        model: The GENIE model (unwrapped nn.Module).
        config: Experiment configuration.
        layer_accessor: Callable that, given the NNsight-wrapped model,
            returns a list/sequence of NNsight envoy layer proxies.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        layer_accessor: Callable | None = None,
    ) -> None:
        from nnsight import NNsight

        self._model = model
        self._config = config
        self._nnsight_model = NNsight(model)
        self._layer_accessor = layer_accessor
        self._diffusion = DiffusionHelper(
            num_timesteps=config.diffusion_steps,
            schedule_name=config.noise_schedule,
        )

    def _get_layers(self):
        """Return NNsight envoy proxies for the decoder layers."""
        if self._layer_accessor is not None:
            return self._layer_accessor(self._nnsight_model)
        return self._nnsight_model.decoder.layers

    @torch.no_grad()
    def collect(self, dataloader: DataLoader) -> ActivationStore:
        """Collect activations from all decoder layers across diffusion timesteps.

        The dataloader yields ``(input_ids, attention_mask)`` tuples from a
        ``TensorDataset``.  For each batch and timestep we:

        1. Convert ``input_ids`` to embeddings via ``model.get_embeds``.
        2. Add diffusion noise at the current timestep.
        3. Run an NNsight trace with the proper GENIE forward-pass arguments.
        4. Save each layer's output to disk.

        Returns:
            An :class:`ActivationStore` pointing at the saved activation files.
        """
        config = self._config
        store_dir = str(Path(config.output_dir) / "activations")
        store = ActivationStore(store_dir)

        layers = self._get_layers()
        num_layers = len(layers)
        timesteps = config.diffusion_timesteps

        logger.info(
            "Starting activation collection: %d layers, %d timesteps, device=%s",
            num_layers, len(timesteps), config.device,
        )

        activation_dim: int | None = None
        total_samples = 0
        batch_counter = 0

        self._model.eval()

        for ts_idx, timestep in enumerate(timesteps):
            for batch_idx, batch in enumerate(dataloader):
                # Unpack dataloader batch — (input_ids, attention_mask)
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_ids = batch[0].to(config.device)
                    attention_mask = batch[1].to(config.device)
                else:
                    # Fallback for non-GENIE models (e.g. tests with plain tensors)
                    input_ids = batch.to(config.device) if isinstance(batch, torch.Tensor) else batch[0].to(config.device)
                    attention_mask = None

                batch_size = input_ids.shape[0]

                # Build GENIE forward-pass inputs
                # 1. Get clean word embeddings
                with torch.no_grad():
                    x_start = self._model.get_embeds(input_ids)

                # 2. Create timestep tensor and add noise
                t_tensor = torch.full(
                    (batch_size,), timestep, dtype=torch.long, device=config.device
                )
                x_noised = self._diffusion.q_sample(x_start, t_tensor)

                # 3. Trace through the model with proper GENIE args
                saved_outputs: list = []
                with self._nnsight_model.trace(
                    x_noised, t_tensor, input_ids, attention_mask
                ):
                    for layer in layers:
                        saved_outputs.append(layer.output.save())

                # 4. Process and save each layer's activations
                for layer_idx, proxy in enumerate(saved_outputs):
                    act = proxy.value if hasattr(proxy, "value") else proxy
                    if isinstance(act, (tuple, list)):
                        act = act[0]
                    act = act.detach().float()

                    # Flatten to 2-D: (batch * seq_len, dim)
                    if act.dim() == 3:
                        act = act.reshape(-1, act.shape[-1])
                    elif act.dim() > 3:
                        act = act.reshape(-1, act.shape[-1])

                    if activation_dim is None:
                        activation_dim = act.shape[-1]

                    total_samples += act.shape[0]
                    store.save_activations(layer_idx, timestep, batch_counter, act)

                batch_counter += 1

                if (batch_idx + 1) % max(1, config.log_interval) == 0:
                    logger.info(
                        "  timestep %d | batch %d/%d collected",
                        timestep, batch_idx + 1,
                        len(dataloader) if hasattr(dataloader, "__len__") else "?",
                    )

            logger.info("Completed timestep %d (%d/%d)", timestep, ts_idx + 1, len(timesteps))

        metadata = {
            "num_layers": num_layers,
            "activation_dim": activation_dim or 0,
            "timesteps": timesteps,
            "num_samples": total_samples,
        }
        store.save_metadata(metadata)

        logger.info(
            "Activation collection complete: %d layers, %d total samples, dim=%d",
            num_layers, total_samples, activation_dim or 0,
        )
        return store
