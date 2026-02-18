"""NNsight-based activation extraction from GENIE decoder layers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from geniesae.genie_model import DiffusionHelper

logger = logging.getLogger("geniesae.activation_collector")


class _LayerDataset(Dataset):
    """PyTorch Dataset backed by .pt files in a single layer directory.

    Builds a lightweight index of (file_path, num_rows) without loading
    full tensors into RAM.  Individual samples are loaded on-demand in
    ``__getitem__``.
    """

    _INDEX_NAME = "_file_index.json"

    def __init__(self, layer_dir: Path) -> None:
        self._layer_dir = layer_dir
        self._files = sorted(layer_dir.glob("timestep_*.pt"))
        if not self._files:
            raise FileNotFoundError(f"No timestep_*.pt files found in {layer_dir}")

        # Try to load a cached index first (avoids touching every .pt file).
        index_path = layer_dir / self._INDEX_NAME
        if index_path.exists():
            self._load_index(index_path)
        else:
            self._build_index()
            self._save_index(index_path)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Scan .pt files to record per-file row counts.

        Uses ``mmap=True`` when available (PyTorch ≥ 2.1) so the tensor
        data is never copied into RAM — we only need ``.shape[0]``.
        Falls back to a regular load + immediate ``del`` otherwise.
        """
        import sys

        self._cumulative: list[int] = []
        total = 0
        n_files = len(self._files)
        for i, f in enumerate(self._files):
            try:
                t = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
            except TypeError:
                # Older PyTorch without mmap kwarg
                t = torch.load(f, map_location="cpu", weights_only=True)
            rows = t.shape[0]
            del t  # free immediately
            total += rows
            self._cumulative.append(total)
            if (i + 1) % 500 == 0:
                print(
                    f"[_LayerDataset] indexed {i + 1}/{n_files} files "
                    f"({total} samples so far)",
                    flush=True,
                )
        self._total = total
        print(
            f"[_LayerDataset] index complete: {n_files} files, "
            f"{total} total samples",
            flush=True,
        )

    def _save_index(self, path: Path) -> None:
        """Persist the cumulative index so subsequent runs skip scanning."""
        import json as _json

        data = {
            "files": [f.name for f in self._files],
            "cumulative": self._cumulative,
            "total": self._total,
        }
        with open(path, "w") as fh:
            _json.dump(data, fh)

    def _load_index(self, path: Path) -> None:
        """Load a previously saved index."""
        import json as _json

        with open(path) as fh:
            data = _json.load(fh)

        # Validate that the file list still matches
        indexed_names = data["files"]
        current_names = [f.name for f in self._files]
        if indexed_names != current_names:
            print(
                "[_LayerDataset] cached index stale — rebuilding",
                flush=True,
            )
            self._build_index()
            self._save_index(path)
            return

        self._cumulative = data["cumulative"]
        self._total = data["total"]
        print(
            f"[_LayerDataset] loaded cached index: {len(self._files)} files, "
            f"{self._total} total samples",
            flush=True,
        )

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


class ChunkedLayerDataset(torch.utils.data.IterableDataset):
    """Memory-efficient iterable dataset that streams .pt file chunks.

    Instead of random-access per sample (which requires loading an entire
    .pt file for every ``__getitem__`` call), this dataset iterates over
    files sequentially, yielding all rows from each file before moving on.

    Shuffling happens at two levels:
    - File order is shuffled each epoch.
    - Rows within each file are shuffled.

    When ``max_samples`` is set, samples are drawn equally from each
    detected timestep so the SAE sees a balanced representation across
    the diffusion process.
    """

    def __init__(
        self,
        layer_dir: Path,
        *,
        shuffle: bool = True,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self._files = sorted(layer_dir.glob("timestep_*.pt"))
        if not self._files:
            raise FileNotFoundError(f"No timestep_*.pt files found in {layer_dir}")
        self._shuffle = shuffle
        self._max_samples = max_samples

        # Group files by timestep for balanced sampling
        self._files_by_timestep = self._group_by_timestep()

        # Compute total length from cached index or by scanning
        index_path = layer_dir / _LayerDataset._INDEX_NAME
        if index_path.exists():
            import json as _json
            with open(index_path) as fh:
                data = _json.load(fh)
            self._total = data["total"]
        else:
            # Quick scan — just need total
            ds = _LayerDataset(layer_dir)
            self._total = len(ds)

        if self._max_samples is not None:
            self._total = min(self._total, self._max_samples)

    def _group_by_timestep(self) -> dict[int, list[Path]]:
        """Group .pt files by timestep parsed from filename."""
        import re
        pattern = re.compile(r"timestep_(\d+)")
        groups: dict[int, list[Path]] = {}
        for f in self._files:
            m = pattern.search(f.stem)
            if m:
                ts = int(m.group(1))
                groups.setdefault(ts, []).append(f)
            else:
                # Files without timestep in name go into a catch-all group
                groups.setdefault(-1, []).append(f)
        return groups

    @property
    def total_samples(self) -> int:
        return self._total

    def _worker_files(self, files: list[Path]) -> list[Path]:
        """Shard files across DataLoader workers to avoid duplicates."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return files
        per_worker = len(files) // worker_info.num_workers
        start = worker_info.id * per_worker
        if worker_info.id == worker_info.num_workers - 1:
            return files[start:]
        return files[start : start + per_worker]

    def _iter_files_unlimited(self):
        """Yield samples with optional shuffle."""
        files = self._worker_files(list(self._files))
        if self._shuffle:
            import random
            random.shuffle(files)
        for f in files:
            chunk = torch.load(f, map_location="cpu", weights_only=False)
            if self._shuffle:
                perm = torch.randperm(chunk.shape[0])
                chunk = chunk[perm]
            yield from chunk

    def _iter_files_limited(self):
        """Yield samples balanced across timesteps up to max_samples."""
        import random as _random

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        worker_budget = self._max_samples // num_workers
        if worker_id < self._max_samples % num_workers:
            worker_budget += 1

        num_timesteps = len(self._files_by_timestep)
        per_timestep = worker_budget // max(num_timesteps, 1)
        remainder = worker_budget - per_timestep * num_timesteps

        timestep_keys = sorted(self._files_by_timestep.keys())
        if self._shuffle:
            _random.shuffle(timestep_keys)

        for ts_i, ts in enumerate(timestep_keys):
            budget = per_timestep + (1 if ts_i < remainder else 0)
            emitted = 0

            files = self._worker_files(list(self._files_by_timestep[ts]))
            if self._shuffle:
                _random.shuffle(files)

            for f in files:
                if emitted >= budget:
                    break
                chunk = torch.load(f, map_location="cpu", weights_only=False)
                if self._shuffle:
                    perm = torch.randperm(chunk.shape[0])
                    chunk = chunk[perm]
                take = min(chunk.shape[0], budget - emitted)
                yield from chunk[:take]
                emitted += take

    def __iter__(self):
        if self._max_samples is not None:
            yield from self._iter_files_limited()
        else:
            yield from self._iter_files_unlimited()

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
        │   ├── timestep_0100.pt
        │   └── ...
        ├── layer_01/
        │   └── ...
        └── metadata.json

    Each ``.pt`` file contains a tensor of shape ``(N, activation_dim)``
    holding all activations for a single timestep (concatenated across batches).
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)
        self._metadata: dict | None = None
        self._buffers: dict[tuple[int, int], list[torch.Tensor]] = {}
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
        """Buffer activations in memory. Call :meth:`flush_timestep` to write."""
        key = (layer_idx, timestep)
        if key not in self._buffers:
            self._buffers[key] = []
        self._buffers[key].append(activations.cpu())

    def flush_timestep(self, timestep: int) -> None:
        """Concatenate all buffered batches for *timestep* and write one file per layer."""
        keys_to_flush = [k for k in self._buffers if k[1] == timestep]
        for key in keys_to_flush:
            layer_idx, ts = key
            layer_dir = self._base_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            combined = torch.cat(self._buffers.pop(key), dim=0)
            filename = f"timestep_{ts:04d}.pt"
            torch.save(combined, layer_dir / filename)

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

    def get_chunked_dataset(
        self, layer_idx: int, *, shuffle: bool = True, max_samples: int | None = None,
    ) -> "ChunkedLayerDataset":
        """Return a memory-efficient iterable dataset for training."""
        layer_dir = self._base_dir / f"layer_{layer_idx:02d}"
        if not layer_dir.exists():
            raise FileNotFoundError(f"No activation directory for layer {layer_idx}: {layer_dir}")
        return ChunkedLayerDataset(layer_dir, shuffle=shuffle, max_samples=max_samples)

    def compute_layer_mean(
        self, layer_idx: int, *, max_samples: int | None = None,
    ) -> torch.Tensor:
        """Compute the mean activation vector for a layer.

        Results are cached to ``layer_XX/mean.pt`` (or
        ``layer_XX/mean_NNNk.pt`` when ``max_samples`` is set) so
        subsequent calls return instantly.

        When ``max_samples`` is set, samples are drawn equally from each
        timestep (matching the training data distribution).
        """
        layer_dir = self._base_dir / f"layer_{layer_idx:02d}"
        if max_samples is not None:
            cache_name = f"mean_{max_samples // 1000}k.pt"
        else:
            cache_name = "mean.pt"
        cache_path = layer_dir / cache_name

        if cache_path.exists():
            print(f"[compute_mean] loading cached mean from {cache_path}", flush=True)
            return torch.load(cache_path, map_location="cpu", weights_only=True)

        ds = self.get_chunked_dataset(
            layer_idx, shuffle=False, max_samples=max_samples,
        )
        mean_acc: torch.Tensor | None = None
        count = 0
        for sample in ds:
            if mean_acc is None:
                mean_acc = torch.zeros(sample.shape[-1], dtype=torch.float64)
            mean_acc += sample.to(torch.float64)
            count += 1
            if count % 500_000 == 0:
                print(f"[compute_mean] {count} samples processed", flush=True)

        if mean_acc is None:
            raise RuntimeError(f"No data found for layer {layer_idx}")
        print(f"[compute_mean] done: {count} samples", flush=True)
        result = (mean_acc / count).float()

        # Cache to disk
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
        print(f"[compute_mean] cached to {cache_path}", flush=True)
        return result

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
        config: Any config object with diffusion_steps, noise_schedule,
            diffusion_timesteps, device, output_dir, and log_interval attrs.
        layer_accessor: Callable that, given the NNsight-wrapped model,
            returns a list/sequence of NNsight envoy layer proxies.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
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
        # Try common GENIE model architectures
        if hasattr(self._nnsight_model, "transformer_blocks"):
            return self._nnsight_model.transformer_blocks
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

            store.flush_timestep(timestep)
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
