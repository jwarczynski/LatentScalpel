"""Tests for ActivationStore and ActivationCollector."""

from __future__ import annotations

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path

import pytest

from geniesae.activation_collector import ActivationStore, ActivationCollector
from geniesae.config import ExperimentConfig


# ---------------------------------------------------------------------------
# ActivationStore tests
# ---------------------------------------------------------------------------


class TestActivationStore:
    """Unit tests for disk-backed ActivationStore."""

    def test_save_and_load_metadata(self, tmp_path: Path) -> None:
        store = ActivationStore(str(tmp_path))
        meta = {"num_layers": 2, "activation_dim": 8, "timesteps": [100], "num_samples": 10}
        store.save_metadata(meta)

        store2 = ActivationStore(str(tmp_path))
        assert store2.metadata == meta
        assert store2.num_layers == 2
        assert store2.activation_dim == 8

    def test_save_and_load_activations(self, tmp_path: Path) -> None:
        store = ActivationStore(str(tmp_path))
        act = torch.randn(4, 8)
        store.save_activations(layer_idx=0, timestep=100, batch_idx=0, activations=act)

        pt_file = tmp_path / "layer_00" / "timestep_0100_batch_000.pt"
        assert pt_file.exists()

        loaded = torch.load(pt_file, map_location="cpu", weights_only=True)
        assert torch.allclose(loaded, act)

    def test_get_layer_dataset(self, tmp_path: Path) -> None:
        store = ActivationStore(str(tmp_path))
        act1 = torch.randn(3, 8)
        act2 = torch.randn(5, 8)
        store.save_activations(0, 100, 0, act1)
        store.save_activations(0, 100, 1, act2)
        store.save_metadata({"num_layers": 1, "activation_dim": 8, "timesteps": [100], "num_samples": 8})

        ds = store.get_layer_dataset(0)
        assert len(ds) == 8
        assert torch.allclose(ds[0], act1[0])
        assert torch.allclose(ds[2], act1[2])
        assert torch.allclose(ds[3], act2[0])
        assert torch.allclose(ds[7], act2[4])

    def test_get_layer_dataset_missing_layer(self, tmp_path: Path) -> None:
        store = ActivationStore(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            store.get_layer_dataset(99)

    def test_metadata_not_found(self, tmp_path: Path) -> None:
        store = ActivationStore(str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            _ = store.metadata

    def test_multiple_layers(self, tmp_path: Path) -> None:
        store = ActivationStore(str(tmp_path))
        for layer in range(3):
            store.save_activations(layer, 100, 0, torch.randn(2, 16))
        store.save_metadata({"num_layers": 3, "activation_dim": 16, "timesteps": [100], "num_samples": 6})

        assert store.num_layers == 3
        assert store.activation_dim == 16
        for layer in range(3):
            ds = store.get_layer_dataset(layer)
            assert len(ds) == 2


# ---------------------------------------------------------------------------
# Mock GENIE-like model for ActivationCollector tests
# ---------------------------------------------------------------------------

EMBED_DIM = 8
VOCAB_SIZE = 32


class _MockDecoderLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        return self.linear(x)


class _MockGENIEModel(nn.Module):
    """Minimal model mimicking CrossAttention_Diffusion_LM's interface.

    Has ``word_embedding``, ``get_embeds``, ``transformer_blocks``, and
    accepts ``(x, timesteps, src_input_ids, src_attention_mask)`` in forward.
    """

    def __init__(self, num_layers: int = 2, dim: int = EMBED_DIM, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, dim)
        self.transformer_blocks = nn.ModuleList(
            [_MockDecoderLayer(dim) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(dim, dim)

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(input_ids)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for block in self.transformer_blocks:
            h = block(h)
        return self.proj(h)


# ---------------------------------------------------------------------------
# ActivationCollector tests
# ---------------------------------------------------------------------------


class TestActivationCollector:
    """Smoke tests for ActivationCollector using a mock GENIE model."""

    def _make_config(self, tmp_path: Path, timesteps: list[int] | None = None) -> ExperimentConfig:
        return ExperimentConfig(
            model_checkpoint_path="/fake/path",
            dataset_name="test",
            max_samples=10,
            device="cpu",
            diffusion_steps=100,
            noise_schedule="sqrt",
            diffusion_timesteps=timesteps or [10, 20],
            collection_batch_size=4,
            output_dir=str(tmp_path / "output"),
            random_seed=42,
            log_interval=1,
        )

    def _make_dataloader(self, num_samples: int = 8, seq_len: int = 4) -> DataLoader:
        """Create a dataloader yielding (input_ids, attention_mask) tuples."""
        input_ids = torch.randint(0, VOCAB_SIZE, (num_samples, seq_len))
        attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)
        return DataLoader(TensorDataset(input_ids, attention_mask), batch_size=4)

    def test_collect_produces_store_with_correct_metadata(self, tmp_path: Path) -> None:
        num_layers = 2
        model = _MockGENIEModel(num_layers=num_layers)
        config = self._make_config(tmp_path, timesteps=[10, 20])
        dl = self._make_dataloader()

        def accessor(nnsight_model):
            return nnsight_model.transformer_blocks

        collector = ActivationCollector(model, config, layer_accessor=accessor)
        store = collector.collect(dl)

        meta = store.metadata
        assert meta["num_layers"] == num_layers
        assert meta["activation_dim"] == EMBED_DIM
        assert meta["timesteps"] == [10, 20]
        assert meta["num_samples"] > 0

    def test_collect_creates_files_for_all_layers(self, tmp_path: Path) -> None:
        num_layers = 3
        model = _MockGENIEModel(num_layers=num_layers)
        config = self._make_config(tmp_path, timesteps=[10])
        dl = self._make_dataloader(num_samples=4)

        def accessor(nnsight_model):
            return nnsight_model.transformer_blocks

        collector = ActivationCollector(model, config, layer_accessor=accessor)
        store = collector.collect(dl)

        act_dir = Path(config.output_dir) / "activations"
        for i in range(num_layers):
            layer_dir = act_dir / f"layer_{i:02d}"
            assert layer_dir.exists(), f"Missing directory for layer {i}"
            pt_files = list(layer_dir.glob("*.pt"))
            assert len(pt_files) > 0, f"No .pt files for layer {i}"

    def test_collect_layer_datasets_are_loadable(self, tmp_path: Path) -> None:
        num_layers = 2
        model = _MockGENIEModel(num_layers=num_layers)
        config = self._make_config(tmp_path, timesteps=[10])
        dl = self._make_dataloader(num_samples=6)

        def accessor(nnsight_model):
            return nnsight_model.transformer_blocks

        collector = ActivationCollector(model, config, layer_accessor=accessor)
        store = collector.collect(dl)

        for layer_idx in range(num_layers):
            ds = store.get_layer_dataset(layer_idx)
            assert len(ds) > 0
            sample = ds[0]
            assert sample.shape == (EMBED_DIM,)
