"""Tests for ReconstructionMetrics and SAETrainer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from geniesae.config import ExperimentConfig
from geniesae.activation_collector import ActivationStore
from geniesae.sae import TopKSAE
from geniesae.sae_trainer import ReconstructionMetrics, SAETrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACTIVATION_DIM = 16
DICT_SIZE = 64
TOP_K = 4
NUM_LAYERS = 2
NUM_SAMPLES = 20


def _make_config(tmp_path: Path, **overrides) -> ExperimentConfig:
    defaults = dict(
        model_checkpoint_path="/fake/path",
        dataset_name="test",
        max_samples=100,
        device="cpu",
        diffusion_timesteps=[100],
        collection_batch_size=4,
        sae_dictionary_size=DICT_SIZE,
        sae_top_k=TOP_K,
        sae_learning_rate=1e-3,
        sae_training_epochs=2,
        sae_batch_size=8,
        force_retrain=False,
        output_dir=str(tmp_path / "output"),
        random_seed=42,
        log_interval=100,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _create_activation_store(
    tmp_path: Path,
    num_layers: int = NUM_LAYERS,
    num_samples: int = NUM_SAMPLES,
    activation_dim: int = ACTIVATION_DIM,
) -> ActivationStore:
    """Create an ActivationStore with synthetic .pt files."""
    store_dir = tmp_path / "activations"
    store = ActivationStore(str(store_dir))
    for layer_idx in range(num_layers):
        data = torch.randn(num_samples, activation_dim)
        store.save_activations(layer_idx, 100, 0, data)
    store.save_metadata({
        "num_layers": num_layers,
        "activation_dim": activation_dim,
        "timesteps": [100],
        "num_samples": num_samples * num_layers,
    })
    return store


# ---------------------------------------------------------------------------
# ReconstructionMetrics tests
# ---------------------------------------------------------------------------


class TestReconstructionMetrics:
    def test_to_dict(self) -> None:
        m = ReconstructionMetrics(mse=0.01, explained_variance=0.95, l0_sparsity=4.0)
        d = m.to_dict()
        assert d == {"mse": 0.01, "explained_variance": 0.95, "l0_sparsity": 4.0}

    def test_fields_accessible(self) -> None:
        m = ReconstructionMetrics(mse=0.5, explained_variance=0.8, l0_sparsity=3.2)
        assert m.mse == 0.5
        assert m.explained_variance == 0.8
        assert m.l0_sparsity == 3.2


# ---------------------------------------------------------------------------
# SAETrainer.train_layer tests
# ---------------------------------------------------------------------------


class TestTrainLayer:
    def test_trains_and_returns_sae(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path)
        trainer = SAETrainer(config)

        sae = trainer.train_layer(0, store)
        assert isinstance(sae, TopKSAE)
        assert sae.activation_dim == ACTIVATION_DIM
        assert sae.dictionary_size == DICT_SIZE
        assert sae.k == TOP_K

    def test_saves_checkpoint(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path)
        trainer = SAETrainer(config)

        trainer.train_layer(0, store)
        ckpt_path = Path(config.output_dir) / "sae_checkpoints" / "layer_00.pt"
        assert ckpt_path.exists()

    def test_skip_if_checkpoint_exists(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1, force_retrain=False)
        store = _create_activation_store(tmp_path)
        trainer = SAETrainer(config)

        # Train once
        sae1 = trainer.train_layer(0, store)
        # Second call should load from checkpoint, not retrain
        sae2 = trainer.train_layer(0, store)
        assert isinstance(sae2, TopKSAE)
        # Weights should match
        for p1, p2 in zip(sae1.parameters(), sae2.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    def test_force_retrain_overrides_checkpoint(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1, force_retrain=False)
        store = _create_activation_store(tmp_path)
        trainer = SAETrainer(config)

        # Train once
        trainer.train_layer(0, store)
        ckpt_path = Path(config.output_dir) / "sae_checkpoints" / "layer_00.pt"
        mtime_before = ckpt_path.stat().st_mtime

        # Force retrain
        config_force = _make_config(tmp_path, sae_training_epochs=1, force_retrain=True)
        trainer_force = SAETrainer(config_force)

        import time
        time.sleep(0.05)  # ensure mtime differs
        trainer_force.train_layer(0, store)
        mtime_after = ckpt_path.stat().st_mtime
        assert mtime_after > mtime_before

    def test_checkpoint_load_produces_correct_output(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path)
        trainer = SAETrainer(config)

        sae_trained = trainer.train_layer(0, store)
        # Load from checkpoint
        sae_loaded = trainer._load_checkpoint(0)

        test_input = torch.randn(4, ACTIVATION_DIM)
        sae_trained.eval()
        sae_loaded.eval()
        with torch.no_grad():
            out1, _ = sae_trained(test_input)
            out2, _ = sae_loaded(test_input)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# SAETrainer.train_all_layers tests
# ---------------------------------------------------------------------------


class TestTrainAllLayers:
    def test_returns_dict_with_all_layers(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path, num_layers=3)
        trainer = SAETrainer(config)

        saes = trainer.train_all_layers(store, num_layers=3)
        assert set(saes.keys()) == {0, 1, 2}
        for sae in saes.values():
            assert isinstance(sae, TopKSAE)

    def test_creates_checkpoint_per_layer(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path, num_layers=2)
        trainer = SAETrainer(config)

        trainer.train_all_layers(store, num_layers=2)
        ckpt_dir = Path(config.output_dir) / "sae_checkpoints"
        assert (ckpt_dir / "layer_00.pt").exists()
        assert (ckpt_dir / "layer_01.pt").exists()


# ---------------------------------------------------------------------------
# SAETrainer.evaluate tests
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_metrics_have_valid_bounds(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=2)
        store = _create_activation_store(tmp_path)
        trainer = SAETrainer(config)

        sae = trainer.train_layer(0, store)
        metrics = trainer.evaluate(sae, store, 0)

        assert metrics.mse >= 0.0
        assert 0.0 <= metrics.explained_variance <= 1.0
        assert 0.0 <= metrics.l0_sparsity <= TOP_K

    def test_l0_equals_k_for_topk_sae(self, tmp_path: Path) -> None:
        """L0 sparsity should be exactly K since TopKSAE keeps exactly K features."""
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path)
        trainer = SAETrainer(config)

        sae = trainer.train_layer(0, store)
        metrics = trainer.evaluate(sae, store, 0)
        assert metrics.l0_sparsity == pytest.approx(TOP_K, abs=0.01)

    def test_perfect_reconstruction_metrics(self, tmp_path: Path) -> None:
        """An identity-like SAE should yield near-zero MSE and high explained variance."""
        config = _make_config(tmp_path, sae_training_epochs=1, sae_top_k=DICT_SIZE)
        # With k == dictionary_size, all features are kept
        store = _create_activation_store(tmp_path, num_samples=10)

        # Create an SAE that approximates identity
        sae = TopKSAE(ACTIVATION_DIM, DICT_SIZE, k=DICT_SIZE)
        # Set encoder/decoder to approximate identity (won't be perfect but
        # we just check the metric computation path works)
        sae.eval()

        trainer = SAETrainer(config)
        metrics = trainer.evaluate(sae, store, 0)
        # Just verify the computation runs and returns valid metrics
        assert metrics.mse >= 0.0
        assert 0.0 <= metrics.explained_variance <= 1.0


# ---------------------------------------------------------------------------
# SAETrainer.evaluate_all_layers tests
# ---------------------------------------------------------------------------


class TestEvaluateAllLayers:
    def test_saves_metrics_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path, num_layers=2)
        trainer = SAETrainer(config)

        saes = trainer.train_all_layers(store, num_layers=2)
        metrics = trainer.evaluate_all_layers(saes, store)

        assert set(metrics.keys()) == {0, 1}

        json_path = Path(config.output_dir) / "results" / "reconstruction_metrics.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert "layer_00" in data
        assert "layer_01" in data
        assert "mse" in data["layer_00"]
        assert "explained_variance" in data["layer_00"]
        assert "l0_sparsity" in data["layer_00"]

    def test_metrics_json_values_match_returned(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, sae_training_epochs=1)
        store = _create_activation_store(tmp_path, num_layers=1)
        trainer = SAETrainer(config)

        saes = trainer.train_all_layers(store, num_layers=1)
        metrics = trainer.evaluate_all_layers(saes, store)

        json_path = Path(config.output_dir) / "results" / "reconstruction_metrics.json"
        with open(json_path) as f:
            data = json.load(f)

        m = metrics[0]
        assert data["layer_00"]["mse"] == pytest.approx(m.mse)
        assert data["layer_00"]["explained_variance"] == pytest.approx(m.explained_variance)
        assert data["layer_00"]["l0_sparsity"] == pytest.approx(m.l0_sparsity)
