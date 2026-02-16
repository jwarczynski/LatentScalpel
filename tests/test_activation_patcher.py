"""Tests for PatchingResult and ActivationPatcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from geniesae.activation_patcher import PatchingResult, ActivationPatcher
from geniesae.config import ExperimentConfig
from geniesae.sae import TopKSAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 8
VOCAB_SIZE = 32
DICT_SIZE = 16
TOP_K = 4
NUM_LAYERS = 2


def _make_config(tmp_path: Path, **overrides) -> ExperimentConfig:
    defaults = dict(
        model_checkpoint_path="/fake/path",
        dataset_name="test",
        max_samples=10,
        device="cpu",
        diffusion_steps=100,
        noise_schedule="sqrt",
        diffusion_timesteps=[10],
        collection_batch_size=4,
        sae_dictionary_size=DICT_SIZE,
        sae_top_k=TOP_K,
        sae_learning_rate=1e-3,
        sae_training_epochs=1,
        sae_batch_size=8,
        force_retrain=False,
        output_dir=str(tmp_path / "output"),
        random_seed=42,
        log_interval=100,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# Mock GENIE-like model that supports get_embeds / get_logits
# ---------------------------------------------------------------------------


class _MockDecoderLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        return self.linear(x)


class _MockGENIEModel(nn.Module):
    """Minimal model mimicking CrossAttention_Diffusion_LM for patching tests.

    Accepts ``(x, timesteps, src_input_ids, src_attention_mask)`` and returns
    a tensor of shape ``(batch, seq_len, embed_dim)``.  Also provides
    ``get_embeds`` and ``get_logits`` for loss computation.
    """

    def __init__(
        self, num_layers: int = NUM_LAYERS, dim: int = EMBED_DIM, vocab_size: int = VOCAB_SIZE
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.transformer_blocks = nn.ModuleList(
            [_MockDecoderLayer(dim) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(dim, dim)

    def get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_repr)

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


def _make_dataloader(num_samples: int = 8, seq_len: int = 4) -> DataLoader:
    input_ids = torch.randint(0, VOCAB_SIZE, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)
    return DataLoader(TensorDataset(input_ids, attention_mask), batch_size=4)


def _make_saes(num_layers: int = NUM_LAYERS) -> dict[int, TopKSAE]:
    return {i: TopKSAE(EMBED_DIM, DICT_SIZE, TOP_K) for i in range(num_layers)}


def _layer_accessor(nnsight_model):
    return nnsight_model.transformer_blocks


# ---------------------------------------------------------------------------
# PatchingResult tests
# ---------------------------------------------------------------------------


class TestPatchingResult:
    def test_to_dict(self) -> None:
        r = PatchingResult(layer_idx=0, baseline_loss=3.0, patched_loss=3.5, loss_delta=0.5)
        d = r.to_dict()
        assert d == {
            "layer_idx": 0,
            "baseline_loss": 3.0,
            "patched_loss": 3.5,
            "loss_delta": 0.5,
        }

    def test_to_dict_none_layer(self) -> None:
        r = PatchingResult(layer_idx=None, baseline_loss=2.0, patched_loss=2.5, loss_delta=0.5)
        d = r.to_dict()
        assert d["layer_idx"] is None

    def test_loss_delta_arithmetic(self) -> None:
        r = PatchingResult(layer_idx=1, baseline_loss=3.0, patched_loss=2.8, loss_delta=-0.2)
        assert r.loss_delta == pytest.approx(r.patched_loss - r.baseline_loss)


# ---------------------------------------------------------------------------
# ActivationPatcher.compute_baseline_loss tests
# ---------------------------------------------------------------------------


class TestComputeBaselineLoss:
    def test_returns_float(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        loss = patcher.compute_baseline_loss(dl)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_deterministic(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        # Seed before each call so the diffusion noise is identical
        torch.manual_seed(42)
        loss1 = patcher.compute_baseline_loss(dl)
        torch.manual_seed(42)
        loss2 = patcher.compute_baseline_loss(dl)
        assert loss1 == pytest.approx(loss2)


# ---------------------------------------------------------------------------
# ActivationPatcher.patch_single_layer tests
# ---------------------------------------------------------------------------


class TestPatchSingleLayer:
    def test_returns_patching_result(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        baseline = patcher.compute_baseline_loss(dl)
        result = patcher.patch_single_layer(0, dl, baseline)

        assert isinstance(result, PatchingResult)
        assert result.layer_idx == 0
        assert result.baseline_loss == baseline
        assert isinstance(result.patched_loss, float)

    def test_loss_delta_is_correct(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        baseline = patcher.compute_baseline_loss(dl)
        result = patcher.patch_single_layer(0, dl, baseline)

        assert result.loss_delta == pytest.approx(result.patched_loss - result.baseline_loss)

    def test_missing_sae_raises_key_error(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = {0: TopKSAE(EMBED_DIM, DICT_SIZE, TOP_K)}
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        baseline = patcher.compute_baseline_loss(dl)

        with pytest.raises(KeyError, match="layer 5"):
            patcher.patch_single_layer(5, dl, baseline)


# ---------------------------------------------------------------------------
# ActivationPatcher.patch_all_layers tests
# ---------------------------------------------------------------------------


class TestPatchAllLayers:
    def test_returns_patching_result_with_none_layer(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        baseline = patcher.compute_baseline_loss(dl)
        result = patcher.patch_all_layers(dl, baseline)

        assert isinstance(result, PatchingResult)
        assert result.layer_idx is None
        assert result.baseline_loss == baseline

    def test_loss_delta_is_correct(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        baseline = patcher.compute_baseline_loss(dl)
        result = patcher.patch_all_layers(dl, baseline)

        assert result.loss_delta == pytest.approx(result.patched_loss - result.baseline_loss)


# ---------------------------------------------------------------------------
# ActivationPatcher.run_full_evaluation tests
# ---------------------------------------------------------------------------


class TestRunFullEvaluation:
    def test_returns_all_results(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        results = patcher.run_full_evaluation(dl)

        # 2 per-layer + 1 all-layers = 3 results
        assert len(results) == 3
        assert results[0].layer_idx == 0
        assert results[1].layer_idx == 1
        assert results[2].layer_idx is None

    def test_saves_json(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        patcher.run_full_evaluation(dl)

        json_path = Path(config.output_dir) / "results" / "patching_results.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "baseline_loss" in data
        assert "per_layer" in data
        assert "all_layers" in data
        assert len(data["per_layer"]) == 2
        assert "patched_loss" in data["all_layers"]
        assert "loss_delta" in data["all_layers"]

    def test_json_structure_matches_spec(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        results = patcher.run_full_evaluation(dl)

        json_path = Path(config.output_dir) / "results" / "patching_results.json"
        with open(json_path) as f:
            data = json.load(f)

        for entry in data["per_layer"]:
            assert "layer_idx" in entry
            assert "patched_loss" in entry
            assert "loss_delta" in entry

        baseline = results[0].baseline_loss
        assert data["baseline_loss"] == pytest.approx(baseline)

    def test_all_deltas_are_consistent(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _make_config(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        results = patcher.run_full_evaluation(dl)

        for r in results:
            assert r.loss_delta == pytest.approx(r.patched_loss - r.baseline_loss)
