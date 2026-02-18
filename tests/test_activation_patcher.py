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
from geniesae.sae import TopKSAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 8
VOCAB_SIZE = 32
DICT_SIZE = 16
TOP_K = 4
NUM_LAYERS = 2


class _FakePatcherConfig:
    """Lightweight config stand-in for ActivationPatcher."""

    def __init__(self, tmp_path: Path, **overrides):
        self.diffusion_steps = 100
        self.noise_schedule = "sqrt"
        self.diffusion_timesteps = [10]
        self.device = "cpu"
        self.output_dir = str(tmp_path / "output")
        for k, v in overrides.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Mock GENIE-like model
# ---------------------------------------------------------------------------


class _MockDecoderLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        return self.linear(x)


class _MockGENIEModel(nn.Module):
    def __init__(self, num_layers: int = NUM_LAYERS, dim: int = EMBED_DIM, vocab_size: int = VOCAB_SIZE) -> None:
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

    def forward(self, x, timesteps, src_input_ids, src_attention_mask=None):
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
        assert d == {"layer_idx": 0, "baseline_loss": 3.0, "patched_loss": 3.5, "loss_delta": 0.5}

    def test_to_dict_none_layer(self) -> None:
        r = PatchingResult(layer_idx=None, baseline_loss=2.0, patched_loss=2.5, loss_delta=0.5)
        assert r.to_dict()["layer_idx"] is None

    def test_loss_delta_arithmetic(self) -> None:
        r = PatchingResult(layer_idx=1, baseline_loss=3.0, patched_loss=2.8, loss_delta=-0.2)
        assert r.loss_delta == pytest.approx(r.patched_loss - r.baseline_loss)


# ---------------------------------------------------------------------------
# ActivationPatcher.compute_baseline_loss tests
# ---------------------------------------------------------------------------


class TestComputeBaselineLoss:
    def test_returns_float(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _FakePatcherConfig(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        loss = patcher.compute_baseline_loss(dl)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_deterministic(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _FakePatcherConfig(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
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
        config = _FakePatcherConfig(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        baseline = patcher.compute_baseline_loss(dl)
        result = patcher.patch_single_layer(0, dl, baseline)

        assert isinstance(result, PatchingResult)
        assert result.layer_idx == 0
        assert result.baseline_loss == baseline

    def test_loss_delta_is_correct(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _FakePatcherConfig(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        baseline = patcher.compute_baseline_loss(dl)
        result = patcher.patch_single_layer(0, dl, baseline)
        assert result.loss_delta == pytest.approx(result.patched_loss - result.baseline_loss)

    def test_missing_sae_raises_key_error(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _FakePatcherConfig(tmp_path)
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
        config = _FakePatcherConfig(tmp_path)
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
        config = _FakePatcherConfig(tmp_path)
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
        config = _FakePatcherConfig(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        results = patcher.run_full_evaluation(dl)

        assert len(results) == 3  # 2 per-layer + 1 all-layers
        assert results[0].layer_idx == 0
        assert results[1].layer_idx == 1
        assert results[2].layer_idx is None

    def test_saves_json(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _FakePatcherConfig(tmp_path)
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

    def test_all_deltas_are_consistent(self, tmp_path: Path) -> None:
        model = _MockGENIEModel()
        config = _FakePatcherConfig(tmp_path)
        saes = _make_saes()
        dl = _make_dataloader()

        patcher = ActivationPatcher(model, saes, config, layer_accessor=_layer_accessor)
        results = patcher.run_full_evaluation(dl)

        for r in results:
            assert r.loss_delta == pytest.approx(r.patched_loss - r.baseline_loss)
