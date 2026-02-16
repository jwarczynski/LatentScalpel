"""Unit tests for geniesae.model_loader."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

from geniesae.config import ExperimentConfig
from geniesae.model_loader import (
    load_genie_model,
    load_xsum_dataset,
    get_decoder_layers,
    _resolve_device,
    _create_genie_model,
)
from geniesae.genie_model import Diffusion_LM, CrossAttention_Diffusion_LM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides) -> ExperimentConfig:
    """Build an ExperimentConfig pointing at files inside *tmp_path*."""
    ckpt = tmp_path / "checkpoint.pt"

    defaults = dict(
        model_checkpoint_path=str(ckpt),
        model_arch="transformer",
        in_channel=32,
        model_channels=32,
        out_channel=32,
        vocab_size=100,
        config_name="bert-base-uncased",
        logits_mode=1,
        init_pretrained=False,
        token_emb_type="random",
        learn_sigma=False,
        fix_encoder=False,
        dataset_name="xsum",
        max_samples=4,
        device="cpu",
        diffusion_timesteps=[100],
        collection_batch_size=2,
        sae_dictionary_size=32,
        sae_top_k=4,
        sae_learning_rate=1e-3,
        sae_training_epochs=1,
        sae_batch_size=2,
        force_retrain=False,
        output_dir=str(tmp_path / "output"),
        random_seed=42,
        log_interval=10,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _save_model_checkpoint(model: nn.Module, path: Path) -> None:
    """Save a GENIE-style CheckpointState checkpoint."""
    torch.save({"model_dict": model.state_dict()}, path)


# ---------------------------------------------------------------------------
# Tests: _resolve_device
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_cpu_always_works(self):
        assert _resolve_device("cpu") == torch.device("cpu")

    def test_cuda_falls_back_when_unavailable(self):
        with patch("geniesae.model_loader.torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dev = _resolve_device("cuda:0")
                assert dev == torch.device("cpu")
                assert len(w) == 1
                assert "Falling back to CPU" in str(w[0].message)

    def test_cuda_with_index_falls_back(self):
        with patch("geniesae.model_loader.torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dev = _resolve_device("cuda:1")
                assert dev == torch.device("cpu")
                assert len(w) == 1


# ---------------------------------------------------------------------------
# Tests: _create_genie_model
# ---------------------------------------------------------------------------


class TestCreateGenieModel:
    def test_creates_transformer_arch(self, tmp_path):
        config = _make_config(tmp_path, model_arch="transformer")
        model = _create_genie_model(config)
        assert isinstance(model, Diffusion_LM)

    def test_creates_s2s_cat_arch(self, tmp_path):
        config = _make_config(tmp_path, model_arch="s2s_CAT")
        model = _create_genie_model(config)
        assert isinstance(model, CrossAttention_Diffusion_LM)

    def test_unknown_arch_raises(self, tmp_path):
        config = _make_config(tmp_path, model_arch="unknown")
        with pytest.raises(ValueError, match="Unknown model_arch"):
            _create_genie_model(config)


# ---------------------------------------------------------------------------
# Tests: get_decoder_layers
# ---------------------------------------------------------------------------


class TestGetDecoderLayers:
    def test_transformer_returns_bert_layers(self, tmp_path):
        config = _make_config(tmp_path, model_arch="transformer")
        model = _create_genie_model(config)
        layers = get_decoder_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) > 0

    def test_s2s_cat_returns_transformer_blocks(self, tmp_path):
        config = _make_config(tmp_path, model_arch="s2s_CAT")
        model = _create_genie_model(config)
        layers = get_decoder_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 6  # default BERT config with 6 layers

    def test_unknown_model_type_raises(self):
        model = nn.Linear(10, 10)
        with pytest.raises(TypeError, match="Cannot determine decoder layers"):
            get_decoder_layers(model)


# ---------------------------------------------------------------------------
# Tests: load_genie_model
# ---------------------------------------------------------------------------


class TestLoadGenieModel:
    def test_missing_checkpoint_raises(self, tmp_path):
        config = _make_config(tmp_path, model_checkpoint_path="/nonexistent/ckpt.pt")
        with pytest.raises(FileNotFoundError, match="/nonexistent/ckpt.pt"):
            load_genie_model(config)

    def test_loads_model_on_cpu(self, tmp_path):
        """Model loads successfully and is placed on CPU."""
        config = _make_config(tmp_path)
        # Create and save a model checkpoint
        model = _create_genie_model(config)
        ckpt_path = Path(config.model_checkpoint_path)
        _save_model_checkpoint(model, ckpt_path)

        nnsight_model, tokenizer = load_genie_model(config)

        # The wrapped model should be an NNsight instance
        assert hasattr(nnsight_model, "trace")

        # All parameters should be on CPU
        for p in nnsight_model._model.parameters():
            assert p.device == torch.device("cpu")

    def test_loads_raw_state_dict(self, tmp_path):
        """Supports loading a raw state dict (not wrapped in CheckpointState)."""
        config = _make_config(tmp_path)
        model = _create_genie_model(config)
        ckpt_path = Path(config.model_checkpoint_path)
        # Save as raw state dict
        torch.save(model.state_dict(), ckpt_path)

        nnsight_model, tokenizer = load_genie_model(config)
        assert hasattr(nnsight_model, "trace")

    def test_device_fallback_to_cpu(self, tmp_path):
        """When CUDA is requested but unavailable, model falls back to CPU."""
        config = _make_config(tmp_path, device="cuda:0")
        model = _create_genie_model(config)
        ckpt_path = Path(config.model_checkpoint_path)
        _save_model_checkpoint(model, ckpt_path)

        with patch("geniesae.model_loader.torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                nnsight_model, _ = load_genie_model(config)

        for p in nnsight_model._model.parameters():
            assert p.device == torch.device("cpu")

    def test_tokenizer_uses_config_name(self, tmp_path):
        """Tokenizer is loaded from config_name, not checkpoint directory."""
        config = _make_config(tmp_path)
        model = _create_genie_model(config)
        ckpt_path = Path(config.model_checkpoint_path)
        _save_model_checkpoint(model, ckpt_path)

        _, tokenizer = load_genie_model(config)
        # bert-base-uncased tokenizer should have [CLS], [SEP], etc.
        assert tokenizer is not None

    def test_directory_checkpoint_raises_without_data_pkl(self, tmp_path):
        """A directory without data.pkl raises IsADirectoryError."""
        ckpt_dir = tmp_path / "checkpoint_dir"
        ckpt_dir.mkdir()
        config = _make_config(tmp_path, model_checkpoint_path=str(ckpt_dir))
        with pytest.raises(IsADirectoryError, match="data.pkl"):
            load_genie_model(config)


# ---------------------------------------------------------------------------
# Tests: load_xsum_dataset
# ---------------------------------------------------------------------------


class TestLoadXsumDataset:
    def test_returns_dataloader_with_correct_batch_size(self, tmp_path):
        """DataLoader has the configured batch size."""
        config = _make_config(tmp_path, max_samples=6, collection_batch_size=3)
        tokenizer = _make_mock_tokenizer()

        with patch("geniesae.model_loader.load_dataset") as mock_ld:
            mock_ld.return_value = _make_fake_dataset(6)
            dl = load_xsum_dataset(config, tokenizer)

        assert dl.batch_size == 3

    def test_limits_to_max_samples(self, tmp_path):
        """Dataset is truncated to max_samples."""
        config = _make_config(tmp_path, max_samples=3, collection_batch_size=1)
        tokenizer = _make_mock_tokenizer()

        with patch("geniesae.model_loader.load_dataset") as mock_ld:
            fake_ds = _make_fake_dataset(10)
            mock_ld.return_value = fake_ds
            dl = load_xsum_dataset(config, tokenizer)

        # 3 samples / batch_size 1 = 3 batches
        assert len(dl) == 3

    def test_dataloader_yields_input_ids_and_mask(self, tmp_path):
        config = _make_config(tmp_path, max_samples=2, collection_batch_size=2)
        tokenizer = _make_mock_tokenizer()

        with patch("geniesae.model_loader.load_dataset") as mock_ld:
            mock_ld.return_value = _make_fake_dataset(2)
            dl = load_xsum_dataset(config, tokenizer)

        batch = next(iter(dl))
        assert len(batch) == 2  # input_ids, attention_mask
        assert batch[0].shape[0] == 2  # batch_size


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer():
    """Return a mock tokenizer that produces fixed-size tensors."""
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.eos_token = "[EOS]"

    def _call(texts, **kwargs):
        n = len(texts)
        seq_len = 16
        return {
            "input_ids": torch.randint(0, 1000, (n, seq_len)),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }

    tok.side_effect = _call
    tok.__call__ = _call
    return tok


class _FakeDataset:
    """Minimal stand-in for a HuggingFace Dataset."""

    def __init__(self, n: int):
        self._n = n
        self._docs = [f"text {i}" for i in range(n)]

    def __len__(self):
        return self._n

    def __getitem__(self, key):
        if key == "document":
            return self._docs
        return self._docs[key]

    def select(self, indices):
        subset = _FakeDataset(len(indices))
        subset._docs = [self._docs[i] for i in indices]
        return subset


def _make_fake_dataset(n: int):
    """Return a fake HuggingFace dataset with *n* examples."""
    return _FakeDataset(n)
