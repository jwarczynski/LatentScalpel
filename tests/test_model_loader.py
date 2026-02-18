"""Unit tests for geniesae.model_loader."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn

from geniesae.model_loader import (
    load_genie_model,
    get_decoder_layers,
    _resolve_device,
    _create_genie_model,
)
from geniesae.genie_model import Diffusion_LM, CrossAttention_Diffusion_LM


# ---------------------------------------------------------------------------
# Helpers — lightweight config stand-in (duck-typed)
# ---------------------------------------------------------------------------


class _FakeModelConfig:
    """Minimal config object with the fields model_loader expects."""

    def __init__(self, tmp_path: Path, **overrides):
        self.model_checkpoint_path = str(tmp_path / "checkpoint.pt")
        self.model_arch = "transformer"
        self.in_channel = 32
        self.model_channels = 32
        self.out_channel = 32
        self.vocab_size = 100
        self.config_name = "bert-base-uncased"
        self.logits_mode = 1
        self.init_pretrained = False
        self.token_emb_type = "random"
        self.learn_sigma = False
        self.fix_encoder = False
        self.device = "cpu"
        for k, v in overrides.items():
            setattr(self, k, v)


def _save_model_checkpoint(model: nn.Module, path: Path) -> None:
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
        config = _FakeModelConfig(tmp_path, model_arch="transformer")
        model = _create_genie_model(config)
        assert isinstance(model, Diffusion_LM)

    def test_creates_s2s_cat_arch(self, tmp_path):
        config = _FakeModelConfig(tmp_path, model_arch="s2s_CAT")
        model = _create_genie_model(config)
        assert isinstance(model, CrossAttention_Diffusion_LM)

    def test_unknown_arch_raises(self, tmp_path):
        config = _FakeModelConfig(tmp_path, model_arch="unknown")
        with pytest.raises(ValueError, match="Unknown model_arch"):
            _create_genie_model(config)


# ---------------------------------------------------------------------------
# Tests: get_decoder_layers
# ---------------------------------------------------------------------------


class TestGetDecoderLayers:
    def test_transformer_returns_bert_layers(self, tmp_path):
        config = _FakeModelConfig(tmp_path, model_arch="transformer")
        model = _create_genie_model(config)
        layers = get_decoder_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) > 0

    def test_s2s_cat_returns_transformer_blocks(self, tmp_path):
        config = _FakeModelConfig(tmp_path, model_arch="s2s_CAT")
        model = _create_genie_model(config)
        layers = get_decoder_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 6

    def test_unknown_model_type_raises(self):
        model = nn.Linear(10, 10)
        with pytest.raises(TypeError, match="Cannot determine decoder layers"):
            get_decoder_layers(model)


# ---------------------------------------------------------------------------
# Tests: load_genie_model
# ---------------------------------------------------------------------------


class TestLoadGenieModel:
    def test_missing_checkpoint_raises(self, tmp_path):
        config = _FakeModelConfig(tmp_path, model_checkpoint_path="/nonexistent/ckpt.pt")
        with pytest.raises(FileNotFoundError, match="/nonexistent/ckpt.pt"):
            load_genie_model(config)

    def test_loads_model_on_cpu(self, tmp_path):
        config = _FakeModelConfig(tmp_path)
        model = _create_genie_model(config)
        _save_model_checkpoint(model, Path(config.model_checkpoint_path))

        nnsight_model, tokenizer = load_genie_model(config)
        assert hasattr(nnsight_model, "trace")
        for p in nnsight_model._model.parameters():
            assert p.device == torch.device("cpu")

    def test_loads_raw_state_dict(self, tmp_path):
        config = _FakeModelConfig(tmp_path)
        model = _create_genie_model(config)
        torch.save(model.state_dict(), Path(config.model_checkpoint_path))

        nnsight_model, tokenizer = load_genie_model(config)
        assert hasattr(nnsight_model, "trace")

    def test_device_fallback_to_cpu(self, tmp_path):
        config = _FakeModelConfig(tmp_path, device="cuda:0")
        model = _create_genie_model(config)
        _save_model_checkpoint(model, Path(config.model_checkpoint_path))

        with patch("geniesae.model_loader.torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                nnsight_model, _ = load_genie_model(config)

        for p in nnsight_model._model.parameters():
            assert p.device == torch.device("cpu")

    def test_tokenizer_uses_config_name(self, tmp_path):
        config = _FakeModelConfig(tmp_path)
        model = _create_genie_model(config)
        _save_model_checkpoint(model, Path(config.model_checkpoint_path))

        _, tokenizer = load_genie_model(config)
        assert tokenizer is not None

    def test_directory_checkpoint_raises_without_data_pkl(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoint_dir"
        ckpt_dir.mkdir()
        config = _FakeModelConfig(tmp_path, model_checkpoint_path=str(ckpt_dir))
        with pytest.raises(IsADirectoryError, match="data.pkl"):
            load_genie_model(config)
