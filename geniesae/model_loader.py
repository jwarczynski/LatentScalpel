"""GENIE model and XSum dataset loading.

All public functions accept any config object that has the required
attributes (duck typing), so they work with ActivationCollectionConfig,
EvaluationConfig, or any other Pydantic model that carries the same
model fields.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Protocol

import nnsight
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from geniesae.genie_model import Diffusion_LM, CrossAttention_Diffusion_LM

logger = logging.getLogger("geniesae.model_loader")


class _ModelConfig(Protocol):
    """Structural protocol for any config that carries GENIE model fields."""

    model_checkpoint_path: str
    model_arch: str
    in_channel: int
    model_channels: int
    out_channel: int
    vocab_size: int
    config_name: str
    logits_mode: int
    init_pretrained: bool
    token_emb_type: str
    learn_sigma: bool
    fix_encoder: bool
    device: str


def _resolve_device(requested: str) -> torch.device:
    """Resolve the requested device string, falling back to CPU if CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            f"Requested device '{requested}' is unavailable (CUDA not found). "
            "Falling back to CPU.",
            stacklevel=2,
        )
        logger.warning(
            "Requested device '%s' is unavailable. Falling back to CPU.", requested
        )
        return torch.device("cpu")
    return torch.device(requested)


def _create_genie_model(config: Any) -> nn.Module:
    """Instantiate the GENIE model from config parameters (no weights loaded yet)."""
    out_channels = config.out_channel if not config.learn_sigma else config.out_channel * 2

    if config.model_arch == "transformer":
        return Diffusion_LM(
            in_channels=config.in_channel,
            model_channels=config.model_channels,
            out_channels=out_channels,
            dropout=0.1,
            config_name=config.config_name,
            vocab_size=config.vocab_size,
            logits_mode=config.logits_mode,
            init_pretrained=config.init_pretrained,
            token_emb_type=config.token_emb_type,
        )
    elif config.model_arch == "s2s_CAT":
        return CrossAttention_Diffusion_LM(
            in_channels=config.in_channel,
            model_channels=config.model_channels,
            out_channels=out_channels,
            dropout=0.1,
            config_name=config.config_name,
            vocab_size=config.vocab_size,
            logits_mode=config.logits_mode,
            init_pretrained=config.init_pretrained,
            token_emb_type=config.token_emb_type,
            fix_encoder=config.fix_encoder,
        )
    else:
        raise ValueError(f"Unknown model_arch: {config.model_arch!r}")


def load_genie_model(
    config: Any,
) -> tuple[nn.Module, PreTrainedTokenizer]:
    """Load GENIE model weights onto the configured device.

    Accepts any config object with the standard model fields
    (model_checkpoint_path, model_arch, device, config_name, etc.).

    Returns:
        A tuple of ``(nnsight_model, tokenizer)``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint_path = Path(config.model_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {config.model_checkpoint_path}"
        )

    device = _resolve_device(config.device)

    logger.info("Creating GENIE model (arch=%s)", config.model_arch)
    model = _create_genie_model(config)

    # Resolve the actual file to load.
    load_path: Path = checkpoint_path
    if checkpoint_path.is_dir():
        data_pkl = checkpoint_path / "data.pkl"
        if data_pkl.exists():
            logger.info(
                "Checkpoint path is a directory with data.pkl — "
                "loading extracted PyTorch archive from %s",
                data_pkl,
            )
            load_path = data_pkl
        else:
            raise IsADirectoryError(
                f"Checkpoint path is a directory but does not contain "
                f"data.pkl: {config.model_checkpoint_path}. "
                f"Point model_checkpoint_path at the .pt file directly, "
                f"or at the extracted directory containing data.pkl."
            )

    logger.info("Loading model checkpoint from %s", load_path)

    if load_path.name == "data.pkl":
        import pickle

        data_dir = load_path.parent / "data"

        class _DirectoryUnpickler(pickle.Unpickler):
            def persistent_load(self, saved_id):
                assert saved_id[0] == "storage"
                storage_type, key, location, numel = saved_id[1:]
                dtype = storage_type.dtype if hasattr(storage_type, "dtype") else torch.float32
                storage_file = data_dir / key
                if storage_file.exists():
                    storage = torch.UntypedStorage.from_file(
                        str(storage_file), shared=False, nbytes=numel * torch._utils._element_size(dtype)
                    )
                    return torch.storage.TypedStorage(
                        wrap_storage=storage, dtype=dtype, _internal=True
                    )
                return storage_type(numel)

        with open(load_path, "rb") as f:
            checkpoint = _DirectoryUnpickler(f).load()
    else:
        checkpoint = torch.load(
            load_path, map_location=device, weights_only=False
        )

    if isinstance(checkpoint, dict) and "model_dict" in checkpoint:
        state_dict = checkpoint["model_dict"]
    elif hasattr(checkpoint, "model_dict"):
        state_dict = checkpoint.model_dict
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unexpected checkpoint format: {type(checkpoint).__name__}. "
            "Expected a dict with 'model_dict' key, a CheckpointState "
            "namedtuple, or a raw state dict."
        )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info("Model placed on %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.config_name)
    nnsight_model = nnsight.NNsight(model)

    return nnsight_model, tokenizer


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder layer modules from a GENIE model."""
    if isinstance(model, CrossAttention_Diffusion_LM):
        return model.transformer_blocks
    elif isinstance(model, Diffusion_LM):
        return model.input_transformers.layer
    else:
        raise TypeError(
            f"Cannot determine decoder layers for model type {type(model).__name__}. "
            "Expected Diffusion_LM or CrossAttention_Diffusion_LM."
        )
