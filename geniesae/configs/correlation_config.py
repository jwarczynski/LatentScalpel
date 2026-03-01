"""Exca config for cross-model feature correlation analysis.

Computes pairwise Pearson correlation between T5 SAE features and Genie SAE
features based on co-activation patterns across a shared evaluation dataset
(XSum validation set by default).

Usage:
    uv run python main.py correlate-features configs/correlate_features.yaml

Submit to Slurm:
    uv run python main.py correlate-features configs/correlate_features.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import typing as tp
from pathlib import Path

import exca
import torch
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.correlation")


class CorrelationConfig(BaseModel):
    """Compute cross-model feature correlation from co-activation patterns."""

    # -- SAE checkpoints ------------------------------------------------------
    genie_sae_checkpoint: str = Field(min_length=1, description="Path to Genie SAE .ckpt")
    t5_sae_checkpoint: str = Field(min_length=1, description="Path to T5 SAE .ckpt")

    # -- Activation sources ---------------------------------------------------
    genie_activation_dir: str = Field(min_length=1, description="Genie activation store dir")
    t5_activation_dir: str = Field(min_length=1, description="T5 activation store dir")
    genie_layer_idx: int = Field(ge=0, description="Layer index for Genie SAE")
    t5_layer_idx: int = Field(ge=0, description="Layer index for T5 SAE")

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "xsum"
    # NOTE: dataset_split is metadata only (correlation reads from activation
    # dirs, not HF directly). Should reflect the split activations came from.
    dataset_split: str = "train"
    max_samples: int | None = Field(default=None, description="Max examples (None=all)")

    # -- Correlation ----------------------------------------------------------
    top_k_summary: int = Field(default=10, gt=0, description="Top-K features in summary")

    # -- Output ---------------------------------------------------------------
    output_dir: str = "./experiments/correlation"
    device: str = "cuda:0"

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = ("device",)

    @infra.apply
    @notify_on_completion("correlate-features")
    def apply(self) -> str:
        """Build co-activation matrices and compute Pearson correlation.

        Returns:
            Path to the output directory containing correlation_matrix.pt
            and correlation_summary.json.
        """
        from geniesae.activation_collector import ActivationStore
        from geniesae.sae_lightning import SAELightningModule

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # -- Validate and load SAE checkpoints --------------------------------
        genie_ckpt_path = Path(self.genie_sae_checkpoint)
        t5_ckpt_path = Path(self.t5_sae_checkpoint)

        if not genie_ckpt_path.exists():
            raise FileNotFoundError(
                f"Genie SAE checkpoint not found: {genie_ckpt_path}"
            )
        if not t5_ckpt_path.exists():
            raise FileNotFoundError(
                f"T5 SAE checkpoint not found: {t5_ckpt_path}"
            )

        print("[Correlation] Loading Genie SAE checkpoint...", flush=True)
        genie_module = SAELightningModule.load_trained(
            str(genie_ckpt_path), map_location=str(device)
        )
        genie_sae = genie_module.sae.to(device)
        genie_sae.eval()

        print("[Correlation] Loading T5 SAE checkpoint...", flush=True)
        t5_module = SAELightningModule.load_trained(
            str(t5_ckpt_path), map_location=str(device)
        )
        t5_sae = t5_module.sae.to(device)
        t5_sae.eval()

        # -- Load stored activations ------------------------------------------
        print("[Correlation] Loading activation stores...", flush=True)
        genie_store = ActivationStore(self.genie_activation_dir)
        t5_store = ActivationStore(self.t5_activation_dir)

        # Validate activation dimensions match SAE input dims
        genie_act_dim = genie_store.activation_dim
        t5_act_dim = t5_store.activation_dim

        if genie_act_dim != genie_sae.activation_dim:
            raise ValueError(
                f"Genie activation dim ({genie_act_dim}) does not match "
                f"Genie SAE input dim ({genie_sae.activation_dim})"
            )
        if t5_act_dim != t5_sae.activation_dim:
            raise ValueError(
                f"T5 activation dim ({t5_act_dim}) does not match "
                f"T5 SAE input dim ({t5_sae.activation_dim})"
            )

        # Determine number of examples from dataset sizes
        genie_layer_dir = Path(genie_store._base_dir) / f"layer_{self.genie_layer_idx:02d}"
        t5_layer_dir = Path(t5_store._base_dir) / f"layer_{self.t5_layer_idx:02d}"
        genie_total = sum(
            torch.load(f, map_location="cpu", weights_only=True, mmap=True).shape[0]
            for f in sorted(genie_layer_dir.glob("timestep_*.pt"))
        )
        t5_total = sum(
            torch.load(f, map_location="cpu", weights_only=True, mmap=True).shape[0]
            for f in sorted(t5_layer_dir.glob("timestep_*.pt"))
        )
        num_examples = min(genie_total, t5_total)

        if self.max_samples is not None:
            num_examples = min(num_examples, self.max_samples)

        if num_examples == 0:
            raise ValueError(
                "No examples available for correlation. "
                f"Genie dataset has {num_genie} samples, "
                f"T5 dataset has {num_t5} samples."
            )

        print(
            f"[Correlation] Processing {num_examples} examples "
            f"(Genie features: {genie_sae.dictionary_size}, "
            f"T5 features: {t5_sae.dictionary_size})",
            flush=True,
        )

        # -- Build co-activation matrices -------------------------------------
        num_genie_features = genie_sae.dictionary_size
        num_t5_features = t5_sae.dictionary_size

        batch_size = 4096

        def _encode_store_batched(
            store: "ActivationStore",
            layer_idx: int,
            sae: torch.nn.Module,
            max_rows: int,
            label: str,
        ) -> torch.Tensor:
            """Load timestep files directly and encode in batches."""
            layer_dir = Path(store._base_dir) / f"layer_{layer_idx:02d}"
            ts_files = sorted(layer_dir.glob("timestep_*.pt"))
            chunks: list[torch.Tensor] = []
            total = 0
            for ts_file in ts_files:
                tensor = torch.load(ts_file, map_location="cpu", weights_only=True)
                remaining = max_rows - total
                if remaining <= 0:
                    break
                if tensor.shape[0] > remaining:
                    tensor = tensor[:remaining]
                # Encode in batches
                for start in range(0, tensor.shape[0], batch_size):
                    end = min(start + batch_size, tensor.shape[0])
                    batch_gpu = tensor[start:end].to(device)
                    with torch.no_grad():
                        z = sae.encode(batch_gpu)
                    chunks.append((z > 0).cpu())
                total += tensor.shape[0]
                del tensor
                print(f"  {label}: {total}/{max_rows} rows", flush=True)
            return torch.cat(chunks, dim=0)

        print("[Correlation] Encoding Genie activations...", flush=True)
        genie_coact = _encode_store_batched(
            genie_store, self.genie_layer_idx, genie_sae, num_examples, "Genie",
        )
        print("[Correlation] Encoding T5 activations...", flush=True)
        t5_coact = _encode_store_batched(
            t5_store, self.t5_layer_idx, t5_sae, num_examples, "T5",
        )

        # Verify both co-activation matrices have identical first dimension
        assert genie_coact.shape[0] == t5_coact.shape[0], (
            f"Co-activation matrix row mismatch: "
            f"Genie={genie_coact.shape[0]}, T5={t5_coact.shape[0]}"
        )

        print("[Correlation] Computing Pearson correlation...", flush=True)

        # -- Normalize to float for correlation computation -------------------
        genie_float = genie_coact.float()  # (num_examples, num_genie_features)
        t5_float = t5_coact.float()  # (num_examples, num_t5_features)

        # -- Compute pairwise Pearson correlation -----------------------------
        # Shape: (num_t5_features, num_genie_features)
        correlation_matrix = _compute_pearson_correlation(t5_float, genie_float)

        # Handle zero-variance columns: replace NaN with 0.0
        nan_mask = torch.isnan(correlation_matrix)
        num_nan = nan_mask.sum().item()
        if num_nan > 0:
            logger.warning(
                "Replaced %d NaN values (zero-variance features) with 0.0",
                num_nan,
            )
            correlation_matrix = correlation_matrix.nan_to_num(nan=0.0)

        # -- Save correlation matrix ------------------------------------------
        matrix_path = out_dir / "correlation_matrix.pt"
        torch.save(correlation_matrix, matrix_path)
        print(f"[Correlation] Saved correlation matrix to {matrix_path}", flush=True)

        # -- Build and save summary JSON --------------------------------------
        summary = _build_summary(
            correlation_matrix=correlation_matrix,
            top_k=self.top_k_summary,
            genie_sae_checkpoint=self.genie_sae_checkpoint,
            t5_sae_checkpoint=self.t5_sae_checkpoint,
            num_genie_features=num_genie_features,
            num_t5_features=num_t5_features,
            num_examples=num_examples,
            dataset_split=self.dataset_split,
        )

        summary_path = out_dir / "correlation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Correlation] Saved summary to {summary_path}", flush=True)

        print(
            f"[Correlation] Done: mean={summary['statistics']['mean_correlation']:.4f}, "
            f"max={summary['statistics']['max_correlation']:.4f}",
            flush=True,
        )
        return str(out_dir)


def _compute_pearson_correlation(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise Pearson correlation between columns of a and b.

    Args:
        a: Tensor of shape (n, num_a_features), float.
        b: Tensor of shape (n, num_b_features), float.

    Returns:
        Correlation matrix of shape (num_a_features, num_b_features).
        Entries corresponding to zero-variance columns will be NaN.
    """
    # Center each column
    a_centered = a - a.mean(dim=0, keepdim=True)
    b_centered = b - b.mean(dim=0, keepdim=True)

    # Compute standard deviations
    a_std = a_centered.pow(2).sum(dim=0).sqrt()  # (num_a,)
    b_std = b_centered.pow(2).sum(dim=0).sqrt()  # (num_b,)

    # Cross-correlation: (num_a, num_b)
    numerator = a_centered.T @ b_centered

    # Denominator: outer product of stds
    denominator = a_std.unsqueeze(1) * b_std.unsqueeze(0)

    # Pearson correlation (NaN where denominator is 0)
    correlation = numerator / denominator

    return correlation


def _build_summary(
    *,
    correlation_matrix: torch.Tensor,
    top_k: int,
    genie_sae_checkpoint: str,
    t5_sae_checkpoint: str,
    num_genie_features: int,
    num_t5_features: int,
    num_examples: int,
    dataset_split: str,
) -> dict:
    """Build the correlation summary dictionary.

    Args:
        correlation_matrix: Shape (num_t5_features, num_genie_features).
        top_k: Number of top correlated features to include per direction.
        Other args: metadata fields.

    Returns:
        Summary dict matching the design doc format.
    """
    # Statistics
    stats = {
        "mean_correlation": float(correlation_matrix.mean().item()),
        "median_correlation": float(correlation_matrix.median().item()),
        "max_correlation": float(correlation_matrix.max().item()),
        "std_correlation": float(correlation_matrix.std().item()),
    }

    # Top-K T5 → Genie: for each T5 feature, find top-K Genie features
    actual_k_t2g = min(top_k, num_genie_features)
    t5_to_genie_topk: dict[str, list[dict]] = {}
    if actual_k_t2g > 0:
        vals, idxs = torch.topk(correlation_matrix, actual_k_t2g, dim=1)
        for t5_idx in range(num_t5_features):
            t5_to_genie_topk[str(t5_idx)] = [
                {
                    "genie_feature": int(idxs[t5_idx, j].item()),
                    "correlation": float(vals[t5_idx, j].item()),
                }
                for j in range(actual_k_t2g)
            ]

    # Top-K Genie → T5: for each Genie feature, find top-K T5 features
    actual_k_g2t = min(top_k, num_t5_features)
    genie_to_t5_topk: dict[str, list[dict]] = {}
    if actual_k_g2t > 0:
        # Transpose: (num_genie_features, num_t5_features)
        corr_t = correlation_matrix.T
        vals, idxs = torch.topk(corr_t, actual_k_g2t, dim=1)
        for genie_idx in range(num_genie_features):
            genie_to_t5_topk[str(genie_idx)] = [
                {
                    "t5_feature": int(idxs[genie_idx, j].item()),
                    "correlation": float(vals[genie_idx, j].item()),
                }
                for j in range(actual_k_g2t)
            ]

    return {
        "metadata": {
            "genie_sae_checkpoint": genie_sae_checkpoint,
            "t5_sae_checkpoint": t5_sae_checkpoint,
            "num_genie_features": num_genie_features,
            "num_t5_features": num_t5_features,
            "num_examples": num_examples,
            "dataset_split": dataset_split,
        },
        "statistics": stats,
        "t5_to_genie_top_k": t5_to_genie_topk,
        "genie_to_t5_top_k": genie_to_t5_topk,
    }
