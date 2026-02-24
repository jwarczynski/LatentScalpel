"""Exca config for top-examples discovery stage.

Identifies the dataset examples that maximally activate each SAE feature
by running SAE inference over stored activations and maintaining per-feature
top-K heaps.

Single layer (inline)::

    uv run python main.py find-top-examples configs/find_top_examples.yaml

Submit to Slurm::

    uv run python main.py find-top-examples configs/find_top_examples.yaml \
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import heapq
import json
import logging
import typing as tp
from pathlib import Path

import exca
import torch
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.top_examples")


# ---------------------------------------------------------------------------
# Pure helper: top-K heap update (testable independently from I/O)
# ---------------------------------------------------------------------------


def update_topk_heaps(
    heaps: dict[int, list[tuple[float, int, int, int]]],
    sparse_codes: torch.Tensor,
    feature_indices: set[int],
    top_k: int,
    row_offset: int,
    seq_len: int,
    timestep: int,
    unique_examples: bool = True,
    seen_examples: dict[int, dict[int, int]] | None = None,
) -> None:
    """Update per-feature min-heaps with activations from a batch of sparse codes.

    Each heap entry is ``(activation_value, example_id, timestep, token_position)``.
    ``heapq`` maintains a min-heap so the smallest activation sits at index 0;
    we push when the heap is not full and replace when a new value exceeds the
    current minimum.

    Args:
        heaps: Mutable dict mapping feature index -> min-heap list.
        sparse_codes: Tensor of shape ``(batch_size, dictionary_size)``.
        feature_indices: Set of feature indices to track.
        top_k: Maximum heap size per feature.
        row_offset: Global row offset for this batch within the timestep
            tensor (used to compute ``example_id`` and ``token_position``).
        seq_len: Sequence length used during activation collection.
        timestep: The diffusion timestep these activations came from.
        unique_examples: If True, keep only the highest-activation entry
            per ``(feature, example_id)`` pair, guaranteeing ``top_k``
            unique dataset examples per feature.
        seen_examples: Mutable dict mapping ``feature_idx`` ->
            ``{example_id: heap_index}``.  Required when
            ``unique_examples=True``; ignored otherwise.  Callers must
            create this once and pass it across all batches/timesteps.
    """
    # Iterate over rows; for each row inspect only its non-zero features.
    # This is efficient because sparse codes have exactly K non-zero entries
    # per row (K << dictionary_size).
    for local_idx in range(sparse_codes.shape[0]):
        global_row = row_offset + local_idx
        example_id = global_row // seq_len
        token_pos = global_row % seq_len

        nonzero_feats = sparse_codes[local_idx].nonzero(as_tuple=True)[0]
        for feat_idx_t in nonzero_feats:
            feat_idx = feat_idx_t.item()
            if feat_idx not in feature_indices:
                continue
            val = sparse_codes[local_idx, feat_idx].item()
            heap = heaps[feat_idx]
            entry = (val, example_id, timestep, token_pos)

            if unique_examples and seen_examples is not None:
                feat_seen = seen_examples[feat_idx]
                if example_id in feat_seen:
                    # Already have this example — replace only if better
                    old_idx = feat_seen[example_id]
                    if old_idx < len(heap) and val > heap[old_idx][0]:
                        # Remove old entry, push new one, rebuild heap
                        heap[old_idx] = entry
                        heapq.heapify(heap)
                        # Update index mapping for all entries after heapify
                        feat_seen.clear()
                        for i, h_entry in enumerate(heap):
                            feat_seen[h_entry[1]] = i
                    continue
                # New example_id for this feature
                if len(heap) < top_k:
                    feat_seen[example_id] = len(heap)
                    heapq.heappush(heap, entry)
                elif val > heap[0][0]:
                    evicted = heapq.heapreplace(heap, entry)
                    feat_seen.pop(evicted[1], None)
                    # Rebuild index mapping after replacement
                    feat_seen.clear()
                    for i, h_entry in enumerate(heap):
                        feat_seen[h_entry[1]] = i
            else:
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif val > heap[0][0]:
                    heapq.heapreplace(heap, entry)



class TopExamplesConfig(BaseModel):
    """Discovers top-activating dataset examples for each SAE feature.

    Loads a trained SAE checkpoint and stored activations, runs
    ``TopKSAE.encode()`` over every activation batch, and maintains a
    per-feature min-heap of the top-K highest-activation entries.  The
    result is written as a JSON mapping suitable for downstream
    interpretation via ``InterpretFeaturesConfig``.
    """

    sae_checkpoint_path: str = Field(min_length=1)
    activation_dir: str = Field(min_length=1)
    layer_idx: int = Field(ge=0)
    dataset_name: str = "xsum"
    dataset_split: str = "train"
    top_k: int = Field(default=20, gt=0)
    unique_examples: bool = Field(
        default=True,
        description=(
            "If True (default), each dataset example appears at most once "
            "per feature in the top-K list (keeping the highest activation "
            "across timesteps/tokens). Set to False to allow duplicates, "
            "which reveals when the same example activates strongly at "
            "multiple timesteps."
        ),
    )
    features: list[int] | None = Field(
        default=None,
        description="Subset of feature indices to process. None = all features.",
    )
    timesteps: list[int] | None = Field(
        default=None,
        description=(
            "Diffusion timesteps to consider. None = all available "
            "timesteps from the activation directory."
        ),
    )
    output_dir: str = "./experiments/top_examples"
    device: str = "cuda:0"
    batch_size: int = Field(default=4096, gt=0)
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device",
        "batch_size",
    )

    @infra.apply
    @notify_on_completion("find-top-examples")
    def apply(self) -> str:
        """Run top-examples discovery. Returns the output file path."""
        from geniesae.sae_lightning import SAELightningModule

        # 1. Load SAE
        lightning_module = SAELightningModule.load_trained(
            self.sae_checkpoint_path, map_location=self.device,
        )
        sae = lightning_module.sae.to(self.device)
        sae.eval()
        dictionary_size = sae.dictionary_size

        print(
            f"[TopExamples] Loaded SAE: activation_dim={sae.activation_dim}, "
            f"dictionary_size={dictionary_size}, k={sae.k}",
            flush=True,
        )

        # 2. Read activation metadata
        act_dir = Path(self.activation_dir)
        meta_path = act_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {act_dir}. "
                "Run activation collection first."
            )
        with open(meta_path) as f:
            metadata = json.load(f)

        num_layers = metadata["num_layers"]
        activation_dim = metadata["activation_dim"]
        available_timesteps: list[int] = metadata["timesteps"]

        # 3. Validate layer_idx
        if self.layer_idx >= num_layers:
            raise ValueError(
                f"layer_idx={self.layer_idx} out of range "
                f"(activation store has {num_layers} layers, 0-{num_layers - 1})"
            )

        # 4. Determine timesteps to process
        if self.timesteps is not None:
            timesteps_to_use = self.timesteps
        else:
            timesteps_to_use = available_timesteps

        # 5. Determine features to process
        if self.features is not None:
            feature_indices = self.features
        else:
            feature_indices = list(range(dictionary_size))

        # Validate feature indices
        out_of_range = [f for f in feature_indices if f < 0 or f >= dictionary_size]
        if out_of_range:
            raise ValueError(
                f"Feature indices out of range [0, {dictionary_size}): {out_of_range}"
            )

        feature_set = set(feature_indices)

        # 6. Initialize per-feature heaps
        heaps: dict[int, list[tuple[float, int, int, int]]] = {
            f: [] for f in feature_indices
        }

        # Track seen example_ids per feature for uniqueness enforcement
        seen_examples: dict[int, dict[int, int]] | None = None
        if self.unique_examples:
            seen_examples = {f: {} for f in feature_indices}

        print(
            f"[TopExamples] Processing {len(timesteps_to_use)} timesteps, "
            f"{len(feature_indices)} features, top_k={self.top_k}",
            flush=True,
        )

        layer_dir = act_dir / f"layer_{self.layer_idx:02d}"
        if not layer_dir.exists():
            raise FileNotFoundError(
                f"No activation directory for layer {self.layer_idx}: {layer_dir}"
            )

        # Infer seq_len from the first timestep file and metadata.
        # Each timestep file has shape (num_examples * seq_len, activation_dim).
        # num_samples in metadata is total across all timesteps and layers,
        # so we infer seq_len from the file itself.
        seq_len: int | None = None

        # 7. Process each timestep
        for ts_idx, timestep in enumerate(timesteps_to_use):
            ts_file = layer_dir / f"timestep_{timestep:04d}.pt"
            if not ts_file.exists():
                logger.warning(
                    "Timestep file not found, skipping: %s", ts_file,
                )
                continue

            # Load full timestep tensor: (total_tokens, activation_dim)
            ts_tensor = torch.load(ts_file, map_location="cpu", weights_only=True)

            # Infer seq_len on first file if not yet known.
            # The tokenizer pads/truncates to max_length=512 in the collection
            # config, so we try to read it from metadata first.
            if seq_len is None:
                seq_len = metadata.get("seq_len")
                if seq_len is None:
                    # Heuristic: assume 512 (the default max_length in collection).
                    # This is safe because the collection pipeline always uses
                    # padding=True, truncation=True, max_length=512.
                    seq_len = 512
                    logger.info(
                        "seq_len not in metadata; defaulting to %d", seq_len,
                    )

            total_rows = ts_tensor.shape[0]

            # Process in batches
            for start in range(0, total_rows, self.batch_size):
                end = min(start + self.batch_size, total_rows)
                batch = ts_tensor[start:end].to(self.device)

                with torch.no_grad():
                    sparse_codes = sae.encode(batch)

                # Move to CPU for heap operations
                sparse_codes_cpu = sparse_codes.cpu()

                update_topk_heaps(
                    heaps=heaps,
                    sparse_codes=sparse_codes_cpu,
                    feature_indices=feature_set,
                    top_k=self.top_k,
                    row_offset=start,
                    seq_len=seq_len,
                    timestep=timestep,
                    unique_examples=self.unique_examples,
                    seen_examples=seen_examples,
                )

            if (ts_idx + 1) % 5 == 0 or ts_idx == len(timesteps_to_use) - 1:
                print(
                    f"[TopExamples] Processed timestep {timestep} "
                    f"({ts_idx + 1}/{len(timesteps_to_use)})",
                    flush=True,
                )

        # 8. Build output JSON
        features_output: dict[str, list[dict]] = {}
        for feat_idx in feature_indices:
            # Sort descending by activation value
            entries = sorted(heaps[feat_idx], key=lambda e: e[0], reverse=True)
            features_output[str(feat_idx)] = [
                {
                    "example_id": entry[1],
                    "activation": entry[0],
                    "timestep": entry[2],
                    "token_position": entry[3],
                }
                for entry in entries
            ]

        output = {
            "metadata": {
                "dataset_name": self.dataset_name,
                "dataset_split": self.dataset_split,
                "layer_idx": self.layer_idx,
                "sae_checkpoint": self.sae_checkpoint_path,
                "top_k": self.top_k,
                "num_features": len(feature_indices),
                "activation_dim": activation_dim,
                "timesteps_used": timesteps_to_use,
                "seq_len": seq_len or 512,
                "unique_examples": self.unique_examples,
            },
            "features": features_output,
        }

        # 9. Write output
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"layer_{self.layer_idx:02d}_top_examples.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[TopExamples] Wrote {out_path}", flush=True)
        return str(out_path)
