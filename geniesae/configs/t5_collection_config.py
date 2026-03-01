"""Exca config for collecting activations from a T5 encoder-decoder model.

T5 is an autoregressive encoder-decoder model (not a diffusion model), so
activations are collected at a single "timestep" (timestep 0) from the
decoder's residual stream using forward hooks.

Usage:
    uv run python main.py collect-t5-activations configs/collect_t5_activations.yaml

Submit to Slurm:
    uv run python main.py collect-t5-activations configs/collect_t5_activations.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import typing as tp
from pathlib import Path

import exca
import torch
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.t5_collection")


class T5CollectionConfig(BaseModel):
    """Collect decoder residual-stream activations from a T5 model."""

    # -- Model ----------------------------------------------------------------
    model_name: str = "sysresearch101/t5-large-finetuned-xsum"

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "xsum"
    dataset_split: str = "train"
    max_samples: int = Field(default=10000, gt=0)
    max_length: int = Field(default=512, gt=0)

    # -- Collection -----------------------------------------------------------
    layers: list[int] | None = Field(
        default=None,
        description="Decoder layer indices to collect. None = all decoder layers.",
    )
    batch_size: int = Field(default=16, gt=0)
    output_dir: str = "./experiments/t5_activations"
    force_overwrite: bool = False
    device: str = "cuda:0"
    random_seed: int = 42

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size", "force_overwrite",
    )

    @infra.apply
    @notify_on_completion("collect-t5-activations")
    def apply(self) -> str:
        """Load T5, collect decoder activations, return output dir path."""
        from datasets import load_dataset as hf_load_dataset
        from geniesae.utils import set_seed

        set_seed(self.random_seed)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        store_dir = Path(self.output_dir)

        if store_dir.exists() and self.force_overwrite:
            logger.info("Force overwrite: removing %s", store_dir)
            shutil.rmtree(store_dir)

        # -- Load model and tokenizer -----------------------------------------
        print(f"[T5Collection] Loading model {self.model_name}...", flush=True)
        try:
            from transformers import T5ForConditionalGeneration, AutoTokenizer

            model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load T5 model '{self.model_name}': {e}"
            ) from e

        model = model.to(device)
        model.eval()

        # -- Resolve layer indices --------------------------------------------
        num_decoder_blocks = len(model.decoder.block)
        if self.layers is not None:
            layer_indices = self.layers
            for li in layer_indices:
                if li < 0 or li >= num_decoder_blocks:
                    raise IndexError(
                        f"Layer index {li} is out of range for T5 decoder "
                        f"with {num_decoder_blocks} blocks (valid: 0-{num_decoder_blocks - 1})"
                    )
        else:
            layer_indices = list(range(num_decoder_blocks))

        num_layers = len(layer_indices)

        # -- Load dataset -----------------------------------------------------
        print(f"[T5Collection] Loading dataset {self.dataset_name} "
              f"split={self.dataset_split}...", flush=True)
        ds = hf_load_dataset(self.dataset_name, split=self.dataset_split)
        if self.max_samples < len(ds):
            ds = ds.select(range(self.max_samples))
        num_examples = len(ds)

        # Tokenize source (encoder input) and target (decoder input)
        source_texts: list[str] = list(ds["document"])
        target_texts: list[str] = list(ds["summary"])

        source_encodings = tokenizer(
            source_texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        target_encodings = tokenizer(
            target_texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        source_input_ids = source_encodings["input_ids"]
        source_attention_mask = source_encodings["attention_mask"]
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
            target_encodings["input_ids"]
        )

        del source_texts, target_texts, source_encodings, target_encodings, ds
        gc.collect()

        # -- Set up forward hooks ---------------------------------------------
        captured: dict[int, torch.Tensor] = {}

        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # T5 decoder block output is a tuple; first element is hidden state
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden.detach().cpu().float()
            return hook_fn

        handles = []
        for li in layer_indices:
            h = model.decoder.block[li].register_forward_hook(_make_hook(li))
            handles.append(h)

        # -- Create output directories ----------------------------------------
        store_dir.mkdir(parents=True, exist_ok=True)
        for li in layer_indices:
            (store_dir / f"layer_{li:02d}").mkdir(parents=True, exist_ok=True)

        # -- Run inference and collect activations ----------------------------
        activation_dim: int | None = None
        total_samples = 0
        batch_counter = 0

        print(f"[T5Collection] Collecting layers {layer_indices}, "
              f"{num_examples} examples, batch_size={self.batch_size}", flush=True)

        with torch.no_grad():
            for batch_start in range(0, num_examples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_examples)

                b_src_ids = source_input_ids[batch_start:batch_end].to(device)
                b_src_mask = source_attention_mask[batch_start:batch_end].to(device)
                b_dec_ids = decoder_input_ids[batch_start:batch_end].to(device)

                captured.clear()

                # Encoder-decoder forward pass
                model(
                    input_ids=b_src_ids,
                    attention_mask=b_src_mask,
                    decoder_input_ids=b_dec_ids,
                )

                # Write each layer's activations to disk
                for li in layer_indices:
                    act = captured[li]
                    if act.dim() == 3:
                        act = act.reshape(-1, act.shape[-1])

                    if activation_dim is None:
                        activation_dim = act.shape[-1]

                    total_samples += act.shape[0]
                    batch_path = (
                        store_dir / f"layer_{li:02d}"
                        / f"ts_0000_batch_{batch_counter:05d}.pt"
                    )
                    torch.save(act, batch_path)

                captured.clear()
                del b_src_ids, b_src_mask, b_dec_ids
                torch.cuda.empty_cache()
                batch_counter += 1

                if (batch_counter % 50) == 0:
                    print(f"  Processed {batch_end}/{num_examples} examples...",
                          flush=True)

        # -- Concatenate per-batch files into timestep_0000.pt ----------------
        for li in layer_indices:
            layer_dir = store_dir / f"layer_{li:02d}"
            batch_files = sorted(layer_dir.glob("ts_0000_batch_*.pt"))
            chunks = [
                torch.load(bf, map_location="cpu", weights_only=True)
                for bf in batch_files
            ]
            combined = torch.cat(chunks, dim=0)
            torch.save(combined, layer_dir / "timestep_0000.pt")
            del chunks, combined
            for bf in batch_files:
                bf.unlink()

        gc.collect()
        print("[T5Collection] Flushed timestep 0", flush=True)

        # -- Remove hooks -----------------------------------------------------
        for h in handles:
            h.remove()

        # -- Save metadata ----------------------------------------------------
        metadata = {
            "model": self.model_name,
            "num_layers": num_layers,
            "layer_indices": layer_indices,
            "activation_dim": activation_dim or 0,
            "timesteps": [0],
            "num_samples": total_samples,
            "num_examples": num_examples,
            "dataset_split": self.dataset_split,
            "dataset_name": self.dataset_name,
        }
        meta_path = store_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[T5Collection] Done: {num_layers} layers, {total_samples} samples, "
              f"dim={activation_dim}", flush=True)
        return str(store_dir)
