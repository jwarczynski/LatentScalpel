"""Exca config for evaluating SAE reconstruction impact on PLAID model.

Patches transformer block activations with SAE reconstructions during
PLAID's reverse diffusion and measures the impact on output quality.

Usage:
    uv run python main.py evaluate-plaid configs/plaid_evaluation.yaml

Submit to Slurm:
    uv run python main.py evaluate-plaid configs/plaid_evaluation.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import math
import typing as tp
from pathlib import Path

import exca
from pydantic import BaseModel, Field

from geniesae.notify import notify_on_completion

logger = logging.getLogger("geniesae.configs.plaid_evaluation")


class PlaidEvaluationConfig(BaseModel):
    """Evaluate SAE reconstruction impact on PLAID diffusion model.

    Runs iterative reverse diffusion with and without SAE patching,
    comparing cross-entropy loss at each step.
    """

    # -- Model ----------------------------------------------------------------
    weights_path: str = Field(min_length=1)
    dim: int = Field(default=2048, gt=0)
    embed_dim: int = Field(default=16, gt=0)
    n_blocks: int = Field(default=24, gt=0)
    n_heads: int = Field(default=32, gt=0)
    vocab_size: int = Field(default=32768, gt=0)
    gamma_0: float = -3.0
    gamma_1: float = 6.0

    # -- SAE checkpoints ------------------------------------------------------
    sae_checkpoint_dir: str = Field(min_length=1)
    sae_layers: list[int] | None = Field(
        default=None,
        description="Layer indices to evaluate. null -> auto-detect.",
    )
    prefer_best: bool = True

    # -- Dataset --------------------------------------------------------------
    dataset_name: str = "openwebtext"
    # NOTE: openwebtext only has a "train" split. For evaluation we should
    # ideally use a held-out set — use skip_samples to carve out a disjoint
    # slice, or switch to a dataset with a proper validation split.
    dataset_split: str = "train"
    max_samples: int = Field(default=200, gt=0)
    seq_len: int = Field(default=256, gt=0)

    # -- Evaluation -----------------------------------------------------------
    eval_mode: str = Field(
        default="iterative",
        description='"iterative" (full reverse chain) or "single_step" (per-t eval).',
    )
    sampling_timesteps: int = Field(default=256, gt=0)
    score_temp: float = 0.9
    batch_size: int = Field(default=4, gt=0)
    device: str = "cuda:0"

    # -- Single-step config ---------------------------------------------------
    eval_t_values: list[float] = Field(
        default=[0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95],
        description="Continuous t values for single-step evaluation.",
    )

    # -- Logging --------------------------------------------------------------
    use_wandb: bool = True
    wandb_project: str = "plaid-sae"
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None

    # -- Output ---------------------------------------------------------------
    output_dir: str = "./experiments/plaid_results"

    # -- Exca -----------------------------------------------------------------
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "device", "batch_size", "use_wandb", "wandb_project",
        "wandb_run_name", "wandb_run_id",
    )

    def _discover_sae_checkpoints(self) -> dict[int, str]:
        """Find available SAE checkpoint files."""
        import re

        sae_dir = Path(self.sae_checkpoint_dir)
        if not sae_dir.is_dir():
            raise FileNotFoundError(f"SAE checkpoint dir not found: {sae_dir}")

        pattern = re.compile(r"layer_(\d+)(_best)?\.ckpt$")
        found: dict[int, dict[str, str]] = {}

        for f in sorted(sae_dir.iterdir()):
            m = pattern.match(f.name)
            if m:
                layer_idx = int(m.group(1))
                kind = "best" if m.group(2) else "last"
                found.setdefault(layer_idx, {})[kind] = str(f)

        if self.sae_layers is not None:
            found = {k: v for k, v in found.items() if k in self.sae_layers}

        result: dict[int, str] = {}
        for layer_idx, paths in sorted(found.items()):
            if self.prefer_best and "best" in paths:
                result[layer_idx] = paths["best"]
            elif "last" in paths:
                result[layer_idx] = paths["last"]
            elif "best" in paths:
                result[layer_idx] = paths["best"]

        if not result:
            raise FileNotFoundError(
                f"No SAE checkpoints found in {sae_dir} (layers={self.sae_layers})"
            )
        return result

    @infra.apply
    @notify_on_completion("evaluate-plaid")
    def apply(self) -> dict:
        """Run PLAID evaluation and return results dict."""
        import gc

        import torch
        import torch.nn.functional as F

        from geniesae.plaid_model import PlaidDiffusionHelper, load_plaid_modules
        from geniesae.sae import TopKSAE
        from geniesae.sae_lightning import SAELightningModule
        from geniesae.utils import set_seed

        set_seed(42)
        torch.set_float32_matmul_precision("high")
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        # --- Load PLAID model ---
        print("[PlaidEval] Loading PLAID model...", flush=True)
        modules = load_plaid_modules(
            self.weights_path,
            dim=self.dim, embed_dim=self.embed_dim,
            n_blocks=self.n_blocks, n_heads=self.n_heads,
            vocab_size=self.vocab_size,
            gamma_0=self.gamma_0, gamma_1=self.gamma_1,
            device=str(device),
        )
        model = modules["model"]
        helper = PlaidDiffusionHelper(
            modules,
            sampling_timesteps=self.sampling_timesteps,
            score_temp=self.score_temp,
        )

        # --- Load SAEs ---
        ckpt_map = self._discover_sae_checkpoints()
        print(f"[PlaidEval] SAE checkpoints: {sorted(ckpt_map.keys())}", flush=True)

        saes: dict[int, TopKSAE] = {}
        for layer_idx, ckpt_path in ckpt_map.items():
            lm = SAELightningModule.load_trained(ckpt_path, map_location=str(device))
            saes[layer_idx] = lm.sae.to(device)
            saes[layer_idx].eval()
            print(f"[PlaidEval] Loaded SAE layer {layer_idx} from {ckpt_path}", flush=True)

        # --- Load dataset ---
        print(f"[PlaidEval] Loading dataset {self.dataset_name}...", flush=True)
        from datasets import load_dataset as hf_load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = hf_load_dataset(self.dataset_name, split=self.dataset_split, streaming=True)
        texts: list[str] = []
        for i, row in enumerate(ds):
            if i >= self.max_samples:
                break
            texts.append(row["text"])

        encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.seq_len, return_tensors="pt",
        )
        input_ids_all = encodings["input_ids"]
        del texts, encodings
        gc.collect()
        print(f"[PlaidEval] {input_ids_all.shape[0]} samples loaded", flush=True)

        # --- WandB ---
        wandb_run = None
        if self.use_wandb:
            import wandb
            mode_tag = "iter" if self.eval_mode == "iterative" else "ss"
            run_name = self.wandb_run_name or (
                f"plaid_{mode_tag}_" + "_".join(f"L{l}" for l in sorted(saes.keys()))
            )
            wkw: dict = dict(project=self.wandb_project, name=run_name)
            if self.wandb_run_id:
                wkw["id"] = self.wandb_run_id
                wkw["resume"] = "must"
            wandb_run = wandb.init(**wkw)

        # --- Run evaluation ---
        if self.eval_mode == "iterative":
            results = self._run_iterative(
                model, helper, saes, input_ids_all, device, wandb_run,
            )
        else:
            results = self._run_single_step(
                model, modules, saes, input_ids_all, device, wandb_run,
            )

        # --- Save ---
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "plaid_evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[PlaidEval] Results saved to {results_path}", flush=True)

        if wandb_run is not None:
            wandb_run.finish()
        return results

    def _patch_activations(
        self,
        model,
        saes: dict[int, "TopKSAE"],
        layer_set: dict[int, "TopKSAE"],
    ) -> list:
        """Register forward hooks that replace block outputs with SAE reconstructions."""
        handles = []

        def _make_hook(sae):
            def hook_fn(module, input, output):
                with torch.no_grad():
                    x = output.float()
                    orig_shape = x.shape
                    flat = x.reshape(-1, x.shape[-1])
                    reconstructed, _ = sae(flat)
                    return reconstructed.reshape(orig_shape).to(output.dtype)
            return hook_fn

        import torch
        for layer_idx, sae in layer_set.items():
            h = model.blocks[layer_idx].register_forward_hook(_make_hook(sae))
            handles.append(h)
        return handles

    def _compute_ce_loss(self, logits, input_ids, device) -> float:
        """Compute cross-entropy loss between model logits and input tokens."""
        import torch
        import torch.nn.functional as F
        # logits: (bs, seq_len, vocab_size), input_ids: (bs, seq_len)
        ids = input_ids.to(device).clamp(max=logits.shape[-1] - 1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ids.reshape(-1),
            reduction="mean",
        )
        return loss.item()

    def _reverse_step(
        self, z, x_reconst_raw, gamma_t, gamma_s, t_val, score_temp,
    ):
        """Inline VDM reverse step (t -> s) using already-computed x_reconst.

        Avoids calling p_sample_step (which does its own forward pass).
        """
        import torch

        alpha_squared_t = torch.sigmoid(-gamma_t)
        sigma_squared_t = torch.sigmoid(gamma_t)
        alpha_squared_s = torch.sigmoid(-gamma_s)
        alpha_t = alpha_squared_t.sqrt()
        sigma_t = sigma_squared_t.sqrt()

        x_reconst = x_reconst_raw.double()

        # Score temperature
        epsilon_pred = (z.double() - alpha_t[:, None, None] * x_reconst) / sigma_t[:, None, None]
        epsilon_pred /= score_temp
        x_reconst = (z.double() - sigma_t[:, None, None] * epsilon_pred) / alpha_t[:, None, None]

        if t_val > 0:
            c = -torch.expm1(gamma_s - gamma_t)  # (bs,)
            c = c[:, None, None]  # (bs, 1, 1)
            alpha_s = alpha_squared_s.sqrt()[:, None, None]
            alpha_t_exp = alpha_squared_t.sqrt()[:, None, None]
            one_minus_alpha_sq_s = (1 - alpha_squared_s)[:, None, None]

            z_new = (1 - c) * alpha_s / alpha_t_exp * z.double()
            z_new += c * alpha_s * x_reconst
            z_new += (c * one_minus_alpha_sq_s).sqrt() * torch.randn_like(z).double()
            z = z_new.float()

        return z

    def _run_iterative(self, model, helper, saes, input_ids_all, device, wandb_run):
        """Full reverse diffusion chain evaluation.

        Runs the model forward once per chain per step (baseline + per-layer
        patched), captures logits for loss, and performs the VDM reverse step
        inline using the same x_reconst — no double forward pass.
        """
        import torch

        layer_indices = sorted(saes.keys())
        T = self.sampling_timesteps
        embedding_matrix_module = helper.embedding_matrix_module
        score_temp = helper.score_temp

        print(f"[PlaidEval:iterative] T={T}, layers={layer_indices}", flush=True)

        bl_losses: dict[int, list[float]] = {}
        patched_losses: dict[int, dict[int, list[float]]] = {}
        n_batches = 0

        for batch_start in range(0, input_ids_all.shape[0], self.batch_size):
            batch_end = min(batch_start + self.batch_size, input_ids_all.shape[0])
            b_ids = input_ids_all[batch_start:batch_end].to(device)
            bs = b_ids.shape[0]

            embedding_matrix = embedding_matrix_module()
            b_ids_clamped = b_ids.clamp(max=self.vocab_size - 1)
            x_embed = embedding_matrix[b_ids_clamped]

            # Initialize from pure noise
            z_bl = torch.randn_like(x_embed.float())
            z_patched = {li: z_bl.clone() for li in layer_indices}
            x_sc_bl = torch.zeros_like(z_bl)
            x_sc_patched = {li: torch.zeros_like(z_bl) for li in layer_indices}

            log_interval = max(1, T // 20)

            for step_idx in range(T):
                t_val = 1.0 - step_idx / T
                s_val = max(1.0 - (step_idx + 1) / T, 0.0)

                # Batch tensors for get_gamma / model forward
                t_batch = torch.full((bs,), t_val, device=device)
                s_batch = torch.full((bs,), s_val, device=device)
                gamma_t = helper.get_gamma(t_batch)
                gamma_s = helper.get_gamma(s_batch)

                # --- Baseline (no patching) ---
                with torch.no_grad():
                    logits_bl, x_reconst_bl = model(
                        z_bl.float(), gamma_t.float(), embedding_matrix,
                        1.0, x_sc_bl,
                    )
                    x_sc_bl = x_reconst_bl.clone().detach()
                    z_bl = self._reverse_step(
                        z_bl, x_reconst_bl, gamma_t, gamma_s, t_val, score_temp,
                    )

                # --- Per-layer patched ---
                logits_patched = {}
                for li in layer_indices:
                    hooks = self._patch_activations(model, saes, {li: saes[li]})
                    with torch.no_grad():
                        logits_p, x_reconst_p = model(
                            z_patched[li].float(), gamma_t.float(),
                            embedding_matrix, 1.0, x_sc_patched[li],
                        )
                        x_sc_patched[li] = x_reconst_p.clone().detach()
                        z_patched[li] = self._reverse_step(
                            z_patched[li], x_reconst_p, gamma_t, gamma_s,
                            t_val, score_temp,
                        )
                    for h in hooks:
                        h.remove()
                    logits_patched[li] = logits_p

                # --- Log losses ---
                if step_idx % log_interval == 0 or step_idx == T - 1:
                    bl_loss = self._compute_ce_loss(logits_bl, b_ids, device)
                    bl_losses.setdefault(step_idx, []).append(bl_loss)

                    patched_losses.setdefault(step_idx, {})
                    for li in layer_indices:
                        pl = self._compute_ce_loss(logits_patched[li], b_ids, device)
                        patched_losses[step_idx].setdefault(li, []).append(pl)

            n_batches += 1
            print(
                f"[PlaidEval:iterative] Batch {n_batches} done "
                f"({batch_end}/{input_ids_all.shape[0]})",
                flush=True,
            )

        # --- Aggregate ---
        def _avg(vals):
            return sum(vals) / len(vals) if vals else 0.0

        per_step = []
        for step_idx in sorted(bl_losses.keys()):
            bl = _avg(bl_losses[step_idx])
            entry = {
                "step": step_idx,
                "t_value": round(1.0 - step_idx / T, 4),
                "baseline_loss": round(bl, 6),
                "baseline_ppl": round(math.exp(min(bl, 100)), 2),
                "per_layer": {},
            }
            log_d = {
                "step": step_idx,
                "iterative/baseline_loss": bl,
            }
            for li in layer_indices:
                pl = _avg(patched_losses.get(step_idx, {}).get(li, []))
                delta = pl - bl
                entry["per_layer"][li] = {
                    "patched_loss": round(pl, 6),
                    "loss_delta": round(delta, 6),
                }
                log_d[f"iterative/layer_{li:02d}_loss"] = pl
                log_d[f"iterative/layer_{li:02d}_delta"] = delta

            per_step.append(entry)
            if wandb_run is not None:
                wandb_run.log(log_d, step=step_idx)

        # Final step
        final_step = max(bl_losses.keys())
        final_bl = _avg(bl_losses[final_step])
        aggregate = {
            "final_baseline_loss": round(final_bl, 6),
            "per_layer": {},
        }
        for li in layer_indices:
            fl = _avg(patched_losses.get(final_step, {}).get(li, []))
            aggregate["per_layer"][li] = {
                "final_patched_loss": round(fl, 6),
                "final_loss_delta": round(fl - final_bl, 6),
            }

        if wandb_run is not None:
            wandb_run.log({
                "iterative_summary/final_baseline_loss": final_bl,
                **{f"iterative_summary/layer_{li:02d}_delta": aggregate["per_layer"][li]["final_loss_delta"]
                   for li in layer_indices},
            })

        print(f"[PlaidEval:iterative] Done. Final baseline loss={final_bl:.4f}", flush=True)
        return {"eval_mode": "iterative", "per_step": per_step, "aggregate": aggregate}

    def _run_single_step(self, model, modules, saes, input_ids_all, device, wandb_run):
        """Per-timestep independent evaluation."""
        import torch

        layer_indices = sorted(saes.keys())
        embedding_matrix_module = modules["embedding_matrix"]
        noise_schedule = modules["noise_schedule"]
        gamma_bounds = modules["gamma_bounds"]

        print(
            f"[PlaidEval:single_step] t_values={self.eval_t_values}, "
            f"layers={layer_indices}",
            flush=True,
        )

        per_t_results = []

        for t_val in self.eval_t_values:
            bl_losses_t = []
            patched_losses_t: dict[int, list[float]] = {li: [] for li in layer_indices}

            for batch_start in range(0, input_ids_all.shape[0], self.batch_size):
                batch_end = min(batch_start + self.batch_size, input_ids_all.shape[0])
                b_ids = input_ids_all[batch_start:batch_end].to(device)
                bs = b_ids.shape[0]

                embedding_matrix = embedding_matrix_module()
                b_ids_clamped = b_ids.clamp(max=self.vocab_size - 1)
                x_embed = embedding_matrix[b_ids_clamped]

                # Forward diffusion to noise level t
                t_tensor = torch.full((bs,), t_val, device=device)
                gamma_0, gamma_1 = gamma_bounds()
                gamma_norm = noise_schedule(t_tensor).double()
                gamma = gamma_0 + (gamma_1 - gamma_0) * gamma_norm

                alpha_sq = torch.sigmoid(-gamma)[:, None, None]
                sigma_sq = torch.sigmoid(gamma)[:, None, None]
                noise = torch.randn_like(x_embed.double())
                z = (alpha_sq.sqrt() * x_embed.double() + sigma_sq.sqrt() * noise).float()
                x_selfcond = torch.zeros_like(z)

                # Baseline
                with torch.no_grad():
                    logits_bl, _ = model(
                        z, gamma.float(), embedding_matrix, 1.0, x_selfcond,
                    )
                bl_losses_t.append(self._compute_ce_loss(logits_bl, b_ids, device))

                # Per-layer patched
                for li in layer_indices:
                    hooks = self._patch_activations(model, saes, {li: saes[li]})
                    with torch.no_grad():
                        logits_p, _ = model(
                            z, gamma.float(), embedding_matrix, 1.0, x_selfcond,
                        )
                    for h in hooks:
                        h.remove()
                    patched_losses_t[li].append(
                        self._compute_ce_loss(logits_p, b_ids, device)
                    )

            bl = sum(bl_losses_t) / len(bl_losses_t)
            entry = {
                "t_value": t_val,
                "baseline_loss": round(bl, 6),
                "per_layer": {},
            }
            log_d = {"t_value": t_val, "single_step/baseline_loss": bl}

            for li in layer_indices:
                pl = sum(patched_losses_t[li]) / len(patched_losses_t[li])
                entry["per_layer"][li] = {
                    "patched_loss": round(pl, 6),
                    "loss_delta": round(pl - bl, 6),
                }
                log_d[f"single_step/layer_{li:02d}_loss"] = pl
                log_d[f"single_step/layer_{li:02d}_delta"] = pl - bl

            per_t_results.append(entry)
            if wandb_run is not None:
                wandb_run.log(log_d)

            print(f"  t={t_val:.2f}: baseline_loss={bl:.4f}", flush=True)

        print("[PlaidEval:single_step] Done.", flush=True)
        return {"eval_mode": "single_step", "per_t": per_t_results}
