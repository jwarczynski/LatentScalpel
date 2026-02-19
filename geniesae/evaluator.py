"""Evaluation: measure SAE reconstruction impact on GENIE model loss/perplexity.

Two evaluation modes:

1. **Single-step** (``eval_mode="single_step"``):
   For each timestep independently, noise clean embeddings to that level,
   run one forward pass with/without SAE patching, compare loss.

2. **Iterative** (``eval_mode="iterative"``):
   Run the full reverse diffusion chain from t=T-1 down to t=0.
   At each step, the model predicts x_0, we compute loss against
   the target tokens, then update latents via the posterior
   q(x_{t-1} | x_t, x_0_pred).  Two chains are run in parallel:
   baseline (no patching) and patched (SAE reconstructions).
   Per-step and final losses are logged.

Supports single-layer patching, per-layer patching, and all-layers-at-once.
Reports per-timestep and aggregate metrics, logs to WandB and saves JSON.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from geniesae.genie_model import DiffusionHelper
from geniesae.sae import TopKSAE

logger = logging.getLogger("geniesae.evaluator")


class Evaluator:
    """Evaluates SAE patching impact across diffusion timesteps.

    Args:
        model: The GENIE model (unwrapped nn.Module).
        saes: Mapping from layer index to trained TopKSAE.
        config: Config with diffusion_steps, noise_schedule, device, etc.
        timesteps: For single-step mode: which timesteps to evaluate.
            For iterative mode: which steps to log intermediate losses at.
    """

    def __init__(
        self,
        model: nn.Module,
        saes: dict[int, TopKSAE],
        config: Any,
        timesteps: list[int] | None = None,
    ) -> None:
        from nnsight import NNsight

        self._model = model
        self._saes = saes
        self._config = config
        self._timesteps = timesteps or config.diffusion_timesteps
        self._nnsight_model = NNsight(model)
        self._diffusion = DiffusionHelper(
            num_timesteps=config.diffusion_steps,
            schedule_name=config.noise_schedule,
        )

    def _get_layers(self):
        """Return NNsight envoy proxies for the decoder layers."""
        if hasattr(self._nnsight_model, "transformer_blocks"):
            return self._nnsight_model.transformer_blocks
        return self._nnsight_model.decoder.layers

    def _compute_loss(
        self, model_output: torch.Tensor, target_ids: torch.Tensor,
    ) -> float:
        """Cross-entropy loss from GENIE denoised output (x_0 prediction)."""
        with torch.no_grad():
            logits = self._model.get_logits(model_output)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="mean",
            )
        return loss.item()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_baseline(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single unpatched forward pass, return model output (x_0 pred)."""
        saved = []
        with self._nnsight_model.trace(x_t, t_tensor, input_ids, attention_mask):
            saved.append(self._nnsight_model.output.save())
        output = saved[0].value if hasattr(saved[0], "value") else saved[0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    @torch.no_grad()
    def _forward_patched(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_layers: dict[int, TopKSAE],
    ) -> torch.Tensor:
        """Run a single patched forward pass, return model output (x_0 pred)."""
        layers = self._get_layers()
        saved = []
        with self._nnsight_model.trace(x_t, t_tensor, input_ids, attention_mask):
            for li, sae in patch_layers.items():
                layer_out = layers[li].output
                if isinstance(layer_out, (tuple, list)):
                    act = layer_out[0]
                else:
                    act = layer_out
                reconstructed, _ = sae(act)
                if isinstance(layer_out, (tuple, list)):
                    layers[li].output = (reconstructed, *layer_out[1:])
                else:
                    layers[li].output = reconstructed
            saved.append(self._nnsight_model.output.save())
        output = saved[0].value if hasattr(saved[0], "value") else saved[0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    # ------------------------------------------------------------------
    # Single-step evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_baseline(self, dataloader: DataLoader, timestep: int) -> float:
        device = self._config.device
        total_loss, n = 0.0, 0
        for batch in dataloader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            x_start = self._model.get_embeds(input_ids)
            t = torch.full((input_ids.shape[0],), timestep, dtype=torch.long, device=device)
            x_noised = self._diffusion.q_sample(x_start, t)
            output = self._forward_baseline(x_noised, t, input_ids, attention_mask)
            total_loss += self._compute_loss(output, input_ids)
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _run_patched(
        self, dataloader: DataLoader, timestep: int, patch_layers: dict[int, TopKSAE],
    ) -> float:
        device = self._config.device
        total_loss, n = 0.0, 0
        for batch in dataloader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            x_start = self._model.get_embeds(input_ids)
            t = torch.full((input_ids.shape[0],), timestep, dtype=torch.long, device=device)
            x_noised = self._diffusion.q_sample(x_start, t)
            output = self._forward_patched(x_noised, t, input_ids, attention_mask, patch_layers)
            total_loss += self._compute_loss(output, input_ids)
            n += 1
        return total_loss / max(n, 1)

    def run_single_step(
        self, dataloader: DataLoader, wandb_run: Any | None = None,
    ) -> dict:
        """Single-step evaluation across configured timesteps."""
        self._model.eval()
        for sae in self._saes.values():
            sae.eval()

        timesteps = self._timesteps
        layer_indices = sorted(self._saes.keys())

        print(
            f"[Eval:single_step] layers={layer_indices}, "
            f"{len(timesteps)} timesteps",
            flush=True,
        )

        agg_bl = 0.0
        agg_patched: dict[int, float] = {li: 0.0 for li in layer_indices}
        agg_all = 0.0
        per_ts: list[dict] = []

        for ts_idx, timestep in enumerate(timesteps):
            print(f"[Eval] Timestep {timestep} ({ts_idx+1}/{len(timesteps)})", flush=True)

            bl = self._run_baseline(dataloader, timestep)
            bl_ppl = math.exp(min(bl, 100))

            ts_r: dict = {
                "timestep": timestep,
                "baseline_loss": round(bl, 6),
                "baseline_ppl": round(bl_ppl, 2),
                "per_layer": {},
            }
            log_d: dict = {"timestep": timestep, "baseline/loss": bl, "baseline/ppl": bl_ppl}

            for li in layer_indices:
                pl = self._run_patched(dataloader, timestep, {li: self._saes[li]})
                pp = math.exp(min(pl, 100))
                d = pl - bl
                ts_r["per_layer"][li] = {
                    "patched_loss": round(pl, 6), "patched_ppl": round(pp, 2),
                    "loss_delta": round(d, 6),
                }
                log_d[f"layer_{li:02d}/patched_loss"] = pl
                log_d[f"layer_{li:02d}/patched_ppl"] = pp
                log_d[f"layer_{li:02d}/loss_delta"] = d
                agg_patched[li] += pl

            if len(layer_indices) > 1:
                ap = self._run_patched(dataloader, timestep, self._saes)
                app = math.exp(min(ap, 100))
                ad = ap - bl
                ts_r["all_layers"] = {
                    "patched_loss": round(ap, 6), "patched_ppl": round(app, 2),
                    "loss_delta": round(ad, 6),
                }
                log_d["all_layers/patched_loss"] = ap
                log_d["all_layers/patched_ppl"] = app
                log_d["all_layers/loss_delta"] = ad
                agg_all += ap

            agg_bl += bl
            per_ts.append(ts_r)
            if wandb_run is not None:
                wandb_run.log(log_d, step=ts_idx)

        n_ts = len(timesteps)
        avg_bl = agg_bl / n_ts
        avg_bl_ppl = math.exp(min(avg_bl, 100))
        aggregate: dict = {
            "avg_baseline_loss": round(avg_bl, 6),
            "avg_baseline_ppl": round(avg_bl_ppl, 2),
            "per_layer": {},
        }
        summary: dict = {"summary/avg_baseline_loss": avg_bl, "summary/avg_baseline_ppl": avg_bl_ppl}

        for li in layer_indices:
            al = agg_patched[li] / n_ts
            ap = math.exp(min(al, 100))
            ad = al - avg_bl
            aggregate["per_layer"][li] = {
                "avg_patched_loss": round(al, 6), "avg_patched_ppl": round(ap, 2),
                "avg_loss_delta": round(ad, 6),
            }
            summary[f"summary/layer_{li:02d}_avg_loss"] = al
            summary[f"summary/layer_{li:02d}_avg_ppl"] = ap
            summary[f"summary/layer_{li:02d}_avg_delta"] = ad

        if len(layer_indices) > 1:
            aa = agg_all / n_ts
            aap = math.exp(min(aa, 100))
            aggregate["all_layers"] = {
                "avg_patched_loss": round(aa, 6), "avg_patched_ppl": round(aap, 2),
                "avg_loss_delta": round(aa - avg_bl, 6),
            }
            summary["summary/all_layers_avg_loss"] = aa
            summary["summary/all_layers_avg_ppl"] = aap

        if wandb_run is not None:
            wandb_run.log(summary)

        print(f"[Eval] Done. Avg baseline loss={avg_bl:.4f}, ppl={avg_bl_ppl:.2f}", flush=True)
        for li in layer_indices:
            info = aggregate["per_layer"][li]
            print(
                f"[Eval] Layer {li}: avg_loss={info['avg_patched_loss']:.4f}, "
                f"ppl={info['avg_patched_ppl']:.2f}, delta={info['avg_loss_delta']:.4f}",
                flush=True,
            )

        return {
            "eval_mode": "single_step",
            "per_timestep": per_ts,
            "aggregate": aggregate,
            "config": {
                "sae_layers": layer_indices, "timesteps": timesteps,
                "dataset_split": self._config.dataset_split,
                "max_samples": self._config.max_samples,
            },
        }

    # ------------------------------------------------------------------
    # Iterative reverse-diffusion evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_iterative(
        self, dataloader: DataLoader, wandb_run: Any | None = None,
    ) -> dict:
        """Full reverse-chain evaluation.

        For each batch, runs the complete denoising chain t=T-1 → 0.
        Maintains parallel chains: one baseline and one per SAE layer
        (plus all-layers if multiple SAEs).

        At each step:
        1. Model predicts x_0 from current latents x_t.
        2. Loss is computed from x_0 prediction vs target tokens.
        3. Latents are updated: x_{t-1} ~ q(x_{t-1} | x_t, x_0_pred).

        Logs per-step loss and final loss to WandB.
        """
        self._model.eval()
        for sae in self._saes.values():
            sae.eval()

        device = self._config.device
        layer_indices = sorted(self._saes.keys())
        T = self._diffusion.num_timesteps
        log_steps = set(self._timesteps) if self._timesteps else set(range(T))
        multi_layer = len(layer_indices) > 1

        print(
            f"[Eval:iterative] Full reverse chain T={T}, "
            f"layers={layer_indices}, logging {len(log_steps)} steps",
            flush=True,
        )

        # Accumulators: {timestep: [loss_per_batch]}
        bl_losses: dict[int, list[float]] = {}
        patched_losses: dict[int, dict[int, list[float]]] = {}  # t -> {layer -> [losses]}
        all_patched_losses: dict[int, list[float]] = {}

        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            bs = input_ids.shape[0]

            # Initialize all chains from the same pure noise
            noise = torch.randn(bs, input_ids.shape[1], self._config.in_channel, device=device)
            # Actually, we need the embedding dim. Get it from the model.
            x_start_dummy = self._model.get_embeds(input_ids)
            noise = torch.randn_like(x_start_dummy)

            x_bl = noise.clone()
            x_per_layer = {li: noise.clone() for li in layer_indices}
            x_all = noise.clone() if multi_layer else None

            for t in reversed(range(T)):
                t_tensor = torch.full((bs,), t, dtype=torch.long, device=device)

                # Baseline chain
                out_bl = self._forward_baseline(x_bl, t_tensor, input_ids, attention_mask)
                x_bl = self._diffusion.p_sample(out_bl, x_bl, t_tensor)

                # Per-layer patched chains
                out_per_layer: dict[int, torch.Tensor] = {}
                for li in layer_indices:
                    out_li = self._forward_patched(
                        x_per_layer[li], t_tensor,
                        input_ids, attention_mask, {li: self._saes[li]},
                    )
                    x_per_layer[li] = self._diffusion.p_sample(out_li, x_per_layer[li], t_tensor)
                    out_per_layer[li] = out_li

                # All-layers patched chain
                out_all = None
                if x_all is not None:
                    out_all = self._forward_patched(
                        x_all, t_tensor, input_ids, attention_mask, self._saes,
                    )
                    x_all = self._diffusion.p_sample(out_all, x_all, t_tensor)

                # Log losses at selected timesteps
                if t in log_steps:
                    bl_loss = self._compute_loss(out_bl, input_ids)
                    bl_losses.setdefault(t, []).append(bl_loss)

                    patched_losses.setdefault(t, {})
                    for li in layer_indices:
                        pl = self._compute_loss(out_per_layer[li], input_ids)
                        patched_losses[t].setdefault(li, []).append(pl)

                    if out_all is not None:
                        al = self._compute_loss(out_all, input_ids)
                        all_patched_losses.setdefault(t, []).append(al)

            # Also log the final x_0 prediction (t=0 output) explicitly
            # (already captured above if 0 is in log_steps)

            n_batches += 1
            if (batch_idx + 1) % 4 == 0 or batch_idx == 0:
                print(
                    f"[Eval:iterative] Batch {batch_idx+1}/{len(dataloader)} done",
                    flush=True,
                )

        # --- Aggregate results ---
        def _avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        per_step_results: list[dict] = []
        sorted_steps = sorted(bl_losses.keys(), reverse=True)  # T-1 down to 0

        for step_idx, t in enumerate(sorted_steps):
            bl = _avg(bl_losses[t])
            bl_ppl = math.exp(min(bl, 100))

            step_r: dict = {
                "timestep": t,
                "baseline_loss": round(bl, 6),
                "baseline_ppl": round(bl_ppl, 2),
                "per_layer": {},
            }
            log_d: dict = {
                "timestep": t,
                "iterative/baseline_loss": bl,
                "iterative/baseline_ppl": bl_ppl,
            }

            for li in layer_indices:
                pl = _avg(patched_losses.get(t, {}).get(li, []))
                pp = math.exp(min(pl, 100))
                delta = pl - bl
                step_r["per_layer"][li] = {
                    "patched_loss": round(pl, 6),
                    "patched_ppl": round(pp, 2),
                    "loss_delta": round(delta, 6),
                }
                log_d[f"iterative/layer_{li:02d}_loss"] = pl
                log_d[f"iterative/layer_{li:02d}_ppl"] = pp
                log_d[f"iterative/layer_{li:02d}_delta"] = delta

            if multi_layer and t in all_patched_losses:
                al = _avg(all_patched_losses[t])
                alp = math.exp(min(al, 100))
                step_r["all_layers"] = {
                    "patched_loss": round(al, 6),
                    "patched_ppl": round(alp, 2),
                    "loss_delta": round(al - bl, 6),
                }
                log_d["iterative/all_layers_loss"] = al
                log_d["iterative/all_layers_ppl"] = alp
                log_d["iterative/all_layers_delta"] = al - bl

            per_step_results.append(step_r)
            if wandb_run is not None:
                wandb_run.log(log_d, step=step_idx)

        # Final step (t=0) is the headline metric
        final_bl = _avg(bl_losses.get(0, []))
        final_bl_ppl = math.exp(min(final_bl, 100))

        aggregate: dict = {
            "final_baseline_loss": round(final_bl, 6),
            "final_baseline_ppl": round(final_bl_ppl, 2),
            "per_layer": {},
        }
        summary: dict = {
            "iterative_summary/final_baseline_loss": final_bl,
            "iterative_summary/final_baseline_ppl": final_bl_ppl,
        }

        for li in layer_indices:
            fl = _avg(patched_losses.get(0, {}).get(li, []))
            fp = math.exp(min(fl, 100))
            fd = fl - final_bl
            aggregate["per_layer"][li] = {
                "final_patched_loss": round(fl, 6),
                "final_patched_ppl": round(fp, 2),
                "final_loss_delta": round(fd, 6),
            }
            summary[f"iterative_summary/layer_{li:02d}_final_loss"] = fl
            summary[f"iterative_summary/layer_{li:02d}_final_delta"] = fd

        if multi_layer:
            fa = _avg(all_patched_losses.get(0, []))
            fap = math.exp(min(fa, 100))
            aggregate["all_layers"] = {
                "final_patched_loss": round(fa, 6),
                "final_patched_ppl": round(fap, 2),
                "final_loss_delta": round(fa - final_bl, 6),
            }
            summary["iterative_summary/all_layers_final_loss"] = fa

        if wandb_run is not None:
            wandb_run.log(summary)

        print(
            f"[Eval:iterative] Done. Final baseline loss={final_bl:.4f}, "
            f"ppl={final_bl_ppl:.2f}",
            flush=True,
        )
        for li in layer_indices:
            info = aggregate["per_layer"][li]
            print(
                f"[Eval:iterative] Layer {li}: final_loss={info['final_patched_loss']:.4f}, "
                f"ppl={info['final_patched_ppl']:.2f}, "
                f"delta={info['final_loss_delta']:.4f}",
                flush=True,
            )

        return {
            "eval_mode": "iterative",
            "per_step": per_step_results,
            "aggregate": aggregate,
            "config": {
                "sae_layers": layer_indices,
                "total_steps": T,
                "logged_steps": sorted(bl_losses.keys()),
                "dataset_split": self._config.dataset_split,
                "max_samples": self._config.max_samples,
            },
        }

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def run(
        self,
        dataloader: DataLoader,
        wandb_run: Any | None = None,
        eval_mode: str = "single_step",
    ) -> dict:
        """Run evaluation in the specified mode.

        Args:
            dataloader: Evaluation data yielding (input_ids, attention_mask).
            wandb_run: Optional wandb run for logging.
            eval_mode: ``"single_step"`` or ``"iterative"``.

        Returns:
            Results dict.
        """
        if eval_mode == "iterative":
            return self.run_iterative(dataloader, wandb_run)
        return self.run_single_step(dataloader, wandb_run)
