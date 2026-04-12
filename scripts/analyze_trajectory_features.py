#!/usr/bin/env python3
"""Analyze trajectory feature dynamics — identify phase transitions, midpoint
effects, and temporal feature groups.

Produces:
- Summary of midpoint-transition features (on→off or off→on around step ~1000)
- Finishing-phase features (spike near step 0)
- Features active around step ~1000 specifically
- Temporal clustering of features by activation profile shape
- Cross-reference with interpretation results if available

Usage:
    python scripts/analyze_trajectory_features.py experiments/trajectory_features.json
    python scripts/analyze_trajectory_features.py experiments/trajectory_features.json \
        --interp experiments/interpretation_results.json \
        --out-dir experiments/plots/trajectory_analysis
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_matrix(layer_data: dict[str, dict[str, float]]):
    """Build (features × timesteps) matrix. Returns timesteps, feature_ids, matrix."""
    timesteps = sorted(int(t) for t in layer_data.keys())
    all_features: set[int] = set()
    for feats in layer_data.values():
        all_features.update(int(f) for f in feats.keys())
    feature_ids = sorted(all_features)
    fid_to_row = {fid: i for i, fid in enumerate(feature_ids)}

    matrix = np.zeros((len(feature_ids), len(timesteps)))
    for t_idx, t_val in enumerate(timesteps):
        feats = layer_data.get(str(t_val), {})
        for f_str, val in feats.items():
            fid = int(f_str)
            if fid in fid_to_row:
                matrix[fid_to_row[fid], t_idx] = val

    return np.array(timesteps), feature_ids, matrix



def classify_features(timesteps, feature_ids, matrix, diffusion_steps: int):
    """Classify each feature by its temporal activation pattern.

    Categories:
    - 'early_only': active in first half (high noise), dies in second half
    - 'late_only': inactive in first half, activates in second half (low noise)
    - 'midpoint_transition': sharp change around the midpoint
    - 'finishing_spike': spike in the last ~10% of denoising (near step 0)
    - 'stable': relatively constant activation throughout
    - 'variable': high variance but no clear pattern
    """
    n_features, n_timesteps = matrix.shape
    mid_idx = n_timesteps // 2
    last_10pct = int(n_timesteps * 0.9)

    results = []
    for i, fid in enumerate(feature_ids):
        row = matrix[i]
        total = row.sum()
        if total < 1e-6:
            continue  # skip dead features

        row_max = row.max()
        row_mean = row.mean()
        row_var = row.var()

        first_half_mean = row[:mid_idx].mean()
        second_half_mean = row[mid_idx:].mean()
        last_10pct_mean = row[last_10pct:].mean()
        first_10pct_mean = row[:int(n_timesteps * 0.1)].mean()

        # Normalized profile for shape analysis
        profile = row / (row_max + 1e-10)

        # Detect midpoint transition: large ratio between halves
        half_ratio = (first_half_mean + 1e-8) / (second_half_mean + 1e-8)

        # Detect finishing spike
        finishing_ratio = (last_10pct_mean + 1e-8) / (row_mean + 1e-8)

        # Detect early spike (features active only at high noise)
        early_ratio = (first_10pct_mean + 1e-8) / (row_mean + 1e-8)

        # Coefficient of variation
        cv = np.sqrt(row_var) / (row_mean + 1e-10)

        # Find the timestep of maximum activation
        peak_step_idx = np.argmax(row)
        peak_step = timesteps[peak_step_idx]

        # Find where the biggest single-step change happens
        diffs = np.abs(np.diff(row))
        max_diff_idx = np.argmax(diffs)
        max_diff_step = timesteps[max_diff_idx]

        category = "variable"
        if cv < 0.3:
            category = "stable"
        elif half_ratio > 3.0:
            category = "early_only"
        elif half_ratio < 0.33:
            category = "late_only"
        elif finishing_ratio > 2.5:
            category = "finishing_spike"
        elif early_ratio > 2.5:
            category = "starting_spike"
        elif abs(np.log(half_ratio)) > 0.5:
            category = "midpoint_transition"

        results.append({
            "feature_id": fid,
            "category": category,
            "total_activation": float(total),
            "mean_activation": float(row_mean),
            "max_activation": float(row_max),
            "variance": float(row_var),
            "cv": float(cv),
            "first_half_mean": float(first_half_mean),
            "second_half_mean": float(second_half_mean),
            "half_ratio": float(half_ratio),
            "finishing_ratio": float(finishing_ratio),
            "early_ratio": float(early_ratio),
            "peak_step": int(peak_step),
            "max_change_step": int(max_diff_step),
        })

    return results


def print_category_summary(classified: list[dict], interp_data: dict | None, layer_idx: int):
    """Print a summary of feature categories with optional interpretations."""
    categories = defaultdict(list)
    for f in classified:
        categories[f["category"]].append(f)

    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx} — Feature Temporal Classification")
    print(f"{'='*80}")
    print(f"Total features with nonzero activation: {len(classified)}")
    print()

    for cat in ["early_only", "late_only", "midpoint_transition",
                 "finishing_spike", "starting_spike", "stable", "variable"]:
        feats = categories.get(cat, [])
        if not feats:
            continue
        # Sort by variance (most interesting first)
        feats.sort(key=lambda x: x["variance"], reverse=True)
        print(f"\n--- {cat.upper()} ({len(feats)} features) ---")
        for f in feats[:15]:  # top 15 per category
            fid = f["feature_id"]
            line = (f"  F{fid:>6d}  mean={f['mean_activation']:.3f}  "
                    f"max={f['max_activation']:.3f}  var={f['variance']:.4f}  "
                    f"half_ratio={f['half_ratio']:.2f}  "
                    f"peak@step={f['peak_step']}  max_change@step={f['max_change_step']}")
            if interp_data and str(fid) in interp_data:
                interp = interp_data[str(fid)]
                expl = interp.get("explanation", "")[:80]
                score = interp.get("interpretability_score", "?")
                line += f"\n          interp(score={score}): {expl}..."
            print(line)



def analyze_midpoint_region(timesteps, feature_ids, matrix, diffusion_steps: int):
    """Deep dive into what happens around the midpoint (~step 1000).

    Looks at features that have their maximum rate of change near the midpoint.
    """
    mid_step = diffusion_steps // 2
    # Find timestep indices near the midpoint (within 20% of total)
    window = int(len(timesteps) * 0.2)
    mid_t_idx = np.argmin(np.abs(timesteps - mid_step))
    lo = max(0, mid_t_idx - window)
    hi = min(len(timesteps), mid_t_idx + window)

    print(f"\n{'='*80}")
    print(f"MIDPOINT ANALYSIS (around step ~{mid_step})")
    print(f"Window: steps {timesteps[lo]} to {timesteps[hi-1]}")
    print(f"{'='*80}")

    # For each feature, compute the activation change in the midpoint window
    midpoint_changes = []
    for i, fid in enumerate(feature_ids):
        row = matrix[i]
        if row.max() < 1e-6:
            continue
        before = row[lo:mid_t_idx].mean() if mid_t_idx > lo else 0
        after = row[mid_t_idx:hi].mean() if hi > mid_t_idx else 0
        change = after - before
        rel_change = abs(change) / (row.mean() + 1e-10)
        midpoint_changes.append({
            "feature_id": fid,
            "before_mid": float(before),
            "after_mid": float(after),
            "change": float(change),
            "relative_change": float(rel_change),
            "direction": "increases" if change > 0 else "decreases",
        })

    # Sort by absolute relative change
    midpoint_changes.sort(key=lambda x: abs(x["relative_change"]), reverse=True)

    print(f"\nTop 20 features with biggest change around midpoint:")
    for mc in midpoint_changes[:20]:
        fid = mc["feature_id"]
        print(f"  F{fid:>6d}  before={mc['before_mid']:.4f}  after={mc['after_mid']:.4f}  "
              f"change={mc['change']:+.4f} ({mc['direction']})  "
              f"rel_change={mc['relative_change']:.2f}x")

    # Features that appear ONLY around the midpoint
    print(f"\nFeatures with peak activation in midpoint window:")
    for i, fid in enumerate(feature_ids):
        row = matrix[i]
        if row.max() < 1e-6:
            continue
        peak_idx = np.argmax(row)
        if lo <= peak_idx < hi:
            peak_val = row[peak_idx]
            outside_mean = np.concatenate([row[:lo], row[hi:]]).mean() if (lo > 0 or hi < len(timesteps)) else 0
            if peak_val > 3 * (outside_mean + 1e-10):
                print(f"  F{fid:>6d}  peak={peak_val:.4f} @ step {timesteps[peak_idx]}  "
                      f"outside_mean={outside_mean:.4f}  ratio={peak_val/(outside_mean+1e-10):.1f}x")


def plot_feature_profiles(timesteps, feature_ids, matrix, classified, layer_idx, out_dir: Path):
    """Plot activation profiles grouped by category."""
    categories = defaultdict(list)
    for f in classified:
        categories[f["category"]].append(f)

    # Plot each category
    for cat in ["early_only", "late_only", "midpoint_transition",
                 "finishing_spike", "starting_spike"]:
        feats = categories.get(cat, [])
        if not feats:
            continue
        feats.sort(key=lambda x: x["variance"], reverse=True)
        top = feats[:min(12, len(feats))]

        fig, ax = plt.subplots(figsize=(14, 6))
        fid_to_row = {fid: i for i, fid in enumerate(feature_ids)}
        cmap = plt.cm.get_cmap("tab20", 20)

        for rank, f in enumerate(top):
            fid = f["feature_id"]
            row_idx = fid_to_row[fid]
            ax.plot(timesteps, matrix[row_idx], label=f"F{fid}",
                    color=cmap(rank % 20), linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Denoising timestep (T → 0)")
        ax.set_ylabel("Mean activation")
        ax.set_title(f"Layer {layer_idx}: {cat.replace('_', ' ').title()} features (top {len(top)})")
        ax.invert_xaxis()
        ax.legend(fontsize=7, ncol=3, loc="best")
        ax.axvline(x=timesteps[len(timesteps)//2], color="red", linestyle="--",
                   alpha=0.3, label="midpoint")
        plt.tight_layout()
        save_path = out_dir / f"layer_{layer_idx:02d}_{cat}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # Special plot: all-features heatmap sorted by peak timestep
    fig, ax = plt.subplots(figsize=(16, 10))
    # Filter to features with meaningful activation
    active_mask = matrix.max(axis=1) > 0.01
    active_matrix = matrix[active_mask]
    active_fids = [fid for fid, m in zip(feature_ids, active_mask) if m]

    if active_matrix.shape[0] > 0:
        # Normalize each row to [0,1]
        row_maxes = active_matrix.max(axis=1, keepdims=True)
        norm_matrix = active_matrix / (row_maxes + 1e-10)
        # Sort by peak timestep
        peak_indices = np.argmax(norm_matrix, axis=1)
        sort_order = np.argsort(peak_indices)
        sorted_matrix = norm_matrix[sort_order]

        im = ax.imshow(sorted_matrix, aspect="auto", cmap="inferno",
                       extent=[timesteps[-1], timesteps[0], len(active_fids), 0])
        ax.set_xlabel("Denoising timestep (T → 0)")
        ax.set_ylabel("Features (sorted by peak timestep)")
        ax.set_title(f"Layer {layer_idx}: All active features — normalized activation heatmap\n"
                     f"({active_matrix.shape[0]} features sorted by when they peak)")
        plt.colorbar(im, ax=ax, label="Normalized activation", shrink=0.8)
        ax.axvline(x=timesteps[len(timesteps)//2], color="cyan", linestyle="--",
                   alpha=0.5, linewidth=1)
    plt.tight_layout()
    save_path = out_dir / f"layer_{layer_idx:02d}_heatmap.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # Active feature count over time
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: number of active features per timestep
    active_counts = (matrix > 0.01).sum(axis=0)
    axes[0].plot(timesteps, active_counts, "o-", markersize=2, color="steelblue")
    axes[0].set_xlabel("Denoising timestep (T → 0)")
    axes[0].set_ylabel("# active features (activation > 0.01)")
    axes[0].set_title(f"Layer {layer_idx}: Active feature count")
    axes[0].invert_xaxis()
    axes[0].axvline(x=timesteps[len(timesteps)//2], color="red", linestyle="--", alpha=0.3)

    # Right: total activation mass per timestep
    total_mass = matrix.sum(axis=0)
    axes[1].plot(timesteps, total_mass, "o-", markersize=2, color="coral")
    axes[1].set_xlabel("Denoising timestep (T → 0)")
    axes[1].set_ylabel("Total activation mass")
    axes[1].set_title(f"Layer {layer_idx}: Total activation mass")
    axes[1].invert_xaxis()
    axes[1].axvline(x=timesteps[len(timesteps)//2], color="red", linestyle="--", alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / f"layer_{layer_idx:02d}_counts.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")



def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory feature dynamics")
    parser.add_argument("data_path", help="Path to trajectory_features.json")
    parser.add_argument("--interp", type=str, default=None,
                        help="Path to interpretation_results.json for cross-reference")
    parser.add_argument("--out-dir", type=str, default="./experiments/plots/trajectory_analysis",
                        help="Output directory for analysis plots")
    parser.add_argument("--save-json", action="store_true",
                        help="Save classification results as JSON")
    args = parser.parse_args()

    data = load_json(args.data_path)
    meta = data["metadata"]
    layers_data = data["layers"]
    diffusion_steps = meta["diffusion_steps"]

    # Load interpretation data if available
    interp_data = None
    if args.interp:
        interp_raw = load_json(args.interp)
        interp_data = interp_raw.get("features", {})
        print(f"Loaded {len(interp_data)} feature interpretations")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Trajectory data: {meta['num_samples']} samples, "
          f"{len(meta['sampled_timesteps'])} timesteps, "
          f"layers {meta['layers']}, diffusion_steps={diffusion_steps}")

    all_results = {}

    for layer_str, layer_data in layers_data.items():
        layer_idx = int(layer_str)
        print(f"\n{'#'*80}")
        print(f"# LAYER {layer_idx}")
        print(f"{'#'*80}")

        timesteps, feature_ids, matrix = build_matrix(layer_data)
        print(f"Matrix shape: {matrix.shape} ({len(feature_ids)} features × {len(timesteps)} timesteps)")
        print(f"Timestep range: {timesteps[0]} → {timesteps[-1]}")

        # Classify features
        classified = classify_features(timesteps, feature_ids, matrix, diffusion_steps)
        print_category_summary(classified, interp_data, layer_idx)

        # Midpoint analysis
        analyze_midpoint_region(timesteps, feature_ids, matrix, diffusion_steps)

        # Generate plots
        print(f"\nGenerating plots for layer {layer_idx}...")
        plot_feature_profiles(timesteps, feature_ids, matrix, classified, layer_idx, out_dir)

        all_results[layer_str] = classified

    # Save JSON results
    if args.save_json:
        json_path = out_dir / "trajectory_classification.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nClassification saved to {json_path}")

    print(f"\nAll analysis plots saved to {out_dir}/")


# --- Explanation of "Most Variable Features" plot ---
EXPLANATION = """
WHAT "MOST VARIABLE FEATURES" MEANS:
=====================================
The "Top-N most variable features" plot in visualize_trajectory.py shows features
ranked by their VARIANCE across all sampled timesteps.

Variance = how much a feature's mean activation fluctuates over the denoising process.

- HIGH variance feature: its activation changes dramatically during denoising.
  It might be very active at high noise (early steps) and silent at low noise,
  or vice versa. These are the features most sensitive to the denoising phase.

- LOW variance feature: roughly constant activation regardless of noise level.
  These encode information that's present throughout the entire process.

The plot shows the activation trajectory (y-axis) over timesteps (x-axis) for
the top-N features by variance. Each line is one feature.

Features that show a sharp transition at the midpoint likely encode information
about the noise→signal boundary — they might detect when the model transitions
from "mostly noise" to "mostly signal" in the diffusion process.
"""


if __name__ == "__main__":
    print(EXPLANATION)
    main()
