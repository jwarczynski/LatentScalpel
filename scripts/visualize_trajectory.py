#!/usr/bin/env python3
"""Visualize SAE feature activations across the denoising trajectory.

Produces 4 plots per layer:
1. Dot/bubble plot: feature ID vs timestep, dot size = activation magnitude
2. Jaccard similarity: feature set overlap between adjacent timesteps
3. Phase transition: count of uniquely active features per timestep
4. Top-N variable features: line plots of features with highest variance

Usage:
    python scripts/visualize_trajectory.py experiments/trajectory_features.json
    python scripts/visualize_trajectory.py experiments/trajectory_features.json --top-n 20 --out-dir plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_matrix(layer_data: dict[str, dict[str, float]], top_n_features: int | None = None):
    """Build a (features x timesteps) matrix from the layer data.

    Returns:
        timesteps: sorted list of int timesteps
        feature_ids: sorted list of int feature IDs
        matrix: np.ndarray of shape (len(feature_ids), len(timesteps))
    """
    timesteps = sorted(int(t) for t in layer_data.keys())
    # Collect all feature IDs that appear at any timestep
    all_features: set[int] = set()
    for t_str, feats in layer_data.items():
        all_features.update(int(f) for f in feats.keys())

    if top_n_features and len(all_features) > top_n_features:
        # Keep features with highest total activation
        feat_totals: dict[int, float] = {}
        for t_str, feats in layer_data.items():
            for f_str, val in feats.items():
                fid = int(f_str)
                feat_totals[fid] = feat_totals.get(fid, 0.0) + val
        ranked = sorted(feat_totals.items(), key=lambda x: x[1], reverse=True)
        all_features = {fid for fid, _ in ranked[:top_n_features]}

    feature_ids = sorted(all_features)
    fid_to_row = {fid: i for i, fid in enumerate(feature_ids)}

    matrix = np.zeros((len(feature_ids), len(timesteps)))
    for t_idx, t_val in enumerate(timesteps):
        feats = layer_data.get(str(t_val), {})
        for f_str, val in feats.items():
            fid = int(f_str)
            if fid in fid_to_row:
                matrix[fid_to_row[fid], t_idx] = val

    return timesteps, feature_ids, matrix


def _active_sets(layer_data: dict[str, dict[str, float]], threshold: float = 0.0):
    """Return {timestep: set of active feature IDs} for each timestep."""
    result = {}
    for t_str, feats in layer_data.items():
        t = int(t_str)
        result[t] = {int(f) for f, v in feats.items() if v > threshold}
    return result


def plot_bubble(layer_data: dict, layer_idx: int, ax: plt.Axes, top_n: int = 100):
    """Plot 1: Dot/bubble plot — X=timestep, Y=feature ID, size=magnitude."""
    timesteps, feature_ids, matrix = _build_matrix(layer_data, top_n_features=top_n)

    # Collect non-zero points
    xs, ys, sizes = [], [], []
    for t_idx, t_val in enumerate(timesteps):
        for f_idx, fid in enumerate(feature_ids):
            val = matrix[f_idx, t_idx]
            if val > 0:
                xs.append(t_val)
                ys.append(fid)
                sizes.append(val)

    if not xs:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    sizes_arr = np.array(sizes)
    # Normalize dot sizes for readability
    size_norm = (sizes_arr / sizes_arr.max()) * 80 + 2

    scatter = ax.scatter(xs, ys, s=size_norm, c=sizes_arr, cmap="viridis",
                         alpha=0.6, edgecolors="none")
    ax.set_xlabel("Denoising timestep (T → 0)")
    ax.set_ylabel("Feature ID")
    ax.set_title(f"Layer {layer_idx}: Active features across timesteps")
    ax.invert_xaxis()  # T decreases left to right (denoising direction)
    plt.colorbar(scatter, ax=ax, label="Mean activation", shrink=0.8)


def plot_jaccard(layer_data: dict, layer_idx: int, ax: plt.Axes):
    """Plot 2: Jaccard similarity between adjacent timestep feature sets."""
    active = _active_sets(layer_data)
    timesteps = sorted(active.keys(), reverse=True)  # T -> 0

    jaccards = []
    ts_pairs = []
    for i in range(len(timesteps) - 1):
        t1, t2 = timesteps[i], timesteps[i + 1]
        s1, s2 = active[t1], active[t2]
        union = s1 | s2
        if len(union) == 0:
            jaccards.append(1.0)
        else:
            jaccards.append(len(s1 & s2) / len(union))
        ts_pairs.append((t1 + t2) / 2)  # midpoint for x-axis

    ax.plot(ts_pairs, jaccards, "o-", markersize=2, linewidth=1, color="steelblue")
    ax.set_xlabel("Denoising timestep (T → 0)")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title(f"Layer {layer_idx}: Feature set stability")
    ax.set_ylim(-0.05, 1.05)
    ax.invert_xaxis()
    ax.axhline(y=np.mean(jaccards), color="red", linestyle="--", alpha=0.5,
               label=f"mean={np.mean(jaccards):.2f}")
    ax.legend(fontsize=8)


def plot_phase_transition(layer_data: dict, layer_idx: int, ax: plt.Axes):
    """Plot 3: Count of uniquely active features per timestep.

    A feature is 'unique' at timestep t if it's active at t but NOT active
    at either of the adjacent sampled timesteps.
    """
    active = _active_sets(layer_data)
    timesteps = sorted(active.keys(), reverse=True)

    unique_counts = []
    for i, t in enumerate(timesteps):
        neighbors: set[int] = set()
        if i > 0:
            neighbors |= active[timesteps[i - 1]]
        if i < len(timesteps) - 1:
            neighbors |= active[timesteps[i + 1]]
        unique = active[t] - neighbors
        unique_counts.append(len(unique))

    ax.bar(timesteps, unique_counts, width=max(1, timesteps[0] - timesteps[1]) * 0.8 if len(timesteps) > 1 else 10,
           color="coral", alpha=0.7)
    ax.set_xlabel("Denoising timestep (T → 0)")
    ax.set_ylabel("# uniquely active features")
    ax.set_title(f"Layer {layer_idx}: Phase transitions (unique features)")
    ax.invert_xaxis()


def plot_top_variable(layer_data: dict, layer_idx: int, ax: plt.Axes, top_n: int = 15):
    """Plot 4: Line plots of the top-N most variable features across timesteps."""
    timesteps, feature_ids, matrix = _build_matrix(layer_data)

    if matrix.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    # Variance across timesteps for each feature
    variances = matrix.var(axis=1)
    top_indices = np.argsort(variances)[-top_n:][::-1]

    cmap = plt.cm.get_cmap("tab20", min(top_n, 20))
    for rank, feat_row_idx in enumerate(top_indices):
        fid = feature_ids[feat_row_idx]
        ax.plot(timesteps, matrix[feat_row_idx], label=f"F{fid}",
                color=cmap(rank % 20), linewidth=1, alpha=0.8)

    ax.set_xlabel("Denoising timestep (T → 0)")
    ax.set_ylabel("Mean activation")
    ax.set_title(f"Layer {layer_idx}: Top-{top_n} most variable features")
    ax.invert_xaxis()
    ax.legend(fontsize=6, ncol=3, loc="upper right")


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory feature activations")
    parser.add_argument("data_path", help="Path to trajectory_features.json")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Top N features to show in bubble plot (default: 100)")
    parser.add_argument("--top-var", type=int, default=15,
                        help="Top N variable features for line plot (default: 15)")
    parser.add_argument("--out-dir", type=str, default="./experiments/plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    data = load_data(args.data_path)
    meta = data["metadata"]
    layers_data = data["layers"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded trajectory data: {meta['num_samples']} samples, "
          f"{len(meta['sampled_timesteps'])} timesteps, "
          f"layers {meta['layers']}")

    for layer_str, layer_data in layers_data.items():
        layer_idx = int(layer_str)
        print(f"\nGenerating plots for layer {layer_idx}...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(
            f"Layer {layer_idx} — SAE Feature Activations Across Denoising Trajectory\n"
            f"({meta['num_samples']} samples, {meta['dataset_name']}/{meta['dataset_split']}, "
            f"subsample={meta['timestep_subsample']})",
            fontsize=13,
        )

        plot_bubble(layer_data, layer_idx, axes[0, 0], top_n=args.top_n)
        plot_jaccard(layer_data, layer_idx, axes[0, 1])
        plot_phase_transition(layer_data, layer_idx, axes[1, 0])
        plot_top_variable(layer_data, layer_idx, axes[1, 1], top_n=args.top_var)

        plt.tight_layout()
        save_path = out_dir / f"trajectory_layer_{layer_idx:02d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
