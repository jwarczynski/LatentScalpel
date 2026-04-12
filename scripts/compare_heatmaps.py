#!/usr/bin/env python3
"""Compare heatmaps between standard and modified denoising schedules.

Generates side-by-side heatmaps (features × timesteps) for standard and
modified schedules from a schedule comparison JSON produced by
``run-schedule-experiment``.

Features are sorted by peak NDS under the standard schedule. Features
with >threshold peak NDS shift are highlighted with distinct markers.

Usage:
    python scripts/compare_heatmaps.py experiments/schedule_experiment/schedule_comparison.json
    python scripts/compare_heatmaps.py experiments/schedule_experiment/schedule_comparison.json \
        --shift-threshold 0.15 --out-dir plots/schedule/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_comparison(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _sort_features_by_peak_nds(
    profiles: dict[str, list[float]],
    timesteps: list[int],
    total_steps: int,
) -> list[str]:
    """Return feature IDs sorted by peak NDS (ascending) under the given profiles."""
    def _peak_nds(fid: str) -> float:
        profile = profiles[fid]
        if not profile or max(profile) == 0:
            return 0.0
        peak_idx = int(np.argmax(profile))
        if peak_idx < len(timesteps):
            return timesteps[peak_idx] / total_steps
        return 0.0

    return sorted(profiles.keys(), key=_peak_nds)


def _build_heatmap_matrix(
    profiles: dict[str, list[float]],
    sorted_fids: list[str],
) -> np.ndarray:
    """Build a (features × timesteps) matrix from profiles in sorted order."""
    if not sorted_fids or not profiles:
        return np.zeros((0, 0))
    n_timesteps = max(len(profiles[fid]) for fid in sorted_fids)
    matrix = np.zeros((len(sorted_fids), n_timesteps))
    for row, fid in enumerate(sorted_fids):
        profile = profiles[fid]
        matrix[row, :len(profile)] = profile
    return matrix


def _find_shifted_fids(
    original_profiles: dict[str, list[float]],
    modified_profiles: dict[str, list[float]],
    original_timesteps: list[int],
    modified_timesteps: list[int],
    total_steps: int,
    shift_threshold_pct: float,
) -> set[str]:
    """Return set of feature IDs whose peak NDS shifted by more than threshold."""
    shifted: set[str] = set()
    common = set(original_profiles.keys()) & set(modified_profiles.keys())
    for fid in common:
        orig = original_profiles[fid]
        mod = modified_profiles[fid]
        if not orig or not mod or max(orig) == 0 or max(mod) == 0:
            continue
        orig_peak_idx = int(np.argmax(orig))
        mod_peak_idx = int(np.argmax(mod))
        if orig_peak_idx >= len(original_timesteps) or mod_peak_idx >= len(modified_timesteps):
            continue
        orig_nds = original_timesteps[orig_peak_idx] / total_steps
        mod_nds = modified_timesteps[mod_peak_idx] / total_steps
        if abs(orig_nds - mod_nds) > shift_threshold_pct:
            shifted.add(fid)
    return shifted


def plot_comparison_heatmap(
    comparison_json_path: str,
    output_dir: str,
    shift_threshold_pct: float = 0.10,
) -> None:
    """Generate side-by-side heatmaps from schedule comparison data.

    Args:
        comparison_json_path: Path to schedule_comparison.json.
        output_dir: Directory to save PNG output.
        shift_threshold_pct: Fraction threshold for highlighting shifted features.
    """
    data = _load_comparison(comparison_json_path)
    meta = data["metadata"]
    total_steps = meta["diffusion_steps"]
    orig_timesteps = sorted(meta["sampled_standard_timesteps"])
    mod_timesteps = sorted(meta["sampled_modified_timesteps"])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_str, layer_data in data["layers"].items():
        orig_profiles = layer_data["original_profiles"]
        mod_profiles = layer_data["modified_profiles"]

        # Sort features by peak NDS under standard schedule
        sorted_fids = _sort_features_by_peak_nds(orig_profiles, orig_timesteps, total_steps)

        orig_matrix = _build_heatmap_matrix(orig_profiles, sorted_fids)
        mod_matrix = _build_heatmap_matrix(mod_profiles, sorted_fids)

        shifted = _find_shifted_fids(
            orig_profiles, mod_profiles,
            orig_timesteps, mod_timesteps,
            total_steps, shift_threshold_pct,
        )

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, len(sorted_fids) * 0.15)))
        fig.suptitle(
            f"Layer {layer_str} — Schedule Comparison Heatmap\n"
            f"{meta['modification_type']} {meta['modification_params']}",
            fontsize=13,
        )

        vmax = max(orig_matrix.max(), mod_matrix.max()) if orig_matrix.size and mod_matrix.size else 1.0

        im1 = ax1.imshow(orig_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax1.set_title("Standard Schedule")
        ax1.set_xlabel("Timestep index")
        ax1.set_ylabel("Feature (sorted by peak NDS)")

        im2 = ax2.imshow(mod_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax2.set_title("Modified Schedule")
        ax2.set_xlabel("Timestep index")

        # Highlight shifted features with red markers on the y-axis
        for row, fid in enumerate(sorted_fids):
            if fid in shifted:
                ax1.plot(-0.8, row, "r>", markersize=4)
                ax2.plot(-0.8, row, "r>", markersize=4)

        plt.colorbar(im2, ax=[ax1, ax2], label="Mean activation", shrink=0.8)
        plt.tight_layout()

        save_path = out_dir / f"comparison_heatmap_layer_{layer_str}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"All comparison heatmaps saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Compare schedule heatmaps")
    parser.add_argument("data_path", help="Path to schedule_comparison.json")
    parser.add_argument("--shift-threshold", type=float, default=0.10,
                        help="Peak NDS shift threshold for highlighting (default: 0.10)")
    parser.add_argument("--out-dir", type=str, default="./experiments/plots/schedule",
                        help="Output directory for heatmap PNGs")
    args = parser.parse_args()

    plot_comparison_heatmap(args.data_path, args.out_dir, args.shift_threshold)


if __name__ == "__main__":
    main()
