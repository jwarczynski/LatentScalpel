#!/usr/bin/env python3
"""Generate trajectory analysis plots with organized directory structure.

Output structure:
    experiments/plots/trajectory_analysis/<model>/<layer_XX>/<plot>.png

Usage:
    uv run python scripts/plot_trajectory_organized.py \
        experiments/trajectory_features.json --model genie

    uv run python scripts/plot_trajectory_organized.py \
        experiments/plaid_trajectory_features.json --model plaid

Also reorganizes existing flat plots if --reorganize-existing is passed.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_matrix(layer_data: dict[str, dict[str, float]]):
    """Build (n_features, n_timesteps) matrix from trajectory data."""
    all_timesteps = set()
    all_features = set()
    for t_str, feat_dict in layer_data.items():
        all_timesteps.add(int(t_str))
        all_features.update(feat_dict.keys())

    timesteps = sorted(all_timesteps, reverse=True)
    feature_ids = sorted(all_features, key=lambda x: int(x))
    fid_to_col = {fid: i for i, fid in enumerate(feature_ids)}

    matrix = np.zeros((len(feature_ids), len(timesteps)))
    for t_idx, t in enumerate(timesteps):
        t_str = str(t)
        if t_str in layer_data:
            for fid, val in layer_data[t_str].items():
                matrix[fid_to_col[fid], t_idx] = val

    return np.array(timesteps), feature_ids, matrix


def classify_features(timesteps, feature_ids, matrix, diffusion_steps: int):
    """Classify features by their temporal activation pattern."""
    n_features, n_timesteps = matrix.shape
    mid_idx = n_timesteps // 2
    early_slice = slice(0, n_timesteps // 4)
    mid_slice = slice(n_timesteps // 4, 3 * n_timesteps // 4)
    late_slice = slice(3 * n_timesteps // 4, n_timesteps)

    classified = []
    for i, fid in enumerate(feature_ids):
        profile = matrix[i]
        total = profile.sum()
        if total < 1e-6:
            continue

        early_mass = profile[early_slice].sum() / total
        mid_mass = profile[mid_slice].sum() / total
        late_mass = profile[late_slice].sum() / total
        peak_idx = np.argmax(profile)
        peak_frac = peak_idx / max(n_timesteps - 1, 1)
        variance = np.var(profile)

        if early_mass > 0.6 and peak_frac < 0.3:
            cat = "early_only"
        elif late_mass > 0.6 and peak_frac > 0.7:
            cat = "late_only"
        elif mid_mass > 0.5 and 0.3 <= peak_frac <= 0.7:
            cat = "midpoint_transition"
        elif peak_frac > 0.85 and profile[peak_idx] > 2 * np.mean(profile):
            cat = "finishing_spike"
        elif peak_frac < 0.15 and profile[peak_idx] > 2 * np.mean(profile):
            cat = "starting_spike"
        else:
            cat = "other"

        classified.append({
            "feature_id": fid,
            "category": cat,
            "peak_frac": float(peak_frac),
            "variance": float(variance),
            "early_mass": float(early_mass),
            "mid_mass": float(mid_mass),
            "late_mass": float(late_mass),
        })

    return classified


def plot_feature_profiles(timesteps, feature_ids, matrix, classified,
                          layer_idx, out_dir: Path, model_name: str):
    """Plot activation profiles grouped by category."""
    categories = defaultdict(list)
    for f in classified:
        categories[f["category"]].append(f)

    title_prefix = f"{model_name.upper()} Layer {layer_idx}"

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
        ax.set_title(f"{title_prefix}: {cat.replace('_', ' ').title()} features (top {len(top)})")
        ax.invert_xaxis()
        ax.legend(fontsize=7, ncol=3, loc="best")
        ax.axvline(x=timesteps[len(timesteps) // 2], color="red",
                   linestyle="--", alpha=0.3, label="midpoint")
        plt.tight_layout()
        fig.savefig(out_dir / f"{cat}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir / f'{cat}.png'}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    active_mask = matrix.max(axis=1) > 0.01
    active_matrix = matrix[active_mask]
    active_fids = [fid for fid, m in zip(feature_ids, active_mask) if m]

    if active_matrix.shape[0] > 0:
        row_maxes = active_matrix.max(axis=1, keepdims=True)
        norm_matrix = active_matrix / (row_maxes + 1e-10)
        peak_indices = np.argmax(norm_matrix, axis=1)
        sort_order = np.argsort(peak_indices)
        sorted_matrix = norm_matrix[sort_order]

        im = ax.imshow(sorted_matrix, aspect="auto", cmap="inferno",
                       extent=[timesteps[-1], timesteps[0], len(active_fids), 0])
        ax.set_xlabel("Denoising timestep (T → 0)")
        ax.set_ylabel("Features (sorted by peak timestep)")
        ax.set_title(f"{title_prefix}: Normalized activation heatmap "
                     f"({active_matrix.shape[0]} features)")
        plt.colorbar(im, ax=ax, label="Normalized activation", shrink=0.8)
        ax.axvline(x=timesteps[len(timesteps) // 2], color="cyan",
                   linestyle="--", alpha=0.5, linewidth=1)
    plt.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'heatmap.png'}")

    # Counts
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    active_counts = (matrix > 0.01).sum(axis=0)
    axes[0].plot(timesteps, active_counts, "o-", markersize=2, color="steelblue")
    axes[0].set_xlabel("Denoising timestep (T → 0)")
    axes[0].set_ylabel("# active features (activation > 0.01)")
    axes[0].set_title(f"{title_prefix}: Active feature count")
    axes[0].invert_xaxis()
    axes[0].axvline(x=timesteps[len(timesteps) // 2], color="red",
                    linestyle="--", alpha=0.3)

    total_mass = matrix.sum(axis=0)
    axes[1].plot(timesteps, total_mass, "o-", markersize=2, color="coral")
    axes[1].set_xlabel("Denoising timestep (T → 0)")
    axes[1].set_ylabel("Total activation mass")
    axes[1].set_title(f"{title_prefix}: Total activation mass")
    axes[1].invert_xaxis()
    axes[1].axvline(x=timesteps[len(timesteps) // 2], color="red",
                    linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'counts.png'}")


def reorganize_existing(src_dir: Path, model_name: str, base_out: Path):
    """Move existing flat layer_XX_*.png files into model/layer_XX/ structure."""
    import re
    pattern = re.compile(r"layer_(\d+)_(.+)\.png")
    for f in sorted(src_dir.glob("layer_*_*.png")):
        m = pattern.match(f.name)
        if not m:
            continue
        layer_str = f"layer_{int(m.group(1)):02d}"
        plot_name = m.group(2) + ".png"
        dest_dir = base_out / model_name / layer_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / plot_name
        shutil.copy2(f, dest)
        print(f"  Copied: {f.name} -> {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectory plots with organized directory structure")
    parser.add_argument("data_path", help="Path to trajectory_features.json")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. genie, plaid)")
    parser.add_argument("--base-dir", default="./experiments/plots/trajectory_analysis",
                        help="Base output directory")
    parser.add_argument("--save-json", action="store_true",
                        help="Save classification results as JSON per layer")
    parser.add_argument("--reorganize-existing", action="store_true",
                        help="Also reorganize existing flat plots into new structure")
    args = parser.parse_args()

    base_out = Path(args.base_dir)

    if args.reorganize_existing:
        print(f"Reorganizing existing plots from {base_out} into {base_out}/{args.model}/...")
        reorganize_existing(base_out, args.model, base_out)
        print()

    data = load_json(args.data_path)
    meta = data["metadata"]
    layers_data = data["layers"]
    diffusion_steps = meta.get("diffusion_steps", meta.get("sampling_timesteps", 256))
    sampled_timesteps = meta.get("sampled_timesteps", meta.get("sampled_steps", []))

    print(f"Model: {args.model}")
    print(f"Trajectory data: {meta['num_samples']} samples, "
          f"{len(sampled_timesteps)} timesteps, "
          f"layers {meta['layers']}, diffusion_steps={diffusion_steps}")

    for layer_str, layer_data in layers_data.items():
        layer_idx = int(layer_str)
        layer_dir = base_out / args.model / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#' * 60}")
        print(f"# {args.model.upper()} LAYER {layer_idx} -> {layer_dir}")
        print(f"{'#' * 60}")

        timesteps, feature_ids, matrix = build_matrix(layer_data)
        print(f"Matrix: {matrix.shape} ({len(feature_ids)} features x "
              f"{len(timesteps)} timesteps)")

        classified = classify_features(timesteps, feature_ids, matrix,
                                       diffusion_steps)

        # Print summary
        cats = defaultdict(int)
        for f in classified:
            cats[f["category"]] += 1
        for c, n in sorted(cats.items()):
            print(f"  {c}: {n}")

        plot_feature_profiles(timesteps, feature_ids, matrix, classified,
                              layer_idx, layer_dir, args.model)

        if args.save_json:
            json_path = layer_dir / "classification.json"
            with open(json_path, "w") as f:
                json.dump(classified, f, indent=2)
            print(f"  Saved: {json_path}")

    print(f"\nAll plots saved under {base_out}/{args.model}/")


if __name__ == "__main__":
    main()
