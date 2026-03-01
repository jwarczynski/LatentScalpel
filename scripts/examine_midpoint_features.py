#!/usr/bin/env python3
"""Deep examination of midpoint (~step 999) features from trajectory data.

Cross-references with interpretation results and top examples to understand
what these phase-boundary detector features actually encode.

Also examines early-only, late-only, and finishing-spike features.

Usage:
    python scripts/examine_midpoint_features.py \
        --trajectory experiments/trajectory_features.json \
        --interp-l0 experiments/interpretation_results_layer_00.json \
        --interp-l5 experiments/interpretation_results_layer_05.json \
        --top-examples-l0 experiments/results/top_examples/layer_00_top_examples.json \
        --top-examples-l5 experiments/results/top_examples/layer_05_top_examples.json \
        --out-dir experiments/plots/midpoint_analysis
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        print(f"  [skip] {path} not found")
        return None
    with open(p) as f:
        return json.load(f)


def build_matrix(layer_data: dict):
    """Build (features x timesteps) matrix."""
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


def classify_feature(vals, timesteps, mid_idx):
    """Classify a single feature's temporal profile."""
    total = vals.sum()
    if total < 1e-6:
        return "dead", {}

    n = len(vals)
    mid = n // 2
    fh_mean = vals[:mid].mean()
    sh_mean = vals[mid:].mean()
    last_10 = vals[int(n * 0.9):].mean()
    first_10 = vals[:int(n * 0.1)].mean()
    mean = vals.mean()
    var = vals.var()
    cv = np.sqrt(var) / (mean + 1e-10)
    peak_idx = np.argmax(vals)
    peak_step = timesteps[peak_idx]

    # Midpoint-exclusive: high at midpoint, near-zero elsewhere
    mid_val = vals[mid_idx]
    outside = np.concatenate([vals[:max(0, mid_idx - 2)], vals[mid_idx + 3:]])
    outside_mean = outside.mean() if len(outside) > 0 else 0

    half_ratio = (fh_mean + 1e-8) / (sh_mean + 1e-8)
    finishing_ratio = (last_10 + 1e-8) / (mean + 1e-8)
    early_ratio = (first_10 + 1e-8) / (mean + 1e-8)

    info = dict(
        mean=float(mean), var=float(var), cv=float(cv),
        fh_mean=float(fh_mean), sh_mean=float(sh_mean),
        mid_val=float(mid_val), outside_mean=float(outside_mean),
        peak_step=int(peak_step), half_ratio=float(half_ratio),
    )

    if mid_val > 0.01 and (outside_mean < 0.001 or mid_val / (outside_mean + 1e-10) > 10):
        return "midpoint_exclusive", info
    if cv < 0.3:
        return "stable", info
    if half_ratio > 3.0:
        return "early_only", info
    if half_ratio < 0.33:
        return "late_only", info
    if finishing_ratio > 2.5:
        return "finishing_spike", info
    if early_ratio > 2.5:
        return "starting_spike", info
    if abs(np.log(half_ratio)) > 0.5:
        return "midpoint_transition", info
    return "variable", info


def analyze_layer(
    layer_idx: int,
    layer_data: dict,
    interp_data: dict | None,
    top_examples_data: dict | None,
    out_dir: Path,
):
    """Full analysis of one layer's trajectory features."""
    timesteps, feature_ids, matrix = build_matrix(layer_data)
    mid_idx = min(range(len(timesteps)), key=lambda i: abs(timesteps[i] - 999))

    # Classify all features
    categories = defaultdict(list)
    for i, fid in enumerate(feature_ids):
        cat, info = classify_feature(matrix[i], timesteps, mid_idx)
        if cat != "dead":
            categories[cat].append((fid, info, matrix[i]))

    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx} — {len(feature_ids)} features")
    print(f"{'='*80}")
    for cat in ["midpoint_exclusive", "early_only", "late_only",
                 "finishing_spike", "starting_spike", "stable",
                 "midpoint_transition", "variable"]:
        feats = categories.get(cat, [])
        print(f"  {cat}: {len(feats)}")

    # --- MIDPOINT EXCLUSIVE FEATURES ---
    midpoint_feats = categories.get("midpoint_exclusive", [])
    midpoint_feats.sort(key=lambda x: x[1]["mid_val"], reverse=True)

    print(f"\n--- MIDPOINT-EXCLUSIVE FEATURES ({len(midpoint_feats)}) ---")
    print("These features fire almost exclusively at step ~999 (SNR ≈ 1 boundary)")
    for fid, info, vals in midpoint_feats[:30]:
        line = f"  F{fid:>6d}  mid_act={info['mid_val']:.4f}  outside={info['outside_mean']:.6f}"
        if interp_data and str(fid) in interp_data:
            interp = interp_data[str(fid)]
            expl = interp.get("explanation", "")[:100]
            score = interp.get("interpretability_score", "?")
            line += f"\n           interp(score={score}): {expl}"
        if top_examples_data and str(fid) in top_examples_data:
            examples = top_examples_data[str(fid)]
            if isinstance(examples, list) and len(examples) > 0:
                # Show first example snippet
                ex = examples[0]
                text = ex.get("text", ex.get("tokens", ""))
                if isinstance(text, str):
                    text = text[:120]
                line += f"\n           top_example: {text}..."
        print(line)

    # --- Plot midpoint features zoomed in ---
    if midpoint_feats:
        _plot_midpoint_zoom(timesteps, midpoint_feats[:15], mid_idx, layer_idx, out_dir)
        _plot_midpoint_activation_distribution(midpoint_feats, layer_idx, out_dir)

    # --- EARLY-ONLY and LATE-ONLY with interpretations ---
    for cat_name in ["early_only", "late_only", "finishing_spike"]:
        feats = categories.get(cat_name, [])
        if not feats:
            continue
        feats.sort(key=lambda x: x[1]["var"], reverse=True)
        print(f"\n--- {cat_name.upper()} ({len(feats)}) — top 15 by variance ---")
        for fid, info, vals in feats[:15]:
            line = f"  F{fid:>6d}  mean={info['mean']:.4f}  var={info['var']:.4f}  peak@{info['peak_step']}"
            if interp_data and str(fid) in interp_data:
                interp = interp_data[str(fid)]
                expl = interp.get("explanation", "")[:100]
                score = interp.get("interpretability_score", "?")
                line += f"\n           interp(score={score}): {expl}"
            print(line)

    return categories


def _plot_midpoint_zoom(timesteps, midpoint_feats, mid_idx, layer_idx, out_dir):
    """Zoomed plot of midpoint features around step 999."""
    # Zoom window: ±10 timestep indices around midpoint
    window = 10
    lo = max(0, mid_idx - window)
    hi = min(len(timesteps), mid_idx + window + 1)
    zoom_ts = timesteps[lo:hi]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: zoomed profiles
    cmap = plt.cm.get_cmap("tab20", 20)
    for rank, (fid, info, vals) in enumerate(midpoint_feats):
        axes[0].plot(zoom_ts, vals[lo:hi], "o-", label=f"F{fid}",
                     color=cmap(rank % 20), linewidth=1.5, markersize=3)
    axes[0].set_xlabel("Denoising timestep")
    axes[0].set_ylabel("Mean activation")
    axes[0].set_title(f"Layer {layer_idx}: Midpoint features — zoomed around step 999")
    axes[0].invert_xaxis()
    axes[0].axvline(x=timesteps[mid_idx], color="red", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=6, ncol=2)

    # Right: full trajectory for top 5
    for rank, (fid, info, vals) in enumerate(midpoint_feats[:5]):
        axes[1].plot(timesteps, vals, label=f"F{fid}", color=cmap(rank % 20), linewidth=1.5)
    axes[1].set_xlabel("Denoising timestep (T → 0)")
    axes[1].set_ylabel("Mean activation")
    axes[1].set_title(f"Layer {layer_idx}: Top 5 midpoint features — full trajectory")
    axes[1].invert_xaxis()
    axes[1].axvline(x=timesteps[mid_idx], color="red", linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    save_path = out_dir / f"layer_{layer_idx:02d}_midpoint_zoom.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_midpoint_activation_distribution(midpoint_feats, layer_idx, out_dir):
    """Distribution of midpoint activation magnitudes."""
    mid_vals = [info["mid_val"] for _, info, _ in midpoint_feats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mid_vals, bins=30, color="coral", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Activation magnitude at step 999")
    ax.set_ylabel("Count")
    ax.set_title(f"Layer {layer_idx}: Distribution of midpoint-exclusive feature activations\n"
                 f"({len(midpoint_feats)} features)")
    ax.axvline(x=np.median(mid_vals), color="blue", linestyle="--",
               label=f"median={np.median(mid_vals):.3f}")
    ax.legend()
    plt.tight_layout()
    save_path = out_dir / f"layer_{layer_idx:02d}_midpoint_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Examine midpoint features")
    parser.add_argument("--trajectory", default="experiments/trajectory_features.json")
    parser.add_argument("--interp-l0", default="experiments/interpretation_results_layer_00.json")
    parser.add_argument("--interp-l5", default="experiments/interpretation_results_layer_05.json")
    parser.add_argument("--top-examples-l0", default="experiments/results/top_examples/layer_00_top_examples.json")
    parser.add_argument("--top-examples-l5", default="experiments/results/top_examples/layer_05_top_examples.json")
    parser.add_argument("--out-dir", default="experiments/plots/midpoint_analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = load_json(args.trajectory)
    if traj is None:
        print("ERROR: trajectory data not found")
        return

    interp_l0_raw = load_json(args.interp_l0)
    interp_l5_raw = load_json(args.interp_l5)
    top_ex_l0 = load_json(args.top_examples_l0)
    top_ex_l5 = load_json(args.top_examples_l5)

    interp_l0 = interp_l0_raw.get("features", {}) if interp_l0_raw else None
    interp_l5 = interp_l5_raw.get("features", {}) if interp_l5_raw else None
    top_l0 = top_ex_l0.get("features", top_ex_l0) if top_ex_l0 else None
    top_l5 = top_ex_l5.get("features", top_ex_l5) if top_ex_l5 else None

    layers_data = traj["layers"]

    if "0" in layers_data:
        cats_l0 = analyze_layer(0, layers_data["0"], interp_l0, top_l0, out_dir)
    if "5" in layers_data:
        cats_l5 = analyze_layer(5, layers_data["5"], interp_l5, top_l5, out_dir)

    # --- Summary comparison ---
    print(f"\n{'='*80}")
    print("CROSS-LAYER COMPARISON")
    print(f"{'='*80}")
    if "0" in layers_data and "5" in layers_data:
        for cat in ["midpoint_exclusive", "early_only", "late_only", "stable"]:
            n0 = len(cats_l0.get(cat, []))
            n5 = len(cats_l5.get(cat, []))
            print(f"  {cat}: L0={n0}, L5={n5}")

    # --- Save feature lists for further analysis ---
    summary = {}
    for layer_str in ["0", "5"]:
        if layer_str not in layers_data:
            continue
        cats = cats_l0 if layer_str == "0" else cats_l5
        interp = interp_l0 if layer_str == "0" else interp_l5
        layer_summary = {}
        for cat, feats in cats.items():
            layer_summary[cat] = []
            for fid, info, _ in feats:
                entry = {"feature_id": fid, **info}
                if interp and str(fid) in interp:
                    entry["explanation"] = interp[str(fid)].get("explanation", "")
                    entry["interpretability_score"] = interp[str(fid)].get("interpretability_score")
                layer_summary[cat].append(entry)
        summary[f"layer_{layer_str}"] = layer_summary

    summary_path = out_dir / "midpoint_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
