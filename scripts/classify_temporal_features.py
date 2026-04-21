#!/usr/bin/env python3
"""Run TemporalClassifier on a trajectory JSON and save per-layer profiles.

Handles both GENIE (diffusion_steps) and PLAID (sampling_timesteps) metadata.

Usage:
    uv run python scripts/classify_temporal_features.py \
        experiments/results/plaid_finetuned_v3b/trajectory_features.json \
        --output_dir experiments/results/plaid_finetuned_v3b/temporal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geniesae.temporal_classifier import TemporalClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_path", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--midpoint_ratio", type=float, default=10.0)
    parser.add_argument("--midpoint_window_pct", type=float, default=0.10)
    args = parser.parse_args()

    with open(args.trajectory_path) as f:
        trajectory_data = json.load(f)

    # Bridge PLAID metadata: sampling_timesteps -> diffusion_steps
    meta = trajectory_data.get("metadata", {})
    if "diffusion_steps" not in meta and "sampling_timesteps" in meta:
        meta["diffusion_steps"] = meta["sampling_timesteps"]

    classifier = TemporalClassifier(
        midpoint_ratio_threshold=args.midpoint_ratio,
        midpoint_window_pct=args.midpoint_window_pct,
    )

    layers = sorted(int(li) for li in trajectory_data.get("layers", {}))
    print(f"Classifying {len(layers)} layers: {layers}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        result = classifier.classify_to_json(trajectory_data, layer)
        n_feats = len(result["features"])
        category_counts: dict[str, int] = {}
        for profile in result["features"].values():
            cat = profile["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"Layer {layer:02d}: {n_feats} features classified")
        for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {cnt}")

        out_path = out_dir / f"layer_{layer:02d}_temporal_profiles.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
