#!/usr/bin/env python3
"""Merge trajectory job-array shard files into a single combined file.

Each shard is named layer_XX_samples_START-END.json and contains
per-timestep feature activations summed over its batch of samples.
This script averages across all shards and produces the combined format
expected by downstream scripts (visualize_trajectory.py, etc.).

Usage:
    uv run python scripts/merge_trajectory_shards.py \
        experiments/results/genie/trajectory \
        --output experiments/results/genie/trajectory_features.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def merge_shards(shard_dir: Path, output_path: Path) -> None:
    """Merge per-layer per-shard trajectory files into one combined file."""
    shard_dir = Path(shard_dir)
    pattern = re.compile(r"layer_(\d+)_samples_(\d+)-(\d+)\.json")

    # Discover all shards grouped by layer
    layer_shards: dict[int, list[Path]] = defaultdict(list)
    for f in sorted(shard_dir.iterdir()):
        m = pattern.match(f.name)
        if m:
            layer_idx = int(m.group(1))
            layer_shards[layer_idx].append(f)

    if not layer_shards:
        print(f"No shard files found in {shard_dir}")
        return

    print(f"Found shards for layers: {sorted(layer_shards.keys())}")

    # Merge: for each layer, accumulate feature activations across shards
    # then average by total number of shards (each shard already averaged
    # internally across its own batches).
    merged_layers: dict[str, dict[str, dict[str, float]]] = {}
    metadata = None

    for layer_idx in sorted(layer_shards.keys()):
        shards = layer_shards[layer_idx]
        print(f"  Layer {layer_idx}: {len(shards)} shards")

        # Accumulate: timestep -> feature_id -> sum of activations
        accumulated: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_samples = 0

        for shard_path in shards:
            with open(shard_path) as f:
                shard = json.load(f)

            shard_meta = shard["metadata"]
            n = shard_meta["num_samples"]
            total_samples += n

            if metadata is None:
                metadata = {
                    k: v for k, v in shard_meta.items()
                    if k not in ("layer", "num_samples", "sample_start", "sample_end")
                }

            for ts, features in shard["timesteps"].items():
                for fid, val in features.items():
                    # Each shard's values are already averaged over its batches.
                    # Weight by num_samples to get proper global average.
                    accumulated[ts][fid] += val * n

        # Average across all samples
        layer_data: dict[str, dict[str, float]] = {}
        for ts in sorted(accumulated.keys(), key=lambda x: int(x)):
            layer_data[ts] = {
                fid: round(val / total_samples, 6)
                for fid, val in accumulated[ts].items()
            }

        merged_layers[str(layer_idx)] = layer_data
        print(f"    {len(layer_data)} timesteps, {total_samples} total samples")

    # Build combined output
    assert metadata is not None
    metadata["num_samples"] = total_samples
    metadata["layers"] = sorted(layer_shards.keys())

    output = {
        "metadata": metadata,
        "layers": merged_layers,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nMerged trajectory saved to {output_path}")
    print(f"  Layers: {sorted(layer_shards.keys())}")
    print(f"  Total samples: {total_samples}")


def main():
    parser = argparse.ArgumentParser(description="Merge trajectory shards")
    parser.add_argument("shard_dir", help="Directory containing shard files")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: <shard_dir>/../trajectory_features.json)",
    )
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = shard_dir.parent / "trajectory_features.json"

    merge_shards(shard_dir, output_path)


if __name__ == "__main__":
    main()
