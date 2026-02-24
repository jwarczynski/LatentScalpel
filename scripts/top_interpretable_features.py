#!/usr/bin/env python3
"""Show top X features sorted by interpretability score, filtered by min predicted indices.

Usage:
    python scripts/top_interpretable_features.py results.json --top 20 --min-predicted 3
    python scripts/top_interpretable_features.py results.json  # defaults: top=10, min-predicted=1
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Top interpretable features from interpretation results.")
    parser.add_argument("results_path", help="Path to interpretation_results.json")
    parser.add_argument("--top", "-x", type=int, default=10, help="Number of top features to show (default: 10)")
    parser.add_argument("--min-predicted", "-m", type=int, default=1, help="Min number of predicted indices required (default: 1)")
    args = parser.parse_args()

    with open(args.results_path) as f:
        data = json.load(f)

    features = data.get("features", {})

    # Filter: must have predicted_indices with at least m entries
    filtered = []
    for fid, info in features.items():
        preds = info.get("predicted_indices")
        if preds is not None and len(preds) >= args.min_predicted:
            filtered.append((fid, info))

    # Sort by interpretability_score descending
    filtered.sort(key=lambda x: x[1].get("interpretability_score", 0.0), reverse=True)

    top = filtered[: args.top]

    print(f"\nTop {len(top)} features (min predicted indices >= {args.min_predicted}):\n")
    for rank, (fid, info) in enumerate(top, 1):
        score = info.get("interpretability_score", 0.0)
        n_pred = len(info.get("predicted_indices", []))
        expl = info.get("explanation", "").strip()
        print(f"#{rank}  Feature {fid}  |  Score: {score:.4f}  |  #Predicted: {n_pred}")
        print(f"  {expl}")
        print()


if __name__ == "__main__":
    main()
