"""Generate a detailed report of trajectory features from Genie SAE analysis.

Produces a JSON report with sections for each temporal category (midpoint,
early-only, late-only, stable), including LLM judge explanations, per-timestep
activation profiles, and top-activating dataset examples.

Usage:
    uv run python scripts/trajectory_report.py [--output experiments/trajectory_report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_data():
    """Load all required data files."""
    with open("experiments/trajectory_features.json") as f:
        trajectory = json.load(f)
    with open("experiments/plots/trajectory_analysis/trajectory_classification.json") as f:
        classification = json.load(f)

    interp = {}
    top_examples = {}
    for layer in ["00", "05"]:
        layer_int = str(int(layer))
        ip = Path(f"experiments/interpretation_results_layer_{layer}.json")
        if ip.exists():
            with open(ip) as f:
                interp[layer_int] = json.load(f)["features"]
        tp_ = Path(f"experiments/top_examples/layer_{layer}_top_examples.json")
        if tp_.exists():
            with open(tp_) as f:
                d = json.load(f)
                top_examples[layer_int] = {
                    "metadata": d["metadata"],
                    "features": d["features"],
                }
    return trajectory, classification, interp, top_examples


def get_activation_profile(trajectory_layers: dict, layer: str, feature_id: int) -> dict:
    """Extract per-timestep activation profile for a feature."""
    layer_data = trajectory_layers.get(layer, {})
    fid_str = str(feature_id)
    profile = {}
    for ts_str, feats in sorted(layer_data.items(), key=lambda x: int(x[0])):
        if fid_str in feats:
            profile[int(ts_str)] = round(feats[fid_str], 6)
    return profile


def build_feature_entry(
    feat_class: dict,
    trajectory_layers: dict,
    interp: dict,
    top_ex: dict,
    layer: str,
) -> dict:
    """Build a detailed entry for one feature."""
    fid = feat_class["feature_id"]
    fid_str = str(fid)

    # Activation profile across timesteps
    profile = get_activation_profile(trajectory_layers, layer, fid)

    # Normalize profile (0-1 scale relative to peak)
    peak_val = max(profile.values()) if profile else 1.0
    normalized = {ts: round(v / peak_val, 4) for ts, v in profile.items()} if peak_val > 0 else {}

    # LLM interpretation
    interpretation = interp.get(layer, {}).get(fid_str, {})

    # Top examples
    examples = []
    if layer in top_ex:
        feat_examples = top_ex[layer]["features"].get(fid_str, [])
        dataset_name = top_ex[layer]["metadata"]["dataset_name"]
        dataset_split = top_ex[layer]["metadata"]["dataset_split"]
        for ex in feat_examples[:10]:
            examples.append({
                "dataset": dataset_name,
                "split": dataset_split,
                "example_id": ex["example_id"],
                "activation": round(ex["activation"], 4),
                "timestep": ex["timestep"],
                "token_position": ex["token_position"],
            })

    entry = {
        "feature_id": fid,
        "layer": int(layer),
        "category": feat_class["category"],
        "peak_step": feat_class["peak_step"],
        "max_activation": round(feat_class["max_activation"], 4),
        "mean_activation": round(feat_class["mean_activation"], 4),
        "coefficient_of_variation": round(feat_class["cv"], 4),
        "first_half_mean": round(feat_class["first_half_mean"], 6),
        "second_half_mean": round(feat_class["second_half_mean"], 6),
        "llm_explanation": interpretation.get("explanation", None),
        "interpretability_score": interpretation.get("interpretability_score", None),
        "predicted_indices": interpretation.get("predicted_indices", None),
        "ground_truth_indices": interpretation.get("ground_truth_indices", None),
        "activation_profile": profile,
        "normalized_profile": normalized,
        "top_examples": examples,
    }
    return entry


def build_report(trajectory, classification, interp, top_examples) -> dict:
    """Build the full report."""
    trajectory_layers = trajectory["layers"]

    report = {
        "metadata": {
            "description": (
                "Trajectory feature analysis for Genie diffusion model SAEs. "
                "Features are classified by their temporal activation pattern "
                "across the 2000-step diffusion process."
            ),
            "model": "GENIE (XSum)",
            "diffusion_steps": trajectory["metadata"]["diffusion_steps"],
            "timestep_subsample": trajectory["metadata"]["timestep_subsample"],
            "interpretation_model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "top_examples_dataset": "xsum",
            "top_examples_split": "validation",
        },
        "sections": {},
    }

    # Define sections with filters
    sections = [
        {
            "name": "midpoint_features",
            "title": "Midpoint Features (peak at step ~999, SNR ≈ 1)",
            "description": (
                "Features that activate most strongly at the exact midpoint of the "
                "diffusion process (step 999-1000), where the signal-to-noise ratio "
                "is approximately 1. These are 'phase boundary detectors' that fire "
                "at the critical noise→signal transition."
            ),
            "filter": lambda f: f["peak_step"] == 999,
            "sort_key": lambda f: -f["max_activation"],
        },
        {
            "name": "early_only",
            "title": "Early-Only Features (active at high noise, die off)",
            "description": (
                "Features active primarily in the early (noisy) phase of diffusion. "
                "These likely encode noise-level information, structural uncertainty, "
                "or coarse document-type signals detectable even under heavy noise."
            ),
            "filter": lambda f: f["category"] == "early_only",
            "sort_key": lambda f: -f["max_activation"],
        },
        {
            "name": "late_only",
            "title": "Late-Only Features (activate as signal emerges)",
            "description": (
                "Features that activate only in the late (clean) phase. These encode "
                "actual content and semantics that only become distinguishable as "
                "noise is removed."
            ),
            "filter": lambda f: f["category"] == "late_only" and f["peak_step"] != 999,
            "sort_key": lambda f: -f["max_activation"],
        },
        {
            "name": "stable",
            "title": "Stable Features (constant throughout diffusion)",
            "description": (
                "Features with near-constant activation across all timesteps. "
                "These likely encode positional, structural, or dataset-level "
                "information independent of noise level."
            ),
            "filter": lambda f: f["category"] == "stable",
            "sort_key": lambda f: -f["mean_activation"],
        },
        {
            "name": "midpoint_transition",
            "title": "Midpoint Transition Features",
            "description": (
                "Features that show a clear transition around the midpoint — "
                "their activation pattern changes significantly near step 999."
            ),
            "filter": lambda f: f["category"] == "midpoint_transition",
            "sort_key": lambda f: -f["max_activation"],
        },
    ]

    for section_def in sections:
        section = {
            "title": section_def["title"],
            "description": section_def["description"],
            "layers": {},
        }

        for layer_key in ["0", "5"]:
            layer_feats = classification[layer_key]
            filtered = [f for f in layer_feats if section_def["filter"](f)]
            filtered.sort(key=section_def["sort_key"])

            entries = []
            for feat in filtered[:20]:  # Top 20 per section per layer
                entry = build_feature_entry(
                    feat, trajectory_layers, interp, top_examples, layer_key,
                )
                entries.append(entry)

            section["layers"][f"layer_{layer_key}"] = {
                "total_count": len(filtered),
                "shown": len(entries),
                "features": entries,
            }

        report["sections"][section_def["name"]] = section

    # Summary statistics
    summary = {}
    for layer_key in ["0", "5"]:
        cats = {}
        for f in classification[layer_key]:
            c = f["category"]
            cats[c] = cats.get(c, 0) + 1
        step999_count = sum(1 for f in classification[layer_key] if f["peak_step"] == 999)
        summary[f"layer_{layer_key}"] = {
            "total_trajectory_features": len(classification[layer_key]),
            "category_counts": cats,
            "step_999_features": step999_count,
        }
    report["summary"] = summary

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory feature report")
    parser.add_argument(
        "--output", default="experiments/trajectory_report.json",
        help="Output path for the report JSON",
    )
    args = parser.parse_args()

    print("[TrajectoryReport] Loading data...", flush=True)
    trajectory, classification, interp, top_examples = load_data()

    print("[TrajectoryReport] Building report...", flush=True)
    report = build_report(trajectory, classification, interp, top_examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    for layer_key in ["0", "5"]:
        s = report["summary"][f"layer_{layer_key}"]
        print(f"  Layer {layer_key}: {s['total_trajectory_features']} features, "
              f"{s['step_999_features']} at step 999")
    print(f"[TrajectoryReport] Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
