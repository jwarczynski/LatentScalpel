#!/usr/bin/env python3
"""Show LLM judge interpretations for top correlated feature pairs."""
import json

pairs = [
    (1700, 7352, 1.0000),
    (5559, 7352, 1.0000),
    (95, 10239, 0.7071),
    (777, 10570, 0.5792),
    (15002, 10230, 0.5773),
    (16198, 10230, 0.5773),
    (13162, 8832, 0.5773),
    (11309, 10570, 0.5169),
    (14308, 10570, 0.5102),
]

with open("experiments/results/t5/interpretation_results_layer_00.json") as f:
    t5_interp = json.load(f)["features"]

with open("experiments/results/genie/interpretation_results_layer_00.json") as f:
    genie_interp = json.load(f)["features"]

print("=" * 80)
print("TOP CORRELATED FEATURE PAIRS (r > 0.5) - Genie L0 vs T5 L0")
print("=" * 80)

for t5_f, g_f, r in pairs:
    print(f"\n--- T5 feature {t5_f} <-> Genie feature {g_f} (r={r:.4f}) ---")

    t5_data = t5_interp.get(str(t5_f))
    if t5_data:
        print(f"  T5 F{t5_f} explanation: {t5_data['explanation']}")
        print(f"  T5 F{t5_f} score: {t5_data['interpretability_score']}")
    else:
        print(f"  T5 F{t5_f}: not in interpretation results")

    g_data = genie_interp.get(str(g_f))
    if g_data:
        print(f"  Genie F{g_f} explanation: {g_data['explanation']}")
        print(f"  Genie F{g_f} score: {g_data['interpretability_score']}")
    else:
        print(f"  Genie F{g_f}: not in interpretation results")
