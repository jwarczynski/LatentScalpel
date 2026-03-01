# Feature Classification Categories

This document explains how SAE features are classified by their temporal activation patterns during the denoising trajectory. The classification is performed by `scripts/analyze_trajectory_features.py` and `scripts/plot_trajectory_organized.py`.

## How Classification Works

The denoising trajectory is split into three regions based on timestep position:
- **Early region**: first 25% of steps (high noise)
- **Middle region**: middle 50% of steps
- **Late region**: last 25% of steps (near-clean output)

For each feature, the script computes:
- What fraction of total activation mass falls in each region
- Where the peak activation occurs (as a fraction of total steps)
- Whether the peak is sharp (spike) or broad (plateau)

## Categories

### early_only
- **Criterion**: >60% of activation mass in the first quarter, peak in the first 30% of steps
- **Behavior**: Fires strongly at the start of denoising (high noise) and fades out gradually
- **Shape**: Broad early activation, not necessarily a sharp spike

### starting_spike
- **Criterion**: Peak in the first 15% of steps AND peak value > 2x the feature's mean activation
- **Behavior**: Sharp transient burst right at the beginning of denoising
- **Difference from early_only**: starting_spike is a sharp, narrow burst; early_only is a broader sustained activation in the early phase. A feature can be early_only without being spiky.

### midpoint_transition
- **Criterion**: >50% of activation mass in the middle half, peak between 30-70% of steps
- **Behavior**: Activates primarily during the middle phase of denoising
- **Note**: These are the "midpoint features" used in intervention experiments. They represent features that track the transition from noisy to clean representations.

### late_only
- **Criterion**: >60% of activation mass in the last quarter, peak after 70% of steps
- **Behavior**: Inactive during early denoising, activates as the output approaches clean text/structure

### finishing_spike
- **Criterion**: Peak after 85% of steps AND peak value > 2x the feature's mean activation
- **Behavior**: Sharp burst at the very end of denoising, near the final output
- **Difference from late_only**: Same relationship as starting_spike vs early_only — finishing_spike is a sharp narrow burst, late_only is broader sustained late activation.

### other
- **Criterion**: Doesn't match any of the above patterns
- **Behavior**: Mixed or flat activation profiles, no clear temporal specialization

## How Features Are Chosen for Plots

Two filtering stages determine which features appear on heatmap and trajectory plots:

1. **Collection-time filtering (`top_k_to_record`)**: During trajectory collection, only the top-K most active features per token position per timestep are recorded (default: 64). Features that never rank in the top-K at any timestep are not present in the trajectory JSON at all. This is a hard upper bound on what can appear in any downstream plot.

2. **Plot-time filtering (max activation > 0.01)**: When generating the heatmap, the plotting script (`scripts/plot_trajectory_organized.py`) further filters to "active" features — those whose maximum activation across all timesteps exceeds 0.01. Features with negligible activations everywhere are excluded.

The combination of these two filters is why the heatmap typically shows far fewer features than the total SAE dictionary size. For example, a PLAID layer-0 SAE with 32,768 features may show only ~125 on the heatmap because most features either never rank in the top-64 during collection or have max activation below 0.01.

## Model-Specific Notes

- **Genie**: 2000 diffusion steps. Midpoint = step 1000. Trajectory sampled at 101 timesteps.
- **PLAID**: 256 diffusion steps. Midpoint = step 128. Trajectory sampled at all 256 steps.
- Classification thresholds are relative (percentages), so they apply consistently across different step counts.
