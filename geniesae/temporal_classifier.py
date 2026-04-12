"""Temporal feature classification for diffusion SAE trajectory data.

Classifies SAE features into temporal categories based on their activation
profiles across denoising timesteps. Supports both Genie (discrete 2000-step)
and PLAID (continuous learned schedule) trajectory data via NDS normalization.

Categories:
    midpoint_exclusive: Peak within 10% of midpoint, ratio to outside > threshold
    early_only: Active in first half (high noise), dies in second half
    late_only: Inactive in first half, activates in second half (low noise)
    midpoint_transition: Sharp change around the midpoint
    finishing_spike: Spike in the last ~10% of denoising (near step 0)
    starting_spike: Spike in the first ~10% of denoising (near max noise)
    stable: Relatively constant activation throughout
    variable: High variance but no clear pattern
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


VALID_CATEGORIES = frozenset({
    "midpoint_exclusive",
    "early_only",
    "late_only",
    "midpoint_transition",
    "finishing_spike",
    "starting_spike",
    "stable",
    "variable",
})


@dataclass
class TemporalProfile:
    """Classification result for a single SAE feature."""

    feature_id: int
    category: str
    mean_activation: float
    peak_nds: float  # normalized [0, 1]
    peak_nds_raw: int  # original timestep index
    coefficient_of_variation: float
    first_half_mean: float
    second_half_mean: float
    midpoint_activation: float
    midpoint_to_outside_ratio: float


class TemporalClassifier:
    """Classify SAE features by their temporal activation patterns.

    Args:
        midpoint_ratio_threshold: Ratio of midpoint activation to outside-midpoint
            mean required for ``midpoint_exclusive`` classification.
        midpoint_window_pct: Fraction of total steps defining the midpoint window
            (applied symmetrically around the midpoint).
    """

    def __init__(
        self,
        midpoint_ratio_threshold: float = 10.0,
        midpoint_window_pct: float = 0.10,
    ) -> None:
        self.midpoint_ratio_threshold = midpoint_ratio_threshold
        self.midpoint_window_pct = midpoint_window_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_nds(timestep: int, total_steps: int) -> float:
        """Normalize a timestep to the [0.0, 1.0] range.

        Convention: 0 → 0.0 (clean), ``total_steps - 1`` → 1.0 (max noise).

        Args:
            timestep: Non-negative integer timestep.
            total_steps: Positive integer total number of diffusion steps.

        Returns:
            Float in [0.0, 1.0].
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if total_steps == 1:
            return 0.0
        return float(timestep) / float(total_steps - 1)

    def classify(
        self, trajectory_data: dict, layer: int
    ) -> dict[int, TemporalProfile]:
        """Classify all features in a layer's trajectory data.

        Args:
            trajectory_data: Loaded trajectory JSON (from ``collect-trajectory``
                or ``collect-plaid-trajectory``).
            layer: Layer index to classify.

        Returns:
            Mapping from feature index to :class:`TemporalProfile`.

        Raises:
            ValueError: If the trajectory data is missing expected keys.
        """
        self._validate_trajectory_data(trajectory_data)

        layers_data = trajectory_data["layers"]
        layer_key = str(layer)

        if layer_key not in layers_data:
            return {}

        layer_data = layers_data[layer_key]
        if not layer_data:
            return {}

        metadata = trajectory_data["metadata"]
        diffusion_steps: int = metadata["diffusion_steps"]

        timesteps, feature_ids, matrix = self._build_matrix(layer_data)

        if matrix.size == 0:
            return {}

        return self._classify_matrix(
            timesteps, feature_ids, matrix, diffusion_steps
        )

    def classify_to_json(self, trajectory_data: dict, layer: int) -> dict:
        """Classify and return a JSON-serializable dict.

        The output matches the classification output schema from the design doc.

        Args:
            trajectory_data: Loaded trajectory JSON.
            layer: Layer index to classify.

        Returns:
            Dict with ``metadata`` and ``features`` keys.
        """
        profiles = self.classify(trajectory_data, layer)

        metadata = trajectory_data.get("metadata", {})

        features_json: dict[str, dict] = {}
        for fid, profile in profiles.items():
            d = asdict(profile)
            d.pop("feature_id")  # redundant with the key
            features_json[str(fid)] = d

        return {
            "metadata": {
                "layer": layer,
                "total_features_classified": len(profiles),
                "midpoint_ratio_threshold": self.midpoint_ratio_threshold,
                "midpoint_window_pct": self.midpoint_window_pct,
                "diffusion_steps": metadata.get("diffusion_steps"),
            },
            "features": features_json,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_trajectory_data(trajectory_data: dict) -> None:
        """Raise ``ValueError`` if the trajectory data is malformed."""
        if not isinstance(trajectory_data, dict):
            raise ValueError("Trajectory data must be a dict")
        if "metadata" not in trajectory_data:
            raise ValueError(
                "Trajectory data missing 'metadata' key — unknown format"
            )
        if "layers" not in trajectory_data:
            raise ValueError(
                "Trajectory data missing 'layers' key — unknown format"
            )
        meta = trajectory_data["metadata"]
        if "diffusion_steps" not in meta:
            raise ValueError(
                "Trajectory metadata missing 'diffusion_steps'"
            )

    @staticmethod
    def _build_matrix(
        layer_data: dict[str, dict[str, float]],
    ) -> tuple[np.ndarray, list[int], np.ndarray]:
        """Build a (features × timesteps) matrix from layer data.

        Returns:
            Tuple of (timesteps array, feature_ids list, matrix).
        """
        timesteps = sorted(int(t) for t in layer_data.keys())
        all_features: set[int] = set()
        for feats in layer_data.values():
            all_features.update(int(f) for f in feats.keys())
        feature_ids = sorted(all_features)

        if not feature_ids or not timesteps:
            return np.array([], dtype=int), [], np.empty((0, 0))

        fid_to_row = {fid: i for i, fid in enumerate(feature_ids)}
        matrix = np.zeros((len(feature_ids), len(timesteps)))

        for t_idx, t_val in enumerate(timesteps):
            feats = layer_data.get(str(t_val), {})
            for f_str, val in feats.items():
                fid = int(f_str)
                if fid in fid_to_row:
                    matrix[fid_to_row[fid], t_idx] = val

        return np.array(timesteps), feature_ids, matrix

    def _classify_matrix(
        self,
        timesteps: np.ndarray,
        feature_ids: list[int],
        matrix: np.ndarray,
        diffusion_steps: int,
    ) -> dict[int, TemporalProfile]:
        """Classify every feature in the matrix."""
        n_features, n_timesteps = matrix.shape
        mid_idx = n_timesteps // 2

        # Midpoint window indices
        window_half = max(1, int(n_timesteps * self.midpoint_window_pct / 2))
        mid_lo = max(0, mid_idx - window_half)
        mid_hi = min(n_timesteps, mid_idx + window_half + 1)

        results: dict[int, TemporalProfile] = {}

        for i, fid in enumerate(feature_ids):
            row = matrix[i]
            total = row.sum()
            if total < 1e-6:
                continue  # skip dead features

            row_mean = float(row.mean())
            row_std = float(row.std())
            cv = row_std / (row_mean + 1e-10)

            first_half_mean = float(row[:mid_idx].mean()) if mid_idx > 0 else 0.0
            second_half_mean = float(row[mid_idx:].mean()) if mid_idx < n_timesteps else 0.0

            # Midpoint region stats
            midpoint_activation = float(row[mid_lo:mid_hi].mean())
            outside_indices = np.concatenate([
                np.arange(0, mid_lo),
                np.arange(mid_hi, n_timesteps),
            ])
            if len(outside_indices) > 0:
                outside_mean = float(row[outside_indices].mean())
            else:
                outside_mean = 0.0
            midpoint_to_outside_ratio = midpoint_activation / (outside_mean + 1e-10)

            # Peak info
            peak_step_idx = int(np.argmax(row))
            peak_step_raw = int(timesteps[peak_step_idx])
            peak_nds = self.normalize_nds(peak_step_raw, diffusion_steps)

            # Ratios for classification
            last_10pct_idx = int(n_timesteps * 0.9)
            first_10pct_idx = max(1, int(n_timesteps * 0.1))
            last_10pct_mean = float(row[last_10pct_idx:].mean()) if last_10pct_idx < n_timesteps else 0.0
            first_10pct_mean = float(row[:first_10pct_idx].mean())

            half_ratio = (first_half_mean + 1e-8) / (second_half_mean + 1e-8)
            finishing_ratio = (last_10pct_mean + 1e-8) / (row_mean + 1e-8)
            early_ratio = (first_10pct_mean + 1e-8) / (row_mean + 1e-8)

            # Check midpoint_exclusive first (highest priority)
            midpoint_nds = 0.5
            peak_within_midpoint_window = abs(peak_nds - midpoint_nds) <= self.midpoint_window_pct
            is_midpoint_exclusive = (
                peak_within_midpoint_window
                and midpoint_to_outside_ratio > self.midpoint_ratio_threshold
            )

            # Classify
            if is_midpoint_exclusive:
                category = "midpoint_exclusive"
            elif cv < 0.3:
                category = "stable"
            elif half_ratio > 3.0:
                category = "early_only"
            elif half_ratio < 0.33:
                category = "late_only"
            elif finishing_ratio > 2.5:
                category = "finishing_spike"
            elif early_ratio > 2.5:
                category = "starting_spike"
            elif abs(np.log(half_ratio)) > 0.5:
                category = "midpoint_transition"
            else:
                category = "variable"

            results[fid] = TemporalProfile(
                feature_id=fid,
                category=category,
                mean_activation=row_mean,
                peak_nds=peak_nds,
                peak_nds_raw=peak_step_raw,
                coefficient_of_variation=cv,
                first_half_mean=first_half_mean,
                second_half_mean=second_half_mean,
                midpoint_activation=midpoint_activation,
                midpoint_to_outside_ratio=midpoint_to_outside_ratio,
            )

        return results
