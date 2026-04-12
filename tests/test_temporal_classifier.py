"""Property-based and unit tests for geniesae.temporal_classifier."""

from __future__ import annotations

import math

from hypothesis import given, settings, strategies as st

from geniesae.temporal_classifier import (
    VALID_CATEGORIES,
    TemporalClassifier,
    TemporalProfile,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for diffusion_steps (at least 2 so normalization is meaningful)
_diffusion_steps_st = st.integers(min_value=3, max_value=4000)

# Strategy for the number of sampled timesteps in the trajectory
_n_timesteps_st = st.integers(min_value=3, max_value=30)

# Strategy for the number of features per timestep
_n_features_st = st.integers(min_value=1, max_value=20)


@st.composite
def trajectory_data_st(draw: st.DrawFn) -> dict:
    """Generate a random trajectory data dict with non-zero activation profiles.

    Every generated feature has at least one activation value > 0.01 so that
    the classifier will not skip it as a dead feature.
    """
    diffusion_steps = draw(_diffusion_steps_st)
    # n_timesteps must not exceed diffusion_steps (available unique values)
    max_ts = min(30, diffusion_steps)
    n_timesteps = draw(st.integers(min_value=3, max_value=max_ts))
    n_features = draw(_n_features_st)

    # Pick unique sorted timesteps in [0, diffusion_steps)
    timesteps = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=diffusion_steps - 1),
                min_size=n_timesteps,
                max_size=n_timesteps,
                unique=True,
            )
        )
    )

    feature_ids = list(range(n_features))

    # Build layer data — ensure every feature has at least one non-trivial value
    layer_data: dict[str, dict[str, float]] = {}
    for t in timesteps:
        feats: dict[str, float] = {}
        for fid in feature_ids:
            val = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
            feats[str(fid)] = val
        layer_data[str(t)] = feats

    # Guarantee every feature has total activation >= 0.01 by boosting one
    # random timestep per feature if needed.
    for fid in feature_ids:
        total = sum(layer_data[str(t)].get(str(fid), 0.0) for t in timesteps)
        if total < 0.01:
            boost_t = draw(st.sampled_from(timesteps))
            layer_data[str(boost_t)][str(fid)] = draw(
                st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False)
            )

    return {
        "metadata": {"diffusion_steps": diffusion_steps},
        "layers": {"0": layer_data},
    }


# ---------------------------------------------------------------------------
# Property 1: Classification completeness and validity
# ---------------------------------------------------------------------------


class TestClassificationCompletenessAndValidity:
    """Property 1: Classification completeness and validity.

    For any trajectory data containing features with non-zero activations,
    TemporalClassifier.classify() shall map every such feature to a
    TemporalProfile containing exactly one valid category and all required
    statistics as finite floats.

    **Validates: Requirements 1.1, 1.2, 1.4**
    """

    @given(data=trajectory_data_st())
    @settings(max_examples=200, deadline=None)
    def test_every_feature_classified_with_valid_category_and_finite_stats(
        self, data: dict
    ) -> None:
        # Feature: temporal-feature-analysis, Property 1: Classification completeness and validity
        classifier = TemporalClassifier()
        result = classifier.classify(data, layer=0)

        # Collect feature ids that have non-trivial total activation (>= 1e-6)
        layer_data = data["layers"]["0"]
        timesteps = sorted(int(t) for t in layer_data.keys())
        all_features: set[int] = set()
        for feats in layer_data.values():
            all_features.update(int(f) for f in feats.keys())

        for fid in all_features:
            total = sum(
                layer_data[str(t)].get(str(fid), 0.0) for t in timesteps
            )
            if total < 1e-6:
                # Dead features are allowed to be absent
                continue

            # Every non-dead feature must be classified
            assert fid in result, (
                f"Feature {fid} has total activation {total} but was not classified"
            )

            profile = result[fid]

            # Must be a TemporalProfile
            assert isinstance(profile, TemporalProfile)

            # Exactly one valid category
            assert profile.category in VALID_CATEGORIES, (
                f"Feature {fid} has invalid category '{profile.category}'"
            )

            # All statistics must be finite floats
            stats = {
                "mean_activation": profile.mean_activation,
                "peak_nds": profile.peak_nds,
                "coefficient_of_variation": profile.coefficient_of_variation,
                "first_half_mean": profile.first_half_mean,
                "second_half_mean": profile.second_half_mean,
                "midpoint_activation": profile.midpoint_activation,
                "midpoint_to_outside_ratio": profile.midpoint_to_outside_ratio,
            }
            for stat_name, stat_val in stats.items():
                assert isinstance(stat_val, float), (
                    f"Feature {fid}: {stat_name} is {type(stat_val)}, expected float"
                )
                assert math.isfinite(stat_val), (
                    f"Feature {fid}: {stat_name} = {stat_val} is not finite"
                )

            # peak_nds_raw must be an int (the raw timestep)
            assert isinstance(profile.peak_nds_raw, int)

            # feature_id must match the key
            assert profile.feature_id == fid

# ---------------------------------------------------------------------------
# Property 2: Midpoint exclusive classification correctness
# ---------------------------------------------------------------------------


@st.composite
def midpoint_exclusive_trajectory_st(draw: st.DrawFn) -> dict:
    """Generate trajectory data that satisfies midpoint_exclusive conditions.

    Strategy: build evenly-spaced timesteps so that the array midpoint
    corresponds to the NDS midpoint, then place a large spike at the
    array midpoint and near-zero values everywhere else.

    Guarantees:
    - Peak activation timestep has normalized NDS within 10% of 0.5
    - Midpoint activation to outside-midpoint mean ratio > 10
    """
    # Use an odd number of timesteps so there's a clear center element
    n_timesteps = draw(st.integers(min_value=11, max_value=31)) | 1  # ensure odd
    diffusion_steps = draw(st.integers(min_value=100, max_value=4000))

    # Build evenly-spaced timesteps across [0, diffusion_steps-1]
    timesteps = [
        int(round(i * (diffusion_steps - 1) / (n_timesteps - 1)))
        for i in range(n_timesteps)
    ]
    # Deduplicate while preserving order (can happen with small diffusion_steps)
    timesteps = sorted(set(timesteps))
    n_ts = len(timesteps)
    if n_ts < 5:
        # Not enough unique timesteps; use a safe fallback
        timesteps = [0, diffusion_steps // 4, diffusion_steps // 2,
                     3 * diffusion_steps // 4, diffusion_steps - 1]
        timesteps = sorted(set(t for t in timesteps if 0 <= t < diffusion_steps))
        n_ts = len(timesteps)

    mid_idx = n_ts // 2
    peak_timestep = timesteps[mid_idx]

    # High activation at the midpoint, near-zero elsewhere
    peak_val = draw(st.floats(min_value=50.0, max_value=200.0,
                              allow_nan=False, allow_infinity=False))
    max_outside = peak_val / 100.0  # ensures ratio >> 10

    layer_data: dict[str, dict[str, float]] = {}
    for arr_idx, t in enumerate(timesteps):
        if arr_idx == mid_idx:
            val = peak_val
        else:
            val = draw(st.floats(min_value=0.0, max_value=max_outside,
                                 allow_nan=False, allow_infinity=False))
        layer_data[str(t)] = {"0": val}

    return {
        "metadata": {"diffusion_steps": diffusion_steps},
        "layers": {"0": layer_data},
    }


class TestMidpointExclusiveClassification:
    """Property 2: Midpoint exclusive classification correctness.

    For any activation profile where the peak activation occurs within 10%
    of the midpoint NDS and the ratio of midpoint activation to
    outside-midpoint mean exceeds 10, the TemporalClassifier shall classify
    the feature as midpoint_exclusive.

    **Validates: Requirements 1.3**
    """

    @given(data=midpoint_exclusive_trajectory_st())
    @settings(max_examples=200, deadline=None)
    def test_midpoint_exclusive_classification(self, data: dict) -> None:
        # Feature: temporal-feature-analysis, Property 2: Midpoint exclusive classification correctness
        classifier = TemporalClassifier()
        result = classifier.classify(data, layer=0)

        assert 0 in result, (
            f"Feature 0 was not classified. Keys: {list(result.keys())}"
        )

        profile = result[0]

        # Verify the preconditions hold for this generated data
        assert abs(profile.peak_nds - 0.5) <= 0.10, (
            f"peak_nds {profile.peak_nds} not within 10% of midpoint"
        )
        assert profile.midpoint_to_outside_ratio > 10.0, (
            f"midpoint_to_outside_ratio {profile.midpoint_to_outside_ratio} not > 10"
        )

        # The property: must be classified as midpoint_exclusive
        assert profile.category == "midpoint_exclusive", (
            f"Expected 'midpoint_exclusive' but got '{profile.category}' "
            f"(peak_nds={profile.peak_nds}, ratio={profile.midpoint_to_outside_ratio})"
        )


# ---------------------------------------------------------------------------
# Property 3: NDS normalization range
# ---------------------------------------------------------------------------


class TestNDSNormalizationRange:
    """Property 3: NDS normalization range.

    For any non-negative integer timestep and positive integer total_steps
    where timestep < total_steps, normalize_nds(timestep, total_steps) shall
    return a float in the closed interval [0.0, 1.0].

    **Validates: Requirements 8.1, 1.5**
    """

    @given(
        total_steps=st.integers(min_value=1, max_value=100_000),
        data=st.data(),
    )
    @settings(max_examples=200, deadline=None)
    def test_normalize_nds_returns_float_in_unit_interval(
        self, total_steps: int, data: st.DataObject
    ) -> None:
        # Feature: temporal-feature-analysis, Property 3: NDS normalization range
        timestep = data.draw(
            st.integers(min_value=0, max_value=total_steps - 1),
            label="timestep",
        )

        result = TemporalClassifier.normalize_nds(timestep, total_steps)

        assert isinstance(result, float), (
            f"Expected float, got {type(result)} for "
            f"timestep={timestep}, total_steps={total_steps}"
        )
        assert 0.0 <= result <= 1.0, (
            f"normalize_nds({timestep}, {total_steps}) = {result} "
            f"is outside [0.0, 1.0]"
        )

