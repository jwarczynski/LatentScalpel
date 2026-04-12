"""Property-based tests for geniesae.configs.intervention_config."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from geniesae.configs.intervention_config import InterventionConfig
from geniesae.temporal_classifier import VALID_CATEGORIES


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# All valid categories as a sorted list for sampling
_all_categories = sorted(VALID_CATEGORIES)

# Non-midpoint categories for mixing into classification data
_non_midpoint_categories = sorted(VALID_CATEGORIES - {"midpoint_exclusive"})


@st.composite
def classification_json_st(draw: st.DrawFn) -> dict:
    """Generate a classification JSON dict with various feature categories.

    The dict follows the classification output schema:
    {
        "metadata": {...},
        "features": {
            "<feature_id>": {"category": "<category>", ...},
            ...
        }
    }
    """
    n_features = draw(st.integers(min_value=0, max_value=30))

    # Generate unique feature IDs (as strings, matching the JSON format)
    feature_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=9999),
            min_size=n_features,
            max_size=n_features,
            unique=True,
        )
    )

    features: dict[str, dict] = {}
    for fid in feature_ids:
        category = draw(st.sampled_from(_all_categories))
        features[str(fid)] = {
            "category": category,
            "mean_activation": draw(
                st.floats(min_value=0.0, max_value=10.0,
                           allow_nan=False, allow_infinity=False)
            ),
            "peak_nds": draw(
                st.floats(min_value=0.0, max_value=1.0,
                           allow_nan=False, allow_infinity=False)
            ),
        }

    return {
        "metadata": {
            "layer": 0,
            "total_features_classified": n_features,
        },
        "features": features,
    }


# ---------------------------------------------------------------------------
# Property 8: Auto feature selection from classification
# ---------------------------------------------------------------------------


class TestAutoFeatureSelectionFromClassification:
    """Property 8: Auto feature selection from classification.

    For any classification JSON containing features with various categories,
    when feature_selection is "auto", the InterventionConfig shall select
    exactly the set of feature indices whose category is midpoint_exclusive.

    **Validates: Requirements 4.5**
    """

    @given(data=classification_json_st())
    @settings(max_examples=200, deadline=None)
    def test_auto_selects_exactly_midpoint_exclusive_features(
        self, data: dict
    ) -> None:
        # Feature: temporal-feature-analysis, Property 8: Auto feature selection from classification
        result = InterventionConfig.select_midpoint_exclusive_features(data)

        # Compute expected: all feature indices with midpoint_exclusive category
        expected = sorted(
            int(fid)
            for fid, fdata in data.get("features", {}).items()
            if fdata.get("category") == "midpoint_exclusive"
        )

        assert result == expected, (
            f"Expected midpoint_exclusive features {expected}, got {result}"
        )
