"""Property-based tests for geniesae.feature_intervention."""

from __future__ import annotations

import torch
from hypothesis import given, settings, strategies as st

from geniesae.feature_intervention import _intervene_on_sparse_code
from geniesae.sae import TopKSAE


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_float_st = st.floats(
    min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
)


@st.composite
def sparse_code_and_intervention_st(draw: st.DrawFn) -> dict:
    """Generate a random sparse code tensor and intervention parameters."""
    dict_size = draw(st.integers(min_value=4, max_value=32))
    batch_size = draw(st.integers(min_value=1, max_value=4))

    # Random sparse code
    n = batch_size * dict_size
    values = draw(st.lists(_float_st, min_size=n, max_size=n))
    sparse_code = torch.tensor(values, dtype=torch.float32).reshape(batch_size, dict_size)

    # Pick 1-3 feature indices to intervene on
    n_features = draw(st.integers(min_value=1, max_value=min(3, dict_size)))
    feature_indices = sorted(draw(
        st.lists(
            st.integers(min_value=0, max_value=dict_size - 1),
            min_size=n_features,
            max_size=n_features,
            unique=True,
        )
    ))

    target_magnitude = draw(
        st.floats(min_value=-10.0, max_value=10.0,
                  allow_nan=False, allow_infinity=False)
    )

    return {
        "sparse_code": sparse_code,
        "feature_indices": feature_indices,
        "target_magnitude": target_magnitude,
        "dict_size": dict_size,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Property 7: Feature intervention encode-modify-decode
# ---------------------------------------------------------------------------


class TestFeatureInterventionEncodeModifyDecode:
    """Property 7: Feature intervention encode-modify-decode.

    For any sparse code tensor, when _intervene_on_sparse_code sets specified
    feature indices to a target magnitude, the modified sparse code shall have
    exactly the target value at those indices, the original tensor shall be
    unchanged, and all non-intervened features shall retain their original values.

    This tests the core encode-modify-decode contract: the modification step
    correctly and precisely sets feature activations. The full round-trip
    (decode → re-encode) is inherently lossy due to TopK selection and is
    validated via integration tests on GPU.

    **Validates: Requirements 3.2, 3.5, 3.6, 3.7**
    """

    @given(data=sparse_code_and_intervention_st())
    @settings(max_examples=200, deadline=None)
    def test_intervene_sets_target_and_preserves_others(
        self, data: dict
    ) -> None:
        # Feature: temporal-feature-analysis, Property 7: Feature intervention encode-modify-decode
        sparse_code = data["sparse_code"]
        feature_indices = data["feature_indices"]
        target_magnitude = data["target_magnitude"]

        original = sparse_code.clone()

        modified = _intervene_on_sparse_code(
            sparse_code, feature_indices, target_magnitude
        )

        # 1. Original tensor must be unchanged (no mutation)
        assert torch.equal(sparse_code, original), (
            "Original sparse code was mutated by _intervene_on_sparse_code"
        )

        # 2. Modified tensor must have target value at intervened indices
        for idx in feature_indices:
            actual = modified[..., idx].item() if modified.dim() == 1 else None
            for b in range(modified.shape[0]):
                val = modified[b, idx].item()
                assert abs(val - target_magnitude) < 1e-6, (
                    f"Feature {idx} batch {b}: expected {target_magnitude}, "
                    f"got {val}"
                )

        # 3. Non-intervened features must be unchanged
        intervened_set = set(feature_indices)
        for idx in range(data["dict_size"]):
            if idx not in intervened_set:
                assert torch.equal(modified[..., idx], original[..., idx]), (
                    f"Non-intervened feature {idx} was modified"
                )

    @given(
        target=st.floats(min_value=0.0, max_value=0.0),
        data=sparse_code_and_intervention_st(),
    )
    @settings(max_examples=100, deadline=None)
    def test_suppression_zeros_out_features(
        self, target: float, data: dict
    ) -> None:
        """Suppression (target=0.0) zeroes out the specified features."""
        # Feature: temporal-feature-analysis, Property 7: Feature intervention encode-modify-decode
        modified = _intervene_on_sparse_code(
            data["sparse_code"], data["feature_indices"], 0.0
        )

        for idx in data["feature_indices"]:
            for b in range(modified.shape[0]):
                assert modified[b, idx].item() == 0.0, (
                    f"Feature {idx} batch {b} not zeroed: {modified[b, idx].item()}"
                )
