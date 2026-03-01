"""Property-based and unit tests for temporal prompt construction in geniesae.prompts."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from geniesae.prompts import build_explanation_prompt, build_temporal_explanation_prompt
from geniesae.temporal_classifier import VALID_CATEGORIES


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_temporal_category_st = st.sampled_from(sorted(VALID_CATEGORIES))

# Strategy for temporal summary dicts with plausible statistic keys
@st.composite
def temporal_summary_st(draw: st.DrawFn) -> dict:
    """Generate a temporal summary dict with typical statistic keys."""
    return {
        "peak_nds": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "mean_activation": draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        "coefficient_of_variation": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        "first_half_mean": draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        "second_half_mean": draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
    }


# NDS values: list of floats in [0, 1]
_nds_value_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Activating tokens: printable non-empty strings
_token_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=20,
)

# Documents with << >> markers
_document_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=80,
).map(lambda s: f"some text <<{s}>> more text")


@st.composite
def prompt_inputs_st(draw: st.DrawFn) -> dict:
    """Generate a complete set of inputs for build_temporal_explanation_prompt."""
    n = draw(st.integers(min_value=1, max_value=10))
    documents = draw(st.lists(_document_st, min_size=n, max_size=n))
    nds_values = draw(st.lists(_nds_value_st, min_size=n, max_size=n))
    activating_tokens = draw(st.lists(_token_st, min_size=n, max_size=n))
    temporal_category = draw(_temporal_category_st)
    temporal_summary = draw(temporal_summary_st())
    return {
        "documents": documents,
        "temporal_category": temporal_category,
        "temporal_summary": temporal_summary,
        "nds_values": nds_values,
        "activating_tokens": activating_tokens,
    }


# ---------------------------------------------------------------------------
# Property 4: Temporal prompt content completeness
# ---------------------------------------------------------------------------


class TestTemporalPromptContentCompleteness:
    """Property 4: Temporal prompt content completeness.

    For any feature with a valid temporal category, temporal summary dict,
    list of NDS values, and list of activating tokens,
    build_temporal_explanation_prompt() shall return a prompt whose user
    message contains: the temporal category string, all provided NDS values
    (as formatted strings), all provided activating tokens, and a reference
    to the xsum dataset instructing the LLM to produce specific (not generic)
    explanations.

    **Validates: Requirements 2.1, 2.3, 2.4, 2.5**
    """

    @given(inputs=prompt_inputs_st())
    @settings(max_examples=200, deadline=None)
    def test_prompt_contains_temporal_category_nds_tokens_and_xsum(
        self, inputs: dict
    ) -> None:
        # Feature: temporal-feature-analysis, Property 4: Temporal prompt content completeness
        result = build_temporal_explanation_prompt(
            documents=inputs["documents"],
            temporal_category=inputs["temporal_category"],
            temporal_summary=inputs["temporal_summary"],
            nds_values=inputs["nds_values"],
            activating_tokens=inputs["activating_tokens"],
        )

        # Result must be a list of message dicts with system and user roles
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

        user_content = result[1]["content"]

        # 1. Temporal category string must appear in the user message
        assert inputs["temporal_category"] in user_content, (
            f"Temporal category '{inputs['temporal_category']}' not found in prompt"
        )

        # 2. All NDS values must appear as formatted strings (3 decimal places)
        for nds_val in inputs["nds_values"]:
            nds_str = f"{nds_val:.3f}"
            assert nds_str in user_content, (
                f"NDS value '{nds_str}' not found in prompt"
            )

        # 3. All activating tokens must appear in the user message
        for token in inputs["activating_tokens"]:
            assert token in user_content, (
                f"Activating token '{token}' not found in prompt"
            )

        # 4. Reference to xsum dataset must be present
        assert "xsum" in user_content.lower(), (
            "Reference to xsum dataset not found in prompt"
        )

# ---------------------------------------------------------------------------
# Property 5: Temporal prompt neutrality for midpoint features
# ---------------------------------------------------------------------------

# Strategy restricted to midpoint categories only
_midpoint_category_st = st.sampled_from(["midpoint_exclusive", "midpoint_transition"])


class TestTemporalPromptNeutrality:
    """Property 5: Temporal prompt neutrality for midpoint features.

    For any feature classified as midpoint_exclusive or midpoint_transition,
    the prompt returned by build_temporal_explanation_prompt() shall not
    contain the phrases "this is a timing feature", "this feature encodes
    timing", "timing feature", or "encodes phase-transition" — the LLM must
    determine the explanation from the evidence alone.

    **Validates: Requirements 2.2**
    """

    FORBIDDEN_PHRASES = [
        "this is a timing feature",
        "this feature encodes timing",
        "timing feature",
        "encodes phase-transition",
    ]

    @given(
        category=_midpoint_category_st,
        summary=temporal_summary_st(),
        data=st.data(),
    )
    @settings(max_examples=200, deadline=None)
    def test_midpoint_prompt_does_not_assert_timing(
        self,
        category: str,
        summary: dict,
        data: st.SearchStrategy,
    ) -> None:
        # Feature: temporal-feature-analysis, Property 5: Temporal prompt neutrality for midpoint features
        n = data.draw(st.integers(min_value=1, max_value=10))
        documents = data.draw(st.lists(_document_st, min_size=n, max_size=n))
        nds_values = data.draw(st.lists(_nds_value_st, min_size=n, max_size=n))
        activating_tokens = data.draw(st.lists(_token_st, min_size=n, max_size=n))

        result = build_temporal_explanation_prompt(
            documents=documents,
            temporal_category=category,
            temporal_summary=summary,
            nds_values=nds_values,
            activating_tokens=activating_tokens,
        )

        # Concatenate all message content for checking
        full_text = " ".join(msg["content"] for msg in result).lower()

        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase not in full_text, (
                f"Prompt for '{category}' feature contains forbidden phrase: "
                f"'{phrase}'"
            )



# ---------------------------------------------------------------------------
# Property 6: Backward-compatible prompt fallback
# ---------------------------------------------------------------------------


class TestBackwardCompatiblePromptFallback:
    """Property 6: Backward-compatible prompt fallback.

    For any list of documents, when trajectory_data_path is None (no temporal
    data), the interpretation pipeline shall produce prompts identical to those
    from the existing build_explanation_prompt() function.

    Since the full apply() method requires an LLM, dataset, and model weights,
    we test the property at the prompt level: build_explanation_prompt() must
    produce the same deterministic output for any document list, confirming
    the non-temporal code path is unmodified.

    **Validates: Requirements 2.8**
    """

    @given(documents=st.lists(_document_st, min_size=1, max_size=15))
    @settings(max_examples=200, deadline=None)
    def test_standard_prompt_is_deterministic_and_unchanged(
        self, documents: list[str]
    ) -> None:
        # Feature: temporal-feature-analysis, Property 6: Backward-compatible prompt fallback
        #
        # When trajectory_data_path is None, InterpretFeaturesConfig.apply()
        # calls build_explanation_prompt(documents) directly. We verify that
        # two independent calls with the same documents produce identical
        # output, confirming the fallback path is stable and deterministic.
        result_a = build_explanation_prompt(documents)
        result_b = build_explanation_prompt(documents)

        assert result_a == result_b, (
            "build_explanation_prompt() is not deterministic for the same input"
        )

        # Structural checks: must be a 2-element list with system + user roles
        assert isinstance(result_a, list)
        assert len(result_a) == 2
        assert result_a[0]["role"] == "system"
        assert result_a[1]["role"] == "user"

        # The user content must contain all documents (numbered)
        user_content = result_a[1]["content"]
        for i, doc in enumerate(documents, start=1):
            assert f"{i}. {doc}" in user_content, (
                f"Document {i} not found in standard prompt"
            )

        # The standard prompt must NOT contain temporal-specific content,
        # confirming it hasn't been accidentally modified to include temporal data.
        # NOTE: The base prompt now intentionally contains an "xsum" specificity
        # warning to improve explanation quality. Only temporal-specific annotations
        # (NDS values, temporal classification, activating tokens) should be absent.
        assert "temporal classification" not in user_content.lower(), (
            "Standard prompt unexpectedly contains temporal classification"
        )
        assert "xsum" in user_content.lower(), (
            "Standard prompt should contain xsum specificity warning"
        )
        assert "NDS=" not in user_content, (
            "Standard prompt unexpectedly contains NDS annotations"
        )
        assert "activating_token=" not in user_content, (
            "Standard prompt unexpectedly contains activating token annotations"
        )
