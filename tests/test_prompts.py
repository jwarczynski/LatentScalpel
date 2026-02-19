"""Unit tests for geniesae.prompts — prompt builders, response parsing, and scoring."""

from __future__ import annotations

from geniesae.prompts import (
    build_explanation_prompt,
    build_scoring_prompt,
    compute_interpretability_score,
    parse_scoring_response,
    _EXPLANATION_SYSTEM_MESSAGE,
    _SCORING_SYSTEM_MESSAGE,
)


# ---------------------------------------------------------------------------
# Tests: build_explanation_prompt
# ---------------------------------------------------------------------------


class TestBuildExplanationPrompt:
    def test_returns_two_messages(self):
        msgs = build_explanation_prompt(["doc one"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_message_is_exact_appendix_d(self):
        msgs = build_explanation_prompt(["x"])
        assert msgs[0]["content"] == _EXPLANATION_SYSTEM_MESSAGE

    def test_single_document_numbered(self):
        msgs = build_explanation_prompt(["hello world"])
        assert "1. hello world" in msgs[1]["content"]

    def test_multiple_documents_numbered_sequentially(self):
        docs = ["alpha", "bravo", "charlie"]
        msgs = build_explanation_prompt(docs)
        user = msgs[1]["content"]
        assert "1. alpha" in user
        assert "2. bravo" in user
        assert "3. charlie" in user

    def test_user_message_starts_with_preamble(self):
        msgs = build_explanation_prompt(["x"])
        assert msgs[1]["content"].startswith("The activating documents are given below:")

    def test_preserves_angle_bracket_markers(self):
        doc = "The <<cat>> sat on the mat"
        msgs = build_explanation_prompt([doc])
        assert "<<cat>>" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Tests: build_scoring_prompt
# ---------------------------------------------------------------------------


class TestBuildScoringPrompt:
    def test_returns_two_messages(self):
        msgs = build_scoring_prompt("activates on sports", ["ex1"])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_message_is_exact_appendix_d(self):
        msgs = build_scoring_prompt("x", ["y"])
        assert msgs[0]["content"] == _SCORING_SYSTEM_MESSAGE

    def test_user_message_contains_explanation(self):
        msgs = build_scoring_prompt("activates on numbers", ["ex"])
        assert "activates on numbers" in msgs[1]["content"]

    def test_examples_numbered_sequentially(self):
        examples = ["foo", "bar", "baz"]
        msgs = build_scoring_prompt("test", examples)
        user = msgs[1]["content"]
        assert "1. foo" in user
        assert "2. bar" in user
        assert "3. baz" in user

    def test_user_message_format(self):
        msgs = build_scoring_prompt("some explanation", ["a", "b"])
        user = msgs[1]["content"]
        assert user.startswith("Here is the explanation: some explanation.")
        assert "Here are the examples:" in user


# ---------------------------------------------------------------------------
# Tests: parse_scoring_response
# ---------------------------------------------------------------------------


class TestParseScoringResponse:
    def test_none_returns_empty_set(self):
        assert parse_scoring_response("None", 10) == set()

    def test_none_with_whitespace(self):
        assert parse_scoring_response("  None  ", 10) == set()

    def test_single_index(self):
        assert parse_scoring_response("3", 5) == {3}

    def test_comma_separated_with_spaces(self):
        assert parse_scoring_response("2, 5, 7", 10) == {2, 5, 7}

    def test_comma_separated_no_spaces(self):
        assert parse_scoring_response("1,3,5", 10) == {1, 3, 5}

    def test_index_out_of_range_high(self):
        assert parse_scoring_response("11", 10) is None

    def test_index_out_of_range_zero(self):
        assert parse_scoring_response("0", 10) is None

    def test_index_out_of_range_negative(self):
        assert parse_scoring_response("-1", 10) is None

    def test_non_integer_content(self):
        assert parse_scoring_response("abc", 10) is None

    def test_mixed_valid_invalid(self):
        assert parse_scoring_response("1, abc, 3", 10) is None

    def test_empty_string(self):
        assert parse_scoring_response("", 10) is None

    def test_whitespace_stripped(self):
        assert parse_scoring_response("  2, 5  ", 10) == {2, 5}


# ---------------------------------------------------------------------------
# Tests: compute_interpretability_score
# ---------------------------------------------------------------------------


class TestComputeInterpretabilityScore:
    def test_perfect_score(self):
        predicted = {1, 2, 3}
        ground_truth = {1, 2, 3}
        assert compute_interpretability_score(predicted, ground_truth, 5) == 1.0

    def test_zero_score(self):
        # Predict all wrong: predict {1,2} when truth is {3,4} out of 4
        predicted = {1, 2}
        ground_truth = {3, 4}
        # TP=0, TN=0 (all indices are in one set or the other)
        assert compute_interpretability_score(predicted, ground_truth, 4) == 0.0

    def test_partial_score(self):
        # total=4, predicted={1,2}, truth={1,3}
        # TP=1 (index 1), TN=1 (index 4), FP=1 (index 2), FN=1 (index 3)
        score = compute_interpretability_score({1, 2}, {1, 3}, 4)
        assert score == 0.5

    def test_empty_predicted_and_truth(self):
        # All are true negatives
        assert compute_interpretability_score(set(), set(), 5) == 1.0

    def test_empty_predicted_nonempty_truth(self):
        # truth={1,2}, predicted={}. total=5
        # TP=0, TN=3 (indices 3,4,5)
        assert compute_interpretability_score(set(), {1, 2}, 5) == 3 / 5

    def test_total_zero_returns_zero(self):
        assert compute_interpretability_score(set(), set(), 0) == 0.0

    def test_score_in_unit_interval(self):
        score = compute_interpretability_score({1, 3}, {2, 4}, 10)
        assert 0.0 <= score <= 1.0
