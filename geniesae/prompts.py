"""Prompt construction and scoring utilities for the DLM-Scope interpretation protocol.

Implements the LLM-as-judge auto-interpretation pipeline from DLM-Scope
(arXiv:2602.05859) Appendix D. Provides pure functions for building explanation
and scoring prompts, parsing LLM responses, and computing interpretability scores.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System messages — exact phrasing from DLM-Scope Appendix D
# ---------------------------------------------------------------------------

_EXPLANATION_SYSTEM_MESSAGE = (
    "We're studying neurons in a neural network. Each neuron activates on some "
    "particular word/words/substring/concept in a short document. The activating "
    "words in each document are indicated with << ... >>. We will give you a list "
    "of documents on which the neuron activates, in order from most strongly "
    "activating to least strongly activating. Look at the parts of the document "
    "the neuron activates for and summarize in a single sentence what the neuron "
    "is activating on. Try not to be overly specific in your explanation. Note "
    "that some neurons will activate only on specific words or substrings, but "
    "others will activate on most/all words in a sentence provided that sentence "
    "contains some particular concept. Your explanation should cover most or all "
    "activating words. Pay attention to capitalization and punctuation, since they "
    "might matter."
)

_SCORING_SYSTEM_MESSAGE = (
    "We're studying neurons in a neural network. Each neuron activates on some "
    "particular word/words/substring/concept in a short document. You will be "
    "given a short explanation of what this neuron activates for, and then be "
    "shown several example sequences in random order. You must return a "
    "comma-separated list of the examples where you think the neuron should "
    "activate at least once, on ANY of the words or substrings in the document. "
    'For example, your response might look like "2, 9, 10, 12". Try not to be '
    "overly specific in your interpretation of the explanation. If you think "
    "there are no examples where the neuron will activate, you should just "
    'respond with "None". You should include nothing else in your response '
    "other than comma-separated numbers or the word \"None\" - this is important."
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_explanation_prompt(documents: list[str]) -> list[dict]:
    """Build the explanation prompt messages following DLM-Scope Appendix D.

    Args:
        documents: Top-activating documents with ``<< >>`` markers, ordered
            from most to least strongly activating.

    Returns:
        List of two message dicts ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    numbered = "\n".join(f"{i}. {doc}" for i, doc in enumerate(documents, start=1))
    user_content = f"The activating documents are given below:\n\n{numbered}"

    return [
        {"role": "system", "content": _EXPLANATION_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def build_scoring_prompt(explanation: str, examples: list[str]) -> list[dict]:
    """Build the scoring/discrimination prompt messages following DLM-Scope Appendix D.

    Args:
        explanation: The one-sentence explanation from the explanation stage.
        examples: Shuffled list of documents (mix of activating and non-activating).

    Returns:
        List of two message dicts ``[{"role": "system", ...}, {"role": "user", ...}]``.
    """
    numbered = "\n".join(f"{i}. {ex}" for i, ex in enumerate(examples, start=1))
    user_content = (
        f"Here is the explanation: {explanation}.\n\n"
        f"Here are the examples:\n\n{numbered}"
    )

    return [
        {"role": "system", "content": _SCORING_SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parsing and scoring
# ---------------------------------------------------------------------------


def parse_scoring_response(response: str, num_examples: int) -> set[int] | None:
    """Parse the LLM's scoring response into a set of predicted indices.

    Args:
        response: Raw LLM response string.
        num_examples: Total number of examples shown in the scoring prompt,
            used to validate that all indices are in ``[1, num_examples]``.

    Returns:
        A set of 1-based integer indices, an empty set if the response is
        ``"None"``, or ``None`` if the response cannot be parsed.
    """
    stripped = response.strip()

    if stripped == "None":
        return set()

    try:
        indices = {int(tok) for tok in stripped.split(",")}
    except ValueError:
        return None

    # Validate all indices are within the expected range.
    if any(idx < 1 or idx > num_examples for idx in indices):
        return None

    return indices


def compute_interpretability_score(
    predicted: set[int],
    ground_truth: set[int],
    total: int,
) -> float:
    """Compute interpretability accuracy: (TP + TN) / total.

    Args:
        predicted: Set of 1-based indices the LLM predicted as activating.
        ground_truth: Set of 1-based indices that are truly activating.
        total: Total number of scoring examples.

    Returns:
        Accuracy score in ``[0.0, 1.0]``.
    """
    if total <= 0:
        return 0.0

    all_indices = set(range(1, total + 1))
    tp = len(predicted & ground_truth)
    tn = len(all_indices - predicted - ground_truth)

    return (tp + tn) / total
