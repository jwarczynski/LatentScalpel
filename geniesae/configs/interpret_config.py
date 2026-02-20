"""Exca config for the feature interpretation stage.

Implements the DLM-Scope LLM-as-judge auto-interpretation protocol
(arXiv:2602.05859). For each SAE feature, generates a natural-language
explanation via an LLM and scores it via a discrimination task.

Single run (inline)::

    uv run python main.py interpret-features configs/interpret_features.yaml

Submit to Slurm::

    uv run python main.py interpret-features configs/interpret_features.yaml \\
        --submit --infra.cluster=slurm
"""

from __future__ import annotations

import json
import logging
import random
import typing as tp
from pathlib import Path

import exca
from pydantic import BaseModel, Field

logger = logging.getLogger("geniesae.configs.interpret")


class InterpretFeaturesConfig(BaseModel):
    """Interprets SAE features using the DLM-Scope LLM-as-judge protocol.

    Loads a Top_Examples_Mapping JSON (produced by ``find-top-examples``),
    retrieves the original dataset texts, and for each feature:

    1. Builds an explanation prompt from the top-activating documents.
    2. Sends it to an LLM to get a one-sentence explanation.
    3. Builds a scoring/discrimination prompt mixing activating and
       non-activating examples.
    4. Sends it to the LLM and parses the predicted indices.
    5. Computes an interpretability score (accuracy).

    Results are written as a JSON file with per-feature explanations and
    scores.
    """

    top_examples_path: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)
    vllm_base_url: str | None = None
    batch_size: int = Field(default=32, gt=0)
    num_scoring_examples: int = Field(default=10, gt=0)
    features: list[int] | None = Field(
        default=None,
        description="Subset of feature indices to interpret. None = all features.",
    )
    output_path: str = "./experiments/interpretation_results.json"
    max_tokens_explanation: int = Field(default=256, gt=0)
    max_tokens_scoring: int = Field(default=128, gt=0)
    max_doc_chars: int = Field(
        default=500,
        gt=0,
        description="Truncate each document to this many characters in prompts.",
    )
    temperature: float = Field(default=0.0, ge=0.0)
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    _exclude_from_cls_uid: tp.ClassVar[tuple[str, ...]] = (
        "batch_size",
        "vllm_base_url",
    )

    @infra.apply
    def apply(self) -> str:
        """Run the interpretation pipeline. Returns the output file path."""
        from datasets import load_dataset

        from geniesae.llm_client import LLMClient
        from geniesae.prompts import (
            build_explanation_prompt,
            build_scoring_prompt,
            compute_interpretability_score,
            parse_scoring_response,
        )

        # ------------------------------------------------------------------
        # 1. Load and validate Top_Examples_Mapping JSON
        # ------------------------------------------------------------------
        top_path = Path(self.top_examples_path)
        if not top_path.exists():
            raise FileNotFoundError(
                f"Top_Examples_Mapping file not found: {top_path}"
            )

        with open(top_path) as f:
            top_examples_data = json.load(f)

        metadata = top_examples_data.get("metadata", {})
        required_meta = ("dataset_name", "dataset_split")
        missing = [k for k in required_meta if k not in metadata]
        if missing:
            raise ValueError(
                f"Top_Examples_Mapping metadata missing required fields: {missing}"
            )

        features_data: dict[str, list[dict]] = top_examples_data.get("features", {})

        # ------------------------------------------------------------------
        # 2. Load HuggingFace dataset
        # ------------------------------------------------------------------
        dataset_name = metadata["dataset_name"]
        dataset_split = metadata["dataset_split"]

        print(
            f"[InterpretFeatures] Loading dataset {dataset_name} "
            f"(split={dataset_split})",
            flush=True,
        )
        dataset = load_dataset(dataset_name, split=dataset_split)

        # ------------------------------------------------------------------
        # 3. Initialize LLM client
        # ------------------------------------------------------------------
        llm = LLMClient(model=self.llm_model, base_url=self.vllm_base_url)

        # ------------------------------------------------------------------
        # 4. Determine which features to interpret
        # ------------------------------------------------------------------
        if self.features is not None:
            feature_keys = [str(f) for f in self.features]
            # Validate that requested features exist in the mapping
            available = set(features_data.keys())
            missing_feats = [k for k in feature_keys if k not in available]
            if missing_feats:
                logger.warning(
                    "Requested features not in top-examples mapping, skipping: %s",
                    missing_feats,
                )
                feature_keys = [k for k in feature_keys if k in available]
        else:
            feature_keys = list(features_data.keys())

        print(
            f"[InterpretFeatures] Interpreting {len(feature_keys)} features "
            f"with model={self.llm_model}",
            flush=True,
        )

        # Collect all example_ids across all features for non-activating sampling
        all_top_example_ids: set[int] = set()
        for feat_key in feature_keys:
            for entry in features_data.get(feat_key, []):
                all_top_example_ids.add(entry["example_id"])

        dataset_size = len(dataset)

        # ------------------------------------------------------------------
        # 5. Process each feature
        # ------------------------------------------------------------------
        results_features: dict[str, dict] = {}

        for feat_idx, feat_key in enumerate(feature_keys):
            top_entries = features_data.get(feat_key, [])

            if not top_entries:
                logger.info(
                    "Feature %s has no top examples, skipping.", feat_key,
                )
                continue

            # 5a. Retrieve top-example texts from dataset
            documents: list[str] = []
            top_example_ids_for_feature: set[int] = set()
            for entry in top_entries:
                eid = entry["example_id"]
                top_example_ids_for_feature.add(eid)
                if eid < dataset_size:
                    documents.append(dataset[eid]["document"])
                else:
                    logger.warning(
                        "example_id %d out of dataset range (%d), skipping",
                        eid,
                        dataset_size,
                    )

            if not documents:
                logger.info(
                    "Feature %s: no valid documents found, skipping.", feat_key,
                )
                continue

            # Truncate documents for prompt construction
            documents = [
                d[:self.max_doc_chars] + ("..." if len(d) > self.max_doc_chars else "")
                for d in documents
            ]

            # 5b. Build explanation prompt and get explanation
            explanation_prompt = build_explanation_prompt(documents)
            explanation = llm.generate(
                explanation_prompt,
                max_tokens=self.max_tokens_explanation,
                temperature=self.temperature,
            )

            # 5c. Select non-activating examples
            # Use half activating, half non-activating (or as close as possible)
            num_activating = min(
                len(top_entries), self.num_scoring_examples // 2,
            )
            num_non_activating = self.num_scoring_examples - num_activating

            # Build pool of candidate non-activating IDs (disjoint from top examples)
            candidate_ids = [
                i for i in range(dataset_size)
                if i not in top_example_ids_for_feature
            ]
            non_activating_ids = random.sample(
                candidate_ids,
                min(num_non_activating, len(candidate_ids)),
            )

            # Build scoring example lists
            activating_entries = top_entries[:num_activating]
            activating_texts = []
            for entry in activating_entries:
                eid = entry["example_id"]
                if eid < dataset_size:
                    activating_texts.append(dataset[eid]["document"])

            non_activating_texts = [
                dataset[eid]["document"] for eid in non_activating_ids
            ]

            # Truncate scoring texts
            activating_texts = [
                t[:self.max_doc_chars] + ("..." if len(t) > self.max_doc_chars else "")
                for t in activating_texts
            ]
            non_activating_texts = [
                t[:self.max_doc_chars] + ("..." if len(t) > self.max_doc_chars else "")
                for t in non_activating_texts
            ]

            # 5d. Shuffle and track ground truth
            # Each item: (text, is_activating)
            scoring_items: list[tuple[str, bool]] = [
                (t, True) for t in activating_texts
            ] + [(t, False) for t in non_activating_texts]
            random.shuffle(scoring_items)

            scoring_texts = [item[0] for item in scoring_items]
            # Ground truth: 1-based indices of activating examples
            ground_truth_indices = {
                i + 1
                for i, item in enumerate(scoring_items)
                if item[1]
            }

            total_scoring = len(scoring_texts)

            # 5e. Build scoring prompt and get prediction
            scoring_prompt = build_scoring_prompt(explanation, scoring_texts)
            scoring_response = llm.generate(
                scoring_prompt,
                max_tokens=self.max_tokens_scoring,
                temperature=self.temperature,
            )

            # 5f. Parse response and compute score
            predicted_indices = parse_scoring_response(
                scoring_response, total_scoring,
            )

            if predicted_indices is None:
                # Parse failure
                logger.warning(
                    "Feature %s: failed to parse scoring response: %r",
                    feat_key,
                    scoring_response,
                )
                results_features[feat_key] = {
                    "explanation": explanation,
                    "interpretability_score": 0.0,
                    "predicted_indices": None,
                    "ground_truth_indices": sorted(ground_truth_indices),
                }
            else:
                score = compute_interpretability_score(
                    predicted_indices, ground_truth_indices, total_scoring,
                )
                results_features[feat_key] = {
                    "explanation": explanation,
                    "interpretability_score": score,
                    "predicted_indices": sorted(predicted_indices),
                    "ground_truth_indices": sorted(ground_truth_indices),
                }

            if (feat_idx + 1) % 10 == 0 or feat_idx == len(feature_keys) - 1:
                print(
                    f"[InterpretFeatures] Processed feature {feat_key} "
                    f"({feat_idx + 1}/{len(feature_keys)})",
                    flush=True,
                )

        # ------------------------------------------------------------------
        # 6. Write results JSON
        # ------------------------------------------------------------------
        output = {
            "metadata": {
                "llm_model": self.llm_model,
                "top_examples_path": self.top_examples_path,
                "num_scoring_examples": self.num_scoring_examples,
                "num_features_interpreted": len(results_features),
            },
            "features": results_features,
        }

        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[InterpretFeatures] Wrote {out_path}", flush=True)
        return str(out_path)
