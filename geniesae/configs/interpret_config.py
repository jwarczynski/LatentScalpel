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
    vllm_kwargs: dict[str, tp.Any] = Field(
        default_factory=lambda: {
            "quantization": "awq",
            "enforce_eager": True,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.92,
        },
        description="Extra kwargs passed to vllm.LLM() in offline mode.",
    )
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
        "vllm_kwargs",
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
        llm = LLMClient(model=self.llm_model, base_url=self.vllm_base_url, **self.vllm_kwargs)

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
        # 5. Prepare per-feature data (prompts + scoring metadata)
        # ------------------------------------------------------------------
        results_features: dict[str, dict] = {}

        # Filter to active features and pre-compute everything needed
        # before any LLM calls.
        active_keys: list[str] = []
        explanation_prompts: list[list[dict]] = []
        # Per-feature scoring metadata, built now so we only need
        # the explanation text to construct the scoring prompt later.
        scoring_meta: list[dict] = []

        def _truncate(text: str) -> str:
            if len(text) > self.max_doc_chars:
                return text[: self.max_doc_chars] + "..."
            return text

        for feat_key in feature_keys:
            top_entries = features_data.get(feat_key, [])
            if not top_entries:
                continue

            # Retrieve and truncate top-example texts
            documents: list[str] = []
            top_ids: set[int] = set()
            for entry in top_entries:
                eid = entry["example_id"]
                top_ids.add(eid)
                if eid < dataset_size:
                    documents.append(_truncate(dataset[eid]["document"]))

            if not documents:
                continue

            explanation_prompts.append(build_explanation_prompt(documents))
            active_keys.append(feat_key)

            # Pre-build scoring data
            num_act = min(len(top_entries), self.num_scoring_examples // 2)
            num_non = self.num_scoring_examples - num_act

            candidate_ids = [
                i for i in range(dataset_size) if i not in top_ids
            ]
            non_act_ids = random.sample(
                candidate_ids, min(num_non, len(candidate_ids)),
            )

            act_texts = [
                _truncate(dataset[e["example_id"]]["document"])
                for e in top_entries[:num_act]
                if e["example_id"] < dataset_size
            ]
            non_act_texts = [
                _truncate(dataset[eid]["document"]) for eid in non_act_ids
            ]

            items: list[tuple[str, bool]] = [
                (t, True) for t in act_texts
            ] + [(t, False) for t in non_act_texts]
            random.shuffle(items)

            scoring_meta.append({
                "texts": [it[0] for it in items],
                "ground_truth": {
                    i + 1 for i, it in enumerate(items) if it[1]
                },
            })

        num_active = len(active_keys)
        print(
            f"[InterpretFeatures] {num_active} active features, "
            f"processing in batches of {self.batch_size}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 5b. Batched LLM calls: explanations then scoring
        # ------------------------------------------------------------------
        for batch_start in range(0, num_active, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_active)
            batch_keys = active_keys[batch_start:batch_end]
            batch_exp_prompts = explanation_prompts[batch_start:batch_end]
            batch_scoring_meta = scoring_meta[batch_start:batch_end]

            # --- Explanation batch ---
            explanations = llm.generate_batch(
                batch_exp_prompts,
                max_tokens=self.max_tokens_explanation,
                temperature=self.temperature,
            )

            # --- Scoring batch ---
            scoring_prompts = [
                build_scoring_prompt(exp, sm["texts"])
                for exp, sm in zip(explanations, batch_scoring_meta)
            ]
            scoring_responses = llm.generate_batch(
                scoring_prompts,
                max_tokens=self.max_tokens_scoring,
                temperature=self.temperature,
            )

            # --- Parse and score ---
            for i, feat_key in enumerate(batch_keys):
                explanation = explanations[i]
                gt = batch_scoring_meta[i]["ground_truth"]
                total = len(batch_scoring_meta[i]["texts"])

                predicted = parse_scoring_response(scoring_responses[i], total)

                if predicted is None:
                    logger.warning(
                        "Feature %s: failed to parse scoring response: %r",
                        feat_key, scoring_responses[i],
                    )
                    results_features[feat_key] = {
                        "explanation": explanation,
                        "interpretability_score": 0.0,
                        "predicted_indices": None,
                        "ground_truth_indices": sorted(gt),
                    }
                else:
                    score = compute_interpretability_score(predicted, gt, total)
                    results_features[feat_key] = {
                        "explanation": explanation,
                        "interpretability_score": score,
                        "predicted_indices": sorted(predicted),
                        "ground_truth_indices": sorted(gt),
                    }

            print(
                f"[InterpretFeatures] Batch done: features "
                f"{batch_start + 1}-{batch_end}/{num_active}",
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
