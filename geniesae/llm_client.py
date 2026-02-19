"""Unified LLM client for vLLM API and offline inference.

Provides a thin wrapper around two backends:

- **API mode** (``base_url`` set): Uses the ``openai`` Python client to talk
  to a vLLM server via its OpenAI-compatible HTTP API.
- **Offline mode** (``base_url`` is ``None``): Uses ``vllm.LLM`` for local
  batched inference.

Imports for ``openai`` and ``vllm`` are lazy so the module can be imported
even when only one backend is installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("geniesae.llm_client")


class LLMClient:
    """Unified interface for vLLM API and offline inference."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._kwargs = kwargs

        if base_url is not None:
            # API mode — lazy import openai
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url)
            self._mode = "api"
            logger.info("LLMClient: API mode, base_url=%s, model=%s", base_url, model)
        else:
            # Offline mode — lazy import vllm
            from vllm import LLM

            self._llm = LLM(model=model, **kwargs)
            self._mode = "offline"
            logger.info("LLMClient: offline mode, model=%s", model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Single chat completion. Returns the generated text."""
        if self._mode == "api":
            return self._generate_api(messages, max_tokens, temperature)
        return self._generate_offline([messages], max_tokens, temperature)[0]

    def generate_batch(
        self,
        prompts: list[list[dict]],
        max_tokens: int,
        temperature: float,
    ) -> list[str]:
        """Batched chat completions. Returns list of generated texts."""
        if self._mode == "api":
            return [
                self._generate_api(msgs, max_tokens, temperature)
                for msgs in prompts
            ]
        return self._generate_offline(prompts, max_tokens, temperature)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_api(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Send a single request via the OpenAI-compatible API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _generate_offline(
        self,
        prompts: list[list[dict]],
        max_tokens: int,
        temperature: float,
    ) -> list[str]:
        """Run batched inference via vllm.LLM.chat()."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs = self._llm.chat(
            messages=prompts,
            sampling_params=sampling_params,
        )
        return [output.outputs[0].text for output in outputs]
