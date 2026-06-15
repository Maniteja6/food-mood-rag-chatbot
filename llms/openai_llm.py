"""
llm/openai_llm.py
═════════════════
OpenAI GPT provider for MoodBite.

Wraps the OpenAI Python SDK's chat completions endpoint behind the
LLMBase interface.  Supports all GPT-4o, GPT-4, and GPT-3.5 models.

Features
────────
- Lazy client initialisation — the OpenAI SDK is imported and the client
  object is created on the first complete() call, not at construction time.
  Safe for Streamlit's @st.cache_resource (no I/O in __init__).
- Automatic retry with exponential back-off via LLMBase._call_with_retry().
- Streaming is NOT used here — we wait for the full response and return it
  as a string.  Streamlit's chat_input handles visual streaming via reruns.
- Token usage is logged at DEBUG level for cost monitoring.

Usage
─────
    from llm.openai_llm import OpenAILLM

    llm = OpenAILLM.from_settings()          # reads .env
    response = llm.complete([
        {"role": "system",    "content": "You are a food recommender."},
        {"role": "user",      "content": "I'm tired, what should I eat?"},
    ])
    print(response)
"""

from __future__ import annotations

import logging
from typing import Any

from llms.base import LLMBase, Message

logger = logging.getLogger(__name__)


class OpenAILLM(LLMBase):
    """
    OpenAI GPT chat completions wrapper.

    Parameters
    ──────────
    api_key     OpenAI API key (sk-...).
    model       Model name e.g. "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo".
    temperature Sampling temperature 0.0–2.0.
    max_tokens  Maximum tokens to generate per response.
    """

    def __init__(
        self,
        api_key:     str,
        model:       str   = "gpt-4o",
        temperature: float = 0.7,
        max_tokens:  int   = 1024,
    ) -> None:
        # Store the key even if empty — the hard check runs in _get_client()
        # so pipeline.initialise() succeeds and the app shows a graceful
        # "API key missing" message instead of crashing with RuntimeError.
        self._api_key    = (api_key or "").strip()
        self._model      = model
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._client: Any = None   # lazy-initialised on first call

        logger.info(f"OpenAILLM configured: model={model}  temp={temperature}")

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "OpenAILLM":
        """Build from application Settings object."""
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        return cls(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "OpenAILLM":
        """Build from a raw config dict."""
        return cls(
            api_key=cfg.get("openai_api_key", ""),
            model=cfg.get("llm_model", "gpt-4o"),
            temperature=float(cfg.get("llm_temperature", 0.7)),
            max_tokens=int(cfg.get("llm_max_tokens", 1024)),
        )

    # ── Internal client ───────────────────────────────────────────────────────

    def _get_client(self):
        """Lazy-init the OpenAI client on first use."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI LLM. "
                "Run: pip install openai"
            )
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file and restart the app."
            )
        self._client = OpenAI(api_key=self._api_key)
        logger.debug("OpenAI client initialised.")
        return self._client

    # ── Core complete() ───────────────────────────────────────────────────────

    def complete(self, messages: list[Message]) -> str:
        """
        Send messages to the OpenAI chat completions API.

        Args:
            messages: OpenAI-style message list.
                      System message should be messages[0] with role "system".

        Returns:
            The model's text response as a string.  Returns "" on empty reply.

        Raises:
            RuntimeError: if the API key is invalid.
            ImportError:  if the openai package is not installed.
        """
        return self._call_with_retry(self._do_complete, messages)

    def _do_complete(self, messages: list[Message]) -> str:
        """Single (non-retried) API call."""
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Log token usage for cost visibility
        usage = response.usage
        if usage:
            logger.debug(
                f"OpenAI usage — "
                f"prompt={usage.prompt_tokens}  "
                f"completion={usage.completion_tokens}  "
                f"total={usage.total_tokens}  "
                f"model={response.model}"
            )

        content = response.choices[0].message.content
        return content.strip() if content else ""

    # ── info() ────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return configuration summary dict."""
        return {
            "provider":    "openai",
            "model":       self._model,
            "temperature": self._temperature,
            "max_tokens":  self._max_tokens,
            "api_key_set": bool(self._api_key),
        }