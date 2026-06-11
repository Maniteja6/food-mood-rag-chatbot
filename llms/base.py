"""
llm/base.py
═══════════
Abstract base class defining the contract every LLM provider must satisfy.

All concrete LLM wrappers (OpenAILLM, AnthropicLLM, and any future providers)
inherit from LLMBase and implement its two abstract methods.  This lets
pipeline.py work with any provider through one stable interface.

Interface
─────────
    llm.complete(messages)        → str     single-turn or multi-turn call
    llm.info()                    → dict    model name, provider, config

    LLMBase.from_settings()       → LLMBase correct subclass from .env
    LLMBase.from_config(cfg)      → LLMBase correct subclass from raw dict

Message format (OpenAI-style, universal)
────────────────────────────────────────
All providers receive the same message list:
    [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user",      "content": "..."},
    ]

Concrete implementations convert to their provider's native format internally.
Anthropic separates the system message; Google concatenates everything into a
single string.  The caller never needs to know.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# Universal message type
Message = dict[str, str]   # {"role": "system"|"user"|"assistant", "content": str}


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class LLMBase(ABC):
    """
    Abstract base class for LLM provider wrappers.

    Subclasses must implement:
        complete(messages) → str
        info()             → dict

    All subclasses inherit retry logic from _call_with_retry().
    """

    # Default retry config — subclasses may override
    MAX_RETRIES: int   = 3
    RETRY_BASE_WAIT: float = 2.0   # seconds; actual wait = base^attempt

    # ── Abstract methods ──────────────────────────────────────────────────────

    @abstractmethod
    def complete(self, messages: list[Message]) -> str:
        """
        Send a list of messages to the LLM and return the text response.

        Args:
            messages: OpenAI-style message list:
                      [{"role": "system"|"user"|"assistant", "content": str}, ...]
                      System message is always the first element when present.

        Returns:
            The model's text response as a plain Python string.
            Never returns None — returns "" on empty response.

        Raises:
            RuntimeError:  if the API key is missing or invalid.
            ImportError:   if the provider package is not installed.
            Exception:     provider-specific errors after all retries exhausted.
        """

    @abstractmethod
    def info(self) -> dict:
        """
        Return a dict describing this LLM's configuration.

        Required keys:
            provider    str   e.g. "openai"
            model       str   e.g. "gpt-4o"
            temperature float
            max_tokens  int

        Optional keys (add as needed):
            context_window  int
            supports_vision bool
        """

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "LLMBase":
        """
        Build the correct LLMBase subclass from app Settings.

        Args:
            settings: Settings instance. Reads .env if None.

        Returns:
            Configured OpenAILLM, AnthropicLLM, or GoogleLLM.
        """
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        return cls._build(settings.llm_provider.value, settings=settings)

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMBase":
        """
        Build the correct LLMBase subclass from a raw config dict.

        Expected keys (all optional, with safe defaults):
            llm_provider      str   "openai" | "anthropic" | "google"
            llm_model         str
            llm_temperature   float
            llm_max_tokens    int
            openai_api_key    str
            anthropic_api_key str
            google_api_key    str
        """
        provider = cfg.get("llm_provider", "openai").lower()
        return cls._build(provider, cfg=cfg)

    @classmethod
    def _build(
        cls,
        provider: str,
        settings: Any  = None,
        cfg:      dict = None,
    ) -> "LLMBase":
        """Internal factory dispatcher."""
        from llms.openai_llm    import OpenAILLM
        from llms.anthropic_llm import AnthropicLLM

        if provider == "openai":
            if settings:
                return OpenAILLM.from_settings(settings)
            return OpenAILLM.from_config(cfg or {})

        elif provider == "anthropic":
            if settings:
                return AnthropicLLM.from_settings(settings)
            return AnthropicLLM.from_config(cfg or {})

        elif provider == "google":
            # Google LLM uses the same interface — lazy import to avoid
            # requiring google-generativeai when not configured
            from llms.openai_llm import OpenAILLM   # placeholder until GoogleLLM is added
            logger.warning(
                "GoogleLLM is not yet implemented as a standalone class. "
                "Falling back to OpenAILLM. Set LLM_PROVIDER=openai or anthropic."
            )
            if settings:
                return OpenAILLM.from_settings(settings)
            return OpenAILLM.from_config(cfg or {})

        else:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                "Choose 'openai', 'anthropic', or 'google' in your .env."
            )

    # ── Shared retry wrapper ──────────────────────────────────────────────────

    def _call_with_retry(self, call_fn, *args, **kwargs) -> str:
        """
        Call ``call_fn(*args, **kwargs)`` with exponential back-off retry.

        Retries on all exceptions except:
            - AuthenticationError / InvalidRequestError / ValueError
              (these will not succeed on retry)

        Args:
            call_fn: Callable that makes the API request.
            *args, **kwargs: Passed through to call_fn.

        Returns:
            The string response from call_fn on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception = RuntimeError("No attempts made.")

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return call_fn(*args, **kwargs)
            except Exception as exc:                           # noqa: BLE001
                last_exc = exc
                exc_name = type(exc).__name__

                # Non-retryable — fail immediately
                if any(k in exc_name for k in (
                    "Auth", "Permission", "Invalid", "NotFound",
                    "ValueError", "ImportError",
                )):
                    raise

                if attempt == self.MAX_RETRIES:
                    logger.error(
                        f"LLM call failed after {self.MAX_RETRIES} attempts: {exc}"
                    )
                    raise

                wait = self.RETRY_BASE_WAIT ** attempt
                logger.warning(
                    f"LLM call error [{exc_name}] "
                    f"(attempt {attempt}/{self.MAX_RETRIES}): {exc}. "
                    f"Retrying in {wait:.0f}s …"
                )
                time.sleep(wait)

        raise last_exc

    # ── Convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        i = self.info()
        return (
            f"{type(self).__name__}("
            f"provider={i.get('provider')!r}, "
            f"model={i.get('model')!r})"
        )