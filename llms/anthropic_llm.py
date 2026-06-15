"""
llm/anthropic_llm.py
════════════════════
Anthropic Claude provider for MoodBite.

Wraps the Anthropic Python SDK behind the LLMBase interface.  Supports
all Claude 3.x and Claude 3.5 models.

Key differences from OpenAI
────────────────────────────
The Anthropic Messages API separates the system prompt from the
conversation turns.  Our incoming messages list uses the universal
OpenAI-style format (with role "system"), so AnthropicLLM.complete()
splits the list:

    messages[0] with role="system"  → system= parameter
    remaining messages              → messages= parameter

If no system message is found in the list, the system parameter is
set to an empty string (Anthropic allows this).

Usage
─────
    from llm.anthropic_llm import AnthropicLLM

    llm = AnthropicLLM.from_settings()
    response = llm.complete([
        {"role": "system",    "content": "You are a food recommender."},
        {"role": "user",      "content": "I'm feeling romantic, what should I eat?"},
    ])
    print(response)
"""

from __future__ import annotations

import logging
from typing import Any

from llms.base import LLMBase, Message

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMBase):
    """
    Anthropic Claude messages wrapper.

    Parameters
    ──────────
    api_key     Anthropic API key (sk-ant-...).
    model       Model name e.g. "claude-3-5-sonnet-20241022",
                "claude-3-opus-20240229", "claude-3-haiku-20240307".
    temperature Sampling temperature 0.0–1.0.
    max_tokens  Maximum tokens to generate per response.
    """

    def __init__(
        self,
        api_key:     str,
        model:       str   = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens:  int   = 1024,
    ) -> None:
        # Store even if empty — deferred check runs in _get_client()
        self._api_key     = (api_key or "").strip()
        self._model       = model
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._client: Any = None   # lazy-initialised on first call

        logger.info(f"AnthropicLLM configured: model={model}  temp={temperature}")

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings=None) -> "AnthropicLLM":
        """Build from application Settings object."""
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        return cls(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "AnthropicLLM":
        """Build from a raw config dict."""
        return cls(
            api_key=cfg.get("anthropic_api_key", ""),
            model=cfg.get("anthropic_model", "claude-3-5-sonnet-20241022"),
            temperature=float(cfg.get("llm_temperature", 0.7)),
            max_tokens=int(cfg.get("llm_max_tokens", 1024)),
        )

    # ── Internal client ───────────────────────────────────────────────────────

    def _get_client(self):
        """Lazy-init the Anthropic client on first use."""
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic LLM. "
                "Run: pip install anthropic"
            )
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file and restart the app."
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        logger.debug("Anthropic client initialised.")
        return self._client

    # ── Core complete() ───────────────────────────────────────────────────────

    def complete(self, messages: list[Message]) -> str:
        """
        Send messages to the Anthropic Messages API.

        Splits the OpenAI-style message list into:
            system   — the first message with role "system"
            messages — all remaining user/assistant turns

        Args:
            messages: OpenAI-style message list.

        Returns:
            The model's text response as a string. Returns "" on empty reply.

        Raises:
            RuntimeError: if the API key is invalid.
            ImportError:  if the anthropic package is not installed.
        """
        return self._call_with_retry(self._do_complete, messages)

    def _do_complete(self, messages: list[Message]) -> str:
        """Single (non-retried) API call."""
        client = self._get_client()

        # Split system message from conversation turns
        system_content = ""
        conv_messages: list[Message] = []

        for m in messages:
            if m.get("role") == "system":
                system_content = m.get("content", "")
            else:
                conv_messages.append({
                    "role":    m["role"],
                    "content": m.get("content", ""),
                })

        # Anthropic requires at least one user message
        if not conv_messages:
            conv_messages = [{"role": "user", "content": "Hello"}]

        # Ensure conversation alternates correctly (user/assistant/user…)
        conv_messages = self._fix_alternation(conv_messages)

        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_content,
            messages=conv_messages,
        )

        # Log token usage
        if hasattr(response, "usage"):
            usage = response.usage
            logger.debug(
                f"Anthropic usage — "
                f"input={usage.input_tokens}  "
                f"output={usage.output_tokens}  "
                f"model={response.model}"
            )

        if response.content:
            return response.content[0].text.strip()
        return ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fix_alternation(messages: list[Message]) -> list[Message]:
        """
        Ensure messages strictly alternate user/assistant.

        Anthropic's API rejects consecutive messages from the same role.
        When two consecutive user messages appear (e.g. after history
        trimming), we merge them into one.

        Args:
            messages: List of {role, content} dicts (no system messages).

        Returns:
            List where roles strictly alternate starting with "user".
        """
        if not messages:
            return messages

        fixed: list[Message] = []
        for msg in messages:
            if fixed and fixed[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages
                fixed[-1] = {
                    "role":    fixed[-1]["role"],
                    "content": fixed[-1]["content"] + "\n\n" + msg["content"],
                }
            else:
                fixed.append(dict(msg))

        # Must start with user
        if fixed and fixed[0]["role"] != "user":
            fixed.insert(0, {"role": "user", "content": "(conversation start)"})

        return fixed

    # ── info() ────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return configuration summary dict."""
        return {
            "provider":    "anthropic",
            "model":       self._model,
            "temperature": self._temperature,
            "max_tokens":  self._max_tokens,
            "api_key_set": bool(self._api_key),
        }