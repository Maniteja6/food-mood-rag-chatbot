"""
llm — LLM Provider Abstraction Package
═══════════════════════════════════════

Provides a provider-agnostic interface for calling large language models.
All providers accept the same OpenAI-style message list and return a plain
Python string.

Supported providers
───────────────────
openai      OpenAILLM      GPT-4o, GPT-4-turbo, GPT-3.5-turbo
anthropic   AnthropicLLM   Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku

Quick start
───────────
    # Read .env and return the correct provider automatically
    from llm import get_llm

    llm = get_llm()                         # from .env (LLM_PROVIDER)
    llm = get_llm("openai")                 # explicit
    llm = get_llm("anthropic")              # explicit

    response = llm.complete([
        {"role": "system",    "content": "You are a food recommender."},
        {"role": "user",      "content": "I'm stressed. What should I eat?"},
    ])

    # Or use classes directly
    from llm import OpenAILLM, AnthropicLLM, LLMBase

    llm = OpenAILLM.from_settings()
    llm = AnthropicLLM.from_settings()

Public symbols
──────────────
    LLMBase        Abstract base class — defines the interface
    OpenAILLM      OpenAI implementation
    AnthropicLLM   Anthropic implementation
    Message        Type alias  dict[str, str]
    get_llm()      Factory convenience function
"""

from llms.base          import LLMBase, Message
from llms.openai_llm    import OpenAILLM
from llms.anthropic_llm import AnthropicLLM


def get_llm(
    provider: str | None = None,
    settings=None,
) -> LLMBase:
    """
    Return a configured LLM for the given provider.

    Args:
        provider: "openai" | "anthropic" | None.
                  If None, reads LLM_PROVIDER from .env via Settings.
        settings: Optional Settings instance. Reads .env if None.

    Returns:
        Configured OpenAILLM or AnthropicLLM instance.

    Examples:
        llm = get_llm()              # from .env
        llm = get_llm("openai")      # explicit OpenAI
        llm = get_llm("anthropic")   # explicit Anthropic
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    target = (provider or settings.llm_provider.value).lower()

    if target == "openai":
        return OpenAILLM.from_settings(settings)
    elif target == "anthropic":
        return AnthropicLLM.from_settings(settings)
    else:
        raise ValueError(
            f"Unknown LLM provider '{target}'. "
            "Choose 'openai' or 'anthropic' in your .env file."
        )


__all__ = [
    "LLMBase",
    "OpenAILLM",
    "AnthropicLLM",
    "Message",
    "get_llm",
]