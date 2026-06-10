"""
config/settings.py
══════════════════
Centralised application settings for MoodBite.

All configuration is read from environment variables (via a .env file at the
repo root). Pydantic v2 BaseSettings is used so every field is:
  - Type-validated on startup — bad values raise a clear error before
    the app or pipeline touches any data.
  - Documented with a description that appears in `settings.model_json_schema()`.
  - Available as a typed Python attribute anywhere in the codebase.

Usage
─────
    from config.settings import get_settings

    cfg = get_settings()            # cached singleton — safe to call anywhere
    print(cfg.llm_model)
    print(cfg.vector_db_path)

The `get_settings()` function is decorated with `@lru_cache` so the .env
file is parsed exactly once per process, not on every import.

Environment variables → settings fields
────────────────────────────────────────
All env var names are UPPER_CASE; field names are lower_snake_case.
Pydantic maps them automatically (case-insensitive).

    LLM_PROVIDER               → llm_provider
    OPENAI_API_KEY             → openai_api_key
    VECTOR_STORE_PROVIDER      → vector_store_provider
    ...  (see fields below for full mapping)
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Graceful Pydantic import — works with both v1 and v2
# ---------------------------------------------------------------------------
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, field_validator, model_validator
    _PYDANTIC_V2 = True
except ImportError:
    try:
        # pydantic v2 without pydantic-settings installed separately
        from pydantic import BaseSettings, Field, validator as field_validator  # type: ignore[assignment]
        _PYDANTIC_V2 = False
    except ImportError:
        raise ImportError(
            "pydantic is required. Run: pip install pydantic python-dotenv\n"
            "For pydantic v2 also run: pip install pydantic-settings"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Enums — constrain provider choices to valid strings
# ─────────────────────────────────────────────────────────────────────────────

class LLMProvider(str, Enum):
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE    = "google"


class EmbeddingProvider(str, Enum):
    OPENAI     = "openai"
    HUGGINGFACE = "huggingface"


class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"
    FAISS  = "faiss"


class LogLevel(str, Enum):
    DEBUG   = "DEBUG"
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# Settings model
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    MoodBite application settings.
    All fields are populated from environment variables (case-insensitive).
    Defaults are provided for all non-secret fields so the app can start
    in demo / dry-run mode without a fully configured .env.
    """

    if _PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",           # unknown env vars are silently ignored
            validate_default=True,
        )

    # ── LLM ──────────────────────────────────────────────────────────────────

    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM backend to use: openai | anthropic | google.",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name for the selected LLM provider.",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM (0 = deterministic, 2 = very creative).",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=8192,
        description="Maximum tokens the LLM can generate per response.",
    )

    # ── API Keys ──────────────────────────────────────────────────────────────

    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Required when LLM_PROVIDER=openai or EMBEDDING_PROVIDER=openai.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Required when LLM_PROVIDER=anthropic.",
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Anthropic model name (used when LLM_PROVIDER=anthropic).",
    )
    google_api_key: str = Field(
        default="",
        description="Google Gemini API key. Required when LLM_PROVIDER=google.",
    )
    google_model: str = Field(
        default="gemini-1.5-pro",
        description="Google model name (used when LLM_PROVIDER=google).",
    )

    # ── Embedding ─────────────────────────────────────────────────────────────

    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="Embedding backend: openai | huggingface.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )
    hf_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name (local, no API key needed).",
    )

    # ── Vector Store ──────────────────────────────────────────────────────────

    vector_store_provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.CHROMA,
        description="Vector DB backend: chroma | faiss.",
    )
    vector_db_path: str = Field(
        default="./data/vector_db",
        description="Directory path where the vector database is persisted.",
    )
    chroma_collection_name: str = Field(
        default="food_mood_collection",
        description="ChromaDB collection name.",
    )
    retriever_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top documents returned by the retriever per query.",
    )
    retriever_score_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for a result to be included (0–1).",
    )

    # ── Data Paths ────────────────────────────────────────────────────────────

    food_dataset_path: str = Field(
        default="./data/raw/food_dataset.csv",
        description="Path to the raw food dataset CSV.",
    )
    mood_mapping_path: str = Field(
        default="./data/raw/mood_food_mapping.json",
        description="Path to the mood-to-food-category mapping JSON.",
    )
    processed_chunks_path: str = Field(
        default="./data/processed/chunks.json",
        description="Path to the preprocessed chunks JSON (output of ingestion).",
    )

    # ── Streamlit App ─────────────────────────────────────────────────────────

    app_title: str = Field(
        default="MoodBite — Food Recommendations by Mood",
        description="Browser tab title and Streamlit page title.",
    )
    app_description: str = Field(
        default="Tell me how you're feeling and I'll recommend the perfect meal.",
        description="App tagline displayed below the logo.",
    )
    conversation_memory_limit: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Maximum number of past messages kept in session memory for the LLM context.",
    )
    debug_mode: bool = Field(
        default=False,
        description="When True, show retrieved chunk scores and pipeline debug info in the UI.",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    embed_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        description="Number of chunks embedded per API call during ingestion.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Python logging level: DEBUG | INFO | WARNING | ERROR.",
    )
    log_file: str = Field(
        default="./logs/app.log",
        description="Path to write the log file. Parent directory is created automatically.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────────

    if _PYDANTIC_V2:

        @field_validator("openai_api_key", mode="before")
        @classmethod
        def _strip_openai_key(cls, v: str) -> str:
            return str(v).strip()

        @field_validator("anthropic_api_key", mode="before")
        @classmethod
        def _strip_anthropic_key(cls, v: str) -> str:
            return str(v).strip()

        @field_validator("google_api_key", mode="before")
        @classmethod
        def _strip_google_key(cls, v: str) -> str:
            return str(v).strip()

        @field_validator("vector_db_path", "food_dataset_path",
                         "mood_mapping_path", "processed_chunks_path",
                         "log_file", mode="before")
        @classmethod
        def _normalise_path(cls, v: str) -> str:
            """Expand ~ and resolve relative paths from the repo root."""
            return str(v).strip()

        @model_validator(mode="after")
        def _check_api_key_present(self) -> "Settings":
            """
            Warn (but don't error) if the active provider's API key is missing.
            We warn rather than raise so the app can still start in dry-run /
            demo mode and show a helpful message in the UI.
            """
            import warnings
            if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
                warnings.warn(
                    "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                    "The chatbot will not be able to generate responses. "
                    "Add OPENAI_API_KEY to your .env file.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
                warnings.warn(
                    "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.llm_provider == LLMProvider.GOOGLE and not self.google_api_key:
                warnings.warn(
                    "LLM_PROVIDER=google but GOOGLE_API_KEY is not set.",
                    UserWarning,
                    stacklevel=2,
                )
            if (self.embedding_provider == EmbeddingProvider.OPENAI
                    and not self.openai_api_key):
                warnings.warn(
                    "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set. "
                    "Ingestion will fail. Use EMBEDDING_PROVIDER=huggingface for local embeddings.",
                    UserWarning,
                    stacklevel=2,
                )
            return self

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def active_llm_api_key(self) -> str:
        """Return the API key for the currently configured LLM provider."""
        return {
            LLMProvider.OPENAI:    self.openai_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.GOOGLE:    self.google_api_key,
        }.get(self.llm_provider, "")

    @property
    def active_llm_model(self) -> str:
        """Return the model name for the currently configured LLM provider."""
        return {
            LLMProvider.OPENAI:    self.llm_model,
            LLMProvider.ANTHROPIC: self.anthropic_model,
            LLMProvider.GOOGLE:    self.google_model,
        }.get(self.llm_provider, self.llm_model)

    @property
    def vector_db_path_obj(self) -> Path:
        """Return vector_db_path as a resolved Path object."""
        return Path(self.vector_db_path).resolve()

    @property
    def is_openai_embedding(self) -> bool:
        return self.embedding_provider == EmbeddingProvider.OPENAI

    @property
    def is_huggingface_embedding(self) -> bool:
        return self.embedding_provider == EmbeddingProvider.HUGGINGFACE

    @property
    def is_chroma(self) -> bool:
        return self.vector_store_provider == VectorStoreProvider.CHROMA

    @property
    def is_faiss(self) -> bool:
        return self.vector_store_provider == VectorStoreProvider.FAISS

    @property
    def active_embedding_model(self) -> str:
        """Return the embedding model name for the active provider."""
        if self.is_huggingface_embedding:
            return self.hf_embedding_model
        return self.embedding_model

    def summary(self) -> str:
        """Return a human-readable one-line config summary for logging."""
        return (
            f"LLM={self.llm_provider.value}/{self.active_llm_model}  "
            f"Embed={self.embedding_provider.value}/{self.active_embedding_model}  "
            f"VectorStore={self.vector_store_provider.value}  "
            f"TopK={self.retriever_top_k}  "
            f"Debug={self.debug_mode}"
        )

    def to_ingestion_dict(self) -> dict:
        """
        Return a flat dict of settings consumed by the ingestion pipeline.
        Matches the keys expected by ingestion/ingest.py _get_config().
        """
        return {
            "food_dataset_path":     self.food_dataset_path,
            "mood_mapping_path":     self.mood_mapping_path,
            "processed_chunks_path": self.processed_chunks_path,
            "embedding_provider":    self.embedding_provider.value,
            "embedding_model":       self.embedding_model,
            "hf_embedding_model":    self.hf_embedding_model,
            "vector_store_provider": self.vector_store_provider.value,
            "vector_db_path":        self.vector_db_path,
            "chroma_collection":     self.chroma_collection_name,
            "openai_api_key":        self.openai_api_key,
            "log_level":             self.log_level.value,
            "log_file":              self.log_file,
            "batch_size":            self.embed_batch_size,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    Reads .env once and caches the result for the lifetime of the process.
    Safe to call from anywhere — multiple calls are free after the first.

    Example:
        from config.settings import get_settings
        cfg = get_settings()
        print(cfg.llm_model)
    """
    # Ensure .env is loaded from the repo root regardless of where the
    # process was launched from.
    _repo_root = Path(__file__).resolve().parent.parent
    _env_path  = _repo_root / ".env"

    if _env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env_path, override=False)

    return Settings()


def reload_settings() -> Settings:
    """
    Clear the cache and reload settings from environment.
    Useful in tests or when the .env file has been changed at runtime.
    """
    get_settings.cache_clear()
    return get_settings()